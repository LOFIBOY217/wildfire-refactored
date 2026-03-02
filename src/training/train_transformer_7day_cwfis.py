"""
Train 7-Day Fire Probability Transformer  [CWFIS Hotspot Labels + Precompute + Mixed Sampling]
================================================================================================
Based on train_transformer_7day_posaware.py with CWFIS hotspot labels.

Key differences from posaware (CIFFC) version:
  Labels   : CWFIS satellite hotspot records (VIIRS-M, 2018–2025)
             instead of CIFFC human-reported fires.
  Data span: 2018–2025 (default) vs 2023–2025 for CIFFC.
  Decoder  : FireProb7DayTransformer uses learnable lead-time queries;
             no external decoder input.

Precompute approach (identical to posaware):
  STEP 7 pre-computes ALL frames into two dense arrays:
    meteo_patched : (T, n_patches, P²*3)  float32
    fire_patched  : (T, n_patches, P²)    uint8   ← binary 0/1, 4× smaller than float32
  After that, __getitem__ is pure O(1) array indexing.

NOTE  Memory requirement:
  P=16, Canada grid → n_patches ≈ 24 310
  meteo_patched ≈ T × 74.7 MB  float32  (e.g. T=1000 → ~73 GB, T=2300 → ~172 GB)
  fire_patched  ≈ T ×  6.2 MB  uint8    (e.g. T=1000 →  ~6 GB, T=2300 →  ~14 GB)
  Reduce T by narrowing --data_start if RAM is limited.

Mixed sampling (identical to posaware):
  Training  : pos_pairs (patches with ≥1 fire pixel in forecast window)
              + neg_ratio × neg_pairs (random background patches)
  Validation: unfiltered (all patches) — honest evaluation.

Usage:
    python -m src.training.train_transformer_7day_cwfis \\
        --config configs/default.yaml \\
        --data_start 2018-05-01 \\
        --pred_start 2024-05-01 \\
        --pred_end   2024-10-31 \\
        --neg_ratio  3

Output:
    outputs/transformer7d_cwfis_fire_prob/YYYYMMDD/fire_prob_lead{k:02d}d_YYYYMMDD.tif
"""

import argparse
import glob
import json
import os
import sys
import time
import atexit
from datetime import date, timedelta
from datetime import datetime as dt

import numpy as np
import rasterio
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset, DataLoader

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    from pathlib import Path
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument

from src.utils.date_utils import extract_date_from_filename
from src.utils.raster_io import read_singleband_stack, clean_nodata
from src.utils.patch_utils import patchify, depatchify
from src.data_ops.processing.rasterize_hotspots import load_hotspot_data, rasterize_hotspots_batch
from src.models.transformer_7day import FireProb7DayTransformer


# ------------------------------------------------------------------ #
# Datasets  (identical structure to posaware)
# ------------------------------------------------------------------ #

class Fire7DayDatasetMixed(Dataset):
    """
    Training dataset: pos_pairs (patches with ≥1 fire pixel in forecast window)
    mixed with randomly sampled neg_pairs (pure-background patches).
    __getitem__ is pure O(1) array indexing after precompute.
    """

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid, all_pairs):
        self.meteo     = meteo_patched   # (T, n_patches, in_dim)
        self.fire      = fire_patched    # (T, n_patches, out_dim)
        self.windows   = windows
        self.hw        = hw
        self.grid      = grid
        self.all_pairs = all_pairs       # list of (win_i, patch_i)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        win_i, patch_i = self.all_pairs[idx]
        hs, he, fs, fe = self.windows[win_i]
        x = self.meteo[hs:he, patch_i, :]              # (in_days,  in_dim)  float32
        y = self.fire[fs:fe,  patch_i, :].astype(np.float32)  # uint8 → float32 for BCE
        return torch.from_numpy(x.copy()), torch.from_numpy(y)


class Fire7DayDatasetPrecompute(Dataset):
    """
    Unfiltered dataset for validation — evaluates on all patches
    (including negative ones) for an honest performance estimate.
    """

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid):
        self.meteo     = meteo_patched
        self.fire      = fire_patched
        self.windows   = windows
        self.hw        = hw
        self.grid      = grid
        self.n_patches = meteo_patched.shape[1]

    def __len__(self):
        return len(self.windows) * self.n_patches

    def __getitem__(self, idx):
        win_i   = idx // self.n_patches
        patch_i = idx %  self.n_patches
        hs, he, fs, fe = self.windows[win_i]
        x = self.meteo[hs:he, patch_i, :]                      # float32
        y = self.fire[fs:fe,  patch_i, :].astype(np.float32)  # uint8 → float32 for BCE
        return torch.from_numpy(x.copy()), torch.from_numpy(y)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _build_file_dict(directory, prefix):
    result = {}
    sub_paths = sorted(glob.glob(
        os.path.join(directory, prefix, f"{prefix}_*.tif")
    ))
    paths = sub_paths if sub_paths else sorted(glob.glob(
        os.path.join(directory, f"{prefix}_*.tif")
    ))
    for p in paths:
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            result[d] = p
    return result


def _build_windows(n_days, in_days, out_days):
    windows = []
    for i in range(in_days, n_days - out_days + 1):
        windows.append((i - in_days, i, i, i + out_days))
    return windows


# ------------------------------------------------------------------ #
# Forecast-only helper
# ------------------------------------------------------------------ #

def _run_forecast_only(args, cfg, fwi_dir, obs_root, output_dir, ckpt_dir):
    """
    Load the best checkpoint and generate forecast tifs for specified years.
    Skips all training steps (STEP 1-9).  Normalises each year's meteo data
    using the saved training statistics so the model sees the same feature
    distribution it was trained on.
    """
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {best_ckpt}\n"
            "Run training first (without --forecast_only)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FORECAST ONLY] Loading checkpoint: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)

    saved_args   = ckpt["args"]
    P            = saved_args["patch_size"]
    in_days      = saved_args["in_days"]
    out_days     = saved_args["out_days"]
    meteo_means  = ckpt["meteo_means"]   # shape (3,)
    meteo_stds   = ckpt["meteo_stds"]    # shape (3,)
    patch_dim_in  = ckpt["patch_dim_in"]
    patch_dim_out = ckpt["patch_dim_out"]

    model = FireProb7DayTransformer(
        patch_dim_in=patch_dim_in,
        patch_dim_out=patch_dim_out,
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_encoder_layers=saved_args["enc_layers"],
        num_decoder_layers=saved_args["dec_layers"],
        dim_feedforward=saved_args["d_model"] * 4,
        forecast_days=out_days,
        encoder_days=in_days,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Device={device}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  in_days={in_days}  out_days={out_days}  patch_size={P}")
    print(f"  norm means={meteo_means.round(3)}  stds={meteo_stds.round(3)}")

    # Build full file indices (covers all available years)
    fwi_dict = {}
    for p in sorted(glob.glob(os.path.join(fwi_dir, "*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            fwi_dict[d] = p
    d2m_dict = _build_file_dict(obs_root, "2d")
    t2m_dict = _build_file_dict(obs_root, "2t")

    # Raster profile + dimensions from first FWI file
    first_fwi = sorted(fwi_dict.values())[0]
    with rasterio.open(first_fwi) as src:
        profile = src.profile
        H, W    = profile["height"], profile["width"]
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    # Parse target years
    if args.forecast_years:
        years = [int(y.strip()) for y in args.forecast_years.split(",")]
    else:
        years = [int(args.pred_start.split("-")[0])]
    print(f"  Forecast years: {years}")

    def _clean(stack):
        stack = clean_nodata(stack.astype(np.float32))
        fill  = float(np.nanmean(stack))
        if not np.isfinite(fill):
            fill = 0.0
        return np.nan_to_num(stack, nan=fill, posinf=fill, neginf=fill)

    for year in years:
        print(f"\n{'='*60}")
        print(f"  Year {year}: May 1 – Oct 31")
        print(f"{'='*60}")

        pred_start   = date(year, 5, 1)
        pred_end     = date(year, 10, 31)
        # Need in_days of history before the first prediction date
        data_start   = pred_start - timedelta(days=in_days + 5)
        required_end = pred_end   + timedelta(days=out_days)

        fwi_p, t2m_p, d2m_p, dates_y = [], [], [], []
        cur = data_start
        while cur <= required_end:
            if cur in fwi_dict and cur in t2m_dict and cur in d2m_dict:
                fwi_p.append(fwi_dict[cur])
                t2m_p.append(t2m_dict[cur])
                d2m_p.append(d2m_dict[cur])
                dates_y.append(cur)
            cur += timedelta(days=1)

        T_y = len(dates_y)
        if T_y < in_days + out_days:
            print(f"  Only {T_y} aligned days (need >= {in_days + out_days}). Skipping {year}.")
            continue
        print(f"  Aligned: {T_y} days  ({dates_y[0]} → {dates_y[-1]})")

        fwi_s = read_singleband_stack(fwi_p)
        t2m_s = read_singleband_stack(t2m_p)
        d2m_s = read_singleband_stack(d2m_p)

        meteo_y = np.empty((T_y, H, W, 3), dtype=np.float32)
        meteo_y[..., 0] = _clean(fwi_s); del fwi_s
        meteo_y[..., 1] = _clean(t2m_s); del t2m_s
        meteo_y[..., 2] = _clean(d2m_s); del d2m_s

        # Normalise using TRAINING statistics (not recomputed)
        meteo_y -= meteo_means
        meteo_y /= meteo_stds
        np.clip(meteo_y, -10.0, 10.0, out=meteo_y)

        date_to_idx_y = {d: i for i, d in enumerate(dates_y)}
        n_pred = (pred_end - pred_start).days + 1
        pred_dates_y = [pred_start + timedelta(days=k) for k in range(n_pred)]

        n_done = 0
        for base_date in pred_dates_y:
            if base_date not in date_to_idx_y:
                continue
            base_idx = date_to_idx_y[base_date]
            if base_idx < in_days:
                continue

            hist = meteo_y[base_idx - in_days: base_idx]       # (in_days, H, W, 3)
            xp, pred_hw, pred_grid = patchify(hist, P)          # (n_patches, in_days, in_dim)

            chunk = args.pred_batch_size
            n_p   = xp.shape[0]
            prob_list = []
            with torch.no_grad():
                for cs in range(0, n_p, chunk):
                    ce  = min(cs + chunk, n_p)
                    xb  = torch.from_numpy(xp[cs:ce]).float().to(device)
                    prob_list.append(torch.sigmoid(model(xb)).cpu().numpy())
            probs = np.concatenate(prob_list, axis=0)           # (n_patches, out_days, P²)

            base_str = base_date.strftime("%Y%m%d")
            day_out  = os.path.join(output_dir, base_str)
            os.makedirs(day_out, exist_ok=True)

            for lead in range(1, out_days + 1):
                target_date     = base_date + timedelta(days=lead)
                target_date_str = target_date.strftime("%Y%m%d")
                out_path = os.path.join(
                    day_out, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
                )
                prob_patches_lead = probs[:, lead - 1, :]
                prob_vol = depatchify(
                    prob_patches_lead[:, np.newaxis, :],
                    pred_grid, P, pred_hw, num_channels=1
                )
                prob_map = prob_vol[0] if prob_vol.ndim == 3 else prob_vol
                if prob_map.shape != (H, W):
                    full = np.zeros((H, W), dtype=np.float32)
                    full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                    prob_map = full
                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(prob_map.astype(np.float32), 1)

            n_done += 1
            if n_done % 30 == 0 or base_date == pred_dates_y[-1]:
                print(f"  [{n_done}/{len(pred_dates_y)}] {base_date} → {out_days} tifs")

        del meteo_y
        print(f"  Year {year}: {n_done} base dates, {n_done * out_days} tifs")

    print("\n" + "=" * 70)
    print("FORECAST-ONLY COMPLETE")
    print(f"  Output: {output_dir}")
    print("=" * 70)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    run_started_at  = time.time()
    run_started_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"

    ap = argparse.ArgumentParser(
        description="Train 7-Day Fire Probability Transformer [CWFIS Hotspot + Precompute + Mixed]"
    )
    add_config_argument(ap)
    ap.add_argument("--data_start",   type=str,   default="2018-05-01",
                    help="First date for data loading. Narrow this to reduce RAM usage.")
    ap.add_argument("--pred_start",   type=str,   default="2024-05-01",
                    help="First prediction date; also used as train/val split boundary.")
    ap.add_argument("--pred_end",     type=str,   default="2024-10-31",
                    help="Last prediction date (inclusive).")
    ap.add_argument("--in_days",      type=int,   default=7)
    ap.add_argument("--out_days",     type=int,   default=7)
    ap.add_argument("--patch_size",   type=int,   default=16)
    ap.add_argument("--d_model",      type=int,   default=128)
    ap.add_argument("--nhead",        type=int,   default=4)
    ap.add_argument("--enc_layers",   type=int,   default=2)
    ap.add_argument("--dec_layers",   type=int,   default=2)
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--neg_ratio",      type=float, default=3.0,
                    help="Neg-only patches per positive patch (default=3). "
                         "Higher = more background exposure, better calibration.")
    ap.add_argument("--pred_batch_size", type=int, default=512,
                    help="Patch chunk size for GPU inference in STEP 10 (default=512). "
                         "Reduce if CUDA OOM during forecast generation.")
    ap.add_argument("--dilate_radius",   type=int, default=5,
                    help="Spatial dilation radius applied to CWFIS hotspot labels (pixels). "
                         "Each hotspot point is expanded to a filled circle of this radius. "
                         "Grid resolution is ~2 km/pixel, so radius=5 → 10 km buffer. "
                         "Set 0 to disable. Default=5.")
    ap.add_argument("--forecast_only", action="store_true",
                    help="Skip training — load best checkpoint and generate forecast tifs "
                         "for --forecast_years. Useful for extending predictions to new years.")
    ap.add_argument("--forecast_years", type=str, default=None,
                    help="Comma-separated years for forecast generation (used with "
                         "--forecast_only), e.g. '2022,2023'. Each year runs May 1 – Oct 31.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------
    # Config & paths
    # ----------------------------------------------------------------
    cfg         = load_config(args.config)
    fwi_dir     = get_path(cfg, "fwi_dir")
    paths_cfg   = cfg.get("paths", {})
    obs_root    = get_path(cfg, "observation_dir") if "observation_dir" in paths_cfg \
                  else get_path(cfg, "ecmwf_dir")
    hotspot_csv = get_path(cfg, "hotspot_csv")
    output_dir  = os.path.join(get_path(cfg, "output_dir"), "transformer7d_cwfis_fire_prob")
    ckpt_dir    = os.path.join(get_path(cfg, "checkpoint_dir"), "transformer_7day_cwfis")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    # -- Forecast-only shortcut: skip training entirely --
    if args.forecast_only:
        _run_forecast_only(args, cfg, fwi_dir, obs_root, output_dir, ckpt_dir)
        return

    run_stamp     = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_meta_path = os.path.join(ckpt_dir, f"run_{run_stamp}.json")
    run_meta = {
        "run_started_at_utc": run_started_iso,
        "label_source":       "CWFIS satellite hotspots (VIIRS-M)",
        "sampling":           f"mixed — pos + {args.neg_ratio}x neg patches, precomputed",
        "cli_args": vars(args),
        "status": "running",
    }

    def _flush():
        if run_meta.get("status") == "running":
            run_meta["status"] = "failed_or_interrupted"
            run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
        try:
            with open(run_meta_path, "w") as f:
                json.dump(run_meta, f, indent=2)
        except Exception:
            pass
    atexit.register(_flush)

    def _date(s):
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))

    data_start_date = _date(args.data_start)
    pred_start_date = _date(args.pred_start)
    pred_end_date   = _date(args.pred_end)
    in_days         = args.in_days
    out_days        = args.out_days

    print("\n" + "=" * 70)
    print("7-DAY FIRE PROBABILITY TRANSFORMER  [CWFIS Hotspot + Precompute + Mixed]")
    print("=" * 70)
    print(f"  data_start        : {data_start_date}")
    print(f"  pred_start        : {pred_start_date}  (train/val split boundary)")
    print(f"  pred_end          : {pred_end_date}")
    print(f"  in_days / out_days: {in_days} / {out_days}")
    print(f"  patch_size        : {args.patch_size}")
    print(f"  neg_ratio         : {args.neg_ratio}  (neg patches per pos patch)")
    print(f"  d_model / nhead   : {args.d_model} / {args.nhead}")
    print(f"  epochs / batch    : {args.epochs} / {args.batch_size}  lr={args.lr}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # STEP 1  Build file indices
    # ----------------------------------------------------------------
    print("\n[STEP 1] Building file index...")

    fwi_dict = {}
    for p in sorted(glob.glob(os.path.join(fwi_dir, "*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            fwi_dict[d] = p

    d2m_dict = _build_file_dict(obs_root, "2d")
    t2m_dict = _build_file_dict(obs_root, "2t")

    if not fwi_dict:
        raise RuntimeError(f"No FWI .tif files found in {fwi_dir}")
    if not d2m_dict:
        raise RuntimeError(f"No 2d .tif files found under {obs_root}")
    if not t2m_dict:
        raise RuntimeError(f"No 2t .tif files found under {obs_root}")

    print(f"  FWI: {len(fwi_dict)} days  2t: {len(t2m_dict)} days  2d: {len(d2m_dict)} days")

    # ----------------------------------------------------------------
    # STEP 2  Align dates
    # ----------------------------------------------------------------
    print("\n[STEP 2] Aligning dates...")

    required_end = pred_end_date + timedelta(days=out_days)
    fwi_paths, t2m_paths, d2m_paths, aligned_dates = [], [], [], []
    cur = data_start_date
    while cur <= required_end:
        if cur in fwi_dict and cur in t2m_dict and cur in d2m_dict:
            fwi_paths.append(fwi_dict[cur])
            t2m_paths.append(t2m_dict[cur])
            d2m_paths.append(d2m_dict[cur])
            aligned_dates.append(cur)
        cur += timedelta(days=1)

    if len(aligned_dates) < in_days + out_days:
        raise RuntimeError(
            f"Only {len(aligned_dates)} aligned days, need >= {in_days + out_days}"
        )
    print(f"  Aligned dates: {len(aligned_dates)}  ({aligned_dates[0]} -> {aligned_dates[-1]})")
    run_meta["aligned_days"] = len(aligned_dates)

    # ----------------------------------------------------------------
    # STEP 3  Load raster stacks
    # ----------------------------------------------------------------
    print("\n[STEP 3] Loading raster stacks...")
    fwi_stack = read_singleband_stack(fwi_paths)
    t2m_stack = read_singleband_stack(t2m_paths)
    d2m_stack = read_singleband_stack(d2m_paths)
    T, H, W = fwi_stack.shape
    print(f"  Shape: T={T}  H={H}  W={W}")

    def _clean_stack(stack):
        stack = clean_nodata(stack.astype(np.float32))
        fill  = float(np.nanmean(stack))
        if not np.isfinite(fill):
            fill = 0.0
        return np.nan_to_num(stack, nan=fill, posinf=fill, neginf=fill)

    fwi_stack = _clean_stack(fwi_stack)
    t2m_stack = _clean_stack(t2m_stack)
    d2m_stack = _clean_stack(d2m_stack)
    print(f"  FWI  range after clean: [{fwi_stack.min():.2f}, {fwi_stack.max():.2f}]")
    print(f"  2t   range after clean: [{t2m_stack.min():.2f}, {t2m_stack.max():.2f}]")
    print(f"  2d   range after clean: [{d2m_stack.min():.2f}, {d2m_stack.max():.2f}]")

    with rasterio.open(fwi_paths[0]) as src:
        profile = src.profile

    # ----------------------------------------------------------------
    # STEP 4  Load and rasterize CWFIS hotspots
    # ----------------------------------------------------------------
    print("\n[STEP 4] Loading CWFIS hotspot records...")
    hotspot_df = load_hotspot_data(hotspot_csv)
    print(f"  Total records : {len(hotspot_df):,}")
    print(f"  Date range    : {hotspot_df['date'].min()} to {hotspot_df['date'].max()}")
    run_meta["hotspot_records"] = int(len(hotspot_df))
    fire_stack = rasterize_hotspots_batch(hotspot_df, aligned_dates, profile)   # [T,H,W] uint8
    pos_rate   = fire_stack.mean()
    print(f"  Fire stack shape     : {fire_stack.shape}")
    print(f"  Mean fire pixel rate : {pos_rate:.6%}  (raw hotspot points)")
    run_meta["positive_rate_raw"] = float(pos_rate)

    # -- Spatial dilation: expand each hotspot point to a filled circle --
    if args.dilate_radius > 0:
        r = args.dilate_radius
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk   = (xx ** 2 + yy ** 2 <= r ** 2)          # circular kernel
        print(f"\n  Dilating fire labels: radius={r} px  "
              f"(~{r * 2} km buffer, kernel {disk.shape[0]}×{disk.shape[1]}, "
              f"{disk.sum()} pixels/hotspot)...")
        t0_dil = time.time()
        for t in range(T):
            if fire_stack[t].any():                      # skip empty frames (fast)
                fire_stack[t] = binary_dilation(
                    fire_stack[t], structure=disk
                ).astype(np.uint8)
            if t % 200 == 0 or t == T - 1:
                print(f"    dilate frame {t:4d}/{T}  ({time.time()-t0_dil:.0f}s)")
        pos_rate_dil = fire_stack.mean()
        print(f"  After dilation: positive_rate={pos_rate_dil:.4%}  "
              f"({pos_rate_dil/pos_rate:.1f}× increase)  "
              f"({time.time()-t0_dil:.0f}s)")
        run_meta["dilate_radius"]        = r
        run_meta["positive_rate_dilated"] = float(pos_rate_dil)
    else:
        run_meta["dilate_radius"] = 0
        run_meta["positive_rate_dilated"] = float(pos_rate)

    # ----------------------------------------------------------------
    # STEP 5  Train / val split
    # ----------------------------------------------------------------
    print("\n[STEP 5] Splitting train / val by pred_start...")
    train_end_idx = None
    for i, d in enumerate(aligned_dates):
        if d >= pred_start_date:
            train_end_idx = i
            break
    if train_end_idx is None:
        raise RuntimeError(f"pred_start={pred_start_date} is beyond all aligned dates")

    print(f"  Training : 0 -> {train_end_idx - 1}  "
          f"({aligned_dates[0]} -> {aligned_dates[train_end_idx-1]})")
    print(f"  Val/pred : {train_end_idx} -> {T-1}  "
          f"({aligned_dates[train_end_idx]} -> {aligned_dates[-1]})")

    # ----------------------------------------------------------------
    # STEP 6  Standardise (in-place, memory-efficient)
    # ----------------------------------------------------------------
    print("\n[STEP 6] Standardising per-channel (FWI, 2t, 2d)...")

    meteo_norm = np.empty((T, H, W, 3), dtype=np.float32)
    meteo_norm[..., 0] = fwi_stack;  del fwi_stack
    meteo_norm[..., 1] = t2m_stack;  del t2m_stack
    meteo_norm[..., 2] = d2m_stack;  del d2m_stack

    train_meteo = meteo_norm[:train_end_idx]
    meteo_means = train_meteo.reshape(-1, 3).mean(axis=0)
    meteo_stds  = train_meteo.reshape(-1, 3).std(axis=0) + 1e-6
    del train_meteo

    meteo_norm -= meteo_means
    meteo_norm /= meteo_stds
    np.clip(meteo_norm, -10.0, 10.0, out=meteo_norm)

    print(f"  Means (FWI,2t,2d): {meteo_means.round(3)}")
    print(f"  Stds  (FWI,2t,2d): {meteo_stds.round(3)}")

    run_meta["norm_stats"] = {
        "meteo_means": meteo_means.tolist(),
        "meteo_stds":  meteo_stds.tolist(),
    }
    np.save(os.path.join(ckpt_dir, "norm_stats.npy"),
            np.stack([meteo_means, meteo_stds]))

    # ----------------------------------------------------------------
    # STEP 7  Pre-compute all patch arrays
    # ----------------------------------------------------------------
    P = args.patch_size
    _sample, hw, grid = patchify(meteo_norm[:1], P)
    n_patches = grid[0] * grid[1]
    in_dim    = P * P * 3
    out_dim   = P * P

    meteo_gb  = T * n_patches * in_dim  * 4 / 1e9   # float32 = 4 bytes
    fire_gb   = T * n_patches * out_dim * 1 / 1e9   # uint8   = 1 byte  (4× smaller)
    needed_gb = meteo_gb + fire_gb

    print(f"\n[STEP 7] Pre-computing patches for T={T} frames...")
    print(f"  n_patches={n_patches}  in_dim={in_dim}  out_dim={out_dim}")
    print(f"  Estimated RAM: meteo={meteo_gb:.1f} GB (float32)  "
          f"fire={fire_gb:.1f} GB (uint8)  total={needed_gb:.1f} GB")

    # -- Memory pre-flight check --
    try:
        import psutil
        vm           = psutil.virtual_memory()
        available_gb = vm.available / 1e9
        total_gb     = vm.total    / 1e9
        print(f"  System RAM   : {available_gb:.1f} GB available / {total_gb:.1f} GB total")
        if needed_gb > total_gb * 0.85:
            # Check against TOTAL RAM — available RAM fluctuates with OS cache.
            # Only abort if the data physically cannot fit in the machine.
            bytes_per_frame = n_patches * (in_dim * 4 + out_dim * 1)  # mixed dtypes
            safe_T = int(total_gb * 0.80 * 1e9 / bytes_per_frame)
            safe_date = aligned_dates[max(0, T - safe_T)]
            print(f"\n  *** OOM WARNING ***")
            print(f"  Need {needed_gb:.1f} GB but machine total is only {total_gb:.1f} GB.")
            print(f"  Suggested: --data_start {safe_date}  (reduces T to ~{safe_T} days)")
            print(f"  Aborting to avoid system freeze. Re-run with a later --data_start.\n")
            raise SystemExit(1)
        elif needed_gb > available_gb:
            print(f"  NOTE: Need {needed_gb:.1f} GB, currently {available_gb:.1f} GB free "
                  f"(total {total_gb:.1f} GB). OS will reclaim cache — continuing.")
    except ImportError:
        print("  (pip install psutil to enable RAM pre-flight check)")

    print("  (__getitem__ will be O(1) array indexing thereafter)")

    # -- meteo_patched --
    t0_meteo = time.time()
    try:
        meteo_patched = np.empty((T, n_patches, in_dim), dtype=np.float32)
    except MemoryError:
        safe_T = int(0.8 / ((n_patches * in_dim * 4) / 1e9))
        raise RuntimeError(
            f"MemoryError allocating meteo_patched ({meteo_gb:.1f} GB). "
            f"Try --data_start with a later date (need T <= ~{safe_T} for 80% of total RAM)."
        )
    for t in range(T):
        patches, _, _ = patchify(meteo_norm[t:t + 1], P)
        meteo_patched[t] = patches[:, 0, :]
        if t % 100 == 0 or t == T - 1:
            print(f"  meteo frame {t:4d}/{T}  ({time.time()-t0_meteo:.0f}s)")
    print(f"  meteo_patched: {meteo_patched.shape}  "
          f"{meteo_patched.nbytes/1e9:.1f} GB  ({time.time()-t0_meteo:.0f}s)")

    # -- fire_patched  (uint8 — binary 0/1 fire labels, 4× smaller than float32) --
    t0_fire = time.time()
    try:
        fire_patched = np.empty((T, n_patches, out_dim), dtype=np.uint8)
    except MemoryError:
        raise RuntimeError(
            f"MemoryError allocating fire_patched ({fire_gb:.1f} GB as uint8). "
            f"meteo_patched already occupies {meteo_gb:.1f} GB. "
            f"Try --data_start with a later date."
        )
    for t in range(T):
        # patchify expects float; cast back to uint8 before storing
        fut1 = fire_stack[t:t + 1].astype(np.float32)
        patches, _, _ = patchify(fut1, P)
        fire_patched[t] = patches[:, 0, :].astype(np.uint8)
        if t % 100 == 0 or t == T - 1:
            print(f"  fire  frame {t:4d}/{T}  ({time.time()-t0_fire:.0f}s)")
    print(f"  fire_patched:  {fire_patched.shape}  dtype=uint8  "
          f"{fire_patched.nbytes/1e9:.1f} GB  ({time.time()-t0_fire:.0f}s)")

    all_windows = _build_windows(T, in_days, out_days)
    train_wins  = [w for w in all_windows
                   if aligned_dates[w[1] - 1] < pred_start_date and w[3] <= T]
    val_wins    = [w for w in all_windows
                   if aligned_dates[w[0]] >= pred_start_date and w[3] <= T]
    print(f"\n  Total windows: {len(all_windows)}  train: {len(train_wins)}  val: {len(val_wins)}")

    # ----------------------------------------------------------------
    # STEP 7b  Build positive pairs; sample negative pairs
    # ----------------------------------------------------------------
    print("\n[STEP 7b] Filtering positive patches + sampling negative patches...")
    t0_filter = time.time()

    pos_pairs = []
    for win_i, (hs, he, fs, fe) in enumerate(train_wins):
        for patch_i in range(n_patches):
            if fire_patched[fs:fe, patch_i, :].max() > 0:
                pos_pairs.append((win_i, patch_i))
        if win_i % 200 == 0 or win_i == len(train_wins) - 1:
            print(f"  scanned {win_i+1}/{len(train_wins)} windows  "
                  f"pos_pairs: {len(pos_pairs):,}  ({time.time()-t0_filter:.0f}s)")

    total_pairs = len(train_wins) * n_patches
    pct = 100.0 * len(pos_pairs) / max(total_pairs, 1)
    print(f"  Positive pairs: {len(pos_pairs):,} / {total_pairs:,}  ({pct:.3f}%)")

    if len(pos_pairs) == 0:
        raise RuntimeError(
            "No positive (window, patch) pairs found in training set. "
            "Check hotspot_csv and date alignment."
        )

    n_neg_target = int(len(pos_pairs) * args.neg_ratio)
    pos_set = set(pos_pairs)
    neg_set = set()
    rng     = np.random.default_rng(args.seed)
    n_wins  = len(train_wins)
    while len(neg_set) < n_neg_target:
        win_i   = int(rng.integers(0, n_wins))
        patch_i = int(rng.integers(0, n_patches))
        pair    = (win_i, patch_i)
        if pair not in pos_set:
            neg_set.add(pair)
    neg_pairs = list(neg_set)

    all_pairs = list(pos_pairs) + neg_pairs
    rng.shuffle(all_pairs)
    print(f"  Neg pairs sampled: {len(neg_pairs):,}  (neg_ratio={args.neg_ratio})")
    print(f"  Total train pairs: {len(all_pairs):,}  "
          f"(pos {len(pos_pairs)/len(all_pairs)*100:.1f}% / "
          f"neg {len(neg_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Sampling time: {time.time()-t0_filter:.0f}s")

    run_meta["pos_pairs"]   = len(pos_pairs)
    run_meta["neg_pairs"]   = len(neg_pairs)
    run_meta["total_pairs"] = len(all_pairs)

    # ----------------------------------------------------------------
    # STEP 8  Compute pos_weight; build BCE criterion
    # ----------------------------------------------------------------
    print("\n[STEP 8] Computing BCE pos_weight from pixel counts...")
    pos_pixels = 0
    neg_pixels_in_pos = 0
    for win_i, patch_i in pos_pairs:
        hs, he, fs, fe = train_wins[win_i]
        pf = fire_patched[fs:fe, patch_i, :]
        p  = int(pf.sum())
        pos_pixels        += p
        neg_pixels_in_pos += pf.size - p
    neg_pixels_total = neg_pixels_in_pos + len(neg_pairs) * out_dim * out_days
    raw_ratio        = neg_pixels_total / max(pos_pixels, 1)
    pos_weight_val   = min(raw_ratio, 100.0)
    print(f"  pos_pixels : {pos_pixels:,}")
    print(f"  neg_pixels : {neg_pixels_total:,}")
    print(f"  raw ratio  : {raw_ratio:.1f}   pos_weight (capped 100) = {pos_weight_val:.2f}")
    run_meta["pos_weight"] = pos_weight_val

    # ----------------------------------------------------------------
    # Build datasets and dataloaders
    # ----------------------------------------------------------------
    train_ds = Fire7DayDatasetMixed(
        meteo_patched, fire_patched, train_wins, hw, grid, all_pairs
    )
    if val_wins:
        val_ds = Fire7DayDatasetPrecompute(
            meteo_patched, fire_patched, val_wins, hw, grid
        )
    else:
        val_ds = train_ds
        print("  WARNING: no val windows, using train set as val proxy")

    patch_dim_in  = in_dim
    patch_dim_out = out_dim

    print(f"\n  Train samples (mixed pos+neg): {len(train_ds):,}")
    print(f"  Val   samples (unfiltered)  : {len(val_ds):,}")
    print(f"  Grid: {grid[0]}x{grid[1]} patches/frame  "
          f"(dim_in={patch_dim_in}  dim_out={patch_dim_out})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----------------------------------------------------------------
    # STEP 9  Build model & train
    # ----------------------------------------------------------------
    print("\n[STEP 9] Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = FireProb7DayTransformer(
        patch_dim_in=patch_dim_in,
        patch_dim_out=patch_dim_out,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.d_model * 4,
        forecast_days=out_days,
        encoder_days=in_days,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    run_meta["n_params"] = n_params

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_ckpt     = os.path.join(ckpt_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss    = 0.0
        train_samples = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if not torch.isfinite(loss):          # skip NaN/Inf batches
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * xb.size(0)
            train_samples += xb.size(0)

        # train_samples==0 means ALL batches were NaN → model weights are dead
        if train_samples == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"*** ALL TRAIN BATCHES NaN — model weights corrupted. Stopping.")
            break
        train_loss = train_loss / train_samples

        model.eval()
        val_loss    = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl = criterion(model(xb), yb)
                if torch.isfinite(vl):
                    val_loss    += vl.item() * xb.size(0)
                    val_samples += xb.size(0)

        if val_samples == 0:
            # All val batches NaN — don't update best, just report and stop
            print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  "
                  f"val=NaN (all batches invalid) — stopping early.")
            break
        val_loss = val_loss / val_samples

        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Only save when val_loss is a real finite improvement (not a 0.0 from NaN samples)
        if val_samples > 0 and np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "meteo_means":   meteo_means,
                "meteo_stds":    meteo_stds,
                "patch_dim_in":  patch_dim_in,
                "patch_dim_out": patch_dim_out,
                "hw":            hw,
                "grid":          grid,
                "args":          vars(args),
            }, best_ckpt)

    print(f"\n  Best val loss: {best_val_loss:.6f}  saved -> {best_ckpt}")
    run_meta["best_val_loss"] = best_val_loss

    # ----------------------------------------------------------------
    # STEP 10  Generate forecast GeoTIFFs
    # ----------------------------------------------------------------
    print("\n[STEP 10] Generating forecast tifs...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    pred_dates = []
    cur = pred_start_date
    while cur <= pred_end_date:
        pred_dates.append(cur)
        cur += timedelta(days=1)

    date_to_idx = {d: i for i, d in enumerate(aligned_dates)}

    for base_date in pred_dates:
        if base_date not in date_to_idx:
            continue
        base_idx = date_to_idx[base_date]
        if base_idx < in_days:
            continue

        hist = meteo_norm[base_idx - in_days: base_idx]     # (in_days, H, W, 3)
        xp, pred_hw, pred_grid = patchify(hist, P)          # (n_patches, in_days, in_dim)

        # Chunk-wise inference to avoid CUDA OOM on large grids
        chunk    = args.pred_batch_size
        n_p      = xp.shape[0]
        prob_list = []
        with torch.no_grad():
            for cs in range(0, n_p, chunk):
                ce  = min(cs + chunk, n_p)
                xb  = torch.from_numpy(xp[cs:ce]).float().to(device)
                prob_list.append(torch.sigmoid(model(xb)).cpu().numpy())
        probs = np.concatenate(prob_list, axis=0)           # (n_patches, out_days, P²)

        base_str = base_date.strftime("%Y%m%d")
        day_out  = os.path.join(output_dir, base_str)
        os.makedirs(day_out, exist_ok=True)

        for lead in range(1, out_days + 1):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            out_path        = os.path.join(
                day_out, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
            )

            prob_patches_lead = probs[:, lead - 1, :]
            prob_vol = depatchify(
                prob_patches_lead[:, np.newaxis, :],
                pred_grid, P, pred_hw, num_channels=1
            )
            prob_map = prob_vol[0] if prob_vol.ndim == 3 else prob_vol

            if prob_map.shape != (H, W):
                full = np.zeros((H, W), dtype=np.float32)
                full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                prob_map = full

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(prob_map.astype(np.float32), 1)

        print(f"  [DONE] {base_date} -> {out_days} lead tifs")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE  [CWFIS Hotspot + Precompute + Mixed]")
    print(f"  Forecasts : {output_dir}")
    print(f"  Checkpoint: {best_ckpt}")
    print("=" * 70)

    run_meta["status"]           = "success"
    run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run log: {run_meta_path}")


if __name__ == "__main__":
    main()
