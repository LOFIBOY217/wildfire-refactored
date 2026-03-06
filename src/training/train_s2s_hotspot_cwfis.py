"""
Train S2S Hotspot Transformer  [CWFIS Hotspot Labels, Lead 14–46 Days]
=======================================================================
Seq2Seq Transformer that predicts per-pixel fire probability for the
subseasonal range (14–46 days ahead) using:

  Encoder : 7 days of historical weather  (FWI + 2m temp + 2m dewpoint)
  Decoder : future weather for leads 14–46 (ERA5 oracle by default;
            ECMWF S2S hindcast when --ecmwf_dir is supplied — see below)
  Target  : CWFIS satellite hotspot labels (binary, dilated)

Architecture: S2SHotspotTransformer (src/models/s2s_hotspot.py)
  Encoder → TransformerEncoder → memory
  Decoder → TransformerDecoder(memory) → Linear → logits
  Loss:     BCEWithLogitsLoss with pos_weight cap

Decoder modes
-------------
  ERA5 oracle  (default, no --ecmwf_dir):
      Decoder input = ERA5 future weather for the same 3 channels
      (FWI, 2t, 2d) from the precomputed meteo array.  The decoder
      sees the TRUE future weather — valid for architecture validation
      and upper-bound benchmarking before hindcast data is available.
      patch_dim_dec = P² × 3

  ECMWF S2S hindcast  (--ecmwf_dir <path>):
      Decoder input = 5 ECMWF variables (2t, 2d, tcw, sm20, st20)
      loaded on-the-fly per issue date from <ecmwf_dir>/<YYYYMMDD>/.
      Only issue dates that have ECMWF data contribute to training.
      patch_dim_dec = P² × 5
      *** NOT YET IMPLEMENTED — placeholder for future work ***

Memory note (ERA5 oracle mode)
-------------------------------
  meteo_patched : (T, n_patches, P²×3)  float32 — both encoder and decoder
  fire_patched  : (T, n_patches, P²)    uint8
  For P=16, Canada grid → n_patches ≈ 24 310
  T=600  days  →  meteo ≈ 45 GB,  fire ≈ 3.7 GB   (recommended minimum)
  T=1000 days  →  meteo ≈ 75 GB,  fire ≈ 6.2 GB

Usage (ERA5 oracle, quick test — 1 year training):
    python -m src.training.train_s2s_hotspot_cwfis \\
        --config configs/default.yaml \\
        --data_start 2022-05-01 \\
        --pred_start 2023-05-01 \\
        --pred_end   2023-10-31 \\
        --lead_start 14 --lead_end 46 \\
        --patch_size 16 --d_model 256 --nhead 8 \\
        --enc_layers 4 --dec_layers 4 \\
        --epochs 30 --batch_size 128 --lr 1e-4 \\
        --neg_ratio 20 --pos_weight_cap 10 \\
        --dilate_radius 14

Output:
    outputs/s2s_hotspot_cwfis_fire_prob/YYYYMMDD/fire_prob_lead{k:02d}d_YYYYMMDD.tif
    (compatible with evaluate_topk_cwfis.py)
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
from src.models.s2s_hotspot import S2SHotspotTransformer


# ------------------------------------------------------------------ #
# Datasets
# ------------------------------------------------------------------ #

class S2SHotspotDatasetMixed(Dataset):
    """
    Training dataset: pos_pairs (patches with ≥1 fire pixel in target window)
    mixed with neg_ratio × neg_pairs (pure-background patches).

    Each sample returns (x_enc, x_dec, y):
      x_enc : (encoder_days, patch_dim_enc)  — historical weather
      x_dec : (decoder_days, patch_dim_dec)  — future weather (oracle)
      y     : (decoder_days, P²)             — binary fire labels (float32)

    windows[win_i] = (enc_start, enc_end, target_start, target_end)
      enc:    meteo_patched[enc_start : enc_end]    — 7 days before issue_date
      target: meteo_patched[target_start : target_end] — lead_start..lead_end days after
    """

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid, all_pairs):
        self.meteo     = meteo_patched     # (T, n_patches, enc_dim)
        self.fire      = fire_patched      # (T, n_patches, P²) uint8
        self.windows   = windows           # list of (hs, he, ts, te)
        self.hw        = hw
        self.grid      = grid
        self.all_pairs = all_pairs         # list of (win_i, patch_i)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        win_i, patch_i = self.all_pairs[idx]
        hs, he, ts, te = self.windows[win_i]
        x_enc = self.meteo[hs:he, patch_i, :]                        # (enc_days, enc_dim)
        x_dec = self.meteo[ts:te, patch_i, :]                        # (dec_days, enc_dim)
        y     = self.fire[ts:te, patch_i, :].astype(np.float32)      # (dec_days, P²)
        return (
            torch.from_numpy(x_enc.copy()),
            torch.from_numpy(x_dec.copy()),
            torch.from_numpy(y),
        )


class S2SHotspotDatasetUnfiltered(Dataset):
    """
    Unfiltered dataset for validation — evaluates on all patches for an
    honest performance estimate (no pos/neg sampling).
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
        hs, he, ts, te = self.windows[win_i]
        x_enc = self.meteo[hs:he, patch_i, :]
        x_dec = self.meteo[ts:te, patch_i, :]
        y     = self.fire[ts:te, patch_i, :].astype(np.float32)
        return (
            torch.from_numpy(x_enc.copy()),
            torch.from_numpy(x_dec.copy()),
            torch.from_numpy(y),
        )


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _build_file_dict(directory, prefix):
    """Return {date: filepath} for tif files matching prefix_YYYYMMDD.tif."""
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


def _build_s2s_windows(n_days, in_days, lead_start, lead_end):
    """
    Build (enc_start, enc_end, target_start, target_end) index tuples.

    For each issue_date index i:
      - Encoder window: aligned_dates[i - in_days : i]
      - Gap (not used):  aligned_dates[i : i + lead_start]
      - Target window:   aligned_dates[i + lead_start : i + lead_end + 1]

    Conditions:
      enc_start >= 0         → i >= in_days
      target_end <= n_days   → i + lead_end + 1 <= n_days
                             → i <= n_days - lead_end - 1
    """
    windows = []
    for i in range(in_days, n_days - lead_end):
        windows.append((i - in_days, i, i + lead_start, i + lead_end + 1))
    return windows


# ------------------------------------------------------------------ #
# Forecast-only helper
# ------------------------------------------------------------------ #

def _run_forecast_only(args, cfg, fwi_dir, obs_root, output_dir, ckpt_dir):
    """
    Load the best checkpoint and generate forecast TIFs for --forecast_years.
    Uses ERA5 oracle decoder (precomputed future meteo) — identical
    normalisation stats as training.
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
    lead_start   = saved_args["lead_start"]
    lead_end     = saved_args["lead_end"]
    decoder_days = lead_end - lead_start + 1
    meteo_means  = ckpt["meteo_means"]
    meteo_stds   = ckpt["meteo_stds"]
    patch_dim_enc = ckpt["patch_dim_enc"]
    patch_dim_dec = ckpt["patch_dim_dec"]
    patch_dim_out = ckpt["patch_dim_out"]

    model = S2SHotspotTransformer(
        patch_dim_enc=patch_dim_enc,
        patch_dim_dec=patch_dim_dec,
        patch_dim_out=patch_dim_out,
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_encoder_layers=saved_args["enc_layers"],
        num_decoder_layers=saved_args["dec_layers"],
        dim_feedforward=saved_args["d_model"] * 4,
        encoder_days=in_days,
        decoder_days=decoder_days,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Device={device}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  in_days={in_days}  lead={lead_start}–{lead_end}  patch_size={P}")
    print(f"  norm means={meteo_means.round(3)}  stds={meteo_stds.round(3)}")

    # Build full file indices
    fwi_dict = {}
    for p in sorted(glob.glob(os.path.join(fwi_dir, "*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            fwi_dict[d] = p
    d2m_dict = _build_file_dict(obs_root, "2d")
    t2m_dict = _build_file_dict(obs_root, "2t")

    first_fwi = sorted(fwi_dict.values())[0]
    with rasterio.open(first_fwi) as src:
        profile = src.profile
        H, W = profile["height"], profile["width"]
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="lzw")

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
        print(f"\n{'='*60}\n  Year {year}: May 1 – Oct 31\n{'='*60}")
        pred_start   = date(year, 5, 1)
        pred_end     = date(year, 10, 31)
        # need in_days history + lead_end future days
        data_start   = pred_start - timedelta(days=in_days + 5)
        required_end = pred_end   + timedelta(days=lead_end + 5)

        fwi_p, t2m_p, d2m_p, dates_y = [], [], [], []
        cur = data_start
        while cur <= required_end:
            if cur in fwi_dict and cur in t2m_dict and cur in d2m_dict:
                fwi_p.append(fwi_dict[cur]); t2m_p.append(t2m_dict[cur])
                d2m_p.append(d2m_dict[cur]); dates_y.append(cur)
            cur += timedelta(days=1)

        T_y = len(dates_y)
        if T_y < in_days + lead_end + 1:
            print(f"  Only {T_y} aligned days. Skipping {year}.")
            continue
        print(f"  Aligned: {T_y} days  ({dates_y[0]} → {dates_y[-1]})")

        fwi_s = _clean(read_singleband_stack(fwi_p))
        t2m_s = _clean(read_singleband_stack(t2m_p))
        d2m_s = _clean(read_singleband_stack(d2m_p))

        meteo_y = np.empty((T_y, H, W, 3), dtype=np.float32)
        meteo_y[..., 0] = fwi_s; del fwi_s
        meteo_y[..., 1] = t2m_s; del t2m_s
        meteo_y[..., 2] = d2m_s; del d2m_s

        meteo_y -= meteo_means
        meteo_y /= meteo_stds
        np.clip(meteo_y, -10.0, 10.0, out=meteo_y)

        date_to_idx_y = {d: i for i, d in enumerate(dates_y)}
        pred_dates_y  = [pred_start + timedelta(days=k)
                         for k in range((pred_end - pred_start).days + 1)]

        n_done = 0
        for base_date in pred_dates_y:
            if base_date not in date_to_idx_y:
                continue
            base_idx = date_to_idx_y[base_date]
            if base_idx < in_days or base_idx + lead_end + 1 > T_y:
                continue

            enc_hist = meteo_y[base_idx - in_days: base_idx]              # (7, H, W, 3)
            dec_fut  = meteo_y[base_idx + lead_start: base_idx + lead_end + 1]  # (33, H, W, 3)

            enc_patches, pred_hw, pred_grid = patchify(enc_hist, P)  # (n_patches, 7, enc_dim)
            dec_patches, _, _               = patchify(dec_fut,  P)  # (n_patches, 33, dec_dim)

            chunk, n_p = args.pred_batch_size, enc_patches.shape[0]
            prob_list = []
            with torch.no_grad():
                for cs in range(0, n_p, chunk):
                    ce  = min(cs + chunk, n_p)
                    xb_enc = torch.from_numpy(enc_patches[cs:ce].copy()).float().to(device)
                    xb_dec = torch.from_numpy(dec_patches[cs:ce].copy()).float().to(device)
                    logits = model(xb_enc, xb_dec)
                    prob_list.append(torch.sigmoid(logits).cpu().numpy())
            probs = np.concatenate(prob_list, axis=0)   # (n_patches, decoder_days, P²)

            base_str = base_date.strftime("%Y%m%d")
            day_out  = os.path.join(output_dir, base_str)
            os.makedirs(day_out, exist_ok=True)

            for li, lead in enumerate(range(lead_start, lead_end + 1)):
                target_date     = base_date + timedelta(days=lead)
                target_date_str = target_date.strftime("%Y%m%d")
                out_path = os.path.join(
                    day_out, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
                )
                prob_patches_lead = probs[:, li, :]        # (n_patches, P²)
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
            if n_done % 20 == 0 or base_date == pred_dates_y[-1]:
                print(f"  [{n_done}/{len(pred_dates_y)}] {base_date} → {decoder_days} tifs")

        del meteo_y
        print(f"  Year {year}: {n_done} base dates, {n_done * decoder_days} tifs")

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
        description="Train S2S Hotspot Transformer [CWFIS Hotspot, Lead 14–46 Days]"
    )
    add_config_argument(ap)

    # Data
    ap.add_argument("--data_start",   type=str,   default="2018-05-01",
                    help="First date for data loading. Same span as 7-day model. "
                         "Narrow (e.g. 2021-05-01) to reduce RAM if needed.")
    ap.add_argument("--pred_start",   type=str,   default="2022-05-01",
                    help="First prediction date; also train/val split boundary. "
                         "Default 2022 gives 3 years of val (2022-2024) "
                         "vs 4 years of train (2018-2022).")
    ap.add_argument("--pred_end",     type=str,   default="2024-10-31",
                    help="Last prediction date (inclusive). Dates near pred_end "
                         "where dec_end > T are silently skipped.")
    ap.add_argument("--in_days",      type=int,   default=7,
                    help="Encoder history days (default=7).")
    ap.add_argument("--lead_start",   type=int,   default=14,
                    help="First forecast lead time in days (default=14).")
    ap.add_argument("--lead_end",     type=int,   default=46,
                    help="Last forecast lead time in days (default=46).")

    # Spatial dilation
    ap.add_argument("--dilate_radius", type=int,  default=14,
                    help="Hotspot label dilation radius in pixels. "
                         "Default=14 (28 km buffer, matching FWI effective resolution). "
                         "Each pixel = 2 km; radius 14 = 28 km sphere enclosing "
                         "the ERA5 native grid cell used by FWI.")

    # Model
    ap.add_argument("--patch_size",   type=int,   default=16)
    ap.add_argument("--d_model",      type=int,   default=256)
    ap.add_argument("--nhead",        type=int,   default=8)
    ap.add_argument("--enc_layers",   type=int,   default=4)
    ap.add_argument("--dec_layers",   type=int,   default=4)

    # Training
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--neg_ratio",    type=float, default=20.0,
                    help="Negative patches per positive patch (default=20).")
    ap.add_argument("--pos_weight_cap", type=float, default=10.0,
                    help="Maximum BCE pos_weight (default=10). Reduces over-confidence.")

    # Forecast
    ap.add_argument("--pred_batch_size", type=int, default=256,
                    help="Patch chunk size for GPU inference (default=256).")
    ap.add_argument("--forecast_only", action="store_true",
                    help="Skip training — load best checkpoint and generate forecast tifs.")
    ap.add_argument("--forecast_years", type=str, default=None,
                    help="Comma-separated years for --forecast_only, e.g. '2023,2024'.")

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
    output_dir  = os.path.join(get_path(cfg, "output_dir"), "s2s_hotspot_cwfis_fire_prob")
    ckpt_dir    = os.path.join(get_path(cfg, "checkpoint_dir"), "s2s_hotspot_cwfis")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    if args.forecast_only:
        _run_forecast_only(args, cfg, fwi_dir, obs_root, output_dir, ckpt_dir)
        return

    run_stamp     = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_meta_path = os.path.join(ckpt_dir, f"run_{run_stamp}.json")
    run_meta = {
        "run_started_at_utc": run_started_iso,
        "label_source":       "CWFIS satellite hotspots (VIIRS-M)",
        "model":              "S2SHotspotTransformer",
        "decoder_mode":       "era5_oracle",
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
    lead_start      = args.lead_start
    lead_end        = args.lead_end
    decoder_days    = lead_end - lead_start + 1

    print("\n" + "=" * 70)
    print("S2S HOTSPOT TRANSFORMER  [CWFIS Hotspot + Lead 14–46 + ERA5 Oracle]")
    print("=" * 70)
    print(f"  data_start        : {data_start_date}")
    print(f"  pred_start        : {pred_start_date}  (train/val split boundary)")
    print(f"  pred_end          : {pred_end_date}")
    print(f"  in_days           : {in_days}  (encoder history)")
    print(f"  lead_start–end    : {lead_start}–{lead_end}  (decoder_days={decoder_days})")
    print(f"  patch_size        : {args.patch_size}")
    print(f"  dilate_radius     : {args.dilate_radius} px "
          f"(≈{args.dilate_radius * 2} km, matching FWI effective resolution)")
    print(f"  neg_ratio         : {args.neg_ratio}")
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

    # We need history BEFORE data_start and labels AFTER pred_end
    # aligned_dates must cover: [data_start – in_days, pred_end + lead_end]
    # We keep data_start as the outer boundary; the inner code checks
    # that windows don't go out of range.
    required_end = pred_end_date + timedelta(days=lead_end + 5)

    fwi_paths, t2m_paths, d2m_paths, aligned_dates = [], [], [], []
    cur = data_start_date
    while cur <= required_end:
        if cur in fwi_dict and cur in t2m_dict and cur in d2m_dict:
            fwi_paths.append(fwi_dict[cur])
            t2m_paths.append(t2m_dict[cur])
            d2m_paths.append(d2m_dict[cur])
            aligned_dates.append(cur)
        cur += timedelta(days=1)

    min_needed = in_days + lead_end + 1
    if len(aligned_dates) < min_needed:
        raise RuntimeError(
            f"Only {len(aligned_dates)} aligned days, need >= {min_needed} "
            f"(in_days={in_days} + lead_end={lead_end} + 1)"
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
    print(f"  FWI  range: [{fwi_stack.min():.2f}, {fwi_stack.max():.2f}]")
    print(f"  2t   range: [{t2m_stack.min():.2f}, {t2m_stack.max():.2f}]")
    print(f"  2d   range: [{d2m_stack.min():.2f}, {d2m_stack.max():.2f}]")

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

    fire_stack = rasterize_hotspots_batch(hotspot_df, aligned_dates, profile)  # (T, H, W) uint8
    pos_rate   = fire_stack.mean()
    print(f"  Fire stack shape     : {fire_stack.shape}")
    print(f"  Mean fire pixel rate : {pos_rate:.6%}  (raw hotspot points)")
    run_meta["positive_rate_raw"] = float(pos_rate)

    # -- Spatial dilation: expand each hotspot to a filled circle --
    if args.dilate_radius > 0:
        r = args.dilate_radius
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk   = (xx ** 2 + yy ** 2 <= r ** 2)
        print(f"\n  Dilating fire labels: radius={r} px  "
              f"(~{r * 2} km buffer, kernel {disk.shape[0]}×{disk.shape[1]}, "
              f"{disk.sum()} pixels/hotspot)...")
        t0_dil = time.time()
        for t in range(T):
            if fire_stack[t].any():
                fire_stack[t] = binary_dilation(
                    fire_stack[t], structure=disk
                ).astype(np.uint8)
            if t % 200 == 0 or t == T - 1:
                print(f"    dilate frame {t:4d}/{T}  ({time.time()-t0_dil:.0f}s)")
        pos_rate_dil = fire_stack.mean()
        print(f"  After dilation: positive_rate={pos_rate_dil:.4%}  "
              f"({pos_rate_dil/pos_rate:.1f}× increase)  "
              f"({time.time()-t0_dil:.0f}s)")
        run_meta["dilate_radius"]         = r
        run_meta["positive_rate_dilated"] = float(pos_rate_dil)
    else:
        run_meta["dilate_radius"]         = 0
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

    print(f"  Training : 0 → {train_end_idx - 1}  "
          f"({aligned_dates[0]} → {aligned_dates[train_end_idx-1]})")
    print(f"  Val/pred : {train_end_idx} → {T-1}  "
          f"({aligned_dates[train_end_idx]} → {aligned_dates[-1]})")

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
    meteo_stds  = train_meteo.reshape(-1, 3).std(axis=0)  + 1e-6
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
    n_patches  = grid[0] * grid[1]
    enc_dim    = P * P * 3   # patch_dim_enc = P²×3  (FWI + 2t + 2d)
    dec_dim    = enc_dim     # ERA5 oracle: same 3 channels for decoder
    out_dim    = P * P       # patch_dim_out = P²

    meteo_gb   = T * n_patches * enc_dim * 4 / 1e9
    fire_gb    = T * n_patches * out_dim * 1 / 1e9
    needed_gb  = meteo_gb + fire_gb

    print(f"\n[STEP 7] Pre-computing patches for T={T} frames...")
    print(f"  n_patches={n_patches}  enc_dim={enc_dim}  dec_dim={dec_dim}  out_dim={out_dim}")
    print(f"  Estimated RAM: meteo={meteo_gb:.1f} GB (float32)  "
          f"fire={fire_gb:.1f} GB (uint8)  total={needed_gb:.1f} GB")
    print(f"  NOTE: In ERA5 oracle mode, the same meteo_patched array serves as")
    print(f"        both encoder (past 7 days) and decoder (future {decoder_days} days).")

    try:
        import psutil
        vm           = psutil.virtual_memory()
        available_gb = vm.available / 1e9
        total_gb     = vm.total    / 1e9
        print(f"  System RAM: {available_gb:.1f} GB available / {total_gb:.1f} GB total")
        if needed_gb > total_gb * 0.85:
            bytes_per_frame = n_patches * (enc_dim * 4 + out_dim * 1)
            safe_T    = int(total_gb * 0.80 * 1e9 / bytes_per_frame)
            safe_date = aligned_dates[max(0, T - safe_T)]
            print(f"\n  *** OOM WARNING ***")
            print(f"  Need {needed_gb:.1f} GB but machine total is only {total_gb:.1f} GB.")
            print(f"  Suggested: --data_start {safe_date}  (reduces T to ~{safe_T} days)")
            print(f"  Aborting. Re-run with a later --data_start.\n")
            raise SystemExit(1)
        elif needed_gb > available_gb:
            print(f"  NOTE: Need {needed_gb:.1f} GB, currently {available_gb:.1f} GB free "
                  f"(total {total_gb:.1f} GB). OS will reclaim cache — continuing.")
    except ImportError:
        print("  (pip install psutil to enable RAM pre-flight check)")

    # -- meteo_patched: (T, n_patches, enc_dim) float32 --
    t0_meteo = time.time()
    try:
        meteo_patched = np.empty((T, n_patches, enc_dim), dtype=np.float32)
    except MemoryError:
        raise RuntimeError(
            f"MemoryError allocating meteo_patched ({meteo_gb:.1f} GB). "
            f"Try a later --data_start date."
        )
    for t in range(T):
        patches, _, _ = patchify(meteo_norm[t:t + 1], P)
        meteo_patched[t] = patches[:, 0, :]
        if t % 100 == 0 or t == T - 1:
            print(f"  meteo frame {t:4d}/{T}  ({time.time()-t0_meteo:.0f}s)")
    print(f"  meteo_patched: {meteo_patched.shape}  "
          f"{meteo_patched.nbytes/1e9:.1f} GB  ({time.time()-t0_meteo:.0f}s)")

    # -- fire_patched: (T, n_patches, P²) uint8 --
    t0_fire = time.time()
    try:
        fire_patched = np.empty((T, n_patches, out_dim), dtype=np.uint8)
    except MemoryError:
        raise RuntimeError(
            f"MemoryError allocating fire_patched ({fire_gb:.1f} GB as uint8). "
            f"Try a later --data_start date."
        )
    for t in range(T):
        fut1 = fire_stack[t:t + 1].astype(np.float32)
        patches, _, _ = patchify(fut1, P)
        fire_patched[t] = patches[:, 0, :].astype(np.uint8)
        if t % 100 == 0 or t == T - 1:
            print(f"  fire  frame {t:4d}/{T}  ({time.time()-t0_fire:.0f}s)")
    print(f"  fire_patched:  {fire_patched.shape}  dtype=uint8  "
          f"{fire_patched.nbytes/1e9:.1f} GB  ({time.time()-t0_fire:.0f}s)")
    del meteo_norm  # free the (T, H, W, 3) array — we keep the patched version

    # -- Build S2S windows (with gap between encoder and decoder) --
    all_windows = _build_s2s_windows(T, in_days, lead_start, lead_end)
    train_wins  = [w for w in all_windows
                   if aligned_dates[w[1] - 1] < pred_start_date]
    val_wins    = [w for w in all_windows
                   if aligned_dates[w[0]] >= pred_start_date]
    print(f"\n  S2S windows built (enc_days={in_days}, gap={lead_start-1}, "
          f"target_days={decoder_days})")
    print(f"  Total: {len(all_windows)}  train: {len(train_wins)}  val: {len(val_wins)}")

    # ----------------------------------------------------------------
    # STEP 7b  Build positive pairs; sample negative pairs
    # ----------------------------------------------------------------
    print("\n[STEP 7b] Filtering positive patches + sampling negative patches...")
    print(f"  Positive = any fire pixel in TARGET window (lead {lead_start}–{lead_end})")
    t0_filter = time.time()

    pos_pairs = []
    for win_i, (hs, he, ts, te) in enumerate(train_wins):
        for patch_i in range(n_patches):
            # Check TARGET window (not encoder window) for fire pixels
            if fire_patched[ts:te, patch_i, :].max() > 0:
                pos_pairs.append((win_i, patch_i))
        if win_i % 200 == 0 or win_i == len(train_wins) - 1:
            print(f"  scanned {win_i+1}/{len(train_wins)} windows  "
                  f"pos_pairs: {len(pos_pairs):,}  ({time.time()-t0_filter:.0f}s)")

    total_pairs = len(train_wins) * n_patches
    pct = 100.0 * len(pos_pairs) / max(total_pairs, 1)
    print(f"  Positive pairs: {len(pos_pairs):,} / {total_pairs:,}  ({pct:.3f}%)")

    if len(pos_pairs) == 0:
        raise RuntimeError(
            "No positive (window, patch) pairs found. "
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
    pos_pixels        = 0
    neg_pixels_in_pos = 0
    for win_i, patch_i in pos_pairs:
        hs, he, ts, te = train_wins[win_i]
        pf = fire_patched[ts:te, patch_i, :]   # (decoder_days, P²)
        p  = int(pf.sum())
        pos_pixels        += p
        neg_pixels_in_pos += pf.size - p
    neg_pixels_total = neg_pixels_in_pos + len(neg_pairs) * out_dim * decoder_days
    raw_ratio        = neg_pixels_total / max(pos_pixels, 1)
    pos_weight_val   = min(raw_ratio, args.pos_weight_cap)
    print(f"  pos_pixels : {pos_pixels:,}")
    print(f"  neg_pixels : {neg_pixels_total:,}")
    print(f"  raw ratio  : {raw_ratio:.1f}   pos_weight (capped {args.pos_weight_cap}) = {pos_weight_val:.2f}")
    run_meta["pos_weight"] = pos_weight_val

    # ----------------------------------------------------------------
    # Build datasets and dataloaders
    # ----------------------------------------------------------------
    train_ds = S2SHotspotDatasetMixed(
        meteo_patched, fire_patched, train_wins, hw, grid, all_pairs
    )
    if val_wins:
        val_ds = S2SHotspotDatasetUnfiltered(
            meteo_patched, fire_patched, val_wins, hw, grid
        )
    else:
        val_ds = train_ds
        print("  WARNING: no val windows — using train set as val proxy")

    patch_dim_enc = enc_dim    # P²×3
    patch_dim_dec = dec_dim    # P²×3  (ERA5 oracle — same as encoder)
    patch_dim_out = out_dim    # P²

    print(f"\n  Train samples (mixed pos+neg): {len(train_ds):,}")
    print(f"  Val   samples (unfiltered)  : {len(val_ds):,}")
    print(f"  Grid: {grid[0]}×{grid[1]} patches/frame  "
          f"(enc_dim={patch_dim_enc}  dec_dim={patch_dim_dec}  out_dim={patch_dim_out})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----------------------------------------------------------------
    # STEP 9  Build model & train
    # ----------------------------------------------------------------
    print("\n[STEP 9] Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = S2SHotspotTransformer(
        patch_dim_enc=patch_dim_enc,
        patch_dim_dec=patch_dim_dec,
        patch_dim_out=patch_dim_out,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.d_model * 4,
        encoder_days=in_days,
        decoder_days=decoder_days,
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
        # -- Training --
        model.train()
        train_loss    = 0.0
        train_samples = 0
        for xb_enc, xb_dec, yb in train_dl:
            xb_enc, xb_dec, yb = xb_enc.to(device), xb_dec.to(device), yb.to(device)
            logits = model(xb_enc, xb_dec)            # (B, decoder_days, P²)
            loss   = criterion(logits, yb)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            _gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(_gnorm):
                optimizer.zero_grad()
                continue
            optimizer.step()
            train_loss    += loss.item() * xb_enc.size(0)
            train_samples += xb_enc.size(0)

        if train_samples == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"*** ALL TRAIN BATCHES NaN — stopping.")
            break
        train_loss /= train_samples

        # -- Validation --
        model.eval()
        val_loss    = 0.0
        val_samples = 0
        _lfire_sum, _lfire_n = 0.0, 0
        _lbg_sum,   _lbg_n   = 0.0, 0
        with torch.no_grad():
            for xb_enc, xb_dec, yb in val_dl:
                xb_enc, xb_dec, yb = xb_enc.to(device), xb_dec.to(device), yb.to(device)
                logits_v = model(xb_enc, xb_dec)
                vl = criterion(logits_v, yb)
                if torch.isfinite(vl):
                    val_loss    += vl.item() * xb_enc.size(0)
                    val_samples += xb_enc.size(0)
                # Logit diagnostics
                _lv = logits_v.detach().float()
                _yb = yb.detach().bool()
                _f  = _lv[_yb]
                _bg = _lv[~_yb]
                if _f.numel() > 0:
                    _lfire_sum += _f.sum().item()
                    _lfire_n   += _f.numel()
                _lbg_sum += _bg.sum().item()
                _lbg_n   += _bg.numel()

        if val_samples == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  "
                  f"val=NaN — stopping early.")
            break
        val_loss /= val_samples

        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}")
        if _lfire_n > 0:
            print(f"           logit  fire={_lfire_sum/_lfire_n:+.3f}  "
                  f"bg={_lbg_sum/_lbg_n:+.3f}  "
                  f"({_lfire_n:,} fire px / {_lbg_n:,} bg px)")

        if val_samples > 0 and np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "meteo_means":   meteo_means,
                "meteo_stds":    meteo_stds,
                "patch_dim_enc": patch_dim_enc,
                "patch_dim_dec": patch_dim_dec,
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

    pred_dates  = []
    cur = pred_start_date
    while cur <= pred_end_date:
        pred_dates.append(cur)
        cur += timedelta(days=1)

    date_to_idx = {d: i for i, d in enumerate(aligned_dates)}

    # Rebuild meteo_norm from patched data is not feasible after del.
    # Instead we re-load from patched arrays for inference.
    # For each base_date: gather encoder (past 7 days) and decoder (future 33 days)
    # from meteo_patched, then depatchify or run patch-level inference.

    n_done = 0
    for base_date in pred_dates:
        if base_date not in date_to_idx:
            continue
        base_idx = date_to_idx[base_date]
        # encoder indices
        enc_start = base_idx - in_days
        enc_end   = base_idx
        # decoder indices (ERA5 oracle: future lead_start..lead_end)
        dec_start = base_idx + lead_start
        dec_end   = base_idx + lead_end + 1

        if enc_start < 0 or dec_end > T:
            continue

        # Process in chunks: slice patch axis (axis=1) then transpose to
        # (chunk, days, dim).  np.ascontiguousarray ensures the memory is
        # C-contiguous before torch.from_numpy (required by PyTorch).
        chunk = args.pred_batch_size
        n_p   = n_patches
        prob_list = []
        with torch.no_grad():
            for cs in range(0, n_p, chunk):
                ce = min(cs + chunk, n_p)
                xb_enc = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[enc_start:enc_end, cs:ce, :].transpose(1, 0, 2)
                    )
                ).float().to(device)                    # (chunk, in_days, enc_dim)
                xb_dec = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[dec_start:dec_end, cs:ce, :].transpose(1, 0, 2)
                    )
                ).float().to(device)                    # (chunk, decoder_days, dec_dim)
                logits = model(xb_enc, xb_dec)          # (chunk, dec_days, P²)
                prob_list.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(prob_list, axis=0)   # (n_patches, decoder_days, P²)

        base_str = base_date.strftime("%Y%m%d")
        day_out  = os.path.join(output_dir, base_str)
        os.makedirs(day_out, exist_ok=True)

        for li, lead in enumerate(range(lead_start, lead_end + 1)):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            out_path        = os.path.join(
                day_out, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
            )
            prob_patches_lead = probs[:, li, :]       # (n_patches, P²)
            prob_vol = depatchify(
                prob_patches_lead[:, np.newaxis, :],
                grid, P, hw, num_channels=1
            )
            prob_map = prob_vol[0] if prob_vol.ndim == 3 else prob_vol
            if prob_map.shape != (H, W):
                full = np.zeros((H, W), dtype=np.float32)
                full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                prob_map = full
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(prob_map.astype(np.float32), 1)

        n_done += 1
        if n_done % 20 == 0 or base_date == pred_dates[-1]:
            print(f"  [{n_done}/{len(pred_dates)}] {base_date} → {decoder_days} lead tifs "
                  f"(lead {lead_start}–{lead_end})")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE  [S2S Hotspot, ERA5 Oracle Decoder]")
    print(f"  Forecasts  : {output_dir}")
    print(f"  Checkpoint : {best_ckpt}")
    print(f"  Lead range : {lead_start}–{lead_end} days")
    print("=" * 70)

    run_meta["status"]           = "success"
    run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run log: {run_meta_path}")


if __name__ == "__main__":
    main()
