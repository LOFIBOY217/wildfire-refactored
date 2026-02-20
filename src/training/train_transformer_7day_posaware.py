"""
Train 7-Day Fire Probability Transformer  (Positive-Aware Sampling Version)
============================================================================
This is a variant of train_transformer_7day_precompute.py that fixes the
extreme class imbalance problem (positive rate ≈ 0.000742%) by filtering
training samples to only patches that contain at least one positive fire pixel.

Problem with focal/precompute versions:
    Each 16×16 patch = 256 pixels → avg 0.0019 fire pixels/patch.
    A batch of 256 patches has on average only 0.46 positive pixels.
    Most batches have ZERO positive samples → zero gradient signal.
    The model finds the trivial solution (predict 0 everywhere) and
    training loss collapses to near-0. AUC ≈ 0.50 (random).

Fix — Positive-Aware Patch Filtering:
    After pre-computing fire_patched (T, n_patches, 256), scan all
    training (window, patch) pairs and keep only those where the fire
    label contains ≥1 positive pixel.

    Within a positive-containing patch: neg:pos ≈ 3:1 → manageable with
    standard BCEWithLogitsLoss + true pos_weight (no clamping needed).

    Validation set remains UNFILTERED for honest evaluation on all pixels.

All other logic (pre-computed patches, STEP 1-6, STEP 10) is identical
to train_transformer_7day_precompute.py.

Output format (compatible with evaluate_forecast.py):
    outputs/transformer7d_fire_prob_posaware/YYYYMMDD/fire_prob_lead{k:02d}d_YYYYMMDD.tif

Usage:
    python -m src.training.train_transformer_7day_posaware \\
        --config configs/paths_windows.yaml \\
        --data_start 2023-05-05 \\
        --pred_start 2025-07-01 \\
        --pred_end   2025-08-14
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
from src.data_ops.processing.rasterize_fires import load_ciffc_data, rasterize_fires_batch
from src.models.transformer_7day import FireProb7DayTransformer


# ------------------------------------------------------------------ #
# Datasets
# ------------------------------------------------------------------ #

class Fire7DayDatasetPosAware(Dataset):
    """
    Training dataset: only (window, patch) pairs containing ≥1 positive pixel.

    pos_pairs : list of (win_i, patch_i) indices pre-filtered from training windows.
    Every sample returned is guaranteed to have at least one fire pixel,
    ensuring every training batch contains meaningful gradient signal.
    """

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid, pos_pairs):
        self.meteo     = meteo_patched
        self.fire      = fire_patched
        self.windows   = windows
        self.hw        = hw
        self.grid      = grid
        self.pos_pairs = pos_pairs  # list of (win_i, patch_i)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        win_i, patch_i = self.pos_pairs[idx]
        hs, he, fs, fe = self.windows[win_i]
        x = self.meteo[hs:he, patch_i, :]   # (in_days, in_dim)
        y = self.fire[fs:fe, patch_i, :]     # (out_days, out_dim)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


class Fire7DayDatasetPrecompute(Dataset):
    """
    Unfiltered dataset used for validation — evaluates on all patches
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
        x = self.meteo[hs:he, patch_i, :]
        y = self.fire[fs:fe, patch_i, :]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


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
# Main
# ------------------------------------------------------------------ #

def main():
    run_started_at  = time.time()
    run_started_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"

    ap = argparse.ArgumentParser(
        description="Train 7-Day Fire Probability Transformer (Positive-Aware Sampling)"
    )
    add_config_argument(ap)
    ap.add_argument("--data_start",   type=str,   default="2023-05-05")
    ap.add_argument("--pred_start",   type=str,   default="2025-07-01")
    ap.add_argument("--pred_end",     type=str,   default="2025-08-14")
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------
    # Config & paths
    # ----------------------------------------------------------------
    cfg       = load_config(args.config)
    fwi_dir   = get_path(cfg, "fwi_dir")
    paths_cfg = cfg.get("paths", {})
    obs_root  = get_path(cfg, "observation_dir") if "observation_dir" in paths_cfg \
                else get_path(cfg, "ecmwf_dir")
    ciffc_csv  = get_path(cfg, "ciffc_csv")
    output_dir = os.path.join(get_path(cfg, "output_dir"),
                              "transformer7d_fire_prob_posaware")
    ckpt_dir   = os.path.join(get_path(cfg, "checkpoint_dir"),
                              "transformer_7day_posaware")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    run_stamp     = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_meta_path = os.path.join(ckpt_dir, f"run_{run_stamp}.json")
    run_meta = {
        "run_started_at_utc": run_started_iso,
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
    print("7-DAY FIRE PROBABILITY TRANSFORMER  [Positive-Aware Sampling]")
    print("=" * 70)
    print(f"  data_start        : {data_start_date}")
    print(f"  pred_start        : {pred_start_date}  (train/val split boundary)")
    print(f"  pred_end          : {pred_end_date}")
    print(f"  in_days / out_days: {in_days} / {out_days}")
    print(f"  patch_size        : {args.patch_size}")
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
        fill = float(np.nanmean(stack))
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
    # STEP 4  Rasterize CIFFC fires
    # ----------------------------------------------------------------
    print("\n[STEP 4] Rasterizing CIFFC fire records...")
    ciffc_df   = load_ciffc_data(ciffc_csv)
    fire_stack = rasterize_fires_batch(ciffc_df, aligned_dates, profile)
    pos_rate   = fire_stack.mean()
    print(f"  Fire records: {len(ciffc_df)}  positive_rate: {pos_rate:.6%}")
    run_meta["fire_records"]  = int(len(ciffc_df))
    run_meta["positive_rate"] = float(pos_rate)

    # ----------------------------------------------------------------
    # STEP 5  Find train/val split index
    # ----------------------------------------------------------------
    print("\n[STEP 5] Splitting train / val by pred_start...")

    train_end_idx = None
    for i, d in enumerate(aligned_dates):
        if d >= pred_start_date:
            train_end_idx = i
            break
    if train_end_idx is None:
        raise RuntimeError(
            f"pred_start={pred_start_date} is beyond all aligned dates"
        )
    print(f"  Training dates: 0 -> {train_end_idx - 1}  "
          f"({aligned_dates[0]} -> {aligned_dates[train_end_idx-1]})")
    print(f"  Val/pred dates: {train_end_idx} -> {T-1}  "
          f"({aligned_dates[train_end_idx]} -> {aligned_dates[-1]})")

    # ----------------------------------------------------------------
    # STEP 6  Standardise meteorological data (stats from training only)
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
    print("\n[STEP 7] Pre-computing patches for all T frames...")
    print("  (__getitem__ will be O(1) array indexing thereafter)")

    P = args.patch_size

    _sample, hw, grid = patchify(meteo_norm[:1], P)
    n_patches = grid[0] * grid[1]
    in_dim    = P * P * 3
    out_dim   = P * P

    t0_meteo = time.time()
    meteo_patched = np.empty((T, n_patches, in_dim), dtype=np.float32)
    for t in range(T):
        patches, _, _ = patchify(meteo_norm[t:t + 1], P)
        meteo_patched[t] = patches[:, 0, :]
        if t % 50 == 0 or t == T - 1:
            print(f"  meteo frame {t:4d}/{T}  ({time.time()-t0_meteo:.0f}s elapsed)")
    print(f"  meteo_patched: {meteo_patched.shape}  "
          f"{meteo_patched.nbytes/1e9:.1f} GB  ({time.time()-t0_meteo:.0f}s)")

    t0_fire = time.time()
    fire_patched = np.empty((T, n_patches, out_dim), dtype=np.float32)
    for t in range(T):
        fut1 = fire_stack[t:t + 1].astype(np.float32)
        patches, _, _ = patchify(fut1, P)
        fire_patched[t] = patches[:, 0, :]
        if t % 50 == 0 or t == T - 1:
            print(f"  fire  frame {t:4d}/{T}  ({time.time()-t0_fire:.0f}s elapsed)")
    print(f"  fire_patched:  {fire_patched.shape}  "
          f"{fire_patched.nbytes/1e9:.1f} GB  ({time.time()-t0_fire:.0f}s)")

    all_windows = _build_windows(T, in_days, out_days)
    print(f"\n  Total time windows: {len(all_windows)}")

    train_wins = [w for w in all_windows
                  if aligned_dates[w[1] - 1] < pred_start_date and w[3] <= T]
    val_wins   = [w for w in all_windows
                  if aligned_dates[w[0]] >= pred_start_date and w[3] <= T]

    print(f"  Train windows: {len(train_wins)}  Val windows: {len(val_wins)}")

    # ----------------------------------------------------------------
    # STEP 7b  Filter training pairs to positive-containing patches
    # ----------------------------------------------------------------
    print("\n[STEP 7b] Filtering training pairs to positive-containing patches...")
    t0_filter = time.time()
    pos_pairs = []
    for win_i, (hs, he, fs, fe) in enumerate(train_wins):
        for patch_i in range(n_patches):
            if fire_patched[fs:fe, patch_i, :].max() > 0:
                pos_pairs.append((win_i, patch_i))
        if win_i % 100 == 0 or win_i == len(train_wins) - 1:
            print(f"  scanned {win_i+1}/{len(train_wins)} windows  "
                  f"pos_pairs so far: {len(pos_pairs):,}  "
                  f"({time.time()-t0_filter:.0f}s)")

    total_pairs = len(train_wins) * n_patches
    pct = 100.0 * len(pos_pairs) / max(total_pairs, 1)
    print(f"  Positive pairs: {len(pos_pairs):,} / {total_pairs:,}  ({pct:.3f}%)")
    print(f"  Filter time: {time.time()-t0_filter:.0f}s")
    run_meta["pos_pairs"]   = len(pos_pairs)
    run_meta["total_pairs"] = total_pairs

    if len(pos_pairs) == 0:
        raise RuntimeError(
            "No positive (window, patch) pairs found in training set. "
            "Check ciffc_csv and date alignment."
        )

    # ----------------------------------------------------------------
    # STEP 8  Compute pos_weight from filtered patches; build criterion
    # ----------------------------------------------------------------
    print("\n[STEP 8] Computing BCE pos_weight from positive-containing patches...")
    pos_pixels = 0
    neg_pixels = 0
    for win_i, patch_i in pos_pairs:
        hs, he, fs, fe = train_wins[win_i]
        patch_fire = fire_patched[fs:fe, patch_i, :]
        p = int(patch_fire.sum())
        pos_pixels += p
        neg_pixels += patch_fire.size - p

    pos_weight_val = min(neg_pixels / max(pos_pixels, 1), 10.0)
    print(f"  Within positive patches:  pos={pos_pixels:,}  neg={neg_pixels:,}")
    print(f"  pos_weight = {pos_weight_val:.2f}  (capped at 10; raw ratio = {neg_pixels/max(pos_pixels,1):.1f})")
    run_meta["pos_weight"] = pos_weight_val

    # ----------------------------------------------------------------
    # Build datasets and dataloaders
    # ----------------------------------------------------------------
    train_ds = Fire7DayDatasetPosAware(
        meteo_patched, fire_patched, train_wins, hw, grid, pos_pairs
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

    print(f"\n  Train samples (pos-filtered): {len(train_ds):,}")
    print(f"  Val   samples (unfiltered)  : {len(val_ds):,}")
    print(f"  Grid: {grid[0]}x{grid[1]} patches/frame  "
          f"(dim_in={patch_dim_in}  dim_out={patch_dim_out})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

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
        # -- train --
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # -- val --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss:
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
    # STEP 10  Generate prediction tifs
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

        hist = meteo_norm[base_idx - in_days: base_idx]
        xp, pred_hw, pred_grid = patchify(hist, P)

        xb = torch.from_numpy(xp).float().to(device)
        with torch.no_grad():
            logits = model(xb)
            probs  = torch.sigmoid(logits).cpu().numpy()

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
            if prob_vol.ndim == 3:
                prob_map = prob_vol[0]
            else:
                prob_map = prob_vol

            if prob_map.shape != (H, W):
                full = np.zeros((H, W), dtype=np.float32)
                full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                prob_map = full

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(prob_map.astype(np.float32), 1)

        print(f"  [DONE] {base_date} -> {out_days} lead tifs")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
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
