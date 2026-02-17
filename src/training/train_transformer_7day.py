"""
Train 7-Day Fire Probability Transformer
=========================================
Fair comparison against the Logistic Regression baseline:

  Inputs (identical to Logistic):
    - FWI rasters          (past 7 days)
    - 2m temperature       (past 7 days, from ECMWF reanalysis observation)
    - 2m dewpoint          (past 7 days, from ECMWF reanalysis observation)

  Target (identical to Logistic):
    - CIFFC rasterized fire labels (binary, 7 future days)

  Output format (identical to Logistic, compatible with evaluate_forecast.py):
    outputs/transformer7d_fire_prob/YYYYMMDD/fire_prob_lead{k:02d}d_YYYYMMDD.tif

Usage:
    python -m src.training.train_transformer_7day \\
        --config configs/default.yaml \\
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
from src.utils.raster_io import read_singleband_stack
from src.utils.patch_utils import patchify, depatchify
from src.utils.normalization import standardize
from src.data_ops.processing.rasterize_fires import load_ciffc_data, rasterize_fires_batch
from src.models.transformer_7day import FireProb7DayTransformer


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class Fire7DayDataset(Dataset):
    """
    Each sample is one spatial patch across time.

    X: (in_days, patch_dim_in)  standardised FWI+2t+2d patches
    Y: (out_days, patch_dim_out) binary fire label patches (float32)
    """

    def __init__(self, meteo_patches, fire_patches):
        """
        Args:
            meteo_patches: (N_patches, in_days,  patch_dim_in)  float32
            fire_patches:  (N_patches, out_days, patch_dim_out) float32
        """
        self.X = torch.from_numpy(meteo_patches.astype(np.float32))
        self.Y = torch.from_numpy(fire_patches.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _file_index(directory, pattern, obs_root=None):
    """
    Build {datetime.date: filepath} index.
    Tries subdirectory pattern first, then flat pattern.
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths and obs_root:
        flat_pattern = pattern.replace("*/", "")
        paths = sorted(glob.glob(os.path.join(obs_root, flat_pattern)))
    result = {}
    for p in paths:
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            result[d] = p
    return result


def _build_windows(n_days, in_days, out_days):
    """Return list of (hist_start, hist_end, fut_start, fut_end) index tuples."""
    windows = []
    for i in range(in_days, n_days - out_days + 1):
        windows.append((i - in_days, i, i, i + out_days))
    return windows


def _stack_meteo(fwi_stack, t2m_stack, d2m_stack):
    """
    Stack three [T,H,W] arrays into [T,H,W,3].
    Channels: [FWI, 2t, 2d]
    """
    return np.stack([fwi_stack, t2m_stack, d2m_stack], axis=-1)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    run_started_at = time.time()
    run_started_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"

    ap = argparse.ArgumentParser(description="Train 7-Day Fire Probability Transformer")
    add_config_argument(ap)
    ap.add_argument("--data_start", type=str, default="2023-05-05",
                    help="First date for data loading (YYYY-MM-DD)")
    ap.add_argument("--pred_start", type=str, default="2025-07-01",
                    help="First prediction date; also used as train/val split boundary")
    ap.add_argument("--pred_end",   type=str, default="2025-08-14",
                    help="Last prediction date (inclusive)")
    ap.add_argument("--in_days",    type=int, default=7,  help="History window (days)")
    ap.add_argument("--out_days",   type=int, default=7,  help="Forecast horizon (days)")
    ap.add_argument("--patch_size", type=int, default=16, help="Spatial patch size")
    ap.add_argument("--d_model",    type=int, default=128)
    ap.add_argument("--nhead",      type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=2)
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--num_workers",type=int, default=0)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    fwi_dir        = get_path(cfg, "fwi_dir")
    obs_root       = get_path(cfg, "observation_dir") if "observation_dir" in cfg.get("paths", {}) \
                     else get_path(cfg, "ecmwf_dir")
    ciffc_csv      = get_path(cfg, "ciffc_csv")
    output_dir     = os.path.join(get_path(cfg, "output_dir"), "transformer7d_fire_prob")
    ckpt_dir       = os.path.join(get_path(cfg, "checkpoint_dir"), "transformer_7day")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    run_stamp    = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
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

    # Parse dates
    def _date(s):
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))

    data_start_date  = _date(args.data_start)
    pred_start_date  = _date(args.pred_start)
    pred_end_date    = _date(args.pred_end)

    in_days  = args.in_days
    out_days = args.out_days

    print("\n" + "=" * 70)
    print("7-DAY FIRE PROBABILITY TRANSFORMER")
    print("=" * 70)
    print(f"  data_start : {data_start_date}")
    print(f"  pred_start : {pred_start_date}  (train/val split boundary)")
    print(f"  pred_end   : {pred_end_date}")
    print(f"  in_days    : {in_days}   out_days: {out_days}")
    print(f"  patch_size : {args.patch_size}")
    print(f"  d_model    : {args.d_model}  nhead: {args.nhead}")
    print(f"  epochs     : {args.epochs}  batch: {args.batch_size}  lr: {args.lr}")
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

    d2m_dict = {}
    for p in sorted(glob.glob(os.path.join(obs_root, "2d", "2d_*.tif"))
                    or glob.glob(os.path.join(obs_root, "2d_*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            d2m_dict[d] = p
    if not d2m_dict:
        for p in sorted(glob.glob(os.path.join(obs_root, "2d_*.tif"))):
            d = extract_date_from_filename(os.path.basename(p))
            if d:
                d2m_dict[d] = p

    t2m_dict = {}
    for p in sorted(glob.glob(os.path.join(obs_root, "2t", "2t_*.tif"))
                    or glob.glob(os.path.join(obs_root, "2t_*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            t2m_dict[d] = p
    if not t2m_dict:
        for p in sorted(glob.glob(os.path.join(obs_root, "2t_*.tif"))):
            d = extract_date_from_filename(os.path.basename(p))
            if d:
                t2m_dict[d] = p

    if not fwi_dict:
        raise RuntimeError(f"No FWI files in {fwi_dir}")
    if not d2m_dict:
        raise RuntimeError(f"No 2d files under {obs_root}")
    if not t2m_dict:
        raise RuntimeError(f"No 2t files under {obs_root}")

    print(f"  FWI: {len(fwi_dict)} days  2t: {len(t2m_dict)} days  2d: {len(d2m_dict)} days")

    # ----------------------------------------------------------------
    # STEP 2  Align dates
    # ----------------------------------------------------------------
    print("\n[STEP 2] Aligning dates...")

    # We need data up to pred_end + out_days (target labels extend beyond pred_end)
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
        raise RuntimeError(f"Only {len(aligned_dates)} aligned days, need ≥ {in_days + out_days}")
    print(f"  Aligned dates: {len(aligned_dates)}  ({aligned_dates[0]} → {aligned_dates[-1]})")

    # ----------------------------------------------------------------
    # STEP 3  Load raster stacks
    # ----------------------------------------------------------------
    print("\n[STEP 3] Loading raster stacks...")
    fwi_stack = read_singleband_stack(fwi_paths)   # [T, H, W]
    t2m_stack = read_singleband_stack(t2m_paths)
    d2m_stack = read_singleband_stack(d2m_paths)
    T, H, W = fwi_stack.shape
    print(f"  Shape: T={T}, H={H}, W={W}")

    with rasterio.open(fwi_paths[0]) as src:
        profile = src.profile

    # ----------------------------------------------------------------
    # STEP 4  Rasterize CIFFC fires
    # ----------------------------------------------------------------
    print("\n[STEP 4] Rasterizing CIFFC fire records...")
    ciffc_df   = load_ciffc_data(ciffc_csv)
    fire_stack = rasterize_fires_batch(ciffc_df, aligned_dates, profile)  # [T, H, W] uint8
    pos_rate   = fire_stack.mean()
    print(f"  Fire records: {len(ciffc_df)}  positive_rate: {pos_rate:.4%}")
    run_meta["fire_records"] = int(len(ciffc_df))
    run_meta["positive_rate"] = float(pos_rate)

    # ----------------------------------------------------------------
    # STEP 5  Find train/val split index
    # ----------------------------------------------------------------
    print("\n[STEP 5] Splitting train / val...")

    train_end_idx = None
    for i, d in enumerate(aligned_dates):
        if d >= pred_start_date:
            train_end_idx = i
            break
    if train_end_idx is None:
        raise RuntimeError(f"pred_start={pred_start_date} not found in aligned dates")

    # ----------------------------------------------------------------
    # STEP 6  Standardise meteorological data (stats from training dates only)
    # ----------------------------------------------------------------
    print("\n[STEP 6] Standardising...")

    # Stack FWI+2t+2d into [T, H, W, 3]
    meteo_stack = _stack_meteo(fwi_stack, t2m_stack, d2m_stack)   # [T, H, W, 3]

    # Compute per-channel stats from training period only
    train_meteo = meteo_stack[:train_end_idx]                       # [T_train, H, W, 3]
    meteo_means = train_meteo.reshape(-1, 3).mean(axis=0)
    meteo_stds  = train_meteo.reshape(-1, 3).std(axis=0) + 1e-6

    meteo_norm = (meteo_stack - meteo_means) / meteo_stds
    meteo_norm = np.clip(meteo_norm, -10, 10).astype(np.float32)

    print(f"  Meteo means: {meteo_means.round(3)}")
    print(f"  Meteo stds:  {meteo_stds.round(3)}")

    run_meta["norm_stats"] = {
        "meteo_means": meteo_means.tolist(),
        "meteo_stds":  meteo_stds.tolist(),
    }
    np.save(os.path.join(ckpt_dir, "norm_stats.npy"),
            np.stack([meteo_means, meteo_stds]))

    # ----------------------------------------------------------------
    # STEP 7  Patchify & build windowed samples
    # ----------------------------------------------------------------
    print("\n[STEP 7] Patchifying...")

    # patchify expects (D, H, W, C) for multi-channel
    # We'll build windows first, then patchify per-window to avoid OOM
    windows = _build_windows(T, in_days, out_days)
    print(f"  Total windows: {len(windows)}")

    # Determine patch grid from a single frame
    sample_frame = meteo_norm[0]                                  # (H, W, 3)
    _, hw, grid = patchify(sample_frame[np.newaxis], args.patch_size)
    n_patches = grid[0] * grid[1]
    P = args.patch_size
    patch_dim_in  = P * P * 3
    patch_dim_out = P * P

    # Split windows by pred_start
    train_wins = [(hs, he, fs, fe) for hs, he, fs, fe in windows
                  if aligned_dates[he - 1] < pred_start_date
                  and fe <= len(aligned_dates)]
    val_wins   = [(hs, he, fs, fe) for hs, he, fs, fe in windows
                  if aligned_dates[hs] >= pred_start_date
                  and fe <= len(aligned_dates)]

    print(f"  Train windows: {len(train_wins)}  Val windows: {len(val_wins)}")

    def _build_patches(wins, meteo, fire):
        """Build (N_patches, days, patch_dim) arrays for a list of windows."""
        X_list, Y_list = [], []
        for hs, he, fs, fe in wins:
            hist = meteo[hs:he]   # (in_days, H, W, 3)
            fut  = fire[fs:fe]    # (out_days, H, W)

            # patchify input: (in_days, H, W, 3) → (n_patches, in_days, P²×3)
            xp, _, _ = patchify(hist, P)   # (n_patches, in_days, P²×3)

            # patchify labels: (out_days, H, W) → (n_patches, out_days, P²)
            yp, _, _ = patchify(fut.astype(np.float32), P)

            X_list.append(xp)
            Y_list.append(yp)

        X = np.concatenate(X_list, axis=0)  # (N*n_patches, in_days, P²×3)
        Y = np.concatenate(Y_list, axis=0)
        return X, Y

    print("  Building train patches...")
    X_train, Y_train = _build_patches(train_wins, meteo_norm, fire_stack)
    print(f"  Train patches: {X_train.shape}")

    if val_wins:
        print("  Building val patches...")
        X_val, Y_val = _build_patches(val_wins, meteo_norm, fire_stack)
        print(f"  Val patches: {X_val.shape}")
    else:
        X_val, Y_val = X_train[:1], Y_train[:1]
        print("  Warning: no val windows, using dummy val")

    train_ds = Fire7DayDataset(X_train, Y_train)
    val_ds   = Fire7DayDataset(X_val,   Y_val)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # ----------------------------------------------------------------
    # STEP 8  Compute pos_weight for class imbalance
    # ----------------------------------------------------------------
    # Use training fire labels only
    train_fire = fire_stack[:train_end_idx]
    n_pos = float(train_fire.sum())
    n_neg = float(train_fire.size) - n_pos
    if n_pos == 0:
        pos_w = 1.0
    else:
        pos_w = n_neg / n_pos
    print(f"\n[STEP 8] pos_weight = {pos_w:.1f}  (neg/pos from training set)")
    run_meta["pos_weight"] = pos_w

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

    pos_weight_tensor = torch.tensor([pos_w], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        # -- train --
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
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

        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "meteo_means":  meteo_means,
                "meteo_stds":   meteo_stds,
                "patch_dim_in": patch_dim_in,
                "patch_dim_out":patch_dim_out,
                "hw":           hw,
                "grid":         grid,
                "args":         vars(args),
            }, best_ckpt)

    print(f"\n  Best val loss: {best_val_loss:.5f}  saved to {best_ckpt}")
    run_meta["best_val_loss"] = best_val_loss

    # ----------------------------------------------------------------
    # STEP 10  Generate prediction tifs
    # ----------------------------------------------------------------
    print("\n[STEP 10] Generating forecast tifs...")

    # Reload best model
    ckpt = torch.load(best_ckpt, map_location=device)
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

        hist = meteo_norm[base_idx - in_days: base_idx]   # (in_days, H, W, 3)

        # patchify → (n_patches, in_days, patch_dim_in)
        xp, _, _ = patchify(hist, P)
        xb = torch.from_numpy(xp).float().to(device)      # (n_patches, in_days, P²×3)

        with torch.no_grad():
            logits = model(xb)                             # (n_patches, out_days, P²)
            probs  = torch.sigmoid(logits).cpu().numpy()   # (n_patches, out_days, P²)

        base_str = base_date.strftime("%Y%m%d")
        day_out  = os.path.join(output_dir, base_str)
        os.makedirs(day_out, exist_ok=True)

        for lead in range(1, out_days + 1):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            out_path        = os.path.join(
                day_out, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
            )

            # probs[:, lead-1, :] shape: (n_patches, P²)
            prob_patches = probs[:, lead - 1, :]    # (n_patches, P²)

            # depatchify: expects (n_patches, 1, P²) → (1, Hc, Wc)
            prob_day = depatchify(
                prob_patches[:, np.newaxis, :], grid, P, hw, num_channels=1
            )                                        # (1, Hc, Wc)
            prob_map = prob_day[0]                   # (Hc, Wc)

            # Pad back to original H, W if cropped
            if prob_map.shape != (H, W):
                full = np.zeros((H, W), dtype=np.float32)
                full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                prob_map = full

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(prob_map.astype(np.float32), 1)

        print(f"  [DONE] {base_date} → {out_days} lead tifs")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Forecasts : {output_dir}")
    print(f"  Checkpoint: {best_ckpt}")
    print("=" * 70)

    run_meta["status"] = "success"
    run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run log: {run_meta_path}")


if __name__ == "__main__":
    main()
