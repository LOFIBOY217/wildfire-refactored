"""
Train 7-Day FWI Regression Transformer
=======================================
Replaces binary CIFFC fire labels with **continuous FWI values** as the
prediction target, turning wildfire forecasting into a clean regression task.

Key differences vs. the binary fire-probability variants:
  - Target y  : future FWI (continuous float, raw scale 0–100+)
  - Loss      : MSELoss  (no pos_weight, no class imbalance issues)
  - Sampling  : ALL (window, patch) pairs — no positive/negative filtering needed
  - No CIFFC  : fire rasterization step removed entirely
  - Evaluation: new evaluate_fwi_forecast.py  (MAE / RMSE / Pearson R)

Motivation:
    CIFFC binary labels suffer from sparse/incomplete records, time lag, and
    spatial mismatch — creating irreducible label noise that penalises complex
    models. FWI is computed from complete gridded reanalysis data: every cell,
    every day, no missing values, no reporting lag. Switching to FWI regression
    lets the Transformer learn true meteorological time-series patterns.

Output format (compatible with evaluate_fwi_forecast.py):
    outputs/transformer7d_fwi_pred/YYYYMMDD/fwi_pred_lead{k:02d}d_YYYYMMDD.tif

Usage:
    python -m src.training.train_transformer_7day_fwi \\
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
from src.models.transformer_7day_fwi import FWI7DayTransformer


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class Fire7DayDatasetFWI(Dataset):
    """
    All-patch dataset for FWI regression.

    Unlike the binary variants, there is no positive/negative filtering:
    every (window, patch) pair is a valid training sample because every
    grid cell has a meaningful FWI value on every day.

    Returns (x, y_fwi) where:
        x     : (in_days, patch_dim_in)   normalised meteo input
        y_fwi : (out_days, patch_dim_out) raw FWI target values
    """

    def __init__(self, meteo_patched, fwi_target_patched, windows, n_patches, pairs=None):
        self.meteo  = meteo_patched
        self.fwi    = fwi_target_patched
        self.windows   = windows
        self.n_patches = n_patches
        # If pairs not provided, use all (window, patch) combinations
        if pairs is not None:
            self.pairs = pairs
        else:
            self.pairs = [
                (win_i, patch_i)
                for win_i in range(len(windows))
                for patch_i in range(n_patches)
            ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        win_i, patch_i = self.pairs[idx]
        hs, he, fs, fe = self.windows[win_i]
        x     = self.meteo[hs:he, patch_i, :]   # (in_days, patch_dim_in)
        y_fwi = self.fwi[fs:fe,  patch_i, :]    # (out_days, patch_dim_out)
        return torch.from_numpy(x.copy()), torch.from_numpy(y_fwi.copy())


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
        description="Train 7-Day FWI Regression Transformer"
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
    ap.add_argument("--val_sample",   type=float, default=0.1,
                    help="Fraction of all-patch val pairs to use (default=0.1). "
                         "Full val set is large; subsample for speed.")
    ap.add_argument("--train_sample", type=float, default=0.02,
                    help="Fraction of all (window,patch) train pairs per epoch (default=0.02). "
                         "Full set is ~18M+ on Canada-wide grid; 0.02 → ~375K pairs.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------
    # Config & paths
    # ----------------------------------------------------------------
    cfg        = load_config(args.config)
    fwi_dir    = get_path(cfg, "fwi_dir")
    paths_cfg  = cfg.get("paths", {})
    obs_root   = get_path(cfg, "observation_dir") if "observation_dir" in paths_cfg \
                 else get_path(cfg, "ecmwf_dir")
    output_dir = os.path.join(get_path(cfg, "output_dir"),
                              "transformer7d_fwi_pred")
    ckpt_dir   = os.path.join(get_path(cfg, "checkpoint_dir"),
                              "transformer_7day_fwi")
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
    print("7-DAY FWI REGRESSION TRANSFORMER")
    print("=" * 70)
    print(f"  data_start        : {data_start_date}")
    print(f"  pred_start        : {pred_start_date}  (train/val split boundary)")
    print(f"  pred_end          : {pred_end_date}")
    print(f"  in_days / out_days: {in_days} / {out_days}")
    print(f"  patch_size        : {args.patch_size}")
    print(f"  d_model / nhead   : {args.d_model} / {args.nhead}")
    print(f"  epochs / batch    : {args.epochs} / {args.batch_size}  lr={args.lr}")
    print(f"  train_sample      : {args.train_sample}  val_sample: {args.val_sample}")
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
    # STEP 4  Normalise FWI target (BEFORE meteo normalisation uses train_end_idx)
    # ----------------------------------------------------------------
    # We must normalise the FWI regression target to prevent NaN loss:
    # raw FWI can reach 200+, MSELoss on raw scale causes gradient explosion.
    # Stats are computed from training period only; saved to checkpoint for
    # de-normalisation when writing output tifs.
    print("\n[STEP 4] Normalising FWI regression target (training-set stats)...")
    # train_end_idx not yet computed — derive it here temporarily
    _train_end_tmp = next(
        (i for i, d in enumerate(aligned_dates) if d >= pred_start_date), len(aligned_dates)
    )
    fwi_train_mean = float(fwi_stack[:_train_end_tmp].mean())
    fwi_train_std  = float(fwi_stack[:_train_end_tmp].std()) + 1e-6
    fwi_raw_norm   = (fwi_stack.copy() - fwi_train_mean) / fwi_train_std
    np.clip(fwi_raw_norm, -10.0, 10.0, out=fwi_raw_norm)
    print(f"  fwi_train_mean={fwi_train_mean:.3f}  fwi_train_std={fwi_train_std:.3f}")
    print(f"  fwi_raw_norm range: [{fwi_raw_norm.min():.2f}, {fwi_raw_norm.max():.2f}]")

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
    # STEP 6  Standardise meteorological input (stats from training only)
    # ----------------------------------------------------------------
    print("\n[STEP 6] Standardising per-channel (FWI, 2t, 2d) for encoder input...")

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

    P = args.patch_size

    _sample, hw, grid = patchify(meteo_norm[:1], P)
    n_patches = grid[0] * grid[1]
    in_dim    = P * P * 3   # encoder: FWI + 2t + 2d
    out_dim   = P * P * 1   # target:  FWI only

    print(f"  n_patches={n_patches}  in_dim={in_dim}  out_dim={out_dim}")

    t0_meteo = time.time()
    meteo_patched = np.empty((T, n_patches, in_dim), dtype=np.float32)
    for t in range(T):
        patches, _, _ = patchify(meteo_norm[t:t + 1], P)
        meteo_patched[t] = patches[:, 0, :]
        if t % 50 == 0 or t == T - 1:
            print(f"  meteo frame {t:4d}/{T}  ({time.time()-t0_meteo:.0f}s elapsed)")
    print(f"  meteo_patched: {meteo_patched.shape}  "
          f"{meteo_patched.nbytes/1e9:.2f} GB  ({time.time()-t0_meteo:.0f}s)")

    t0_fwi = time.time()
    fwi_target_patched = np.empty((T, n_patches, out_dim), dtype=np.float32)
    for t in range(T):
        patches, _, _ = patchify(fwi_raw_norm[t:t + 1], P)
        fwi_target_patched[t] = patches[:, 0, :]
        if t % 50 == 0 or t == T - 1:
            print(f"  fwi   frame {t:4d}/{T}  ({time.time()-t0_fwi:.0f}s elapsed)")
    print(f"  fwi_target_patched: {fwi_target_patched.shape}  "
          f"{fwi_target_patched.nbytes/1e9:.2f} GB  ({time.time()-t0_fwi:.0f}s)")

    del fwi_raw_norm  # free memory

    all_windows = _build_windows(T, in_days, out_days)
    print(f"\n  Total time windows: {len(all_windows)}")

    train_wins = [w for w in all_windows
                  if aligned_dates[w[1] - 1] < pred_start_date and w[3] <= T]
    val_wins   = [w for w in all_windows
                  if aligned_dates[w[0]] >= pred_start_date and w[3] <= T]

    print(f"  Train windows: {len(train_wins)}  Val windows: {len(val_wins)}")

    # ----------------------------------------------------------------
    # STEP 7b  Build all training pairs (no filtering needed)
    # ----------------------------------------------------------------
    print("\n[STEP 7b] Building all (window, patch) training pairs...")
    t0 = time.time()

    rng = np.random.default_rng(args.seed)

    # All windows × all patches — no positive/negative discrimination
    all_train_pairs_full = [
        (win_i, patch_i)
        for win_i in range(len(train_wins))
        for patch_i in range(n_patches)
    ]
    n_train   = max(1, int(len(all_train_pairs_full) * args.train_sample))
    train_idx = rng.choice(len(all_train_pairs_full), size=n_train, replace=False)
    all_train_pairs = [all_train_pairs_full[i] for i in sorted(train_idx)]
    rng.shuffle(all_train_pairs)
    print(f"  Total train pairs (sampled {args.train_sample*100:.1f}%): "
          f"{len(all_train_pairs):,} / {len(all_train_pairs_full):,}  ({time.time()-t0:.0f}s)")

    # Subsample validation pairs for speed (full set can be very large)
    all_val_pairs = [
        (win_i, patch_i)
        for win_i in range(len(val_wins))
        for patch_i in range(n_patches)
    ]
    n_val = max(1, int(len(all_val_pairs) * args.val_sample))
    val_indices  = rng.choice(len(all_val_pairs), size=n_val, replace=False)
    sampled_val_pairs = [all_val_pairs[i] for i in sorted(val_indices)]
    print(f"  Val pairs (sampled {args.val_sample*100:.0f}%): {len(sampled_val_pairs):,}")

    run_meta["total_train_pairs"] = len(all_train_pairs)
    run_meta["total_val_pairs"]   = len(sampled_val_pairs)

    # ----------------------------------------------------------------
    # Build datasets and dataloaders
    # ----------------------------------------------------------------
    train_ds = Fire7DayDatasetFWI(
        meteo_patched, fwi_target_patched, train_wins, n_patches,
        pairs=all_train_pairs
    )
    val_ds = Fire7DayDatasetFWI(
        meteo_patched, fwi_target_patched, val_wins, n_patches,
        pairs=sampled_val_pairs
    ) if val_wins else train_ds

    if not val_wins:
        print("  WARNING: no val windows, using train set as val proxy")

    patch_dim_in  = in_dim
    patch_dim_out = out_dim

    print(f"\n  Train samples: {len(train_ds):,}")
    print(f"  Val   samples: {len(val_ds):,}")
    print(f"  Grid: {grid[0]}x{grid[1]} patches/frame  "
          f"(dim_in={patch_dim_in}  dim_out={patch_dim_out})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    # ----------------------------------------------------------------
    # STEP 8  Build MSE criterion
    # ----------------------------------------------------------------
    print("\n[STEP 8] Setting up MSELoss (regression on raw FWI values)...")
    criterion = nn.MSELoss()
    print("  Loss: MSELoss  (no pos_weight, no class imbalance)")

    # ----------------------------------------------------------------
    # STEP 9  Build model & train
    # ----------------------------------------------------------------
    print("\n[STEP 9] Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = FWI7DayTransformer(
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
            pred   = model(xb)
            loss   = criterion(pred, yb)
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
                pred     = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        # MSE → RMSE for interpretability
        train_rmse = train_loss ** 0.5
        val_rmse   = val_loss   ** 0.5

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_mse={train_loss:.4f} (RMSE={train_rmse:.3f})  "
              f"val_mse={val_loss:.4f} (RMSE={val_rmse:.3f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "meteo_means":    meteo_means,
                "meteo_stds":     meteo_stds,
                "fwi_train_mean": fwi_train_mean,
                "fwi_train_std":  fwi_train_std,
                "patch_dim_in":   patch_dim_in,
                "patch_dim_out":  patch_dim_out,
                "hw":             hw,
                "grid":           grid,
                "args":           vars(args),
            }, best_ckpt)

    best_rmse = best_val_loss ** 0.5
    print(f"\n  Best val MSE: {best_val_loss:.4f}  (RMSE={best_rmse:.3f})  saved -> {best_ckpt}")
    run_meta["best_val_mse"]  = best_val_loss
    run_meta["best_val_rmse"] = best_rmse

    # ----------------------------------------------------------------
    # STEP 10  Generate prediction tifs
    # ----------------------------------------------------------------
    print("\n[STEP 10] Generating forecast tifs...")

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    fwi_train_mean = ckpt["fwi_train_mean"]
    fwi_train_std  = ckpt["fwi_train_std"]

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
        n_pred_patches = xp.shape[0]

        # Chunked inference to avoid OOM
        chunk_size = 1024
        preds_list = []
        with torch.no_grad():
            for start in range(0, n_pred_patches, chunk_size):
                end   = min(start + chunk_size, n_pred_patches)
                xb_c  = torch.from_numpy(xp[start:end]).float().to(device)
                pred_c = model(xb_c).cpu().numpy()
                preds_list.append(pred_c)
        preds = np.concatenate(preds_list, axis=0)  # (n_patches, out_days, out_dim)

        base_str = base_date.strftime("%Y%m%d")
        day_out  = os.path.join(output_dir, base_str)
        os.makedirs(day_out, exist_ok=True)

        for lead in range(1, out_days + 1):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            out_path        = os.path.join(
                day_out, f"fwi_pred_lead{lead:02d}d_{target_date_str}.tif"
            )

            # De-normalise: model output is in normalised space, convert back to raw FWI scale
            fwi_patches_lead = preds[:, lead - 1, :] * fwi_train_std + fwi_train_mean

            fwi_vol = depatchify(
                fwi_patches_lead[:, np.newaxis, :],
                pred_grid, P, pred_hw, num_channels=1
            )
            if fwi_vol.ndim == 3:
                fwi_map = fwi_vol[0]
            else:
                fwi_map = fwi_vol

            if fwi_map.shape != (H, W):
                full = np.zeros((H, W), dtype=np.float32)
                full[:fwi_map.shape[0], :fwi_map.shape[1]] = fwi_map
                fwi_map = full

            # Clip to physically plausible FWI range
            fwi_map = np.clip(fwi_map, 0.0, 150.0)

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(fwi_map.astype(np.float32), 1)

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
