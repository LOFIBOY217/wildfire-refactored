"""
Train Patch Temporal Transformer for FWI Forecasting
=====================================================
Loads FWI rasters, builds sliding windows, and trains a per-patch
encoder-decoder Transformer to predict 7-day FWI maps.

Usage:
    python -m src.training.train_transformer_fwi --data_dir data/fwi_data --epochs 20
    python -m src.training.train_transformer_fwi --config configs/default.yaml
    python -m src.training.train_transformer_fwi --start_date 20250101 --end_date 20250630

Based on pytorch_transformer_fwi20260129.py.
"""

import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from src.config import load_config, get_path, add_config_argument
from src.utils.seed import set_seed
from src.utils.date_utils import parse_date_arg
from src.utils.patch_utils import patchify, depatchify, build_windows
from src.utils.normalization import standardize
from src.utils.raster_io import NODATA_THRESHOLD, clean_nodata
from src.models.transformer_fwi import PatchTemporalTransformer
from src.datasets.fwi_weekly import FWIWeeklyDataset

import glob
import re
from datetime import datetime


def load_stack(data_dir, start_date=None, end_date=None):
    """
    Load FWI raster stack from directory.

    Returns:
        frames: (T, H, W) float32, NoData cleaned
        dates: list of datetime
        files: list of file paths
    """
    def _parse_date(path):
        m = re.search(r'(20\d{6})', os.path.basename(path))
        if not m:
            return None
        return datetime.strptime(m.group(1), "%Y%m%d")

    files = sorted(
        glob.glob(os.path.join(data_dir, "*.tif")),
        key=lambda p: _parse_date(p) or datetime.min
    )
    if not files:
        raise FileNotFoundError(f"No .tif files in {data_dir}")

    imgs, dates, filtered_files = [], [], []
    nodata_count = 0
    total_pixels = 0

    for p in files:
        d = _parse_date(p)
        if d is None:
            continue
        if start_date is not None and d < start_date:
            continue
        if end_date is not None and d > end_date:
            continue

        im = Image.open(p)
        arr = np.array(im, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]

        nodata_mask = arr < NODATA_THRESHOLD
        nodata_count += nodata_mask.sum()
        total_pixels += arr.size

        arr = clean_nodata(arr)
        imgs.append(arr)
        dates.append(d)
        filtered_files.append(p)

    if not imgs:
        raise FileNotFoundError("No valid .tif files found in date range")

    frames = np.stack(imgs, axis=0)

    nodata_ratio = nodata_count / total_pixels * 100 if total_pixels > 0 else 0
    print(f"[Data] Loaded {len(imgs)} frames, NoData: {nodata_ratio:.2f}%")
    if dates:
        print(f"[Data] Range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Fill NaN with global mean
    valid_mean = np.nanmean(frames)
    if np.isnan(valid_mean) or np.isinf(valid_mean):
        valid_mean = 0.0
    frames = np.nan_to_num(frames, nan=valid_mean, posinf=valid_mean, neginf=valid_mean)

    return frames, dates, filtered_files


def run_epoch(model, loader, opt, loss_fn, device, train=True):
    """Run one training or validation epoch."""
    model.train(train)
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser(description="Train FWI Transformer")
    add_config_argument(ap)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--in_days", type=int, default=7)
    ap.add_argument("--out_days", type=int, default=7)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--start_date", type=str, default=None)
    ap.add_argument("--end_date", type=str, default=None)
    args = ap.parse_args()

    # Load config and resolve paths
    cfg = load_config(args.config)
    data_dir = args.data_dir or get_path(cfg, 'fwi_dir')
    out_dir = args.out_dir or os.path.join(get_path(cfg, 'output_dir'), 'transformer_fwi')

    start_date = parse_date_arg(args.start_date)
    end_date = parse_date_arg(args.end_date)

    set_seed(args.seed)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    frames, dates, files = load_stack(data_dir, start_date, end_date)
    T, H, W = frames.shape
    print(f"[Info] Frames: {T}, Resolution: {H}x{W}")

    if T < args.in_days + args.out_days:
        warnings.warn(f"Only {T} frames, reducing to in_days=3, out_days=1")
        args.in_days, args.out_days = 3, 1

    # Standardize using first 70%
    cutoff = int(T * 0.7)
    print(f"\n[Standardize] Using first {cutoff} frames...")
    frames_std, mu, sd = standardize(frames[:cutoff], frames)
    np.save(os.path.join(out_dir, "norm_stats.npy"), np.array([mu, sd], dtype=np.float32))

    windows = build_windows(frames_std, args.in_days, args.out_days)
    if not windows:
        raise RuntimeError("Not enough time windows. Add more daily rasters.")
    print(f"[Windows] {len(windows)} time windows")

    # Chronological split
    split_idx = int(len(windows) * (1 - args.val_split))
    train_wins, val_wins = windows[:split_idx], windows[split_idx:]
    print(f"[Split] Train: {len(train_wins)}, Val: {len(val_wins)}")

    train_ds = FWIWeeklyDataset(frames_std, train_wins, args.in_days, args.out_days, args.patch, augment=True)
    val_ds = FWIWeeklyDataset(frames_std, val_wins, args.in_days, args.out_days, args.patch, augment=False)
    print(f"[Dataset] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    patch_dim = train_ds.X.shape[-1]
    model = PatchTemporalTransformer(
        patch_dim, d_model=128, nhead=4, num_enc=2, num_dec=2,
        in_days=args.in_days, out_days=args.out_days
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, opt, loss_fn, device, train=True)
        va = run_epoch(model, val_dl, opt, loss_fn, device, train=False)
        print(f"Epoch {epoch}: train {tr:.4f}  val {va:.4f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    # Generate sample forecast
    print("\n[Predict] Generating sample forecast...")
    last_start = windows[-1][0]
    X_last = frames_std[last_start:last_start + args.in_days]
    Xp, hw, grid = patchify(X_last, args.patch)
    with torch.no_grad():
        pred = model(torch.from_numpy(Xp).to(device).float()).cpu().numpy()
    Y_pred = depatchify(pred, grid, args.patch, hw)
    Y_pred_den = Y_pred * sd + mu

    for i in range(Y_pred_den.shape[0]):
        arr = Y_pred_den[i]
        vmin, vmax = np.nanpercentile(arr, 1), np.nanpercentile(arr, 99)
        img = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.save(os.path.join(out_dir, f"forecast_day{i+1}.png"))

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
