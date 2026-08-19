"""
Train Linear Regression Baseline for 7-Day FWI Forecasting
===========================================================
A pixel-level linear regression model that predicts future FWI values
(continuous regression) using the same 3-feature engineering as the
Logistic Regression fire-probability baseline:

    Feature 1: fwi_max_norm  — max FWI over the 7-day history window
    Feature 2: dryness_norm  — max dewpoint depression (T - Td) over window
    Feature 3: fwi_last      — FWI on the most recent day (persistence signal)

One independent Ridge Regression model is trained per lead day (1–7),
predicting the raw FWI value at that lead day.

Purpose:
    Direct fair comparison against train_transformer_7day_fwi.py.
    Both models:
      - use the same inputs  (7-day history of FWI + 2t + 2d)
      - predict the same target (future FWI, continuous)
      - are evaluated by the same script (evaluate_fwi_forecast.py)

    Unlike the binary Logistic model, NO CIFFC data is used — the target
    is FWI, not fire occurrence.

Output format (compatible with evaluate_fwi_forecast.py):
    outputs/linreg_fwi_pred/YYYYMMDD/fwi_pred_lead{k:02d}d_YYYYMMDD.tif

Usage:
    python -m src.training.train_linear_7day_fwi \\
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
from sklearn.linear_model import Ridge

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


# ------------------------------------------------------------------ #
# Feature engineering
# ------------------------------------------------------------------ #

FWI_MAX_CLIP    = 150.0
DRYNESS_OFFSET  = 5.0
DRYNESS_SCALE   = 15.0


def compute_fwi_features(fwi_window, t2m_window, d2m_window):
    """
    Compute 3 features from a 7-day history window for FWI regression.

    Deliberately mirrors logistic_baseline.compute_features() but drops
    `recent_fire` (CIFFC-dependent) and adds `fwi_last` (persistence signal).

    Args:
        fwi_window : (T, H, W) — raw FWI values over the history window
        t2m_window : (T, H, W) — 2m temperature (K)
        d2m_window : (T, H, W) — 2m dewpoint temperature (K)

    Returns:
        features : (H, W, 3)
            [0] fwi_max_norm  — max FWI in window, clipped and normalised
            [1] dryness_norm  — max dewpoint depression, normalised
            [2] fwi_last      — FWI on final day of window (persistence proxy)
    """
    # Feature 1: max FWI in history window (normalised)
    fwi_max      = np.max(fwi_window, axis=0)
    fwi_max      = np.clip(fwi_max, 0.0, FWI_MAX_CLIP)
    fwi_max_norm = fwi_max / 30.0

    # Feature 2: max dewpoint depression (dryness proxy, normalised)
    dew_dep       = t2m_window - d2m_window           # (T, H, W)
    dryness_max   = np.max(dew_dep, axis=0)
    dryness_norm  = (dryness_max - DRYNESS_OFFSET) / DRYNESS_SCALE
    dryness_norm  = np.clip(dryness_norm, 0.0, 5.0)

    # Feature 3: FWI on the last observed day (persistence signal)
    fwi_last = fwi_window[-1].astype(np.float32)

    return np.stack([fwi_max_norm, dryness_norm, fwi_last], axis=-1)  # (H, W, 3)


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


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    run_started_at  = time.time()
    run_started_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"

    ap = argparse.ArgumentParser(
        description="Train Linear Regression Baseline for 7-Day FWI Forecasting"
    )
    add_config_argument(ap)
    ap.add_argument("--data_start",  type=str,   default="2023-05-05")
    ap.add_argument("--pred_start",  type=str,   default="2025-07-01")
    ap.add_argument("--pred_end",    type=str,   default="2025-08-14")
    ap.add_argument("--in_days",     type=int,   default=7)
    ap.add_argument("--out_days",    type=int,   default=7)
    ap.add_argument("--alpha",       type=float, default=1.0,
                    help="Ridge regularisation strength (default=1.0).")
    ap.add_argument("--n_samples",   type=int,   default=50000,
                    help="Pixels sampled per training day per lead (default=50000).")
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # ----------------------------------------------------------------
    # Config & paths
    # ----------------------------------------------------------------
    cfg        = load_config(args.config)
    fwi_dir    = get_path(cfg, "fwi_dir")
    paths_cfg  = cfg.get("paths", {})
    obs_root   = get_path(cfg, "observation_dir") if "observation_dir" in paths_cfg \
                 else get_path(cfg, "ecmwf_dir")
    output_dir = os.path.join(get_path(cfg, "output_dir"), "linreg_fwi_pred")
    ckpt_dir   = os.path.join(get_path(cfg, "checkpoint_dir"), "linreg_fwi")
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
        p = s.split("-")
        return date(int(p[0]), int(p[1]), int(p[2]))

    data_start_date = _date(args.data_start)
    pred_start_date = _date(args.pred_start)
    pred_end_date   = _date(args.pred_end)
    in_days         = args.in_days
    out_days        = args.out_days

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION BASELINE  —  7-Day FWI Forecast")
    print("=" * 70)
    print(f"  data_start : {data_start_date}")
    print(f"  pred_start : {pred_start_date}  (train/val split)")
    print(f"  pred_end   : {pred_end_date}")
    print(f"  in_days    : {in_days}   out_days: {out_days}")
    print(f"  alpha      : {args.alpha}   n_samples/day: {args.n_samples}")
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
        raise RuntimeError(f"Only {len(aligned_dates)} aligned days, need >= {in_days + out_days}")
    print(f"  Aligned dates: {len(aligned_dates)}  ({aligned_dates[0]} -> {aligned_dates[-1]})")
    run_meta["aligned_days"] = len(aligned_dates)

    # ----------------------------------------------------------------
    # STEP 3  Load raster stacks
    # ----------------------------------------------------------------
    print("\n[STEP 3] Loading raster stacks...")
    fwi_stack = read_singleband_stack(fwi_paths)
    t2m_stack = read_singleband_stack(t2m_paths)
    d2m_stack = read_singleband_stack(d2m_paths)
    T, H, W   = fwi_stack.shape
    print(f"  Shape: T={T}  H={H}  W={W}")

    def _clean(stack):
        s = clean_nodata(stack.astype(np.float32))
        fill = float(np.nanmean(s))
        if not np.isfinite(fill):
            fill = 0.0
        return np.nan_to_num(s, nan=fill, posinf=fill, neginf=fill)

    fwi_stack = _clean(fwi_stack)
    t2m_stack = _clean(t2m_stack)
    d2m_stack = _clean(d2m_stack)

    with rasterio.open(fwi_paths[0]) as src:
        profile = src.profile

    # ----------------------------------------------------------------
    # STEP 4  Find train/val split
    # ----------------------------------------------------------------
    print("\n[STEP 4] Splitting train / val by pred_start...")
    train_end_idx = None
    for i, d in enumerate(aligned_dates):
        if d >= pred_start_date:
            train_end_idx = i
            break
    if train_end_idx is None:
        raise RuntimeError(f"pred_start={pred_start_date} is beyond all aligned dates")
    print(f"  Train: {aligned_dates[0]} -> {aligned_dates[train_end_idx-1]}")
    print(f"  Val  : {aligned_dates[train_end_idx]} -> {aligned_dates[-1]}")

    # ----------------------------------------------------------------
    # STEP 5  Build training data — one model per lead day
    # ----------------------------------------------------------------
    print("\n[STEP 5] Building training samples...")

    rng      = np.random.default_rng(args.seed)
    n_train  = train_end_idx

    # Collect (X, y_lead) for each lead day
    X_lists = [[] for _ in range(out_days)]
    y_lists = [[] for _ in range(out_days)]

    for idx in range(in_days, n_train):
        ws = idx - in_days
        we = idx

        feats = compute_fwi_features(
            fwi_stack[ws:we],
            t2m_stack[ws:we],
            d2m_stack[ws:we],
        )                              # (H, W, 3)

        # Valid pixel mask (no nodata/nan)
        valid = np.isfinite(feats[..., 0]) & (feats[..., 0] >= 0)

        valid_rows, valid_cols = np.where(valid)
        n_valid = len(valid_rows)
        if n_valid == 0:
            continue

        # Sample pixels
        n_take = min(n_valid, args.n_samples)
        chosen = rng.choice(n_valid, size=n_take, replace=False)
        rows   = valid_rows[chosen]
        cols   = valid_cols[chosen]
        X_day  = feats[rows, cols]     # (n_take, 3)

        for lead in range(1, out_days + 1):
            target_idx = idx + lead - 1
            if target_idx >= T:
                continue
            y_day = fwi_stack[target_idx][rows, cols]   # raw FWI at lead day
            X_lists[lead - 1].append(X_day)
            y_lists[lead - 1].append(y_day)

        if idx % 50 == 0 or idx == n_train - 1:
            print(f"  Processed training day {idx}/{n_train}")

    # ----------------------------------------------------------------
    # STEP 6  Train one Ridge model per lead day
    # ----------------------------------------------------------------
    print("\n[STEP 6] Training Ridge regression models (one per lead day)...")
    models = []
    for lead in range(1, out_days + 1):
        X_lead = np.concatenate(X_lists[lead - 1], axis=0)
        y_lead = np.concatenate(y_lists[lead - 1], axis=0)
        print(f"  Lead {lead}  samples={len(X_lead):,}  "
              f"FWI range=[{y_lead.min():.1f}, {y_lead.max():.1f}]  mean={y_lead.mean():.2f}")
        m = Ridge(alpha=args.alpha)
        m.fit(X_lead, y_lead)
        models.append(m)
        print(f"          coef={m.coef_.round(3)}  intercept={m.intercept_:.3f}")

    run_meta["models"] = [
        {"lead": i + 1, "coef": m.coef_.tolist(), "intercept": float(m.intercept_)}
        for i, m in enumerate(models)
    ]

    # ----------------------------------------------------------------
    # STEP 7  Generate prediction tifs
    # ----------------------------------------------------------------
    print("\n[STEP 7] Generating forecast tifs...")

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    pred_dates = []
    cur = pred_start_date
    while cur <= pred_end_date:
        pred_dates.append(cur)
        cur += timedelta(days=1)

    date_to_idx = {d: i for i, d in enumerate(aligned_dates)}
    chunk_size  = 250_000

    for base_date in pred_dates:
        if base_date not in date_to_idx:
            continue
        base_idx = date_to_idx[base_date]
        if base_idx < in_days:
            continue

        feats = compute_fwi_features(
            fwi_stack[base_idx - in_days: base_idx],
            t2m_stack[base_idx - in_days: base_idx],
            d2m_stack[base_idx - in_days: base_idx],
        )                                     # (H, W, 3)

        X_pred = feats.reshape(-1, 3)         # (H*W, 3)

        base_str = base_date.strftime("%Y%m%d")
        day_out  = os.path.join(output_dir, base_str)
        os.makedirs(day_out, exist_ok=True)

        for lead in range(1, out_days + 1):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            out_path        = os.path.join(
                day_out, f"fwi_pred_lead{lead:02d}d_{target_date_str}.tif"
            )

            fwi_map = np.zeros(H * W, dtype=np.float32)
            for i in range(0, len(X_pred), chunk_size):
                chunk = X_pred[i:i + chunk_size]
                fwi_map[i:i + chunk_size] = models[lead - 1].predict(chunk)

            fwi_map = np.clip(fwi_map, 0.0, 150.0).reshape(H, W)

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(fwi_map.astype(np.float32), 1)

        print(f"  [DONE] {base_date} -> {out_days} lead tifs")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Forecasts : {output_dir}")
    print("=" * 70)

    run_meta["status"]           = "success"
    run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run log: {run_meta_path}")


if __name__ == "__main__":
    main()
