#!/usr/bin/env python3
"""
Logistic regression baseline for wildfire ignition prediction.

Trains a per-patch logistic regression on aggregated weather + static
features over the training period, then scores val windows using the
same pipeline as src/evaluation/benchmark_baselines.py. Outputs CSV in
the same schema, so results append cleanly to benchmark_baselines.csv.

Features per (window, patch):
  - FWI mean over forecast window
  - FWI max over forecast window
  - 2t mean over forecast window
  - sm20 mean over forecast window
  - fire_clim (static, per patch)
  - slope (static, per patch)

Label: any fire pixel inside the patch over the forecast window
       (matches benchmark_baselines.py per_window eval).

Usage:
  python -m scripts.benchmark_logreg \
      --config configs/paths_narval.yaml \
      --pred_start 2022-05-01 --pred_end 2024-10-31 \
      --fire_label_npy /path/to/fire_labels.npy \
      --climatology_tif data/fire_clim_annual_nbac/fire_clim_upto_2022.tif \
      --output_csv outputs/benchmark_logreg.csv
"""
import argparse
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Make `src` importable when running as plain script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rasterio  # noqa: E402

from src.evaluation.benchmark_baselines import (  # noqa: E402
    load_data, _build_file_index, _read_tif, _patchify_frame,
    _build_s2s_windows_calendar, eval_per_window,
)


def _load_static_patched(path, Hc, Wc, P):
    """Load a static raster (slope/population/etc.) patched to (n_patches, P*P)."""
    arr = np.nan_to_num(_read_tif(path), nan=0.0).astype(np.float32)
    return _patchify_frame(arr[:Hc, :Wc, np.newaxis], P).astype(np.float32)


def _load_daily_channel(file_index, all_dates, Hc, Wc, P, label):
    """Load a daily raster (e.g. 2t, sm20) into (n_patches, T, P*P) f16."""
    n_patches = (Hc // P) * (Wc // P)
    out = np.zeros((n_patches, len(all_dates), P * P), dtype=np.float16)
    t0 = time.time()
    miss = 0
    for t_idx, d in enumerate(all_dates):
        if d not in file_index:
            miss += 1
            continue
        arr = np.nan_to_num(_read_tif(file_index[d]), nan=0.0)
        if arr.shape[0] < Hc or arr.shape[1] < Wc:
            # Skip any oddly-shaped frame
            miss += 1
            continue
        out[:, t_idx, :] = _patchify_frame(
            arr[:Hc, :Wc, np.newaxis], P).astype(np.float16)
        if t_idx % 500 == 0:
            print(f"  [{label}] {t_idx}/{len(all_dates)} ({time.time()-t0:.0f}s)")
    if miss:
        print(f"  [{label}] WARN: {miss} missing days (zero-filled)")
    return out


def _features_for_window(win, fwi_p, t2_p, sm_p, clim_per_patch, slope_per_patch,
                          n_patches):
    """Return X = (n_patches, n_features) for one window."""
    hs, he, ts, te = win
    fwi_slice = fwi_p[:, ts:te, :].astype(np.float32)   # (n_patches, lead, P*P)
    t2_slice = t2_p[:, ts:te, :].astype(np.float32)
    sm_slice = sm_p[:, ts:te, :].astype(np.float32)

    # Patch-level summaries (mean over P*P pixels and over lead days)
    fwi_mean = fwi_slice.mean(axis=(1, 2))
    fwi_max = fwi_slice.max(axis=(1, 2))
    t2_mean = t2_slice.mean(axis=(1, 2))
    sm_mean = sm_slice.mean(axis=(1, 2))

    X = np.column_stack([
        fwi_mean, fwi_max, t2_mean, sm_mean,
        clim_per_patch, slope_per_patch,
    ]).astype(np.float32)
    return X   # (n_patches, 6)


def _label_for_window(win, fire_p):
    """Return y in {0,1} per patch — any fire pixel inside the lead window."""
    hs, he, ts, te = win
    return (fire_p[ts:te].sum(axis=(0, 2)) > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred_start", default="2022-05-01")
    ap.add_argument("--pred_end", default="2024-10-31")
    ap.add_argument("--in_days", type=int, default=7)
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=45)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--dilate_radius", type=int, default=14)
    ap.add_argument("--k_values", nargs="+", type=int,
                    default=[1000, 2500, 5000, 10000, 25000])
    ap.add_argument("--n_sample_wins", type=int, default=20)
    ap.add_argument("--fire_season_only", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--climatology_tif", type=str, default=None)
    ap.add_argument("--fire_label_npy", type=str, default=None)
    ap.add_argument("--n_train_wins", type=int, default=80,
                    help="Cap on train windows (sampled uniformly) to keep "
                         "logreg fitting tractable")
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("LOGREG BASELINE")
    print("=" * 70)

    # Reuse benchmark_baselines loader for FWI / fire_label / climatology
    (fwi_p, fire_p, clim_p, all_dates, date_to_idx,
     val_wins, val_win_dates, n_patches, grid) = load_data(
        args.config, args.pred_start, args.pred_end,
        args.in_days, args.lead_start, args.lead_end,
        args.patch_size, args.dilate_radius,
        args.fire_season_only, args.climatology_tif,
        args.fire_label_npy,
    )

    # All windows (need train_wins separately)
    all_windows = _build_s2s_windows_calendar(
        all_dates, date_to_idx, args.in_days, args.lead_start, args.lead_end)
    pred_start = date.fromisoformat(args.pred_start)
    train_wins = [w for w in all_windows if all_dates[w[1]] < pred_start]
    print(f"  train_wins: {len(train_wins)}  val_wins: {len(val_wins)}")

    # Sample train windows to cap fitting time
    rng = np.random.default_rng(0)
    if len(train_wins) > args.n_train_wins:
        idxs = rng.choice(len(train_wins), args.n_train_wins, replace=False)
        train_wins_sampled = [train_wins[i] for i in sorted(idxs)]
    else:
        train_wins_sampled = train_wins
    print(f"  train_wins sampled: {len(train_wins_sampled)}")

    # Static features
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["paths"]
    H, W = fwi_p.shape[0] * 0, fwi_p.shape[0] * 0  # placeholder, recompute
    # Recompute Hc, Wc from FWI raster
    fwi_idx = _build_file_index(cfg["fwi_dir"])
    sample_path = fwi_idx[all_dates[0]]
    with rasterio.open(sample_path) as src:
        H, W = src.height, src.width
    P = args.patch_size
    Hc, Wc = H - H % P, W - W % P
    print(f"  Hc={Hc} Wc={Wc}")

    # static (per patch, scalar)
    print("\n  Loading static channels (climatology, slope)...")
    if clim_p is None:
        sys.exit("ERROR: climatology_tif required for logreg baseline")
    clim_per_patch = clim_p.mean(axis=1).astype(np.float32)  # (n_patches,)

    # Slope lives at {terrain_dir}/slope.tif
    terrain_dir = cfg.get("terrain_dir", "data/terrain")
    slope_path = os.path.join(terrain_dir, "slope.tif")
    if not os.path.exists(slope_path):
        sys.exit(f"ERROR: slope TIF missing: {slope_path}")
    slope_arr = _load_static_patched(slope_path, Hc, Wc, P)
    slope_per_patch = slope_arr.mean(axis=1).astype(np.float32)

    # Daily ERA5 channels: observation_dir has per-variable subdirs (2t/, sm20/)
    # containing per-date TIFs like 2t_YYYYMMDD.tif
    print("\n  Loading daily channels (2t, sm20)...")
    obs_root = cfg.get("observation_dir")
    if not obs_root:
        sys.exit(f"ERROR: observation_dir not in config; "
                 f"available keys: {sorted(cfg.keys())}")
    t2_dir = os.path.join(obs_root, "2t")
    sm_dir = os.path.join(obs_root, "sm20")
    for d in (t2_dir, sm_dir):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: ERA5 subdir missing: {d}")
    t2_idx = _build_file_index(t2_dir)
    sm_idx = _build_file_index(sm_dir)
    print(f"  2t: {len(t2_idx)} days  sm20: {len(sm_idx)} days")
    t2_p = _load_daily_channel(t2_idx, all_dates, Hc, Wc, P, "2t")
    sm_p = _load_daily_channel(sm_idx, all_dates, Hc, Wc, P, "sm20")

    # Build train (X, y)
    print(f"\n  Building train features over {len(train_wins_sampled)} windows...")
    X_train_list = []
    y_train_list = []
    for i, w in enumerate(train_wins_sampled):
        X = _features_for_window(w, fwi_p, t2_p, sm_p,
                                 clim_per_patch, slope_per_patch, n_patches)
        y = _label_for_window(w, fire_p)
        X_train_list.append(X)
        y_train_list.append(y)
        if i % 20 == 0:
            print(f"    win {i}/{len(train_wins_sampled)}  pos_rate={y.mean():.4%}")
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"  X_train: {X_train.shape}  pos_rate={y_train.mean():.4%}")

    # Fit
    print("\n  Fitting LogisticRegression(class_weight='balanced')...")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    lr = LogisticRegression(
        class_weight="balanced", max_iter=200, solver="lbfgs", n_jobs=-1
    )
    t0 = time.time()
    lr.fit(X_train_s, y_train)
    print(f"  fit in {time.time()-t0:.0f}s")
    print(f"  coefs: {dict(zip(['fwi_mean','fwi_max','2t_mean','sm_mean','clim','slope'], lr.coef_[0]))}")

    # Score function for eval
    def _logreg_win_score(win):
        X = _features_for_window(win, fwi_p, t2_p, sm_p,
                                 clim_per_patch, slope_per_patch, n_patches)
        Xs = scaler.transform(X)
        proba = lr.predict_proba(Xs)[:, 1].astype(np.float32)  # (n_patches,)
        # broadcast to (n_patches, P*P) — same score per pixel within patch
        return np.broadcast_to(proba[:, None], (n_patches, P * P)).copy()

    print("\n  Running per-window eval on val period...")
    per_win, summary = eval_per_window(
        _logreg_win_score, fire_p, val_wins, args.k_values,
        args.n_sample_wins, "logreg",
        grid=grid, patch_size=args.patch_size,
    )

    # Append to output CSV in benchmark_baselines.csv schema
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        new_file = not os.path.exists(args.output_csv)
        with open(args.output_csv, "a") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["baseline", "k", "lift_k", "precision_k",
                            "lift_coarse", "bss", "f2", "pr_auc",
                            "n_wins_with_fire", "n_fire", "baseline"])
            # NOTE 2026-04-24: eval_per_window returns dicts keyed by
            # `lift_k`, `precision_k`, `pr_auc`, `lift_coarse`, `bss`, `f2`,
            # `mcc`, `n_fire`, `baseline` (not `lift`, `precision`, etc.).
            # Earlier version wrote NaN for everything because of the mismatch.
            for k, row in summary.items():
                w.writerow([
                    "logreg", k,
                    f"{row.get('lift_k', float('nan')):.4f}",
                    f"{row.get('precision_k', float('nan')):.6f}",
                    f"{row.get('lift_coarse', float('nan')):.4f}",   # event-level
                    f"{row.get('bss', float('nan')):.6f}",
                    f"{row.get('f2', float('nan')):.6f}",
                    f"{row.get('pr_auc', float('nan')):.6f}",
                    int(row.get('n_wins_with_fire', 0)),
                    int(row.get('n_fire', 0)),
                    f"{row.get('baseline', float('nan')):.8f}",
                ])
        print(f"\n  appended to {args.output_csv}")

    print("\n=== logreg eval done ===")


if __name__ == "__main__":
    main()
