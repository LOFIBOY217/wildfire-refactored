#!/usr/bin/env python3
"""
ML baseline panel for wildfire ignition prediction (NBAC labels).

Supports four model families on the same 6-feature input:
  - logreg      sklearn LogisticRegression       (linear)
  - rf          sklearn RandomForestClassifier    (tree ensemble)
  - xgboost     xgboost.XGBClassifier             (boosted trees)
  - mlp         sklearn MLPClassifier             (shallow NN)

Features per (window, patch):
  1. FWI mean over forecast window
  2. FWI max  over forecast window
  3. 2t  mean over forecast window
  4. sm20 mean over forecast window
  5. fire_clim   (static, per patch)
  6. slope       (static, per patch)

Use --smoke for a quick (~5 min) sanity test with tiny N before submitting
the full SLURM job.

Usage:
    python -m scripts.benchmark_ml --model rf \\
        --config configs/paths_narval.yaml \\
        --pred_start 2022-05-01 --pred_end 2024-10-31 \\
        --fire_label_npy /path/to/fire_labels.npy \\
        --climatology_tif data/fire_clim_annual_nbac/fire_clim_upto_2022.tif \\
        --output_csv outputs/benchmark_ml.csv
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rasterio  # noqa: E402

from src.evaluation.benchmark_baselines import (  # noqa: E402
    load_data, _build_file_index, _read_tif, _patchify_frame,
    _build_s2s_windows_calendar, eval_per_window,
)


# ── feature utilities (same as benchmark_logreg.py) ───────────────────────

def _load_static_patched(path, Hc, Wc, P):
    arr = np.nan_to_num(_read_tif(path), nan=0.0).astype(np.float32)
    return _patchify_frame(arr[:Hc, :Wc, np.newaxis], P).astype(np.float32)


def _load_daily_channel(file_index, all_dates, Hc, Wc, P, label):
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
            miss += 1
            continue
        out[:, t_idx, :] = _patchify_frame(
            arr[:Hc, :Wc, np.newaxis], P).astype(np.float16)
        if t_idx % 500 == 0:
            print(f"  [{label}] {t_idx}/{len(all_dates)} ({time.time()-t0:.0f}s)")
    if miss:
        print(f"  [{label}] WARN: {miss} missing days (zero-filled)")
    return out


def _features_for_window(win, fwi_p, t2_p, sm_p, clim_per_patch, slope_per_patch):
    hs, he, ts, te = win
    fwi_slice = fwi_p[:, ts:te, :].astype(np.float32)
    t2_slice = t2_p[:, ts:te, :].astype(np.float32)
    sm_slice = sm_p[:, ts:te, :].astype(np.float32)
    fwi_mean = fwi_slice.mean(axis=(1, 2))
    fwi_max  = fwi_slice.max(axis=(1, 2))
    t2_mean  = t2_slice.mean(axis=(1, 2))
    sm_mean  = sm_slice.mean(axis=(1, 2))
    return np.column_stack([fwi_mean, fwi_max, t2_mean, sm_mean,
                             clim_per_patch, slope_per_patch]).astype(np.float32)


def _label_for_window(win, fire_p):
    hs, he, ts, te = win
    return (fire_p[ts:te].sum(axis=(0, 2)) > 0).astype(np.uint8)


# ── model factory ─────────────────────────────────────────────────────────

def build_model(name, n_pos, n_neg, smoke=False):
    """Return a fresh estimator + a name. class_weight handled separately."""
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            class_weight="balanced", max_iter=200, solver="lbfgs", n_jobs=-1)

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        # Smaller forest for smoke; full size for production
        n_estimators = 50 if smoke else 200
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=12,
            min_samples_leaf=200,            # avoid memorizing rare patches
            class_weight="balanced",
            n_jobs=-1, random_state=0)

    if name == "xgboost":
        import xgboost as xgb
        # Use sample_weight in fit() for class balance (xgb's
        # scale_pos_weight only handles binary, which is our case).
        spw = max(1.0, n_neg / max(n_pos, 1))   # negative/positive ratio
        n_rounds = 100 if smoke else 400
        return xgb.XGBClassifier(
            n_estimators=n_rounds, max_depth=6, learning_rate=0.05,
            scale_pos_weight=spw, tree_method="hist",
            n_jobs=-1, eval_metric="aucpr", random_state=0)

    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        # Shallow 2-hidden-layer net; smaller for smoke
        hidden = (16, 8) if smoke else (64, 32)
        return MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu",
            solver="adam", max_iter=50 if smoke else 200,
            batch_size=4096, learning_rate_init=1e-3,
            random_state=0, verbose=False)

    raise ValueError(f"unknown model: {name}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["logreg", "rf", "xgboost", "mlp"])
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
    ap.add_argument("--n_train_wins", type=int, default=80)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke-test mode: tiny N + smaller model. "
                         "Use to verify the pipeline runs end-to-end before "
                         "spending hours on a full run.")
    args = ap.parse_args()

    if args.smoke:
        args.n_train_wins = 5
        args.n_sample_wins = 3
        args.k_values = [5000]
        print(">> SMOKE MODE: tiny N + small model. Not for real numbers.")

    print("=" * 70)
    print(f"ML BASELINE: {args.model}")
    print("=" * 70)

    (fwi_p, fire_p, clim_p, all_dates, date_to_idx,
     val_wins, val_win_dates, n_patches, grid) = load_data(
        args.config, args.pred_start, args.pred_end,
        args.in_days, args.lead_start, args.lead_end,
        args.patch_size, args.dilate_radius,
        args.fire_season_only, args.climatology_tif,
        args.fire_label_npy,
    )

    all_windows = _build_s2s_windows_calendar(
        all_dates, date_to_idx, args.in_days, args.lead_start, args.lead_end)
    pred_start = date.fromisoformat(args.pred_start)
    train_wins = [w for w in all_windows if all_dates[w[1]] < pred_start]
    print(f"  train_wins: {len(train_wins)}  val_wins: {len(val_wins)}")

    # Sample train windows
    rng = np.random.default_rng(0)
    if len(train_wins) > args.n_train_wins:
        idxs = rng.choice(len(train_wins), args.n_train_wins, replace=False)
        train_wins_s = [train_wins[i] for i in sorted(idxs)]
    else:
        train_wins_s = train_wins
    print(f"  train_wins sampled: {len(train_wins_s)}")

    # Static features
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["paths"]
    fwi_idx = _build_file_index(cfg["fwi_dir"])
    sample_path = fwi_idx[all_dates[0]]
    with rasterio.open(sample_path) as src:
        H, W = src.height, src.width
    P = args.patch_size
    Hc, Wc = H - H % P, W - W % P

    if clim_p is None:
        sys.exit("ERROR: climatology_tif required for ML baseline")
    clim_per_patch = clim_p.mean(axis=1).astype(np.float32)

    terrain_dir = cfg.get("terrain_dir", "data/terrain")
    slope_path = os.path.join(terrain_dir, "slope.tif")
    if not os.path.exists(slope_path):
        sys.exit(f"ERROR: slope TIF missing: {slope_path}")
    slope_arr = _load_static_patched(slope_path, Hc, Wc, P)
    slope_per_patch = slope_arr.mean(axis=1).astype(np.float32)

    # Daily ERA5 channels
    print("\n  Loading daily channels (2t, sm20)...")
    obs_root = cfg.get("observation_dir")
    if not obs_root:
        sys.exit(f"ERROR: observation_dir not in config")
    t2_idx = _build_file_index(os.path.join(obs_root, "2t"))
    sm_idx = _build_file_index(os.path.join(obs_root, "sm20"))
    t2_p = _load_daily_channel(t2_idx, all_dates, Hc, Wc, P, "2t")
    sm_p = _load_daily_channel(sm_idx, all_dates, Hc, Wc, P, "sm20")

    # Build train (X, y)
    print(f"\n  Building train features over {len(train_wins_s)} windows...")
    X_list, y_list = [], []
    for i, w in enumerate(train_wins_s):
        X = _features_for_window(w, fwi_p, t2_p, sm_p,
                                 clim_per_patch, slope_per_patch)
        y = _label_for_window(w, fire_p)
        X_list.append(X); y_list.append(y)
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"  X_train: {X_train.shape}  pos={n_pos:,}  neg={n_neg:,}  "
          f"pos_rate={n_pos/len(y_train):.4%}")

    # Standardize (helps logreg + mlp; tree models indifferent but harmless)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    # Fit
    print(f"\n  Fitting {args.model}...")
    model = build_model(args.model, n_pos, n_neg, smoke=args.smoke)
    t0 = time.time()
    model.fit(X_train_s, y_train)
    print(f"  fit in {time.time()-t0:.0f}s")

    # Score function
    def _score_win(win):
        X = _features_for_window(win, fwi_p, t2_p, sm_p,
                                 clim_per_patch, slope_per_patch)
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)[:, 1].astype(np.float32)
        return np.broadcast_to(proba[:, None], (n_patches, P * P)).copy()

    print("\n  Running per-window eval...")
    per_win, summary = eval_per_window(
        _score_win, fire_p, val_wins, args.k_values,
        args.n_sample_wins, args.model,
        grid=grid, patch_size=args.patch_size,
    )

    # CSV output
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        new_file = not os.path.exists(args.output_csv)
        with open(args.output_csv, "a") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["model", "k", "lift_k", "precision_k",
                            "lift_coarse", "bss", "f2", "pr_auc",
                            "n_wins_with_fire", "n_fire", "baseline"])
            for k, row in summary.items():
                w.writerow([
                    args.model, k,
                    f"{row.get('lift_k', float('nan')):.4f}",
                    f"{row.get('precision_k', float('nan')):.6f}",
                    f"{row.get('lift_coarse', float('nan')):.4f}",
                    f"{row.get('bss', float('nan')):.6f}",
                    f"{row.get('f2', float('nan')):.6f}",
                    f"{row.get('pr_auc', float('nan')):.6f}",
                    int(row.get('n_wins_with_fire', 0)),
                    int(row.get('n_fire', 0)),
                    f"{row.get('baseline', float('nan')):.8f}",
                ])
        print(f"\n  appended to {args.output_csv}")

    print(f"\n=== {args.model} eval done ===")


if __name__ == "__main__":
    main()
