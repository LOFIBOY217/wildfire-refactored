#!/usr/bin/env python3
"""
Fit logreg ONCE (same setup as benchmark_logreg.py) then evaluate on
the EXACT 20 windows that a model's save_scores ran on. Produces
total + novel_7d/30d/90d lift, matching the apples-to-apples format
already produced by add_baselines_to_score_comparison.py for
climatology + persistence.

This closes the "logreg is total-only on different windows" hole in
the paper Table 1.

Usage:
    python -m scripts.add_logreg_to_score_comparison \\
        --scores_dir outputs/window_scores/v3_9ch_enc28_4y_2018/ \\
        --fire_label_npy data/fire_labels/fire_labels_nbac_nfdb_*.npy \\
        --climatology_tif data/fire_clim_annual_nbac/fire_clim_upto_2022.tif \\
        --output_csv outputs/logreg_on_enc28_windows.csv
"""
import argparse
import csv
import glob
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rasterio  # noqa: E402

from src.evaluation.benchmark_baselines import (  # noqa: E402
    load_data, _build_file_index, _read_tif, _patchify_frame,
    _build_s2s_windows_calendar,
)
from scripts.benchmark_ml import (  # noqa: E402
    _load_static_patched, _load_daily_channel,
    _features_for_window, _label_for_window,
)
from scripts.compute_lift_from_scores import lift_at_k, patchify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True,
                    help="A model's saved window dir (used to derive matching windows)")
    ap.add_argument("--config", default="configs/paths_narval.yaml")
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--climatology_tif", required=True)
    ap.add_argument("--pred_start", default="2022-05-01")
    ap.add_argument("--pred_end", default="2025-10-31",
                    help="Match the same range used by the model's save_scores")
    ap.add_argument("--in_days", type=int, default=7)
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=45)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--dilate_radius", type=int, default=14)
    ap.add_argument("--n_train_wins", type=int, default=80)
    ap.add_argument("--lookback_days_list", nargs="+", type=int,
                    default=[7, 30, 90])
    ap.add_argument("--k", type=int, default=5000)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    print("=" * 70)
    print("LOGREG on model's exact eval windows + novel labels")
    print("=" * 70)

    # ── 1. Load shared data via benchmark_baselines loader ────────────
    (fwi_p, fire_p, clim_p, all_dates, date_to_idx,
     val_wins, val_win_dates, n_patches, grid) = load_data(
        args.config, args.pred_start, args.pred_end,
        args.in_days, args.lead_start, args.lead_end,
        args.patch_size, args.dilate_radius,
        True, args.climatology_tif, args.fire_label_npy,
    )
    print(f"  loaded data: T={len(all_dates)}  val_wins={len(val_wins)}")

    # ── 2. Load extra logreg features (2t, sm20) ──────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["paths"]
    fwi_idx = _build_file_index(cfg["fwi_dir"])
    sample_path = fwi_idx[all_dates[0]]
    with rasterio.open(sample_path) as src:
        H, W = src.height, src.width
    P = args.patch_size
    Hc, Wc = H - H % P, W - W % P
    print(f"  Hc={Hc} Wc={Wc}")

    if clim_p is None:
        sys.exit("ERROR: climatology_tif required")
    clim_per_patch = clim_p.mean(axis=1).astype(np.float32)

    terrain_dir = cfg.get("terrain_dir", "data/terrain")
    slope_path = os.path.join(terrain_dir, "slope.tif")
    if not os.path.exists(slope_path):
        sys.exit(f"ERROR: slope TIF missing: {slope_path}")
    slope_per_patch = _load_static_patched(slope_path, Hc, Wc, P).mean(axis=1)

    obs_root = cfg.get("observation_dir")
    print("\n  Loading 2t, sm20 daily channels (~few min)...")
    t2_idx = _build_file_index(os.path.join(obs_root, "2t"))
    sm_idx = _build_file_index(os.path.join(obs_root, "sm20"))
    t2_p = _load_daily_channel(t2_idx, all_dates, Hc, Wc, P, "2t")
    sm_p = _load_daily_channel(sm_idx, all_dates, Hc, Wc, P, "sm20")

    # ── 3. Build train windows + fit logreg ───────────────────────────
    all_windows = _build_s2s_windows_calendar(
        all_dates, date_to_idx, args.in_days, args.lead_start, args.lead_end)
    pred_start = date.fromisoformat(args.pred_start)
    train_wins = [w for w in all_windows if all_dates[w[1]] < pred_start]
    print(f"\n  train_wins: {len(train_wins)}  (sampling {args.n_train_wins})")

    rng = np.random.default_rng(0)
    if len(train_wins) > args.n_train_wins:
        idxs = rng.choice(len(train_wins), args.n_train_wins, replace=False)
        train_wins_s = [train_wins[i] for i in sorted(idxs)]
    else:
        train_wins_s = train_wins

    print("\n  Building train features...")
    X_list, y_list = [], []
    for i, w in enumerate(train_wins_s):
        X = _features_for_window(w, fwi_p, t2_p, sm_p,
                                 clim_per_patch, slope_per_patch)
        y = _label_for_window(w, fire_p)
        X_list.append(X); y_list.append(y)
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    print(f"  X_train: {X_train.shape}  pos_rate={y_train.mean():.4%}")

    scaler = StandardScaler().fit(X_train)
    print("  Fitting LogisticRegression(class_weight='balanced')...")
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=200,
                            solver="lbfgs", n_jobs=-1)
    lr.fit(scaler.transform(X_train), y_train)
    print(f"  fit in {time.time()-t0:.0f}s")
    print(f"  coefs: fwi_mean={lr.coef_[0,0]:+.3f} fwi_max={lr.coef_[0,1]:+.3f} "
          f"2t={lr.coef_[0,2]:+.3f} sm20={lr.coef_[0,3]:+.3f} "
          f"clim={lr.coef_[0,4]:+.3f} slope={lr.coef_[0,5]:+.3f}")

    # ── 4. Evaluate on each saved window from model's scores_dir ──────
    fire_full = np.load(args.fire_label_npy, mmap_mode="r")  # (T, H, W)
    sidecar = str(args.fire_label_npy).rsplit(".", 1)[0] + ".json"
    label_start = date.fromisoformat("2000-05-01")
    if os.path.exists(sidecar):
        import json
        with open(sidecar) as f:
            label_start = date.fromisoformat(json.load(f)["date_range"][0])

    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"\n  evaluating on {len(files)} model windows from {args.scores_dir}")

    rows = []
    for fpath in files:
        z = np.load(fpath)
        win_date_str = str(z["win_date"])
        if not win_date_str:
            continue
        win_date = date.fromisoformat(win_date_str)
        hs, he, ts, te = int(z["hs"]), int(z["he"]), int(z["ts"]), int(z["te"])
        win = (hs, he, ts, te)

        # Logreg score for this window
        X_win = _features_for_window(win, fwi_p, t2_p, sm_p,
                                     clim_per_patch, slope_per_patch)
        proba = lr.predict_proba(scaler.transform(X_win))[:, 1].astype(np.float32)
        # Broadcast to (n_patches, P*P) for consistent eval
        score = np.broadcast_to(proba[:, None], (n_patches, P * P)).copy()

        label_total = z["label_agg"].astype(np.uint8)  # use the SAME label as model
        rec = {"win_date": win_date_str,
               "n_total": int(label_total.sum())}

        # total lift
        lift_t, _, _ = lift_at_k(score.reshape(-1), label_total.reshape(-1), args.k)
        rec[f"logreg_lift_total"] = lift_t

        # Novel labels
        for lookback in args.lookback_days_list:
            past_start = win_date - timedelta(days=lookback + 7)
            past_start_idx = max(0, (past_start - label_start).days)
            past_end_idx = min(fire_full.shape[0],
                               (win_date - label_start).days + 1)
            burn_recent = (fire_full[past_start_idx:past_end_idx]
                           .max(axis=0).astype(np.uint8))
            burn_recent_p = patchify(burn_recent, P)
            novel = ((label_total > 0) & (burn_recent_p == 0)).astype(np.uint8)
            rec[f"n_novel_{lookback}d"] = int(novel.sum())
            lift_n, _, _ = lift_at_k(score.reshape(-1), novel.reshape(-1), args.k)
            rec[f"logreg_lift_novel_{lookback}d"] = lift_n

        rows.append(rec)
        print(f"  {win_date_str}: total={rec['logreg_lift_total']:.2f}x  "
              + " ".join(f"novel_{lb}={rec[f'logreg_lift_novel_{lb}d']:.2f}x"
                         for lb in args.lookback_days_list))

    # ── Write CSV + summary ────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n  wrote {args.output_csv}")

    print("\n" + "=" * 70)
    print(f"SUMMARY  Lift@{args.k}  on {len(rows)} model-eval windows")
    print("=" * 70)
    for col in ["logreg_lift_total"] + [f"logreg_lift_novel_{lb}d"
                                         for lb in args.lookback_days_list]:
        vals = [r[col] for r in rows
                if not (isinstance(r[col], float) and np.isnan(r[col]))]
        if vals:
            print(f"  {col:30s} = {np.mean(vals):.2f} ± {np.std(vals):.2f}x  "
                  f"(n={len(vals)})")


if __name__ == "__main__":
    main()
