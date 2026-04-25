#!/usr/bin/env python3
"""
Add climatology + persistence lift on the EXACT SAME 20 windows that
the model save_scores ran on. This makes the comparison apples-to-apples
(prior novel_eval used a different val-window subset).

Reads a model's saved window scores (for the dates), then for each
window:
  - Loads climatology from TIF, patchifies same way
  - Builds persistence score from past 7 days of fire stack
  - Computes total + novel lift on the same labels
  - Outputs a comparison CSV
"""
import argparse
import csv
import glob
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compute_lift_from_scores import lift_at_k, patchify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True,
                    help="A model's saved window dir (used to derive matching windows)")
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--label_start_date", default="2000-05-01")
    ap.add_argument("--climatology_tif", required=True)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--lookback_days_list", nargs="+", type=int,
                    default=[7, 30, 90])
    ap.add_argument("--k", type=int, default=5000)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    print("=== baseline lift on model's exact windows ===")
    fire_full = np.load(args.fire_label_npy, mmap_mode="r")  # (T, H, W) uint8
    label_start = date.fromisoformat(args.label_start_date)
    sidecar = str(args.fire_label_npy).rsplit(".", 1)[0] + ".json"
    if os.path.exists(sidecar):
        import json
        with open(sidecar) as f:
            label_start = date.fromisoformat(json.load(f)["date_range"][0])
        print(f"  label_start (sidecar): {label_start}")

    # Climatology
    P = args.patch_size
    with rasterio.open(args.climatology_tif) as src:
        clim_arr = np.nan_to_num(src.read(1), nan=0.0).astype(np.float32)
    H, W = clim_arr.shape
    Hc, Wc = H - H % P, W - W % P
    clim_p = patchify(clim_arr[:Hc, :Wc], P)   # (n_patches, P²)
    print(f"  climatology patched: {clim_p.shape}  nonzero={np.count_nonzero(clim_p)}")

    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"  {len(files)} windows from {args.scores_dir}")

    rows = []
    for fpath in files:
        z = np.load(fpath)
        label_total_p = z["label_agg"].astype(np.uint8)        # (n_patches, P²)
        win_date_str = str(z["win_date"])
        if not win_date_str:
            continue
        win_date = date.fromisoformat(win_date_str)
        hs, he, ts, te = int(z["hs"]), int(z["he"]), int(z["ts"]), int(z["te"])

        # Persistence score: past 7 days fire density (from absolute calendar)
        past7_start = win_date - timedelta(days=7)
        past7_start_idx = max(0, (past7_start - label_start).days)
        past7_end_idx = min(fire_full.shape[0], (win_date - label_start).days + 1)
        past7_arr = fire_full[past7_start_idx:past7_end_idx].mean(axis=0)
        persist_p = patchify(past7_arr.astype(np.float32), P)

        rec = {"win_date": win_date_str,
               "n_total": int(label_total_p.sum())}

        for lookback in args.lookback_days_list:
            past_start = win_date - timedelta(days=lookback + 7)
            past_start_idx = max(0, (past_start - label_start).days)
            past_end_idx = min(fire_full.shape[0],
                               (win_date - label_start).days + 1)
            burn_recent = (fire_full[past_start_idx:past_end_idx]
                           .max(axis=0).astype(np.uint8))
            burn_recent_p = patchify(burn_recent, P)
            novel = ((label_total_p > 0) & (burn_recent_p == 0)).astype(np.uint8)
            rec[f"n_novel_{lookback}d"] = int(novel.sum())
            for name, score_p in [("clim", clim_p), ("persist", persist_p)]:
                lift_t, _, _ = lift_at_k(score_p.reshape(-1),
                                         label_total_p.reshape(-1), args.k)
                lift_n, _, _ = lift_at_k(score_p.reshape(-1),
                                         novel.reshape(-1), args.k)
                rec[f"{name}_lift_total"] = lift_t
                rec[f"{name}_lift_novel_{lookback}d"] = lift_n
        rows.append(rec)

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n  wrote {args.output_csv}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY  Lift@{args.k}  on {len(rows)} model-eval windows")
    print("=" * 70)
    for name in ["clim", "persist"]:
        for lb in args.lookback_days_list:
            key = f"{name}_lift_novel_{lb}d"
            vals = [r[key] for r in rows if not np.isnan(r[key])]
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                print(f"  {name:8s}  novel_{lb:>2}d  = {m:5.2f} ± {s:.2f}x  (n={len(vals)})")
        # total
        key = f"{name}_lift_total"
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        if vals:
            m, s = float(np.mean(vals)), float(np.std(vals))
            print(f"  {name:8s}  total      = {m:5.2f} ± {s:.2f}x  (n={len(vals)})")


if __name__ == "__main__":
    main()
