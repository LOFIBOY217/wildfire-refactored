#!/usr/bin/env python3
"""
Recompute cluster_lift on existing save_window_scores dirs using the
new tile_size_percentile + stratified output. Compares old (p50) vs
new (p75, p90) and shows whether the heavy-tail bias inflates lift.

Usage:
  python scripts/recompute_cluster_lift_with_fix.py \
      --scores_dir outputs/window_scores_full/v3_9ch_enc14_4y_2018 \
      --out outputs/cluster_lift_audit_enc14_4y.csv

Output: per-window CSV with columns
  win_date, n_clusters, median_size, max_size,
  lift_p50, lift_p75, lift_p90,
  lift_small, lift_medium, lift_large
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Re-implement patches_to_image inline to avoid heavy train_v3 imports
def _patches_to_image(patches, nph, npw, P):
    """(n_patches, P*P) -> (nph*P, npw*P) 2D image."""
    arr = patches.reshape(nph, npw, P, P)
    return arr.transpose(0, 2, 1, 3).reshape(nph * P, npw * P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5000)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--grid", nargs=2, type=int, default=[142, 169])
    args = ap.parse_args()

    # Lazy import after path injection — train_v3 needs torch but only
    # the cluster_lift function is pure numpy
    from src.training.train_v3 import _compute_cluster_lift_k

    nph, npw = args.grid
    P = args.patch_size

    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"  found {len(files)} window files")

    rows = []
    for fi, fpath in enumerate(files):
        z = np.load(fpath)
        prob = z["prob_agg"].astype(np.float32)
        label = z["label_agg"].astype(np.uint8)
        win_date = str(z["win_date"])

        prob_2d = _patches_to_image(prob, nph, npw, P)
        label_2d = _patches_to_image(label, nph, npw, P)

        if int(label_2d.sum()) == 0:
            continue

        r_p50 = _compute_cluster_lift_k(prob_2d, label_2d, args.k,
                                         tile_size_percentile=50,
                                         return_stratified=True)
        r_p75 = _compute_cluster_lift_k(prob_2d, label_2d, args.k,
                                         tile_size_percentile=75)
        r_p90 = _compute_cluster_lift_k(prob_2d, label_2d, args.k,
                                         tile_size_percentile=90)

        # Find max cluster size to characterize tail
        from scipy.ndimage import label as ndimage_label
        cmap, n_raw = ndimage_label(label_2d, structure=np.ones((3, 3), bool))
        max_size = int(np.bincount(cmap.ravel())[1:].max()) if n_raw > 0 else 0

        rows.append({
            "win_date": win_date,
            "n_clusters": r_p50["n_clusters"],
            "median_size": r_p50["median_cluster_size"],
            "max_size": max_size,
            "tile_side_p50": r_p50["tile_side"],
            "tile_side_p75": r_p75["tile_side"],
            "tile_side_p90": r_p90["tile_side"],
            "lift_p50": r_p50["lift_k"],
            "lift_p75": r_p75["lift_k"],
            "lift_p90": r_p90["lift_k"],
            "lift_small": r_p50.get("lift_k_small", float("nan")),
            "lift_medium": r_p50.get("lift_k_medium", float("nan")),
            "lift_large": r_p50.get("lift_k_large", float("nan")),
            "n_small": r_p50.get("n_clusters_small", 0),
            "n_medium": r_p50.get("n_clusters_medium", 0),
            "n_large": r_p50.get("n_clusters_large", 0),
        })

        if (fi + 1) % 50 == 0:
            print(f"  done {fi+1}/{len(files)} windows")

    if not rows:
        print("  no windows with fire — nothing to write")
        return

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {args.out} ({len(rows)} windows)")

    # Summary
    print()
    print("=" * 70)
    print(f"SUMMARY (n={len(rows)} windows with fire)")
    print("=" * 70)
    arr_p50 = np.array([r["lift_p50"] for r in rows])
    arr_p75 = np.array([r["lift_p75"] for r in rows])
    arr_p90 = np.array([r["lift_p90"] for r in rows])
    print(f"  Cluster Lift @ K={args.k}:")
    print(f"    tile p50 (current): mean={arr_p50.mean():.3f}  median={np.median(arr_p50):.3f}")
    print(f"    tile p75 (fix1):    mean={arr_p75.mean():.3f}  median={np.median(arr_p75):.3f}")
    print(f"    tile p90 (fix2):    mean={arr_p90.mean():.3f}  median={np.median(arr_p90):.3f}")
    print()
    print(f"  Inflation from heavy-tail bias:")
    print(f"    p50 vs p75: {100*(arr_p50.mean()/arr_p75.mean()-1):+.1f}%")
    print(f"    p50 vs p90: {100*(arr_p50.mean()/arr_p90.mean()-1):+.1f}%")
    print()

    # Stratified
    arr_s = np.array([r["lift_small"] for r in rows], dtype=np.float64)
    arr_s = arr_s[~np.isnan(arr_s)]
    arr_m = np.array([r["lift_medium"] for r in rows], dtype=np.float64)
    arr_m = arr_m[~np.isnan(arr_m)]
    arr_l = np.array([r["lift_large"] for r in rows], dtype=np.float64)
    arr_l = arr_l[~np.isnan(arr_l)]
    print(f"  Stratified Lift (across windows that have that bin):")
    print(f"    small  (<100 px) :  mean={arr_s.mean():.3f}  n={len(arr_s)}")
    print(f"    medium (100-1k)  :  mean={arr_m.mean():.3f}  n={len(arr_m)}")
    print(f"    large  (>1000)   :  mean={arr_l.mean():.3f}  n={len(arr_l)}")


if __name__ == "__main__":
    main()
