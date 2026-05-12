"""
Audit Lift@30km — identical scores+labels, all 4 algorithmic combinations.

The two production pipelines disagree on the SAME ckpt:
    train_v3 val loop (metrics.py)            → 7.26x
    compute_full_metric_card.py / ensemble    → 4.09x

This script computes Lift@30km on the SAME npz, four ways:
    1. mean-pool score, max-pool label, K = 5000 // (pool²)
    2. max-pool  score, max-pool label, K = 5000 // (pool²)
    3. mean-pool score, max-pool label, K = 5000 (NOT scaled)
    4. max-pool  score, max-pool label, K = 5000 (NOT scaled)

(1) = metrics.py convention (train_v3)
(4) = compute_full_metric_card / ensemble convention

Differences (1) vs (4) isolate the aggregation AND K-scaling axes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def lift_at_k(score, label, k):
    valid = np.isfinite(score) & np.isfinite(label)
    s = score[valid]; y = label[valid]
    if len(s) == 0 or y.sum() == 0:
        return float("nan")
    base = float(y.mean())
    if base <= 0:
        return float("nan")
    k = int(min(k, len(s)))
    top = np.argpartition(-s, k - 1)[:k]
    return float(y[top].mean() / base)


def patches_to_2d(patch_arr, n_rows, n_cols, P):
    return (patch_arr.reshape(n_rows, n_cols, P, P)
            .transpose(0, 2, 1, 3).reshape(n_rows * P, n_cols * P))


def coarsen(arr2d, pool, mode):
    H, W = arr2d.shape
    Hp, Wp = H // pool, W // pool
    block = arr2d[:Hp * pool, :Wp * pool].reshape(Hp, pool, Wp, pool)
    if mode == "mean":
        return block.mean(axis=(1, 3))
    elif mode == "max":
        return block.max(axis=(1, 3))
    else:
        raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True,
                    help="window_scores_full/<run> dir with window_*.npz")
    ap.add_argument("--n_rows", type=int, default=142)
    ap.add_argument("--n_cols", type=int, default=169)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--pool", type=int, default=15)
    ap.add_argument("--k_fine", type=int, default=5000)
    ap.add_argument("--max_windows", type=int, default=200)
    args = ap.parse_args()

    files = sorted(Path(args.scores_dir).glob("window_*.npz"))[:args.max_windows]
    print(f"Found {len(files)} windows in {args.scores_dir}\n")

    P = args.patch_size
    NR, NC = args.n_rows, args.n_cols
    pool = args.pool
    k_fine = args.k_fine
    k_coarse_scaled = max(1, k_fine // (pool * pool))

    # 4 combinations
    res = {("mean", "scaled"): [], ("max", "scaled"): [],
           ("mean", "unscaled"): [], ("max", "unscaled"): []}
    # Also do FINE (no pooling) Lift@5000 as control
    res["fine_k5000"] = []
    # Number of coarse cells & fire cells per window (avg) for context
    n_coarse_cells = []
    n_coarse_fires = []

    for i, f in enumerate(files):
        npz = np.load(f)
        prob = npz["prob_agg"].astype(np.float32)
        label = npz["label_agg"]
        score_2d = patches_to_2d(prob, NR, NC, P)
        label_2d = (patches_to_2d(label, NR, NC, P) > 0).astype(np.uint8)

        if label_2d.sum() == 0:
            continue

        # Coarsen 4 ways
        score_mean = coarsen(score_2d, pool, "mean").ravel()
        score_max  = coarsen(score_2d, pool, "max").ravel()
        label_max  = coarsen(label_2d, pool, "max").ravel().astype(np.float32)

        # Fine control
        res["fine_k5000"].append(
            lift_at_k(score_2d.flatten(), label_2d.flatten(), k_fine))

        # 4 combos
        res[("mean", "scaled")].append(
            lift_at_k(score_mean, label_max, k_coarse_scaled))
        res[("max",  "scaled")].append(
            lift_at_k(score_max,  label_max, k_coarse_scaled))
        res[("mean", "unscaled")].append(
            lift_at_k(score_mean, label_max, k_fine))
        res[("max",  "unscaled")].append(
            lift_at_k(score_max,  label_max, k_fine))

        n_coarse_cells.append(len(label_max))
        n_coarse_fires.append(int(label_max.sum()))

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(files)}")

    print(f"\nValid windows: {len(res[('mean', 'scaled')])}")
    n_cells_avg = np.mean(n_coarse_cells) if n_coarse_cells else 0
    n_fires_avg = np.mean(n_coarse_fires) if n_coarse_fires else 0
    print(f"Avg coarse cells/window: {n_cells_avg:.0f}")
    print(f"Avg coarse fire cells/window: {n_fires_avg:.1f}")
    print(f"K (scaled to coarse) = {k_coarse_scaled}")
    print(f"K (unscaled, fine)   = {k_fine}")
    print(f"K_unscaled / n_cells = {k_fine / n_cells_avg:.1%}  "
          f"(if > 1: K bigger than valid cells, top-K = ALL cells)")
    print()
    print(f"{'Method':<35} {'Mean Lift':>12} {'Notes'}")
    print("-" * 90)
    print(f"{'fine Lift@5000 (CONTROL)':<35} {np.nanmean(res['fine_k5000']):>12.3f} "
          f"sanity: should match prior reports")
    print(f"{'(1) mean-pool, K-scaled  [metrics.py / train_v3]':<35} "
          f"{np.nanmean(res[('mean','scaled')]):>12.3f} "
          f"K={k_coarse_scaled}")
    print(f"{'(2) max-pool,  K-scaled':<35} "
          f"{np.nanmean(res[('max','scaled')]):>12.3f} "
          f"K={k_coarse_scaled}")
    print(f"{'(3) mean-pool, K-unscaled':<35} "
          f"{np.nanmean(res[('mean','unscaled')]):>12.3f} "
          f"K={k_fine}")
    print(f"{'(4) max-pool,  K-unscaled  [metric_card]':<35} "
          f"{np.nanmean(res[('max','unscaled')]):>12.3f} "
          f"K={k_fine}")


if __name__ == "__main__":
    main()
