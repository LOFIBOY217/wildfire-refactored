"""
Post-hoc model + climatology ensemble — sweep α to find the best blend
WITHOUT retraining. 10× cheaper than the GPU clim_blend training jobs.

For each window and each α ∈ [0, 0.1, ..., 1.0]:
    p_ens = α × prob_model + (1 − α) × prob_clim_norm
Then compute:
    Recall@budget at {0.1, 0.5, 1, 5, 10}% of Canada area
    Lift@5000 (pixel-level)
    Lift@30km (event-level, 15-px max-pool)

Reuses existing model npz files (prob_agg + label_agg) + climatology TIFs.
Pure CPU, ~10 min for 11 alphas × ~580 windows.

Usage:
  python -m scripts.posthoc_clim_blend_sweep \\
      --scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
      --fire_clim_dir data/fire_clim_annual_nbac \\
      --output outputs/posthoc_clim_blend_sweep.json
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.recall_at_budget import (
    parse_date, list_score_files, patches_to_2d,
    connected_fire_events, recall_at_budget, bootstrap_ci,
)


def load_clim_for_year(fire_clim_dir, year, cache):
    if year in cache:
        return cache[year]
    import rasterio
    f = Path(fire_clim_dir) / f"fire_clim_upto_{year - 1}.tif"
    if not f.exists():
        f = Path(fire_clim_dir) / "fire_clim_upto_2022.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    cache[year] = arr
    return arr


def lift_at_k(scores_flat, labels_flat, k):
    valid = np.isfinite(scores_flat) & np.isfinite(labels_flat)
    s = scores_flat[valid]
    y = labels_flat[valid]
    if len(s) == 0 or y.sum() == 0:
        return float("nan")
    base = float(y.mean())
    if base <= 0:
        return float("nan")
    k = min(k, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    return float(y[top].mean() / base)


def lift_at_30km(score_2d, label_2d, k, pool=15):
    H, W = score_2d.shape
    Hp, Wp = H // pool, W // pool
    s = score_2d[:Hp * pool, :Wp * pool].reshape(Hp, pool, Wp, pool).max(axis=(1, 3))
    y = label_2d[:Hp * pool, :Wp * pool].reshape(Hp, pool, Wp, pool).max(axis=(1, 3))
    return lift_at_k(s.flatten(), y.flatten(), k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--fire_clim_dir", default="data/fire_clim_annual_nbac")
    ap.add_argument("--pred_start", default="2022-05-01")
    ap.add_argument("--pred_end", default="2025-10-31")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8, 0.9, 1.0])
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.05, 0.10])
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--n_rows", type=int, default=142)
    ap.add_argument("--n_cols", type=int, default=169)
    ap.add_argument("--limit_windows", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    P, NR, NC = args.patch_size, args.n_rows, args.n_cols
    H, W = NR * P, NC * P

    pred_start = parse_date(args.pred_start)
    pred_end = parse_date(args.pred_end)
    score_files = list_score_files(args.scores_dir, pred_start, pred_end, True)
    if args.limit_windows > 0:
        score_files = score_files[:args.limit_windows]
    print(f"Windows: {len(score_files)}")
    print(f"Alphas: {args.alphas}")
    print(f"Budgets: {args.budgets}")

    clim_cache = {}
    # results[alpha][window_idx][budget_idx] = recall
    n_a, n_w, n_b = len(args.alphas), len(score_files), len(args.budgets)
    R = np.full((n_a, n_w, n_b), np.nan)
    L5K = np.full((n_a, n_w), np.nan)
    L30 = np.full((n_a, n_w), np.nan)

    t0 = datetime.now()
    for wi, (t, npz_path) in enumerate(score_files):
        try:
            npz = np.load(npz_path)
            if "prob_agg" not in npz.files or "label_agg" not in npz.files:
                continue
            prob_model_2d = patches_to_2d(npz["prob_agg"].astype(np.float32),
                                          NR, NC, P)
            label_2d = (patches_to_2d(npz["label_agg"], NR, NC, P) > 0).astype(np.uint8)
            if label_2d.sum() == 0:
                continue
            clim = load_clim_for_year(args.fire_clim_dir, t.year, clim_cache)
            if clim is None:
                continue
            clim_2d = clim[:H, :W].astype(np.float32)
            valid = np.isfinite(clim_2d) & np.isfinite(prob_model_2d)
            if valid.sum() == 0:
                continue
            # Min-max normalise both to [0, 1] so α is interpretable
            def _mm01(x):
                v = x[valid]
                lo, hi = float(v.min()), float(v.max())
                if hi - lo < 1e-12:
                    return np.zeros_like(x)
                return np.where(valid, (x - lo) / (hi - lo), 0.0)
            p_m = _mm01(prob_model_2d)
            p_c = _mm01(clim_2d)
            event_lbl, n_events = connected_fire_events(label_2d)

            for ai, a in enumerate(args.alphas):
                p_ens = a * p_m + (1 - a) * p_c
                p_ens = np.where(valid, p_ens, -np.inf)
                # Recall@budget
                for bi, B in enumerate(args.budgets):
                    r = recall_at_budget(p_ens, label_2d, valid, B,
                                         event_lbl, n_events)
                    R[ai, wi, bi] = r["recall"]
                # Lift@5000
                L5K[ai, wi] = lift_at_k(p_ens.flatten(), label_2d.flatten(), 5000)
                # Lift@30km
                p_score_2d = np.where(valid, p_ens, np.nan)
                L30[ai, wi] = lift_at_30km(np.where(np.isfinite(p_score_2d),
                                                    p_score_2d, -np.inf),
                                            label_2d, 5000)
            if (wi + 1) % 50 == 0:
                print(f"  [{wi+1}/{len(score_files)}] "
                      f"({(datetime.now()-t0).total_seconds():.0f}s)")
        except Exception as e:
            print(f"  [WARN] {npz_path}: {e}")
            continue

    # Aggregate
    summary = {"alphas": list(args.alphas), "budgets": list(args.budgets),
               "n_windows": int(np.isfinite(L5K).sum(axis=1).max()),
               "results": []}

    print(f"\n{'α':>5} | {'Recall@1%':>14} | {'Recall@5%':>14} | "
          f"{'Recall@10%':>14} | {'Lift@5000':>14} | {'Lift@30km':>14}")
    print("-" * 100)
    for ai, a in enumerate(args.alphas):
        row = {"alpha": float(a), "by_budget": [],
               "lift_5000": {}, "lift_30km": {}}
        # Pick budget indices for the 3 we want to print
        b_print = [0.01, 0.05, 0.10]
        cells = [f"{a:>5.2f}"]
        for B in b_print:
            bi = args.budgets.index(B)
            recalls = R[ai, :, bi][np.isfinite(R[ai, :, bi])]
            m, lo, hi = bootstrap_ci(list(recalls)) if len(recalls) > 0 else (float('nan'),)*3
            cells.append(f"{m*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]")
        # All budgets in summary
        for bi, B in enumerate(args.budgets):
            recalls = R[ai, :, bi][np.isfinite(R[ai, :, bi])]
            m, lo, hi = bootstrap_ci(list(recalls)) if len(recalls) > 0 else (float('nan'),)*3
            row["by_budget"].append({"budget": float(B), "mean": m,
                                      "ci_lo": lo, "ci_hi": hi})
        # Lifts
        l5 = L5K[ai, :][np.isfinite(L5K[ai, :])]
        l30 = L30[ai, :][np.isfinite(L30[ai, :])]
        m5, lo5, hi5 = bootstrap_ci(list(l5)) if len(l5) > 0 else (float('nan'),)*3
        m30, lo30, hi30 = bootstrap_ci(list(l30)) if len(l30) > 0 else (float('nan'),)*3
        row["lift_5000"] = {"mean": m5, "ci_lo": lo5, "ci_hi": hi5}
        row["lift_30km"] = {"mean": m30, "ci_lo": lo30, "ci_hi": hi30}
        cells.append(f"{m5:5.2f}× [{lo5:4.2f},{hi5:4.2f}]")
        cells.append(f"{m30:5.2f}× [{lo30:4.2f},{hi30:4.2f}]")
        print(" | ".join(cells))
        summary["results"].append(row)

    # Find optima
    print(f"\nOptimal α per metric:")
    for B in [0.01, 0.05, 0.10]:
        bi = args.budgets.index(B)
        means = [r["by_budget"][bi]["mean"] for r in summary["results"]]
        best_ai = int(np.nanargmax(means))
        print(f"  Recall@{B*100:.0f}% : α* = {args.alphas[best_ai]:.2f}  "
              f"(recall = {means[best_ai]*100:.1f}%)")
    means_l5 = [r["lift_5000"]["mean"] for r in summary["results"]]
    best_l5 = int(np.nanargmax(means_l5))
    print(f"  Lift@5000   : α* = {args.alphas[best_l5]:.2f}  "
          f"(lift = {means_l5[best_l5]:.2f}×)")
    means_l30 = [r["lift_30km"]["mean"] for r in summary["results"]]
    best_l30 = int(np.nanargmax(means_l30))
    print(f"  Lift@30km   : α* = {args.alphas[best_l30]:.2f}  "
          f"(lift = {means_l30[best_l30]:.2f}×)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
