#!/usr/bin/env python3
"""
Test ensemble (persistence × climatology) under both total and
novel-ignition labels.

Hypothesis: persistence dominates lift_total but crashes on
lift_novel. Climatology has moderate but stable lift on both.
A weighted combination (alpha * persistence + (1-alpha) * climatology)
might reach the best of both:
  - high lift_total (persistence advantage on continuation)
  - non-zero lift_novel (climatology advantage on new fires)

This is a CHEAP test (no model inference needed) of whether adding a
recency feature would help our transformer. If ensemble beats either
component on novel_30d → recency + spatial prior have synergy →
worth investing in 10ch retrain with fire_recent_density channel.

Usage:
    python -m scripts.evaluate_novel_ensemble \\
        --config configs/paths_narval.yaml \\
        --pred_start 2022-05-01 --pred_end 2024-10-31 \\
        --fire_label_npy ... --climatology_tif ...
"""
import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_baselines import load_data  # noqa: E402
from scripts.evaluate_novel_ignition import (  # noqa: E402
    _per_window_lift, _build_novel_label, _build_total_label,
)


def _normalize(x):
    """Min-max normalize to [0, 1] per pixel-array (avoid div-by-zero)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x.astype(np.float32) - lo) / (hi - lo)


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
    ap.add_argument("--n_sample_wins", type=int, default=20)
    ap.add_argument("--fire_season_only", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--climatology_tif", required=True)
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="Ensemble weight for persistence "
                         "(1.0 = pure persistence, 0.0 = pure climatology)")
    ap.add_argument("--lookback_days", type=int, default=30)
    ap.add_argument("--k", type=int, default=5000)
    ap.add_argument("--output_csv", default="outputs/benchmark_ensemble.csv")
    args = ap.parse_args()

    print("=" * 70)
    print("ENSEMBLE eval: persistence × climatology")
    print("=" * 70)

    (fwi_p, fire_p, clim_p, all_dates, date_to_idx,
     val_wins, val_win_dates, n_patches, grid) = load_data(
        args.config, args.pred_start, args.pred_end,
        args.in_days, args.lead_start, args.lead_end,
        args.patch_size, args.dilate_radius,
        args.fire_season_only, args.climatology_tif,
        args.fire_label_npy,
    )

    rng = np.random.default_rng(0)
    if len(val_wins) > args.n_sample_wins:
        idxs = rng.choice(len(val_wins), args.n_sample_wins, replace=False)
        val_wins_s = [val_wins[i] for i in sorted(idxs)]
    else:
        val_wins_s = val_wins
    print(f"  evaluating on {len(val_wins_s)} val windows")

    # Pre-compute climatology score (constant across windows)
    clim_score = _normalize(clim_p)   # (n_patches, P²)

    rows = []
    for alpha in args.alphas:
        print(f"\n  alpha={alpha} (persistence={alpha}, clim={1-alpha})")
        lifts_total = []
        lifts_novel = []
        for win in val_wins_s:
            hs, he, ts, te = win
            persist_raw = fire_p[hs:he].astype(np.float32).mean(axis=0)
            persist_n = _normalize(persist_raw)
            ensemble = alpha * persist_n + (1 - alpha) * clim_score
            label_t = _build_total_label(win, fire_p)
            label_n = _build_novel_label(win, fire_p, args.lookback_days)
            mt = _per_window_lift(ensemble, label_t, [args.k], grid,
                                  args.patch_size)
            mn = _per_window_lift(ensemble, label_n, [args.k], grid,
                                  args.patch_size)
            if not np.isnan(mt[args.k]["lift"]):
                lifts_total.append(mt[args.k]["lift"])
            if not np.isnan(mn[args.k]["lift"]):
                lifts_novel.append(mn[args.k]["lift"])
        mt_mean = float(np.mean(lifts_total)) if lifts_total else float('nan')
        mn_mean = float(np.mean(lifts_novel)) if lifts_novel else float('nan')
        mt_std = float(np.std(lifts_total)) if lifts_total else float('nan')
        mn_std = float(np.std(lifts_novel)) if lifts_novel else float('nan')
        print(f"    L@{args.k} total      = {mt_mean:.2f} ± {mt_std:.2f}x  "
              f"(n={len(lifts_total)})")
        print(f"    L@{args.k} novel_{args.lookback_days}d  = {mn_mean:.2f} ± {mn_std:.2f}x  "
              f"(n={len(lifts_novel)})")
        rows.append({
            "alpha_persistence": alpha,
            "k": args.k,
            "lift_total_mean": mt_mean,
            "lift_total_std": mt_std,
            f"lift_novel_{args.lookback_days}d_mean": mn_mean,
            f"lift_novel_{args.lookback_days}d_std": mn_std,
            "n_eval_total": len(lifts_total),
            "n_eval_novel": len(lifts_novel),
        })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    print("\n" + "=" * 70)
    print(f"SUMMARY  Lift@{args.k}  (persistence weight α)")
    print("=" * 70)
    print(f"  α      total           novel_{args.lookback_days}d")
    for r in rows:
        a = r["alpha_persistence"]
        t = r["lift_total_mean"]
        n = r[f"lift_novel_{args.lookback_days}d_mean"]
        tag = ""
        if a == 1.0: tag = " ← pure persistence"
        if a == 0.0: tag = " ← pure climatology"
        print(f"  {a:.2f}   {t:6.2f}x        {n:6.2f}x{tag}")
    print(f"\n  wrote {args.output_csv}")


if __name__ == "__main__":
    main()
