#!/usr/bin/env python3
"""
Per-month Lift@K analysis from save_window_scores output.

For each month-of-year (5..10, fire season), aggregate all val windows
that fall in that month and compute:
  - n_windows
  - n_fire_pixels (pooled across windows)
  - mean Lift@K (per K) over windows
  - bootstrap 95% CI

Usage:
  python scripts/analyze_per_month_lift.py \
      --scores_dir outputs/window_scores_full/v3_9ch_enc14_4y_2018 \
      --out outputs/per_month_enc14_4y.csv
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np


def parse_date(fname):
    m = re.search(r"window_\d+_(\d{4})-(\d{2})-(\d{2})\.npz$", fname)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return y, mo, d


def lift_at_k(prob_flat, label_flat, k):
    n = prob_flat.size
    if k > n:
        k = n
    idx = np.argpartition(-prob_flat, k - 1)[:k]
    n_pos = int(label_flat[idx].sum())
    precision_at_k = n_pos / k
    baseline = float(label_flat.mean())
    if baseline <= 0:
        return float("nan"), precision_at_k, baseline
    return precision_at_k / baseline, precision_at_k, baseline


def bootstrap_ci(values, n_boot=2000, seed=0):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan"), float("nan")
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_values", nargs="+", type=int,
                    default=[1000, 2500, 5000, 10000, 25000])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"  found {len(files)} window files in {args.scores_dir}")

    # Group windows by (year, month) and (month_of_year)
    by_month = defaultdict(list)  # mo -> list of (lift_per_K dict, n_fire)

    t0_total = 0
    for i, f in enumerate(files):
        d = parse_date(os.path.basename(f))
        if d is None:
            continue
        y, mo, dd = d
        npz = np.load(f)
        prob = npz["prob_agg"].astype(np.float32).ravel()
        label = npz["label_agg"].astype(np.uint8).ravel()
        n_fire = int(label.sum())
        if n_fire == 0:
            # still record window count for fairness
            by_month[mo].append((None, 0))
            continue
        per_k = {}
        for K in args.k_values:
            lift, prec, base = lift_at_k(prob, label, K)
            per_k[K] = (lift, prec)
        by_month[mo].append((per_k, n_fire))
        if (i + 1) % 50 == 0:
            print(f"  done {i + 1}/{len(files)} windows")

    # Aggregate per month
    rows = []
    for mo in sorted(by_month.keys()):
        entries = by_month[mo]
        n_total_wins = len(entries)
        with_fire = [e for e in entries if e[0] is not None]
        n_with_fire = len(with_fire)
        n_fire_total = sum(e[1] for e in entries)
        row = {
            "month": mo,
            "n_windows": n_total_wins,
            "n_with_fire": n_with_fire,
            "n_fire_total": n_fire_total,
        }
        for K in args.k_values:
            lifts = [e[0][K][0] for e in with_fire if e[0][K][0] == e[0][K][0]]
            precs = [e[0][K][1] for e in with_fire]
            if not lifts:
                row[f"lift_{K}_mean"] = float("nan")
                row[f"lift_{K}_ci_low"] = float("nan")
                row[f"lift_{K}_ci_high"] = float("nan")
                row[f"prec_{K}_mean"] = float("nan")
                continue
            row[f"lift_{K}_mean"] = float(np.mean(lifts))
            ci_lo, ci_hi = bootstrap_ci(lifts)
            row[f"lift_{K}_ci_low"] = ci_lo
            row[f"lift_{K}_ci_high"] = ci_hi
            row[f"prec_{K}_mean"] = float(np.mean(precs))
        rows.append(row)

    fieldnames = ["month", "n_windows", "n_with_fire", "n_fire_total"]
    for K in args.k_values:
        fieldnames += [f"lift_{K}_mean", f"lift_{K}_ci_low",
                       f"lift_{K}_ci_high", f"prec_{K}_mean"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {args.out} ({len(rows)} months)")

    # Pretty print summary
    print()
    print(f"{'mo':>3} {'n_win':>6} {'n_fire':>6} {'pixels':>10}",
          end="")
    for K in args.k_values:
        print(f" {'lift@' + str(K):>12}", end="")
    print()
    for r in rows:
        print(f"{r['month']:>3} {r['n_windows']:>6} "
              f"{r['n_with_fire']:>6} {r['n_fire_total']:>10}", end="")
        for K in args.k_values:
            v = r[f"lift_{K}_mean"]
            print(f" {v:>12.3f}" if v == v else f" {'nan':>12}", end="")
        print()


if __name__ == "__main__":
    main()
