#!/usr/bin/env python3
"""
Compute bootstrap 95% CI for novel-ignition lifts from
model_novel_lift_*.csv files. Reads the per-window CSV (one row per
val window) and resamples 2000 times to estimate confidence interval
on the mean lift.

Usage:
    python scripts/bootstrap_ci.py outputs/model_novel_lift_*.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np


def bootstrap_ci(values, n_boot=2000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float('nan'), float('nan'), float('nan'), 0
    n = len(arr)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = arr[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(arr.mean()), float(lo), float(hi), n


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: bootstrap_ci.py <csv...>")
        sys.exit(1)

    print(f"{'file':40s} {'metric':25s} {'mean':>8s} {'95% CI':>20s}  n")
    print("-" * 100)

    for path in sorted(paths):
        name = Path(path).stem
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        # Find lift columns
        lift_cols = [k for k in rows[0].keys()
                     if k.startswith("lift_") and "_5000" in k]
        for col in sorted(lift_cols):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[col]))
                except (ValueError, KeyError):
                    pass
            mean, lo, hi, n = bootstrap_ci(vals)
            print(f"{name:40s} {col:25s} {mean:6.2f}x  [{lo:5.2f}, {hi:5.2f}]  {n}")
        print()


if __name__ == "__main__":
    main()
