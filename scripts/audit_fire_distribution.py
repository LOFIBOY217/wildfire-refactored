#!/usr/bin/env python3
"""
Per-year fire distribution audit.

Purpose: detect systematic differences between train-extended (2000-2017)
and train-original (2018-2022) + val (2022-2025) periods. If 2000-2017
fire distribution differs sharply from 2018+, the extended training may
teach the model a different phenomenon than the one it's evaluated on.

Reports:
  1. Hotspot CSV: count per year, top 3 provinces per year
  2. burn_count annual: total burned pixels per year
  3. fire_clim_upto: nonzero pixels per year (cumulative history)

Usage:
    python scripts/audit_fire_distribution.py
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--hotspot-csv",
                    default="data/hotspot/hotspot_2000_2025.csv")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()

    os.chdir(args.root)

    print(f"\n{'='*80}")
    print("FIRE DISTRIBUTION AUDIT")
    print(f"{'='*80}\n")

    # ─── 1. Hotspot CSV counts per year ───
    print("[1/3] Hotspot CSV — count per year + fire season share\n")
    if not os.path.exists(args.hotspot_csv):
        print(f"  MISSING: {args.hotspot_csv}")
    else:
        df = pd.read_csv(args.hotspot_csv, usecols=['latitude', 'longitude', 'acq_date'])
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        df['year'] = df['acq_date'].dt.year
        df['month'] = df['acq_date'].dt.month

        print(f"  Total rows: {len(df):,}")
        print(f"  Date range: {df['acq_date'].min().date()} → {df['acq_date'].max().date()}\n")

        print(f"  {'Year':>6s}  {'Count':>12s}  {'FireSeasonFrac':>15s}  {'MedianLat':>10s}")
        print(f"  {'-'*55}")
        for y in range(args.start_year, args.end_year + 1):
            yr = df[df.year == y]
            if len(yr) == 0:
                print(f"  {y:>6d}  {'0':>12s}")
                continue
            fire_season = yr[(yr.month >= 5) & (yr.month <= 10)]
            fs_frac = len(fire_season) / len(yr)
            print(f"  {y:>6d}  {len(yr):>12,}  {fs_frac:>15.3f}  "
                  f"{yr['latitude'].median():>10.2f}")

        # Summary: train-extended (2000-2017) vs train-original (2018-2021) vs val (2022-2024)
        extended_count = df[(df.year >= 2000) & (df.year <= 2017)].shape[0]
        original_count = df[(df.year >= 2018) & (df.year <= 2021)].shape[0]
        val_count = df[(df.year >= 2022) & (df.year <= 2024)].shape[0]
        print(f"\n  Summary (by period):")
        print(f"    Extended train (2000-2017, 18y): {extended_count:>12,}  "
              f"avg {extended_count/18:>10,.0f}/year")
        print(f"    Original train (2018-2021, 4y): {original_count:>12,}  "
              f"avg {original_count/4:>10,.0f}/year")
        print(f"    Val period    (2022-2024, 3y): {val_count:>12,}  "
              f"avg {val_count/3:>10,.0f}/year")

        ratio_ext_orig = (extended_count/18) / (original_count/4) if original_count else float('nan')
        ratio_ext_val  = (extended_count/18) / (val_count/3) if val_count else float('nan')
        print(f"\n  Fire-density ratio:")
        print(f"    Extended vs Original: {ratio_ext_orig:.2f}x  (if ≠ 1.0, DRIFT)")
        print(f"    Extended vs Val     : {ratio_ext_val:.2f}x")
        if abs(ratio_ext_orig - 1.0) > 0.3:
            print(f"  ⚠ WARNING: extended train has >30% different fire density than original.")

    # ─── 2. burn_count annual ───
    print(f"\n\n[2/3] burn_count_{{YEAR}}.tif — burned pixels per year\n")
    import rasterio
    print(f"  {'Year':>6s}  {'NonzeroPixels':>15s}  {'Sum':>12s}  {'Max':>8s}")
    print(f"  {'-'*50}")
    for y in range(args.start_year, args.end_year + 1):
        path = f"data/burn_scars/burn_count_{y}.tif"
        if not os.path.exists(path):
            continue
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            nz = int((arr > 0).sum())
            print(f"  {y:>6d}  {nz:>15,}  {int(arr.sum()):>12,}  {int(arr.max()):>8d}")
        except Exception as e:
            print(f"  {y:>6d}  ERR: {e}")

    # ─── 3. fire_clim cumulative ───
    print(f"\n\n[3/3] fire_clim_upto_{{YEAR}}.tif — cumulative hotspot climatology\n")
    print(f"  {'Year':>6s}  {'NonzeroPixels':>15s}  {'Mean(nz)':>10s}  {'Max':>8s}")
    print(f"  {'-'*50}")
    prev_nz = 0
    for y in range(args.start_year, args.end_year + 1):
        path = f"data/fire_clim_annual/fire_clim_upto_{y}.tif"
        if not os.path.exists(path):
            continue
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            nz_mask = arr > 0
            nz = int(nz_mask.sum())
            mean_nz = float(arr[nz_mask].mean()) if nz > 0 else 0.0
            delta = nz - prev_nz
            delta_str = f"(+{delta:,})" if delta >= 0 else f"({delta:,})"
            print(f"  {y:>6d}  {nz:>15,} {delta_str:>10s}  "
                  f"{mean_nz:>10.3f}  {arr.max():>8.3f}")
            prev_nz = nz
        except Exception as e:
            print(f"  {y:>6d}  ERR: {e}")

    print(f"\n{'='*80}\n")
    print("Interpretation notes:")
    print("  - Hotspot count drift: if extended > 2× original, likely")
    print("    detection technology changed (new satellites, etc.)")
    print("  - burn_count sudden jump: NBAC methodology changes")
    print("  - fire_clim should be monotonically non-decreasing (cumulative)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
