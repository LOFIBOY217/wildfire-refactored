#!/usr/bin/env python3
"""
Per-channel per-year distribution audit.

Computes mean/std/nonzero_fraction/min/max for one representative TIF per
(channel, year) pair. Flags distribution drift (e.g., if 2t 2010 has
mean 15°C but 2012 has -40°C, there's a unit/processing issue).

Output: markdown table to stdout; CSV to --out-csv if given.

Usage:
    python scripts/audit_channel_stats.py
    python scripts/audit_channel_stats.py --out-csv audit.csv

Takes ~5 min on Lustre.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

CHANNELS = [
    # (name, dir, pattern, sample_month_day, kind)
    ('FWI',   'data/fwi_data',                    'fwi_{y}{md}.tif',  '0715', 'daily'),
    ('FFMC',  'data/ffmc_data',                   'ffmc_{y}{md}.tif', '0715', 'daily'),
    ('DC',    'data/dc_data',                     'dc_{y}{md}.tif',   '0715', 'daily'),
    ('2t',    'data/ecmwf_observation/2t',        '2t_{y}{md}.tif',   '0715', 'daily'),
    ('2d',    'data/ecmwf_observation/2d',        '2d_{y}{md}.tif',   '0715', 'daily'),
    ('tcw',   'data/ecmwf_observation/tcw',       'tcw_{y}{md}.tif',  '0715', 'daily'),
    ('sm20',  'data/ecmwf_observation/sm20',      'sm20_{y}{md}.tif', '0715', 'daily'),
    ('st20',  'data/ecmwf_observation/st20',      'st20_{y}{md}.tif', '0715', 'daily'),
    ('u10',   'data/era5_u10',                    'u10_{y}{md}.tif',  '0715', 'daily'),
    ('v10',   'data/era5_v10',                    'v10_{y}{md}.tif',  '0715', 'daily'),
    ('cape',  'data/era5_cape',                   'cape_{y}{md}.tif', '0715', 'daily'),
    ('tp',    'data/era5_precip',                 'tp_{y}{md}.tif',   '0715', 'daily'),
    ('swvl2', 'data/era5_deep_soil',              'swvl2_{y}{md}.tif','0715', 'daily'),
    # Annual
    ('fire_clim',  'data/fire_clim_annual', 'fire_clim_upto_{y}.tif',    None, 'annual'),
    ('burn_age',   'data/burn_scars',       'years_since_burn_{y}.tif',  None, 'annual'),
    ('burn_count', 'data/burn_scars',       'burn_count_{y}.tif',        None, 'annual'),
]


def scan_tif(path):
    """Return (mean, std, nz_frac, vmin, vmax) for a single TIF, handling nodata."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nd = src.nodata
        # Mask both nan/inf and declared nodata
        mask = np.isfinite(arr)
        if nd is not None:
            mask &= (arr != nd)
        valid = arr[mask]
        if valid.size == 0:
            return (np.nan, np.nan, 0.0, np.nan, np.nan)
        return (float(valid.mean()), float(valid.std()),
                float(mask.sum() / arr.size),
                float(valid.min()), float(valid.max()))
    except Exception as e:
        return (np.nan, np.nan, 0.0, np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root (contains data/)")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()

    os.chdir(args.root)

    years = list(range(args.start_year, args.end_year + 1))
    rows = []

    print(f"\n{'='*95}")
    print(f"PER-CHANNEL PER-YEAR STATS AUDIT ({args.start_year}-{args.end_year})")
    print(f"{'='*95}")
    print(f"Sampled date per year: Jul 15 (daily) or annual file")
    print(f"Stats: mean / std / nonzero_frac on masked-valid pixels\n")

    for name, d, pat, md, kind in CHANNELS:
        print(f"\n--- {name} ({d}) ---")
        print(f"{'Year':>6s}  {'mean':>12s}  {'std':>10s}  {'nz_frac':>8s}  {'min':>10s}  {'max':>10s}")

        for y in years:
            if kind == 'daily':
                path = os.path.join(d, pat.format(y=y, md=md))
            else:
                path = os.path.join(d, pat.format(y=y))

            if not os.path.exists(path):
                print(f"{y:>6d}  {'MISSING':>12s}")
                rows.append((name, y, np.nan, np.nan, 0.0, np.nan, np.nan))
                continue

            mean, std, nzf, vmin, vmax = scan_tif(path)
            print(f"{y:>6d}  {mean:>12.4f}  {std:>10.4f}  {nzf:>8.3f}  "
                  f"{vmin:>10.3f}  {vmax:>10.3f}")
            rows.append((name, y, mean, std, nzf, vmin, vmax))

        # Flag distribution drift: if per-year means have std > 3× within-year std
        means = [r[2] for r in rows[-len(years):] if not np.isnan(r[2])]
        if len(means) > 5:
            per_year_std = np.std(means)
            within_year_std = np.nanmean([r[3] for r in rows[-len(years):] if not np.isnan(r[3])])
            if within_year_std > 0 and per_year_std > 0.5 * within_year_std:
                print(f"  ⚠ DRIFT FLAG: per-year mean std = {per_year_std:.3f}, "
                      f"within-year mean std = {within_year_std:.3f}")

    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['channel', 'year', 'mean', 'std', 'nz_frac', 'min', 'max'])
            w.writerows(rows)
        print(f"\nCSV written to {args.out_csv}")


if __name__ == "__main__":
    main()
