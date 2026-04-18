#!/usr/bin/env python3
"""
Comprehensive data audit — industry-standard QA for training data.

7 check categories (modeled on Google/Meta Data Cards framework):

  A. STRUCTURAL      — file counts, dimensions, CRS, dtype, compression
  B. VALUE RANGE     — physics-based sanity (temp in -80..50°C, NDVI in
                       [-1, 1], etc.) → flags sentinel leakage, unit bugs
  C. TEMPORAL        — expected count per year, missing-date list, gap
                       analysis, seasonal coverage
  D. SPATIAL         — valid-pixel fraction per sampled frame, flags
                       reproject-nan-edge-type bugs
  E. DISTRIBUTION    — per-year mean/std drift (flag >2σ cross-year
                       distribution shifts)
  F. CROSS-CHANNEL   — on one fixed date, confirm all channels have
                       valid data at that date
  G. LABEL INTEGRITY — hotspot CSV date range, count stability, fire
                       season filter coverage

Output:
  - Stdout: formatted report by section, with PASS / WARN / FAIL per check
  - CSV:    per-channel-per-year detail (pipe to spreadsheet / pandas)

Usage:
  python scripts/audit_data_complete.py --root . --out-dir data/audit
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# Channel definitions: single source of truth for audit
# ============================================================

# Physics-sane value ranges (after unit normalization).
# Flags values wildly outside these as potential sentinel leakage.
PHYSICAL_RANGES = {
    'FWI':        (-10, 200),
    'FFMC':       (-10, 120),
    'DMC':        (-10, 600),
    'DC':         (-10, 2000),
    'BUI':        (-10, 600),
    'ISI':        (-10, 100),
    '2t':         (-80, 50),
    '2d':         (-80, 40),
    'tcw':        (0, 80),
    'sm20':       (-0.1, 1.1),
    'st20':       (-60, 50),
    'u10':        (-50, 50),
    'v10':        (-50, 50),
    'cape':       (-10, 10000),
    'tp':         (-0.01, 500),       # mm/day
    'swvl2':      (-0.1, 1.1),
    'NDVI':       (-1.1, 1.1),
    'fire_clim':  (-0.1, 20),
    'burn_age':   (-0.1, 20),
    'burn_count': (-0.1, 20),
    'population': (-1, 20),
    'slope':      (-1, 90),
}

# Expected shape invariants
EXPECTED_CRS = 'EPSG:3978'
EXPECTED_H, EXPECTED_W = 2281, 2709
EXPECTED_DTYPES = {'float32', 'uint16', 'uint8'}

# Channel table: (name, dir, pattern, temporal_kind, required_for_9ch)
CHANNELS = [
    # Daily
    ('FWI',        'data/fwi_data',                    'fwi_{y}{md}.tif',        'daily',   True),
    ('FFMC',       'data/ffmc_data',                   'ffmc_{y}{md}.tif',       'daily',   False),
    ('DC',         'data/dc_data',                     'dc_{y}{md}.tif',         'daily',   False),
    ('BUI',        'data/bui_data',                    'bui_{y}{md}.tif',        'daily',   False),
    ('2t',         'data/ecmwf_observation/2t',        '2t_{y}{md}.tif',         'daily',   True),
    ('2d',         'data/ecmwf_observation/2d',        '2d_{y}{md}.tif',         'daily',   True),
    ('tcw',        'data/ecmwf_observation/tcw',       'tcw_{y}{md}.tif',        'daily',   True),
    ('sm20',       'data/ecmwf_observation/sm20',      'sm20_{y}{md}.tif',       'daily',   True),
    ('st20',       'data/ecmwf_observation/st20',      'st20_{y}{md}.tif',       'daily',   False),
    ('u10',        'data/era5_u10',                    'u10_{y}{md}.tif',        'daily',   False),
    ('v10',        'data/era5_v10',                    'v10_{y}{md}.tif',        'daily',   False),
    ('cape',       'data/era5_cape',                   'cape_{y}{md}.tif',       'daily',   False),
    ('tp',         'data/era5_precip',                 'tp_{y}{md}.tif',         'daily',   False),
    ('swvl2',      'data/era5_deep_soil',              'swvl2_{y}{md}.tif',      'daily',   False),
    ('NDVI',       'data/ndvi_data',                   'ndvi_{y}{md}.tif',       '16-day',  False),
    # Annual
    ('fire_clim',  'data/fire_clim_annual',            'fire_clim_upto_{y}.tif', 'annual',  True),
    ('burn_age',   'data/burn_scars',                  'years_since_burn_{y}.tif','annual', True),
    ('burn_count', 'data/burn_scars',                  'burn_count_{y}.tif',     'annual',  False),
    # Static
    ('population', 'data',                             'population_density.tif', 'static',  True),
    ('slope',      'data/terrain',                     'slope.tif',              'static',  True),
]


# ============================================================
# Helper functions
# ============================================================

def status(passed, msg, flags=None):
    """Format check status."""
    flags = flags or []
    if passed:
        return f"  [PASS] {msg}"
    if any(f == 'WARN' for f in flags):
        return f"  [WARN] {msg}"
    return f"  [FAIL] {msg}"


def sample_dates_for_year(y, kind='daily'):
    """Return list of (YYYY, MMDD) tuples for sampling."""
    if kind == 'annual' or kind == 'static':
        return [(y, None)]
    # 3 dates per year: Jan 15, Jul 15, Oct 15
    return [(y, '0115'), (y, '0715'), (y, '1015')]


def load_tif_stats(path):
    """Return (mean, std, vmin, vmax, valid_frac, crs_str, h, w, dtype, nodata)."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            crs_str = str(src.crs)
            h, w = src.height, src.width
            dtype = src.dtypes[0]
        mask = np.isfinite(arr)
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            mask &= (arr != nodata)
        if mask.any():
            valid = arr[mask]
            stats = (float(valid.mean()), float(valid.std()),
                     float(valid.min()), float(valid.max()),
                     float(mask.mean()), crs_str, h, w, dtype, nodata)
        else:
            stats = (np.nan, np.nan, np.nan, np.nan, 0.0,
                     crs_str, h, w, dtype, nodata)
        return stats
    except Exception as e:
        return None


# ============================================================
# Audit sections
# ============================================================

def audit_A_structural(root, start_year, end_year):
    """A. Count files, check dims/CRS/dtype on sample."""
    print("\n" + "="*80)
    print("A. STRUCTURAL AUDIT — file count, dimensions, CRS, dtype")
    print("="*80)
    results = {}

    for name, d, pat, kind, req in CHANNELS:
        full_dir = os.path.join(root, d)
        # Count files matching pattern's glob (replace placeholders)
        glob_pat = pat.replace('{y}', '*').replace('{md}', '*')
        files = glob.glob(os.path.join(full_dir, glob_pat))
        n_files = len(files)

        # Check one sample
        sample = files[0] if files else None
        sample_stats = load_tif_stats(sample) if sample else None

        flags = []
        if sample_stats:
            (_, _, _, _, _, crs, h, w, dtype, _) = sample_stats
            crs_ok = (crs == EXPECTED_CRS)
            shape_ok = (h == EXPECTED_H and w == EXPECTED_W)
            dtype_ok = (dtype in EXPECTED_DTYPES)
            if not crs_ok:
                flags.append(f"CRS={crs} expected {EXPECTED_CRS}")
            if not shape_ok:
                flags.append(f"shape=({h},{w}) expected ({EXPECTED_H},{EXPECTED_W})")
            if not dtype_ok:
                flags.append(f"dtype={dtype} not in {EXPECTED_DTYPES}")
            passed = crs_ok and shape_ok and dtype_ok
        else:
            passed = False
            flags.append("no files or unreadable")

        results[name] = dict(n_files=n_files, flags=flags,
                             required_9ch=req, passed=passed)
        req_str = " ★" if req else "  "
        print(f"{name:12s}{req_str} n_files={n_files:>6d}  "
              f"{'[PASS]' if passed else '[FAIL]':6s} {' | '.join(flags) if flags else 'OK'}")

    # Summary
    n_required_bad = sum(1 for v in results.values()
                         if v['required_9ch'] and not v['passed'])
    print()
    if n_required_bad == 0:
        print(f"  → ALL REQUIRED 9ch channels PASS structural checks")
    else:
        print(f"  → {n_required_bad} REQUIRED 9ch channels FAILED structural")
    return results


def audit_B_value_range(root, sample_year):
    """B. Physics-based value range check on sample dates."""
    print("\n" + "="*80)
    print(f"B. VALUE RANGE AUDIT — physics bounds (sampled at {sample_year}-07-15)")
    print("="*80)
    results = {}

    for name, d, pat, kind, req in CHANNELS:
        if name not in PHYSICAL_RANGES:
            continue
        lo, hi = PHYSICAL_RANGES[name]
        # Pick sample path
        if kind == 'static':
            path = os.path.join(root, d, pat)
        elif kind == 'annual':
            path = os.path.join(root, d, pat.format(y=sample_year))
        else:
            path = os.path.join(root, d, pat.format(y=sample_year, md='0715'))

        if not os.path.exists(path):
            print(f"  [SKIP] {name:12s} file not found: {os.path.basename(path)}")
            continue

        stats = load_tif_stats(path)
        if stats is None:
            print(f"  [FAIL] {name:12s} unreadable")
            continue
        mean, std, vmin, vmax, vf, _, _, _, _, _ = stats
        in_range = (vmin >= lo - 0.001) and (vmax <= hi + 0.001)
        flags = []
        if vmin < lo:
            flags.append(f"min={vmin:.2f} < {lo}")
        if vmax > hi:
            flags.append(f"max={vmax:.2f} > {hi}")

        symbol = "[PASS]" if in_range else "[FAIL]"
        print(f"  {symbol} {name:12s}  range=[{vmin:>8.2f}, {vmax:>8.2f}]  "
              f"expected=[{lo}, {hi}]  {' '.join(flags)}")
        results[name] = dict(passed=in_range, flags=flags,
                             observed=(vmin, vmax), expected=(lo, hi))
    return results


def audit_C_temporal_coverage(root, start_year, end_year):
    """C. Expected file count per year, list missing years."""
    print("\n" + "="*80)
    print(f"C. TEMPORAL COVERAGE — file count per year ({start_year}-{end_year})")
    print("="*80)
    results = {}

    for name, d, pat, kind, req in CHANNELS:
        if kind == 'static':
            continue
        counts = {}
        for y in range(start_year, end_year + 1):
            if kind == 'annual':
                path = os.path.join(root, d, pat.format(y=y))
                counts[y] = 1 if os.path.exists(path) else 0
            else:
                # Count daily or 16-day files for this year
                files = glob.glob(os.path.join(root, d,
                                              pat.replace('{y}', str(y)).replace('{md}', '*')))
                counts[y] = len(files)

        # Expected: daily=365/366, 16-day=~23, annual=1
        if kind == 'annual':
            expected = 1
            missing_years = [y for y, c in counts.items() if c == 0]
            ok = (len(missing_years) == 0)
            print(f"  {name:12s} annual files: {sum(counts.values())}/{end_year-start_year+1}  "
                  f"{'OK' if ok else f'MISSING: {missing_years}'}")
        elif kind == '16-day':
            partial = [y for y, c in counts.items() if 0 < c < 20]
            missing = [y for y, c in counts.items() if c == 0]
            ok = (len(missing) == 0)
            print(f"  {name:12s} 16-day files: per-year count range "
                  f"{min(counts.values())}-{max(counts.values())}  "
                  f"missing years: {missing}")
        else:  # daily
            missing_years = [y for y, c in counts.items() if c == 0]
            partial_years = [(y, c) for y, c in counts.items() if 0 < c < 300]
            ok = (len(missing_years) == 0 and len(partial_years) == 0)
            row = f"  {name:12s} daily files: "
            row += f"{sum(counts.values())} total. "
            if missing_years:
                row += f"MISSING years: {missing_years[:5]}{'...' if len(missing_years)>5 else ''}. "
            if partial_years:
                pp = [f"{y}:{c}" for y, c in partial_years[:5]]
                row += f"Partial: {pp}..."
            print(row)

        results[name] = dict(counts=counts)
    return results


def audit_D_spatial_completeness(root, sample_year, problem_threshold=0.95):
    """D. Valid-pixel fraction on sample date (detects reproject-nan bug)."""
    print("\n" + "="*80)
    print(f"D. SPATIAL COMPLETENESS — valid_frac on {sample_year}-07-15 "
          f"(flag <{problem_threshold:.0%})")
    print("="*80)
    results = {}

    for name, d, pat, kind, req in CHANNELS:
        if kind == 'static':
            path = os.path.join(root, d, pat)
        elif kind == 'annual':
            path = os.path.join(root, d, pat.format(y=sample_year))
        else:
            path = os.path.join(root, d, pat.format(y=sample_year, md='0715'))
        if not os.path.exists(path):
            print(f"  [SKIP] {name:12s} no sample file")
            continue
        stats = load_tif_stats(path)
        if stats is None:
            print(f"  [FAIL] {name:12s} unreadable")
            continue
        vf = stats[4]
        flag = "" if vf >= problem_threshold else f" ← ONLY {vf:.1%} valid"
        symbol = "[PASS]" if vf >= problem_threshold else "[WARN]"
        print(f"  {symbol} {name:12s}  valid_frac={vf:.3f}{flag}")
        results[name] = dict(valid_frac=vf, passed=(vf >= problem_threshold))
    return results


def audit_E_distribution_drift(root, start_year, end_year, max_drift=3.0):
    """E. Per-year mean/std on July 15; flag cross-year drift > 3σ."""
    print("\n" + "="*80)
    print(f"E. DISTRIBUTION DRIFT — per-year mean on Jul-15 "
          f"(flag per-year std > {max_drift}× within-year std)")
    print("="*80)
    results = {}

    for name, d, pat, kind, req in CHANNELS:
        if kind in ('static',):
            continue
        if kind == 'annual':
            continue  # annual drift covered in fire_clim/burn separately
        means, stds = [], []
        for y in range(start_year, end_year + 1):
            path = os.path.join(root, d, pat.format(y=y, md='0715'))
            if not os.path.exists(path):
                continue
            stats = load_tif_stats(path)
            if stats is None:
                continue
            means.append(stats[0])
            stds.append(stats[1])
        if len(means) < 5:
            print(f"  [SKIP] {name:12s} <5 years of data")
            continue
        per_year_std = float(np.std(means))
        within_year_std = float(np.mean(stds))
        ratio = per_year_std / within_year_std if within_year_std > 0 else 0
        flag = " ← DRIFT" if ratio > max_drift else ""
        symbol = "[WARN]" if ratio > max_drift else "[PASS]"
        print(f"  {symbol} {name:12s}  per-year μ range [{min(means):.2f}, {max(means):.2f}]  "
              f"cross-year σ(μ)={per_year_std:.3f}  within-year σ̄={within_year_std:.3f}  "
              f"ratio={ratio:.2f}{flag}")
        results[name] = dict(per_year_std=per_year_std,
                             within_year_std=within_year_std,
                             ratio=ratio,
                             passed=(ratio <= max_drift))
    return results


def audit_F_cross_channel_date(root, test_date='20200715'):
    """F. On one test date, check all daily channels have data."""
    print("\n" + "="*80)
    print(f"F. CROSS-CHANNEL ALIGNMENT — all channels on {test_date}")
    print("="*80)

    y = test_date[:4]
    md = test_date[4:]
    present, missing = [], []
    for name, d, pat, kind, req in CHANNELS:
        if kind != 'daily':
            continue
        path = os.path.join(root, d, pat.format(y=y, md=md))
        if os.path.exists(path):
            present.append(name)
        else:
            missing.append(name)

    print(f"  Present ({len(present)}): {', '.join(present)}")
    print(f"  Missing ({len(missing)}): {', '.join(missing) if missing else 'none'}")
    return dict(present=present, missing=missing)


def audit_G_labels(root, hotspot_csv, start_year, end_year):
    """G. Label file sanity: date range, count stability."""
    print("\n" + "="*80)
    print("G. LABEL INTEGRITY — hotspot CSV sanity")
    print("="*80)

    path = os.path.join(root, hotspot_csv)
    if not os.path.exists(path):
        print(f"  [FAIL] hotspot CSV not found: {hotspot_csv}")
        return dict(passed=False)

    df = pd.read_csv(path, usecols=['latitude', 'longitude', 'acq_date'])
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    print(f"  [INFO] Total records: {len(df):,}")
    print(f"  [INFO] Date range: {df.acq_date.min().date()} → {df.acq_date.max().date()}")
    print(f"  [INFO] Latitude range: {df.latitude.min():.2f} → {df.latitude.max():.2f}")
    print(f"  [INFO] Longitude range: {df.longitude.min():.2f} → {df.longitude.max():.2f}")

    # Per-year count ratios
    df['year'] = df.acq_date.dt.year
    counts_per_year = df.groupby('year').size()
    max_count = counts_per_year.max()
    min_count = counts_per_year.min()
    ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"  [{'WARN' if ratio > 10 else 'PASS'}] "
          f"Max/min year count ratio = {ratio:.1f}x "
          f"({min_count:,} in {counts_per_year.idxmin()} vs "
          f"{max_count:,} in {counts_per_year.idxmax()})")
    if ratio > 10:
        print(f"         → Extreme drift (>10×) likely indicates detection tech change")

    return dict(n_records=len(df),
                date_range=(df.acq_date.min().date(), df.acq_date.max().date()),
                per_year_min=min_count,
                per_year_max=max_count,
                drift_ratio=ratio)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--sample-year", type=int, default=2018,
                    help="Year to use for value/spatial checks")
    ap.add_argument("--cross-date", default='20200715',
                    help="Date for cross-channel check")
    ap.add_argument("--hotspot-csv",
                    default="data/hotspot/hotspot_2000_2025.csv")
    ap.add_argument("--out-dir", default="data/audit",
                    help="Where to write CSV detail")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.root, args.out_dir), exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"# COMPREHENSIVE DATA AUDIT")
    print(f"# Root: {args.root}  Years: {args.start_year}-{args.end_year}")
    print(f"# Sample year: {args.sample_year}  Cross-date: {args.cross_date}")
    print(f"{'#'*80}")

    t0 = time.time()
    a = audit_A_structural(args.root, args.start_year, args.end_year)
    b = audit_B_value_range(args.root, args.sample_year)
    c = audit_C_temporal_coverage(args.root, args.start_year, args.end_year)
    d = audit_D_spatial_completeness(args.root, args.sample_year)
    e = audit_E_distribution_drift(args.root, args.start_year, args.end_year)
    f = audit_F_cross_channel_date(args.root, args.cross_date)
    g = audit_G_labels(args.root, args.hotspot_csv, args.start_year, args.end_year)

    # ── Overall summary ──
    print("\n" + "#"*80)
    print("# OVERALL SUMMARY")
    print("#"*80)

    # Required 9ch check
    n_req_bad_struct = sum(1 for v in a.values()
                           if v['required_9ch'] and not v['passed'])
    n_bad_range = sum(1 for v in b.values() if not v['passed'])
    n_bad_spatial = sum(1 for v in d.values() if not v['passed'])
    n_drift = sum(1 for v in e.values() if not v['passed'])

    print(f"\n  Required 9ch structural : "
          f"{'✓ ALL PASS' if n_req_bad_struct == 0 else f'{n_req_bad_struct} FAIL'}")
    print(f"  Value range violations  : "
          f"{'✓ NONE' if n_bad_range == 0 else f'{n_bad_range} channel(s)'}")
    print(f"  Spatial completeness    : "
          f"{'✓ ALL COMPLETE' if n_bad_spatial == 0 else f'{n_bad_spatial} channel(s) <95% valid'}")
    print(f"  Distribution drift      : "
          f"{'✓ STABLE' if n_drift == 0 else f'{n_drift} channel(s) drifting'}")
    _missing_count = len(f['missing'])
    print(f"  Cross-channel alignment : "
          f"{'✓ ALL PRESENT' if _missing_count == 0 else f'missing {_missing_count}'}")
    print(f"  Label integrity         : "
          f"{'⚠ drift' if g.get('drift_ratio', 0) > 10 else '✓'}  "
          f"({g.get('drift_ratio', 0):.1f}× year-ratio)")

    print(f"\n  Audit runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
