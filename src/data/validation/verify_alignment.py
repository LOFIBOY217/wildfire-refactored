#!/usr/bin/env python3
"""
Verify that FWI and ECMWF data are perfectly aligned.

This script checks:
1. Same number of files for each date
2. Matching date ranges
3. Identical spatial grid (CRS, transform, shape)
4. No missing dates in the sequence

Refactored from: data/verify_data_alignment.py
  - Uses src.utils.date_utils.extract_date_from_filename
  - Uses src.utils.raster_io.get_raster_info
  - Loads paths from YAML config via src.config

Usage:
  python -m src.data.validation.verify_alignment
  python -m src.data.validation.verify_alignment --config configs/paths_mac.yaml
  python -m src.data.validation.verify_alignment --fwi-dir /path/to/fwi --ecmwf-dir /path/to/ecmwf
"""

import argparse
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

from src.config import load_config, get_path, add_config_argument
from src.utils.date_utils import extract_date_from_filename
from src.utils.raster_io import get_raster_info


def check_spatial_alignment(fwi_file, ecmwf_file):
    """
    Check if two rasters are spatially aligned.

    Returns:
        (bool, str): (is_aligned, error_message)
    """
    fwi_info = get_raster_info(fwi_file)
    ecmwf_info = get_raster_info(ecmwf_file)

    errors = []

    # Check CRS
    if fwi_info['crs'] != ecmwf_info['crs']:
        errors.append(f"CRS mismatch: FWI={fwi_info['crs']}, ECMWF={ecmwf_info['crs']}")

    # Check shape
    fwi_shape = (fwi_info['height'], fwi_info['width'])
    ecmwf_shape = (ecmwf_info['height'], ecmwf_info['width'])
    if fwi_shape != ecmwf_shape:
        errors.append(f"Shape mismatch: FWI={fwi_shape}, ECMWF={ecmwf_shape}")

    # Check transform (with small tolerance for floating point)
    fwi_t = fwi_info['transform']
    ecmwf_t = ecmwf_info['transform']

    if not all(abs(a - b) < 1e-6 for a, b in zip(fwi_t, ecmwf_t)):
        errors.append(f"Transform mismatch:\n  FWI:   {fwi_t}\n  ECMWF: {ecmwf_t}")

    # Check bounds (with tolerance)
    fwi_b = fwi_info['bounds']
    ecmwf_b = ecmwf_info['bounds']

    if not all(abs(a - b) < 10 for a, b in zip(fwi_b, ecmwf_b)):  # 10m tolerance
        errors.append(f"Bounds mismatch:\n  FWI:   {fwi_b}\n  ECMWF: {ecmwf_b}")

    if errors:
        return False, "\n  ".join(errors)
    else:
        return True, "OK"


def main():
    parser = argparse.ArgumentParser(
        description='Verify FWI and ECMWF data alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_config_argument(parser)

    parser.add_argument(
        '--fwi-dir',
        default=None,
        help='Directory containing FWI rasters (overrides config)'
    )

    parser.add_argument(
        '--ecmwf-dir',
        default=None,
        help='Directory containing ECMWF rasters aligned to FWI grid (overrides config)'
    )

    parser.add_argument(
        '--ecmwf-vars',
        default=None,
        help='ECMWF variables to check (comma-separated, overrides config)'
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve directories: CLI overrides > config
    if args.fwi_dir:
        fwi_dir = Path(args.fwi_dir)
    else:
        fwi_dir = Path(get_path(cfg, 'fwi_dir'))

    if args.ecmwf_dir:
        ecmwf_dir = Path(args.ecmwf_dir)
    else:
        ecmwf_dir = Path(get_path(cfg, 'ecmwf_dir'))

    if args.ecmwf_vars:
        ecmwf_vars = args.ecmwf_vars.split(',')
    else:
        ecmwf_vars = cfg.get('data', {}).get('ecmwf_vars', ['2t', '2d', 'tcw', 'sm20', 'st20'])

    # Verify directories exist
    if not fwi_dir.exists():
        print(f"Error: FWI directory not found: {fwi_dir}")
        sys.exit(1)

    if not ecmwf_dir.exists():
        print(f"Error: ECMWF directory not found: {ecmwf_dir}")
        sys.exit(1)

    print("=" * 70)
    print("DATA ALIGNMENT VERIFICATION")
    print("=" * 70)
    print(f"FWI directory:   {fwi_dir}")
    print(f"ECMWF directory: {ecmwf_dir}")
    print(f"ECMWF variables: {', '.join(ecmwf_vars)}")
    print("=" * 70)

    # ============================================
    # STEP 1: Inventory files by date
    # ============================================
    print("\n[STEP 1] Inventorying files...")

    fwi_files = sorted(fwi_dir.glob("*.tif"))
    fwi_by_date = {}

    for f in fwi_files:
        date = extract_date_from_filename(f.name)
        if date:
            fwi_by_date[date] = f

    print(f"  Found {len(fwi_by_date)} FWI files")

    ecmwf_by_date = defaultdict(dict)

    for var in ecmwf_vars:
        var_files = sorted(ecmwf_dir.glob(f"{var}_*.tif"))
        for f in var_files:
            date = extract_date_from_filename(f.name)
            if date:
                ecmwf_by_date[date][var] = f

    ecmwf_dates = set(ecmwf_by_date.keys())
    print(f"  Found ECMWF files for {len(ecmwf_dates)} dates")

    # ============================================
    # STEP 2: Check date coverage
    # ============================================
    print("\n[STEP 2] Checking date coverage...")

    fwi_dates = set(fwi_by_date.keys())

    if not fwi_dates:
        print("  ERROR: No FWI files found")
        sys.exit(1)

    if not ecmwf_dates:
        print("  ERROR: No ECMWF files found")
        sys.exit(1)

    # Check overlap
    common_dates = fwi_dates & ecmwf_dates
    fwi_only = fwi_dates - ecmwf_dates
    ecmwf_only = ecmwf_dates - fwi_dates

    print(f"\n  Date coverage:")
    print(f"    FWI dates:     {min(fwi_dates)} to {max(fwi_dates)} ({len(fwi_dates)} days)")
    print(f"    ECMWF dates:   {min(ecmwf_dates)} to {max(ecmwf_dates)} ({len(ecmwf_dates)} days)")
    print(f"    Common dates:  {len(common_dates)} days")

    if fwi_only:
        print(f"\n  WARNING: {len(fwi_only)} dates have FWI but no ECMWF:")
        for date in sorted(fwi_only)[:5]:
            print(f"      {date}")
        if len(fwi_only) > 5:
            print(f"      ... and {len(fwi_only) - 5} more")

    if ecmwf_only:
        print(f"\n  WARNING: {len(ecmwf_only)} dates have ECMWF but no FWI:")
        for date in sorted(ecmwf_only)[:5]:
            print(f"      {date}")
        if len(ecmwf_only) > 5:
            print(f"      ... and {len(ecmwf_only) - 5} more")

    if not common_dates:
        print("\n  CRITICAL: No overlapping dates!")
        sys.exit(1)

    # ============================================
    # STEP 3: Check for missing dates in sequence
    # ============================================
    print("\n[STEP 3] Checking for gaps in date sequence...")

    sorted_common = sorted(common_dates)
    start_date = sorted_common[0]
    end_date = sorted_common[-1]

    expected_dates = set()
    current = start_date
    while current <= end_date:
        expected_dates.add(current)
        current += timedelta(days=1)

    missing_dates = expected_dates - common_dates

    if missing_dates:
        print(f"  WARNING: Found {len(missing_dates)} missing dates in sequence:")
        for date in sorted(missing_dates)[:10]:
            print(f"      {date}")
        if len(missing_dates) > 10:
            print(f"      ... and {len(missing_dates) - 10} more")
    else:
        print(f"  OK: Complete date sequence from {start_date} to {end_date}")

    # ============================================
    # STEP 4: Check variable completeness
    # ============================================
    print("\n[STEP 4] Checking ECMWF variable completeness...")

    incomplete_dates = []
    for date in sorted_common:
        missing_vars = [v for v in ecmwf_vars if v not in ecmwf_by_date[date]]
        if missing_vars:
            incomplete_dates.append((date, missing_vars))

    if incomplete_dates:
        print(f"  WARNING: {len(incomplete_dates)} dates missing some variables:")
        for date, vars in incomplete_dates[:5]:
            print(f"      {date}: missing {', '.join(vars)}")
        if len(incomplete_dates) > 5:
            print(f"      ... and {len(incomplete_dates) - 5} more")
    else:
        print(f"  OK: All {len(sorted_common)} dates have all {len(ecmwf_vars)} variables")

    # ============================================
    # STEP 5: Check spatial alignment (sample)
    # ============================================
    print("\n[STEP 5] Checking spatial alignment (sampling)...")

    # Sample up to 5 dates
    sample_dates = sorted_common[:min(5, len(sorted_common))]
    alignment_errors = []

    for date in sample_dates:
        fwi_file = fwi_by_date[date]

        print(f"\n  Checking {date}:")
        print(f"    FWI: {fwi_file.name}")

        for var in ecmwf_vars:
            if var in ecmwf_by_date[date]:
                ecmwf_file = ecmwf_by_date[date][var]
                print(f"    {var:5s}: {ecmwf_file.name} ", end='')

                is_aligned, msg = check_spatial_alignment(fwi_file, ecmwf_file)

                if is_aligned:
                    print("[OK]")
                else:
                    print("[FAIL]")
                    print(f"      ERROR: {msg}")
                    alignment_errors.append((date, var, msg))

    # ============================================
    # STEP 6: Summary
    # ============================================
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    issues = []

    if fwi_only:
        issues.append(f"WARNING: {len(fwi_only)} dates with FWI only")

    if ecmwf_only:
        issues.append(f"WARNING: {len(ecmwf_only)} dates with ECMWF only")

    if missing_dates:
        issues.append(f"WARNING: {len(missing_dates)} missing dates in sequence")

    if incomplete_dates:
        issues.append(f"WARNING: {len(incomplete_dates)} dates with incomplete variables")

    if alignment_errors:
        issues.append(f"FAIL: {len(alignment_errors)} spatial alignment errors")

    if not issues:
        print("ALL CHECKS PASSED")
        print(f"\n  * {len(common_dates)} dates with complete data")
        print(f"  * {len(ecmwf_vars)} ECMWF variables per date")
        print(f"  * Perfect spatial alignment verified")
        print(f"\n  Ready for logistic baseline training!")
        sys.exit(0)
    else:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")

        if alignment_errors:
            print("\n  Critical alignment errors must be fixed:")
            for date, var, msg in alignment_errors[:3]:
                print(f"    {date} {var}: {msg}")

        print("\n  Please resolve these issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
