#!/usr/bin/env python3
"""
FWI Data Health Checker.

Scans a directory of FWI GeoTIFFs and checks for:
1. Consistent image dimensions across all files
2. Date continuity in filenames
3. Value anomalies (NaN, Inf, extreme values, constant images)
4. Global value range summary

Refactored from: test_scripts/check_data.py
  - Imports from src.config instead of hardcoded G: drive path
  - Uses src.utils.date_utils.parse_date_from_filename
  - Added --config argument

Usage:
  python -m src.data.validation.check_fwi_health
  python -m src.data.validation.check_fwi_health --config configs/paths_mac.yaml
  python -m src.data.validation.check_fwi_health --dir /path/to/fwi_data
"""

import os
import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

from src.config import load_config, get_path, add_config_argument
from src.utils.date_utils import parse_date_from_filename


def check_fwi_health(data_dir):
    """
    Check health of FWI GeoTIFF data in a directory.

    Scans all .tif files and reports on dimension consistency,
    date sequence continuity, and value anomalies.

    Args:
        data_dir: Path to directory containing FWI .tif files.
    """
    print("=" * 60)
    print(f"Starting data health check: {data_dir}")
    print("=" * 60)

    files = sorted(Path(data_dir).glob("*.tif"))
    if not files:
        print("Error: No .tif files found!")
        return

    # Tracking variables
    shapes = []
    dates = []
    issues = []
    stats_summary = []

    print(f"Found {len(files)} files, scanning...")

    for fpath in files:
        fname = fpath.name

        # 1. Parse date from filename
        curr_date = parse_date_from_filename(str(fpath))
        if curr_date is None:
            issues.append(f"[{fname}] Cannot parse date from filename")
            continue
        dates.append(curr_date)

        try:
            with Image.open(str(fpath)) as im:
                # 2. Check dimension consistency
                shapes.append(im.size)  # (width, height)

                arr = np.array(im).astype(np.float32)

                # 3. Check for missing values and anomalies
                n_nan = np.isnan(arr).sum()
                n_inf = np.isinf(arr).sum()
                v_min, v_max = np.nanmin(arr), np.nanmax(arr)
                v_mean = np.nanmean(arr)

                # 4. Check for all-zero or constant-value images
                v_std = np.nanstd(arr)

                # Issue detection
                file_issues = []
                if n_nan > 0:
                    file_issues.append(f"{n_nan} NaN values")
                if n_inf > 0:
                    file_issues.append(f"{n_inf} Inf values")
                if v_min < -100:
                    file_issues.append(f"Suspected NoData extreme low ({v_min})")
                if v_max > 500:
                    file_issues.append(f"Suspected anomalous extreme high ({v_max})")
                if v_std < 1e-6:
                    file_issues.append("Constant value across image (possibly dead frame)")

                if file_issues:
                    issues.append(f"[{fname}] " + ", ".join(file_issues))

                stats_summary.append({
                    'min': v_min, 'max': v_max, 'mean': v_mean
                })

        except Exception as e:
            issues.append(f"[{fname}] Cannot read file: {e}")

    # --- Summary report ---
    print("\n" + "=" * 60)
    print("DATA HEALTH SUMMARY")
    print("=" * 60)

    # 1. Dimension statistics
    shape_counts = Counter(shapes)
    if len(shape_counts) > 1:
        print(f"[FAIL] Multiple resolutions found: {dict(shape_counts)}")
    elif shapes:
        print(f"[OK]   Resolution consistent: {shapes[0]}")

    # 2. Temporal continuity
    if dates:
        dates.sort()
        delta = (dates[-1] - dates[0]).days + 1
        if len(dates) != delta:
            print(f"[FAIL] Time series not continuous: expected {delta} days, found {len(dates)} files")
            # Find missing dates (optional detail)
        else:
            print(f"[OK]   Time series complete ({dates[0].date()} to {dates[-1].date()})")

    # 3. Value distribution
    if stats_summary:
        all_mins = [s['min'] for s in stats_summary]
        all_maxs = [s['max'] for s in stats_summary]
        print(f"  Global value range: [{min(all_mins):.2f}, {max(all_maxs):.2f}]")
        print(f"  Global mean range:  [{min(s['mean'] for s in stats_summary):.2f}, "
              f"{max(s['mean'] for s in stats_summary):.2f}]")

    # 4. Issue list
    print("\n" + "=" * 60)
    if issues:
        print(f"Found {len(issues)} potential issue(s):")
        for issue in issues[:20]:  # Show at most 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("No anomalies detected. Data looks healthy.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Check FWI data health (dimensions, continuity, value anomalies)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_config_argument(parser)

    parser.add_argument(
        '--dir',
        default=None,
        help='Directory containing FWI .tif files (overrides config fwi_dir)'
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve directory: CLI overrides config
    if args.dir:
        data_dir = args.dir
    else:
        data_dir = get_path(cfg, 'fwi_dir')

    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    check_fwi_health(data_dir)


if __name__ == "__main__":
    main()
