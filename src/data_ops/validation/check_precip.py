#!/usr/bin/env python3
"""
Validate ERA5 total precipitation (tp) GRIB downloads.

Checks:
  1. GRIB file count and date coverage
  2. File sizes
  3. Spot-check readability and value range

Usage:
    python -m src.data_ops.validation.check_precip
    python -m src.data_ops.validation.check_precip --config configs/paths_narval.yaml
"""

import argparse
import glob
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


DATE_RE = re.compile(r'era5_tp_(\d{4})_(\d{2})_(\d{2})\.grib$')


def main():
    parser = argparse.ArgumentParser(description="Validate ERA5 precipitation GRIB files")
    add_config_argument(parser)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2025-10-31")
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        precip_dir = Path(get_path(cfg, "precip_dir"))
    except (KeyError, TypeError):
        precip_dir = Path("data/era5_precip")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("=" * 60)
    print("PRECIPITATION (ERA5 tp) VALIDATION")
    print("=" * 60)
    print(f"  Directory: {precip_dir}")

    gribs = sorted(glob.glob(str(precip_dir / "era5_tp_*.grib")))
    print(f"  GRIB files: {len(gribs)}")

    if not gribs:
        print("  [FAIL] No GRIB files found")
        return 1

    # Parse dates
    dates_found = set()
    sizes = []
    for g in gribs:
        m = DATE_RE.search(os.path.basename(g))
        if m:
            dates_found.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        sizes.append(os.path.getsize(g))

    expected = set()
    cur = start
    while cur <= end:
        expected.add(cur)
        cur += timedelta(days=1)

    found_in_range = dates_found & expected
    missing = sorted(expected - dates_found)
    coverage = len(found_in_range) / max(len(expected), 1)

    print(f"  Date range: {min(dates_found)} → {max(dates_found)}")
    print(f"  Coverage: {coverage:.1%} ({len(found_in_range)}/{len(expected)})")
    print(f"  File sizes: min={min(sizes)/1e6:.1f}MB  "
          f"max={max(sizes)/1e6:.1f}MB  mean={np.mean(sizes)/1e6:.1f}MB")

    if missing and len(missing) <= 10:
        print(f"  Missing: {missing}")
    elif missing:
        print(f"  Missing: {len(missing)} dates")

    # Spot-check
    sample = gribs[len(gribs) // 2]
    try:
        import rasterio
        with rasterio.open(sample) as src:
            data = src.read(1)
            print(f"\n  Sample ({os.path.basename(sample)}):")
            print(f"    Shape: {data.shape}  Bands: {src.count}")
            print(f"    Range: [{data.min():.6f}, {data.max():.6f}]  "
                  f"Mean: {data.mean():.6f}")
            if data.min() < -0.001:
                print("    [WARN] Negative precipitation values")
    except Exception as e:
        print(f"  Sample read failed: {e}")
        print("  (GRIB reading may need eccodes)")

    all_ok = coverage > 0.5
    print(f"\n{'=' * 60}")
    print(f"Result: {'PASS' if all_ok else 'INCOMPLETE (download in progress)'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
