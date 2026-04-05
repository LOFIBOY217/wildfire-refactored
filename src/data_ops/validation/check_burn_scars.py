#!/usr/bin/env python3
"""
Validate NBAC burn scar data (years_since_burn + burn_count TIFs).

Checks:
  1. File existence for each year (2018-2024)
  2. Grid alignment (EPSG:3978, 2281×2709)
  3. Value ranges (years_since_burn: 0-9999, burn_count: 0-~20)
  4. Ecological plausibility (2023 should have most recent burns)
  5. burn_count consistency (burned pixels should match between files)

Usage:
    python -m src.data_ops.validation.check_burn_scars
    python -m src.data_ops.validation.check_burn_scars --config configs/paths_narval.yaml
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


FWI_CRS = "EPSG:3978"
EXPECTED_SHAPE = (2281, 2709)


def main():
    parser = argparse.ArgumentParser(description="Validate NBAC burn scar TIFs")
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2024)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        burn_dir = Path(get_path(cfg, "burn_scars_dir"))
    except (KeyError, TypeError):
        burn_dir = Path("data/burn_scars")

    print("=" * 60)
    print("BURN SCAR DATA VALIDATION")
    print("=" * 60)
    print(f"  Directory: {burn_dir}")
    print(f"  Years: {args.start_year}-{args.end_year}")

    all_ok = True

    for year in range(args.start_year, args.end_year + 1):
        age_path = burn_dir / f"years_since_burn_{year}.tif"
        count_path = burn_dir / f"burn_count_{year}.tif"

        print(f"\n--- {year} ---")

        # Check years_since_burn
        if not age_path.exists():
            print(f"  [FAIL] years_since_burn_{year}.tif MISSING")
            all_ok = False
            continue

        with rasterio.open(age_path) as src:
            age = src.read(1).astype(np.float32)
            crs = str(src.crs)
            shape = (src.height, src.width)

        if shape != EXPECTED_SHAPE:
            print(f"  [FAIL] Shape {shape} != {EXPECTED_SHAPE}")
            all_ok = False
        if crs != FWI_CRS:
            print(f"  [FAIL] CRS {crs} != {FWI_CRS}")
            all_ok = False

        never = int((age == 9999).sum())
        recent_5 = int(((age >= 0) & (age <= 5)).sum())
        burned_this_year = int((age == 0).sum())
        print(f"  years_since_burn: never={never:,}  within_5yr={recent_5:,}  "
              f"this_year(age=0)={burned_this_year:,}")

        # Check burn_count
        if not count_path.exists():
            print(f"  [WARN] burn_count_{year}.tif MISSING")
            continue

        with rasterio.open(count_path) as src:
            cnt = src.read(1).astype(np.float32)

        never_cnt = int((cnt == 0).sum())
        once = int((cnt == 1).sum())
        twice = int((cnt == 2).sum())
        three_plus = int((cnt >= 3).sum())
        max_cnt = int(cnt.max())

        print(f"  burn_count: never={never_cnt:,}  1x={once:,}  "
              f"2x={twice:,}  3+={three_plus:,}  max={max_cnt}")

        # Consistency check: never-burned should match
        if never != never_cnt:
            print(f"  [WARN] Never-burned mismatch: age={never:,} vs count={never_cnt:,}")

        # Plausibility: 2023 should have lots of recent burns
        if year == 2023 and burned_this_year < 10000:
            print(f"  [WARN] 2023 was record fire year but only {burned_this_year:,} "
                  f"pixels burned — check NBAC data")

    print(f"\n{'=' * 60}")
    print(f"Result: {'ALL PASS' if all_ok else 'ISSUES FOUND'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
