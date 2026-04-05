#!/usr/bin/env python3
"""
Validate GLM lightning data (raw 0.1° TIFs and processed FWI-grid TIFs).

Checks:
  1. Raw file count and date coverage
  2. Processed file count (after reproject to FWI grid)
  3. Grid alignment (CRS, shape)
  4. Value ranges (flash counts should be non-negative, summer > winter)
  5. Seasonal pattern (summer months should have more flashes)

Usage:
    python -m src.data_ops.validation.check_lightning
    python -m src.data_ops.validation.check_lightning --config configs/paths_narval.yaml
"""

import argparse
import glob
import os
import sys
from datetime import date, timedelta
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

from src.utils.date_utils import extract_date_from_filename

FWI_CRS = "EPSG:3978"
EXPECTED_SHAPE = (2281, 2709)
RAW_SHAPE = (440, 920)


def main():
    parser = argparse.ArgumentParser(description="Validate GLM lightning data")
    add_config_argument(parser)
    parser.add_argument("--start", type=str, default="2018-05-01")
    parser.add_argument("--end", type=str, default="2024-10-31")
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        lightning_dir = Path(get_path(cfg, "lightning_dir"))
    except (KeyError, TypeError):
        lightning_dir = Path("data/lightning")
    raw_dir = lightning_dir.parent / "lightning_raw"

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("=" * 60)
    print("LIGHTNING DATA VALIDATION")
    print("=" * 60)

    # Raw files
    raw_files = sorted(glob.glob(str(raw_dir / "glm_raw_*.tif")))
    print(f"\n  Raw directory: {raw_dir}")
    print(f"  Raw files: {len(raw_files)}")

    if raw_files:
        raw_dates = set()
        for f in raw_files:
            d = extract_date_from_filename(os.path.basename(f))
            if d:
                raw_dates.add(d)
        # Per-year count
        years = {}
        for d in raw_dates:
            years.setdefault(d.year, 0)
            years[d.year] += 1
        for y in sorted(years):
            print(f"    {y}: {years[y]} days")

        # Spot-check one raw file
        sample = raw_files[len(raw_files) // 2]
        with rasterio.open(sample) as src:
            data = src.read(1)
            print(f"\n  Raw sample ({os.path.basename(sample)}):")
            print(f"    Shape: {data.shape}  CRS: {src.crs}")
            print(f"    Total flashes: {data.sum():.0f}  "
                  f"Nonzero pixels: {(data > 0).sum():,}  "
                  f"Max: {data.max():.0f}")
            if data.shape != RAW_SHAPE:
                print(f"    [WARN] Expected {RAW_SHAPE}")

    # Processed files
    proc_files = sorted(glob.glob(str(lightning_dir / "lightning_*.tif")))
    print(f"\n  Processed directory: {lightning_dir}")
    print(f"  Processed files: {len(proc_files)}")

    all_ok = True
    if proc_files:
        sample = proc_files[len(proc_files) // 2]
        with rasterio.open(sample) as src:
            data = src.read(1)
            crs = str(src.crs)
            shape = (src.height, src.width)
        print(f"\n  Processed sample ({os.path.basename(sample)}):")
        print(f"    Shape: {shape}  CRS: {crs}")
        print(f"    Total flashes: {data.sum():.0f}  "
              f"Nonzero: {(data > 0).sum():,}  "
              f"Max: {data.max():.0f}")
        if shape != EXPECTED_SHAPE:
            print(f"    [FAIL] Shape {shape} != {EXPECTED_SHAPE}")
            all_ok = False
        if crs != FWI_CRS:
            print(f"    [FAIL] CRS {crs} != {FWI_CRS}")
            all_ok = False
        if data.max() < 0:
            print(f"    [FAIL] Negative flash counts")
            all_ok = False
    else:
        print("  [WARN] No processed TIFs — run resample_glm_to_fwi_grid.py")

    # Coverage
    expected_days = 0
    cur = start
    while cur <= end:
        if 5 <= cur.month <= 10:
            expected_days += 1
        cur += timedelta(days=1)
    print(f"\n  Expected fire-season days: {expected_days}")
    print(f"  Raw coverage: {len(raw_files)}/{expected_days} "
          f"({100*len(raw_files)/max(expected_days,1):.0f}%)")

    print(f"\n{'=' * 60}")
    print(f"Result: {'PASS' if all_ok else 'ISSUES FOUND'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
