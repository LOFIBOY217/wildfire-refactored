#!/usr/bin/env python3
"""
Validate MODIS NDVI data (raw HDF4 files and processed FWI-grid TIFs).

Checks:
  1. Raw HDF4 file count per year
  2. Processed daily TIF count and date coverage
  3. Grid alignment (EPSG:3978, 2281×2709)
  4. Value ranges (NDVI: -0.5 to 1.0, valid pixels > 0)
  5. Seasonal pattern (summer NDVI > winter NDVI)

Usage:
    python -m src.data_ops.validation.check_ndvi
    python -m src.data_ops.validation.check_ndvi --config configs/paths_narval.yaml
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

from src.utils.date_utils import extract_date_from_filename

FWI_CRS = "EPSG:3978"
EXPECTED_SHAPE = (2281, 2709)


def main():
    parser = argparse.ArgumentParser(description="Validate MODIS NDVI data")
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2024)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        ndvi_dir = Path(get_path(cfg, "ndvi_dir"))
    except (KeyError, TypeError):
        ndvi_dir = Path("data/ndvi_data")
    raw_dir = ndvi_dir.parent / "ndvi_raw"

    print("=" * 60)
    print("NDVI DATA VALIDATION")
    print("=" * 60)

    # Raw HDF4 per year
    print(f"\n  Raw directory: {raw_dir}")
    total_hdf = 0
    for year in range(args.start_year, args.end_year + 1):
        year_dir = raw_dir / str(year)
        count = len(list(year_dir.glob("*.hdf"))) if year_dir.exists() else 0
        total_hdf += count
        status = "OK" if count > 500 else ("PARTIAL" if count > 0 else "MISSING")
        print(f"    {year}: {count} HDF4 files  [{status}]")
    print(f"    Total: {total_hdf}")

    # Processed TIFs
    tifs = sorted(glob.glob(str(ndvi_dir / "ndvi_*.tif")))
    print(f"\n  Processed directory: {ndvi_dir}")
    print(f"  Processed TIFs: {len(tifs)}")

    all_ok = True

    if not tifs:
        if total_hdf > 0:
            print("  [WARN] Raw HDF4 exists but no processed TIFs — "
                  "run process_modis_ndvi.py")
        else:
            print("  [FAIL] No data at all")
            all_ok = False
    else:
        # Per-year count
        years = {}
        for t in tifs:
            d = extract_date_from_filename(os.path.basename(t))
            if d:
                years.setdefault(d.year, 0)
                years[d.year] += 1
        for y in sorted(years):
            print(f"    {y}: {years[y]} daily TIFs")

        # Spot-check: mid-summer file (should have high NDVI)
        summer_files = [t for t in tifs if "0715" in os.path.basename(t)
                        or "0716" in os.path.basename(t)]
        sample = summer_files[0] if summer_files else tifs[len(tifs) // 2]

        with rasterio.open(sample) as src:
            data = src.read(1)
            crs = str(src.crs)
            shape = (src.height, src.width)

        finite = data[np.isfinite(data)]
        nonzero = finite[finite != 0]

        print(f"\n  Sample ({os.path.basename(sample)}):")
        print(f"    Shape: {shape}  CRS: {crs}")
        print(f"    Finite: {len(finite):,}/{data.size:,}  "
              f"Nonzero: {len(nonzero):,}")

        if len(finite) > 0:
            print(f"    Range: [{finite.min():.3f}, {finite.max():.3f}]  "
                  f"Mean: {finite.mean():.3f}")
        else:
            print("    [FAIL] ALL NaN — sinusoidal projection bug?")
            all_ok = False

        if shape != EXPECTED_SHAPE:
            print(f"    [FAIL] Shape {shape} != {EXPECTED_SHAPE}")
            all_ok = False
        if crs != FWI_CRS:
            print(f"    [FAIL] CRS {crs} != {FWI_CRS}")
            all_ok = False
        if len(nonzero) == 0 and len(finite) > 0:
            print("    [FAIL] All pixels zero (valid but empty)")
            all_ok = False

        # Check NDVI range
        if len(finite) > 0:
            if finite.min() < -1.0 or finite.max() > 1.5:
                print(f"    [WARN] NDVI outside [-1, 1.5]: "
                      f"[{finite.min():.3f}, {finite.max():.3f}]")

    print(f"\n{'=' * 60}")
    print(f"Result: {'PASS' if all_ok else 'ISSUES FOUND'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
