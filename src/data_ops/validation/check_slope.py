#!/usr/bin/env python3
"""
Validate SRTM slope and aspect data.

Checks:
  1. slope.tif and aspect.tif exist
  2. Grid alignment (EPSG:3978, 2281×2709)
  3. Value ranges (slope: 0-90°, aspect: -1 to 360°)
  4. Coverage (SRTM only covers ≤60°N, northern Canada will be NaN)
  5. Raw tile count if processed files missing

Usage:
    python -m src.data_ops.validation.check_slope
    python -m src.data_ops.validation.check_slope --config configs/paths_narval.yaml
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Validate SRTM slope/aspect TIFs")
    add_config_argument(parser)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        terrain_dir = Path(get_path(cfg, "terrain_dir"))
    except (KeyError, TypeError):
        terrain_dir = Path("data/terrain")
    raw_dir = terrain_dir.parent / "terrain_raw"

    print("=" * 60)
    print("SLOPE / ASPECT VALIDATION")
    print("=" * 60)

    slope_path = terrain_dir / "slope.tif"
    aspect_path = terrain_dir / "aspect.tif"
    all_ok = True

    if not slope_path.exists():
        hgt_count = len(list(raw_dir.glob("*.hgt"))) if raw_dir.exists() else 0
        print(f"  [FAIL] slope.tif not found: {slope_path}")
        print(f"  Raw .hgt tiles: {hgt_count}")
        if hgt_count > 0:
            print("  → Run process_srtm_slope.py on compute node")
        return 1

    for name, path, vmin, vmax in [
        ("slope", slope_path, 0.0, 90.0),
        ("aspect", aspect_path, -1.0, 360.0),
    ]:
        print(f"\n  --- {name} ---")
        if not path.exists():
            print(f"  [WARN] {name}.tif missing")
            continue

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            crs = str(src.crs)
            shape = (src.height, src.width)

        print(f"  Shape: {shape}  CRS: {crs}")
        if shape != EXPECTED_SHAPE:
            print(f"  [FAIL] Shape != {EXPECTED_SHAPE}")
            all_ok = False
        if crs != FWI_CRS:
            print(f"  [FAIL] CRS != {FWI_CRS}")
            all_ok = False

        finite = data[np.isfinite(data)]
        nan_pct = 100 * (data.size - len(finite)) / data.size
        print(f"  Finite: {len(finite):,}/{data.size:,}  NaN: {nan_pct:.1f}%")

        if len(finite) > 0:
            print(f"  Range: [{finite.min():.2f}, {finite.max():.2f}]  "
                  f"Mean: {finite.mean():.2f}")
            if finite.min() < vmin - 0.1 or finite.max() > vmax + 0.1:
                print(f"  [WARN] Outside expected [{vmin}, {vmax}]")
        else:
            print("  [FAIL] All NaN")
            all_ok = False

        # SRTM coverage check
        if name == "slope" and nan_pct > 60:
            print(f"  [INFO] {nan_pct:.0f}% NaN — expected for SRTM (≤60°N only)")

    print(f"\n{'=' * 60}")
    print(f"Result: {'PASS' if all_ok else 'ISSUES FOUND'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
