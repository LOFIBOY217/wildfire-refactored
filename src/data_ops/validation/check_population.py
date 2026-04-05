#!/usr/bin/env python3
"""
Validate WorldPop population density data.

Checks:
  1. File exists and non-empty
  2. Grid alignment (EPSG:3978, 2281×2709)
  3. Value range (log1p scale: 0 to ~10)
  4. Spatial plausibility (most pixels near zero, cities high)

Usage:
    python -m src.data_ops.validation.check_population
    python -m src.data_ops.validation.check_population --config configs/paths_narval.yaml
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
    parser = argparse.ArgumentParser(description="Validate population density TIF")
    add_config_argument(parser)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        pop_path = Path(get_path(cfg, "population_tif"))
    except (KeyError, TypeError):
        pop_path = Path("data/population_density.tif")

    print("=" * 60)
    print("POPULATION DENSITY VALIDATION")
    print("=" * 60)
    print(f"  Path: {pop_path}")

    all_ok = True

    if not pop_path.exists():
        print("  [FAIL] File not found")
        return 1

    print(f"  Size: {pop_path.stat().st_size / 1e6:.1f} MB")

    with rasterio.open(pop_path) as src:
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
    nonzero = finite[finite > 0]

    print(f"  Finite pixels: {len(finite):,}/{data.size:,}")
    print(f"  Nonzero pixels: {len(nonzero):,} ({100*len(nonzero)/data.size:.1f}%)")
    print(f"  Range: [{finite.min():.3f}, {finite.max():.3f}]")
    print(f"  Mean: {finite.mean():.4f}  Mean(nonzero): {nonzero.mean():.3f}")

    # Plausibility
    if finite.max() < 5.0:
        print("  [WARN] Max < 5.0 — expected ~9-10 for major cities (log1p scale)")
    if len(nonzero) < 100000:
        print("  [WARN] Too few populated pixels — check reprojection")
    if finite.min() < 0:
        print("  [FAIL] Negative values (log1p should be >= 0)")
        all_ok = False

    # Distribution
    pct_uninhabited = 100 * (len(finite) - len(nonzero)) / len(finite)
    print(f"  Uninhabited: {pct_uninhabited:.1f}% (expect >80% for Canada)")

    print(f"\n{'=' * 60}")
    print(f"Result: {'PASS' if all_ok else 'ISSUES FOUND'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
