#!/usr/bin/env python3
"""
Reproject WorldPop population density to the FWI grid.

Input:  {population_raw_dir}/can_pd_2020_1km.tif  (from download_population.py)
Output: {population_tif}  (EPSG:3978, 2709×2281, log1p(people/km²))

Usage:
    python -m src.data_ops.processing.process_population
    python -m src.data_ops.processing.process_population --config configs/paths_narval.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)


def main():
    parser = argparse.ArgumentParser(
        description="Reproject WorldPop population → FWI grid (EPSG:3978)"
    )
    add_config_argument(parser)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        out_path = Path(get_path(cfg, "population_tif"))
    except (KeyError, TypeError):
        out_path = Path("data/population_density.tif")

    raw_dir = out_path.parent / "population_raw"
    raw_path = raw_dir / "can_pd_2020_1km.tif"

    if not raw_path.exists():
        print(f"[ERROR] Raw file not found: {raw_path}", file=sys.stderr)
        print("  Run download_population.py first.", file=sys.stderr)
        sys.exit(1)

    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] Output exists: {out_path}")
        return

    print(f"Reprojecting {raw_path.name} → EPSG:3978 ({FWI_WIDTH}×{FWI_HEIGHT})")

    dst_transform = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    dst_crs = CRS.from_string(FWI_CRS)

    with rasterio.open(raw_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[~np.isfinite(data)] = 0.0
        data[data < 0] = 0.0

        dst_data = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float32)
        reproject(
            source=data, destination=dst_data,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan,
        )

    # log1p transform
    dst_data = np.where(np.isfinite(dst_data), dst_data, 0.0)
    dst_data = np.maximum(dst_data, 0.0)
    dst_data = np.log1p(dst_data).astype(np.float32)

    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": FWI_WIDTH, "height": FWI_HEIGHT, "count": 1,
        "crs": dst_crs, "transform": dst_transform,
        "nodata": np.nan, "compress": "lzw",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dst_data, 1)

    valid = np.isfinite(dst_data) & (dst_data > 0)
    print(f"[SAVED] {out_path}")
    print(f"  Valid pixels: {valid.sum():,}  Range: [{dst_data[valid].min():.2f}, {dst_data[valid].max():.2f}]")


if __name__ == "__main__":
    main()
