#!/usr/bin/env python3
"""
Reproject GLM raw 0.1° WGS84 lightning TIFs to the FWI grid (EPSG:3978).

Input:  {lightning_raw_dir}/glm_raw_YYYYMMDD.tif  (0.1° WGS84, from download_goes_glm.py)
Output: {lightning_dir}/lightning_YYYYMMDD.tif     (EPSG:3978, 2709×2281)

Usage:
    python -m src.data_ops.processing.resample_glm_to_fwi_grid
    python -m src.data_ops.processing.resample_glm_to_fwi_grid --config configs/paths_narval.yaml
"""

import argparse
import glob
import os
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

from src.utils.date_utils import extract_date_from_filename

FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)


def main():
    parser = argparse.ArgumentParser(
        description="Reproject GLM raw TIFs (0.1° WGS84) → FWI grid (EPSG:3978)"
    )
    add_config_argument(parser)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    lightning_dir = Path(get_path(cfg, "lightning_dir"))
    raw_dir = lightning_dir.parent / "lightning_raw"
    lightning_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(glob.glob(str(raw_dir / "glm_raw_*.tif")))
    print(f"Found {len(raw_files)} raw GLM TIFs in {raw_dir}")

    dst_transform = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    dst_crs = CRS.from_string(FWI_CRS)

    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": FWI_WIDTH, "height": FWI_HEIGHT, "count": 1,
        "crs": dst_crs, "transform": dst_transform,
        "nodata": 0.0, "compress": "lzw",
    }

    done = skipped = 0
    for raw_path in raw_files:
        d = extract_date_from_filename(os.path.basename(raw_path))
        if d is None:
            continue
        date_str = d.strftime("%Y%m%d")
        out_path = lightning_dir / f"lightning_{date_str}.tif"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with rasterio.open(raw_path) as src:
            src_data = src.read(1)
            src_transform = src.transform
            src_crs = src.crs

        dst_data = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(dst_data, 1)
        done += 1

    print(f"Done: {done} reprojected, {skipped} skipped")


if __name__ == "__main__":
    main()
