#!/usr/bin/env python3
"""
Process NBAC shapefiles → years-since-burn GeoTIFs on the FWI grid.

Input:  {burn_scars_raw_dir}/nbac_{YYYY}.zip  (from download_nbac_burn_scars.py)
Output: {burn_scars_dir}/years_since_burn_{YYYY}.tif  (EPSG:3978, 2709×2281, uint16)

Usage:
    python -m src.data_ops.processing.process_nbac_burn_scars
    python -m src.data_ops.processing.process_nbac_burn_scars --config configs/paths_narval.yaml

Prerequisites:
    pip install geopandas rasterio
"""

import argparse
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


FWI_CRS       = "EPSG:3978"
FWI_WIDTH     = 2709
FWI_HEIGHT    = 2281
FWI_BOUNDS    = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
FWI_TRANSFORM = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
NODATA_YEARS  = 9999


def _rasterize_burn_year(zip_path: Path, year: int):
    """Read shapefile from zip, rasterize to FWI grid. Returns (H,W) uint8 mask."""
    import geopandas as gpd

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        shp_files = list(Path(tmp).rglob("*.shp"))
        if not shp_files:
            print(f"    [{year}] No .shp found in zip")
            return None
        gdf = gpd.read_file(shp_files[0])

    if gdf.empty:
        return None

    gdf = gdf.to_crs(FWI_CRS)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return None

    mask = rasterize(
        shapes, out_shape=(FWI_HEIGHT, FWI_WIDTH),
        transform=FWI_TRANSFORM, fill=0, dtype="uint8",
        all_touched=True,
    )
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Process NBAC zips → years-since-burn TIFs on FWI grid"
    )
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    burn_dir = Path(get_path(cfg, "burn_scars_dir"))
    raw_dir = burn_dir.parent / "burn_scars_raw"
    burn_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Rasterize each year's shapefile
    burn_masks = {}  # year → (H, W) uint8
    cache_dir = burn_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Need burn history from 1985 onward for years-since-burn calculation
    download_start = 1985

    print(f"Processing NBAC burn scars {download_start}–{args.end_year}")

    for year in range(download_start, args.end_year + 1):
        cache_path = cache_dir / f"nbac_mask_{year}.npy"
        if cache_path.exists() and not args.overwrite:
            burn_masks[year] = np.load(cache_path)
            continue

        zip_path = raw_dir / f"nbac_{year}.zip"
        if not zip_path.exists():
            continue

        try:
            mask = _rasterize_burn_year(zip_path, year)
        except Exception as e:
            print(f"  [{year}] Skipping (error: {e})")
            continue

        if mask is None:
            continue

        burn_masks[year] = mask
        np.save(cache_path, mask)
        print(f"  [{year}] {mask.sum():,} burned pixels")

    if not burn_masks:
        print("[ERROR] No burn masks processed. Download zips first.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Compute years-since-burn for each output year
    profile = {
        "driver": "GTiff", "dtype": "uint16",
        "width": FWI_WIDTH, "height": FWI_HEIGHT, "count": 1,
        "crs": FWI_CRS, "transform": FWI_TRANSFORM,
        "nodata": NODATA_YEARS, "compress": "lzw",
    }

    written = 0
    for target_year in range(args.start_year, args.end_year + 1):
        out_path = burn_dir / f"years_since_burn_{target_year}.tif"
        if out_path.exists() and not args.overwrite:
            continue

        years_since = np.full((FWI_HEIGHT, FWI_WIDTH), NODATA_YEARS, dtype=np.uint16)
        for burn_year in sorted(burn_masks.keys(), reverse=True):
            if burn_year > target_year:
                continue
            mask = burn_masks[burn_year]
            burned = mask > 0
            years_since[burned] = np.minimum(
                years_since[burned], target_year - burn_year
            )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(years_since, 1)
        written += 1

    print(f"\n[COMPLETE] {written} years-since-burn TIFs written to {burn_dir}")


if __name__ == "__main__":
    main()
