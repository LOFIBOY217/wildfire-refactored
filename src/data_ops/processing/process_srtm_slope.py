#!/usr/bin/env python3
"""
Process raw SRTM .hgt tiles → slope.tif and aspect.tif on the FWI grid.

Input:  {terrain_raw_dir}/*.hgt  (from download_srtm_slope.py)
Output: {terrain_dir}/slope.tif   (EPSG:3978, 2709×2281, degrees [0,90])
        {terrain_dir}/aspect.tif  (EPSG:3978, 2709×2281, degrees [0,360))

Processing:
  1. Merge .hgt tiles in batches (avoid OOM)
  2. Reproject DEM mosaic to FWI grid
  3. Compute slope and aspect using Horn's (1981) method

Usage:
    python -m src.data_ops.processing.process_srtm_slope
    python -m src.data_ops.processing.process_srtm_slope --config configs/paths_narval.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
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


def _compute_slope_aspect(dem, cell_size_m):
    """Horn's (1981) 3x3 finite-difference method for slope and aspect."""
    dzdx = (
        (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1))
    ) / (2 * cell_size_m)
    dzdy = (
        (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0))
    ) / (2 * cell_size_m)

    rise = np.sqrt(dzdx**2 + dzdy**2)
    slope_deg = np.degrees(np.arctan(rise))

    aspect_rad = np.arctan2(-dzdy, dzdx)
    aspect_deg = 90.0 - np.degrees(aspect_rad)
    aspect_deg[aspect_deg < 0] += 360.0
    aspect_deg[aspect_deg >= 360.0] -= 360.0

    nan_mask = ~np.isfinite(dem)
    flat_mask = rise < 1e-8
    slope_deg[nan_mask] = np.nan
    aspect_deg[nan_mask] = np.nan
    aspect_deg[flat_mask & ~nan_mask] = -1.0

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Process SRTM .hgt tiles → slope.tif + aspect.tif on FWI grid"
    )
    add_config_argument(parser)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge_batch", type=int, default=100,
                        help="Tiles per merge batch (default: 100)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    terrain_dir = Path(get_path(cfg, "terrain_dir"))
    raw_dir = terrain_dir.parent / "terrain_raw"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    slope_path = terrain_dir / "slope.tif"
    aspect_path = terrain_dir / "aspect.tif"

    if slope_path.exists() and aspect_path.exists() and not args.overwrite:
        print(f"[SKIP] slope.tif and aspect.tif already exist in {terrain_dir}")
        return

    hgt_files = sorted(raw_dir.glob("*.hgt"))
    if not hgt_files:
        print(f"[ERROR] No .hgt files in {raw_dir}. Run download_srtm_slope.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(hgt_files)} SRTM tiles → slope + aspect")

    # Step 1: Merge tiles in batches
    print(f"  Merging tiles (batch={args.merge_batch})…")
    batch = args.merge_batch
    mosaic_arr = None
    mosaic_tf = None
    mosaic_crs = None

    for bi in range(0, len(hgt_files), batch):
        chunk = hgt_files[bi:bi + batch]
        srcs = [rasterio.open(f) for f in chunk]
        chunk_data, chunk_tf = merge(srcs)
        if mosaic_crs is None:
            mosaic_crs = srcs[0].crs
        for s in srcs:
            s.close()

        chunk_arr = chunk_data[0].astype(np.float32)
        chunk_arr[chunk_arr <= -32000] = np.nan

        if mosaic_arr is None:
            mosaic_arr = chunk_arr
            mosaic_tf = chunk_tf
        else:
            # For non-overlapping tiles, merge via temp files
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif") as t1, \
                 tempfile.NamedTemporaryFile(suffix=".tif") as t2:
                for path, arr, tf in [(t1.name, mosaic_arr, mosaic_tf),
                                      (t2.name, chunk_arr, chunk_tf)]:
                    with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                                       width=arr.shape[1], height=arr.shape[0],
                                       count=1, crs=mosaic_crs, transform=tf,
                                       nodata=np.nan) as dst:
                        dst.write(arr, 1)
                s1, s2 = rasterio.open(t1.name), rasterio.open(t2.name)
                combined, combined_tf = merge([s1, s2])
                s1.close()
                s2.close()
                mosaic_arr = combined[0].astype(np.float32)
                mosaic_tf = combined_tf

        print(f"    batch {bi//batch+1}: mosaic {mosaic_arr.shape}")
        del chunk_data, chunk_arr

    # Step 2: Reproject to FWI grid
    print("  Reprojecting to FWI grid…")
    dst_tf = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    dst_crs = CRS.from_string(FWI_CRS)
    dem_fwi = np.full((FWI_HEIGHT, FWI_WIDTH), np.nan, dtype=np.float32)
    reproject(mosaic_arr, dem_fwi,
              src_transform=mosaic_tf, src_crs=mosaic_crs,
              dst_transform=dst_tf, dst_crs=dst_crs,
              resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    del mosaic_arr

    # Step 3: Compute slope and aspect
    cell_size_m = abs(FWI_BOUNDS[2] - FWI_BOUNDS[0]) / FWI_WIDTH
    print(f"  Computing slope & aspect (cell_size ≈ {cell_size_m:.0f} m)…")
    slope, aspect = _compute_slope_aspect(dem_fwi, cell_size_m)

    # Write output
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": FWI_WIDTH, "height": FWI_HEIGHT, "count": 1,
        "crs": dst_crs, "transform": dst_tf,
        "nodata": np.nan, "compress": "lzw",
    }
    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(slope, 1)
    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(aspect, 1)

    print(f"  Saved: {slope_path} (mean={np.nanmean(slope):.1f}° max={np.nanmax(slope):.1f}°)")
    print(f"  Saved: {aspect_path}")
    print("[COMPLETE]")


if __name__ == "__main__":
    main()
