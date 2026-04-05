#!/usr/bin/env python3
"""
Process raw CDEM tiles → slope_cdem.tif and aspect_cdem.tif on the FWI grid.

Strategy: same tile-by-tile reproject as process_srtm_slope.py.
Each CDEM tile is reprojected to EPSG:3978 FWI grid, accumulated, then
slope/aspect computed from the merged DEM.

CDEM covers ALL of Canada (including >60°N), unlike SRTM.

Input:  {cdem_raw_dir}/*.tif  (from download_cdem.py)
Output: {terrain_dir}/slope_cdem.tif   (EPSG:3978, 2709×2281, degrees [0,90])
        {terrain_dir}/aspect_cdem.tif  (EPSG:3978, 2709×2281, degrees [0,360))
        {terrain_dir}/dem_cdem.tif     (EPSG:3978, 2709×2281, elevation in metres)

After cross-validation with SRTM, slope_cdem.tif replaces slope.tif for training.

Usage:
    python -m src.data_ops.processing.process_cdem_slope
    python -m src.data_ops.processing.process_cdem_slope --config configs/paths_narval.yaml
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


def _compute_slope_aspect(dem, cell_size_m):
    """Horn's (1981) 3x3 finite-difference method for slope and aspect."""
    dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * cell_size_m)
    dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * cell_size_m)
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
        description="Process CDEM tiles → slope_cdem.tif + aspect_cdem.tif"
    )
    add_config_argument(parser)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    terrain_dir = Path(get_path(cfg, "terrain_dir"))
    raw_dir = terrain_dir.parent / "cdem_raw"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    slope_path = terrain_dir / "slope_cdem.tif"
    aspect_path = terrain_dir / "aspect_cdem.tif"
    dem_path = terrain_dir / "dem_cdem.tif"

    if slope_path.exists() and aspect_path.exists() and not args.overwrite:
        print(f"[SKIP] slope_cdem.tif and aspect_cdem.tif already exist")
        return

    tif_files = sorted(raw_dir.glob("*.tif"))
    if not tif_files:
        print(f"[ERROR] No .tif files in {raw_dir}", file=sys.stderr)
        print("  Run download_cdem.py first", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(tif_files)} CDEM tiles → FWI grid DEM (tile-by-tile)")

    dst_tf = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    dst_crs = CRS.from_string(FWI_CRS)

    # Accumulate DEM on FWI grid: sum + count for nanmean
    dem_sum = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float64)
    dem_count = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float64)

    for i, tif_path in enumerate(tif_files):
        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                # CDEM uses -32767 or similar for ocean/nodata
                data[data <= -1000] = np.nan

                tile_fwi = np.full((FWI_HEIGHT, FWI_WIDTH), np.nan, dtype=np.float32)
                reproject(
                    source=data, destination=tile_fwi,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_tf, dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan, dst_nodata=np.nan,
                )

                valid = np.isfinite(tile_fwi)
                dem_sum[valid] += tile_fwi[valid]
                dem_count[valid] += 1
        except Exception as e:
            print(f"  [{i}] {tif_path.name}: error {e}")
            continue

        if (i + 1) % 100 == 0 or i == len(tif_files) - 1:
            coverage = (dem_count > 0).sum() / dem_count.size * 100
            print(f"  [{i+1}/{len(tif_files)}] coverage: {coverage:.1f}%")

    # Mean DEM
    dem_fwi = np.where(dem_count > 0, dem_sum / dem_count, np.nan).astype(np.float32)
    coverage_pct = (dem_count > 0).sum() / dem_count.size * 100
    print(f"  Final DEM coverage: {coverage_pct:.1f}%")

    # Compute slope and aspect
    cell_size_m = abs(FWI_BOUNDS[2] - FWI_BOUNDS[0]) / FWI_WIDTH
    print(f"  Computing slope & aspect (cell_size ≈ {cell_size_m:.0f} m)...")
    slope, aspect = _compute_slope_aspect(dem_fwi, cell_size_m)

    # Write output
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": FWI_WIDTH, "height": FWI_HEIGHT, "count": 1,
        "crs": dst_crs, "transform": dst_tf,
        "nodata": np.nan, "compress": "lzw",
    }
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(dem_fwi, 1)
    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(slope, 1)
    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(aspect, 1)

    nan_pct = 100 * np.isnan(slope).sum() / slope.size
    print(f"  Saved: {dem_path}")
    print(f"  Saved: {slope_path} (mean={np.nanmean(slope):.1f}° NaN={nan_pct:.0f}%)")
    print(f"  Saved: {aspect_path}")

    # Also copy to slope.tif as the primary training file
    slope_primary = terrain_dir / "slope.tif"
    aspect_primary = terrain_dir / "aspect.tif"
    import shutil
    shutil.copy2(slope_path, slope_primary)
    shutil.copy2(aspect_path, aspect_primary)
    print(f"  Copied → {slope_primary} (primary training file)")

    print("[COMPLETE]")


if __name__ == "__main__":
    main()
