#!/usr/bin/env python3
"""
Download SRTM 1 Arc-Second DEM tiles for Canada, then compute slope and aspect.

The FWI system assumes flat terrain. Slope strongly affects fire spread rate:
  - Fire spreads faster uphill (roughly doubles per 10° of slope)
  - Aspect (N/S/E/W facing) affects fuel dryness via solar exposure

Output (static, only needs to run once):
    {terrain_dir}/slope.tif   — degrees [0, 90]
    {terrain_dir}/aspect.tif  — degrees [0, 360), 0=North, clockwise

Data source: SRTM GL1 (1 arc-second, ~30 m), NASA EarthData — free, requires account.
    https://lpdaac.usgs.gov/products/srtmgl1v003/

Prerequisites:
    pip install earthaccess
    # Log in once: earthaccess.login(strategy="interactive")
    # Or set env vars: EARTHDATA_USERNAME, EARTHDATA_PASSWORD

Usage:
    python -m src.data_ops.download.download_srtm_slope
    python -m src.data_ops.download.download_srtm_slope --config configs/paths_windows.yaml
    python -m src.data_ops.download.download_srtm_slope --overwrite
"""

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# FWI grid reference (EPSG:3978, 2709×2281)
# ------------------------------------------------------------------ #

FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)  # (left, bottom, right, top)

# Canada bounding box in WGS84 (for tile selection)
CANADA_BBOX = {"N": 84, "S": 41, "W": -142, "E": -52}


# ------------------------------------------------------------------ #
# Slope / aspect computation (pure numpy, no external DEM tool needed)
# ------------------------------------------------------------------ #

def _compute_slope_aspect(dem: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees, 0=N, CW) from a DEM array.

    Uses the standard 3×3 Horn (1981) finite-difference method (same as ArcGIS/GDAL).

    Args:
        dem:        2-D float32 array of elevation in metres. NaN for nodata.
        cell_size_m: approximate horizontal resolution in metres (used for both x and y).

    Returns:
        slope:  same shape as dem, degrees [0, 90]. NaN where dem is NaN.
        aspect: same shape as dem, degrees [0, 360). NaN where dem is NaN.
    """
    # Pad with edge values to handle borders
    pad = np.pad(dem, 1, mode="edge")

    # Horn's method neighbours
    # a b c
    # d e f
    # g h i
    a = pad[:-2, :-2]; b = pad[:-2, 1:-1]; c = pad[:-2, 2:]
    d = pad[1:-1, :-2];                    f = pad[1:-1, 2:]
    g = pad[2:,  :-2]; h = pad[2:,  1:-1]; i = pad[2:,  2:]

    dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8.0 * cell_size_m)
    dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8.0 * cell_size_m)

    rise = np.sqrt(dzdx**2 + dzdy**2)
    slope_rad = np.arctan(rise)
    slope_deg = np.degrees(slope_rad)

    # Aspect: 0 = North, clockwise
    aspect_rad = np.arctan2(-dzdy, dzdx)           # standard math angle
    aspect_deg = 90.0 - np.degrees(aspect_rad)     # convert to compass bearing
    aspect_deg[aspect_deg < 0] += 360.0
    aspect_deg[aspect_deg >= 360.0] -= 360.0

    # Propagate NaN from DEM
    nan_mask   = ~np.isfinite(dem)
    flat_mask  = (rise == 0)
    slope_deg[nan_mask] = np.nan
    aspect_deg[nan_mask] = np.nan
    aspect_deg[flat_mask & ~nan_mask] = -1.0   # flat convention: -1

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


# ------------------------------------------------------------------ #
# SRTM tile download via earthaccess
# ------------------------------------------------------------------ #

def _download_srtm_tiles(tmp_dir: Path) -> list[Path]:
    """
    Search and download all SRTM GL1 tiles covering Canada using earthaccess.
    Returns list of local .hgt or .tif file paths.
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError(
            "earthaccess is required: pip install earthaccess\n"
            "Then authenticate once: python -c \"import earthaccess; earthaccess.login()\""
        )

    print("  Authenticating with NASA Earthdata…")
    earthaccess.login(strategy="environment")   # uses EARTHDATA_USERNAME / EARTHDATA_PASSWORD

    print("  Searching SRTMGL1v003 tiles for Canada bounding box…")
    results = earthaccess.search_data(
        short_name  = "SRTMGL1",
        version     = "003",
        bounding_box = (
            CANADA_BBOX["W"], CANADA_BBOX["S"],
            CANADA_BBOX["E"], CANADA_BBOX["N"],
        ),
    )
    print(f"  Found {len(results)} tiles")

    if not results:
        raise RuntimeError("No SRTM tiles found for Canada bounding box.")

    print(f"  Downloading {len(results)} tiles to {tmp_dir} …")
    downloaded = earthaccess.download(results, local_path=str(tmp_dir))

    # Unzip .hgt.zip files if present
    hgt_files = []
    for f in downloaded:
        p = Path(f)
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as zf:
                zf.extractall(tmp_dir)
            extracted = list(tmp_dir.glob("*.hgt")) + list(tmp_dir.glob("*.tif"))
            hgt_files.extend(extracted)
        elif p.suffix in (".hgt", ".tif"):
            hgt_files.append(p)

    print(f"  Extracted {len(hgt_files)} DEM files")
    return hgt_files


# ------------------------------------------------------------------ #
# Main processing
# ------------------------------------------------------------------ #

def _fwi_transform():
    """Return the affine transform for the FWI grid."""
    left, bottom, right, top = FWI_BOUNDS
    from rasterio.transform import from_bounds as _fb
    return _fb(left, bottom, right, top, FWI_WIDTH, FWI_HEIGHT)


def build_slope_aspect(terrain_dir: Path, overwrite: bool = False) -> None:
    slope_path  = terrain_dir / "slope.tif"
    aspect_path = terrain_dir / "aspect.tif"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    if slope_path.exists() and aspect_path.exists() and not overwrite:
        print(f"  [SKIP] slope.tif and aspect.tif already exist in {terrain_dir}")
        print("         Use --overwrite to recompute.")
        return

    # ── Step 1: Download SRTM tiles ───────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir   = Path(tmp)
        hgt_files = _download_srtm_tiles(tmp_dir)

        if not hgt_files:
            raise RuntimeError("No DEM files were extracted from SRTM download.")

        # ── Step 2: Merge all tiles into one mosaic ───────────────────
        print(f"\n  Merging {len(hgt_files)} DEM tiles…")
        open_files = [rasterio.open(f) for f in hgt_files]
        mosaic, mosaic_transform = merge(open_files)
        mosaic_crs = open_files[0].crs
        mosaic_arr = mosaic[0].astype(np.float32)

        # Replace SRTM nodata (−32768) with NaN
        mosaic_arr[mosaic_arr <= -32000] = np.nan

        for src in open_files:
            src.close()

        print(f"  Mosaic shape: {mosaic_arr.shape}  CRS: {mosaic_crs}")

        # ── Step 3: Reproject mosaic to FWI grid (EPSG:3978) ─────────
        print(f"  Reprojecting to FWI grid ({FWI_CRS}, {FWI_WIDTH}×{FWI_HEIGHT})…")
        fwi_transform = _fwi_transform()
        dem_fwi = np.full((FWI_HEIGHT, FWI_WIDTH), np.nan, dtype=np.float32)

        reproject(
            source           = mosaic_arr,
            destination      = dem_fwi,
            src_transform    = mosaic_transform,
            src_crs          = mosaic_crs,
            dst_transform    = fwi_transform,
            dst_crs          = FWI_CRS,
            resampling       = Resampling.bilinear,
            src_nodata       = np.nan,
            dst_nodata       = np.nan,
        )

        # ── Step 4: Compute slope and aspect ─────────────────────────
        # FWI grid cell size ≈ (3039835 - (-2378164)) / 2709 ≈ 2000 m (Lambert conformal)
        cell_size_m = (FWI_BOUNDS[2] - FWI_BOUNDS[0]) / FWI_WIDTH
        print(f"  Computing slope & aspect (cell_size ≈ {cell_size_m:.0f} m)…")
        slope, aspect = _compute_slope_aspect(dem_fwi, cell_size_m)

    # ── Step 5: Write output TIFs ─────────────────────────────────────
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     FWI_WIDTH,
        "height":    FWI_HEIGHT,
        "count":     1,
        "crs":       FWI_CRS,
        "transform": _fwi_transform(),
        "nodata":    np.nan,
        "compress":  "lzw",
    }

    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(slope, 1)
    print(f"  Saved: {slope_path}   "
          f"(mean slope: {np.nanmean(slope):.1f}°  max: {np.nanmax(slope):.1f}°)")

    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(aspect, 1)
    print(f"  Saved: {aspect_path}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Redownload and recompute even if output files already exist.",
    )
    args = parser.parse_args()
    cfg  = load_config(args.config)

    terrain_dir = Path(get_path(cfg, "terrain_dir"))
    print("SRTM Slope / Aspect builder")
    print(f"  Output dir: {terrain_dir}")
    print()
    build_slope_aspect(terrain_dir, overwrite=args.overwrite)
    print("\nDone.  terrain/slope.tif and terrain/aspect.tif are ready.")


if __name__ == "__main__":
    main()
