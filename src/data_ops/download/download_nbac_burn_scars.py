#!/usr/bin/env python3
"""
Download NBAC (National Burned Area Composite) annual shapefiles from NRCan
and rasterize them to the FWI grid as "years since last burn" layers.

Burned areas have significantly reduced fuel loads for 3-10 years after a fire
(the "fire shadow" effect). FWI has no knowledge of this — it only tracks
current meteorological dryness, not the actual available fuel at each location.

Output (one file per year):
    {burn_scars_dir}/years_since_burn_{YYYY}.tif
        Value = number of years since that pixel last burned (as of Jan 1 of YYYY)
        0  = burned in YYYY-1 (most recent fire season)
        1  = burned in YYYY-2
        ...
        No-data (9999) = never burned in the available record (1985–present)

Data source: NRCan NBAC — free, no account required.
    https://opendata.nfis.org/mapserver/nfis-change_eng.html

Prerequisites:
    pip install requests geopandas
    (rasterio and numpy already in project environment)

Usage:
    python -m src.data_ops.download.download_nbac_burn_scars
    python -m src.data_ops.download.download_nbac_burn_scars --start_year 2000 --end_year 2024
    python -m src.data_ops.download.download_nbac_burn_scars --config configs/paths_windows.yaml
    python -m src.data_ops.download.download_nbac_burn_scars --overwrite
"""

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
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


# ------------------------------------------------------------------ #
# FWI grid reference
# ------------------------------------------------------------------ #

FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)  # (left, bottom, right, top)
FWI_TRANSFORM = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)

# Sentinel for "never burned"
NODATA_YEARS = 9999

# NBAC download URL templates (NRCan open data)
# Multiple URL patterns to handle different naming conventions across years
NBAC_URL_TEMPLATES = [
    "https://opendata.nfis.org/downloads/forest_change/CA_Forest_Fire_NBAC_{year}_r9_20210810.zip",
    "https://opendata.nfis.org/downloads/forest_change/CA_Forest_Fire_NBAC_{year}.zip",
    "https://opendata.nfis.org/downloads/forest_change/nbac_{year}_20220624.zip",
    "https://opendata.nfis.org/downloads/forest_change/nbac_{year}_r9.zip",
]

TIMEOUT = 120
RETRY   = 3


# ------------------------------------------------------------------ #
# Download helper
# ------------------------------------------------------------------ #

def _download_nbac_zip(year: int) -> bytes | None:
    """Try each URL template until one succeeds. Returns raw zip bytes or None."""
    for template in NBAC_URL_TEMPLATES:
        url = template.format(year=year)
        for attempt in range(1, RETRY + 1):
            try:
                r = requests.get(url, timeout=TIMEOUT, stream=True)
                if r.status_code == 404:
                    break   # try next template
                r.raise_for_status()
                data = r.content
                print(f"    [OK] {url}  ({len(data)/1e6:.1f} MB)")
                return data
            except requests.exceptions.HTTPError:
                break   # 4xx → next template
            except Exception as e:
                if attempt < RETRY:
                    print(f"    [retry {attempt}] {e}")
                    time.sleep(3 * attempt)
    print(f"    [FAIL] No working URL found for year {year}")
    return None


# ------------------------------------------------------------------ #
# Rasterize one year's burn shapefile → binary burned mask
# ------------------------------------------------------------------ #

def _rasterize_burn_year(zip_bytes: bytes, year: int) -> np.ndarray | None:
    """
    Extract shapefile from zip bytes, reproject to FWI grid,
    rasterize to a binary uint8 mask (1 = burned, 0 = not burned).
    Returns None on failure.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas required: pip install geopandas")

    # Extract shapefile from zip into memory
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        shp_files = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shp_files:
            print(f"    [WARN] No .shp file found in NBAC {year} zip")
            return None

        # Extract all shapefile components (.shp, .dbf, .shx, .prj) to temp dir
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            shp_path = Path(tmp) / Path(shp_files[0]).name
            # Search recursively for the shp
            shp_candidates = list(Path(tmp).rglob("*.shp"))
            if not shp_candidates:
                return None
            shp_path = shp_candidates[0]

            gdf = gpd.read_file(shp_path)

    if gdf.empty:
        print(f"    [WARN] Empty GeoDataFrame for NBAC {year}")
        return None

    # Reproject to FWI CRS
    if str(gdf.crs) != FWI_CRS:
        gdf = gdf.to_crs(FWI_CRS)

    # Rasterize: 1 where polygon exists, 0 elsewhere
    shapes  = ((geom, 1) for geom in gdf.geometry if geom is not None)
    burned  = rasterize(
        shapes,
        out_shape   = (FWI_HEIGHT, FWI_WIDTH),
        transform   = FWI_TRANSFORM,
        fill        = 0,
        dtype       = np.uint8,
        all_touched = True,
    )
    n_burned = burned.sum()
    print(f"    Rasterized: {n_burned:,} burned pixels  "
          f"({n_burned / (FWI_HEIGHT * FWI_WIDTH) * 100:.2f}% of grid)")
    return burned


# ------------------------------------------------------------------ #
# Main processing
# ------------------------------------------------------------------ #

def build_years_since_burn(
    burn_scars_dir: Path,
    start_year: int,
    end_year:   int,
    overwrite:  bool = False,
) -> None:
    burn_scars_dir.mkdir(parents=True, exist_ok=True)

    # Check how many output files already exist
    existing = [
        burn_scars_dir / f"years_since_burn_{y}.tif"
        for y in range(start_year, end_year + 1)
        if (burn_scars_dir / f"years_since_burn_{y}.tif").exists()
    ]
    if existing and not overwrite:
        print(f"  [RESUME] {len(existing)} output files already exist; "
              f"skipping those years (use --overwrite to redo all).")

    # ── Download annual burn masks (all years in range + 10 years before
    #    start_year so early records have context for "years since burn") ──
    download_start = max(1985, start_year - 15)
    burn_masks: dict[int, np.ndarray] = {}

    print(f"\n  Downloading NBAC burn shapefiles {download_start}–{end_year}…")
    for year in range(download_start, end_year + 1):
        cache_path = burn_scars_dir / f"_nbac_mask_{year}.npy"

        if cache_path.exists() and not overwrite:
            burn_masks[year] = np.load(cache_path)
            print(f"  [{year}] Loaded cached mask  "
                  f"({burn_masks[year].sum():,} burned pixels)")
            continue

        print(f"  [{year}] Downloading NBAC…")
        zip_bytes = _download_nbac_zip(year)
        if zip_bytes is None:
            print(f"  [{year}] Skipping (no data)")
            continue

        mask = _rasterize_burn_year(zip_bytes, year)
        if mask is None:
            continue

        burn_masks[year] = mask
        np.save(cache_path, mask)   # cache raw mask for future runs

    if not burn_masks:
        raise RuntimeError("No NBAC burn masks could be downloaded.")

    # ── Compute "years since last burn" for each output year ──────────
    print(f"\n  Computing years-since-burn for {start_year}–{end_year}…")

    profile = {
        "driver":    "GTiff",
        "dtype":     "uint16",
        "width":     FWI_WIDTH,
        "height":    FWI_HEIGHT,
        "count":     1,
        "crs":       FWI_CRS,
        "transform": FWI_TRANSFORM,
        "nodata":    NODATA_YEARS,
        "compress":  "lzw",
    }

    for target_year in range(start_year, end_year + 1):
        out_path = burn_scars_dir / f"years_since_burn_{target_year}.tif"
        if out_path.exists() and not overwrite:
            print(f"  [{target_year}] [SKIP] already exists")
            continue

        # For each pixel, find the most recent year it burned (before target_year)
        years_since = np.full((FWI_HEIGHT, FWI_WIDTH), NODATA_YEARS, dtype=np.uint16)

        for burn_year in sorted(burn_masks.keys(), reverse=True):
            if burn_year >= target_year:
                continue   # only consider past fires
            age   = target_year - burn_year        # e.g. burned in 2020, target 2023 → age=3
            mask  = burn_masks[burn_year].astype(bool)
            # Update pixels where this is the most recent fire AND age < current value
            update = mask & (years_since == NODATA_YEARS)
            years_since[update] = age

        n_burned_ever = (years_since < NODATA_YEARS).sum()
        pct = n_burned_ever / (FWI_HEIGHT * FWI_WIDTH) * 100
        print(f"  [{target_year}] {n_burned_ever:,} pixels have burn history  "
              f"({pct:.1f}%)  median age={np.median(years_since[years_since < NODATA_YEARS]):.0f} yr")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(years_since, 1)
        print(f"           → {out_path}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2018,
                        help="First output year (default: 2018)")
    parser.add_argument("--end_year",   type=int, default=2024,
                        help="Last output year (default: 2024)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-download and recompute all years")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    burn_scars_dir = Path(get_path(cfg, "burn_scars_dir"))
    print("NBAC Burn Scars → Years-Since-Burn rasterizer")
    print(f"  Output dir : {burn_scars_dir}")
    print(f"  Years      : {args.start_year} – {args.end_year}")
    print()
    build_years_since_burn(
        burn_scars_dir = burn_scars_dir,
        start_year     = args.start_year,
        end_year       = args.end_year,
        overwrite      = args.overwrite,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
