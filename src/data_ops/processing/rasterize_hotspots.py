#!/usr/bin/env python3
"""
Rasterize CWFIS/FIRMS Satellite Hotspot Records to Gridded Rasters.

Drop-in replacement for rasterize_fires.py that uses satellite hotspot data
(VIIRS-M detections from CWFIS) instead of human-reported CIFFC fire records.

Key differences from rasterize_fires.py:
    - Input CSV has columns: latitude, longitude, acq_date  (not CIFFC format)
    - No area expansion: each hotspot marks exactly one pixel (~375m satellite
      resolution vs CIFFC point with reported fire size in hectares)
    - Vectorised rasterization: hotspots can be 10-100x more numerous per day
      than CIFFC records, so we avoid Python-level loops over individual points

Three public functions (identical signatures to rasterize_fires.py):
    load_hotspot_data(path)               -> DataFrame with 'date' column
    rasterize_hotspots_batch(df, dates, profile) -> [T, H, W] uint8
    rasterize_hotspots_single(df, date, profile) -> [H, W] uint8

Usage as a library:
    from src.data_ops.processing.rasterize_hotspots import (
        load_hotspot_data, rasterize_hotspots_batch, rasterize_hotspots_single,
    )

Usage as CLI (smoke-test):
    python -m src.data_ops.processing.rasterize_hotspots \\
        --config configs/default.yaml --date 2023-08-01

    python -m src.data_ops.processing.rasterize_hotspots \\
        --hotspot data/hotspot/hotspot_2018_2025.csv \\
        --reference data/fwi_data/fwi_20230801.tif \\
        --date 2023-08-01 --output /tmp/hotspot_20230801.tif
"""

import argparse
import glob
import os
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hotspot_data(hotspot_path):
    """
    Load CWFIS/FIRMS hotspot data from CSV.

    Expects columns: latitude, longitude, acq_date
    (output format of download_hotspots.py).

    Args:
        hotspot_path: Path to the hotspot CSV file.

    Returns:
        pandas.DataFrame with columns:
            - date           (datetime.date)
            - field_latitude  (float)   — named for API compatibility with
            - field_longitude (float)     rasterize_fires.py callers
    """
    df = pd.read_csv(str(hotspot_path))

    # Parse date column
    df['date'] = pd.to_datetime(df['acq_date']).dt.date

    # Rename to match the field names expected by callers that were previously
    # using load_ciffc_data (makes it a drop-in replacement).
    df = df.rename(columns={
        'latitude':  'field_latitude',
        'longitude': 'field_longitude',
    })

    return df[['date', 'field_latitude', 'field_longitude']]


def load_nfdb_as_hotspot_df(nfdb_shp_or_zip_path,
                            min_size_ha=0.0,
                            year_min=None, year_max=None,
                            causes=None):
    """
    Load NFDB point shapefile and return in the same DataFrame format as
    load_hotspot_data() so it can be fed into rasterize_hotspots_batch().

    NFDB = Canadian National Fire Database (point version): 442k fires,
    1930-2024, with reported ignition date + lat/lon + size + cause.
    CRS is already EPSG:3978 in the source but we convert to lat/lon
    (EPSG:4326) for uniform handling with the hotspot rasterizer.

    Args:
        nfdb_shp_or_zip_path: Path to NFDB .shp or .zip containing it.
        min_size_ha: Exclude fires smaller than this (default 0 = include all).
                     Useful to downweight small, noisy early records.
        year_min, year_max: Optional year filter.
        causes: Optional iterable of CAUSE codes to keep
                (e.g. {"N", "H", "U"}). Default: all.

    Returns:
        DataFrame with columns: date, field_latitude, field_longitude.
        date is datetime.date parsed from REP_DATE (or YEAR/MONTH/DAY
        fallback if REP_DATE missing).
    """
    import zipfile as _zf

    p = Path(nfdb_shp_or_zip_path)
    if p.suffix.lower() == ".zip":
        extract_dir = p.parent / (p.stem + "_extract")
        extract_dir.mkdir(parents=True, exist_ok=True)
        shps = list(extract_dir.glob("*.shp"))
        if not shps:
            with _zf.ZipFile(p) as zf:
                for name in zf.namelist():
                    if name.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                        zf.extract(name, extract_dir)
            shps = list(extract_dir.rglob("*.shp"))
        if not shps:
            raise RuntimeError(f"No .shp inside {p}")
        shp = shps[0]
    else:
        shp = p

    import geopandas as gpd
    gdf = gpd.read_file(shp)

    # Reproject to EPSG:4326 so _rasterize_points transformer works uniformly
    if str(gdf.crs).lower() != "epsg:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Build date column. Prefer REP_DATE; fallback to YEAR-MONTH-DAY.
    if "REP_DATE" in gdf.columns:
        d = pd.to_datetime(gdf["REP_DATE"], errors="coerce")
    else:
        d = pd.to_datetime({
            "year": gdf.get("YEAR"),
            "month": gdf.get("MONTH"),
            "day": gdf.get("DAY"),
        }, errors="coerce")
    gdf = gdf.assign(_date=d)

    # Filters
    mask = gdf["_date"].notna()
    if year_min is not None:
        mask &= gdf["_date"].dt.year >= year_min
    if year_max is not None:
        mask &= gdf["_date"].dt.year <= year_max
    if "SIZE_HA" in gdf.columns and min_size_ha > 0:
        mask &= gdf["SIZE_HA"].fillna(0) >= min_size_ha
    if causes is not None and "CAUSE" in gdf.columns:
        mask &= gdf["CAUSE"].isin(set(causes))

    gdf = gdf.loc[mask].copy()

    # Latitude / longitude from geometry (since we reprojected)
    lat = gdf.geometry.y.values
    lon = gdf.geometry.x.values
    date_arr = gdf["_date"].dt.date.values

    # Prefer LATITUDE/LONGITUDE columns if present and geometry missing
    if len(lat) == 0 and "LATITUDE" in gdf.columns:
        lat = gdf["LATITUDE"].values
        lon = gdf["LONGITUDE"].values

    df = pd.DataFrame({
        "date": date_arr,
        "field_latitude": lat,
        "field_longitude": lon,
    })
    return df


# ---------------------------------------------------------------------------
# Rasterization helpers
# ---------------------------------------------------------------------------

def _build_transformer(profile):
    """Build a pyproj Transformer from EPSG:4326 to the raster's CRS."""
    return Transformer.from_crs(
        "EPSG:4326",
        profile['crs'],
        always_xy=True,
    )


def _rasterize_points(lats, lons, transformer, transform, height, width):
    """
    Convert arrays of (lat, lon) points to a binary [H, W] raster.

    Vectorised: projects all points at once and uses numpy fancy indexing
    to mark pixels — no Python loop over individual hotspots.

    Args:
        lats, lons:   1-D float arrays (EPSG:4326).
        transformer:  pyproj Transformer (EPSG:4326 → raster CRS).
        transform:    Rasterio affine transform.
        height, width: Raster dimensions.

    Returns:
        numpy.ndarray [H, W], dtype uint8.
    """
    raster = np.zeros((height, width), dtype=np.uint8)

    if len(lats) == 0:
        return raster

    # Project all points at once
    xs, ys = transformer.transform(lons, lats)

    # Convert projected coords to pixel (row, col) indices
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.intp)
    cols = np.asarray(cols, dtype=np.intp)

    # Keep only pixels that fall within the raster extent
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    raster[rows[valid], cols[valid]] = 1

    return raster


# ---------------------------------------------------------------------------
# Public API  (same signatures as rasterize_fires.py)
# ---------------------------------------------------------------------------

def rasterize_hotspots_batch(hotspot_df, date_list, profile, nodata_value=-9999):
    """
    Rasterize hotspot points to a binary grid for each date in *date_list*.

    Each hotspot marks exactly one pixel (no area expansion — satellite
    detections are already at ~375m pixel resolution).

    Args:
        hotspot_df: DataFrame from :func:`load_hotspot_data`.
        date_list:  Iterable of ``datetime.date`` objects.
        profile:    Rasterio profile dict (must contain 'crs', 'height',
                    'width', 'transform').
        nodata_value: Unused; kept for API compatibility with rasterize_fires.py.

    Returns:
        numpy.ndarray of shape ``[T, H, W]``, dtype ``uint8``.
        ``1`` = hotspot present, ``0`` = no hotspot.
    """
    transformer = _build_transformer(profile)
    height = profile['height']
    width  = profile['width']
    transform = profile['transform']

    fire_stack = []

    for target_date in date_list:
        today = hotspot_df[hotspot_df['date'] == target_date]

        raster = _rasterize_points(
            lats=today['field_latitude'].values,
            lons=today['field_longitude'].values,
            transformer=transformer,
            transform=transform,
            height=height,
            width=width,
        )
        fire_stack.append(raster)

    return np.stack(fire_stack, axis=0)   # [T, H, W]


def rasterize_hotspots_single(hotspot_df, target_date, profile):
    """
    Rasterize hotspot points to a binary grid for a single date.

    Args:
        hotspot_df:  DataFrame from :func:`load_hotspot_data`.
        target_date: A ``datetime.date`` object.
        profile:     Rasterio profile dict.

    Returns:
        numpy.ndarray of shape ``[H, W]``, dtype ``uint8``.
    """
    today = hotspot_df[hotspot_df['date'] == target_date]

    transformer = _build_transformer(profile)

    return _rasterize_points(
        lats=today['field_latitude'].values,
        lons=today['field_longitude'].values,
        transformer=transformer,
        transform=profile['transform'],
        height=profile['height'],
        width=profile['width'],
    )


# ---------------------------------------------------------------------------
# CLI (smoke-test)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rasterize CWFIS hotspot points onto the FWI grid (smoke-test CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)

    parser.add_argument(
        "--hotspot", type=str, default=None,
        help="Path to hotspot CSV file (overrides config hotspot_csv)",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Path to a reference FWI GeoTIFF (overrides config)",
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="Target date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output GeoTIFF path (optional; if omitted prints stats only)",
    )

    args = parser.parse_args()
    cfg  = load_config(args.config)

    # Resolve hotspot path
    hotspot_path = args.hotspot
    if hotspot_path is None:
        try:
            hotspot_path = get_path(cfg, 'hotspot_csv')
        except Exception:
            hotspot_path = "data/hotspot/hotspot_2018_2025.csv"
    if not os.path.exists(hotspot_path):
        print(f"Error: hotspot file not found: {hotspot_path}")
        sys.exit(1)

    # Resolve reference raster path
    ref_path = args.reference
    if ref_path is None:
        fwi_dir    = get_path(cfg, 'fwi_dir')
        candidates = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
        if candidates:
            ref_path = candidates[0]
        else:
            print(f"Error: No .tif files in fwi_dir ({fwi_dir}). Use --reference.")
            sys.exit(1)

    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    # Load data
    print(f"Loading hotspot data from: {hotspot_path}")
    hotspot_df = load_hotspot_data(hotspot_path)
    print(f"  Total records : {len(hotspot_df):,}")
    print(f"  Date range    : {hotspot_df['date'].min()} to {hotspot_df['date'].max()}")

    # Get profile from reference raster
    print(f"Reading reference grid from: {ref_path}")
    with rasterio.open(ref_path) as src:
        profile = src.profile

    # Rasterize
    print(f"\nRasterizing hotspots for {target_date}...")
    raster = rasterize_hotspots_single(hotspot_df, target_date, profile)

    n_fire = int(raster.sum())
    print(f"  Hotspot pixels : {n_fire}")
    print(f"  Raster shape   : {raster.shape}")
    print(f"  Fire fraction  : {n_fire / raster.size * 100:.3f}%")

    # Optionally write output GeoTIFF
    if args.output:
        out_profile = profile.copy()
        out_profile.update(dtype='uint8', count=1, nodata=0, compress='lzw')
        with rasterio.open(args.output, 'w', **out_profile) as dst:
            dst.write(raster, 1)
        print(f"  Saved: {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
