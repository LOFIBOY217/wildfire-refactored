#!/usr/bin/env python3
"""
Rasterize CIFFC Fire Point Records to Gridded Rasters.

Provides reusable functions for loading CIFFC fire data (CSV/JSON) and
converting point observations to binary raster grids that align with the
FWI reference grid.

Extracted and unified from:
    - basic_regression/simple_logistic_7day.py   (load_ciffc_data, rasterize_fires)
    - basic_regression/evaluate_with_confusion_matrix.py (load_ciffc_data, rasterize_fires)

Three public functions:
    load_ciffc_data(path)           -> DataFrame with 'date' column
    rasterize_fires_batch(df, dates, profile) -> [T, H, W] uint8
    rasterize_fires_single(df, date, profile) -> [H, W] uint8

Usage as a library:
    from src.data_ops.processing.rasterize_fires import (
        load_ciffc_data, rasterize_fires_batch, rasterize_fires_single,
    )

Usage as CLI (quick smoke-test):
    python -m src.data_ops.processing.rasterize_fires \\
        --ciffc data/ciffc.csv \\
        --reference data/fwi_data/fwi_20250101.tif \\
        --date 2025-08-01

    # With config
    python -m src.data_ops.processing.rasterize_fires \\
        --config configs/default.yaml --date 2025-08-01
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime, date

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ciffc_data(ciffc_path):
    """
    Load CIFFC fire data from CSV or JSON.

    Expects a column named ``field_situation_report_date`` which is parsed
    into a ``date`` column (Python ``datetime.date`` objects).

    Args:
        ciffc_path: Path to a ``.csv`` or ``.json`` CIFFC file.

    Returns:
        pandas.DataFrame with at least these columns:
            - date (datetime.date)
            - field_longitude (float)
            - field_latitude (float)
    """
    ciffc_path = str(ciffc_path)

    if ciffc_path.endswith('.csv'):
        df = pd.read_csv(ciffc_path)
    elif ciffc_path.endswith('.json'):
        with open(ciffc_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"CIFFC file must be .csv or .json, got: {ciffc_path}")

    # Parse date column
    df['date'] = pd.to_datetime(df['field_situation_report_date']).dt.date

    return df


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def _build_transformer(profile):
    """Build a pyproj Transformer from EPSG:4326 to the raster's CRS."""
    return Transformer.from_crs(
        "EPSG:4326",
        profile['crs'],
        always_xy=True,
    )


def rasterize_fires_batch(fire_df, date_list, profile, nodata_value=-9999):
    """
    Rasterize fire points to a binary grid for each date in *date_list*.

    Args:
        fire_df: DataFrame from :func:`load_ciffc_data` (must have 'date',
                 'field_longitude', 'field_latitude').
        date_list: Iterable of ``datetime.date`` objects.
        profile: Rasterio profile dict (must contain 'crs', 'height',
                 'width', 'transform').
        nodata_value: Not used in output but kept for API compatibility.

    Returns:
        numpy.ndarray of shape ``[T, H, W]``, dtype ``uint8``.
        ``1`` = fire present, ``0`` = no fire.
    """
    transformer = _build_transformer(profile)
    height, width = profile['height'], profile['width']
    transform = profile['transform']

    fire_stack = []

    for target_date in date_list:
        fires_today = fire_df[fire_df['date'] == target_date]

        raster = np.zeros((height, width), dtype=np.uint8)

        if len(fires_today) > 0:
            lons = fires_today['field_longitude'].values
            lats = fires_today['field_latitude'].values
            xs, ys = transformer.transform(lons, lats)

            for x, y in zip(xs, ys):
                try:
                    row, col = rowcol(transform, x, y)
                    if 0 <= row < height and 0 <= col < width:
                        raster[row, col] = 1
                except Exception:
                    continue

        fire_stack.append(raster)

    return np.stack(fire_stack, axis=0)  # [T, H, W]


def rasterize_fires_single(fire_df, target_date, profile):
    """
    Rasterize fire points to a binary grid for a single date.

    Args:
        fire_df: DataFrame from :func:`load_ciffc_data`.
        target_date: A ``datetime.date`` object.
        profile: Rasterio profile dict.

    Returns:
        numpy.ndarray of shape ``[H, W]``, dtype ``uint8``.
    """
    fires_today = fire_df[fire_df['date'] == target_date]

    height, width = profile['height'], profile['width']
    raster = np.zeros((height, width), dtype=np.uint8)

    if len(fires_today) > 0:
        transformer = _build_transformer(profile)

        lons = fires_today['field_longitude'].values
        lats = fires_today['field_latitude'].values
        xs, ys = transformer.transform(lons, lats)

        for x, y in zip(xs, ys):
            try:
                row, col = rowcol(profile['transform'], x, y)
                if 0 <= row < height and 0 <= col < width:
                    raster[row, col] = 1
            except Exception:
                continue

    return raster  # [H, W]


# ---------------------------------------------------------------------------
# CLI (smoke-test)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rasterize CIFFC fire points onto the FWI grid (smoke-test CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)

    parser.add_argument(
        "--ciffc", type=str, default=None,
        help="Path to CIFFC CSV or JSON file (overrides config)",
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

    # Load config
    cfg = load_config(args.config)

    # Resolve CIFFC path
    ciffc_path = args.ciffc
    if ciffc_path is None:
        ciffc_path = get_path(cfg, 'ciffc_csv')
    if not os.path.exists(ciffc_path):
        print(f"Error: CIFFC file not found: {ciffc_path}")
        sys.exit(1)

    # Resolve reference path
    ref_path = args.reference
    if ref_path is None:
        fwi_dir = get_path(cfg, 'fwi_dir')
        candidates = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
        if candidates:
            ref_path = candidates[0]
        else:
            print(f"Error: No .tif files in fwi_dir ({fwi_dir}). Use --reference.")
            sys.exit(1)

    # Parse target date
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    # Load data
    print(f"Loading CIFFC data from: {ciffc_path}")
    ciffc_df = load_ciffc_data(ciffc_path)
    print(f"  Total records: {len(ciffc_df)}")
    print(f"  Date range: {ciffc_df['date'].min()} to {ciffc_df['date'].max()}")

    # Get profile from reference raster
    print(f"Reading reference grid from: {ref_path}")
    with rasterio.open(ref_path) as src:
        profile = src.profile

    # Rasterize
    print(f"\nRasterizing fires for {target_date}...")
    fire_raster = rasterize_fires_single(ciffc_df, target_date, profile)

    n_fire = int(fire_raster.sum())
    print(f"  Fire pixels: {n_fire}")
    print(f"  Raster shape: {fire_raster.shape}")

    # Optionally write output
    if args.output:
        out_profile = profile.copy()
        out_profile.update(dtype='uint8', count=1, nodata=0, compress='lzw')
        with rasterio.open(args.output, 'w', **out_profile) as dst:
            dst.write(fire_raster, 1)
        print(f"  Saved: {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
