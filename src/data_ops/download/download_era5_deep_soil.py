#!/usr/bin/env python3
"""
Download ERA5 deep soil moisture (swvl2, 7-28 cm depth) and reproject to FWI grid.

Deep soil moisture is a slow-varying drought indicator that complements FWI:
  - FWI components respond to recent weather (days to weeks)
  - swvl2 responds to accumulated precipitation deficit over months
  - Low deep soil moisture → dry deep duff → fires harder to suppress

Source: Copernicus CDS reanalysis-era5-single-levels
Variable: volumetric_soil_water_layer_2 (swvl2, 7-28 cm, m^3/m^3)
Temporal: Daily average (hourly mean over 00-23 UTC)
Spatial:  0.25° x 0.25° (native) → EPSG:3978 2709x2281 (reprojected)

Pipeline:
  1. CDS API monthly GRIB requests (one file per month)
  2. Extract daily fields from monthly GRIB
  3. Reproject each daily field to FWI grid (bilinear)
  4. Save as deep_soil_YYYYMMDD.tif

Usage:
    python -m src.data_ops.download.download_era5_deep_soil \\
        2018-01-01 2025-10-31 --workers 2

    # Convert-only (GRIB files already downloaded on login node):
    python -m src.data_ops.download.download_era5_deep_soil \\
        2018-01-01 2025-10-31 --convert-only

Prerequisites:
    pip install cdsapi cfgrib eccodes
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, date
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


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)

DEFAULT_AREA = [83, -141, 41, -52]  # [N, W, S, E]
DEFAULT_CDS_API_KEY = "d952a10c-f9c0-4ff3-92e1-aac8756dd123"


# ------------------------------------------------------------------ #
# Download
# ------------------------------------------------------------------ #

def _make_cds_client(api_key: str):
    import cdsapi
    return cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=api_key,
    )


def download_month(client, year: int, month: int, outdir: Path,
                    area=None, verbose=True) -> bool:
    """Download one month of swvl2 data as GRIB."""
    if area is None:
        area = DEFAULT_AREA

    target = outdir / f"era5_swvl2_{year:04d}_{month:02d}.grib"
    if target.exists() and target.stat().st_size > 0:
        if verbose:
            print(f"[SKIP] {year}-{month:02d} already exists")
        return True

    import calendar
    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    req = {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": "volumetric_soil_water_layer_2",
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
    }

    try:
        if verbose:
            print(f"[DOWNLOAD] {year}-{month:02d} -> {target}")
        client.retrieve("reanalysis-era5-single-levels", req, str(target))
        if target.exists() and target.stat().st_size > 0:
            if verbose:
                print(f"[OK] {year}-{month:02d}: {target.stat().st_size / 1e6:.1f} MB")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] {year}-{month:02d}: {e}", file=sys.stderr)
        if target.exists():
            target.unlink()
        return False


# ------------------------------------------------------------------ #
# Convert GRIB -> daily GeoTIFF
# ------------------------------------------------------------------ #

def convert_month_grib(grib_path: Path, tif_dir: Path, verbose=True):
    """Extract daily means from monthly GRIB and save as individual GeoTIFFs."""
    try:
        import cfgrib
    except ImportError:
        print("[ERROR] cfgrib not installed. Run: pip install cfgrib", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"[CONVERT] {grib_path.name}")

    # Open GRIB with cfgrib/xarray
    import xarray as xr
    ds = xr.open_dataset(str(grib_path), engine="cfgrib",
                         backend_kwargs={"indexpath": ""})

    # Variable might be named 'swvl2' or 'SWVL2'
    var_name = None
    for name in ["swvl2", "SWVL2", "volumetric_soil_water_layer_2"]:
        if name in ds.data_vars:
            var_name = name
            break
    if var_name is None:
        # Use the first data variable
        var_name = list(ds.data_vars)[0]
        if verbose:
            print(f"  [WARN] Expected swvl2, found: {var_name}")

    data = ds[var_name]  # (time, latitude, longitude)

    # Compute daily means
    daily = data.resample(time="1D").mean()

    dst_transform = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    dst_crs = CRS.from_string(FWI_CRS)

    n_written = 0
    for t_idx in range(len(daily.time)):
        dt = daily.time.values[t_idx]
        date_str = str(dt)[:10].replace("-", "")
        tif_path = tif_dir / f"deep_soil_{date_str}.tif"

        if tif_path.exists():
            continue

        frame = daily.isel(time=t_idx).values.astype(np.float32)
        if frame.ndim != 2:
            continue

        # Handle NaN / nodata
        frame = np.where(np.isfinite(frame), frame, np.nan)

        # Build source transform from lat/lon
        lats = daily.latitude.values
        lons = daily.longitude.values
        # GRIB lats typically descending (N to S)
        if lats[0] < lats[-1]:
            frame = frame[::-1, :]
            lats = lats[::-1]

        src_transform = from_bounds(
            float(lons.min()), float(lats.min()),
            float(lons.max()), float(lats.max()),
            len(lons), len(lats),
        )
        src_crs = CRS.from_epsg(4326)

        # Reproject to FWI grid
        dst_data = np.full((FWI_HEIGHT, FWI_WIDTH), np.nan, dtype=np.float32)
        reproject(
            source=frame,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        # Write GeoTIFF
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": FWI_WIDTH,
            "height": FWI_HEIGHT,
            "count": 1,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": np.nan,
            "compress": "lzw",
        }
        tif_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(dst_data, 1)

        n_written += 1

    ds.close()
    if verbose:
        print(f"  [DONE] {n_written} daily TIFs written")
    return n_written


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 deep soil moisture (swvl2) -> FWI grid"
    )
    parser.add_argument("start_date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("end_date", type=str, help="End date YYYY-MM-DD")
    add_config_argument(parser)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel download workers (default: 1)")
    parser.add_argument("--convert-only", action="store_true",
                        help="Skip download, only convert existing GRIB files")
    parser.add_argument("--cds-api-key", type=str, default=None,
                        help="CDS API key (default: from environment or built-in)")
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        tif_dir = Path(get_path(cfg, "deep_soil_dir"))
    except (KeyError, TypeError):
        tif_dir = Path("data/era5_deep_soil")
    grib_dir = tif_dir / "grib"
    grib_dir.mkdir(parents=True, exist_ok=True)
    tif_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Build list of (year, month) to process
    months = []
    cur = date(start.year, start.month, 1)
    end_m = date(end.year, end.month, 1)
    while cur <= end_m:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    print(f"[INFO] Processing {len(months)} months: {months[0]} -> {months[-1]}")
    print(f"  GRIB dir: {grib_dir}")
    print(f"  TIF dir:  {tif_dir}")

    # Step 1: Download monthly GRIB files
    if not args.convert_only:
        api_key = args.cds_api_key or os.environ.get("CDS_API_KEY", DEFAULT_CDS_API_KEY)
        client = _make_cds_client(api_key)

        for year, month in months:
            ok = download_month(client, year, month, grib_dir)
            if not ok:
                print(f"[WARN] Failed to download {year}-{month:02d}, continuing...",
                      file=sys.stderr)

    # Step 2: Convert GRIB -> daily TIFs
    total_written = 0
    for year, month in months:
        grib_path = grib_dir / f"era5_swvl2_{year:04d}_{month:02d}.grib"
        if not grib_path.exists():
            print(f"[SKIP] GRIB not found: {grib_path}")
            continue
        n = convert_month_grib(grib_path, tif_dir)
        total_written += n

    print(f"[COMPLETE] {total_written} daily TIFs written to {tif_dir}")


if __name__ == "__main__":
    main()
