#!/usr/bin/env python3
"""
Download ERA5 deep soil moisture (swvl2, 7-28 cm depth) as daily GeoTIFFs.

Uses the SAME per-day CDS API pattern as download_ecmwf_reanalysis_observations.py
to avoid OOM: one small GRIB per day (~3 MB), immediately averaged and reprojected.

Source: Copernicus CDS reanalysis-era5-single-levels
Variable: volumetric_soil_water_layer_2 (swvl2, 7-28 cm, m^3/m^3)
Output: {deep_soil_dir}/deep_soil_YYYYMMDD.tif (float32, EPSG:3978)

Usage:
    python -m src.data_ops.download.download_era5_deep_soil \\
        2018-01-01 2025-10-31 --workers 2
"""

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _make_cds_client(api_key: str):
    import cdsapi
    return cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=api_key,
    )


def _download_and_process_day(client, date_str, tif_dir, grib_tmp_dir,
                               area=None, verbose=True):
    """Download one day of swvl2, compute daily mean, reproject to FWI grid."""
    if area is None:
        area = DEFAULT_AREA

    date_str_compact = date_str.replace("-", "")
    tif_path = tif_dir / f"deep_soil_{date_str_compact}.tif"

    if tif_path.exists() and tif_path.stat().st_size > 0:
        return date_str, True, "skip"

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    nc_path = grib_tmp_dir / f"swvl2_{date_str_compact}.nc"

    # Download single day as NetCDF (~5 MB) — avoids cfgrib/eccodes dependency
    req = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "volumetric_soil_water_layer_2",
        "year": date_obj.strftime("%Y"),
        "month": date_obj.strftime("%m"),
        "day": date_obj.strftime("%d"),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
    }

    try:
        client.retrieve("reanalysis-era5-single-levels", req, str(nc_path))
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {date_str}: {e}", file=sys.stderr)
        return date_str, False, str(e)

    if not nc_path.exists() or nc_path.stat().st_size == 0:
        return date_str, False, "empty nc"

    # Read NetCDF and compute daily mean (no cfgrib needed)
    try:
        import netCDF4 as nc4
        with nc4.Dataset(str(nc_path), "r") as ds:
            # Variable name: 'swvl2' in CDS NetCDF output
            var_name = None
            for name in ["swvl2", "SWVL2", "volumetric_soil_water_layer_2"]:
                if name in ds.variables:
                    var_name = name
                    break
            if var_name is None:
                # Use first non-coordinate variable
                coords = {"time", "latitude", "longitude", "lat", "lon"}
                for name in ds.variables:
                    if name not in coords:
                        var_name = name
                        break
            if var_name is None:
                nc_path.unlink()
                return date_str, False, "no data var found"

            # Shape: (24, nlat, nlon) — compute daily mean
            data = ds[var_name][:].data.astype(np.float32)
            frame = np.nanmean(data, axis=0)

            # Get lat/lon
            lat_name = "latitude" if "latitude" in ds.variables else "lat"
            lon_name = "longitude" if "longitude" in ds.variables else "lon"
            lats = ds[lat_name][:].data.astype(np.float64)
            lons = ds[lon_name][:].data.astype(np.float64)

    except Exception as e:
        if verbose:
            print(f"  [ERROR] {date_str} convert: {e}", file=sys.stderr)
        if nc_path.exists():
            nc_path.unlink()
        return date_str, False, str(e)

    # Cleanup NC immediately
    if nc_path.exists():
        nc_path.unlink()

    # Handle NaN
    frame = np.where(np.isfinite(frame), frame, np.nan)

    # Lats may be descending
    if lats[0] < lats[-1]:
        frame = frame[::-1, :]
        lats = lats[::-1]

    src_transform = from_bounds(
        float(lons.min()), float(lats.min()),
        float(lons.max()), float(lats.max()),
        len(lons), len(lats),
    )
    src_crs = CRS.from_epsg(4326)
    dst_transform = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    dst_crs = CRS.from_string(FWI_CRS)

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

    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": FWI_WIDTH, "height": FWI_HEIGHT, "count": 1,
        "crs": dst_crs, "transform": dst_transform,
        "nodata": np.nan, "compress": "lzw",
    }
    tif_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(dst_data, 1)

    return date_str, True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 deep soil moisture (swvl2) -> daily TIFs on FWI grid"
    )
    parser.add_argument("start_date", type=str, help="YYYY-MM-DD")
    parser.add_argument("end_date", type=str, help="YYYY-MM-DD")
    add_config_argument(parser)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cds-api-key", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        tif_dir = Path(get_path(cfg, "deep_soil_dir"))
    except (KeyError, TypeError):
        tif_dir = Path("data/era5_deep_soil")
    tif_dir.mkdir(parents=True, exist_ok=True)

    grib_tmp = tif_dir / "grib_tmp"
    grib_tmp.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    print(f"[INFO] {len(dates)} days: {dates[0]} -> {dates[-1]}")
    print(f"  TIF dir: {tif_dir}")

    api_key = args.cds_api_key or os.environ.get("CDS_API_KEY", DEFAULT_CDS_API_KEY)
    client = _make_cds_client(api_key)

    n_ok, n_skip, n_fail = 0, 0, 0
    for i, d in enumerate(dates):
        d_str, ok, status = _download_and_process_day(
            client, d, tif_dir, grib_tmp)
        if status == "skip":
            n_skip += 1
        elif ok:
            n_ok += 1
        else:
            n_fail += 1
        if (i + 1) % 30 == 0 or i == len(dates) - 1:
            print(f"  [{i+1}/{len(dates)}] ok={n_ok} skip={n_skip} fail={n_fail}")

    print(f"[COMPLETE] ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
