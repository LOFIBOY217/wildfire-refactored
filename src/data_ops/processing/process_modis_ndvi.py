#!/usr/bin/env python3
"""
Process raw MODIS MOD13A2 HDF4 files → daily NDVI/EVI TIFs on the FWI grid.

Input:  {ndvi_raw_dir}/{YYYY}/*.hdf  (from download_modis_ndvi.py)
Output: {ndvi_dir}/ndvi_YYYYMMDD.tif  (EPSG:3978, 2709×2281, float32)
        {evi_dir}/evi_YYYYMMDD.tif

Processing steps per year:
  1. Group HDF4 files by composite date
  2. For each composite: read tiles, mosaic, resample to FWI grid
  3. Linearly interpolate composites to daily
  4. Write daily TIFs

Usage:
    python -m src.data_ops.processing.process_modis_ndvi
    python -m src.data_ops.processing.process_modis_ndvi --config configs/paths_narval.yaml
    python -m src.data_ops.processing.process_modis_ndvi --start_year 2023 --end_year 2023

Prerequisites:
    pip install pyhdf rasterio
    (HDF4 files must be downloaded first by download_modis_ndvi.py)
"""

import argparse
import os
import sys
from datetime import date, timedelta
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


FWI_CRS       = "EPSG:3978"
FWI_WIDTH     = 2709
FWI_HEIGHT    = 2281
FWI_BOUNDS    = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
FWI_TRANSFORM = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)

NDVI_SCALE = 0.0001
EVI_SCALE  = 0.0001
MAX_PIXEL_RELIABILITY = 1
CANADA_BBOX_WGS84 = (-142.0, 41.0, -52.0, 84.0)


def _read_hdf4_sds(hdf_path: Path, sds_name: str) -> np.ndarray:
    """Read a named SDS from a MODIS HDF4 file."""
    from pyhdf.SD import SD, SDC
    hdf = SD(str(hdf_path), SDC.READ)
    sds = hdf.select(sds_name)
    data = sds.get().astype(np.float32)
    fill = sds.attributes().get("_FillValue", -28672)
    sds.end()
    hdf.end()
    data[data == fill] = np.nan
    return data


def _extract_modis_date(hdf_path: Path) -> date | None:
    """Extract acquisition date from MOD13A2 filename (AYYYYDDD)."""
    for p in hdf_path.stem.split("."):
        if p.startswith("A") and len(p) == 8:
            try:
                return date(int(p[1:5]), 1, 1) + timedelta(days=int(p[5:8]) - 1)
            except ValueError:
                continue
    return None


def _get_modis_tile_bounds(hdf_path: Path):
    """Extract tile bounds from MODIS HDF4 metadata (sinusoidal projection).
    Returns (left, bottom, right, top) in sinusoidal meters."""
    from pyhdf.SD import SD, SDC
    hdf = SD(str(hdf_path), SDC.READ)
    meta = hdf.attributes().get("StructMetadata.0", "")
    hdf.end()

    # Parse UpperLeftPointMtrs and LowerRightMtrs from metadata
    import re
    ul = re.search(r"UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)", meta)
    lr = re.search(r"LowerRightMtrs=\(([-\d.]+),([-\d.]+)\)", meta)
    if not ul or not lr:
        return None
    left, top = float(ul.group(1)), float(ul.group(2))
    right, bottom = float(lr.group(1)), float(lr.group(2))
    return (left, bottom, right, top)


# MODIS sinusoidal projection WKT
MODIS_SIN_CRS = CRS.from_proj4(
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)


def _process_composite(hdf_files: list[Path]):
    """Reproject each tile individually to FWI grid, then nanmean.
    Returns (ndvi_fwi, evi_fwi) as (H, W) float32 arrays."""
    dst_crs = CRS.from_string(FWI_CRS)

    # Accumulate reprojected tiles on FWI grid
    ndvi_sum = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float64)
    evi_sum = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float64)
    count = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float64)

    for hdf_path in hdf_files:
        try:
            ndvi_raw = _read_hdf4_sds(hdf_path, "1 km 16 days NDVI")
            evi_raw = _read_hdf4_sds(hdf_path, "1 km 16 days EVI")
            qa_raw = _read_hdf4_sds(hdf_path, "1 km 16 days pixel reliability")
        except Exception:
            continue

        bad = (qa_raw > MAX_PIXEL_RELIABILITY) | (~np.isfinite(qa_raw))
        ndvi_raw[bad] = np.nan
        evi_raw[bad] = np.nan

        ndvi_raw = np.where(np.isfinite(ndvi_raw), ndvi_raw * NDVI_SCALE, np.nan)
        evi_raw = np.where(np.isfinite(evi_raw), evi_raw * EVI_SCALE, np.nan)
        ndvi_raw = np.clip(ndvi_raw, -1.0, 1.0)
        evi_raw = np.clip(evi_raw, -1.0, 1.0)

        # Get tile bounds in sinusoidal CRS
        bounds = _get_modis_tile_bounds(hdf_path)
        if bounds is None:
            continue
        left, bottom, right, top = bounds
        h, w = ndvi_raw.shape
        src_tf = from_bounds(left, bottom, right, top, w, h)

        # Reproject this tile to FWI grid
        ndvi_fwi_tile = np.full((FWI_HEIGHT, FWI_WIDTH), np.nan, dtype=np.float32)
        evi_fwi_tile = np.full((FWI_HEIGHT, FWI_WIDTH), np.nan, dtype=np.float32)

        reproject(ndvi_raw, ndvi_fwi_tile,
                  src_transform=src_tf, src_crs=MODIS_SIN_CRS,
                  dst_transform=FWI_TRANSFORM, dst_crs=dst_crs,
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        reproject(evi_raw, evi_fwi_tile,
                  src_transform=src_tf, src_crs=MODIS_SIN_CRS,
                  dst_transform=FWI_TRANSFORM, dst_crs=dst_crs,
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)

        # Accumulate (running mean for overlapping areas)
        valid = np.isfinite(ndvi_fwi_tile)
        ndvi_sum[valid] += ndvi_fwi_tile[valid]
        evi_sum[valid] += evi_fwi_tile[valid]
        count[valid] += 1

        # Free memory
        del ndvi_raw, evi_raw, ndvi_fwi_tile, evi_fwi_tile

    # Compute mean
    ndvi_fwi = np.where(count > 0, ndvi_sum / count, np.nan).astype(np.float32)
    evi_fwi = np.where(count > 0, evi_sum / count, np.nan).astype(np.float32)
    return ndvi_fwi, evi_fwi


def _interpolate_to_daily(comp_dates, comp_ndvi, comp_evi, target_dates):
    """Linearly interpolate composites to daily grids."""
    daily = {}
    for td in target_dates:
        # Find bracketing composites
        before_i = None
        after_i = None
        for ci, cd in enumerate(comp_dates):
            if cd <= td:
                before_i = ci
            if cd > td and after_i is None:
                after_i = ci

        if before_i is None and after_i is None:
            continue
        elif before_i is None:
            daily[td] = (comp_ndvi[after_i].copy(), comp_evi[after_i].copy())
        elif after_i is None:
            daily[td] = (comp_ndvi[before_i].copy(), comp_evi[before_i].copy())
        else:
            gap = (comp_dates[after_i] - comp_dates[before_i]).days
            if gap <= 0:
                daily[td] = (comp_ndvi[before_i].copy(), comp_evi[before_i].copy())
            else:
                w = (td - comp_dates[before_i]).days / gap
                ndvi = (1 - w) * comp_ndvi[before_i] + w * comp_ndvi[after_i]
                evi = (1 - w) * comp_evi[before_i] + w * comp_evi[after_i]
                daily[td] = (ndvi.astype(np.float32), evi.astype(np.float32))
    return daily


def _valid_date(year, month, day):
    try:
        date(year, month, day)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Process MODIS HDF4 → daily NDVI/EVI TIFs")
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--months", type=int, nargs="+",
                        default=[4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ndvi_dir = Path(get_path(cfg, "ndvi_dir"))
    evi_dir = Path(get_path(cfg, "evi_dir"))
    raw_dir = ndvi_dir.parent / "ndvi_raw"
    ndvi_dir.mkdir(parents=True, exist_ok=True)
    evi_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": FWI_WIDTH, "height": FWI_HEIGHT, "count": 1,
        "crs": FWI_CRS, "transform": FWI_TRANSFORM,
        "nodata": np.nan, "compress": "lzw",
    }

    for year in range(args.start_year, args.end_year + 1):
        year_dir = raw_dir / str(year)
        hdf_files = sorted(year_dir.glob("*.hdf")) if year_dir.exists() else []
        if not hdf_files:
            print(f"  [{year}] No HDF4 files in {year_dir} — skipping")
            continue

        # Build target dates
        year_dates = [
            date(year, m, d)
            for m in args.months
            for d in range(1, 32)
            if _valid_date(year, m, d)
        ]

        # Check existing
        missing = [d for d in year_dates
                   if not (ndvi_dir / f"ndvi_{d.strftime('%Y%m%d')}.tif").exists()]
        if not missing and not args.overwrite:
            print(f"  [{year}] All daily files exist — skipping")
            continue

        # Group by composite date
        date_groups = {}
        for hdf_path in hdf_files:
            acq = _extract_modis_date(hdf_path)
            if acq is not None:
                date_groups.setdefault(acq, []).append(hdf_path)

        print(f"  [{year}] {len(hdf_files)} HDF4 files → {len(date_groups)} composites")

        # Process composites one at a time (low memory)
        comp_dates = sorted(date_groups.keys())
        comp_ndvi = []
        comp_evi = []
        for cd in comp_dates:
            ndvi_arr, evi_arr = _process_composite(date_groups[cd])
            comp_ndvi.append(ndvi_arr)
            comp_evi.append(evi_arr)
            print(f"    {cd}  ({len(date_groups[cd])} tiles)  "
                  f"valid={np.isfinite(ndvi_arr).mean()*100:.0f}%")

        if not comp_dates:
            continue

        # Interpolate to daily
        daily = _interpolate_to_daily(comp_dates, comp_ndvi, comp_evi, year_dates)

        written = 0
        for d, (ndvi_arr, evi_arr) in sorted(daily.items()):
            date_str = d.strftime("%Y%m%d")
            ndvi_path = ndvi_dir / f"ndvi_{date_str}.tif"
            evi_path = evi_dir / f"evi_{date_str}.tif"
            if ndvi_path.exists() and not args.overwrite:
                continue
            with rasterio.open(ndvi_path, "w", **profile) as dst:
                dst.write(ndvi_arr, 1)
            with rasterio.open(evi_path, "w", **profile) as dst:
                dst.write(evi_arr, 1)
            written += 1

        print(f"  [{year}] Written {written} daily NDVI+EVI files")

    print("\n[COMPLETE]")


if __name__ == "__main__":
    main()
