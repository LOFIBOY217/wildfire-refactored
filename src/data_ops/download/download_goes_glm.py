#!/usr/bin/env python3
"""
Download GOES-16/17/18 GLM (Geostationary Lightning Mapper) data and produce
daily lightning flash-count rasters on a 0.1° WGS84 accumulation grid.

Download only — saves raw accumulation grids to disk.
Reprojection to the FWI grid (EPSG:3978) is done separately by
processing/resample_glm_to_fwi_grid.py.

Data source: NOAA GOES GLM-L2-LCFA on AWS S3 — open, no account needed.
  GOES-16 (75.0°W):  s3://noaa-goes16/GLM-L2-LCFA/   ← eastern Canada
  GOES-18 (137.2°W): s3://noaa-goes18/GLM-L2-LCFA/   ← western Canada (2023-present)
  GOES-17 (137.2°W): s3://noaa-goes17/GLM-L2-LCFA/   ← western Canada (2019-2023)

Pipeline:
  1. Lists all granule files for each day from every visible satellite.
  2. Streams flash lat/lon, accumulates on a 0.1° count grid over Canada.
  3. Saves: {lightning_raw_dir}/glm_raw_{YYYYMMDD}.tif  (float32, WGS84, 0.1°)

Usage:
    python -m src.data_ops.download.download_goes_glm \\
        --start 20180501 --end 20241031
    python -m src.data_ops.download.download_goes_glm \\
        --start 20230601 --end 20230630 --workers 8

Prerequisites:
    pip install s3fs netCDF4
"""

import argparse
import concurrent.futures
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
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
# Intermediate accumulation grid (0.1° WGS84 over Canada)
# ------------------------------------------------------------------ #

ACC_LAT_MIN =  41.0
ACC_LAT_MAX =  85.0
ACC_LON_MIN = -142.0
ACC_LON_MAX =  -50.0
ACC_DEG     =    0.1

ACC_NLAT = round((ACC_LAT_MAX - ACC_LAT_MIN) / ACC_DEG)   # 440
ACC_NLON = round((ACC_LON_MAX - ACC_LON_MIN) / ACC_DEG)   # 920

ACC_TRANSFORM = from_bounds(
    ACC_LON_MIN, ACC_LAT_MIN, ACC_LON_MAX, ACC_LAT_MAX,
    ACC_NLON, ACC_NLAT,
)
ACC_CRS = CRS.from_epsg(4326)


# ------------------------------------------------------------------ #
# GOES satellite config
# ------------------------------------------------------------------ #

GOES_SATELLITES = [
    ("noaa-goes16", "GOES-16 (75°W)", date(2017, 4, 1)),
    ("noaa-goes18", "GOES-18 (137°W)", date(2023, 1, 10)),
    ("noaa-goes17", "GOES-17 (137°W)", date(2019, 2, 12)),
]


# ------------------------------------------------------------------ #
# S3 helpers
# ------------------------------------------------------------------ #

def _get_s3():
    """Return an anonymous s3fs filesystem (no AWS credentials needed)."""
    try:
        import s3fs
    except ImportError:
        raise ImportError(
            "s3fs is required: pip install s3fs\n"
            "Also: pip install netCDF4"
        )
    return s3fs.S3FileSystem(anon=True)


def _list_day_files(s3, bucket: str, d: date) -> list[str]:
    """Return all GLM-L2-LCFA granule paths for a given satellite bucket and day."""
    doy = d.timetuple().tm_yday
    prefix = f"{bucket}/GLM-L2-LCFA/{d.year}/{doy:03d}/"
    try:
        files = s3.glob(f"{prefix}**/*.nc")
        return files
    except Exception:
        return []


# ------------------------------------------------------------------ #
# NetCDF reading (single granule → flash lat/lon arrays)
# ------------------------------------------------------------------ #

def _read_granule_flashes(s3, path: str):
    """
    Open one GLM-L2-LCFA NetCDF4 file on S3 and return (flash_lat, flash_lon).
    Returns None on any error.
    """
    try:
        import netCDF4 as nc4
    except ImportError:
        raise ImportError("netCDF4 is required: pip install netCDF4")

    import tempfile as _tf
    try:
        with s3.open(path, "rb") as fobj:
            raw = fobj.read()

        with _tf.NamedTemporaryFile(suffix=".nc", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            with nc4.Dataset(tmp.name, "r") as ds:
                if "flash_lat" not in ds.variables or "flash_lon" not in ds.variables:
                    return None
                lat = ds["flash_lat"][:].data.astype(np.float32)
                lon = ds["flash_lon"][:].data.astype(np.float32)
                if len(lat) == 0:
                    return None
                return lat, lon
    except Exception:
        return None


# ------------------------------------------------------------------ #
# Per-day accumulation
# ------------------------------------------------------------------ #

def _accumulate_day(s3, d: date, max_workers: int = 4,
                    verbose: bool = False) -> np.ndarray:
    """
    Download all GLM granules for one day, accumulate flash counts
    on the 0.1° grid. Returns (ACC_NLAT, ACC_NLON) float32.
    """
    count_grid = np.zeros((ACC_NLAT, ACC_NLON), dtype=np.float32)

    for bucket, sat_name, first_date in GOES_SATELLITES:
        if d < first_date:
            continue

        files = _list_day_files(s3, bucket, d)
        if not files:
            continue

        if verbose:
            print(f"      [{sat_name}] {len(files)} granules")

        local_grid = np.zeros_like(count_grid)
        ok = 0

        # Process in batches of 200 to limit memory (4000+ futures at once = OOM)
        BATCH = 200
        for bi in range(0, len(files), BATCH):
            batch_files = files[bi:bi + BATCH]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_read_granule_flashes, s3, f): f
                           for f in batch_files}
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    if result is None:
                        continue
                    lat_arr, lon_arr = result

                    mask = (
                        (lat_arr >= ACC_LAT_MIN) & (lat_arr < ACC_LAT_MAX) &
                        (lon_arr >= ACC_LON_MIN) & (lon_arr < ACC_LON_MAX)
                    )
                    lat_f = lat_arr[mask]
                    lon_f = lon_arr[mask]

                    if len(lat_f) == 0:
                        continue

                    row = ((ACC_LAT_MAX - lat_f) / ACC_DEG).astype(np.int32)
                    col = ((lon_f - ACC_LON_MIN) / ACC_DEG).astype(np.int32)
                    row = np.clip(row, 0, ACC_NLAT - 1)
                    col = np.clip(col, 0, ACC_NLON - 1)

                    np.add.at(local_grid, (row, col), 1)
                    ok += 1

        count_grid += local_grid
        if verbose:
            print(f"      [{sat_name}] {ok} granules OK  "
                  f"→ {int(local_grid.sum()):,} flashes")

    return count_grid


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Download GOES GLM lightning → 0.1° WGS84 daily TIFs (download only)"
    )
    add_config_argument(parser)
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--workers", type=int, default=4,
                        help="S3 reader threads per day (default: 4)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Output to lightning_raw/ (intermediate, not lightning/ which is FWI-grid)
    lightning_dir = Path(get_path(cfg, "lightning_dir"))
    raw_dir = lightning_dir.parent / "lightning_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:8]))
    end = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:8]))
    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    # Output profile: 0.1° WGS84 accumulation grid
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": ACC_NLON, "height": ACC_NLAT, "count": 1,
        "crs": ACC_CRS, "transform": ACC_TRANSFORM,
        "nodata": 0.0, "compress": "lzw",
    }

    print("=" * 70)
    print("GOES GLM Lightning — Download Only (0.1° WGS84)")
    print("=" * 70)
    print(f"  Dates     : {start} – {end}  ({len(all_days)} days)")
    print(f"  Output    : {raw_dir}/glm_raw_YYYYMMDD.tif")
    print(f"  Workers   : {args.workers}")
    print(f"  NOTE: Run processing/resample_glm_to_fwi_grid.py after download")
    print("=" * 70)

    s3 = _get_s3()
    done = skipped = failed = 0

    for i, d in enumerate(all_days, 1):
        date_str = d.strftime("%Y%m%d")
        out_path = raw_dir / f"glm_raw_{date_str}.tif"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        print(f"  [{i:04d}/{len(all_days)}] {date_str}  accumulating…")

        try:
            acc_grid = _accumulate_day(s3, d, max_workers=args.workers,
                                       verbose=True)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(acc_grid, 1)

            done += 1
            print(f"           → {int(acc_grid.sum()):,} flashes → {out_path.name}")
        except Exception as exc:
            failed += 1
            print(f"  [{i:04d}/{len(all_days)}] {date_str} [FAIL] {exc}")

    print(f"\n  Summary: {done} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
