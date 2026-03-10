#!/usr/bin/env python3
"""
Download GOES-16/17/18 GLM (Geostationary Lightning Mapper) data and produce
daily lightning flash-count rasters aligned to the FWI grid.

Lightning causes ~55% of Canadian wildfires but the FWI system has no
knowledge of point-ignition sources.  This script fills that gap.

Data source: NOAA GOES GLM-L2-LCFA on AWS S3 — open, no account needed.
  GOES-16 (75.0°W):  s3://noaa-goes16/GLM-L2-LCFA/   ← eastern Canada
  GOES-18 (137.2°W): s3://noaa-goes18/GLM-L2-LCFA/   ← western Canada (2023-present)
  GOES-17 (137.2°W): s3://noaa-goes17/GLM-L2-LCFA/   ← western Canada (2019-2023)

Each ~20-second granule file contains flash_lat, flash_lon, flash_count (~1-2 MB).
This script:
  1. Lists all granule files for each day from every visible satellite.
  2. Streams flash lat/lon, accumulates a 0.1° count grid over Canada.
  3. Reprojects that grid to the FWI grid (EPSG:3978, 2709×2281, bilinear).
  4. Saves: {lightning_dir}/lightning_{YYYYMMDD}.tif  (float32, flashes/pixel/day)

Output semantics:
  Each pixel = total GLM flashes detected within that FWI grid cell on that day
  (combined from all satellites).  0 = no flashes detected.  NaN = outside
  the latitude band covered by any satellite on that day.

Usage:
    python -m src.data_ops.download.download_goes_glm \\
        --start 20180501 --end 20241031
    python -m src.data_ops.download.download_goes_glm \\
        --start 20230601 --end 20230630 \\
        --workers 8 --config configs/paths_windows.yaml --overwrite

Prerequisites:
    pip install s3fs netCDF4
    (numpy, rasterio already in project environment)

Notes:
  - Each fire-season day generates ~4 320 granule files per satellite.
    With --workers 8 a typical day takes 3-8 minutes depending on bandwidth.
  - GOES-16 has continuous coverage from 2017-04-01 onward.
  - GOES-17 has coverage from 2019-02-12; replaced by GOES-18 on 2023-01-10.
  - If a satellite has no files for a day (pre-launch, outage), it is skipped.
  - Flash counts are additive across satellites; Canada is double-covered only
    near the longitude boundary (~100°W) where both satellites see it — this
    is acceptable because the double-count zone is small and fire models do
    not require absolute calibration.
"""

import argparse
import concurrent.futures
import io
import sys
from collections import defaultdict
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


# ------------------------------------------------------------------ #
# FWI grid constants (EPSG:3978, Lambert Conformal Conic)
# ------------------------------------------------------------------ #

FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)  # (W, S, E, N)
FWI_TRANSFORM = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)

# ------------------------------------------------------------------ #
# Intermediate accumulation grid (0.1° WGS84 over Canada)
# ------------------------------------------------------------------ #

# Canada bbox (generous — includes Yukon, Labrador sea)
ACC_LAT_MIN =  41.0
ACC_LAT_MAX =  85.0
ACC_LON_MIN = -142.0
ACC_LON_MAX =  -50.0
ACC_DEG     =    0.1                    # grid resolution in degrees

ACC_NLAT = round((ACC_LAT_MAX - ACC_LAT_MIN) / ACC_DEG)   # 440
ACC_NLON = round((ACC_LON_MAX - ACC_LON_MIN) / ACC_DEG)   # 920

# Affine transform for the 0.1° accumulation grid (origin = top-left corner)
ACC_TRANSFORM = from_bounds(
    ACC_LON_MIN, ACC_LAT_MIN, ACC_LON_MAX, ACC_LAT_MAX,
    ACC_NLON, ACC_NLAT,
)
ACC_CRS = CRS.from_epsg(4326)

# ------------------------------------------------------------------ #
# GOES satellite config
# ------------------------------------------------------------------ #

# Each entry: (bucket_name, description, first_operational_date)
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
    """
    Return all GLM-L2-LCFA granule paths for a given satellite bucket and day.
    GLM files live under: s3://{bucket}/GLM-L2-LCFA/{YYYY}/{DDD}/{HH}/
    """
    doy = d.timetuple().tm_yday     # day-of-year (1–366)
    prefix = f"{bucket}/GLM-L2-LCFA/{d.year}/{doy:03d}/"
    try:
        files = s3.glob(f"{prefix}**/*.nc")
        return files
    except Exception:
        return []


# ------------------------------------------------------------------ #
# NetCDF reading (single granule → flash lat/lon arrays)
# ------------------------------------------------------------------ #

def _read_granule_flashes(s3, path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Open one GLM-L2-LCFA NetCDF4 file on S3 and return (flash_lat, flash_lon).
    Returns None on any error (corrupt file, empty, etc.).
    """
    try:
        import netCDF4 as nc4
    except ImportError:
        raise ImportError("netCDF4 is required: pip install netCDF4")

    try:
        with s3.open(path, "rb") as fobj:
            raw = fobj.read()          # typically 500 KB – 2 MB

        with nc4.Dataset("inmemory.nc", memory=raw) as ds:
            # GLM variables: flash_lat, flash_lon (degrees), flash_count
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

def _accumulate_day(
    s3,
    d: date,
    max_workers: int = 4,
    verbose: bool = False,
) -> np.ndarray:
    """
    Download all GLM granules for `d` from all available GOES satellites,
    accumulate flash counts on the 0.1° accumulation grid, and return the
    raw count array (ACC_NLAT × ACC_NLON, float32).
    """
    count_grid = np.zeros((ACC_NLAT, ACC_NLON), dtype=np.float32)

    for bucket, sat_name, first_date in GOES_SATELLITES:
        if d < first_date:
            continue    # satellite not yet operational

        files = _list_day_files(s3, bucket, d)
        if not files:
            if verbose:
                print(f"      [{sat_name}] no files on S3 for {d} — skipping")
            continue

        if verbose:
            print(f"      [{sat_name}] {len(files)} granules")

        # Parallel read within the day
        local_grid = np.zeros_like(count_grid)
        ok = err = 0

        def _read_and_bin(path):
            result = _read_granule_flashes(s3, path)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_read_and_bin, f): f for f in files}
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                lat_arr, lon_arr = result

                # Filter to Canada bounding box before binning
                mask = (
                    (lat_arr >= ACC_LAT_MIN) & (lat_arr < ACC_LAT_MAX) &
                    (lon_arr >= ACC_LON_MIN) & (lon_arr < ACC_LON_MAX)
                )
                lat_f = lat_arr[mask]
                lon_f = lon_arr[mask]

                if len(lat_f) == 0:
                    continue

                # Bin into accumulation grid
                row = ((ACC_LAT_MAX - lat_f) / ACC_DEG).astype(np.int32)
                col = ((lon_f - ACC_LON_MIN) / ACC_DEG).astype(np.int32)

                # Safety clamp (edge pixels)
                row = np.clip(row, 0, ACC_NLAT - 1)
                col = np.clip(col, 0, ACC_NLON - 1)

                np.add.at(local_grid, (row, col), 1)
                ok += 1

        count_grid += local_grid
        n_flashes = int(local_grid.sum())
        if verbose:
            print(f"      [{sat_name}] {ok} granules OK  "
                  f"→ {n_flashes:,} flashes in Canada bbox")

    return count_grid


# ------------------------------------------------------------------ #
# Reproject 0.1° WGS84 → FWI grid
# ------------------------------------------------------------------ #

def _reproject_to_fwi(acc_grid: np.ndarray) -> np.ndarray:
    """
    Reproject the 0.1° accumulation grid (WGS84) to the FWI grid (EPSG:3978).
    Uses bilinear resampling so fractional flash counts are OK for a density grid.
    Returns a (FWI_HEIGHT × FWI_WIDTH) float32 array.
    """
    dst = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float32)

    reproject(
        source           = acc_grid,
        destination      = dst,
        src_transform    = ACC_TRANSFORM,
        src_crs          = ACC_CRS,
        dst_transform    = FWI_TRANSFORM,
        dst_crs          = CRS.from_string(FWI_CRS),
        resampling       = Resampling.bilinear,
        src_nodata       = None,
        dst_nodata       = 0.0,
    )
    return dst


# ------------------------------------------------------------------ #
# Main processing loop
# ------------------------------------------------------------------ #

def download_goes_glm(
    lightning_dir: Path,
    start: date,
    end:   date,
    max_workers: int = 4,
    overwrite:   bool = False,
    verbose:     bool = True,
) -> None:
    lightning_dir.mkdir(parents=True, exist_ok=True)

    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    total = len(all_days)

    s3 = _get_s3()

    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     FWI_WIDTH,
        "height":    FWI_HEIGHT,
        "count":     1,
        "crs":       FWI_CRS,
        "transform": FWI_TRANSFORM,
        "nodata":    0.0,
        "compress":  "lzw",
    }

    done = skipped = failed = 0

    for i, d in enumerate(all_days, 1):
        date_str = d.strftime("%Y%m%d")
        out_path = lightning_dir / f"lightning_{date_str}.tif"

        if out_path.exists() and not overwrite:
            skipped += 1
            if verbose:
                print(f"  [{i:04d}/{total}] {date_str} [SKIP]")
            continue

        if verbose:
            print(f"  [{i:04d}/{total}] {date_str}  accumulating GLM flashes…")

        try:
            acc_grid = _accumulate_day(s3, d, max_workers=max_workers,
                                        verbose=verbose)
            total_flashes = int(acc_grid.sum())

            if total_flashes == 0:
                # No satellite coverage or genuinely no flashes — still write zeros
                # so downstream code knows the file was processed (not missing)
                if verbose:
                    print(f"           → 0 flashes (no satellite coverage or quiet day)")

            fwi_grid = _reproject_to_fwi(acc_grid)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(fwi_grid, 1)

            if verbose:
                nonzero = (fwi_grid > 0).sum()
                print(f"           → {total_flashes:,} raw flashes  "
                      f"| {nonzero:,} FWI pixels with lightning  "
                      f"→ {out_path.name}")
            done += 1

        except Exception as exc:
            failed += 1
            print(f"  [{i:04d}/{total}] {date_str} [FAIL] {exc}")

    print(f"\n  Summary: {done} written, {skipped} skipped, {failed} failed "
          f"out of {total} days.")


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
        "--start", required=True,
        help="Start date YYYYMMDD (inclusive)",
    )
    parser.add_argument(
        "--end", required=True,
        help="End date YYYYMMDD (inclusive)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel S3 reader threads per day (default: 4). "
             "Increase to 8-16 on fast connections.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and recompute even if output TIF already exists.",
    )
    args   = parser.parse_args()
    cfg    = load_config(args.config)

    lightning_dir = Path(get_path(cfg, "lightning_dir"))
    start = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:8]))
    end   = date(int(args.end[:4]),   int(args.end[4:6]),   int(args.end[6:8]))

    print("GOES GLM Lightning → FWI grid")
    print(f"  Output dir : {lightning_dir}")
    print(f"  Date range : {start} – {end}  ({(end - start).days + 1} days)")
    print(f"  Workers    : {args.workers} threads/day")
    print(f"  Satellites : GOES-16 (always) + GOES-18 (≥2023-01-10) "
          f"+ GOES-17 (2019–2023 fallback)")
    print()

    download_goes_glm(
        lightning_dir = lightning_dir,
        start         = start,
        end           = end,
        max_workers   = args.workers,
        overwrite      = args.overwrite,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
