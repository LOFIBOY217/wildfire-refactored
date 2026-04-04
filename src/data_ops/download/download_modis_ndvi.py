#!/usr/bin/env python3
"""
Download MODIS MOD13A2 NDVI/EVI HDF4 granules for Canada.

Download only — saves raw HDF4 files to disk.
Processing (tile mosaic, reproject, daily interpolation) is done separately
by processing/process_modis_ndvi.py.

Product: MOD13A2 v6.1 (Terra MODIS, 16-day composite, 1 km, sinusoidal)
Source: NASA Earthdata LAADS — free, requires NASA Earthdata account.
Output: {ndvi_raw_dir}/{YYYY}/MOD13A2.*.hdf  (raw HDF4 files, organized by year)

Usage:
    python -m src.data_ops.download.download_modis_ndvi --start_year 2018 --end_year 2024
    python -m src.data_ops.download.download_modis_ndvi --start_year 2023 --end_year 2023

Prerequisites:
    pip install earthaccess
    Set env vars: EARTHDATA_USERNAME, EARTHDATA_PASSWORD
"""

import argparse
import sys
from pathlib import Path

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


CANADA_BBOX = (-142.0, 41.0, -52.0, 84.0)  # (W, S, E, N)


def main():
    parser = argparse.ArgumentParser(
        description="Download MODIS MOD13A2 HDF4 granules (download only)"
    )
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--months", type=int, nargs="+",
                        default=[4, 5, 6, 7, 8, 9, 10, 11])
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Granules per download batch (default: 50)")
    args = parser.parse_args()

    try:
        import earthaccess
    except ImportError:
        print("earthaccess required: pip install earthaccess", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    ndvi_dir = Path(get_path(cfg, "ndvi_dir"))
    raw_dir = ndvi_dir.parent / "ndvi_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODIS MOD13A2 NDVI/EVI — Download Only (raw HDF4)")
    print("=" * 70)
    print(f"  Years   : {args.start_year} – {args.end_year}")
    print(f"  Output  : {raw_dir}/YYYY/*.hdf")
    print(f"  NOTE: Run processing/process_modis_ndvi.py after download")
    print("=" * 70)

    print("\n  Authenticating with NASA Earthdata…")
    earthaccess.login(strategy="environment")

    for year in range(args.start_year, args.end_year + 1):
        year_dir = raw_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # Check if year already downloaded
        existing = list(year_dir.glob("*.hdf"))
        if len(existing) > 100:
            print(f"\n  [{year}] {len(existing)} HDF4 files exist — skipping")
            continue

        print(f"\n  [{year}] Searching MOD13A2 composites…")
        results = earthaccess.search_data(
            short_name="MOD13A2",
            version="061",
            bounding_box=CANADA_BBOX,
            temporal=(f"{year}-01-01", f"{year}-12-31"),
        )
        print(f"  [{year}] Found {len(results)} granules")

        if not results:
            continue

        # Download in batches
        batch = args.batch_size
        for bi in range(0, len(results), batch):
            chunk = results[bi:bi + batch]
            print(f"  [{year}] Downloading batch {bi//batch+1}/"
                  f"{(len(results)+batch-1)//batch}: {len(chunk)} files…")
            earthaccess.download(chunk, local_path=str(year_dir))

        final = list(year_dir.glob("*.hdf"))
        print(f"  [{year}] Done: {len(final)} HDF4 files in {year_dir}")

    print("\n[COMPLETE]")


if __name__ == "__main__":
    main()
