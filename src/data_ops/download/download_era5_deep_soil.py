#!/usr/bin/env python3
"""
Download ERA5 deep soil moisture (swvl2, 7-28 cm depth) GRIB files.

Follows the SAME pattern as download_ecmwf_reanalysis_observations.py:
  - Download only — no processing, no reprojection
  - One GRIB file per day (~3 MB), saved to disk
  - Processing (GRIB → daily TIF → FWI grid) is done separately by
    processing/era5_to_daily.py + processing/resample_to_fwi_grid.py

Source: Copernicus CDS reanalysis-era5-single-levels
Variable: volumetric_soil_water_layer_2 (swvl2, 7-28 cm, m^3/m^3)
Output: {deep_soil_dir}/era5_swvl2_YYYY_MM_DD.grib

Usage:
    python -m src.data_ops.download.download_era5_deep_soil 2018-01-01 2025-10-31
    python -m src.data_ops.download.download_era5_deep_soil 2018-01-01 2025-10-31 --workers 2
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Constants (same as download_ecmwf_reanalysis_observations.py)
# ------------------------------------------------------------------ #

DEFAULT_AREA = [83, -141, 41, -52]  # [N, W, S, E] — Canada bounding box
DEFAULT_CDS_API_KEY = "d952a10c-f9c0-4ff3-92e1-aac8756dd123"


def _make_cds_client(api_key: str):
    import cdsapi
    return cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=api_key,
    )


def download_single_date(client, date_str, outdir, area=None, verbose=True):
    """Download one day of swvl2 as GRIB. Returns True on success."""
    safe_date = date_str.replace("-", "_")
    target = outdir / f"era5_swvl2_{safe_date}.grib"

    if target.exists() and target.stat().st_size > 0:
        if verbose:
            print(f"[SKIP] {date_str} - already exists: {target}")
        return True

    if area is None:
        area = DEFAULT_AREA

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    req = {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": "volumetric_soil_water_layer_2",
        "year": date_obj.strftime("%Y"),
        "month": date_obj.strftime("%m"),
        "day": date_obj.strftime("%d"),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
    }

    try:
        if verbose:
            print(f"[DOWNLOADING] {date_str} -> {target}")
        client.retrieve("reanalysis-era5-single-levels", req, str(target))

        if target.exists() and target.stat().st_size > 0:
            if verbose:
                print(f"[SUCCESS] {date_str} - {target.stat().st_size / 1e6:.1f} MB")
            return True
        else:
            if verbose:
                print(f"[ERROR] {date_str} - file missing or empty", file=sys.stderr)
            return False

    except KeyboardInterrupt:
        if target.exists():
            target.unlink()
        raise
    except Exception as e:
        if verbose:
            print(f"[ERROR] {date_str} - {e}", file=sys.stderr)
        if target.exists():
            target.unlink()
        return False


def download_with_retries(date_str, outdir, area, retries, retry_wait,
                          cds_api_key, client=None, verbose=True):
    """Download one date with retries. Returns (date_str, success)."""
    local_client = client or _make_cds_client(cds_api_key)
    for attempt in range(1, retries + 1):
        ok = download_single_date(local_client, date_str, outdir,
                                  area=area, verbose=verbose)
        if ok:
            return date_str, True
        if attempt < retries:
            if verbose:
                print(f"[RETRY] {date_str} attempt {attempt + 1}/{retries} "
                      f"after {retry_wait}s")
            time.sleep(max(0, retry_wait))
    return date_str, False


def generate_date_list(start_date, end_date):
    """Generate list of YYYY-MM-DD strings between start and end (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 deep soil moisture (swvl2) GRIB files"
    )
    parser.add_argument("start_date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("end_date", type=str, help="End date YYYY-MM-DD")
    add_config_argument(parser)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel download workers (default: 1)")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries per date (default: 2)")
    parser.add_argument("--wait", type=int, default=2,
                        help="Seconds between retries (default: 2)")
    parser.add_argument("--cds-api-key", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        outdir = Path(get_path(cfg, "deep_soil_dir"))
    except (KeyError, TypeError):
        outdir = Path("data/era5_deep_soil")
    outdir.mkdir(parents=True, exist_ok=True)

    dates = generate_date_list(args.start_date, args.end_date)

    api_key = (args.cds_api_key
               or os.environ.get("CDS_API_KEY")
               or cfg.get("credentials", {}).get("cds_api_key", "")
               or DEFAULT_CDS_API_KEY)

    print("=" * 70)
    print("ERA5 Deep Soil Moisture (swvl2) — Download Only")
    print("=" * 70)
    print(f"  Dates   : {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"  Output  : {outdir}")
    print(f"  Format  : GRIB (one file per day, ~3 MB each)")
    print(f"  Workers : {args.workers}")
    print()
    print("  NOTE: After download, run processing/era5_to_daily.py")
    print("        then processing/resample_to_fwi_grid.py")
    print("=" * 70)

    success_count = 0
    fail_count = 0
    area = DEFAULT_AREA
    retries = max(1, args.retries)

    client = _make_cds_client(api_key)
    for i, d in enumerate(dates, 1):
        _, ok = download_with_retries(
            d, outdir, area=area, retries=retries,
            retry_wait=args.wait, cds_api_key=api_key,
            client=client, verbose=True,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

        if i % 30 == 0 or i == len(dates):
            print(f"\n  Progress: {i}/{len(dates)}  "
                  f"ok={success_count}  fail={fail_count}\n")

    print(f"\n[COMPLETE] ok={success_count}  fail={fail_count}  "
          f"output={outdir}")


if __name__ == "__main__":
    main()
