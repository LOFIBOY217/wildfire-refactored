"""
Download ECMWF S2S Fire Danger Seasonal Forecast (SEAS5-driven FWI) from
the Copernicus CDS, restricted to Canada.

This is the operational baseline for our paper: at each model issue date
t in the validation period (2022-05 to 2025-10), the SEAS5 seasonal
forecast issued on the most recent month-start gives us a 1-7 month
forecast of FWI. We compare our model against this product at lead 14-46.

Dataset: cems-fire-seasonal  (operational forecasts 2017-present, on EWDS)
  - SEAS5 ensemble mean (or members) drives the FWI calculation
  - Monthly issue dates (1st of each month)
  - Lead times: 1-215 days (~7 months)
  - Spatial resolution: 1° lat/lon (SEAS5 native ~1° on O320 reduced Gaussian)
  - Variables: fwinx (FWI), ffmc, dc, dmc, bui, isi
  - Hosted on EWDS (Early Warning Data Store) at ewds.climate.copernicus.eu,
    NOT the standard CDS at cds.climate.copernicus.eu.

  For hindcasts pre-2017, use dataset 'cems-fire-seasonal-reforecast' instead.

Usage:
  # Local download (login node, no GPU needed):
  python -m src.data_ops.download.download_ecmwf_s2s_fire \\
      --start_year 2022 --end_year 2025 \\
      --output_dir data/ecmwf_s2s_fire \\
      --workers 2 \\
      --variables fwinx
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Canada bounding box (N, W, S, E) — same as fwi_historical.py
CANADA_AREA = [85, -141, 41, -52]

# Variable name -> CDS variable code mapping
S2S_VARIABLES = {
    "fwinx": "fire_weather_index",
    "ffmc": "fine_fuel_moisture_code",
    "dc": "drought_code",
    "dmc": "duff_moisture_code",
    "bui": "buildup_index",
    "isi": "initial_spread_index",
}

# SEAS5 forecasts at 12:00 UTC each day. lead day k → leadtime_hour =
# 12 + (k - 1) * 24, for k = 1..215. Valid enum range: 12, 36, ..., 5148.
ALL_LEADTIMES = [str(12 + (k - 1) * 24) for k in range(1, 216)]


def download_one_issue(client, year, month, var, output_dir, area=None):
    """
    Download one (issue_year, issue_month, variable) from CDS.

    Each forecast issue has all leadtimes 1-215 in one NetCDF file.
    """
    area = area or CANADA_AREA
    yr = str(year)
    mo = f"{int(month):02d}"

    var_dir = output_dir / var
    var_dir.mkdir(parents=True, exist_ok=True)
    # Switch from .nc to .grib — the EWDS API delivers GRIB by default and
    # NetCDF post-conversion is sometimes truncated for multi-leadtime asks.
    outfile = var_dir / f"s2s_{var}_{yr}{mo}.grib"

    if outfile.exists():
        sz = outfile.stat().st_size / 1024 / 1024
        if sz > 0.1:                 # > 100 KB = real file
            print(f"  [SKIP] {outfile.name} ({sz:.1f} MB) already exists")
            return outfile
        else:
            outfile.unlink()         # tiny file = corrupt; redownload

    # cems-fire-seasonal schema (verified 2026-05-02 against EWDS):
    #   release_version: "5"   (NOT "system")
    #   day: "01" (only valid value — monthly issue)
    #   leadtime_hour: 12, 36, ..., 5148 (12 + (k-1)*24 for lead day k)
    #   no originating_centre, no product_type
    request = {
        "release_version": "5",
        "variable": S2S_VARIABLES[var],
        "year": yr,
        "month": mo,
        "day": "01",
        "leadtime_hour": ALL_LEADTIMES,
        "area": area,
        "data_format": "grib",
    }

    print(f"  [DOWNLOAD] {var} {yr}-{mo} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        # Dataset name "cems-fire-seasonal" — operational SEAS5 forecasts
        # 2017-present. For pre-2017 hindcasts use "cems-fire-seasonal-reforecast".
        client.retrieve("cems-fire-seasonal", request, str(outfile))
        elapsed = time.time() - t0
        size_mb = outfile.stat().st_size / 1024 / 1024
        print(f"[OK] {size_mb:.1f} MB in {elapsed:.0f} s")
        return outfile
    except Exception as e:
        print(f"[FAIL] {e}")
        if outfile.exists():
            outfile.unlink()
        return None


def _worker(task):
    year, month, var, output_dir, area, client_kwargs = task
    try:
        import cdsapi
        client = cdsapi.Client(**client_kwargs, quiet=True)
        return download_one_issue(client, year, month, var, output_dir, area)
    except Exception as e:
        print(f"  [WORKER ERROR] {var} {year}-{month}: {e}")
        return None


def download_range(client_kwargs, start_year, end_year, months, variables,
                   output_dir, area=None, workers=2):
    """Threaded parallel download across (year × month × variable)."""
    tasks = [
        (year, month, var, output_dir, area, client_kwargs)
        for year in range(start_year, end_year + 1)
        for month in months
        for var in variables
    ]
    total = len(tasks)
    print(f"  Parallel download: {total} (year-month-var) requests, "
          f"{workers} workers")

    files = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            yr, mo, var = futures[fut][0], futures[fut][1], futures[fut][2]
            result = fut.result()
            status = "OK" if result else "FAIL"
            print(f"  [{done}/{total}] {var} {yr}-{mo:02d} [{status}]")
            if result:
                files.append(result)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(
        description="Download ECMWF S2S Fire Danger forecast from CDS")
    ap.add_argument("--start_year", type=int, default=2022)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--months", type=int, nargs="+",
                    default=[4, 5, 6, 7, 8, 9, 10],
                    help="Issue months (default = fire season Apr-Oct)")
    ap.add_argument("--variables", type=str, nargs="+",
                    default=["fwinx"],
                    choices=list(S2S_VARIABLES.keys()),
                    help="Which fire indices to download (default = FWI only)")
    ap.add_argument("--output_dir", type=str,
                    default="data/ecmwf_s2s_fire")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cds_url", type=str,
                    default=os.environ.get("CDS_API_URL",
                                           "https://ewds.climate.copernicus.eu/api"),
                    help="EWDS API endpoint (CEMS-fire datasets live on "
                         "ewds.climate.copernicus.eu, NOT cds.* — but the "
                         "same CDS_API_KEY works on both).")
    ap.add_argument("--cds_key", type=str,
                    default=os.environ.get("CDS_API_KEY"))
    ap.add_argument("--area", type=float, nargs=4, default=None,
                    metavar=("N", "W", "S", "E"),
                    help="Bounding box; default Canada [85,-141,41,-52]")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    area = list(args.area) if args.area else CANADA_AREA

    print("="*60)
    print("ECMWF S2S Fire Danger download")
    print("="*60)
    print(f"  Years:      {args.start_year} - {args.end_year}")
    print(f"  Months:     {args.months}")
    print(f"  Variables:  {args.variables}")
    print(f"  Area:       {area}")
    print(f"  Output:     {output_dir.absolute()}")
    print(f"  Workers:    {args.workers}")

    try:
        import cdsapi  # noqa: F401
    except ImportError:
        print("\n[ERROR] cdsapi not installed. Run: pip install cdsapi")
        sys.exit(1)

    client_kwargs = {}
    if args.cds_url:
        client_kwargs["url"] = args.cds_url
    if args.cds_key:
        client_kwargs["key"] = args.cds_key

    if not client_kwargs.get("key"):
        print("\n[WARN] No CDS_API_KEY set; falling back to ~/.cdsapirc")

    t0 = time.time()
    files = download_range(
        client_kwargs,
        args.start_year, args.end_year, args.months,
        args.variables, output_dir, area, args.workers
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min — {len(files)} files saved.")


if __name__ == "__main__":
    main()
