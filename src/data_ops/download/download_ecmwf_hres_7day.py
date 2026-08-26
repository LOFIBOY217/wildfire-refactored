#!/usr/bin/env python3
"""
Download ECMWF TIGGE 7-day ensemble control forecasts for a date range.
=======================================================================
TIGGE (THORPEX Interactive Grand Global Ensemble) contains real historical
forecast values (not reanalysis), making it suitable as a 7-day decoder input
for comparison experiments with the logistic baseline.

Data source: the ECMWF Data Store (ECDS, https://ecds.ecmwf.int), collection
``tigge-forecasts``. TIGGE was migrated here off the retired ECMWF Web-API
(api.ecmwf.int/v1) in 2026; this script uses the `cdsapi` client against ECDS.

    S2S   (download_ecmwf_s2s.py)       : 14-46 day forecasts -> data/ecmwf_s2s/
    TIGGE (download_ecmwf_hres_7day.py) : 1-7   day forecasts -> data/ecmwf_hres/

Request (ECDS tigge-forecasts):
    origin        : ecmwf
    forecast_type : control_forecast
    level_type    : single_level
    leadtime_hour : 24/48/72/96/120/144/168  (days 1-7)
    variable      : tcw / 2t / 2d / sm20 / st20

Prerequisites (one-time, per user):
  1. Create an ECMWF account and log in at https://ecds.ecmwf.int.
  2. Accept the TIGGE dataset licence on the dataset's Download tab.
  3. Put your personal access token in ~/.cdsapirc (same token as CDS/ADS/CEMS),
     or set ECDS_API_KEY / CDS_API_KEY. See https://ecds.ecmwf.int/how-to-api.

Output files:
    data/ecmwf_hres/tigge_ecmf_<YYYY-MM-DD>.grib

Usage:
    Single date:   python -m src.data_ops.download.download_ecmwf_hres_7day 2023-04-28
    Date range:    python -m src.data_ops.download.download_ecmwf_hres_7day 2023-04-28 2025-08-21
    Batch mode:    python -m src.data_ops.download.download_ecmwf_hres_7day --batch
                   python -m src.data_ops.download.download_ecmwf_hres_7day --batch-start 2023-04-28 --batch-end 2025-08-21
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from src.config import load_config, get_path, add_config_argument
    from src.data_ops.download._common import make_ecds_client
except ModuleNotFoundError:
    from pathlib import Path as _Path
    for _parent in _Path(__file__).resolve().parents:
        if (_parent / "src" / "config.py").exists():
            sys.path.insert(0, str(_parent))
            break
    from src.config import load_config, get_path, add_config_argument
    from src.data_ops.download._common import make_ecds_client


# ------------------------------------------------------------------ #
# ECDS tigge-forecasts request constants
# ------------------------------------------------------------------ #

# ECDS collection id for the TIGGE forecasts.
TIGGE_COLLECTION = "tigge-forecasts"

# Leadtime hours: 24h to 168h (day 1 through day 7), point steps.
LEADTIME_HOUR = ["24", "48", "72", "96", "120", "144", "168"]

# Single-level variables (exact ECDS API names); same fields as the S2S core set.
VARIABLES = [
    "total_column_water",          # tcw
    "2_m_temperature",             # 2t
    "2_m_dewpoint_temperature",    # 2d
    "soil_moisture_top_20_cm",     # sm20
    "soil_temperature_top_20_cm",  # st20
]

# Canada bounding box [North, West, South, East].
AREA_CANADA = [83, -141, 41, -52]


# ------------------------------------------------------------------ #
# Core download logic
# ------------------------------------------------------------------ #

def download_single_date(server, date_str, outdir):
    """
    Download ECMWF TIGGE 7-day control forecast for a single date.

    Args:
        server:   cdsapi client for ECDS (from make_ecds_client)
        date_str: Forecast initialisation date, YYYY-MM-DD
        outdir:   Output directory (Path)

    Returns:
        True on success, False on failure.
    """
    target = outdir / f"tigge_ecmf_{date_str}.grib"

    if target.exists() and target.stat().st_size > 0:
        print(f"[SKIP] {date_str} - already exists: {target}")
        return True

    year, month, day = date_str.split("-")
    req = {
        "origin":        "ecmwf",
        "forecast_type": "control_forecast",
        "level_type":    "single_level",
        "variable":      VARIABLES,
        "year":          year,
        "month":         month,
        "day":           day,
        "time":          "00:00",
        "leadtime_hour": LEADTIME_HOUR,
        "area":          AREA_CANADA,
        "data_format":   "grib",
    }

    try:
        print(f"[DOWNLOADING] {date_str} -> {target}")
        server.retrieve(TIGGE_COLLECTION, req, str(target))

        if target.exists() and target.stat().st_size > 0:
            print(f"[SUCCESS] {date_str} - {target.stat().st_size / 1e6:.1f} MB")
            return True
        else:
            print(f"[ERROR] {date_str} - file missing or empty", file=sys.stderr)
            return False

    except KeyboardInterrupt:
        print(f"\n[CANCELLED] {date_str} - partial file: {target}")
        raise
    except Exception as e:
        print(f"[ERROR] {date_str} - {e}", file=sys.stderr)
        return False


# ------------------------------------------------------------------ #
# Date utilities
# ------------------------------------------------------------------ #

def generate_date_list(start_date, end_date):
    """Return list of YYYY-MM-DD strings from start to end (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    dates = []
    cur   = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


# ------------------------------------------------------------------ #
# Batch download loop
# ------------------------------------------------------------------ #

def download_batch(server, dates, outdir, wait_time=5):
    """
    Download a list of dates with progress reporting and rate limiting.

    Returns:
        Tuple (success_count, fail_count, failed_dates)
    """
    success_count = 0
    fail_count    = 0
    failed_dates  = []

    try:
        for i, date in enumerate(dates, 1):
            print(f"\n{'='*60}")
            print(f"Progress: {i}/{len(dates)} ({i/len(dates)*100:.1f}%)")
            print(f"{'='*60}")

            success = download_single_date(server, date, outdir)

            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_dates.append(date)

            if i < len(dates):
                print(f"Waiting {wait_time}s before next download...")
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Download cancelled by user")

    finally:
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Total dates:    {len(dates)}")
        print(f"Successful:     {success_count}")
        print(f"Failed:         {fail_count}")

        if failed_dates:
            print("\nFailed dates:")
            for d in failed_dates:
                print(f"  - {d}")
            fail_file = outdir / "failed_downloads.txt"
            with open(fail_file, "w") as f:
                f.write("\n".join(failed_dates))
            print(f"\nFailed dates saved to: {fail_file}")

    return success_count, fail_count, failed_dates


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Download ECMWF TIGGE 7-day control forecasts (sfc, 0.5°, Canada)"
    )
    add_config_argument(parser)

    parser.add_argument(
        "dates", nargs="*",
        help=(
            "One date (YYYY-MM-DD), or two dates (start end) for a range. "
            "Omit when using --batch."
        ),
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Download the full default date range (2023-04-28 to 2025-08-21)",
    )
    parser.add_argument(
        "--batch-start", type=str, default="2023-04-28",
        help="Override batch start date (default: 2023-04-28)",
    )
    parser.add_argument(
        "--batch-end", type=str, default="2025-08-21",
        help="Override batch end date (default: 2025-08-21)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Override output directory (default: data/ecmwf_hres from config)",
    )
    parser.add_argument(
        "--wait", type=int, default=5,
        help="Seconds to wait between requests (default: 5)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ---- Load config and credentials ----
    cfg = load_config(args.config)

    # ECDS access token: ECDS_API_KEY / CDS_API_KEY env, then config, else
    # cdsapi falls back to ~/.cdsapirc. The token is the unified ECMWF one.
    ecds_key = (
        os.environ.get("ECDS_API_KEY")
        or os.environ.get("CDS_API_KEY")
        or cfg.get("credentials", {}).get("ecds_api_key", "")
        or cfg.get("credentials", {}).get("cds_api_key", "")
    )
    cdsapirc = Path.home() / ".cdsapirc"
    if not ecds_key and not cdsapirc.exists():
        print(
            "ERROR: ECDS credentials not found.\n"
            "Set ECDS_API_KEY (or CDS_API_KEY), add it to your YAML config under "
            "'credentials', or create ~/.cdsapirc. Get a personal access token "
            "at https://ecds.ecmwf.int/how-to-api and accept the TIGGE licence at "
            "https://ecds.ecmwf.int/datasets/tigge-forecasts.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Resolve output directory ----
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        project_root = Path(get_path(cfg, "fwi_dir")).parent.parent
        outdir = project_root / "data" / "ecmwf_hres"

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # ---- Determine date list ----
    if args.batch:
        dates = generate_date_list(args.batch_start, args.batch_end)
        print(f"[BATCH MODE] {len(dates)} dates: {args.batch_start} to {args.batch_end}\n")
    elif len(args.dates) == 2:
        dates = generate_date_list(args.dates[0], args.dates[1])
        print(f"[RANGE MODE] {len(dates)} dates: {args.dates[0]} to {args.dates[1]}\n")
    elif len(args.dates) == 1:
        dates = [args.dates[0]]
        print(f"[SINGLE MODE] 1 date: {dates[0]}\n")
    else:
        parser.print_help()
        sys.exit(2)

    # ---- Connect to ECDS ----
    # cdsapi client pointed at the ECDS API root (see make_ecds_client). Passing
    # an empty key lets cdsapi read the token from ~/.cdsapirc.
    server = make_ecds_client(ecds_key or None)

    # ---- Download ----
    if len(dates) == 1:
        try:
            success = download_single_date(server, dates[0], outdir)
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            sys.exit(130)
    else:
        _, fail_count, _ = download_batch(server, dates, outdir, wait_time=args.wait)
        sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
