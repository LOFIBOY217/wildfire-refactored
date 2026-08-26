#!/usr/bin/env python3
"""
Download ECMWF S2S (Realtime) control forecasts for specific dates.

Data source: the ECMWF Data Store (ECDS, https://ecds.ecmwf.int), collection
``s2s-forecasts``. S2S was migrated here off the retired ECMWF Web-API
(api.ecmwf.int/v1) in 2026; this script uses the `cdsapi` client against ECDS.

Prerequisites (one-time, per user):
  1. Create an ECMWF account and log in at https://ecds.ecmwf.int.
  2. Accept the S2S dataset licence on the dataset's Download tab.
  3. Put your personal access token in ~/.cdsapirc (the token is the same one
     used for CDS/ADS/CEMS), or set ECDS_API_KEY / CDS_API_KEY. See
     https://ecds.ecmwf.int/how-to-api.

Three param sets (--param-set):
  core     [default] : tcw / 2t / 2d / sm20 / st20   → s2s_ecmf_cf_YYYY-MM-DD.grib
                       (daily-averaged single-level fields)
  extended           : 10u / 10v / cp / tp           → s2s_ecmf_cf_ext_YYYY-MM-DD.grib
                       (instantaneous/accumulated single-level fields)
  pressure           : gh500 (geopotential @ 500 hPa) → s2s_ecmf_cf_pl_YYYY-MM-DD.grib

Use --param-set extended (or pressure) to download supplementary channels needed
for FWI computation and large-scale fire-weather features. The extended and
pressure sets use separate filenames so existing core downloads are untouched.

Usage:
    # Core (daily-averaged) for one date:
    python -m src.data_ops.download.download_ecmwf_s2s 2023-05-01

    # Batch all Mon/Thu issue dates in a range:
    python -m src.data_ops.download.download_ecmwf_s2s --batch \\
        --batch-start 2023-05-01 --batch-end 2023-05-31 --mon-thu-only

    # Supplement wind + precip:
    python -m src.data_ops.download.download_ecmwf_s2s 2023-05-01 --param-set extended

Maintenance (if this breaks, the ECDS collection or request form changed):
    Source of truth : https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download
                      (open the Download tab, "Show API request code" = ground truth
                      for variable names / leadtime_hour values)
    Last verified   : 2026-08
    See docs/DATA_SOURCES.md for the full endpoint registry and the
    "what to check when a downloader breaks" playbook.
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
    import sys
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument
    from src.data_ops.download._common import make_ecds_client

# ECDS collection id for the S2S real-time forecasts.
S2S_COLLECTION = "s2s-forecasts"


# ------------------------------------------------------------------ #
# Core download logic
# ------------------------------------------------------------------ #

# ECDS leadtime_hour values, lead days 14-46 (336h-1104h).
#  - Daily-averaged fields use 24h-mean RANGE strings "<start>_<end>" (24h stride).
#  - Instantaneous/accumulated fields use point-step strings (24h stride here).
# Both subsets exist in the ECDS s2s-forecasts form; values verified against it.
LEADTIME_DAILY_AVG = [f"{h}_{h + 24}" for h in range(336, 1080 + 1, 24)]  # 336_360 .. 1080_1104
LEADTIME_INSTANT   = [str(h) for h in range(336, 1104 + 1, 24)]           # "336" .. "1104"

# ------------------------------------------------------------------ #
# Param sets  (ECDS s2s-forecasts request templates)
# ------------------------------------------------------------------ #
# Variable names, level_type, forecast_type and leadtime_hour values are the
# exact ECDS API strings (from the s2s-forecasts Download form). A request must
# not mix daily-averaged and instantaneous variables — they take different
# leadtime families — so each set is internally consistent.

PARAM_SETS = {
    "core": {
        "level_type": "single_level",
        "variable": [
            "total_column_water",          # tcw
            "2_m_temperature",             # 2t
            "2_m_dewpoint_temperature",    # 2d
            "soil_moisture_top_20_cm",     # sm20
            "soil_temperature_top_20_cm",  # st20
        ],
        "leadtime_hour": LEADTIME_DAILY_AVG,
        "prefix": "s2s_ecmf_cf_",
        "desc":   "tcw / 2t / 2d / sm20 / st20 (daily averaged)",
    },
    "extended": {
        "level_type": "single_level",
        "variable": [
            "10_m_u_component_of_wind",    # 10u
            "10_m_v_component_of_wind",    # 10v
            "convective_precipitation",    # cp
            "total_precipitation",         # tp
        ],
        "leadtime_hour": LEADTIME_INSTANT,
        "prefix": "s2s_ecmf_cf_ext_",
        "desc":   "10u / 10v / cp / tp (instantaneous/accumulated)",
    },
    "pressure": {
        "level_type": "pressure",
        "level_value": ["500"],
        "variable": ["geopotential_height"],   # gh500 (blocking index)
        "leadtime_hour": LEADTIME_INSTANT,
        "prefix": "s2s_ecmf_cf_pl_",
        "desc":   "gh500 (geopotential @ 500 hPa)",
    },
}


def _build_request(date_str, ps):
    """Build the ECDS s2s-forecasts request dict for one issue date + param set."""
    year, month, day = date_str.split("-")
    req = {
        "origin":        "ecmwf",
        "forecast_type": "control_forecast",
        "level_type":    ps["level_type"],
        "variable":      ps["variable"],
        "year":          year,
        "month":         month,
        "day":           day,
        "time":          "00:00",
        "leadtime_hour": ps["leadtime_hour"],
        "data_format":   "grib",
    }
    if "level_value" in ps:
        req["level_value"] = ps["level_value"]
    return req


def download_single_date(server, date_str, outdir, param_set="core"):
    """
    Download ECMWF S2S data for a single date.

    Args:
        server:     cdsapi client for ECDS (from make_ecds_client)
        date_str:   Date in YYYY-MM-DD format
        outdir:     Output directory (Path)
        param_set:  One of 'core', 'extended', 'pressure'

    Returns:
        True on success, False on failure.
    """
    ps = PARAM_SETS[param_set]
    safe_date = date_str.replace("/", "_")
    target = outdir / f"{ps['prefix']}{safe_date}.grib"

    # Skip if already downloaded
    if target.exists() and target.stat().st_size > 0:
        print(f"[SKIP] {date_str} ({param_set}) - already exists: {target}")
        return True

    req = _build_request(date_str, ps)

    try:
        print(f"[DOWNLOADING] {date_str} ({param_set}: {ps['desc']}) -> {target.name}")
        server.retrieve(S2S_COLLECTION, req, str(target))

        if target.exists() and target.stat().st_size > 0:
            print(f"[SUCCESS] {date_str} ({param_set}) - {target.stat().st_size / 1e6:.1f} MB")
            return True
        else:
            print(f"[ERROR] {date_str} ({param_set}) - file missing or empty", file=sys.stderr)
            return False

    except KeyboardInterrupt:
        print(f"\n[CANCELLED] {date_str} ({param_set}) - partial file: {target}")
        raise
    except Exception as e:
        print(f"[ERROR] {date_str} ({param_set}) - {e}", file=sys.stderr)
        return False


# ------------------------------------------------------------------ #
# Date utilities
# ------------------------------------------------------------------ #

def generate_date_list(start_date, end_date, mon_thu_only=False):
    """Generate list of date strings between *start_date* and *end_date* (inclusive).

    Args:
        mon_thu_only: If True, restrict to Mondays and Thursdays only (older
                      ECMWF S2S schedule, pre ~2023). Default False = try every
                      day and let the API skip non-issue dates automatically.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        if not mon_thu_only or current.weekday() in (0, 3):  # 0=Mon, 3=Thu
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


# ------------------------------------------------------------------ #
# Batch download loop
# ------------------------------------------------------------------ #

def download_batch(server, dates, outdir, wait_time=5, param_set="core"):
    """
    Download a list of dates with progress reporting and rate limiting.

    Args:
        server:     cdsapi client for ECDS (from make_ecds_client)
        dates:      List of date strings
        outdir:     Output directory (Path)
        wait_time:  Seconds to sleep between requests
        param_set:  One of 'core', 'extended', 'pressure'

    Returns:
        Tuple (success_count, fail_count, failed_dates)
    """
    success_count = 0
    fail_count = 0
    failed_dates = []

    try:
        for i, date in enumerate(dates, 1):
            print(f"\n{'='*60}")
            print(f"Progress: {i}/{len(dates)} ({i/len(dates)*100:.1f}%)")
            print(f"{'='*60}")

            success = download_single_date(server, date, outdir, param_set=param_set)

            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_dates.append(date)

            # Rate limiting
            if i < len(dates):
                print(f"Waiting {wait_time}s before next download...")
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Download cancelled by user")

    finally:
        # Summary
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Total dates:    {len(dates)}")
        print(f"Successful:     {success_count}")
        print(f"Failed:         {fail_count}")

        if failed_dates:
            print("\nFailed dates:")
            for date in failed_dates:
                print(f"  - {date}")

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
        description="Download ECMWF S2S realtime forecasts (control, sfc)"
    )
    add_config_argument(parser)

    parser.add_argument(
        "dates", nargs="*",
        help=(
            "One date (YYYY-MM-DD), two dates (start end) for a range, "
            "or omit when using --batch"
        ),
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Download all S2S issue dates (Mon/Thu) from batch-start to batch-end",
    )
    parser.add_argument(
        "--batch-start", type=str, default="2017-01-01",
        help="Batch start date (default: 2017-01-01)",
    )
    parser.add_argument(
        "--batch-end", type=str, default=datetime.today().strftime("%Y-%m-%d"),
        help="Batch end date (default: today)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Override output directory (default: s2s_dir from config)",
    )
    parser.add_argument(
        "--wait", type=int, default=5,
        help="Seconds to wait between requests (default: 5)",
    )
    parser.add_argument(
        "--mon-thu-only", action="store_true",
        help="Only request Mondays and Thursdays (pre-2023 ECMWF S2S schedule)",
    )
    parser.add_argument(
        "--param-set", type=str, default="core",
        choices=list(PARAM_SETS.keys()),
        help=(
            "Which variable set to download (default: core). "
            "'core' = tcw/2t/2d/sm20/st20 (already downloaded); "
            "'extended' = 10u/10v/cp/tp/sm100 (wind+precip+deep soil); "
            "'pressure' = gh500 (500 hPa geopotential, blocking index)."
        ),
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

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
            "at https://ecds.ecmwf.int/how-to-api and accept the S2S licence at "
            "https://ecds.ecmwf.int/datasets/s2s-forecasts.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Resolve output directory ----
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        s2s_dir = cfg.get("paths", {}).get("s2s_dir") or get_path(cfg, "ecmwf_dir")
        outdir = Path(s2s_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Determine date list ----
    mon_thu_only = getattr(args, "mon_thu_only", False)
    if args.batch:
        dates = generate_date_list(args.batch_start, args.batch_end, mon_thu_only=mon_thu_only)
        print(f"[BATCH MODE] {len(dates)} dates: "
              f"{args.batch_start} to {args.batch_end}"
              f"{' (Mon/Thu only)' if mon_thu_only else ''}\n")
    elif len(args.dates) == 2:
        start_date, end_date = args.dates
        dates = generate_date_list(start_date, end_date, mon_thu_only=mon_thu_only)
        print(f"[RANGE MODE] {len(dates)} dates: "
              f"{start_date} to {end_date}\n")
    elif len(args.dates) == 1:
        dates = [args.dates[0]]
        print(f"[SINGLE MODE] Will download 1 date: {dates[0]}\n")
    else:
        parser.print_help()
        sys.exit(2)

    # ---- Connect to ECDS ----
    # cdsapi client pointed at the ECDS API root (see make_ecds_client). Passing
    # an empty key lets cdsapi read the token from ~/.cdsapirc.
    server = make_ecds_client(ecds_key or None)

    param_set = args.param_set
    ps = PARAM_SETS[param_set]
    print(f"[PARAM SET] {param_set}: {ps['desc']}  (prefix: {ps['prefix']})")

    # ---- Download ----
    if len(dates) == 1:
        # Single date: match original single-date script behaviour (exit codes)
        try:
            print(f"Requesting {dates[0]} ({param_set}) -> {outdir}")
            success = download_single_date(server, dates[0], outdir, param_set=param_set)
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            prefix = ps["prefix"]
            target = outdir / f"{prefix}{dates[0].replace('/', '_')}.grib"
            print(f"\nCancelled by user. Partial file (if any): {target}")
            sys.exit(130)
    else:
        _, fail_count, _ = download_batch(
            server, dates, outdir, wait_time=args.wait, param_set=param_set
        )
        sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
