#!/usr/bin/env python3
"""
Download ECMWF S2S (Realtime, Daily averaged) for specific dates.

Two param sets (--param-set):
  core     [default] : tcw / 2t / 2d / sm20 / st20        → s2s_ecmf_cf_YYYY-MM-DD.grib
  extended           : 10u / 10v / tp / cp / sm100         → s2s_ecmf_cf_ext_YYYY-MM-DD.grib
  pressure           : gh500 (geopotential @ 500 hPa)       → s2s_ecmf_cf_pl_YYYY-MM-DD.grib

Use --param-set extended (or pressure) to download supplementary channels needed
for FWI computation and large-scale fire-weather features.  The extended and
pressure sets use separate filenames so existing core downloads are untouched.

Usage:
    # Core (already downloaded):
    python -m src.data_ops.download.s2s_ecmwf --batch

    # Supplement wind + precip + deep soil moisture:
    python -m src.data_ops.download.s2s_ecmwf --batch --param-set extended

    # Supplement 500 hPa geopotential (blocking index):
    python -m src.data_ops.download.s2s_ecmwf --batch --param-set pressure
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from ecmwfapi import ECMWFDataServer

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Core download logic
# ------------------------------------------------------------------ #

# Daily-average step ranges for core set (lead days 14-46, 24h averages)
STEP_STRING_DAILY_AVG = (
    "336-360/360-384/384-408/408-432/432-456/456-480/480-504/"
    "504-528/528-552/552-576/576-600/600-624/624-648/648-672/"
    "672-696/696-720/720-744/744-768/768-792/792-816/816-840/"
    "840-864/864-888/888-912/912-936/936-960/960-984/984-1008/"
    "1008-1032/1032-1056/1056-1080/1080-1104"
)

# Instantaneous steps for extended set (wind/precip not available as daily avg)
# Every 24h from lead day 14 (336h) to day 46 (1104h), 33 steps
STEP_STRING_INSTANT = "/".join(str(h) for h in range(336, 1104 + 1, 24))

# Backward-compatible alias
STEP_STRING = STEP_STRING_DAILY_AVG

# ------------------------------------------------------------------ #
# Param sets
# ------------------------------------------------------------------ #

PARAM_SETS = {
    "core": {
        "levtype": "sfc",
        "param":   "136/167/168/228086/228095",  # tcw/2t/2d/sm20/st20
        "prefix":  "s2s_ecmf_cf_",
        "desc":    "tcw / 2t / 2d / sm20 / st20",
    },
    "extended": {
        "levtype": "sfc",
        # 10u / 10v / cp / tp — wind + precip for FWI computation
        # sm100 (228088) excluded: not available in MARS for S2S
        "param":   "165/166/143/228",
        # Uses default STEP_STRING (daily-average step ranges) — same as core
        "prefix":  "s2s_ecmf_cf_ext_",
        "desc":    "10u / 10v / cp / tp",
    },
    "pressure": {
        "levtype":  "pl",
        "levelist": "500",
        "param":    "129",   # geopotential → gh500 after dividing by g
        "prefix":   "s2s_ecmf_cf_pl_",
        "desc":     "gh500 (geopotential @ 500 hPa)",
    },
}


def download_single_date(server, date_str, outdir, param_set="core"):
    """
    Download ECMWF S2S data for a single date.

    Args:
        server:     ECMWFDataServer instance
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

    req = {
        "class":   "s2",
        "dataset": "s2s",
        "date":    date_str,
        "expver":  "prod",
        "levtype": ps["levtype"],
        "model":   "glob",
        "origin":  "ecmf",
        "param":   ps["param"],
        "step":    ps.get("step", STEP_STRING),
        "stream":  "enfo",
        "time":    "00:00:00",
        "type":    "cf",
        "target":  str(target),
    }
    if "levelist" in ps:
        req["levelist"] = ps["levelist"]

    try:
        print(f"[DOWNLOADING] {date_str} ({param_set}: {ps['desc']}) -> {target.name}")
        server.retrieve(req)

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
        server:     ECMWFDataServer instance
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

    # ---- Load config and credentials from environment ----
    cfg = load_config(args.config)

    ecmwf_email = os.environ.get(
        "ECMWF_EMAIL",
        cfg.get("credentials", {}).get("ecmwf_email", ""),
    )
    ecmwf_key = os.environ.get(
        "ECMWF_KEY",
        cfg.get("credentials", {}).get("ecmwf_key", ""),
    )

    if not ecmwf_email or not ecmwf_key:
        print(
            "ERROR: ECMWF credentials not found.\n"
            "Set ECMWF_EMAIL and ECMWF_KEY environment variables, "
            "or configure them in your YAML config under 'credentials'.",
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

    # ---- Connect to ECMWF ----
    server = ECMWFDataServer(
        url="https://api.ecmwf.int/v1",
        key=ecmwf_key,
        email=ecmwf_email,
    )

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
