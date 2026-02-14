#!/usr/bin/env python3
"""
Download ECMWF S2S (Realtime, Daily averaged, levtype=sfc) for specific dates.
Defaults: params=2t/2d/sm20/st20/tcw, step=336-1104 by 24h, control (cf).

Merges the single-date and batch download scripts into one unified CLI.

Usage:
    Single date:    python -m src.data_ops.download.s2s_ecmwf 2025-01-26
    Date range:     python -m src.data_ops.download.s2s_ecmwf 2025-01-26 2025-11-15
    Batch mode:     python -m src.data_ops.download.s2s_ecmwf --batch
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

STEP_STRING = (
    "336-360/360-384/384-408/408-432/432-456/456-480/480-504/"
    "504-528/528-552/552-576/576-600/600-624/624-648/648-672/"
    "672-696/696-720/720-744/744-768/768-792/792-816/816-840/"
    "840-864/864-888/888-912/912-936/936-960/960-984/984-1008/"
    "1008-1032/1032-1056/1056-1080/1080-1104"
)


def download_single_date(server, date_str, outdir):
    """
    Download ECMWF S2S data for a single date.

    Args:
        server: ECMWFDataServer instance
        date_str: Date in YYYY-MM-DD format (or YYYY-MM-DD/to/YYYY-MM-DD)
        outdir: Output directory (Path)

    Returns:
        True on success, False on failure.
    """
    safe_date = date_str.replace("/", "_")
    target = outdir / f"s2s_ecmf_cf_{safe_date}.grib"

    # Skip if already downloaded
    if target.exists() and target.stat().st_size > 0:
        print(f"[SKIP] {date_str} - file already exists: {target}")
        return True

    req = {
        "class":   "s2",
        "dataset": "s2s",
        "date":    date_str,
        "expver":  "prod",
        "levtype": "sfc",
        "model":   "glob",
        "origin":  "ecmf",
        "param":   "136/167/168/228086/228095",  # tcw/2t/2d/sm20/st20
        "step":    STEP_STRING,
        "stream":  "enfo",
        "time":    "00:00:00",
        "type":    "cf",
        "target":  str(target),
    }

    try:
        print(f"[DOWNLOADING] {date_str} -> {target}")
        server.retrieve(req)

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
    """Generate list of date strings between *start_date* and *end_date* (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


# ------------------------------------------------------------------ #
# Batch download loop
# ------------------------------------------------------------------ #

def download_batch(server, dates, outdir, wait_time=5):
    """
    Download a list of dates with progress reporting and rate limiting.

    Args:
        server: ECMWFDataServer instance
        dates: List of date strings
        outdir: Output directory (Path)
        wait_time: Seconds to sleep between requests

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

            success = download_single_date(server, date, outdir)

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
        help="Download the full default date range (2025-01-26 to 2025-11-15)",
    )
    parser.add_argument(
        "--batch-start", type=str, default="2025-01-26",
        help="Override batch start date (default: 2025-01-26)",
    )
    parser.add_argument(
        "--batch-end", type=str, default="2025-11-15",
        help="Override batch end date (default: 2025-11-15)",
    )
    parser.add_argument(
        "--wait", type=int, default=5,
        help="Seconds to wait between requests (default: 5)",
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
    outdir = Path(get_path(cfg, "ecmwf_dir"))
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Determine date list ----
    if args.batch:
        dates = generate_date_list(args.batch_start, args.batch_end)
        print(f"[BATCH MODE] Will download {len(dates)} dates: "
              f"{args.batch_start} to {args.batch_end}\n")
    elif len(args.dates) == 2:
        start_date, end_date = args.dates
        dates = generate_date_list(start_date, end_date)
        print(f"[RANGE MODE] Will download {len(dates)} dates: "
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

    # ---- Download ----
    if len(dates) == 1:
        # Single date: match original single-date script behaviour (exit codes)
        try:
            print(f"Requesting {dates[0]} -> {outdir}")
            success = download_single_date(server, dates[0], outdir)
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            target = outdir / f"s2s_ecmf_cf_{dates[0].replace('/', '_')}.grib"
            print(f"\nCancelled by user. Partial file (if any): {target}")
            sys.exit(130)
    else:
        _, fail_count, _ = download_batch(
            server, dates, outdir, wait_time=args.wait
        )
        sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
