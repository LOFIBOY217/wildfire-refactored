#!/usr/bin/env python3
"""
Download ECMWF ERA5 Reanalysis (observed/analyzed) data for specific dates.
This data represents historical observations and is suitable for training
baseline models.

ERA5 is ECMWF's 5th generation reanalysis - the "truth" for historical periods.
Use this instead of S2S forecasts when you need observed conditions.

Variables: 2m temperature, 2m dewpoint, total column water,
           soil moisture (0-7cm), soil temp (0-7cm)
Temporal:  Daily average (00-23 UTC)
Spatial:   0.25 x 0.25 deg (will need resampling to FWI grid)

Usage:
    Single date:    python -m src.data_ops.download.era5_observations 2025-09-12
    Date range:     python -m src.data_ops.download.era5_observations 2025-09-12 2025-10-10
    Batch mode:     python -m src.data_ops.download.era5_observations --batch
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi

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

# Default bounding box for Canada: [north, west, south, east]
DEFAULT_AREA = [83, -141, 41, -52]


def download_single_date(client, date_str, outdir, area=None):
    """
    Download ERA5 reanalysis data for a single date.

    Args:
        client: CDS API client
        date_str: Date in YYYY-MM-DD format
        outdir: Output directory (Path)
        area: Optional bounding box [north, west, south, east] in degrees.
              Default covers Canada: [83, -141, 41, -52]

    Returns:
        True on success, False on failure.
    """
    safe_date = date_str.replace("-", "_")
    target = outdir / f"era5_sfc_{safe_date}.grib"

    # Skip if already exists
    if target.exists() and target.stat().st_size > 0:
        print(f"[SKIP] {date_str} - file already exists: {target}")
        return True

    if area is None:
        area = DEFAULT_AREA

    # Convert date to year/month/day for API
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    day = date_obj.strftime("%d")

    req = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            '2m_temperature',                # 2t  (167)
            '2m_dewpoint_temperature',        # 2d  (168)
            'total_column_water',             # tcw (136)
            'volumetric_soil_water_layer_1',  # swvl1 (closest to sm20)
            'soil_temperature_level_1',       # stl1  (closest to st20)
        ],
        'year': year,
        'month': month,
        'day': day,
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        'area': area,  # [north, west, south, east]
    }

    try:
        print(f"[DOWNLOADING] {date_str} -> {target}")
        print(f"  Area: N={area[0]}, W={area[1]}, S={area[2]}, E={area[3]}")

        client.retrieve(
            'reanalysis-era5-single-levels',
            req,
            str(target),
        )

        if target.exists() and target.stat().st_size > 0:
            print(f"[SUCCESS] {date_str} - {target.stat().st_size / 1e6:.1f} MB")
            return True
        else:
            print(f"[ERROR] {date_str} - file missing or empty", file=sys.stderr)
            return False

    except KeyboardInterrupt:
        print(f"\n[CANCELLED] {date_str} - partial file: {target}")
        if target.exists():
            target.unlink()  # Delete partial file
        raise
    except Exception as e:
        print(f"[ERROR] {date_str} - {e}", file=sys.stderr)
        if target.exists():
            target.unlink()  # Delete partial file
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
# CLI
# ------------------------------------------------------------------ #

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Download ERA5 reanalysis data (single-level surface variables)"
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
        help="Download the full default date range (2025-05-01 to 2025-10-31)",
    )
    parser.add_argument(
        "--batch-start", type=str, default="2025-05-01",
        help="Override batch start date (default: 2025-05-01)",
    )
    parser.add_argument(
        "--batch-end", type=str, default="2025-10-31",
        help="Override batch end date (default: 2025-10-31)",
    )
    parser.add_argument(
        "--wait", type=int, default=2,
        help="Seconds to wait between requests (default: 2)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Load config and credentials from environment ----
    cfg = load_config(args.config)

    cds_api_key = os.environ.get(
        "CDS_API_KEY",
        cfg.get("credentials", {}).get("cds_api_key", ""),
    )

    if not cds_api_key:
        print(
            "ERROR: CDS API key not found.\n"
            "Set CDS_API_KEY environment variable, "
            "or configure it in your YAML config under 'credentials.cds_api_key'.\n"
            "Get your key from: https://cds.climate.copernicus.eu/api-how-to",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Resolve output directory ----
    outdir = Path(get_path(cfg, "era5_dir"))
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Create CDS API client ----
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=cds_api_key,
    )

    # ---- Determine date list ----
    if args.batch:
        dates = generate_date_list(args.batch_start, args.batch_end)
        print(f"[BATCH MODE] Will download {len(dates)} dates: "
              f"{args.batch_start} to {args.batch_end}")
        print("(Fire season + shoulder months)\n")
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

    # ---- Info banner ----
    print("=" * 70)
    print("IMPORTANT NOTES")
    print("=" * 70)
    print("  ERA5 data is at 0.25 deg resolution (~28km)")
    print("  You'll need to resample to FWI grid (2km, EPSG:3978)")
    print("  Each download includes 24 hourly timesteps (daily data)")
    print("  Process with: cfgrib + xarray to extract daily averages")
    print("  Area: Canada bounding box [83N, -141W, 41N, -52W]")
    print("=" * 70)
    print()

    # ---- Download loop ----
    success_count = 0
    fail_count = 0
    failed_dates = []

    try:
        for i, date in enumerate(dates, 1):
            print(f"\n{'='*70}")
            print(f"Progress: {i}/{len(dates)} ({i/len(dates)*100:.1f}%)")
            print(f"{'='*70}")

            success = download_single_date(client, date, outdir)

            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_dates.append(date)

            # Rate limiting
            if i < len(dates):
                print(f"Waiting {args.wait}s before next request...")
                time.sleep(args.wait)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Download cancelled by user")

    finally:
        # ---- Summary ----
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
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

        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("1. Process GRIB files to daily averages:")
        print("   python process_era5_to_daily.py")
        print("\n2. Resample to FWI grid:")
        print("   python resample_to_fwi_grid.py")
        print("\n3. Verify alignment with FWI:")
        print("   python verify_data_alignment.py")
        print("=" * 70)

        sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
