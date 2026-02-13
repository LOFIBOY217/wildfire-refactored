#!/usr/bin/env python3
"""
Test FWI date availability before downloading.

Sends lightweight requests (100x100 pixels) to check which dates have data
on the CWFIS WCS service, without downloading full-resolution files.

Usage:
    python -m src.data.validation.test_fwi_availability 20240901 20241231
    python -m src.data.validation.test_fwi_availability 20250501 20251031 --config configs/default.yaml
"""

import sys
import argparse
import urllib.request
from datetime import datetime, timedelta

from src.config import load_config, add_config_argument


WCS_BASE = (
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs"
    "?service=WCS&version=1.0.0&request=GetCoverage"
    "&format=GeoTIFF&crs=EPSG:3978"
    "&bbox=-2378164,-707617,3039835,3854382"
    "&width=100&height=100"
)


def test_date(date_str):
    """Test if FWI data exists for a single date (lightweight 100x100 request)."""
    url = f"{WCS_BASE}&coverage=public:fwi{date_str}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'FWI-Test/1.0'})
        resp = urllib.request.urlopen(req, timeout=10)
        return len(resp.read()) > 1000
    except Exception:
        return False


def test_range(start_date, end_date):
    """
    Test availability for a date range.

    Args:
        start_date: Start date string YYYYMMDD
        end_date: End date string YYYYMMDD

    Returns:
        tuple: (available_list, unavailable_list)
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    available = []
    unavailable = []
    current = start

    print(f"Testing {start.date()} -> {end.date()}\n")

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        print(f"{date_str} ", end="", flush=True)

        if test_date(date_str):
            print("[OK]")
            available.append(date_str)
        else:
            print("[MISS]")
            unavailable.append(date_str)

        current += timedelta(days=1)

    total = len(available) + len(unavailable)
    print(f"\n{'='*40}")
    print(f"Available: {len(available)}/{total}")
    print(f"{'='*40}")

    if unavailable and len(unavailable) <= 30:
        print(f"\nUnavailable dates:")
        for d in unavailable:
            print(f"  {d}")

    return available, unavailable


def main():
    parser = argparse.ArgumentParser(
        description="Test FWI data availability on CWFIS WCS before downloading",
    )
    add_config_argument(parser)
    parser.add_argument("start_date", help="Start date YYYYMMDD")
    parser.add_argument("end_date", help="End date YYYYMMDD")

    args = parser.parse_args()
    _ = load_config(args.config)

    test_range(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
