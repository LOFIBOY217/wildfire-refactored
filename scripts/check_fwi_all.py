#!/usr/bin/env python3
"""
Batch check FWI availability from 20150101 to today.
Uses concurrent requests for speed, outputs CSV summary.

Usage:
    python scripts/check_fwi_all.py
    python scripts/check_fwi_all.py --workers 20
    python scripts/check_fwi_all.py --output my_report.csv
"""

import argparse
import csv
import urllib.request
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed

WCS_BASE = (
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs"
    "?service=WCS&version=1.0.0&request=GetCoverage"
    "&format=GeoTIFF&crs=EPSG:3978"
    "&bbox=-2378164,-707617,3039835,3854382"
    "&width=100&height=100"
)


def test_date(date_str):
    """Return (date_str, True/False)."""
    url = f"{WCS_BASE}&coverage=public:fwi{date_str}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FWI-Test/1.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        ok = len(resp.read()) > 1000
        return date_str, ok
    except Exception:
        return date_str, False


def generate_dates(start_str, end_str):
    start = datetime.strptime(start_str, "%Y%m%d")
    end = datetime.strptime(end_str, "%Y%m%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(description="Batch check FWI availability")
    parser.add_argument("--start", default="20150101", help="Start date YYYYMMDD")
    parser.add_argument("--end", default=date.today().strftime("%Y%m%d"), help="End date YYYYMMDD")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers (default: 10)")
    parser.add_argument("--output", default="fwi_availability.csv", help="Output CSV path")
    args = parser.parse_args()

    dates = generate_dates(args.start, args.end)
    total = len(dates)
    print(f"Checking {total} dates ({args.start} -> {args.end}) with {args.workers} workers...\n")

    results = {}
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(test_date, d): d for d in dates}

        for future in as_completed(futures):
            date_str, ok = future.result()
            results[date_str] = ok
            done += 1

            if done % 100 == 0 or done == total:
                avail_so_far = sum(1 for v in results.values() if v)
                print(f"  Progress: {done}/{total}  (available so far: {avail_so_far})")

    # Sort by date
    sorted_dates = sorted(results.keys())
    available = [d for d in sorted_dates if results[d]]
    unavailable = [d for d in sorted_dates if not results[d]]

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "available"])
        for d in sorted_dates:
            writer.writerow([d, results[d]])

    # Summary
    print(f"\n{'='*50}")
    print(f"TOTAL:       {total}")
    print(f"AVAILABLE:   {len(available)}")
    print(f"UNAVAILABLE: {len(unavailable)}")
    print(f"{'='*50}")

    if available:
        print(f"\nAvailable range: {available[0]} -> {available[-1]}")

        # Find contiguous gaps
        gaps = []
        gap_start = None
        for d in sorted_dates:
            if not results[d]:
                if gap_start is None:
                    gap_start = d
            else:
                if gap_start is not None:
                    gaps.append((gap_start, prev_d))
                    gap_start = None
            prev_d = d
        if gap_start is not None:
            gaps.append((gap_start, sorted_dates[-1]))

        if gaps:
            print(f"\nGaps ({len(gaps)} total):")
            for g_start, g_end in gaps[:30]:
                if g_start == g_end:
                    print(f"  {g_start}")
                else:
                    print(f"  {g_start} -> {g_end}")
            if len(gaps) > 30:
                print(f"  ... and {len(gaps) - 30} more gaps")

    # Year-by-year summary
    print(f"\nYear-by-year:")
    years = sorted(set(d[:4] for d in sorted_dates))
    for year in years:
        year_dates = [d for d in sorted_dates if d[:4] == year]
        year_avail = sum(1 for d in year_dates if results[d])
        pct = year_avail / len(year_dates) * 100
        bar = "#" * int(pct / 2.5)
        print(f"  {year}: {year_avail:>3}/{len(year_dates):>3} ({pct:5.1f}%) {bar}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
