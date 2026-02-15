#!/usr/bin/env python3
"""
Batch check ECMWF S2S availability by issue date.

This script probes each issue date with a minimal ECMWF request and records
whether data is available before running large downloads.

Usage:
    python scripts/check_s2s_ecmwf_availability.py
    python scripts/check_s2s_ecmwf_availability.py --start 20250126 --end 20251115
    python scripts/check_s2s_ecmwf_availability.py --workers 2 --output s2s_avail.csv
"""

import argparse
import csv
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from src.config import load_config, add_config_argument
except ModuleNotFoundError:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import load_config, add_config_argument


def generate_dates(start_yyyymmdd: str, end_yyyymmdd: str) -> List[str]:
    """Generate inclusive YYYY-MM-DD date strings."""
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def build_probe_request(date_str: str, target_path: str) -> Dict[str, str]:
    """
    Build a minimal request for availability probing.

    We request one parameter and one lead range so each check stays lightweight.
    """
    return {
        "class": "s2",
        "dataset": "s2s",
        "date": date_str,
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": "167",          # 2t
        "step": "336-360",       # day 14 one-day averaged step
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf",
        "target": target_path,
    }


def probe_date(date_str: str, email: str, key: str, tmp_root: str) -> Tuple[str, bool, str]:
    """Return (date_str, available, detail)."""
    from ecmwfapi import ECMWFDataServer

    date_compact = date_str.replace("-", "")
    target = os.path.join(tmp_root, f"probe_{date_compact}.grib")
    server = ECMWFDataServer(url="https://api.ecmwf.int/v1", key=key, email=email)
    req = build_probe_request(date_str, target)

    try:
        server.retrieve(req)
        size = os.path.getsize(target) if os.path.exists(target) else 0
        ok = size > 0
        detail = f"bytes={size}" if ok else "empty_file"
        return date_str, ok, detail
    except Exception as exc:
        return date_str, False, str(exc).splitlines()[0][:300]
    finally:
        if os.path.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass


def summarize_gaps(sorted_dates: List[str], results: Dict[str, bool]) -> List[Tuple[str, str]]:
    """Return unavailable contiguous date ranges in YYYY-MM-DD strings."""
    gaps = []
    gap_start = None
    prev_d = None
    for d in sorted_dates:
        if not results[d]:
            if gap_start is None:
                gap_start = d
        else:
            if gap_start is not None and prev_d is not None:
                gaps.append((gap_start, prev_d))
                gap_start = None
        prev_d = d
    if gap_start is not None:
        gaps.append((gap_start, sorted_dates[-1]))
    return gaps


def main():
    parser = argparse.ArgumentParser(description="Batch check ECMWF S2S availability")
    add_config_argument(parser)
    parser.add_argument("--start", default="20180101", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="End date YYYYMMDD")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers (default: 1)")
    parser.add_argument("--output", default="s2s_ecmwf_availability.csv", help="Output CSV path")
    args = parser.parse_args()

    try:
        import ecmwfapi  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: ecmwfapi. Install it before running this checker."
        ) from exc

    cfg = load_config(args.config)
    ecmwf_email = os.environ.get("ECMWF_EMAIL", cfg.get("credentials", {}).get("ecmwf_email", ""))
    ecmwf_key = os.environ.get("ECMWF_KEY", cfg.get("credentials", {}).get("ecmwf_key", ""))
    if not ecmwf_email or not ecmwf_key:
        raise RuntimeError("ECMWF credentials missing: set ECMWF_EMAIL and ECMWF_KEY")

    dates = generate_dates(args.start, args.end)
    total = len(dates)
    print(f"Checking {total} issue dates ({dates[0]} -> {dates[-1]}) with {args.workers} worker(s)")
    print("Probe request: param=167, step=336-360, type=cf, stream=enfo")

    tmp_root = tempfile.mkdtemp(prefix="s2s_probe_")
    results: Dict[str, bool] = {}
    details: Dict[str, str] = {}

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(probe_date, d, ecmwf_email, ecmwf_key, tmp_root): d for d in dates
            }
            done = 0
            for future in as_completed(futures):
                date_str, ok, detail = future.result()
                results[date_str] = ok
                details[date_str] = detail
                done += 1
                if done % 20 == 0 or done == total:
                    avail_so_far = sum(1 for v in results.values() if v)
                    print(f"  Progress: {done}/{total} (available so far: {avail_so_far})")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    sorted_dates = sorted(results.keys())
    available = [d for d in sorted_dates if results[d]]
    unavailable = [d for d in sorted_dates if not results[d]]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "available", "detail"])
        for d in sorted_dates:
            writer.writerow([d, results[d], details.get(d, "")])

    print("\n" + "=" * 56)
    print(f"TOTAL:       {total}")
    print(f"AVAILABLE:   {len(available)}")
    print(f"UNAVAILABLE: {len(unavailable)}")
    print("=" * 56)

    if available:
        print(f"Available range: {available[0]} -> {available[-1]}")
    gaps = summarize_gaps(sorted_dates, results)
    if gaps:
        print(f"Gaps ({len(gaps)} total):")
        for g_start, g_end in gaps[:30]:
            if g_start == g_end:
                print(f"  {g_start}")
            else:
                print(f"  {g_start} -> {g_end}")
        if len(gaps) > 30:
            print(f"  ... and {len(gaps) - 30} more gaps")

    print("\nYear-by-year:")
    years = sorted({d[:4] for d in sorted_dates})
    for year in years:
        year_dates = [d for d in sorted_dates if d[:4] == year]
        year_avail = sum(1 for d in year_dates if results[d])
        pct = year_avail / max(len(year_dates), 1) * 100
        bar = "#" * int(pct / 2.5)
        print(f"  {year}: {year_avail:>3}/{len(year_dates):>3} ({pct:5.1f}%) {bar}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
