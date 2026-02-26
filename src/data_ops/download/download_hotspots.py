"""
Download CWFIS fire hotspot data (2018–2025) via WFS.

Data source: CWFIS GeoServer public:hotspots layer
  - Satellite VIIRS-M detections processed by Canadian Forest Service
  - Complete historical archive (tested back to 2018)
  - No account or API key required (same approach as download_fwi_grids.py)

Output: CSV with columns: latitude, longitude, acq_date

Usage:
    python -m src.data_ops.download.download_hotspots
    python -m src.data_ops.download.download_hotspots --start_year 2020 --end_year 2021
    python -m src.data_ops.download.download_hotspots --config configs/default.yaml
"""

import argparse
import calendar
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

WFS_BASE = "https://cwfis.cfs.nrcan.gc.ca/geoserver/ows"
WFS_PARAMS = dict(
    service="WFS",
    version="2.0.0",
    request="GetFeature",
    typeName="public:hotspots",
    outputFormat="application/json",
    srsName="EPSG:4326",
)

PAGE_SIZE   = 10_000   # Records per WFS page
RETRY_WAIT  = 5        # Seconds between retries
MAX_RETRIES = 3        # Per-page retry attempts
REQUEST_DELAY = 0.5    # Seconds between page requests (be polite to server)

OUTPUT_COLS = ["latitude", "longitude", "acq_date"]


# ------------------------------------------------------------------ #
# Core download logic
# ------------------------------------------------------------------ #

def _fetch_page(month_start: str, month_end: str, start_index: int,
                session: requests.Session) -> dict | None:
    """
    Fetch one page of WFS hotspot features for a given date range.

    Args:
        month_start: ISO date string 'YYYY-MM-DD' (first day of month).
        month_end:   ISO date string 'YYYY-MM-DD' (last day of month).
        start_index: Pagination offset.
        session:     Requests session to reuse.

    Returns:
        Parsed JSON dict or None on failure.
    """
    params = {
        **WFS_PARAMS,
        "CQL_FILTER": f"rep_date >= '{month_start}' AND rep_date <= '{month_end}'",
        "startIndex": start_index,
        "count": PAGE_SIZE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(WFS_BASE, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"      [retry {attempt}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_WAIT * attempt)
            else:
                print(f"      [FAIL] {e}")
                return None


def _extract_rows(geojson: dict) -> list[tuple]:
    """
    Extract (latitude, longitude, acq_date) tuples from a WFS GeoJSON response.

    rep_date format: '2018-07-01T07:15:00Z' → acq_date: '2018-07-01'
    geometry.coordinates: [longitude, latitude]
    """
    rows = []
    for feature in geojson.get("features", []):
        try:
            coords = feature["geometry"]["coordinates"]
            lon = coords[0]
            lat = coords[1]
            rep_date = feature["properties"].get("rep_date", "")
            acq_date = rep_date[:10]          # 'YYYY-MM-DD'
            if acq_date and lat and lon:
                rows.append((lat, lon, acq_date))
        except (KeyError, TypeError, IndexError):
            continue
    return rows


def download_month(year: int, month: int, writer: csv.writer,
                   session: requests.Session) -> int:
    """
    Download all hotspot records for one calendar month and write to CSV.

    Args:
        year, month: Target year/month.
        writer:      CSV writer (already past header row).
        session:     Requests session.

    Returns:
        Total number of rows written for this month.
    """
    month_start = date(year, month, 1).isoformat()
    last_day    = calendar.monthrange(year, month)[1]
    month_end   = date(year, month, last_day).isoformat()

    total_written = 0
    start_index   = 0

    while True:
        data = _fetch_page(month_start, month_end, start_index, session)
        if data is None:
            print(f"      [ERROR] page fetch failed at startIndex={start_index}")
            break

        rows = _extract_rows(data)
        if not rows:
            break  # No more data

        writer.writerows(rows)
        total_written += len(rows)

        # Check if there are more pages
        n_returned = len(data.get("features", []))
        if n_returned < PAGE_SIZE:
            break   # Last page

        start_index += PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    return total_written


# ------------------------------------------------------------------ #
# Resumable download (skip already-downloaded months)
# ------------------------------------------------------------------ #

def _load_existing_months(csv_path: str) -> set[tuple[int, int]]:
    """
    Read existing CSV and return set of (year, month) tuples already downloaded.
    A month is considered complete if it has at least 1 row for the last day.
    """
    if not os.path.exists(csv_path):
        return set()

    done = set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=["acq_date"], parse_dates=["acq_date"])
        for (y, m), _ in df.groupby([df["acq_date"].dt.year, df["acq_date"].dt.month]):
            done.add((int(y), int(m)))
    except Exception as e:
        print(f"  [WARN] Could not read existing CSV for resume check: {e}")
    return done


# ------------------------------------------------------------------ #
# Main download function
# ------------------------------------------------------------------ #

def download_hotspots(output_path: str,
                      start_year: int = 2018, end_year: int = 2025,
                      start_month: int = 5,  end_month: int = 10) -> None:
    """
    Download CWFIS hotspot data for fire season months and save to CSV.

    Args:
        output_path:  Output CSV file path.
        start_year:   First year to download (inclusive).
        end_year:     Last year to download (inclusive).
        start_month:  First month of fire season (default: 5 = May).
        end_month:    Last month of fire season (default: 10 = October).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Resume: find already-downloaded months
    done_months = _load_existing_months(output_path)
    if done_months:
        print(f"  [RESUME] Found {len(done_months)} already-downloaded months in {output_path}")

    # Build list of months to download
    months_to_do = [
        (y, m)
        for y in range(start_year, end_year + 1)
        for m in range(start_month, end_month + 1)
        if (y, m) not in done_months
    ]

    if not months_to_do:
        print("  [DONE] All months already downloaded.")
        return

    total_months  = len(months_to_do)
    total_rows    = 0
    file_exists   = os.path.exists(output_path)

    session = requests.Session()
    session.headers.update({"User-Agent": "CWFIS-Hotspot-Downloader/1.0"})

    # Open CSV in append mode (so resume works)
    open_mode = "a" if file_exists else "w"
    with open(output_path, open_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(OUTPUT_COLS)   # Header only on new file

        for idx, (year, month) in enumerate(months_to_do, 1):
            month_label = date(year, month, 1).strftime("%Y-%m")
            print(f"  [{idx:3d}/{total_months}] {month_label} ...", end=" ", flush=True)

            n = download_month(year, month, writer, session)
            total_rows += n

            print(f"{n:>9,} rows")
            f.flush()    # Ensure data is written before next month starts

    print(f"\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"  New rows written : {total_rows:,}")
    print(f"  Output file      : {output_path}")
    print(f"{'='*50}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _build_parser():
    p = argparse.ArgumentParser(
        description="Download CWFIS fire hotspot data (2018-2025) via WFS."
    )
    add_config_argument(p)

    p.add_argument("--start_year",  type=int, default=2018,
                   help="First year to download (default: 2018).")
    p.add_argument("--end_year",    type=int, default=2025,
                   help="Last year to download (default: 2025).")
    p.add_argument("--start_month", type=int, default=5,
                   help="First fire-season month (default: 5 = May).")
    p.add_argument("--end_month",   type=int, default=10,
                   help="Last fire-season month (default: 10 = October).")
    p.add_argument("--output",      type=str, default=None,
                   help="Output CSV path (default: from config hotspot_csv, "
                        "or data/hotspot/hotspot_2018_2025.csv).")
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        try:
            output_path = get_path(cfg, "hotspot_csv")
        except Exception:
            output_path = "data/hotspot/hotspot_2018_2025.csv"

    print("\n" + "="*50)
    print("CWFIS Hotspot Downloader")
    print("="*50)
    print(f"  Years   : {args.start_year} – {args.end_year}")
    print(f"  Months  : {args.start_month} – {args.end_month} (fire season)")
    print(f"  Output  : {output_path}")
    print("="*50 + "\n")

    download_hotspots(
        output_path=output_path,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
    )


if __name__ == "__main__":
    main()
