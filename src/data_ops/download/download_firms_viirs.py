"""
Download NASA FIRMS VIIRS S-NPP active fire hotspot data (2012-2017) via API.

Data source: NASA FIRMS VIIRS S-NPP Standard Processing (VIIRS_SNPP_SP)
  - 375m resolution (same as CWFIS hotspot data used for 2018+)
  - Covers Canada bounding box: lon [-141, -52.6], lat [41.7, 83.1]
  - Requires a free NASA FIRMS MAP_KEY

Output: CSV with columns: latitude, longitude, acq_date
        (identical format to data/hotspot/hotspot_2018_2025.csv)

Usage:
    python -m src.data_ops.download.download_firms_viirs \\
        --key YOUR_MAP_KEY --start 2012 --end 2017 \\
        --output data/hotspot/hotspot_firms_viirs_2012_2017.csv

Get a free MAP_KEY at: https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""

import argparse
import csv
import io
import sys
import time
from datetime import date, timedelta
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

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
PRODUCT        = "VIIRS_SNPP_SP"      # Standard Processing (historical archive)
CANADA_BBOX    = "-141,41.7,-52.6,83.1"   # west,south,east,north
CHUNK_DAYS     = 5                    # Max days per API request (FIRMS limit: 1-5)
# VIIRS confidence: 'l'=low, 'n'=nominal, 'h'=high
# Keep nominal and high (exclude low)
KEEP_CONFIDENCE = {"n", "h"}
RETRY_WAIT     = 10                   # Seconds between retries
MAX_RETRIES    = 3
REQUEST_DELAY  = 1.0                  # Seconds between requests (rate limit: 5000/10min)

OUTPUT_COLS = ["latitude", "longitude", "acq_date"]


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------

def _build_url(map_key: str, start_date: date, n_days: int) -> str:
    """Build FIRMS API URL for a date chunk."""
    return (
        f"{FIRMS_API_BASE}/{map_key}/{PRODUCT}"
        f"/{CANADA_BBOX}/{n_days}/{start_date.isoformat()}"
    )


def _fetch_chunk(map_key: str, start_date: date, n_days: int,
                 session: requests.Session) -> list[tuple] | None:
    """
    Fetch one chunk of VIIRS fire data from FIRMS API.

    Returns list of (latitude, longitude, acq_date) tuples, or None on failure.
    """
    url = _build_url(map_key, start_date, n_days)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 400:
                # No data for this period (normal for off-season)
                return []
            r.raise_for_status()

            # Parse CSV response
            rows = []
            reader = csv.DictReader(io.StringIO(r.text))
            for row in reader:
                try:
                    lat  = float(row["latitude"])
                    lon  = float(row["longitude"])
                    adate = row["acq_date"].strip()   # already YYYY-MM-DD
                    conf = row.get("confidence", "l").strip().lower()
                    if conf not in KEEP_CONFIDENCE:
                        continue
                    rows.append((lat, lon, adate))
                except (KeyError, ValueError):
                    continue
            return rows

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"      [retry {attempt}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_WAIT * attempt)
            else:
                print(f"      [FAIL] {e}")
                return None

    return None


# ------------------------------------------------------------------
# Resume helpers
# ------------------------------------------------------------------

def _load_done_dates(csv_path: str) -> set[str]:
    """
    Return set of acq_date strings ('YYYY-MM-DD') already in the output CSV.
    Used to skip already-downloaded date ranges.
    """
    if not Path(csv_path).exists():
        return set()
    done = set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=["acq_date"])
        done = set(df["acq_date"].dropna().unique())
    except Exception as e:
        print(f"  [WARN] Could not read existing CSV for resume: {e}")
    return done


# ------------------------------------------------------------------
# Main download function
# ------------------------------------------------------------------

def download_firms_viirs(map_key: str, output_path: str,
                         start_year: int = 2012, end_year: int = 2017) -> None:
    """
    Download FIRMS VIIRS S-NPP hotspot data for Canada and save to CSV.

    Args:
        map_key:     NASA FIRMS MAP_KEY (free, register at firms.modaps.eosdis.nasa.gov).
        output_path: Output CSV file path.
        start_year:  First year to download (inclusive). VIIRS available from 2012-01-20.
        end_year:    Last year to download (inclusive).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build list of all date chunks for the requested years
    start_dt = date(start_year, 1, 1)
    end_dt   = date(end_year, 12, 31)

    # VIIRS S-NPP starts 2012-01-20
    if start_dt < date(2012, 1, 20):
        start_dt = date(2012, 1, 20)

    # Resume: find already-downloaded dates
    done_dates = _load_done_dates(output_path)
    if done_dates:
        print(f"  [RESUME] {len(done_dates)} dates already in {output_path}")

    # Generate all 10-day chunks
    chunks = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end_dt)
        n_days = (chunk_end - cur).days + 1
        chunks.append((cur, n_days))
        cur = chunk_end + timedelta(days=1)

    # Filter chunks that are already fully downloaded
    def _chunk_done(chunk_start: date, n_days: int) -> bool:
        for i in range(n_days):
            d = (chunk_start + timedelta(days=i)).isoformat()
            if d not in done_dates:
                return False
        return True

    todo_chunks = [(s, n) for s, n in chunks if not _chunk_done(s, n)]
    total_chunks = len(todo_chunks)

    if not todo_chunks:
        print("  [DONE] All chunks already downloaded.")
        return

    print(f"  Downloading {total_chunks} chunks "
          f"({start_year}-01-01 to {end_year}-12-31) ...")

    file_exists = Path(output_path).exists()
    open_mode   = "a" if file_exists else "w"

    session = requests.Session()
    session.headers.update({"User-Agent": "FIRMS-VIIRS-Downloader/1.0"})

    total_rows = 0

    with open(output_path, open_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(OUTPUT_COLS)

        for idx, (chunk_start, n_days) in enumerate(todo_chunks, 1):
            chunk_end_dt = chunk_start + timedelta(days=n_days - 1)
            label = (f"{chunk_start.isoformat()} – {chunk_end_dt.isoformat()}"
                     f"  ({n_days}d)")
            print(f"  [{idx:4d}/{total_chunks}] {label} ...",
                  end=" ", flush=True)

            rows = _fetch_chunk(map_key, chunk_start, n_days, session)

            if rows is None:
                print("[ERROR] skipping chunk")
                continue

            writer.writerows(rows)
            total_rows += len(rows)
            print(f"{len(rows):>7,} pts")
            f.flush()

            time.sleep(REQUEST_DELAY)

    print(f"\n{'='*60}")
    print(f"FIRMS VIIRS DOWNLOAD COMPLETE")
    print(f"  New rows written : {total_rows:,}")
    print(f"  Output file      : {output_path}")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Download NASA FIRMS VIIRS S-NPP hotspot data for Canada."
    )
    add_config_argument(p)
    p.add_argument("--key",    type=str, required=True,
                   help="NASA FIRMS MAP_KEY (free from firms.modaps.eosdis.nasa.gov/api/map_key/).")
    p.add_argument("--start",  type=int, default=2012,
                   help="Start year (default: 2012, earliest VIIRS data).")
    p.add_argument("--end",    type=int, default=2017,
                   help="End year (default: 2017).")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: data/hotspot/hotspot_firms_viirs_{start}_{end}.csv).")
    p.add_argument("--include_low_confidence", action="store_true",
                   help="Also include low-confidence detections (default: exclude).")
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    cfg = load_config(args.config)

    output_path = args.output or f"data/hotspot/hotspot_firms_viirs_{args.start}_{args.end}.csv"

    print("\n" + "="*60)
    print("NASA FIRMS VIIRS S-NPP Hotspot Downloader")
    print("="*60)
    print(f"  Product    : {PRODUCT} (375m, same as CWFIS 2018+)")
    print(f"  Region     : Canada ({CANADA_BBOX})")
    print(f"  Years      : {args.start} – {args.end}")
    print(f"  Confidence : nominal + high (l=low excluded)")
    print(f"  Output     : {output_path}")
    print("="*60 + "\n")

    download_firms_viirs(
        map_key=args.key,
        output_path=output_path,
        start_year=args.start,
        end_year=args.end,
    )


if __name__ == "__main__":
    main()
