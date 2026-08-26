#!/usr/bin/env python3
"""
Download monthly climate indices (ENSO/PDO/NAO/AO) from NOAA.

These large-scale climate modes modulate Canadian fire season severity:
  ONI  (Oceanic Niño Index)  — El Niño / La Niña, key for western Canada dryness
  PDO  (Pacific Decadal Oscillation) — decade-scale Pacific SST pattern
  NAO  (North Atlantic Oscillation)  — affects eastern Canada moisture
  AO   (Arctic Oscillation)          — polar vortex, winter/spring temperature anomalies

All data is freely available from NOAA with no API key required.
Output: a single CSV with columns: year, month, ONI, PDO, NAO, AO

Usage:
    python -m src.data_ops.download.download_climate_indices
    python -m src.data_ops.download.download_climate_indices --config configs/paths_windows.yaml
    python -m src.data_ops.download.download_climate_indices --overwrite

Maintenance (if a column comes out empty, NOAA reformatted that ASCII file):
    Source of truth : https://www.cpc.ncep.noaa.gov/data/teledoc/telecontents.shtml
    Last verified   : 2026-08
    curl each URL in SOURCES and eyeball the layout. See docs/DATA_SOURCES.md
    for the full endpoint registry and the debugging playbook.
"""

import argparse
import csv
import os
import sys
import time
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
# NOAA data sources (no API key required)
# ------------------------------------------------------------------ #

SOURCES = {
    "ONI": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        "description": "Oceanic Niño Index (3-month running mean of ERSST.v5 SST anomalies, Niño 3.4)",
    },
    "PDO": {
        "url": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
        "description": "Pacific Decadal Oscillation (ERSST.v5)",
    },
    "NAO": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table",
        "description": "North Atlantic Oscillation (standardized monthly)",
    },
    "AO": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table",
        "description": "Arctic Oscillation (standardized monthly)",
    },
}

TIMEOUT = 30   # seconds per request
RETRY   = 3


# ------------------------------------------------------------------ #
# Parsers — each NOAA file has a unique ASCII format
# ------------------------------------------------------------------ #

def _fetch(url: str, name: str) -> str:
    """Download URL text with retry."""
    for attempt in range(1, RETRY + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            print(f"  [{name}] Downloaded {len(r.content):,} bytes")
            return r.text
        except Exception as e:
            if attempt < RETRY:
                print(f"  [{name}] Attempt {attempt} failed: {e}  (retrying…)")
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"[{name}] Failed after {RETRY} attempts: {e}") from e


# ONI is a 3-month running mean labelled by an overlapping season code; we map
# each season to its centre month (DJF -> Jan, JFM -> Feb, ..., NDJ -> Dec).
_ONI_SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def _is_missing(val: float) -> bool:
    """True for NOAA missing-value sentinels (e.g. 99.99, -99.99, -9.9)."""
    return abs(val) >= 90.0


def _parse_oni(text: str) -> dict[tuple[int, int], float]:
    """
    ONI format (whitespace-delimited, header ``SEAS YR TOTAL ANOM``):
        DJF  1950  25.01  -1.32
        JFM  1950  25.36  -1.20
        ...
    Column 0 is the overlapping-season code, column 1 the year, and column 3
    the SST anomaly (the ONI value). We keep the anomaly, keyed by centre month.
    """
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] not in _ONI_SEASON_TO_MONTH:
            continue
        try:
            year  = int(parts[1])
            month = _ONI_SEASON_TO_MONTH[parts[0]]
            anom  = float(parts[3])
        except ValueError:
            continue
        if _is_missing(anom):
            continue
        result[(year, month)] = anom
    return result


def _parse_year_month_matrix(text: str) -> dict[tuple[int, int], float]:
    """
    Parse a NOAA "year row x month column" table, shared by PDO/NAO/AO.

        <year>  <Jan> <Feb> ... <Dec>

    Robust to the two header styles NOAA ships (``Year Jan ... Dec`` for PDO,
    a bare ``Jan ... Dec`` for NAO/AO) because header rows have a non-numeric
    first token and are skipped. Trailing months of the current year may be
    absent (short row) or filled with a sentinel; both are dropped.
    """
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if not (1800 <= year <= 2100):
            continue
        for m_idx, val_str in enumerate(parts[1:13], start=1):
            try:
                val = float(val_str)
            except ValueError:
                continue
            if _is_missing(val):
                continue
            result[(year, m_idx)] = val
    return result


# PDO, NAO, and AO all ship as a "year row x month column" table, so they share
# one parser. (NAO/AO drop the ``Year`` header label; PDO keeps it — both work
# because header rows have a non-numeric first token.)
PARSERS = {
    "ONI": _parse_oni,
    "PDO": _parse_year_month_matrix,
    "NAO": _parse_year_month_matrix,
    "AO":  _parse_year_month_matrix,
}


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def download_climate_indices(output_csv: Path, overwrite: bool = False) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists() and not overwrite:
        print(f"  [SKIP] {output_csv} already exists (use --overwrite to refresh)")
        return

    # Download and parse each index
    data: dict[str, dict[tuple[int, int], float]] = {}
    for name, info in SOURCES.items():
        print(f"  Fetching {name}:  {info['url']}")
        text   = _fetch(info["url"], name)
        parsed = PARSERS[name](text)
        data[name] = parsed
        print(f"    → {len(parsed)} month-records parsed")

    # Merge: collect all (year, month) keys present in at least one index
    all_keys = sorted(set().union(*[d.keys() for d in data.values()]))

    rows_written = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "month", "ONI", "PDO", "NAO", "AO"])
        for (year, month) in all_keys:
            row = [
                year,
                month,
                data["ONI"].get((year, month), ""),
                data["PDO"].get((year, month), ""),
                data["NAO"].get((year, month), ""),
                data["AO"].get((year, month), ""),
            ]
            writer.writerow(row)
            rows_written += 1

    print(f"\n  Written {rows_written} rows → {output_csv}")
    print("  Columns: year, month, ONI (ENSO), PDO, NAO, AO")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and overwrite existing output CSV.",
    )
    args = parser.parse_args()
    cfg  = load_config(args.config)

    output_csv = Path(get_path(cfg, "climate_indices_csv"))
    print(f"Climate indices download")
    print(f"  Output: {output_csv}")
    print()
    download_climate_indices(output_csv, overwrite=args.overwrite)
    print("\nDone.")


if __name__ == "__main__":
    main()
