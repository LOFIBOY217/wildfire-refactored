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


def _parse_oni(text: str) -> dict[tuple[int, int], float]:
    """
    ONI format (fixed-width):
        YR  MON  TOTAL  CLIM  ANOM  ...
        ...
    We want the ANOM column (index 4).
    """
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            anom  = float(parts[4])
            result[(year, month)] = anom
        except ValueError:
            continue
    return result


def _parse_pdo(text: str) -> dict[tuple[int, int], float]:
    """
    PDO format:
        Year Jan Feb Mar ... Dec
        1900  val val ...
    """
    result = {}
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    header_found = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        # Detect header row
        if parts[0] == "Year":
            header_found = True
            continue
        if not header_found:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for m_idx, val_str in enumerate(parts[1:13], start=1):
            try:
                val = float(val_str)
                if val < -99:   # missing sentinel
                    continue
                result[(year, m_idx)] = val
            except ValueError:
                continue
    return result


def _parse_nao_table(text: str) -> dict[tuple[int, int], float]:
    """
    NAO format (tabular, year as first column, months as subsequent columns):
        Year  Jan   Feb  ...  Dec
        1950  val   val  ...  val
    """
    result = {}
    header_found = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ("Year", "YEAR"):
            header_found = True
            continue
        if not header_found:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for m_idx, val_str in enumerate(parts[1:13], start=1):
            try:
                val = float(val_str)
                if val < -99:
                    continue
                result[(year, m_idx)] = val
            except ValueError:
                continue
    return result


def _parse_ao_table(text: str) -> dict[tuple[int, int], float]:
    """
    AO format:  Year Jan Feb ... Dec  (same as NAO)
    """
    return _parse_nao_table(text)


PARSERS = {
    "ONI": _parse_oni,
    "PDO": _parse_pdo,
    "NAO": _parse_nao_table,
    "AO":  _parse_ao_table,
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
