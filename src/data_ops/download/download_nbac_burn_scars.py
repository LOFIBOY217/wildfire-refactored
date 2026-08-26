#!/usr/bin/env python3
"""
Download NBAC (National Burned Area Composite) annual shapefiles from NRCan.

Download only — saves raw zip files to disk.
Processing (shapefile → rasterize → years-since-burn TIFs) is done separately
by processing/process_nbac_burn_scars.py.

Source: CWFIS Datamart (https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/)
Output: {burn_scars_raw_dir}/nbac_{YYYY}.zip

Usage:
    python -m src.data_ops.download.download_nbac_burn_scars
    python -m src.data_ops.download.download_nbac_burn_scars --start_year 2000 --end_year 2024
"""

import argparse
import re
import sys
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


# CWFIS Datamart NBAC directory. Per-year files are named
# ``NBAC_{year}_{release}.zip`` where {release} is a YYYYMMDD stamp that changes
# on every NBAC re-release, so we scrape the listing rather than hardcode it.
NBAC_DIR_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/"

# Legacy NFIS templates (kept as a fallback; NFIS retired these in 2025).
NBAC_URL_TEMPLATES = [
    "https://opendata.nfis.org/downloads/forest_change/CA_Forest_Fire_NBAC_{year}_r9_20210810.zip",
    "https://opendata.nfis.org/downloads/forest_change/nbac_{year}.zip",
]


def _discover_nbac_urls() -> dict[int, str]:
    """Scrape the CWFIS NBAC directory listing → {year: absolute_zip_url}.

    Returns {} if the listing cannot be fetched (caller falls back to the
    legacy per-year templates).
    """
    try:
        resp = requests.get(NBAC_DIR_URL, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [WARN] could not list {NBAC_DIR_URL}: {e}")
        return {}
    urls = {}
    # Match e.g. NBAC_2024_20260513.zip (single-year composites only, not the
    # multi-year NBAC_1972to2025_..._shp.zip bundle).
    for m in re.finditer(r"NBAC_(\d{4})_(\d{8})\.zip", resp.text):
        year = int(m.group(1))
        # Prefer the most recent release stamp if a year appears more than once.
        if year not in urls or m.group(2) > urls[year][1]:
            urls[year] = (NBAC_DIR_URL + m.group(0), m.group(2))
    return {y: u for y, (u, _stamp) in urls.items()}


def _fetch_zip(url: str) -> bytes | None:
    """GET a URL, returning its bytes only if they are a valid (PK) zip."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=120)
            if (resp.status_code == 200 and len(resp.content) > 1000
                    and resp.content[:2] == b"PK"):  # valid zip header
                print(f"    [OK] {url}  ({len(resp.content)/1e6:.1f} MB)")
                return resp.content
            return None  # 404 or HTML "not found" page → caller tries next
        except Exception:
            if attempt == 2:
                return None
    return None


def _download_nbac_zip(year: int, discovered: dict[int, str]) -> bytes | None:
    """Download the NBAC zip for one year (CWFIS listing first, then legacy)."""
    if year in discovered:
        data = _fetch_zip(discovered[year])
        if data is not None:
            return data
    for tmpl in NBAC_URL_TEMPLATES:
        data = _fetch_zip(tmpl.format(year=year))
        if data is not None:
            return data
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download NBAC burn scar shapefiles (download only)"
    )
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=1985)
    parser.add_argument("--end_year", type=int, default=2025)
    args = parser.parse_args()

    cfg = load_config(args.config)
    burn_dir = Path(get_path(cfg, "burn_scars_dir"))
    raw_dir = burn_dir.parent / "burn_scars_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NBAC Burn Scars — Download Only (raw zip files)")
    print("=" * 70)
    print(f"  Years  : {args.start_year} – {args.end_year}")
    print(f"  Output : {raw_dir}/nbac_YYYY.zip")
    print(f"  NOTE: Run processing/process_nbac_burn_scars.py after download")
    print("=" * 70)

    discovered = _discover_nbac_urls()
    if discovered:
        print(f"  Listed {len(discovered)} NBAC years on CWFIS "
              f"({min(discovered)}–{max(discovered)})")

    ok = skip = fail = 0
    for year in range(args.start_year, args.end_year + 1):
        out_path = raw_dir / f"nbac_{year}.zip"
        if out_path.exists() and out_path.stat().st_size > 100:
            skip += 1
            continue

        print(f"  [{year}] Downloading…")
        data = _download_nbac_zip(year, discovered)
        if data is None:
            print(f"  [{year}] No data available")
            fail += 1
            continue

        out_path.write_bytes(data)
        ok += 1

    print(f"\n[COMPLETE] ok={ok}  skip={skip}  fail={fail}")


if __name__ == "__main__":
    main()
