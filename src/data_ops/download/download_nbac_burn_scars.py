#!/usr/bin/env python3
"""
Download NBAC (National Burned Area Composite) annual shapefiles from NRCan.

Download only — saves raw zip files to disk.
Processing (shapefile → rasterize → years-since-burn TIFs) is done separately
by processing/process_nbac_burn_scars.py.

Source: NRCan Open Data (https://opendata.nfis.org/)
Output: {burn_scars_raw_dir}/nbac_{YYYY}.zip

Usage:
    python -m src.data_ops.download.download_nbac_burn_scars
    python -m src.data_ops.download.download_nbac_burn_scars --start_year 2000 --end_year 2024
"""

import argparse
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


# NRCan URL templates (naming varies by year)
NBAC_URL_TEMPLATES = [
    "https://opendata.nfis.org/downloads/forest_change/CA_Forest_Fire_NBAC_{year}_r9_20210810.zip",
    "https://opendata.nfis.org/downloads/forest_change/CA_Forest_Fire_NBAC_{year}.zip",
    "https://opendata.nfis.org/downloads/forest_change/nbac_{year}_20220624.zip",
    "https://opendata.nfis.org/downloads/forest_change/nbac_{year}_r9.zip",
    "https://opendata.nfis.org/downloads/forest_change/nbac_{year}.zip",
]


def _download_nbac_zip(year: int) -> bytes | None:
    """Try multiple URL templates to download NBAC shapefile zip for one year."""
    for tmpl in NBAC_URL_TEMPLATES:
        url = tmpl.format(year=year)
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                if (resp.status_code == 200 and len(resp.content) > 1000
                        and resp.content[:2] == b"PK"):  # valid zip header
                    print(f"    [OK] {url}  ({len(resp.content)/1e6:.1f} MB)")
                    return resp.content
                break  # 404 or empty → try next template
            except Exception:
                if attempt < 2:
                    continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download NBAC burn scar shapefiles (download only)"
    )
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=1985)
    parser.add_argument("--end_year", type=int, default=2024)
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

    ok = skip = fail = 0
    for year in range(args.start_year, args.end_year + 1):
        out_path = raw_dir / f"nbac_{year}.zip"
        if out_path.exists() and out_path.stat().st_size > 100:
            skip += 1
            continue

        print(f"  [{year}] Downloading…")
        data = _download_nbac_zip(year)
        if data is None:
            print(f"  [{year}] No data available")
            fail += 1
            continue

        out_path.write_bytes(data)
        ok += 1

    print(f"\n[COMPLETE] ok={ok}  skip={skip}  fail={fail}")


if __name__ == "__main__":
    main()
