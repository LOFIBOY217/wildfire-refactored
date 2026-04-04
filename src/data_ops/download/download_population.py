#!/usr/bin/env python3
"""
Download WorldPop 2020 population density for Canada.

Download only — saves raw GeoTIFF to disk.
Processing (clip, reproject, log1p) is done separately by
processing/process_population.py.

Source: WorldPop 2020 Canada 1km (CC BY 4.0)
Output: {data_dir}/population_raw/can_pd_2020_1km.tif

Usage:
    python -m src.data_ops.download.download_population
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


WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population_Density/"
    "Global_2000_2020_1km/2020/CAN/can_pd_2020_1km.tif"
)


def main():
    parser = argparse.ArgumentParser(
        description="Download WorldPop population density for Canada (download only)"
    )
    add_config_argument(parser)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        pop_tif = Path(get_path(cfg, "population_tif"))
    except (KeyError, TypeError):
        pop_tif = Path("data/population_density.tif")

    raw_dir = pop_tif.parent / "population_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "can_pd_2020_1km.tif"

    if raw_path.exists() and raw_path.stat().st_size > 1_000_000:
        print(f"[SKIP] Raw file already exists: {raw_path}")
        print(f"  Run processing/process_population.py to reproject")
        return

    print("=" * 70)
    print("WorldPop Population Density — Download Only")
    print("=" * 70)
    print(f"  URL    : {WORLDPOP_URL}")
    print(f"  Output : {raw_path}")
    print(f"  NOTE: Run processing/process_population.py after download")
    print("=" * 70)

    print("\n  Downloading…")
    resp = requests.get(WORLDPOP_URL, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(raw_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB "
                      f"({100*downloaded/total:.0f}%)", end="", flush=True)
    print()
    print(f"\n[COMPLETE] {raw_path}  ({raw_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
