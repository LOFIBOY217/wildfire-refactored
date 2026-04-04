#!/usr/bin/env python3
"""
Download SRTM GL1 (1 arc-second, ~30m) DEM tiles covering Canada.

Download only — saves raw .hgt files to disk.
Processing (merge, reproject, slope/aspect computation) is done separately
by processing/process_srtm_slope.py.

Source: NASA EarthData SRTMGL1 v003
Output: {terrain_raw_dir}/*.hgt  (raw SRTM tiles)

NOTE: SRTM only covers up to 60°N. Northern Canada (Yukon, NWT, Nunavut)
      will have no data. Consider CDEM for full coverage.

Usage:
    python -m src.data_ops.download.download_srtm_slope
    python -m src.data_ops.download.download_srtm_slope --config configs/paths_narval.yaml

Prerequisites:
    pip install earthaccess
    Set env vars: EARTHDATA_USERNAME, EARTHDATA_PASSWORD
"""

import argparse
import sys
import zipfile
from pathlib import Path

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


CANADA_BBOX = {"W": -141, "S": 41, "E": -52, "N": 60}  # SRTM max 60°N


def main():
    parser = argparse.ArgumentParser(
        description="Download SRTM GL1 DEM tiles for Canada (download only)"
    )
    add_config_argument(parser)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import earthaccess
    except ImportError:
        print("earthaccess required: pip install earthaccess", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    terrain_dir = Path(get_path(cfg, "terrain_dir"))
    raw_dir = terrain_dir.parent / "terrain_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(raw_dir.glob("*.hgt"))
    if len(existing) > 100 and not args.overwrite:
        print(f"[SKIP] {len(existing)} .hgt files already in {raw_dir}")
        return

    print("=" * 70)
    print("SRTM GL1 DEM — Download Only (raw .hgt tiles)")
    print("=" * 70)
    print(f"  Output : {raw_dir}")
    print(f"  NOTE: SRTM covers ≤60°N only. Northern Canada will be missing.")
    print(f"  NOTE: Run processing/process_srtm_slope.py after download")
    print("=" * 70)

    print("\n  Authenticating with NASA Earthdata…")
    earthaccess.login(strategy="environment")

    print("  Searching SRTMGL1v003 tiles…")
    results = earthaccess.search_data(
        short_name="SRTMGL1",
        version="003",
        bounding_box=(CANADA_BBOX["W"], CANADA_BBOX["S"],
                      CANADA_BBOX["E"], CANADA_BBOX["N"]),
    )
    print(f"  Found {len(results)} tiles")

    if not results:
        print("[ERROR] No SRTM tiles found", file=sys.stderr)
        sys.exit(1)

    # Download in batches
    batch = args.batch_size
    for bi in range(0, len(results), batch):
        chunk = results[bi:bi + batch]
        print(f"  Downloading batch {bi//batch+1}/"
              f"{(len(results)+batch-1)//batch}: {len(chunk)} tiles…")
        earthaccess.download(chunk, local_path=str(raw_dir))

    # Unzip if needed
    for zp in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(raw_dir)

    hgt_count = len(list(raw_dir.glob("*.hgt")))
    print(f"\n[COMPLETE] {hgt_count} .hgt files in {raw_dir}")


if __name__ == "__main__":
    main()
