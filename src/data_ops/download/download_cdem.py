#!/usr/bin/env python3
"""
Download CDEM (Canadian Digital Elevation Model) tiles from NRCan.

CDEM covers ALL of Canada including >60°N (unlike SRTM which stops at 60°N).
Resolution: ~20m (0.75 arcsec), datum: NAD83/CSRS.

Source: NRCan Open Canada via HTTPS FTP mirror
URL:    https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/

Output: {cdem_raw_dir}/*.tif  (raw CDEM tiles)

Processing (reproject, slope/aspect computation) is done separately
by processing/process_cdem_slope.py.

Usage:
    python -m src.data_ops.download.download_cdem
    python -m src.data_ops.download.download_cdem --config configs/paths_narval.yaml

Prerequisites:
    pip install requests
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


BASE_URL = "https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/"


def _list_subdirs(url, session):
    """Parse HTML directory listing for NTS sheet subdirectories (001/, 002/, ...)."""
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    # NTS sheet directories are numeric: 001/, 002/, ..., 120/
    return re.findall(r'href="(\d{3}/)"', resp.text)


def _find_zips_in_dir(url, session):
    """Find .zip files in a CDEM NTS sheet subdirectory."""
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    zips = re.findall(r'href="([^"]+\.zip)"', resp.text)
    return [url + z for z in zips]


def _download_file(url, out_path, session):
    """Download a single file with resume support."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        return "skip"
    tmp = out_path.with_suffix(".tmp")
    try:
        resp = session.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(out_path)
        return "ok"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"err: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Download CDEM tiles for Canada (download only)"
    )
    add_config_argument(parser)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    terrain_dir = Path(get_path(cfg, "terrain_dir"))
    raw_dir = terrain_dir.parent / "cdem_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = list(raw_dir.glob("*.zip"))
    if len(existing) > 100 and not args.overwrite:
        print(f"[SKIP] {len(existing)} .zip files already in {raw_dir}")
        return

    print("=" * 70)
    print("CDEM (Canadian Digital Elevation Model) — Download")
    print("=" * 70)
    print(f"  Source : {BASE_URL}")
    print(f"  Output : {raw_dir}")
    print(f"  Workers: {args.workers}")
    print(f"  CDEM covers ALL of Canada including >60°N")
    print("=" * 70)

    session = requests.Session()
    session.headers["User-Agent"] = "wildfire-refactored/1.0"

    # Step 1: crawl directory listing for subdirectories
    print("\n  Crawling directory listing...")
    subdirs = _list_subdirs(BASE_URL, session)
    print(f"  Found {len(subdirs)} subdirectories")

    if not subdirs:
        print("[ERROR] No CDEM subdirectories found. URL structure may have changed.",
              file=sys.stderr)
        sys.exit(1)

    # Step 2: find .zip files in each subdirectory
    print("  Scanning for .zip files...")
    all_urls = []
    for i, sd in enumerate(subdirs):
        try:
            zips = _find_zips_in_dir(BASE_URL + sd, session)
            all_urls.extend(zips)
        except Exception as e:
            print(f"  [WARN] {sd}: {e}")
        if (i + 1) % 20 == 0:
            print(f"    scanned {i+1}/{len(subdirs)} dirs, {len(all_urls)} zips so far")

    print(f"  Total zip files to download: {len(all_urls)}")

    if not all_urls:
        print("[ERROR] No .zip files found in CDEM directories", file=sys.stderr)
        sys.exit(1)

    # Step 3: download in parallel
    done, skipped, failed = 0, 0, 0

    def _task(url):
        fname = url.split("/")[-1]
        out_path = raw_dir / fname
        return _download_file(url, out_path, session)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, u): u for u in all_urls}
        for fut in as_completed(futures):
            result = fut.result()
            if result == "skip":
                skipped += 1
            elif result == "ok":
                done += 1
            else:
                failed += 1
            total = done + skipped + failed
            if total % 20 == 0:
                print(f"  Progress: {total}/{len(all_urls)}  "
                      f"(new={done} skip={skipped} fail={failed})")

    # Step 4: unzip all downloaded files to extract .tif
    print(f"\n  Extracting .tif from zip files...")
    import zipfile
    tif_extracted = 0
    for zp in raw_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                for name in zf.namelist():
                    if name.endswith(".tif"):
                        zf.extract(name, raw_dir)
                        tif_extracted += 1
        except Exception as e:
            print(f"  [WARN] {zp.name}: {e}")

    tif_count = len(list(raw_dir.glob("**/*.tif")))
    print(f"\n[COMPLETE] {tif_count} .tif files in {raw_dir}")
    print(f"  Downloaded: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"  Extracted: {tif_extracted} TIFs from zips")
    if failed > 0:
        print("  [WARN] Some files failed — re-run to retry")


if __name__ == "__main__":
    main()
