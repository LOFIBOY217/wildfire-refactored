#!/usr/bin/env python3
"""
Scan GeoTIFF directories for corrupt / unreadable files.
=========================================================
Tests every *.tif file by opening it with rasterio and reading one pixel.
Files that fail are reported with their error message.

Usage:
    # Scan FWI and observation dirs from default config
    python scripts/check_corrupt_tifs.py

    # Explicit directories (can repeat --dir)
    python scripts/check_corrupt_tifs.py \\
        --dir data/fwi_data \\
        --dir data/ecmwf_observation/2t \\
        --dir data/ecmwf_observation/2d

    # With a specific config
    python scripts/check_corrupt_tifs.py --config configs/paths_windows.yaml

    # Save CSV report (handy on HPC)
    python scripts/check_corrupt_tifs.py --output_csv corrupt_report.csv

Exit code:
    0  all files OK
    1  one or more corrupt files found
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import rasterio

# ---------------------------------------------------------------------------
# Resolve src on the path so this script works from any working directory
# ---------------------------------------------------------------------------
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "config.py").exists():
        sys.path.insert(0, str(_parent))
        break

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    load_config = None
    get_path = None
    add_config_argument = None


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_tif(path):
    """
    Try to open *path* and read its first band.

    Returns:
        (ok: bool, error_msg: str)  error_msg is "" when ok=True
    """
    try:
        with rasterio.open(path) as src:
            src.read(1)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def scan_directory(directory, pattern="**/*.tif", recursive=True):
    """Return sorted list of all *.tif paths under *directory*."""
    return sorted(
        glob.glob(os.path.join(directory, pattern), recursive=recursive)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan GeoTIFF directories for corrupt / unreadable files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    if add_config_argument:
        add_config_argument(parser)
    else:
        parser.add_argument("--config", default=None)

    parser.add_argument(
        "--dir", dest="dirs", action="append", default=None,
        help="Directory to scan (may be repeated). Overrides config.",
    )
    parser.add_argument(
        "--output_csv", default=None,
        help="Optional path to write a CSV report of all corrupt files.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve directories to scan
    # -----------------------------------------------------------------------
    scan_dirs = args.dirs or []

    if not scan_dirs and load_config is not None:
        cfg = load_config(args.config)
        fwi_dir = get_path(cfg, "fwi_dir")
        obs_root = get_path(cfg, "observation_dir") if "observation_dir" in cfg.get("paths", {}) \
                   else get_path(cfg, "ecmwf_dir")

        scan_dirs.append(fwi_dir)
        # Add per-variable subdirs if they exist, otherwise the root
        for var in ("2t", "2d"):
            subdir = os.path.join(obs_root, var)
            if os.path.isdir(subdir):
                scan_dirs.append(subdir)
            else:
                scan_dirs.append(obs_root)
                break

    if not scan_dirs:
        print("ERROR: No directories to scan. Pass --dir or use a config file.")
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    unique_dirs = []
    for d in scan_dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)
    scan_dirs = unique_dirs

    # -----------------------------------------------------------------------
    # Collect all tif files
    # -----------------------------------------------------------------------
    all_files = []
    for d in scan_dirs:
        if not os.path.isdir(d):
            print(f"  WARNING: directory not found, skipping: {d}")
            continue
        files = scan_directory(d)
        all_files.extend(files)
        print(f"  {d}  ->  {len(files)} file(s)")

    if not all_files:
        print("No *.tif files found in any of the specified directories.")
        sys.exit(0)

    print()
    print("=" * 72)
    print(f"CORRUPT TIF SCANNER  —  {len(all_files)} files to check")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Check every file
    # -----------------------------------------------------------------------
    corrupt = []
    total = len(all_files)
    for i, path in enumerate(all_files, 1):
        ok, err = check_tif(path)
        if not ok:
            corrupt.append((path, err))
            # Clear the progress line, then print the corrupt file on its own line
            print(f"\r{' ' * 78}\r  [CORRUPT]  {path}")
            print(f"             {err}")

        # Overwrite the same line with current progress
        bar_done = int(40 * i / total)
        bar = "#" * bar_done + "-" * (40 - bar_done)
        print(
            f"\r  [{bar}] {i}/{total}  corrupt: {len(corrupt)}",
            end="",
            flush=True,
        )

    print()  # newline after the progress bar finishes

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"  Total checked : {len(all_files)}")
    print(f"  OK            : {len(all_files) - len(corrupt)}")
    print(f"  CORRUPT       : {len(corrupt)}")
    print("=" * 72)

    if corrupt:
        print("\nCorrupt files:")
        for path, err in corrupt:
            print(f"  {path}")
            print(f"    -> {err}")

    # -----------------------------------------------------------------------
    # Optional CSV report
    # -----------------------------------------------------------------------
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "error"])
            for path, err in corrupt:
                writer.writerow([path, err])
        print(f"\nCorrupt file list saved: {args.output_csv}")

    sys.exit(0 if not corrupt else 1)


if __name__ == "__main__":
    main()
