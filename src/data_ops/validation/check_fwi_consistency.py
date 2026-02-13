#!/usr/bin/env python3
"""
Check FWI GeoTIFF consistency.

Verifies that all .tif files in a directory match a reference file
in CRS, resolution, extent, and dimensions.

Refactored from: data/check_FWI_consistency.py
  - Uses src.utils.raster_io.get_raster_info
  - Loads paths from YAML config via src.config

Usage:
  python -m src.data_ops.validation.check_fwi_consistency
  python -m src.data_ops.validation.check_fwi_consistency --config configs/paths_mac.yaml
  python -m src.data_ops.validation.check_fwi_consistency --dir /path/to/tifs --ref /path/to/ref.tif
"""

import os
import argparse
import sys
from pathlib import Path

import numpy as np

from src.config import load_config, get_path, add_config_argument
from src.utils.raster_io import get_raster_info


def check_tif_consistency(directory: str, ref_file: str):
    """
    Check all .tif files in a directory against a reference file for
    CRS, resolution, extent, and dimension consistency.

    Args:
        directory: Path to directory containing .tif files to check.
        ref_file: Path to the reference .tif file.
    """
    if not os.path.exists(ref_file):
        print(f"Error: Reference file not found: {ref_file}")
        return

    # 1. Read reference metadata
    ref_info = get_raster_info(ref_file)
    ref_crs = ref_info['crs']
    ref_bounds = ref_info['bounds']
    ref_width = ref_info['width']
    ref_height = ref_info['height']
    ref_transform = ref_info['transform']
    ref_res = (abs(ref_transform.a), abs(ref_transform.e))

    print("=" * 60)
    print(f"Reference file: {os.path.basename(ref_file)}")
    print(f"  CRS:        {ref_crs}")
    print(f"  Resolution: {ref_res}")
    print(f"  Bounds:     {ref_bounds}")
    print(f"  Dimensions: {ref_width} x {ref_height} (W x H)")
    print("=" * 60)

    tif_files = sorted(Path(directory).glob("*.tif"))
    if not tif_files:
        print("No .tif files found in the specified directory.")
        return

    inconsistent_files = []

    for tif in tif_files:
        # Skip the reference file itself
        if os.path.abspath(str(tif)) == os.path.abspath(ref_file):
            continue

        filename = tif.name
        issues = []

        try:
            info = get_raster_info(str(tif))

            # Check CRS
            if info['crs'] != ref_crs:
                issues.append(f"CRS mismatch (got: {info['crs']})")

            # Check resolution (allow tiny floating point error)
            file_res = (abs(info['transform'].a), abs(info['transform'].e))
            if not np.allclose(file_res, ref_res):
                issues.append(f"Resolution mismatch (got: {file_res})")

            # Check dimensions
            if (info['width'], info['height']) != (ref_width, ref_height):
                issues.append(f"Dimension mismatch (got: {info['width']}x{info['height']})")

            # Check bounds
            if not np.allclose(info['bounds'], ref_bounds, atol=1e-6):
                issues.append("Bounds mismatch")

        except Exception as e:
            issues.append(f"Cannot read file: {e}")

        if issues:
            print(f"[FAIL] {filename}")
            for issue in issues:
                print(f"   - {issue}")
            inconsistent_files.append(filename)
        else:
            print(f"[OK]   {filename}")

    print("\n" + "=" * 60)
    if not inconsistent_files:
        print("All files passed. Format is consistent.")
    else:
        print(f"Check complete: {len(inconsistent_files)} file(s) inconsistent.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Check FWI GeoTIFF consistency against a reference file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_config_argument(parser)

    parser.add_argument(
        '--dir',
        default=None,
        help='Directory containing .tif files to check (overrides config fwi_dir)'
    )

    parser.add_argument(
        '--ref',
        default=None,
        help='Path to reference .tif file (defaults to first file in directory)'
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve directory: CLI overrides config
    if args.dir:
        target_dir = args.dir
    else:
        target_dir = get_path(cfg, 'fwi_dir')

    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)

    # Resolve reference file
    if args.ref:
        reference_tif = args.ref
    else:
        # Use first .tif in directory as reference
        tif_files = sorted(Path(target_dir).glob("*.tif"))
        if not tif_files:
            print(f"Error: No .tif files found in {target_dir}")
            sys.exit(1)
        reference_tif = str(tif_files[0])
        print(f"No --ref specified, using first file as reference: {Path(reference_tif).name}\n")

    check_tif_consistency(target_dir, reference_tif)


if __name__ == "__main__":
    main()
