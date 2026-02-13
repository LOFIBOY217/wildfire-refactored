#!/usr/bin/env python3
"""
Convert ECMWF GRIB to FWI GeoTIFF using rasterio only (no pygrib, no cfgrib).

This is the lightweight fallback converter that uses rasterio's built-in GRIB
driver to read subdatasets, extract metadata, reproject to the FWI reference
grid, and write single-band GeoTIFFs.

Usage:
    # Single file
    python -m src.data_ops.processing.ecmwf_to_fwi --grib ecmwf_20240901.grib --ref fwi_reference.tif

    # Directory batch
    python -m src.data_ops.processing.ecmwf_to_fwi --grib ecmwf_data/ --ref fwi_reference.tif

    # With config
    python -m src.data_ops.processing.ecmwf_to_fwi --grib ecmwf_data/ --config configs/paths_mac.yaml
"""

import os
import sys
import glob
import re
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from src.config import load_config, get_path, add_config_argument
from src.utils.raster_io import clean_array, write_geotiff


# ---------------------------------------------------------------------------
# FWI reference grid
# ---------------------------------------------------------------------------

def read_fwi_reference(ref_path):
    """
    Open FWI GeoTIFF reference file and return the open dataset + output profile.

    Returns:
        ref: Open rasterio DatasetReader (caller must close)
        profile: dict suitable for writing aligned output GeoTIFFs
    """
    ref = rasterio.open(ref_path)
    profile = ref.profile.copy()
    profile.update(
        dtype='float32',
        nodata=np.nan,
        count=1,
        compress='deflate',
        predictor=3,
        tiled=True,
    )
    return ref, profile


# ---------------------------------------------------------------------------
# GRIB subdataset helpers
# ---------------------------------------------------------------------------

def list_grib_subdatasets(grib_path):
    """List all subdatasets in a GRIB file."""
    with rasterio.open(str(grib_path)) as src:
        subdatasets = src.subdatasets
        if not subdatasets:
            return [str(grib_path)]
        return subdatasets


def extract_metadata_from_subdataset(sub_path):
    """Extract parameter name, level, and date from a GRIB subdataset."""
    info = {
        'path': sub_path,
        'param': 'unknown',
        'level': 0,
        'date': None,
    }

    try:
        with rasterio.open(sub_path) as src:
            tags = src.tags()

            # Parameter name
            for key in ['GRIB_ELEMENT', 'GRIB_SHORT_NAME', 'GRIB_COMMENT']:
                if key in tags:
                    info['param'] = tags[key]
                    break

            # Level
            for key in ['GRIB_LAYER', 'GRIB_SHORT_NAME_LEVEL']:
                if key in tags:
                    info['level'] = tags[key]
                    break

            # Date/time
            for key in ['GRIB_VALID_TIME', 'GRIB_REF_TIME']:
                if key in tags and tags[key]:
                    try:
                        dt = datetime.fromisoformat(tags[key].replace('Z', ''))
                        info['date'] = dt.strftime('%Y%m%dT%H%M')
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------

def reproject_subdataset(sub_path, ref_ds, profile):
    """Reproject a single GRIB subdataset to the FWI reference grid."""
    with rasterio.open(sub_path) as src:
        data = src.read(1).astype(np.float32)

        # Handle nodata
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)

        # Destination array
        dst_array = np.empty((ref_ds.height, ref_ds.width), dtype=np.float32)
        dst_array.fill(np.nan)

        # Reproject
        reproject(
            source=data,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs if src.crs else 'EPSG:4326',
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        return dst_array


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_grib_file(grib_path, ref_path, output_dir):
    """Process a single GRIB file: read subdatasets, reproject, write GeoTIFFs."""
    grib_path = Path(grib_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {grib_path.name}")

    # Load reference
    ref, profile = read_fwi_reference(ref_path)

    # List subdatasets
    subdatasets = list_grib_subdatasets(grib_path)
    print(f"Found {len(subdatasets)} layers")

    # Process each subdataset
    for i, sub in enumerate(subdatasets, 1):
        try:
            print(f"  [{i}/{len(subdatasets)}]", end=" ", flush=True)

            info = extract_metadata_from_subdataset(sub)
            reprojected = reproject_subdataset(sub, ref, profile)

            # Build output filename
            param = info['param'].replace(' ', '_')
            level = info['level']
            date_tag = info['date'] or f"msg{i}"
            out_name = f"{param}_lev{level}_{date_tag}.tif"
            out_path = output_dir / out_name

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(reprojected, 1)

            print(f"{info['param']} -> {out_name}")

        except Exception as e:
            print(f"ERROR: {e}")

    ref.close()
    print(f"Output: {output_dir}/")


def batch_process(input_dir, ref_path, output_dir):
    """Process all GRIB files in a directory."""
    grib_files = sorted(glob.glob(os.path.join(input_dir, "*.grib")))

    if not grib_files:
        print(f"No GRIB files in {input_dir}")
        return

    print(f"Processing {len(grib_files)} files\n")

    for grib_file in grib_files:
        try:
            process_grib_file(grib_file, ref_path, output_dir)
        except Exception as e:
            print(f"FAILED {grib_file}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert ECMWF GRIB to FWI GeoTIFF (rasterio-only, no pygrib)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)

    parser.add_argument(
        "--grib", type=str, required=True,
        help="Path to a single GRIB file or a directory of GRIB files",
    )
    parser.add_argument(
        "--ref", type=str, default=None,
        help="Path to the FWI reference GeoTIFF (overrides config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: from config or 'fwi_output')",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve reference path
    ref_path = args.ref
    if ref_path is None:
        fwi_dir = get_path(cfg, 'fwi_dir')
        # Use first .tif in fwi_dir as reference
        candidates = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
        if not candidates:
            print(f"Error: No .tif files found in fwi_dir ({fwi_dir}). "
                  f"Use --ref to specify a reference file.")
            sys.exit(1)
        ref_path = candidates[0]
        print(f"Using reference from config: {ref_path}")

    if not os.path.exists(ref_path):
        print(f"Error: Reference file not found: {ref_path}")
        sys.exit(1)

    # Resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(get_path(cfg, 'output_dir'), "ecmwf_fwi_rasterio")

    # Process
    input_path = args.grib
    if os.path.isdir(input_path):
        batch_process(input_path, ref_path, output_dir)
    else:
        process_grib_file(input_path, ref_path, output_dir)


if __name__ == "__main__":
    main()
