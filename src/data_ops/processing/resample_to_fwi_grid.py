#!/usr/bin/env python3
"""
Resample ERA5 daily-averaged data to match the FWI grid.

Steps:
    1. Read ERA5 GeoTIFFs (WGS84, 0.25 deg)
    2. Resample to the FWI grid (e.g. EPSG:3978, 2 km)
    3. Ensure perfect spatial alignment for downstream pipelines

Output structure:
    <output_dir>/
        2t/2t_20250912.tif   (aligned to FWI)
        2d/2d_20250912.tif
        ...

Usage:
    # Basic usage with reference
    python -m src.data_ops.processing.resample_to_fwi_grid --reference fwi/fwi_20250912.tif

    # Specify resampling method
    python -m src.data_ops.processing.resample_to_fwi_grid --reference fwi/ref.tif --method cubic

    # Process single variable
    python -m src.data_ops.processing.resample_to_fwi_grid --reference fwi/ref.tif --variable 2t

    # With YAML config (reference auto-detected from fwi_dir)
    python -m src.data_ops.processing.resample_to_fwi_grid --config configs/paths_mac.yaml
"""

import sys
import os
import glob
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument
from src.utils.raster_io import get_raster_info


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

DATE_RE = re.compile(r'(20\d{6})')


def _extract_date_from_name(path_obj):
    """Extract YYYYMMDD date token from filename."""
    m = DATE_RE.search(path_obj.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None

def get_fwi_grid_params(reference_tif):
    """
    Extract grid parameters from a reference FWI raster.

    Returns:
        dict with keys: crs, transform, width, height, bounds, res
    """
    with rasterio.open(reference_tif) as src:
        return {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'res': src.res,
        }


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_fwi_grid(src_path, dst_path, fwi_params, resampling_method='bilinear'):
    """
    Resample a single ERA5 file to the FWI grid.

    Args:
        src_path: Input ERA5 GeoTIFF
        dst_path: Output path
        fwi_params: Grid parameters from get_fwi_grid_params()
        resampling_method: 'bilinear', 'cubic', 'average', 'nearest', or 'mode'
    """
    method_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average,
        'mode': Resampling.mode,
    }
    resampling = method_map.get(resampling_method, Resampling.bilinear)

    with rasterio.open(src_path) as src:
        src_data = src.read(1)

        dst_data = np.empty(
            (fwi_params['height'], fwi_params['width']),
            dtype=src_data.dtype,
        )

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=fwi_params['transform'],
            dst_crs=fwi_params['crs'],
            resampling=resampling,
        )

        profile = {
            'driver': 'GTiff',
            'height': fwi_params['height'],
            'width': fwi_params['width'],
            'count': 1,
            'dtype': dst_data.dtype,
            'crs': fwi_params['crs'],
            'transform': fwi_params['transform'],
            'compress': 'lzw',
            'nodata': src.nodata if src.nodata is not None else -9999,
        }

        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(dst_data, 1)

            # Copy and extend metadata tags
            if src.tags():
                dst.update_tags(**src.tags())
            dst.update_tags(
                resampled_from='ERA5',
                resampling_method=resampling_method,
                target_crs=str(fwi_params['crs']),
                target_resolution=f"{fwi_params['res'][0]}m",
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resample ERA5 data to FWI grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)

    parser.add_argument(
        '--reference', '-r', type=str, default=None,
        help='Path to reference FWI raster (defines target grid). '
             'If omitted, first .tif in config fwi_dir is used.',
    )
    parser.add_argument(
        '--method', '-m', default='bilinear',
        choices=['nearest', 'bilinear', 'cubic', 'average'],
        help='Resampling method (default: bilinear)',
    )
    parser.add_argument(
        '--variable', '-v', default=None,
        help='Process only a specific variable (e.g. 2t, 2d). Default: all',
    )
    parser.add_argument(
        '--input-dir', '-i', type=str, default=None,
        help='Input directory with ERA5 GeoTIFFs (default: data/era5_daily_averages)',
    )
    parser.add_argument(
        '--start-date', type=str, default=None,
        help='Optional start date YYYYMMDD or YYYY-MM-DD for filtering input files',
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='Optional end date YYYYMMDD or YYYY-MM-DD for filtering input files',
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help='Output root directory (default: from config observation_dir or data/ecmwf_observation)',
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve reference
    reference_tif = args.reference
    if reference_tif is None:
        fwi_dir = get_path(cfg, 'fwi_dir')
        candidates = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
        if candidates:
            reference_tif = candidates[0]
            print(f"Using FWI reference from config: {reference_tif}")
        else:
            print(f"Error: No .tif files found in fwi_dir ({fwi_dir}). Use --reference.")
            sys.exit(1)

    reference_tif = Path(reference_tif)
    if not reference_tif.exists():
        print(f"Error: Reference file not found: {reference_tif}")
        sys.exit(1)

    # Resolve input directory
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
    else:
        from src.config import PROJECT_ROOT
        input_dir = PROJECT_ROOT / "data" / "era5_daily_averages"

    # Resolve output directory
    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    else:
        paths_cfg = cfg.get('paths', {})
        if 'observation_dir' in paths_cfg:
            output_root = Path(get_path(cfg, 'observation_dir'))
        else:
            from src.config import PROJECT_ROOT
            output_root = PROJECT_ROOT / "data" / "ecmwf_observation"

    output_root.mkdir(parents=True, exist_ok=True)

    # Get FWI grid parameters
    print("=" * 70)
    print("RESAMPLING ERA5 TO FWI GRID")
    print("=" * 70)
    print(f"Reference FWI: {reference_tif}")

    fwi_params = get_fwi_grid_params(reference_tif)

    print(f"\nTarget grid parameters:")
    print(f"  CRS:        {fwi_params['crs']}")
    print(f"  Resolution: {fwi_params['res'][0]:.0f}m x {fwi_params['res'][1]:.0f}m")
    print(f"  Size:       {fwi_params['width']} x {fwi_params['height']} pixels")
    print(f"  Bounds:     {fwi_params['bounds']}")
    print(f"\nResampling method: {args.method}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    # Find input files
    if args.variable:
        pattern = f"{args.variable}_*.tif"
        print(f"\nProcessing only variable: {args.variable}")
    else:
        pattern = "*.tif"
        print(f"\nProcessing all variables")

    input_files = sorted(input_dir.glob(pattern))

    # Optional date-range filtering based on YYYYMMDD in filename.
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date.replace('-', ''), "%Y%m%d").date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date.replace('-', ''), "%Y%m%d").date()
    if start_date or end_date:
        filtered = []
        for f in input_files:
            d = _extract_date_from_name(f)
            if d is None:
                continue
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            filtered.append(f)
        input_files = filtered
        print(
            f"Date filter active: "
            f"{start_date if start_date else '-inf'} to {end_date if end_date else '+inf'}"
        )

    if not input_files:
        print(f"\nError: No files found matching {pattern} in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} files to process\n")

    # Process files
    success_count = 0
    fail_count = 0

    for i, src_file in enumerate(input_files, 1):
        print(f"\nProgress: {i}/{len(input_files)} ({i/len(input_files)*100:.1f}%)")
        print(f"Processing: {src_file.name}")

        var_name = src_file.stem.split("_")[0]
        var_dir = output_root / var_name
        var_dir.mkdir(parents=True, exist_ok=True)
        dst_file = var_dir / src_file.name

        # Skip existing
        if dst_file.exists():
            print(f"  [SKIP] Already exists: {dst_file.name}")
            success_count += 1
            continue

        try:
            resample_to_fwi_grid(src_file, dst_file, fwi_params, args.method)
            print(f"  [SUCCESS] -> {dst_file.name}")
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("RESAMPLING SUMMARY")
    print("=" * 70)
    print(f"Total files:  {len(input_files)}")
    print(f"Successful:   {success_count}")
    print(f"Failed:       {fail_count}")
    print(f"\nOutput root: {output_root}")

    print("\n" + "=" * 70)
    print("NEXT STEP: Verify alignment")
    print("=" * 70)
    print("Run verification to ensure perfect alignment:")
    print(f"  python -m src.data_ops.validation.verify_alignment \\")
    print(f"    --fwi-dir <fwi_dir> \\")
    print(f"    --ecmwf-dir {output_dir}")
    print("=" * 70)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
