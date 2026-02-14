#!/usr/bin/env python3
"""
Process ERA5 GRIB files to daily averaged GeoTIFF files.

Steps:
    1. Read hourly ERA5 data from GRIB files (24 timesteps per day)
    2. Compute daily averages for each variable
    3. Save as individual GeoTIFF files for each variable and date

Output structure:
    <output_dir>/
        2t_20250912.tif   (2m temperature, daily avg)
        2d_20250912.tif   (2m dewpoint, daily avg)
        tcw_20250912.tif  (total column water, daily avg)
        ...

Usage:
    # Process all files using config paths
    python -m src.data_ops.processing.era5_to_daily

    # Single file
    python -m src.data_ops.processing.era5_to_daily --input era5_sfc_2025_09_12.grib

    # Custom directories
    python -m src.data_ops.processing.era5_to_daily \\
        --input-dir download_ecmwf_reanalysis_observations --output-dir era5_daily_averages

    # With YAML config
    python -m src.data_ops.processing.era5_to_daily --config configs/paths_mac.yaml
"""

import sys
import os
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import cfgrib
import rasterio
from rasterio.transform import from_bounds

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


# Variable mapping: GRIB short name -> output name
VARIABLE_MAP = {
    't2m': '2t',      # 2m temperature
    'd2m': '2d',      # 2m dewpoint
    'tcw': 'tcw',     # total column water
    'swvl1': 'sm20',  # soil moisture layer 1 (approximate sm20)
    'stl1': 'st20',   # soil temperature layer 1 (approximate st20)
}


def process_single_grib(grib_path, output_dir):
    """
    Process a single ERA5 GRIB file to daily-averaged GeoTIFFs.

    Args:
        grib_path: Path to ERA5 GRIB file
        output_dir: Output directory for GeoTIFF files

    Returns:
        True on success, False on failure
    """
    grib_path = Path(grib_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract date from filename: era5_sfc_2025_09_12.grib
    filename = grib_path.stem
    try:
        date_part = filename.split('era5_sfc_')[1]  # "2025_09_12"
        date_str = date_part.replace('_', '')        # "20250912"
    except Exception:
        print(f"[ERROR] Cannot parse date from filename: {filename}")
        return False

    print(f"\n{'='*70}")
    print(f"Processing: {grib_path.name}")
    print(f"Date: {date_str}")
    print(f"{'='*70}")

    try:
        print(f"  Opening GRIB file...")
        grib_datasets = cfgrib.open_datasets(str(grib_path))
        print(f"  Found {len(grib_datasets)} dataset(s)")

        # Collect all variables from all datasets
        datasets = {}
        for dataset_idx, ds in enumerate(grib_datasets):
            print(f"  Scanning dataset {dataset_idx}...")
            for grib_var, out_var in VARIABLE_MAP.items():
                if grib_var in ds.data_vars:
                    datasets[out_var] = ds
                    print(f"    Found {grib_var} -> {out_var}")

        if not datasets:
            print("[ERROR] No matching variables found in GRIB file")
            return False

        print(f"  Total variables loaded: {len(datasets)}")

        # Process each variable
        for out_var, ds in datasets.items():
            try:
                var_name = list(ds.data_vars)[0]
                data = ds[var_name]

                # Daily average
                if 'time' in data.dims:
                    daily_avg = data.mean(dim='time')
                    print(f"  Averaged {len(data.time)} timesteps for {out_var}")
                else:
                    daily_avg = data
                    print(f"  Single timestep for {out_var}, no averaging needed")

                # Spatial coordinates
                lats = ds.latitude.values
                lons = ds.longitude.values

                output_file = output_dir / f"{out_var}_{date_str}.tif"

                data_array = daily_avg.values
                if data_array.ndim == 2:
                    pass
                elif data_array.ndim == 3:
                    data_array = np.squeeze(data_array)

                # ERA5 is regular lat/lon grid
                lat_min, lat_max = lats.min(), lats.max()
                lon_min, lon_max = lons.min(), lons.max()

                # ERA5 typically has latitude descending (N to S)
                if lats[0] > lats[-1]:
                    data_array = np.flipud(data_array)
                    lat_min, lat_max = lat_max, lat_min

                height, width = data_array.shape
                transform = from_bounds(
                    lon_min, lat_min, lon_max, lat_max, width, height
                )

                # Write GeoTIFF
                with rasterio.open(
                    output_file, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=data_array.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    compress='lzw',
                ) as dst:
                    dst.write(data_array, 1)

                # Add metadata tags
                with rasterio.open(output_file, 'r+') as dst:
                    dst.update_tags(
                        date=date_str,
                        variable=out_var,
                        source='ERA5',
                        processing='daily_average',
                    )

                print(f"  Saved: {output_file.name} ({data_array.shape})")

            except Exception as e:
                print(f"  Failed to process {out_var}: {e}")
                traceback.print_exc()

        return True

    except Exception as e:
        print(f"[ERROR] Failed to process {grib_path}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process ERA5 GRIB files to daily averaged GeoTIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)

    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a single ERA5 GRIB file (overrides --input-dir)",
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory containing ERA5 GRIB files (default: from config or 'download_ecmwf_reanalysis_observations')",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for daily GeoTIFFs (default: 'era5_daily_averages')",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve paths
    if args.input is not None:
        grib_files = [Path(args.input)]
    else:
        input_dir_str = args.input_dir
        if input_dir_str is None:
            # Use project-relative default
            from src.config import PROJECT_ROOT
            input_dir_str = str(PROJECT_ROOT / "download_ecmwf_reanalysis_observations")
        input_dir = Path(input_dir_str)
        grib_files = sorted(input_dir.glob("era5_sfc_*.grib"))

    output_dir_str = args.output_dir
    if output_dir_str is None:
        from src.config import PROJECT_ROOT
        output_dir_str = str(PROJECT_ROOT / "era5_daily_averages")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not grib_files:
        print("Error: No GRIB files found")
        if args.input is None:
            print(f"Looking in: {input_dir_str}")
        sys.exit(1)

    print("=" * 70)
    print("ERA5 DAILY AVERAGING PROCESSOR")
    print("=" * 70)
    if args.input is None:
        print(f"Input directory:  {input_dir_str}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(grib_files)}")
    print("=" * 70)

    # Process files
    success_count = 0
    fail_count = 0

    for i, grib_file in enumerate(grib_files, 1):
        print(f"\nProgress: {i}/{len(grib_files)} ({i/len(grib_files)*100:.1f}%)")

        success = process_single_grib(grib_file, output_dir)
        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total files:  {len(grib_files)}")
    print(f"Successful:   {success_count}")
    print(f"Failed:       {fail_count}")
    print(f"\nOutput: {output_dir}")

    output_files = list(output_dir.glob("*.tif"))
    print(f"Generated {len(output_files)} GeoTIFF files")

    if output_files:
        print("\nSample output files:")
        for f in sorted(output_files)[:10]:
            print(f"  {f.name}")
        if len(output_files) > 10:
            print(f"  ... and {len(output_files) - 10} more")

    print("\n" + "=" * 70)
    print("NEXT STEP: Resample to FWI grid")
    print("=" * 70)
    print("These files are in WGS84 (EPSG:4326) at 0.25 deg resolution.")
    print("Resample them to match the FWI grid:")
    print("  python -m src.data_ops.processing.resample_to_fwi_grid --reference <fwi_ref.tif>")
    print("=" * 70)


if __name__ == "__main__":
    main()
