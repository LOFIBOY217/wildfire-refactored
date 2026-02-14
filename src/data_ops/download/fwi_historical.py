#!/usr/bin/env python3
"""
Download historical FWI data from Copernicus CEMS (ERA5-based reanalysis).

The CWFIS WCS service only provides FWI from ~2023 onward.
For 2015-2022, we use the Copernicus Climate Data Store (CDS) dataset:
    cems-fire-historical-v1  (ERA5-based, 0.25 deg, 1940-present)

The downloaded NetCDFs are then converted to per-day GeoTIFFs and
reprojected to the FWI grid (EPSG:3978, 2709x2281) so they match the
CWFIS rasters.

Prerequisites:
    pip install cdsapi netCDF4
    # Create ~/.cdsapirc  or set CDS_API_KEY env var
    # Accept licence at: https://cds.climate.copernicus.eu/datasets/cems-fire-historical-v1

Usage:
    # Download + convert one year
    python -m src.data_ops.download.fwi_historical --year 2020

    # Download full range 2015-2022
    python -m src.data_ops.download.fwi_historical --start 2015 --end 2022

    # Download specific months (fire season only)
    python -m src.data_ops.download.fwi_historical --start 2015 --end 2022 --months 5 6 7 8 9 10

    # Skip reprojection (keep raw NetCDFs only)
    python -m src.data_ops.download.fwi_historical --year 2020 --no-reproject

    # With custom config
    python -m src.data_ops.download.fwi_historical --year 2020 --config configs/paths_mac.yaml
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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


# ------------------------------------------------------------------ #
# CDS download
# ------------------------------------------------------------------ #

# Canada bounding box: [North, West, South, East]
CANADA_AREA = [84, -142, 41, -52]

# All days in a month template
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
DEFAULT_CDS_API_KEY = "d952a10c-f9c0-4ff3-92e1-aac8756dd123"


def download_fwi_year_month(client, year, month, output_dir, area=None):
    """
    Download FWI for one year-month from CDS.

    Args:
        client: cdsapi.Client instance
        year: int or str, e.g. 2020
        month: int or str, e.g. 7
        output_dir: Path to save NetCDF
        area: [N, W, S, E] bounding box (default: Canada)

    Returns:
        Path to downloaded file, or None on failure.
    """
    area = area or CANADA_AREA
    year_str = str(year)
    month_str = f"{int(month):02d}"

    outfile = output_dir / f"fwi_historical_{year_str}{month_str}.nc"

    if outfile.exists():
        print(f"  [SKIP] {outfile.name} already exists")
        return outfile

    request = {
        'product_type': 'reanalysis',
        'variable': 'fire_weather_index',
        'system_version': '4_1',
        'dataset_type': 'consolidated_dataset',
        'year': year_str,
        'month': month_str,
        'day': ALL_DAYS,
        'grid': '0.25/0.25',
        'area': area,
        'data_format': 'netcdf_legacy',
    }

    print(f"  [DOWNLOAD] {year_str}-{month_str} ...", end=" ", flush=True)
    t0 = time.time()

    try:
        client.retrieve('cems-fire-historical-v1', request, str(outfile))
        elapsed = time.time() - t0
        size_mb = outfile.stat().st_size / 1024 / 1024
        print(f"[OK] {size_mb:.1f}MB in {elapsed:.0f}s")
        return outfile
    except Exception as e:
        print(f"[FAIL] {e}")
        # Clean up partial file
        if outfile.exists():
            outfile.unlink()
        return None


def download_fwi_range(client, start_year, end_year, months, output_dir, area=None):
    """
    Download FWI for a range of years and months.

    Returns:
        list of successfully downloaded file paths
    """
    files = []
    total = (end_year - start_year + 1) * len(months)
    done = 0

    for year in range(start_year, end_year + 1):
        for month in months:
            done += 1
            print(f"\n[{done}/{total}] Year {year}, Month {month:02d}")
            f = download_fwi_year_month(client, year, month, output_dir, area)
            if f:
                files.append(f)

    return files


# ------------------------------------------------------------------ #
# NetCDF -> per-day GeoTIFF conversion + reprojection
# ------------------------------------------------------------------ #

def nc_to_daily_tifs(nc_path, tif_dir, reproject_to_fwi=True, fwi_reference=None):
    """
    Convert a monthly FWI NetCDF to per-day GeoTIFFs.

    If reproject_to_fwi=True and fwi_reference is provided, reprojects
    each day from WGS84 0.25deg to the FWI grid (EPSG:3978, 2709x2281).

    Args:
        nc_path: Path to NetCDF file
        tif_dir: Output directory for GeoTIFFs
        reproject_to_fwi: Whether to reproject to FWI grid
        fwi_reference: Path to a reference FWI GeoTIFF (for target grid)

    Returns:
        int: Number of files created
    """
    import netCDF4
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    ds = netCDF4.Dataset(str(nc_path))

    # Find FWI variable
    fwi_var = None
    for name in ['fwi', 'FWI', 'fire_weather_index', 'mark_4']:
        if name in ds.variables:
            fwi_var = name
            break

    if fwi_var is None:
        # Try any variable that isn't a coordinate
        coord_names = {'time', 'latitude', 'longitude', 'lat', 'lon'}
        for name in ds.variables:
            if name not in coord_names:
                fwi_var = name
                break

    if fwi_var is None:
        print(f"  [WARN] Cannot find FWI variable in {nc_path.name}, available: {list(ds.variables.keys())}")
        ds.close()
        return 0

    # Get time, lat, lon
    time_var = ds.variables['time']
    times = netCDF4.num2date(time_var[:], time_var.units, time_var.calendar if hasattr(time_var, 'calendar') else 'standard')

    # Handle lat/lon naming
    lat_name = 'latitude' if 'latitude' in ds.variables else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.variables else 'lon'
    lats = ds.variables[lat_name][:]
    lons = ds.variables[lon_name][:]

    fwi_data = ds.variables[fwi_var]  # [time, lat, lon]

    # Source grid info (WGS84)
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    nlat, nlon = len(lats), len(lons)

    # Check if lat is descending (north-up)
    lat_descending = lats[0] > lats[-1]

    src_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nlon, nlat)
    src_crs = 'EPSG:4326'

    # Load FWI reference grid if reprojecting
    dst_profile = None
    if reproject_to_fwi and fwi_reference and Path(fwi_reference).exists():
        with rasterio.open(fwi_reference) as ref:
            dst_profile = ref.profile.copy()
            dst_transform = ref.transform
            dst_crs = ref.crs
            dst_height = ref.height
            dst_width = ref.width
    elif reproject_to_fwi:
        print("  [WARN] No FWI reference provided, saving in WGS84")
        reproject_to_fwi = False

    count = 0
    for t_idx in range(len(times)):
        dt = times[t_idx]
        # Handle different datetime types
        if hasattr(dt, 'strftime'):
            date_str = dt.strftime("%Y%m%d")
        else:
            date_str = str(dt)[:10].replace('-', '')

        outfile = tif_dir / f"fwi_{date_str}.tif"
        if outfile.exists():
            count += 1
            continue

        # Read slice
        arr = np.array(fwi_data[t_idx, :, :], dtype=np.float32)

        # Ensure north-up orientation
        if not lat_descending:
            arr = np.flipud(arr)

        # Replace fill values with NaN
        if hasattr(fwi_data, '_FillValue'):
            arr[arr == fwi_data._FillValue] = np.nan
        arr[~np.isfinite(arr)] = np.nan

        if reproject_to_fwi:
            # Reproject to FWI grid
            dst_arr = np.empty((dst_height, dst_width), dtype=np.float32)
            dst_arr[:] = np.nan

            reproject(
                source=arr,
                destination=dst_arr,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

            out_profile = dst_profile.copy()
            out_profile.update(dtype='float32', count=1, compress='lzw', nodata=np.nan)

            with rasterio.open(outfile, 'w', **out_profile) as dst:
                dst.write(dst_arr, 1)
        else:
            # Save in WGS84
            profile = {
                'driver': 'GTiff',
                'height': nlat,
                'width': nlon,
                'count': 1,
                'dtype': 'float32',
                'crs': src_crs,
                'transform': src_transform,
                'compress': 'lzw',
                'nodata': np.nan,
            }
            with rasterio.open(outfile, 'w', **profile) as dst:
                dst.write(arr, 1)

        count += 1

    ds.close()
    return count


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Download historical FWI (2015-2022) from Copernicus CEMS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single year
    python -m src.data_ops.download.fwi_historical --year 2020

    # Full range, fire season only
    python -m src.data_ops.download.fwi_historical --start 2015 --end 2022 --months 5 6 7 8 9 10

    # Full range, all months
    python -m src.data_ops.download.fwi_historical --start 2015 --end 2022
        """,
    )
    add_config_argument(parser)

    parser.add_argument('--year', type=int, default=None,
                        help='Single year to download (shortcut for --start X --end X)')
    parser.add_argument('--start', type=int, default=2015,
                        help='Start year (default: 2015)')
    parser.add_argument('--end', type=int, default=2022,
                        help='End year (default: 2022)')
    parser.add_argument('--months', type=int, nargs='+',
                        default=list(range(1, 13)),
                        help='Months to download (default: 1-12). Example: --months 5 6 7 8 9 10')
    parser.add_argument('--no-reproject', action='store_true',
                        help='Skip reprojection, keep raw WGS84 GeoTIFFs')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference FWI GeoTIFF for reprojection grid')
    parser.add_argument('--nc-dir', type=str, default=None,
                        help='Directory to store raw NetCDF files (default: data/fwi_historical_nc)')
    parser.add_argument('--cds-url', type=str, default=None,
                        help='CDS API URL (if different from default)')
    parser.add_argument('--cds-key', type=str, default=None,
                        help='CDS API key (overrides env var / .cdsapirc)')

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve years
    if args.year:
        start_year = end_year = args.year
    else:
        start_year, end_year = args.start, args.end

    # Resolve directories
    fwi_dir = Path(get_path(cfg, 'fwi_dir'))
    fwi_dir.mkdir(parents=True, exist_ok=True)

    if args.nc_dir:
        nc_dir = Path(args.nc_dir)
    else:
        nc_dir = fwi_dir.parent / 'fwi_historical_nc'
    nc_dir.mkdir(parents=True, exist_ok=True)

    # Resolve reference file for reprojection
    ref_path = args.reference
    if ref_path is None and not args.no_reproject:
        # Try to find an existing FWI tif as reference
        existing = sorted(fwi_dir.glob('fwi_*.tif'))
        if existing:
            ref_path = str(existing[0])
            print(f"Using reference grid: {ref_path}")
        else:
            print("[WARN] No reference FWI GeoTIFF found in fwi_dir.")
            print("       Will save in WGS84. Use --reference to specify one.")
            print("       Or download a recent FWI first:")
            print("         python -m src.data_ops.download.download_fwi_grids 20240901")
            args.no_reproject = True

    # Initialize CDS client
    print("=" * 60)
    print("HISTORICAL FWI DOWNLOAD (Copernicus CEMS)")
    print("=" * 60)
    print(f"Years:   {start_year} - {end_year}")
    print(f"Months:  {args.months}")
    print(f"NC dir:  {nc_dir}")
    print(f"TIF dir: {fwi_dir}")
    print(f"Reproject: {'no' if args.no_reproject else 'yes'}")
    print("=" * 60)

    try:
        import cdsapi
    except ImportError:
        print("\n[ERROR] cdsapi not installed. Run: pip install cdsapi")
        sys.exit(1)

    client_kwargs = {}
    if args.cds_url:
        client_kwargs['url'] = args.cds_url
    if args.cds_key:
        client_kwargs['key'] = args.cds_key
    elif os.environ.get('CDS_API_KEY'):
        client_kwargs['key'] = os.environ['CDS_API_KEY']
    else:
        client_kwargs['key'] = DEFAULT_CDS_API_KEY

    try:
        client = cdsapi.Client(**client_kwargs)
    except Exception as e:
        print(f"\n[ERROR] Cannot create CDS client: {e}")
        print("Make sure you have:")
        print("  1. Installed cdsapi:  pip install cdsapi")
        print("  2. Created ~/.cdsapirc with your credentials, OR")
        print("     Set CDS_API_KEY environment variable")
        print("  3. Accepted the dataset licence at:")
        print("     https://cds.climate.copernicus.eu/datasets/cems-fire-historical-v1")
        sys.exit(1)

    # Step 1: Download NetCDFs
    print("\n--- STEP 1: Downloading from CDS ---\n")
    nc_files = download_fwi_range(client, start_year, end_year, args.months, nc_dir)

    if not nc_files:
        print("\n[ERROR] No files downloaded!")
        sys.exit(1)

    print(f"\nDownloaded {len(nc_files)} NetCDF files")

    # Step 2: Convert to daily GeoTIFFs
    print("\n--- STEP 2: Converting to daily GeoTIFFs ---\n")
    total_tifs = 0

    for nc_file in sorted(nc_files):
        print(f"Processing {nc_file.name} ...")
        n = nc_to_daily_tifs(
            nc_file, fwi_dir,
            reproject_to_fwi=not args.no_reproject,
            fwi_reference=ref_path,
        )
        total_tifs += n
        print(f"  -> {n} daily TIFs")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"NetCDF files:  {len(nc_files)}")
    print(f"Daily GeoTIFFs: {total_tifs}")
    print(f"Output:        {fwi_dir}")
    all_tifs = sorted(fwi_dir.glob('fwi_*.tif'))
    if all_tifs:
        print(f"Date range:    {all_tifs[0].stem.replace('fwi_','')} - {all_tifs[-1].stem.replace('fwi_','')}")
        print(f"Total FWI files in dir: {len(all_tifs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
