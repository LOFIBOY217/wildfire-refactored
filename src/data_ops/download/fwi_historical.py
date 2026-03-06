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

# CDS request variable name → (candidate NetCDF variable names, short output prefix)
# The NetCDF internal name sometimes differs from the CDS request name, so we try
# a list of candidates in order.
FWI_VARIABLES = {
    'fire_weather_index':      (['fwi', 'fire_weather_index', 'mark_4'], 'fwi'),
    'fine_fuel_moisture_code': (['ffmc', 'fine_fuel_moisture_code'],      'ffmc'),
    'duff_moisture_code':      (['dmc',  'duff_moisture_code'],           'dmc'),
    'drought_code':            (['dc',   'drought_code'],                 'dc'),
    'initial_spread_index':    (['isi',  'initial_spread_index'],         'isi'),
    'build_up_index':          (['bui',  'build_up_index'],               'bui'),
}


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

    outfile = output_dir / f"fwi_components_{year_str}{month_str}.nc"

    if outfile.exists():
        print(f"  [SKIP] {outfile.name} already exists")
        return outfile

    request = {
        'product_type': 'reanalysis',
        'variable': list(FWI_VARIABLES.keys()),   # all 6 components in one request
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
    Download FWI for a range of years and months (sequential).

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


def _worker_download(task):
    """Thread-worker: creates its own CDS client and downloads one year-month."""
    year, month, output_dir, area, client_kwargs = task
    try:
        import cdsapi
        client = cdsapi.Client(**client_kwargs, quiet=True)
        return download_fwi_year_month(client, year, month, output_dir, area)
    except Exception as e:
        print(f"  [WORKER ERROR] {year}-{month:02d}: {e}")
        return None


def download_fwi_range_parallel(client_kwargs, start_year, end_year, months,
                                 output_dir, area=None, workers=4):
    """
    Download FWI for a range of years and months using a thread pool.

    Each thread creates its own CDS client so they don't share state.
    CDS API accepts concurrent requests from the same key.

    Args:
        client_kwargs: dict passed to cdsapi.Client (url, key, etc.)
        workers:       number of parallel download threads (default 4)

    Returns:
        list of successfully downloaded file paths
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = [
        (year, month, output_dir, area, client_kwargs)
        for year in range(start_year, end_year + 1)
        for month in months
    ]
    total = len(tasks)
    print(f"  Parallel download: {total} year-months, {workers} workers")

    files = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_download, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            year, month = futures[future][0], futures[future][1]
            result = future.result()
            status = "OK" if result else "FAIL"
            print(f"  [{done}/{total}] {year}-{month:02d} [{status}]")
            if result:
                files.append(result)

    return sorted(files)


# ------------------------------------------------------------------ #
# NetCDF -> per-day GeoTIFF conversion + reprojection
# ------------------------------------------------------------------ #

def nc_to_daily_tifs(nc_path, var_dirs, reproject_to_fwi=True, fwi_reference=None):
    """
    Convert a monthly FWI-components NetCDF to per-day GeoTIFFs for each variable.

    Each variable is written to its own directory:
        var_dirs['fwi']  / fwi_YYYYMMDD.tif
        var_dirs['ffmc'] / ffmc_YYYYMMDD.tif
        var_dirs['dmc']  / dmc_YYYYMMDD.tif
        ...

    If reproject_to_fwi=True and fwi_reference is provided, reprojects
    each day from WGS84 0.25deg to the FWI grid (EPSG:3978, 2709x2281).

    Args:
        nc_path:         Path to NetCDF file
        var_dirs:        dict mapping short name → output Path, e.g.
                         {'fwi': Path(...), 'dmc': Path(...), ...}
        reproject_to_fwi: Whether to reproject to FWI grid
        fwi_reference:   Path to a reference FWI GeoTIFF (for target grid)

    Returns:
        int: Total number of TIF files written across all variables
    """
    import netCDF4
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    ds = netCDF4.Dataset(str(nc_path))

    # Get time, lat, lon
    time_var = ds.variables['time']
    times = netCDF4.num2date(
        time_var[:], time_var.units,
        time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
    )

    lat_name = 'latitude' if 'latitude' in ds.variables else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.variables else 'lon'
    lats = ds.variables[lat_name][:]
    lons = ds.variables[lon_name][:]

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    nlat, nlon = len(lats), len(lons)
    lat_descending = lats[0] > lats[-1]

    src_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nlon, nlat)
    src_crs = 'EPSG:4326'

    # Load target grid for reprojection
    dst_profile = dst_transform = dst_crs = dst_height = dst_width = None
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

    # Build date strings once
    date_strs = []
    for dt in times:
        if hasattr(dt, 'strftime'):
            date_strs.append(dt.strftime("%Y%m%d"))
        else:
            date_strs.append(str(dt)[:10].replace('-', ''))

    total_count = 0

    # Loop over each FWI component
    for cds_name, (nc_candidates, short_name) in FWI_VARIABLES.items():
        out_dir = var_dirs.get(short_name)
        if out_dir is None:
            continue

        # Find the variable name actually used inside this NetCDF
        nc_var_name = None
        for candidate in nc_candidates:
            if candidate in ds.variables:
                nc_var_name = candidate
                break
        if nc_var_name is None:
            print(f"  [WARN] {cds_name} not found in {nc_path.name} "
                  f"(tried: {nc_candidates}), skipping")
            continue

        var_data = ds.variables[nc_var_name]  # [time, lat, lon]
        fill_val = getattr(var_data, '_FillValue', None)
        count = 0

        for t_idx, date_str in enumerate(date_strs):
            outfile = out_dir / f"{short_name}_{date_str}.tif"
            if outfile.exists():
                count += 1
                continue

            arr = np.array(var_data[t_idx, :, :], dtype=np.float32)
            if not lat_descending:
                arr = np.flipud(arr)
            if fill_val is not None:
                arr[arr == fill_val] = np.nan
            arr[~np.isfinite(arr)] = np.nan

            if reproject_to_fwi:
                dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
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
                profile = {
                    'driver': 'GTiff', 'height': nlat, 'width': nlon,
                    'count': 1, 'dtype': 'float32', 'crs': src_crs,
                    'transform': src_transform, 'compress': 'lzw', 'nodata': np.nan,
                }
                with rasterio.open(outfile, 'w', **profile) as dst:
                    dst.write(arr, 1)
            count += 1

        print(f"    {short_name}: {count} TIFs → {out_dir}")
        total_count += count

    ds.close()
    return total_count


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
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel download threads (default=1 sequential). '
                             'Recommended: 3-4. Each thread opens its own CDS client.')

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve years
    if args.year:
        start_year = end_year = args.year
    else:
        start_year, end_year = args.start, args.end

    # Resolve output directories for each FWI component
    var_dirs = {}
    for short_name, config_key in [
        ('fwi',  'fwi_dir'),
        ('ffmc', 'ffmc_dir'),
        ('dmc',  'dmc_dir'),
        ('dc',   'dc_dir'),
        ('isi',  'isi_dir'),
        ('bui',  'bui_dir'),
    ]:
        p = Path(get_path(cfg, config_key))
        p.mkdir(parents=True, exist_ok=True)
        var_dirs[short_name] = p

    fwi_dir = var_dirs['fwi']  # used for reference file lookup and summary

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

    # Build CDS client kwargs.
    # cdsapi >= 1.0 (post-2024 ECMWF migration) requires BOTH url AND key;
    # without url it falls back to ~/.cdsapirc which may not exist on the server.
    DEFAULT_CDS_URL = "https://cds.climate.copernicus.eu/api"
    client_kwargs = {}
    client_kwargs['url'] = args.cds_url if args.cds_url else DEFAULT_CDS_URL
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
        print("Possible fixes:")
        print("  1. Pass key explicitly:  --cds-key YOUR_KEY")
        print("  2. Set env var:          export CDS_API_KEY=YOUR_KEY")
        print("  3. Create ~/.cdsapirc with:")
        print("       url: https://cds.climate.copernicus.eu/api")
        print("       key: YOUR_KEY")
        print("  4. Accept the dataset licence at:")
        print("     https://cds.climate.copernicus.eu/datasets/cems-fire-historical-v1")
        sys.exit(1)

    # Step 1: Download NetCDFs
    print("\n--- STEP 1: Downloading from CDS ---\n")
    if args.workers > 1:
        nc_files = download_fwi_range_parallel(
            client_kwargs, start_year, end_year, args.months, nc_dir,
            workers=args.workers,
        )
    else:
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
            nc_file, var_dirs,
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
