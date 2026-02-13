#!/usr/bin/env python3
"""
ECMWF GRIB to FWI Grid Reprojection Tool (cfgrib + xarray backend).

Reads ECMWF S2S GRIB forecast files using cfgrib/xarray, reprojects every
variable and time step to the FWI reference grid, and writes one GeoTIFF per
(variable, target-date) pair.

Output directory structure:
    <out_dir>/<issue_date>/<target_date>/<variable>.tif

Usage:
    # Single file
    python -m src.data_ops.processing.ecmwf_to_fwi_batch \\
        --grib_path ecmwf/forecast.grib --fwi_ref fwi/reference.tif

    # Batch over date range
    python -m src.data_ops.processing.ecmwf_to_fwi_batch \\
        --grib_dir ecmwf/ --fwi_ref fwi/reference.tif \\
        --start_date 2025-01-01 --end_date 2025-12-31

    # Validate outputs
    python -m src.data_ops.processing.ecmwf_to_fwi_batch \\
        --validate --fwi_ref fwi/reference.tif --check_file output/tcw.tif

    # With YAML config
    python -m src.data_ops.processing.ecmwf_to_fwi_batch \\
        --grib_path ecmwf/forecast.grib --config configs/paths_mac.yaml
"""

import os
import sys
import glob
import re
import argparse
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS

from src.config import load_config, get_path, add_config_argument
from src.utils.raster_io import write_geotiff

# cfgrib / xarray are required for this module
try:
    import xarray as xr
    import cfgrib
    HAS_CFGRIB = True
except ImportError:
    HAS_CFGRIB = False

# Variable name mapping: GRIB short name -> display name
VAR_NAME_MAP = {
    't2m': '2t',
    'd2m': '2d',
    'tcw': 'tcw',
    'sm20': 'sm20',
    'st20': 'st20',
}

# Constants
GRIB_FILL32 = np.float32(-3.4028235e38)
DST_NODATA = np.nan
DST_DTYPE = "float32"


# ============================================================================
# FWI Reference Grid
# ============================================================================

def read_fwi_reference(path: str) -> Tuple[rasterio.DatasetReader, dict]:
    """Open FWI GeoTIFF reference file, return dataset + output profile."""
    ref = rasterio.open(path)
    profile = ref.profile.copy()
    profile.update(
        driver="GTiff",
        dtype=DST_DTYPE,
        nodata=DST_NODATA,
        count=1,
        compress="deflate",
        tiled=True,
        predictor=3,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return ref, profile


# ============================================================================
# GRIB Loading (cfgrib + xarray)
# ============================================================================

def load_grib_datasets(grib_path: str) -> List[xr.Dataset]:
    """
    Load GRIB file using cfgrib; returns a list of xarray Datasets.

    Tries several ``filter_by_keys`` configurations to capture all level types.
    """
    if not HAS_CFGRIB:
        raise RuntimeError(
            "cfgrib/xarray not installed. "
            "Run: conda install -c conda-forge cfgrib xarray eccodes rasterio numpy"
        )

    datasets: List[xr.Dataset] = []
    loaded_vars: set = set()

    filter_configs = [
        {'typeOfLevel': 'surface'},
        {'typeOfLevel': 'heightAboveGround'},
        {'typeOfLevel': 'depthBelowLandLayer'},
        {'typeOfLevel': 'entireAtmosphere'},
        {'typeOfLevel': 'atmosphereSingleLayer'},
        {},  # fallback
    ]

    for filter_keys in filter_configs:
        try:
            ds = xr.open_dataset(
                grib_path,
                engine='cfgrib',
                backend_kwargs={
                    'filter_by_keys': filter_keys,
                    'errors': 'ignore',
                    'indexpath': '',
                },
            )
            new_vars = set(ds.data_vars) - loaded_vars
            if new_vars:
                datasets.append(ds)
                loaded_vars.update(new_vars)
                print(f"  Loaded variables ({filter_keys or 'default'}): {list(new_vars)}")
        except Exception:
            continue

    if not datasets:
        try:
            ds = xr.open_dataset(
                grib_path,
                engine='cfgrib',
                backend_kwargs={'errors': 'ignore', 'indexpath': ''},
            )
            if ds.data_vars:
                datasets.append(ds)
                print(f"  Loaded variables (fallback): {list(ds.data_vars)}")
        except Exception as e:
            raise RuntimeError(f"Cannot load GRIB file: {e}")

    return datasets


def get_time_values(da: xr.DataArray) -> Tuple[Optional[str], List[Any]]:
    """Return (time_dim_name, list_of_time_values) for the given DataArray."""
    for dim in ['time', 'valid_time', 'step', 'forecast_time']:
        if dim in da.dims:
            return dim, list(da[dim].values)
    return None, [None]


def get_level_info(da: xr.DataArray) -> Optional[str]:
    """Return a level-info string such as ``heightAboveGround2`` or None."""
    level_coords = [
        'level', 'heightAboveGround', 'depthBelowLandLayer',
        'surface', 'isobaricInhPa',
    ]
    for coord in level_coords:
        if coord in da.coords:
            val = da[coord].values
            if np.ndim(val) == 0:
                return f"{coord}{int(val)}"
            elif len(val) == 1:
                return f"{coord}{int(val[0])}"
    return None


def build_transform_from_coords(da: xr.DataArray) -> Tuple[Affine, CRS, int, int]:
    """Build an Affine transform from the latitude/longitude coordinates."""
    lats = lons = None
    for name in ['latitude', 'lat', 'y']:
        if name in da.coords:
            lats = da[name].values
            break
    for name in ['longitude', 'lon', 'x']:
        if name in da.coords:
            lons = da[name].values
            break

    if lats is None or lons is None:
        raise ValueError("Cannot find lat/lon coordinates")
    if lats.ndim > 1 or lons.ndim > 1:
        raise ValueError("Irregular grid (2-D lat/lon) not supported")

    dlat = abs(lats[1] - lats[0]) if len(lats) > 1 else 1.0
    dlon = abs(lons[1] - lons[0]) if len(lons) > 1 else 1.0

    lat_descending = lats[0] > lats[-1] if len(lats) > 1 else True

    if lat_descending:
        y0 = lats[0] + dlat / 2
    else:
        y0 = lats[-1] + dlat / 2

    x0 = lons[0] - dlon / 2
    if lons[0] > 180:
        x0 = (lons[0] - 360) - dlon / 2

    transform = Affine.translation(x0, y0) * Affine.scale(dlon, -dlat)
    crs = CRS.from_epsg(4326)
    return transform, crs, len(lats), len(lons)


def format_time_string(time_val: Any) -> str:
    """Convert various time types to a filename-safe string."""
    if time_val is None:
        return ""
    if isinstance(time_val, np.datetime64):
        ts = (time_val - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        try:
            dt = datetime.utcfromtimestamp(float(ts))
            return dt.strftime('%Y%m%dT%H%M')
        except Exception:
            return str(time_val).replace(':', '').replace('-', '')[:13]
    if isinstance(time_val, np.timedelta64):
        hours = int(time_val / np.timedelta64(1, 'h'))
        return f"step{hours:03d}h"
    if hasattr(time_val, 'strftime'):
        return time_val.strftime('%Y%m%dT%H%M')
    if isinstance(time_val, datetime):
        return time_val.strftime('%Y%m%dT%H%M')
    return str(time_val).replace(':', '').replace('-', '').replace(' ', 'T')[:13]


# ============================================================================
# Reprojection
# ============================================================================

def clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace GRIB fill values and non-finite values with NaN."""
    arr = arr.astype(np.float32, copy=True)
    arr = np.where(arr <= (GRIB_FILL32 / 10.0), np.nan, arr)
    arr = np.where(~np.isfinite(arr), np.nan, arr)
    return arr


def reproject_array(
    src_arr: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    ref_ds: rasterio.DatasetReader,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject *src_arr* to the FWI reference grid."""
    dst_arr = np.full((ref_ds.height, ref_ds.width), np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=ref_ds.transform,
        dst_crs=ref_ds.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst_arr


def _write_geotiff(arr, out_path, ref_ds, out_profile):
    """Write a single-band GeoTIFF aligned to *ref_ds*."""
    profile = out_profile.copy()
    profile.update(
        driver="GTiff",
        width=ref_ds.width,
        height=ref_ds.height,
        transform=ref_ds.transform,
        crs=ref_ds.crs,
        count=1,
        dtype="float32",
        nodata=np.nan,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


# ============================================================================
# Validation
# ============================================================================

def validate_reprojection(fwi_ref_path, check_file_path, output_report=True):
    """Validate that a reprojected file matches the FWI reference grid."""
    results = {
        "crs_match": False,
        "size_match": False,
        "transform_match": False,
        "bounds_match": False,
        "data_valid": False,
        "all_passed": False,
        "details": {},
    }

    with rasterio.open(fwi_ref_path) as fwi:
        with rasterio.open(check_file_path) as check:
            results["crs_match"] = fwi.crs == check.crs
            results["details"]["fwi_crs"] = str(fwi.crs)
            results["details"]["check_crs"] = str(check.crs)

            results["size_match"] = (fwi.width == check.width) and (fwi.height == check.height)
            results["details"]["fwi_size"] = (fwi.width, fwi.height)
            results["details"]["check_size"] = (check.width, check.height)

            results["transform_match"] = fwi.transform == check.transform
            results["details"]["fwi_transform"] = str(fwi.transform)
            results["details"]["check_transform"] = str(check.transform)

            bounds_diff = max(
                abs(fwi.bounds.left - check.bounds.left),
                abs(fwi.bounds.right - check.bounds.right),
                abs(fwi.bounds.top - check.bounds.top),
                abs(fwi.bounds.bottom - check.bounds.bottom),
            )
            results["bounds_match"] = bounds_diff < 1.0
            results["details"]["bounds_diff"] = bounds_diff

            check_data = check.read(1)
            valid_pixels = np.sum(np.isfinite(check_data) & (check_data > GRIB_FILL32 / 10))
            total_pixels = check_data.size
            valid_ratio = valid_pixels / total_pixels

            results["data_valid"] = valid_ratio > 0.01
            results["details"]["valid_pixels"] = int(valid_pixels)
            results["details"]["total_pixels"] = int(total_pixels)
            results["details"]["valid_ratio"] = float(valid_ratio)

            valid_data = check_data[np.isfinite(check_data) & (check_data > GRIB_FILL32 / 10)]
            if len(valid_data) > 0:
                results["details"]["data_min"] = float(np.min(valid_data))
                results["details"]["data_max"] = float(np.max(valid_data))
                results["details"]["data_mean"] = float(np.mean(valid_data))
                results["details"]["data_std"] = float(np.std(valid_data))

            results["all_passed"] = all([
                results["crs_match"],
                results["size_match"],
                results["transform_match"],
                results["bounds_match"],
                results["data_valid"],
            ])

    if output_report:
        _print_validation_report(results, fwi_ref_path, check_file_path)

    return results


def _print_validation_report(results, fwi_ref_path, check_file_path):
    """Print a human-readable validation report."""
    def status(passed):
        return "PASS" if passed else "FAIL"

    print("\n" + "=" * 70)
    print("Reprojection Validation Report")
    print("=" * 70)
    print(f"FWI reference: {fwi_ref_path}")
    print(f"Check file:    {check_file_path}")
    print("-" * 70)

    d = results["details"]
    print(f"  CRS match:       {status(results['crs_match'])}")
    print(f"  Size match:      {status(results['size_match'])}  "
          f"FWI={d['fwi_size']}  check={d['check_size']}")
    print(f"  Transform match: {status(results['transform_match'])}")
    print(f"  Bounds match:    {status(results['bounds_match'])}  "
          f"max_diff={d['bounds_diff']:.6f}")
    valid_pct = d['valid_ratio'] * 100
    print(f"  Data valid:      {status(results['data_valid'])}  "
          f"{d['valid_pixels']:,}/{d['total_pixels']:,} ({valid_pct:.2f}%)")

    if 'data_mean' in d:
        print(f"  Stats: min={d['data_min']:.4f}  max={d['data_max']:.4f}  "
              f"mean={d['data_mean']:.4f}  std={d['data_std']:.4f}")

    overall = "ALL PASSED" if results["all_passed"] else "ISSUES FOUND"
    print(f"\n  Overall: {overall}")
    print("=" * 70)


def batch_validate(fwi_ref_path, check_dir):
    """Validate all GeoTIFF files in *check_dir* against the FWI reference."""
    files = glob.glob(os.path.join(check_dir, "*.tif"))
    if not files:
        print(f"No .tif files found in {check_dir}")
        return

    print(f"\nBatch validation: {len(files)} files")
    print("=" * 70)

    passed = failed = 0
    for f in sorted(files):
        try:
            r = validate_reprojection(fwi_ref_path, f, output_report=False)
            tag = "PASS" if r["all_passed"] else "FAIL"
            pct = r['details']['valid_ratio'] * 100
            print(f"  {tag} {os.path.basename(f):<45} valid: {pct:5.1f}%")
            if r["all_passed"]:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL {os.path.basename(f):<45} error: {e}")
            failed += 1

    print("=" * 70)
    print(f"Total: {passed} passed, {failed} failed")


# ============================================================================
# Batch date-range processing
# ============================================================================

def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate inclusive list of ``YYYY-MM-DD`` strings."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def find_grib_file(grib_dir: str, date_str: str) -> Optional[str]:
    """Find the GRIB file for a given date in *grib_dir*."""
    patterns = [
        f"s2s_ecmf_cf_{date_str}.grib",
        f"s2s_ecmf_cf_{date_str.replace('-', '_')}.grib",
        f"s2s_ecmf_cf_{date_str.replace('-', '')}.grib",
        f"*{date_str}*.grib",
    ]
    for pattern in patterns:
        files = glob.glob(os.path.join(grib_dir, pattern))
        if files:
            return files[0]
    return None


def check_outputs_exist(out_dir, issue_date_str, variables=None):
    """Return True if outputs for *issue_date_str* already exist."""
    if variables is None:
        variables = ['t2m', 'd2m', 'sm20', 'st20', 'tcw']

    var_map = {'t2m': '2t', 'd2m': '2d', 'tcw': 'tcw', 'sm20': 'sm20', 'st20': 'st20'}
    issue_dir = os.path.join(out_dir, issue_date_str)

    if not os.path.exists(issue_dir):
        return False

    target_dirs = [
        d for d in os.listdir(issue_dir)
        if os.path.isdir(os.path.join(issue_dir, d))
    ]
    if not target_dirs:
        return False

    check_dirs = [target_dirs[0]]
    if len(target_dirs) > 1:
        check_dirs.append(target_dirs[-1])

    for target_date in check_dirs:
        target_dir = os.path.join(issue_dir, target_date)
        for var in variables:
            display_name = var_map.get(var, var)
            file_path = os.path.join(target_dir, f"{display_name}.tif")
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return False
    return True


def batch_process_gribs(
    grib_dir: str,
    fwi_ref_path: str,
    out_dir: str,
    start_date: str,
    end_date: str,
    variables: Optional[List[str]] = None,
    resampling: Resampling = Resampling.bilinear,
    validate: bool = True,
    force: bool = False,
):
    """Batch-process a date range of GRIB files."""
    dates = generate_date_range(start_date, end_date)

    print("=" * 60)
    print("Batch processing mode")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date} ({len(dates)} days)")
    print(f"GRIB dir:   {grib_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Force:      {force}")
    print("=" * 60)

    success_count = skip_count = already_count = error_count = 0
    failed_dates: List[str] = []

    for i, date_str in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] {date_str}")

        grib_file = find_grib_file(grib_dir, date_str)
        if grib_file is None or not os.path.exists(grib_file):
            print(f"  [SKIP] no GRIB file found")
            skip_count += 1
            continue

        basename = os.path.basename(grib_file)
        date_match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', basename)
        issue_date_str = (
            date_match.group(1).replace('-', '').replace('_', '')
            if date_match
            else date_str.replace('-', '')
        )

        if not force and check_outputs_exist(out_dir, issue_date_str, variables):
            print(f"  [ALREADY DONE] outputs exist, skipping (use --force to redo)")
            already_count += 1
            continue

        try:
            process_grib_file(
                grib_path=grib_file,
                fwi_ref_path=fwi_ref_path,
                out_dir=out_dir,
                variables=variables,
                resampling=resampling,
                validate=False,
            )
            success_count += 1
        except Exception as e:
            error_count += 1
            failed_dates.append(date_str)
            print(f"  [ERROR] {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Batch processing summary")
    print("=" * 60)
    print(f"Total dates:      {len(dates)}")
    print(f"Processed:        {success_count}")
    print(f"Already done:     {already_count}")
    print(f"No GRIB file:     {skip_count}")
    print(f"Failed:           {error_count}")
    if failed_dates:
        print(f"\nFailed dates:")
        for d in failed_dates:
            print(f"  - {d}")
        fail_file = os.path.join(out_dir, "failed_dates.txt")
        with open(fail_file, "w") as f:
            f.write("\n".join(failed_dates))
        print(f"Saved to: {fail_file}")
    print("=" * 60)


# ============================================================================
# Main processing flow
# ============================================================================

def process_grib_file(
    grib_path: str,
    fwi_ref_path: str,
    out_dir: str,
    variables: Optional[List[str]] = None,
    resampling: Resampling = Resampling.bilinear,
    validate: bool = True,
) -> List[str]:
    """
    Process a single GRIB file: load via cfgrib, reproject, write GeoTIFFs.

    Output layout::

        out_dir/<issue_date>/<target_date>/<variable>.tif
    """
    # Extract issue date from filename
    basename = os.path.basename(grib_path)
    date_match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', basename)
    if date_match:
        issue_date_str = date_match.group(1).replace('-', '').replace('_', '')
    else:
        issue_date_str = datetime.now().strftime('%Y%m%d')
        print(f"  [WARNING] Cannot extract date from filename; using today: {issue_date_str}")

    # Read FWI reference
    print("=" * 60)
    print("Reading FWI reference grid...")
    ref, out_profile = read_fwi_reference(fwi_ref_path)
    print(f"  CRS:  {ref.crs}")
    print(f"  Size: {ref.width} x {ref.height}")
    print(f"  Res:  {ref.res}")

    # Load GRIB
    print("\nLoading GRIB via cfgrib...")
    print(f"  Issue date: {issue_date_str}")
    datasets = load_grib_datasets(grib_path)
    print(f"  Loaded {len(datasets)} dataset(s)")

    # Base time
    base_time = None
    for ds in datasets:
        if 'time' in ds.coords:
            base_time = ds['time'].values
            if isinstance(base_time, np.ndarray):
                base_time = base_time.flat[0]
            break
    if base_time is None:
        base_time = np.datetime64(
            f"{issue_date_str[:4]}-{issue_date_str[4:6]}-{issue_date_str[6:8]}"
        )
    print(f"  Base time: {base_time}")

    # Collect variables
    all_vars = []
    for ds_idx, ds in enumerate(datasets):
        for var_name in ds.data_vars:
            if variables and var_name not in variables:
                continue
            all_vars.append((ds_idx, var_name))

    print(f"  Variables to process: {[v[1] for v in all_vars]}")

    # Count total files
    total_files = 0
    for ds_idx, var_name in all_vars:
        da = datasets[ds_idx][var_name]
        _, time_values = get_time_values(da)
        total_files += len(time_values)

    print(f"\nReprojecting {total_files} layer(s)...")

    success_count = error_count = current = 0
    output_files: List[str] = []

    for ds_idx, var_name in all_vars:
        ds = datasets[ds_idx]
        da = ds[var_name]

        try:
            src_transform, src_crs, src_h, src_w = build_transform_from_coords(da)
        except Exception as e:
            print(f"  Skip {var_name}: cannot build transform - {e}")
            error_count += 1
            continue

        level_info = get_level_info(da)
        time_dim, time_values = get_time_values(da)

        lat_name = 'latitude' if 'latitude' in da.coords else 'lat'
        lats = da[lat_name].values
        need_flip = len(lats) > 1 and lats[0] < lats[-1]

        for t_idx, t_val in enumerate(time_values):
            current += 1
            try:
                if time_dim is not None:
                    src_arr = da.isel({time_dim: t_idx}).values
                else:
                    src_arr = da.values

                while src_arr.ndim > 2:
                    src_arr = src_arr.squeeze()
                if src_arr.ndim != 2:
                    raise ValueError(f"Expected 2-D array, got {src_arr.ndim}-D")

                if need_flip:
                    src_arr = np.flip(src_arr, axis=0)

                src_arr = clean_array(src_arr)
                dst_arr = reproject_array(src_arr, src_transform, src_crs, ref, resampling)

                # Compute target date
                if isinstance(t_val, np.timedelta64):
                    valid_time = base_time + t_val
                    ts = (valid_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                    target_date_str = datetime.utcfromtimestamp(float(ts)).strftime('%Y%m%d')
                elif isinstance(t_val, np.datetime64):
                    ts = (t_val - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                    target_date_str = datetime.utcfromtimestamp(float(ts)).strftime('%Y%m%d')
                else:
                    target_date_str = format_time_string(t_val)
                    if len(target_date_str) > 8:
                        target_date_str = target_date_str[:8]

                # Write output
                target_dir = os.path.join(out_dir, issue_date_str, target_date_str)
                os.makedirs(target_dir, exist_ok=True)

                display_name = VAR_NAME_MAP.get(var_name, var_name)
                out_path = os.path.join(target_dir, f"{display_name}.tif")

                _write_geotiff(dst_arr, out_path, ref, out_profile)
                print(f"  [{current}/{total_files}] {var_name} -> "
                      f"{issue_date_str}/{target_date_str}/{display_name}.tif")
                success_count += 1
                output_files.append(out_path)

            except Exception as e:
                print(f"  [{current}/{total_files}] {var_name} ERROR: {e}")
                error_count += 1

    # Cleanup
    for ds in datasets:
        ds.close()
    ref.close()

    print(f"\nDone: {success_count} succeeded, {error_count} failed")

    if validate and output_files:
        print("\nValidating first output...")
        validate_reprojection(fwi_ref_path, output_files[0])

    return output_files


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ECMWF GRIB to FWI Grid (cfgrib + xarray backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)

    # Mode
    parser.add_argument("--validate", action="store_true", help="Validation mode")

    # Common
    parser.add_argument("--fwi_ref", type=str, default=None,
                        help="FWI reference GeoTIFF (overrides config)")

    # Reprojection
    parser.add_argument("--grib_path", type=str, help="Single GRIB file path")
    parser.add_argument("--grib_dir", type=str, help="GRIB directory (batch mode)")
    parser.add_argument("--start_date", type=str, help="Start date YYYY-MM-DD (batch)")
    parser.add_argument("--end_date", type=str, help="End date YYYY-MM-DD (batch)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resampling", type=str, default="bilinear",
                        choices=["bilinear", "nearest", "cubic"],
                        help="Resampling method (default: bilinear)")
    parser.add_argument("--variables", type=str, nargs="*",
                        help="Only process these variables (e.g. tcw t2m d2m)")
    parser.add_argument("--skip_validate", action="store_true",
                        help="Skip automatic validation after processing")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing of existing outputs")

    # Validation
    parser.add_argument("--check_file", type=str, help="File to validate")
    parser.add_argument("--check_dir", type=str, help="Directory to batch-validate")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve fwi_ref
    fwi_ref = args.fwi_ref
    if fwi_ref is None:
        fwi_dir = get_path(cfg, 'fwi_dir')
        candidates = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
        if candidates:
            fwi_ref = candidates[0]
            print(f"Using FWI reference from config: {fwi_ref}")
        else:
            print(f"Error: No .tif files in fwi_dir ({fwi_dir}). Use --fwi_ref.")
            sys.exit(1)

    # Resolve output dir
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(get_path(cfg, 'output_dir'), "ecmwf_reprojected")

    resampling_methods = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
    }

    if args.validate:
        if args.check_file:
            validate_reprojection(fwi_ref, args.check_file)
        elif args.check_dir:
            batch_validate(fwi_ref, args.check_dir)
        else:
            print("Error: validation mode requires --check_file or --check_dir")
    elif args.grib_path:
        if not os.path.exists(args.grib_path):
            print(f"Error: GRIB file not found: {args.grib_path}")
            sys.exit(1)
        process_grib_file(
            grib_path=args.grib_path,
            fwi_ref_path=fwi_ref,
            out_dir=out_dir,
            variables=args.variables,
            resampling=resampling_methods[args.resampling],
            validate=not args.skip_validate,
        )
    elif args.grib_dir and args.start_date and args.end_date:
        if not os.path.exists(args.grib_dir):
            print(f"Error: GRIB directory not found: {args.grib_dir}")
            sys.exit(1)
        batch_process_gribs(
            grib_dir=args.grib_dir,
            fwi_ref_path=fwi_ref,
            out_dir=out_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            variables=args.variables,
            resampling=resampling_methods[args.resampling],
            validate=not args.skip_validate,
            force=args.force,
        )
    else:
        print("Error: specify one of:")
        print("  Single file: --grib_path <file>")
        print("  Batch:       --grib_dir <dir> --start_date YYYY-MM-DD --end_date YYYY-MM-DD")


if __name__ == "__main__":
    main()
