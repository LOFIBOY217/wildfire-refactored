"""
Raster I/O Utilities
====================
GeoTIFF read/write, NoData cleaning, and raster metadata extraction.

Consolidates duplicated functions from:
- ecmwf_to_fwi.py / ecmwf_to_fwi_cfgrib_batch.py (read_fwi_reference, clean_array, write_geotiff)
- simple_logistic_7day.py (read_singleband_stack)
- train_s2s_transformer.py (load_fwi_file, clean_nodata)
- verify_data_alignment.py (get_raster_info)
"""

import numpy as np
import rasterio
from PIL import Image

# Default NoData threshold (float32 minimum is ~-3.4e38)
NODATA_THRESHOLD = -1e30


def clean_nodata(arr, nodata_threshold=NODATA_THRESHOLD, fill_value=None):
    """
    Clean NoData values: replace extreme negatives, NaN, and Inf with NaN.

    Args:
        arr: numpy array
        nodata_threshold: Values below this are treated as NoData
        fill_value: If provided, fill NaN with this value after cleaning

    Returns:
        Cleaned copy of the array
    """
    arr = arr.copy()
    arr[arr < nodata_threshold] = np.nan
    arr[np.isinf(arr)] = np.nan

    if fill_value is not None:
        arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)

    return arr


def clean_array(arr):
    """
    Clean array for GRIB conversion: replace NaN/Inf with 0.

    Args:
        arr: numpy array

    Returns:
        Cleaned array with NaN/Inf replaced by 0
    """
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    return arr


def read_fwi_reference(path):
    """
    Read a reference FWI GeoTIFF and return its metadata.

    Args:
        path: Path to reference GeoTIFF

    Returns:
        tuple: (profile, transform, crs, shape) from the reference file
    """
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    return profile, transform, crs, shape


def read_singleband_stack(tif_paths):
    """
    Read list of single-band GeoTIFFs into a 3D array.

    Corrupt or unreadable files are replaced by the most recent previously
    loaded array (i.e. the previous day's data).  If the very first file is
    corrupt an IOError is raised because there is no earlier array to fall
    back to.  A warning is printed for every substitution so the caller can
    see which dates were affected.

    Args:
        tif_paths: List of file paths to single-band GeoTIFFs

    Returns:
        numpy array of shape [T, H, W]
    """
    arrays = []
    last_good = None  # most recent successfully loaded array

    for path in tif_paths:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1)
            last_good = arr
        except Exception as exc:
            if last_good is None:
                raise IOError(
                    f"First file in stack is corrupt and there is no previous "
                    f"array to fall back to.\n  File: {path}\n  Error: {exc}"
                ) from exc
            import warnings
            warnings.warn(
                f"[read_singleband_stack] Corrupt file, substituting previous day:\n"
                f"  {path}\n  {exc}",
                stacklevel=2,
            )
            arr = last_good  # reuse previous day's data

        arrays.append(arr)

    return np.stack(arrays, axis=0)


def load_fwi_file(filepath):
    """
    Load a single FWI GeoTIFF file with NoData cleaning.

    Uses PIL as primary loader (lightweight), with rasterio fallback.

    Args:
        filepath: Path to .tif file

    Returns:
        numpy array of shape [H, W] with NoData replaced by NaN
    """
    im = Image.open(filepath)
    arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = clean_nodata(arr)
    return arr


def load_rasterio_file(filepath):
    """
    Load a single GeoTIFF using rasterio with NoData cleaning.

    Args:
        filepath: Path to .tif file

    Returns:
        numpy array of shape [H, W] with NoData replaced by NaN
    """
    try:
        with rasterio.open(filepath) as src:
            arr = src.read(1).astype(np.float32)
    except Exception:
        im = Image.open(filepath)
        arr = np.array(im, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
    arr = clean_nodata(arr)
    return arr


def write_geotiff(arr, path, profile, nodata=None):
    """
    Write a 2D array as a single-band GeoTIFF.

    Args:
        arr: 2D numpy array [H, W]
        path: Output file path
        profile: Rasterio profile dict (from reference file)
        nodata: NoData value to set in output
    """
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    if nodata is not None:
        out_profile['nodata'] = nodata

    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def get_raster_info(filepath):
    """
    Get metadata from a raster file.

    Args:
        filepath: Path to raster file

    Returns:
        dict with keys: crs, transform, width, height, bounds, dtype, nodata
    """
    with rasterio.open(filepath) as src:
        return {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'dtype': src.dtypes[0],
            'nodata': src.nodata
        }
