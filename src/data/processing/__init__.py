"""
Data Processing
===============
Transform raw data into model-ready format.

Modules:
    ecmwf_to_fwi       - Convert single GRIB file to GeoTIFF (rasterio-only, no pygrib)
    ecmwf_to_fwi_batch  - Batch GRIB to GeoTIFF conversion using cfgrib + xarray
    era5_to_daily       - Aggregate ERA5 hourly data to daily averages
    resample_to_fwi_grid - Reproject rasters to FWI grid (EPSG:3978, 2709x2281)
    rasterize_fires     - Convert CIFFC point fire records to raster grids

GRIB Backend:
    All GRIB processing uses cfgrib + xarray (Windows compatible).
    The rasterio-only backend (ecmwf_to_fwi.py) serves as a fallback.
    pygrib is NOT used anywhere in this project.
"""

try:
    import cfgrib
    HAS_CFGRIB = True
except ImportError:
    HAS_CFGRIB = False
