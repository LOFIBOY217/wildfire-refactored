"""
Data Pipeline
=============
Data collection, processing, and validation for the wildfire prediction project.

Submodules:
    download/    - Download raw data from ECMWF, CWFIS, CIFFC, etc.
    processing/  - Convert GRIB to GeoTIFF, reproject, compute daily averages, rasterize fires
    validation/  - Verify spatial alignment, data quality, and consistency checks
"""
