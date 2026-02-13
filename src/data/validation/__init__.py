"""
Data Validation
===============
Quality checks and consistency verification for all datasets.

Modules:
    verify_alignment      - Check spatial/temporal alignment across FWI, ECMWF, ERA5
    check_fwi_consistency - Verify all FWI GeoTIFFs share the same CRS, resolution, bounds
    check_data_quality    - Detect NaN, Inf, extreme values, all-zero frames
    check_fwi_health      - FWI-specific health checks (value ranges, NoData patterns)
"""
