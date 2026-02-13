"""
Shared Utilities
================
Common functions used across multiple modules, deduplicated from original scripts.

Modules:
    seed          - Random seed setting for reproducibility (PyTorch + NumPy + Python)
    date_utils    - Date parsing from filenames, date range generation, date argument parsing
    raster_io     - GeoTIFF read/write, NoData cleaning, raster metadata extraction
    patch_utils   - Patchify/depatchify for converting spatial grids to patch sequences
    normalization - Data standardization with safety checks
"""
