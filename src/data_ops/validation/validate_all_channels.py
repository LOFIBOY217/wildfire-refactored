"""
Comprehensive channel data validation.

Checks for ALL training channels:
1. File count and date range coverage
2. Corrupt files (size way below median)
3. CRS + dimensions (must be EPSG:3978, 2709x2281)
4. Pixel value ranges (physical plausibility)
5. NaN/nodata fraction
6. Suspicious "constant value dominance" (>99% same value = likely sentinel leak)

Usage:
    python -m src.data_ops.validation.validate_all_channels --config configs/paths_narval.yaml

Exits with non-zero status if any CRITICAL issue found.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# Expected CRS and dimensions
EXPECTED_CRS = "EPSG:3978"
EXPECTED_H, EXPECTED_W = 2281, 2709

# Physical plausibility ranges (after ERA5 unit conversion)
# Used to flag values WAY outside expected (sentinels, corrupt data)
PLAUSIBLE_RANGES = {
    "FWI":       (-10, 200),     # 0-150 typical
    "FFMC":      (-10, 120),     # 0-100 typical
    "DMC":       (-10, 600),     # 0-300 typical
    "DC":        (-10, 2000),    # 0-1500 typical
    "ISI":       (-10, 100),     # 0-50 typical
    "BUI":       (-10, 600),     # 0-400 typical
    "2t":        (-80, 50),      # Celsius, Canada
    "2d":        (-80, 40),      # Celsius, dewpoint
    "tcw":       (0, 80),        # kg/m², total column water
    "sm20":      (-0.1, 1.1),    # volumetric, 0-1
    "st20":      (-60, 50),      # Celsius
    "u10":       (-50, 50),      # m/s
    "v10":       (-50, 50),      # m/s
    "cape":      (-10, 10000),   # J/kg
    "tp":        (-0.01, 0.5),   # m/day (precip)
    "swvl2":     (-0.1, 1.1),    # volumetric, 0-1
    "deep_soil": (-0.1, 1.1),    # same
    "precip":    (-0.01, 0.5),   # same
    "NDVI":      (-1.1, 1.1),    # -1 to 1
    "fire_clim": (-0.1, 20),     # log1p transform
    "burn_age":  (-0.1, 20),     # log1p of years (max ~3 if masked properly)
    "burn_count": (-0.1, 10),    # log1p of counts
    "population": (-1, 20),      # log-scaled
    "slope":     (-1, 90),       # degrees
}


def check_directory(name, directory, pattern, expected_min_files=0, year_range=None):
    """Check a directory of TIF files. Returns (pass, issues)."""
    issues = []

    if not os.path.isdir(directory):
        return False, [f"Directory does not exist: {directory}"]

    files = sorted(glob.glob(os.path.join(directory, pattern)))
    n_files = len(files)

    if n_files == 0:
        return False, [f"No files matching {pattern}"]

    if n_files < expected_min_files:
        issues.append(f"file count {n_files} < expected min {expected_min_files}")

    # Check date coverage if year_range specified
    if year_range:
        start_year, end_year = year_range
        years_found = set()
        for f in files:
            bn = os.path.basename(f)
            # Extract YYYY from various patterns
            import re
            m = re.search(r'(\d{4})', bn)
            if m:
                years_found.add(int(m.group(1)))
        missing_years = [y for y in range(start_year, end_year + 1) if y not in years_found]
        if missing_years:
            issues.append(f"missing years: {missing_years[:5]}{'...' if len(missing_years) > 5 else ''}")

    # Check for corrupt files (size way below median)
    sizes = [os.path.getsize(f) for f in files]
    median_size = np.median(sizes)
    threshold = median_size * 0.3  # flag files <30% of median size
    corrupt = [(f, s) for f, s in zip(files, sizes) if s < threshold]
    if corrupt:
        issues.append(f"{len(corrupt)} possibly corrupt files (size < 30% median {median_size/1e6:.1f}MB): "
                      f"{[os.path.basename(c[0]) for c in corrupt[:3]]}")

    # Sample a few files for deep inspection
    sample_files = [files[0], files[len(files)//2], files[-1]]  # first, middle, last
    for sf in sample_files:
        try:
            with rasterio.open(sf) as src:
                if str(src.crs) != EXPECTED_CRS:
                    issues.append(f"wrong CRS in {os.path.basename(sf)}: {src.crs}")
                if (src.height, src.width) != (EXPECTED_H, EXPECTED_W):
                    issues.append(f"wrong shape in {os.path.basename(sf)}: ({src.height}, {src.width})")

                arr = src.read(1).astype(np.float32)
                nodata = src.nodata

                # Mask nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan

                # NaN fraction
                nan_frac = np.isnan(arr).mean()
                if nan_frac > 0.5:
                    issues.append(f"{os.path.basename(sf)}: {nan_frac:.0%} NaN (likely corrupt)")

                # Value range check
                if name in PLAUSIBLE_RANGES:
                    lo, hi = PLAUSIBLE_RANGES[name]
                    valid = arr[~np.isnan(arr)]
                    if valid.size > 0:
                        vmin, vmax = float(valid.min()), float(valid.max())
                        if vmin < lo or vmax > hi:
                            issues.append(f"{os.path.basename(sf)}: values [{vmin:.2f}, {vmax:.2f}] "
                                          f"outside plausible [{lo}, {hi}]")

                        # Constant value dominance check
                        vals, counts = np.unique(valid, return_counts=True)
                        if vals.size > 0:
                            top_frac = counts.max() / valid.size
                            top_val = vals[counts.argmax()]
                            if top_frac > 0.99 and top_val != 0:
                                issues.append(f"{os.path.basename(sf)}: {top_frac:.0%} of pixels = {top_val:.2f} "
                                              f"(likely sentinel leak)")

        except Exception as e:
            issues.append(f"Error reading {os.path.basename(sf)}: {type(e).__name__}: {e}")

    # Report success
    if not issues:
        return True, [f"OK: {n_files} files, size range {min(sizes)/1e6:.1f}-{max(sizes)/1e6:.1f} MB"]
    return False, issues


def main():
    parser = argparse.ArgumentParser(description="Validate all training channel data")
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2000,
                        help="Expected start year for time series data")
    parser.add_argument("--end_year", type=int, default=2025,
                        help="Expected end year")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with non-zero status on any issue")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", cfg)

    # Channel definitions: (name, directory_key_or_path, file_pattern, min_files, year_range)
    channels = [
        # FWI components — daily, EPSG:3978
        ("FWI",   paths["fwi_dir"],   "fwi_*.tif",  3000, (args.start_year, args.end_year)),
        ("FFMC",  paths["ffmc_dir"],  "ffmc_*.tif", 3000, (args.start_year, args.end_year)),
        ("DMC",   paths["dmc_dir"],   "dmc_*.tif",  3000, (args.start_year, args.end_year)),
        ("DC",    paths["dc_dir"],    "dc_*.tif",   3000, (args.start_year, args.end_year)),
        ("BUI",   paths["bui_dir"],   "bui_*.tif",  3000, (args.start_year, args.end_year)),

        # ERA5 observation — daily, EPSG:3978
        ("2t",    os.path.join(paths.get("observation_dir", paths.get("ecmwf_dir", "")), "2t"),
         "2t_*.tif", 6000, (args.start_year, args.end_year)),
        ("2d",    os.path.join(paths.get("observation_dir", paths.get("ecmwf_dir", "")), "2d"),
         "2d_*.tif", 6000, (args.start_year, args.end_year)),
        ("tcw",   os.path.join(paths.get("observation_dir", paths.get("ecmwf_dir", "")), "tcw"),
         "tcw_*.tif", 6000, (args.start_year, args.end_year)),
        ("sm20",  os.path.join(paths.get("observation_dir", paths.get("ecmwf_dir", "")), "sm20"),
         "sm20_*.tif", 6000, (args.start_year, args.end_year)),
        ("st20",  os.path.join(paths.get("observation_dir", paths.get("ecmwf_dir", "")), "st20"),
         "st20_*.tif", 6000, (args.start_year, args.end_year)),

        # Wind / CAPE — daily, EPSG:3978
        ("u10",   paths.get("wind_u_dir", "data/era5_u10"), "u10_*.tif", 3000, (args.start_year, args.end_year)),
        ("v10",   paths.get("wind_v_dir", "data/era5_v10"), "v10_*.tif", 3000, (args.start_year, args.end_year)),
        ("cape",  paths.get("cape_dir", "data/era5_cape"),  "cape_*.tif", 3000, (args.start_year, args.end_year)),

        # Optional (may not cover 2000-2017)
        ("deep_soil", paths.get("deep_soil_dir", "data/era5_deep_soil"), "swvl2_*.tif", 0, None),
        ("precip",    paths.get("precip_dir", "data/era5_precip"),       "tp_*.tif",    0, None),
        ("NDVI",      paths.get("ndvi_dir", "data/ndvi_data"),           "ndvi_*.tif",  0, None),

        # Annual
        ("fire_clim",  paths.get("fire_clim_dir", "data/fire_clim_annual"),
         "fire_clim_upto_*.tif", 0, None),
        ("burn_age",   paths.get("burn_scars_dir", "data/burn_scars"),
         "years_since_burn_*.tif", 0, None),
        ("burn_count", paths.get("burn_scars_dir", "data/burn_scars"),
         "burn_count_*.tif", 0, None),
    ]

    print("=" * 70)
    print(f"CHANNEL VALIDATION (expected range: {args.start_year}-{args.end_year})")
    print("=" * 70)

    results = []
    for name, directory, pattern, min_files, year_range in channels:
        if not directory:
            print(f"  [SKIP] {name}: no directory configured")
            continue

        status, issues = check_directory(name, directory, pattern, min_files, year_range)
        status_symbol = "✓" if status else "✗"
        print(f"\n[{status_symbol}] {name:12s} ({directory})")
        for msg in issues:
            prefix = "    " if status else "    ⚠ "
            print(f"{prefix}{msg}")
        results.append((name, status, issues))

    # Summary
    n_pass = sum(1 for _, s, _ in results if s)
    n_fail = len(results) - n_pass

    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed ({len(results)} total)")
    if n_fail:
        print("  Failed channels:")
        for name, status, issues in results:
            if not status:
                print(f"    - {name}: {issues[0] if issues else 'unknown'}")
    print("=" * 70)

    if args.strict and n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
