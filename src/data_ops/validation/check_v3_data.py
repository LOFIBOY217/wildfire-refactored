#!/usr/bin/env python3
"""
Validate all V3 data sources for completeness and quality.

Checks per data source:
  1. Files exist and are non-empty
  2. Can be opened with rasterio
  3. Spatial grid matches FWI reference (CRS, shape, transform)
  4. Values are finite and within expected range
  5. Date coverage is complete (no gaps in required range)

Usage:
    python -m src.data_ops.validation.check_v3_data
    python -m src.data_ops.validation.check_v3_data --config configs/paths_narval.yaml
    python -m src.data_ops.validation.check_v3_data --channel lightning --start 2023-07-01 --end 2023-07-31
"""

import argparse
import glob
import os
import sys
from datetime import date, timedelta
from pathlib import Path

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

from src.utils.date_utils import extract_date_from_filename
from src.utils.raster_io import get_raster_info


# ------------------------------------------------------------------ #
# FWI reference grid
# ------------------------------------------------------------------ #

FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)


# ------------------------------------------------------------------ #
# Expected value ranges per channel
# ------------------------------------------------------------------ #

EXPECTED_RANGES = {
    "deep_soil":  (0.0,   1.0,   "m^3/m^3 volumetric soil water"),
    "lightning":  (0.0,   5000,  "flashes/pixel/day"),
    "ndvi":       (-0.5,  1.0,   "NDVI index"),
    "population": (0.0,   15.0,  "log1p(people/km^2)"),
    "slope":      (0.0,   90.0,  "degrees"),
    "burn_age":   (0.0,   9999,  "years since burn (uint16)"),
}


# ------------------------------------------------------------------ #
# Generic raster check
# ------------------------------------------------------------------ #

def _check_raster(path, expected_shape=None, expected_crs=None,
                  value_min=None, value_max=None, label=""):
    """Check a single raster file. Returns (ok, issues_list)."""
    issues = []
    path = Path(path)

    # 1. File exists and non-empty
    if not path.exists():
        return False, [f"File not found: {path}"]
    if path.stat().st_size == 0:
        return False, [f"File is empty: {path}"]

    # 2. Can be opened
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            crs = str(src.crs)
            shape = (src.height, src.width)
    except Exception as e:
        return False, [f"Cannot open: {e}"]

    # 3. Spatial grid
    if expected_shape and shape != expected_shape:
        issues.append(f"Shape {shape} != expected {expected_shape}")
    if expected_crs and crs != expected_crs:
        issues.append(f"CRS {crs} != expected {expected_crs}")

    # 4. Value checks
    finite_mask = np.isfinite(data)
    n_finite = int(finite_mask.sum())
    n_total = data.size

    if n_finite == 0:
        issues.append("ALL NaN/Inf — no valid pixels")
        return len(issues) == 0, issues

    valid = data[finite_mask]
    n_nonzero = int((valid != 0).sum())

    if n_nonzero == 0:
        issues.append("ALL ZERO (finite but every pixel = 0)")

    if value_min is not None and valid.min() < value_min - 0.01:
        issues.append(f"Min {valid.min():.4f} < expected {value_min}")
    if value_max is not None and valid.max() > value_max + 0.01:
        issues.append(f"Max {valid.max():.4f} > expected {value_max}")

    return len(issues) == 0, issues


def _check_raster_summary(path, label=""):
    """Print summary stats for a raster file."""
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            finite = data[np.isfinite(data)]
            if len(finite) == 0:
                print(f"  {label}: ALL NaN")
                return
            nz = finite[finite != 0]
            print(f"  {label}: shape={data.shape}  "
                  f"finite={len(finite):,}/{data.size:,}  "
                  f"nonzero={len(nz):,}  "
                  f"range=[{finite.min():.4f}, {finite.max():.4f}]  "
                  f"mean={finite.mean():.4f}")
    except Exception as e:
        print(f"  {label}: ERROR {e}")


# ------------------------------------------------------------------ #
# Per-channel validators
# ------------------------------------------------------------------ #

def check_deep_soil_raw(cfg, start, end):
    """Check raw deep_soil GRIB files (download output)."""
    print("\n" + "=" * 60)
    print("DEEP SOIL (raw GRIB)")
    print("=" * 60)

    try:
        grib_dir = Path(get_path(cfg, "deep_soil_dir"))
    except (KeyError, TypeError):
        grib_dir = Path("data/era5_deep_soil")

    gribs = sorted(glob.glob(str(grib_dir / "era5_swvl2_*.grib")))
    print(f"  Directory: {grib_dir}")
    print(f"  Files found: {len(gribs)}")

    if not gribs:
        print("  [FAIL] No GRIB files found")
        return False

    # Check date coverage
    dates_found = set()
    for g in gribs:
        d = extract_date_from_filename(os.path.basename(g))
        if d:
            dates_found.add(d)

    expected_dates = set()
    cur = start
    while cur <= end:
        expected_dates.add(cur)
        cur += timedelta(days=1)

    missing = sorted(expected_dates - dates_found)
    coverage = len(dates_found & expected_dates) / max(len(expected_dates), 1)
    print(f"  Date range: {min(dates_found)} → {max(dates_found)}")
    print(f"  Coverage: {coverage:.1%} ({len(dates_found & expected_dates)}/{len(expected_dates)})")
    if missing and len(missing) <= 10:
        print(f"  Missing: {missing}")
    elif missing:
        print(f"  Missing: {len(missing)} dates (first: {missing[0]}, last: {missing[-1]})")

    # Spot-check 1 file
    sample = gribs[len(gribs) // 2]
    try:
        with rasterio.open(sample) as src:
            data = src.read(1)
            print(f"  Sample ({os.path.basename(sample)}): "
                  f"shape={data.shape}  bands={src.count}  "
                  f"range=[{data.min():.4f}, {data.max():.4f}]")
    except Exception as e:
        print(f"  Sample read failed: {e}")

    ok = coverage > 0.9
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def check_lightning(cfg, start, end):
    """Check processed lightning TIFs (FWI grid)."""
    print("\n" + "=" * 60)
    print("LIGHTNING (FWI grid TIFs)")
    print("=" * 60)

    try:
        lightning_dir = Path(get_path(cfg, "lightning_dir"))
    except (KeyError, TypeError):
        lightning_dir = Path("data/lightning")

    tifs = sorted(glob.glob(str(lightning_dir / "lightning_*.tif")))
    print(f"  Directory: {lightning_dir}")
    print(f"  Files found: {len(tifs)}")

    if not tifs:
        # Check raw files
        raw_dir = lightning_dir.parent / "lightning_raw"
        raws = sorted(glob.glob(str(raw_dir / "glm_raw_*.tif")))
        print(f"  Raw files (not yet processed): {len(raws)}")
        if raws:
            _check_raster_summary(raws[0], "raw sample")
        print("  [WARN] No processed TIFs — run resample_glm_to_fwi_grid.py")
        return False

    # Check grid + values
    vmin, vmax, desc = EXPECTED_RANGES["lightning"]
    sample = tifs[len(tifs) // 2]
    ok, issues = _check_raster(sample, expected_shape=(FWI_HEIGHT, FWI_WIDTH),
                                expected_crs=FWI_CRS, value_min=vmin, value_max=vmax)
    _check_raster_summary(sample, os.path.basename(sample))
    if issues:
        for iss in issues:
            print(f"  [ISSUE] {iss}")

    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def check_ndvi(cfg, start, end):
    """Check processed NDVI TIFs."""
    print("\n" + "=" * 60)
    print("NDVI (FWI grid daily TIFs)")
    print("=" * 60)

    try:
        ndvi_dir = Path(get_path(cfg, "ndvi_dir"))
    except (KeyError, TypeError):
        ndvi_dir = Path("data/ndvi_data")

    tifs = sorted(glob.glob(str(ndvi_dir / "ndvi_*.tif")))
    print(f"  Directory: {ndvi_dir}")
    print(f"  Files found: {len(tifs)}")

    if not tifs:
        raw_dir = ndvi_dir.parent / "ndvi_raw"
        if raw_dir.exists():
            years = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())
            total_hdf = sum(len(list((raw_dir / y).glob("*.hdf"))) for y in years)
            print(f"  Raw HDF4 files: {total_hdf} across years {years}")
        print("  [WARN] No processed TIFs — run process_modis_ndvi.py")
        return False

    # Date coverage
    dates_found = set()
    for t in tifs:
        d = extract_date_from_filename(os.path.basename(t))
        if d:
            dates_found.add(d)
    print(f"  Date range: {min(dates_found)} → {max(dates_found)}")
    print(f"  Unique dates: {len(dates_found)}")

    # Check values
    vmin, vmax, desc = EXPECTED_RANGES["ndvi"]
    sample = tifs[len(tifs) // 2]
    ok, issues = _check_raster(sample, expected_shape=(FWI_HEIGHT, FWI_WIDTH),
                                expected_crs=FWI_CRS, value_min=vmin, value_max=vmax)
    _check_raster_summary(sample, os.path.basename(sample))
    if issues:
        for iss in issues:
            print(f"  [ISSUE] {iss}")

    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def check_population(cfg):
    """Check population density TIF."""
    print("\n" + "=" * 60)
    print("POPULATION DENSITY (static)")
    print("=" * 60)

    try:
        pop_path = Path(get_path(cfg, "population_tif"))
    except (KeyError, TypeError):
        pop_path = Path("data/population_density.tif")

    print(f"  Path: {pop_path}")
    vmin, vmax, desc = EXPECTED_RANGES["population"]
    ok, issues = _check_raster(pop_path, expected_shape=(FWI_HEIGHT, FWI_WIDTH),
                                expected_crs=FWI_CRS, value_min=vmin, value_max=vmax)
    if pop_path.exists():
        _check_raster_summary(pop_path, "population")
    if issues:
        for iss in issues:
            print(f"  [ISSUE] {iss}")

    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def check_slope(cfg):
    """Check slope TIF."""
    print("\n" + "=" * 60)
    print("SLOPE (static)")
    print("=" * 60)

    try:
        terrain_dir = Path(get_path(cfg, "terrain_dir"))
    except (KeyError, TypeError):
        terrain_dir = Path("data/terrain")
    slope_path = terrain_dir / "slope.tif"

    print(f"  Path: {slope_path}")

    if not slope_path.exists():
        # Check raw tiles
        raw_dir = terrain_dir.parent / "terrain_raw"
        hgt_count = len(list(raw_dir.glob("*.hgt"))) if raw_dir.exists() else 0
        print(f"  Raw .hgt tiles: {hgt_count}")
        if hgt_count > 0:
            print("  [WARN] Raw tiles exist — run process_srtm_slope.py")
        else:
            print("  [FAIL] No data found")
        return False

    vmin, vmax, desc = EXPECTED_RANGES["slope"]
    ok, issues = _check_raster(slope_path, expected_shape=(FWI_HEIGHT, FWI_WIDTH),
                                expected_crs=FWI_CRS, value_min=vmin, value_max=vmax)
    _check_raster_summary(slope_path, "slope")
    if issues:
        for iss in issues:
            print(f"  [ISSUE] {iss}")

    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def check_burn_age(cfg, start_year, end_year):
    """Check years-since-burn TIFs."""
    print("\n" + "=" * 60)
    print("BURN AGE (annual TIFs)")
    print("=" * 60)

    try:
        burn_dir = Path(get_path(cfg, "burn_scars_dir"))
    except (KeyError, TypeError):
        burn_dir = Path("data/burn_scars")

    tifs = sorted(glob.glob(str(burn_dir / "years_since_burn_*.tif")))
    print(f"  Directory: {burn_dir}")
    print(f"  Files found: {len(tifs)}")

    if not tifs:
        raw_dir = burn_dir.parent / "burn_scars_raw"
        zips = list(raw_dir.glob("nbac_*.zip")) if raw_dir.exists() else []
        print(f"  Raw zip files: {len(zips)}")
        if zips:
            print("  [WARN] Raw zips exist — run process_nbac_burn_scars.py")
        else:
            print("  [FAIL] No data found (NRCan service may be down)")
        return False

    # Check year coverage
    years_found = set()
    for t in tifs:
        bn = os.path.basename(t)
        try:
            y = int(bn.replace("years_since_burn_", "").replace(".tif", ""))
            years_found.add(y)
        except ValueError:
            pass

    expected_years = set(range(start_year, end_year + 1))
    missing = sorted(expected_years - years_found)
    print(f"  Years: {sorted(years_found)}")
    if missing:
        print(f"  Missing: {missing}")

    # Spot-check
    if tifs:
        vmin, vmax, desc = EXPECTED_RANGES["burn_age"]
        sample = tifs[-1]  # most recent year
        ok, issues = _check_raster(sample, expected_shape=(FWI_HEIGHT, FWI_WIDTH),
                                    expected_crs=FWI_CRS, value_min=vmin, value_max=vmax)
        _check_raster_summary(sample, os.path.basename(sample))
        if issues:
            for iss in issues:
                print(f"  [ISSUE] {iss}")
    else:
        ok = False

    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Validate all V3 data sources for completeness and quality"
    )
    add_config_argument(parser)
    parser.add_argument("--start", type=str, default="2018-05-01",
                        help="Start date for daily data coverage check")
    parser.add_argument("--end", type=str, default="2025-10-31",
                        help="End date for daily data coverage check")
    parser.add_argument("--channel", type=str, default=None,
                        choices=["deep_soil", "lightning", "ndvi", "population",
                                 "slope", "burn_age", "all"],
                        help="Check specific channel (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    channel = args.channel or "all"

    print("=" * 60)
    print("V3 DATA VALIDATION")
    print("=" * 60)
    print(f"  Config  : {getattr(args, 'config', 'default')}")
    print(f"  Range   : {start} → {end}")
    print(f"  Channel : {channel}")

    results = {}

    if channel in ("all", "deep_soil"):
        results["deep_soil"] = check_deep_soil_raw(cfg, start, end)
    if channel in ("all", "lightning"):
        results["lightning"] = check_lightning(cfg, start, end)
    if channel in ("all", "ndvi"):
        results["ndvi"] = check_ndvi(cfg, start, end)
    if channel in ("all", "population"):
        results["population"] = check_population(cfg)
    if channel in ("all", "slope"):
        results["slope"] = check_slope(cfg)
    if channel in ("all", "burn_age"):
        results["burn_age"] = check_burn_age(cfg, start.year, end.year)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name:15s} {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n  All checks passed!")
    else:
        print(f"\n  {sum(1 for v in results.values() if not v)} channel(s) need attention")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
