#!/usr/bin/env python3
"""
Process ECMWF S2S GRIB files → reprojected GeoTIFFs aligned to the EPSG:3978
Canada Lambert grid (same as FWI/ERA5 training data).

For each issue date, extracts 6 channels at each lead day (14–45):
  Band 1 : 2t   – 2m temperature (°C)
  Band 2 : 2d   – 2m dewpoint    (°C)
  Band 3 : tcw  – total column water (kg/m²)
  Band 4 : sm20 – volumetric soil water layer 1 (m³/m³)
  Band 5 : st20 – soil temperature layer 1 (°C)
  Band 6 : VPD  – vapour pressure deficit (kPa), computed from 2t + 2d

Output layout:
  {out_dir}/{YYYY-MM-DD}/lead{kk}.tif   (6-band float32, EPSG:3978)

where kk = lead day (14 … 45).

Usage:
    python -m src.data_ops.processing.process_s2s_to_tif \\
        --s2s-dir  data/s2s_forecast \\
        --out-dir  data/s2s_processed \\
        --reference data/fwi_data/fwi_20250615.tif

    # Single file:
    python -m src.data_ops.processing.process_s2s_to_tif \\
        --input data/s2s_forecast/s2s_ecmf_cf_2020-06-01.grib \\
        --out-dir data/s2s_processed \\
        --reference data/fwi_data/fwi_20250615.tif
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

try:
    import eccodes as _eccodes
    _ECCODES_OK = True
except Exception:
    _ECCODES_OK = False

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GRIB shortName → output channel name  (eccodes uses ECMWF shortNames directly)
GRIB_TO_CHAN = {
    "2t":   "2t",    # 2m temperature
    "2d":   "2d",    # 2m dewpoint
    "tcw":  "tcw",   # total column water
    "sm20": "sm20",  # volumetric soil water layer 1
    "st20": "st20",  # soil temperature layer 1
}

# Channels stored in Kelvin in GRIB → convert to Celsius
KELVIN_CHANS = {"2t", "2d", "st20"}

# Extended S2S variables (wind + precipitation)
GRIB_TO_CHAN_EXT = {
    "10u":  "10u",   # 10m U-wind component (m/s)
    "10v":  "10v",   # 10m V-wind component (m/s)
    "tp":   "tp",    # total precipitation (m)
}
KELVIN_CHANS_EXT = set()  # no Kelvin conversion for extended vars

# Lead days covered by the downloaded STEP_STRING (336-360 … 1080-1104)
# step_start / 24 gives the lead day index
LEAD_DAYS = list(range(14, 46))   # 14 … 45  (32 lead days)


# ---------------------------------------------------------------------------
# VPD helper
# ---------------------------------------------------------------------------

def compute_vpd(t_celsius: np.ndarray, td_celsius: np.ndarray) -> np.ndarray:
    """
    Vapour Pressure Deficit  [kPa].

    VPD = e_sat(T) - e_sat(Td)   with the Magnus formula.
    """
    def _esat(tc):
        return 0.6108 * np.exp(17.27 * tc / (tc + 237.3))

    vpd = _esat(t_celsius) - _esat(td_celsius)
    return vpd.astype(np.float32)


# ---------------------------------------------------------------------------
# Reprojection helper
# ---------------------------------------------------------------------------

def reproject_to_ref(
    data: np.ndarray,
    src_crs: CRS,
    src_transform,
    ref_path: Path,
) -> np.ndarray:
    """
    Reproject a 2-D float32 array to match the reference TIF grid
    (CRS, transform, height, width).

    Returns a 2-D float32 array of shape (ref_height, ref_width).
    """
    with rasterio.open(ref_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_height    = ref.height
        dst_width     = ref.width

    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    reproject(
        source        = data.astype(np.float32),
        destination   = dst,
        src_transform = src_transform,
        src_crs       = src_crs,
        dst_transform = dst_transform,
        dst_crs       = dst_crs,
        resampling    = Resampling.bilinear,
        src_nodata    = np.nan,
        dst_nodata    = np.nan,
    )
    return dst


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _read_grib_eccodes(grib_path: Path, chan_map=None, kelvin_chans=None) -> dict:
    """
    Read all messages from a GRIB file using eccodes (fast, no index building).

    Parameters:
        grib_path: Path to the GRIB file.
        chan_map: dict mapping GRIB shortName → output channel name.
                 Defaults to GRIB_TO_CHAN.
        kelvin_chans: set of channel names that need Kelvin → Celsius conversion.
                      Defaults to KELVIN_CHANS.

    Returns:
        chan_arrays: dict  chan_name → {step_start_h: (arr_2d float32, src_crs, src_transform)}
    """
    import eccodes

    if chan_map is None:
        chan_map = GRIB_TO_CHAN
    if kelvin_chans is None:
        kelvin_chans = KELVIN_CHANS

    chan_arrays = {}
    src_crs = CRS.from_epsg(4326)
    grid_info = {}   # shortName → (nrows, ncols, lat_min, lat_max, lon_min, lon_max, flip_lat)

    with open(grib_path, "rb") as f:
        while True:
            msg = eccodes.codes_grib_new_from_file(f)
            if msg is None:
                break

            try:
                short_name = eccodes.codes_get(msg, "shortName")
                chan_name  = chan_map.get(short_name)
                if chan_name is None:
                    continue

                # Parse step range to determine lead day.
                # Two formats exist in ECMWF S2S GRIB:
                #   "336-360"  — daily-average range (core vars: 2t, 2d, tcw, sm20, st20)
                #                → use start of range: 336 // 24 = lead 14
                #   "0-336"    — accumulated from forecast start (precip: tp, cp)
                #                → use END of range: 336 // 24 = lead 14
                #   "336"      — instantaneous (wind: 10u, 10v)
                #                → use value directly
                step_range = eccodes.codes_get(msg, "stepRange")
                parts = str(step_range).split("-")
                if len(parts) == 2:
                    start_h, end_h = int(parts[0]), int(parts[1])
                    # Accumulated field (start=0): use end step for lead day
                    step_start_h = end_h if start_h == 0 else start_h
                else:
                    step_start_h = int(step_range)

                lead_day = step_start_h // 24   # 336 → 14

                # Grid dimensions
                nrows = eccodes.codes_get(msg, "Nj")
                ncols = eccodes.codes_get(msg, "Ni")

                # Values — eccodes applies bitmap automatically (missing → nan)
                missing_val = eccodes.codes_get(msg, "missingValue")
                values = eccodes.codes_get_values(msg).reshape(nrows, ncols).astype(np.float32)

                # Replace ECMWF missing value sentinel with NaN
                values[values == missing_val] = np.nan

                # Lat/lon for transform (only need to compute once per shortName)
                if short_name not in grid_info:
                    lats_1d = eccodes.codes_get_array(msg, "latitudes")
                    lons_1d = eccodes.codes_get_array(msg, "longitudes")
                    lats_2d = lats_1d.reshape(nrows, ncols)
                    lon_row  = lons_1d[:ncols]   # first-row longitudes (0 … 358.5)

                    lat_min  = float(lats_2d.min())
                    lat_max  = float(lats_2d.max())
                    flip_lat = (lats_2d[0, 0] > lats_2d[-1, 0])   # N→S → True

                    # Detect 0-360 grid and compute roll split
                    uses_360 = bool(lon_row[-1] > 180)
                    if uses_360:
                        dx = float(lon_row[1] - lon_row[0])          # 1.5°
                        split = int(np.searchsorted(lon_row, 180.0, side="right"))
                        # After roll: westernmost center = lon_row[split] - 360
                        lon_min = float(lon_row[split] - 360) - dx / 2
                        lon_max = float(lon_row[split - 1]) + dx / 2
                    else:
                        dx = float(lon_row[1] - lon_row[0])
                        split = 0
                        lon_min = float(lon_row[0])  - dx / 2
                        lon_max = float(lon_row[-1]) + dx / 2

                    dy = abs(lats_2d[1, 0] - lats_2d[0, 0])
                    grid_info[short_name] = (nrows, ncols,
                                             lat_min, lat_max, dy,
                                             lon_min, lon_max, dx,
                                             flip_lat, split)
                else:
                    (nrows, ncols,
                     lat_min, lat_max, dy,
                     lon_min, lon_max, dx,
                     flip_lat, split) = grid_info[short_name]

                # Kelvin → Celsius
                if chan_name in kelvin_chans:
                    values -= 273.15

                # Flip N→S to S→N so rasterio from_bounds works correctly
                if flip_lat:
                    values = np.flipud(values)

                # Roll 0-360 → -180-180 column ordering
                if split > 0:
                    values = np.roll(values, -split, axis=1)

                src_transform = from_bounds(lon_min, lat_min - dy / 2,
                                            lon_max, lat_max + dy / 2,
                                            ncols, nrows)

                if chan_name not in chan_arrays:
                    chan_arrays[chan_name] = {}
                chan_arrays[chan_name][lead_day] = (values, src_crs, src_transform)

            finally:
                eccodes.codes_release(msg)

    return chan_arrays


def process_grib_file(grib_path: Path, out_dir: Path, ref_path: Path,
                      skip_existing: bool = True) -> bool:
    """
    Process one S2S GRIB file → per-lead-day 6-band GeoTIFFs.

    Returns True on success, False on failure.
    """
    if not _ECCODES_OK:
        print("[ERROR] eccodes not available. Run: module load eccodes/2.31.0", file=sys.stderr)
        return False

    # Parse issue date from filename: s2s_ecmf_cf_YYYY-MM-DD.grib
    stem = grib_path.stem
    try:
        issue_date = stem.split("s2s_ecmf_cf_")[1]  # "YYYY-MM-DD"
    except IndexError:
        print(f"[ERROR] Cannot parse issue date from: {grib_path.name}", file=sys.stderr)
        return False

    date_dir = out_dir / issue_date
    date_dir.mkdir(parents=True, exist_ok=True)

    # Check if all lead-day TIFs already exist
    if skip_existing:
        existing = [date_dir / f"lead{k:02d}.tif" for k in LEAD_DAYS]
        if all(p.exists() and p.stat().st_size > 0 for p in existing):
            print(f"[SKIP] {issue_date} — all {len(LEAD_DAYS)} lead TIFs already exist")
            return True

    print(f"\n{'='*70}")
    print(f"Processing: {grib_path.name}  →  {date_dir}")
    print(f"{'='*70}")

    try:
        chan_arrays = _read_grib_eccodes(grib_path)
    except Exception as exc:
        print(f"[ERROR] reading {grib_path.name}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return False

    if not chan_arrays:
        print(f"[ERROR] No recognised variables found in {grib_path.name}", file=sys.stderr)
        return False

    found_chans = sorted(chan_arrays.keys())
    print(f"  Found channels: {found_chans}")

    # Get reference grid shape
    with rasterio.open(ref_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_height    = ref.height
        dst_width     = ref.width

    # Write one TIF per lead day
    n_ok = 0
    for lead_day in LEAD_DAYS:
        out_tif = date_dir / f"lead{lead_day:02d}.tif"

        if skip_existing and out_tif.exists() and out_tif.stat().st_size > 0:
            n_ok += 1
            continue

        bands = []
        missing = []
        for chan in ["2t", "2d", "tcw", "sm20", "st20"]:
            if chan in chan_arrays and lead_day in chan_arrays[chan]:
                arr, src_crs_c, src_tf = chan_arrays[chan][lead_day]
                reprojected = reproject_to_ref(arr, src_crs_c, src_tf, ref_path)
                bands.append(reprojected)
            else:
                missing.append(chan)
                bands.append(np.full((dst_height, dst_width), np.nan, dtype=np.float32))

        if missing:
            print(f"  [WARN] lead{lead_day:02d}: missing channels {missing}")

        # Compute VPD from reprojected 2t and 2d
        t_arr  = bands[0]   # 2t in °C
        td_arr = bands[1]   # 2d in °C
        # Avoid log(0) issues: clip temperature difference
        td_clipped = np.minimum(td_arr, t_arr)
        vpd = compute_vpd(t_arr, td_clipped)
        bands.append(vpd)   # Band 6

        stack = np.stack(bands, axis=0)   # (6, H, W)

        with rasterio.open(
            out_tif, "w",
            driver    = "GTiff",
            height    = dst_height,
            width     = dst_width,
            count     = 6,
            dtype     = "float32",
            crs       = dst_crs,
            transform = dst_transform,
            compress  = "lzw",
        ) as dst:
            dst.write(stack)
            dst.update_tags(
                issue_date  = issue_date,
                lead_day    = str(lead_day),
                channels    = "2t,2d,tcw,sm20,st20,VPD",
                source      = "ECMWF_S2S",
            )

        n_ok += 1

    print(f"  Written {n_ok}/{len(LEAD_DAYS)} lead-day TIFs → {date_dir}")
    return n_ok > 0


def process_ext_grib_file(grib_path: Path, out_dir: Path, ref_path: Path,
                          skip_existing: bool = True) -> bool:
    """
    Process one S2S *extended* GRIB file → per-lead-day 3-band GeoTIFFs.

    Bands: 10u, 10v, tp  (no VPD computation).
    Output: {out_dir}/{YYYY-MM-DD}/lead{kk}_ext.tif

    Returns True on success, False on failure.
    """
    if not _ECCODES_OK:
        print("[ERROR] eccodes not available. Run: module load eccodes/2.31.0", file=sys.stderr)
        return False

    # Parse issue date from filename: s2s_ecmf_cf_ext_YYYY-MM-DD.grib
    stem = grib_path.stem
    try:
        issue_date = stem.split("s2s_ecmf_cf_ext_")[1]  # "YYYY-MM-DD"
    except IndexError:
        print(f"[ERROR] Cannot parse issue date from: {grib_path.name}", file=sys.stderr)
        return False

    date_dir = out_dir / issue_date
    date_dir.mkdir(parents=True, exist_ok=True)

    # Check if all lead-day ext TIFs already exist
    if skip_existing:
        existing = [date_dir / f"lead{k:02d}_ext.tif" for k in LEAD_DAYS]
        if all(p.exists() and p.stat().st_size > 0 for p in existing):
            print(f"[SKIP] {issue_date} ext — all {len(LEAD_DAYS)} lead TIFs already exist")
            return True

    print(f"\n{'='*70}")
    print(f"Processing ext: {grib_path.name}  →  {date_dir}")
    print(f"{'='*70}")

    try:
        chan_arrays = _read_grib_eccodes(
            grib_path, chan_map=GRIB_TO_CHAN_EXT, kelvin_chans=KELVIN_CHANS_EXT
        )
    except Exception as exc:
        print(f"[ERROR] reading {grib_path.name}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return False

    if not chan_arrays:
        print(f"[ERROR] No recognised variables found in {grib_path.name}", file=sys.stderr)
        return False

    found_chans = sorted(chan_arrays.keys())
    print(f"  Found channels: {found_chans}")

    # Get reference grid shape
    with rasterio.open(ref_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_height    = ref.height
        dst_width     = ref.width

    ext_chan_order = ["10u", "10v", "tp"]

    # Write one TIF per lead day
    n_ok = 0
    for lead_day in LEAD_DAYS:
        out_tif = date_dir / f"lead{lead_day:02d}_ext.tif"

        if skip_existing and out_tif.exists() and out_tif.stat().st_size > 0:
            n_ok += 1
            continue

        bands = []
        missing = []
        for chan in ext_chan_order:
            if chan in chan_arrays and lead_day in chan_arrays[chan]:
                arr, src_crs_c, src_tf = chan_arrays[chan][lead_day]
                reprojected = reproject_to_ref(arr, src_crs_c, src_tf, ref_path)
                bands.append(reprojected)
            else:
                missing.append(chan)
                bands.append(np.full((dst_height, dst_width), np.nan, dtype=np.float32))

        if missing:
            print(f"  [WARN] lead{lead_day:02d} ext: missing channels {missing}")

        stack = np.stack(bands, axis=0)   # (3, H, W)

        with rasterio.open(
            out_tif, "w",
            driver    = "GTiff",
            height    = dst_height,
            width     = dst_width,
            count     = 3,
            dtype     = "float32",
            crs       = dst_crs,
            transform = dst_transform,
            compress  = "lzw",
        ) as dst:
            dst.write(stack)
            dst.update_tags(
                issue_date  = issue_date,
                lead_day    = str(lead_day),
                channels    = ",".join(ext_chan_order),
                source      = "ECMWF_S2S_EXT",
            )

        n_ok += 1

    print(f"  Written {n_ok}/{len(LEAD_DAYS)} ext lead-day TIFs → {date_dir}")
    return n_ok > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process ECMWF S2S GRIB files → EPSG:3978 GeoTIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_config_argument(parser)

    parser.add_argument(
        "--input", type=str, default=None,
        help="Single GRIB file to process (overrides --s2s-dir)",
    )
    parser.add_argument(
        "--s2s-dir", type=str, default=None,
        help="Directory of S2S GRIB files (default: s2s_dir from config)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output root directory (default: data/s2s_processed)",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Reference GeoTIFF for target grid (CRS + transform + shape). "
             "Default: fwi_reference_tif from config.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process files even if output TIFs already exist",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--ext-dir", type=str, default=None,
        help="Directory of S2S ext GRIB files (s2s_ecmf_cf_ext_*.grib)",
    )
    parser.add_argument(
        "--ext-only", action="store_true",
        help="Process only extended GRIB files, skip core",
    )

    args = parser.parse_args()
    cfg  = load_config(args.config)

    # ---- Resolve paths ----
    s2s_dir = Path(
        args.s2s_dir or
        cfg.get("paths", {}).get("s2s_dir") or
        get_path(cfg, "ecmwf_dir")
    )

    out_dir = Path(
        args.out_dir or
        cfg.get("paths", {}).get("s2s_processed") or
        "data/s2s_processed"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(
        args.reference or
        cfg.get("paths", {}).get("fwi_reference_tif") or
        get_path(cfg, "fwi_reference_tif")
    )
    if not ref_path.exists():
        print(f"[ERROR] Reference TIF not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    skip_existing = not args.overwrite

    # ---- Core GRIB processing ----
    if not args.ext_only:
        if args.input:
            grib_files = [Path(args.input)]
        else:
            grib_files = sorted(s2s_dir.glob("s2s_ecmf_cf_*.grib"))
            # Exclude ext files from core glob
            grib_files = [f for f in grib_files if "_ext_" not in f.name]
            print(f"Found {len(grib_files)} S2S core GRIB files in {s2s_dir}")

        print(f"\nS2S → EPSG:3978 processing (core)")
        print(f"  GRIB files  : {len(grib_files)}")
        print(f"  Output dir  : {out_dir}")
        print(f"  Reference   : {ref_path}")
        print(f"  Lead days   : {LEAD_DAYS[0]}–{LEAD_DAYS[-1]}  ({len(LEAD_DAYS)} days)")
        print(f"  Channels    : 2t / 2d / tcw / sm20 / st20 / VPD")
        print()

        if args.workers > 1:
            from multiprocessing import Pool
            tasks = [(f, out_dir, ref_path, skip_existing) for f in grib_files]
            with Pool(args.workers) as pool:
                results = pool.starmap(process_grib_file, tasks)
        else:
            results = [
                process_grib_file(f, out_dir, ref_path, skip_existing)
                for f in grib_files
            ]

        n_ok   = sum(results)
        n_fail = len(results) - n_ok
        print(f"\n{'='*70}")
        print(f"Core done.  {n_ok} succeeded  |  {n_fail} failed")
        print(f"Output: {out_dir}")

    # ---- Extended GRIB processing ----
    if args.ext_only or args.ext_dir:
        ext_dir = Path(args.ext_dir) if args.ext_dir else s2s_dir
        ext_files = sorted(ext_dir.glob("s2s_ecmf_cf_ext_*.grib"))
        print(f"\nFound {len(ext_files)} S2S ext GRIB files in {ext_dir}")

        print(f"\nS2S → EPSG:3978 processing (extended)")
        print(f"  GRIB files  : {len(ext_files)}")
        print(f"  Output dir  : {out_dir}")
        print(f"  Reference   : {ref_path}")
        print(f"  Lead days   : {LEAD_DAYS[0]}–{LEAD_DAYS[-1]}  ({len(LEAD_DAYS)} days)")
        print(f"  Channels    : 10u / 10v / tp")
        print()

        if args.workers > 1:
            from multiprocessing import Pool
            tasks = [(f, out_dir, ref_path, skip_existing) for f in ext_files]
            with Pool(args.workers) as pool:
                ext_results = pool.starmap(process_ext_grib_file, tasks)
        else:
            ext_results = [
                process_ext_grib_file(f, out_dir, ref_path, skip_existing)
                for f in ext_files
            ]

        n_ok   = sum(ext_results)
        n_fail = len(ext_results) - n_ok
        print(f"\n{'='*70}")
        print(f"Ext done.  {n_ok} succeeded  |  {n_fail} failed")
        print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
