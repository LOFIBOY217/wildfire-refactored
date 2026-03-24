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
    import cfgrib
    _CFGRIB_OK = True
except Exception:
    _CFGRIB_OK = False

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject

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

# GRIB short-name / cfgrib name → output channel name
# cfgrib uses ERA5-style short names; S2S uses the same ECMWF parametrisation
GRIB_TO_CHAN = {
    "t2m":   "2t",    # 2m temperature
    "d2m":   "2d",    # 2m dewpoint
    "tcw":   "tcw",   # total column water
    "swvl1": "sm20",  # volumetric soil water layer 1
    "stl1":  "st20",  # soil temperature layer 1
}

# Channels stored in Kelvin in GRIB → convert to Celsius
KELVIN_CHANS = {"2t", "2d", "st20"}

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

def process_grib_file(grib_path: Path, out_dir: Path, ref_path: Path,
                      skip_existing: bool = True) -> bool:
    """
    Process one S2S GRIB file → per-lead-day 6-band GeoTIFFs.

    Returns True on success, False on failure.
    """
    if not _CFGRIB_OK:
        print("[ERROR] cfgrib not available. Load the eccodes module first.", file=sys.stderr)
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
        # Open all GRIB messages grouped by step / variable
        datasets = cfgrib.open_datasets(str(grib_path))
    except Exception as exc:
        print(f"[ERROR] cfgrib.open_datasets failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return False

    # Gather available channels: chan_name → xarray DataArray indexed by stepRange
    chan_arrays = {}   # chan_name → {step_start_h: np.ndarray 2-D}

    for ds in datasets:
        for grib_name, chan_name in GRIB_TO_CHAN.items():
            if grib_name not in ds.data_vars:
                continue

            da = ds[grib_name]

            # Latitude / longitude
            lats = ds.latitude.values
            lons = ds.longitude.values

            # Build source transform (EPSG:4326)
            lat_min, lat_max = float(lats.min()), float(lats.max())
            lon_min, lon_max = float(lons.min()), float(lons.max())
            if lats[0] > lats[-1]:
                flip_lat = True
            else:
                flip_lat = False

            src_crs = CRS.from_epsg(4326)

            # Iterate over step dimension
            # cfgrib may store step as timedelta or as a coordinate
            step_dim = None
            for dim in da.dims:
                if "step" in dim.lower():
                    step_dim = dim
                    break

            if step_dim is None:
                # Single step
                steps = [None]
            else:
                steps = da[step_dim].values

            if chan_name not in chan_arrays:
                chan_arrays[chan_name] = {}

            for step_val in steps:
                if step_val is not None:
                    arr = da.sel({step_dim: step_val}).values
                    # stepRange attribute: e.g. "336-360"
                    try:
                        step_range = str(da.sel({step_dim: step_val}).attrs.get("GRIB_stepRange", ""))
                        if "-" in step_range:
                            step_start_h = int(step_range.split("-")[0])
                        else:
                            # Instantaneous: step is end hour
                            import pandas as pd
                            step_start_h = int(pd.Timedelta(step_val).total_seconds() / 3600)
                    except Exception:
                        import pandas as pd
                        step_start_h = int(pd.Timedelta(step_val).total_seconds() / 3600)
                else:
                    arr = da.values
                    step_start_h = 336  # default to first lead

                if arr.ndim != 2:
                    arr = np.squeeze(arr)
                if arr.ndim != 2:
                    continue

                # Kelvin → Celsius
                if chan_name in KELVIN_CHANS:
                    arr = arr - 273.15

                # Flip latitude if needed (N→S to S→N)
                if flip_lat:
                    arr = np.flipud(arr)
                    _lat_min, _lat_max = lat_min, lat_max
                else:
                    _lat_min, _lat_max = lat_min, lat_max

                h, w = arr.shape
                src_transform = from_bounds(lon_min, _lat_min, lon_max, _lat_max, w, h)

                chan_arrays[chan_name][step_start_h] = (
                    arr.astype(np.float32), src_crs, src_transform
                )

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
        step_start_h = lead_day * 24          # e.g. lead 14 → step 336
        out_tif = date_dir / f"lead{lead_day:02d}.tif"

        if skip_existing and out_tif.exists() and out_tif.stat().st_size > 0:
            n_ok += 1
            continue

        bands = []
        missing = []
        for chan in ["2t", "2d", "tcw", "sm20", "st20"]:
            if chan in chan_arrays and step_start_h in chan_arrays[chan]:
                arr, src_crs_c, src_tf = chan_arrays[chan][step_start_h]
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

    args = parser.parse_args()
    cfg  = load_config(args.config)

    # ---- Resolve paths ----
    if args.input:
        grib_files = [Path(args.input)]
    else:
        s2s_dir = Path(
            args.s2s_dir or
            cfg.get("paths", {}).get("s2s_dir") or
            get_path(cfg, "ecmwf_dir")
        )
        grib_files = sorted(s2s_dir.glob("s2s_ecmf_cf_*.grib"))
        print(f"Found {len(grib_files)} S2S GRIB files in {s2s_dir}")

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

    print(f"\nS2S → EPSG:3978 processing")
    print(f"  GRIB files  : {len(grib_files)}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Reference   : {ref_path}")
    print(f"  Lead days   : {LEAD_DAYS[0]}–{LEAD_DAYS[-1]}  ({len(LEAD_DAYS)} days)")
    print(f"  Channels    : 2t / 2d / tcw / sm20 / st20 / VPD")
    print()

    skip_existing = not args.overwrite

    if args.workers > 1:
        from multiprocessing import Pool
        tasks = [(str(f), str(out_dir), str(ref_path), skip_existing) for f in grib_files]

        def _worker(t):
            return process_grib_file(Path(t[0]), Path(t[1]), Path(t[2]), t[3])

        with Pool(args.workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [
            process_grib_file(f, out_dir, ref_path, skip_existing)
            for f in grib_files
        ]

    n_ok   = sum(results)
    n_fail = len(results) - n_ok
    print(f"\n{'='*70}")
    print(f"Done.  {n_ok} succeeded  |  {n_fail} failed")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
