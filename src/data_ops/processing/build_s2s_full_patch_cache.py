"""
Build S2S Full-Patch Decoder Cache
===================================
Converts per-date S2S forecast TIF files into a full-resolution patchified
float16 memmap cache that matches the Oracle encoder format exactly.

Output layout: (n_dates, n_patches, 32, 2048) float16
  - n_dates   : number of YYYY-MM-DD subdirectories found in --s2s-dir
  - n_patches : (H_crop//P * W_crop//P) = 142*169 = 23998  (P=16)
  - 32        : lead days 14..45
  - 2048      : P*P*8 = 256*8 (full patch pixels, 8 channels)

Channel layout (matches Oracle encoder):
  0: FWI   (computed via Van Wagner from S2S weather)
  1: 2t    (2m temperature, deg C)
  2: 2d    (2m dewpoint, deg C)
  3: FFMC  (computed via Van Wagner)
  4: DMC   (computed via Van Wagner)
  5: DC    (computed via Van Wagner)
  6: BUI   (computed via Van Wagner)
  7: fire_clim  (static, log1p-transformed historical fire frequency)

Values are PRE-NORMALIZED using meteo_means/meteo_stds (same as Oracle encoder),
then clipped to [-10, 10].

FWI computation per issue date:
  1. Load observed FFMC/DMC/DC from issue_date (fallback to day before)
  2. Spin-up leads 0..13: advance FWI state using S2S weather (if TIFs exist)
  3. Cache leads 14..45: load weather, advance FWI, stack 8 channels,
     normalize, patchify → (n_patches, 2048)

Companion file: {out_file}.dates.npy — sorted date strings.

Usage:
    python -m src.data_ops.processing.build_s2s_full_patch_cache \\
        --s2s-dir data/s2s_processed \\
        --out-file /scratch/jiaqi217/meteo_cache/s2s_full_patch_cache.dat \\
        --reference data/fwi_data/fwi_20250615.tif \\
        --fire-clim data/fire_climatology.tif \\
        --ffmc-dir data/ffmc_data \\
        --dmc-dir data/dmc_data \\
        --dc-dir data/dc_data \\
        --norm-stats data/norm_stats.npy \\
        --patch-size 16 \\
        --workers 4
"""

import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import rasterio

from src.data_ops.processing.fwi_calculator import (
    FWICalculator,
    dewpoint_to_rh,
    wind_components_to_speed,
    DEFAULT_FFMC,
    DEFAULT_DMC,
    DEFAULT_DC,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S2S_N_LEADS = 32          # lead14 .. lead45
LEAD_START = 14
LEAD_END = 45
N_CHANNELS = 8            # FWI, 2t, 2d, FFMC, DMC, DC, BUI, fire_clim
DEFAULT_WIND_KMH = 12.0   # moderate wind fallback when ext TIF is missing
DEFAULT_RAIN_MM = 0.0

# ---------------------------------------------------------------------------
# Module-level globals set by worker initializer
# ---------------------------------------------------------------------------
_fire_clim = None          # (Hc, Wc) float32, log1p-transformed
_meteo_means = None        # (8,) float32
_meteo_stds = None         # (8,) float32
_ffmc_dict = None          # {YYYY-MM-DD: filepath}
_dmc_dict = None
_dc_dict = None


def _worker_init(fire_clim_path, norm_stats_path, Hc, Wc,
                 ffmc_dir, dmc_dir, dc_dir):
    """Per-worker initializer: load fire_clim, norm_stats, and obs dicts once."""
    global _fire_clim, _meteo_means, _meteo_stds
    global _ffmc_dict, _dmc_dict, _dc_dict

    with rasterio.open(fire_clim_path) as src:
        clim = src.read(1).astype(np.float32)
    clim = np.nan_to_num(clim, nan=0.0)
    clim = np.log1p(clim)
    _fire_clim = clim[:Hc, :Wc]

    stats = np.load(norm_stats_path)  # (2, 8)
    _meteo_means = stats[0].astype(np.float32)
    _meteo_stds = stats[1].astype(np.float32)

    _ffmc_dict = _build_obs_dict(ffmc_dir)
    _dmc_dict = _build_obs_dict(dmc_dir)
    _dc_dict = _build_obs_dict(dc_dir)


# ---------------------------------------------------------------------------
# Date / file helpers
# ---------------------------------------------------------------------------
def _parse_date_dirs(s2s_dir):
    """Return sorted list of YYYY-MM-DD subdirectory names in s2s_dir."""
    entries = []
    for name in os.listdir(s2s_dir):
        full = os.path.join(s2s_dir, name)
        if os.path.isdir(full) and len(name) == 10 and name[4] == '-' and name[7] == '-':
            try:
                parts = name.split('-')
                int(parts[0]); int(parts[1]); int(parts[2])
                entries.append(name)
            except ValueError:
                continue
    return sorted(entries)


def _extract_date_from_filename(fname):
    """Extract YYYY-MM-DD from filenames like ffmc_20220601.tif."""
    m = re.search(r'(\d{4})(\d{2})(\d{2})', fname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _build_obs_dict(directory):
    """Build {YYYY-MM-DD: filepath} dict for all TIFs in directory."""
    import glob as _glob
    result = {}
    for p in sorted(_glob.glob(os.path.join(directory, "*.tif"))):
        d = _extract_date_from_filename(os.path.basename(p))
        if d:
            result[d] = p
    return result


def _load_obs_grid(obs_dict, date_str, default_val, Hc, Wc):
    """
    Load a single-band observed TIF for the given date.
    Falls back to the day before. Returns (Hc, Wc) float64 array.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    for offset in [0, -1]:
        key = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        if key in obs_dict:
            with rasterio.open(obs_dict[key]) as src:
                arr = src.read(1).astype(np.float64)
            arr = np.nan_to_num(arr, nan=default_val)
            return arr[:Hc, :Wc]
    # No observation found — return default
    return np.full((Hc, Wc), default_val, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core worker function
# ---------------------------------------------------------------------------
def _process_one_date(args_tuple):
    """
    Worker: for one issue date, compute FWI from S2S weather and write
    the result directly to the cache file at the correct byte offset.

    Returns (date_str, True) on success, (date_str, False) on error.
    No large arrays are returned through IPC — this avoids the OOM caused
    by Python's multiprocessing queue accumulating many 5.85 GB pickled arrays.
    """
    (date_str, s2s_dir,
     patch_size, n_patches, nph, npw, H, W, Hc, Wc,
     out_file, date_idx, n_dates) = args_tuple

    global _fire_clim, _meteo_means, _meteo_stds
    global _ffmc_dict, _dmc_dict, _dc_dict

    P = patch_size
    C = N_CHANNELS
    feat_dim = P * P * C  # 2048

    result = np.zeros((n_patches, S2S_N_LEADS, feat_dim), dtype=np.float16)

    # ------------------------------------------------------------------
    # 1) Initialize FWI calculator from observed state
    # ------------------------------------------------------------------
    ffmc0 = _load_obs_grid(_ffmc_dict, date_str, DEFAULT_FFMC, Hc, Wc)
    dmc0 = _load_obs_grid(_dmc_dict, date_str, DEFAULT_DMC, Hc, Wc)
    dc0 = _load_obs_grid(_dc_dict, date_str, DEFAULT_DC, Hc, Wc)

    calc = FWICalculator(shape=(Hc, Wc))
    calc.set_state(ffmc0, dmc0, dc0)

    issue_dt = datetime.strptime(date_str, "%Y-%m-%d")
    date_dir = os.path.join(s2s_dir, date_str)

    # ------------------------------------------------------------------
    # 3) Spin-up leads 0..13 (advance FWI state without caching)
    # ------------------------------------------------------------------
    for lead in range(0, LEAD_START):
        weather = _load_weather_for_lead(date_dir, lead, Hc, Wc)
        if weather is None:
            # No TIF for this lead — skip (state stays the same)
            continue
        temp_c, dewp_c, wind_kmh, rain_mm = weather
        rh = dewpoint_to_rh(temp_c, dewp_c)
        valid_dt = issue_dt + timedelta(days=lead)
        month = valid_dt.month
        calc.update(temp_c, rh, wind_kmh, rain_mm, month)

    # ------------------------------------------------------------------
    # 4) Cache leads 14..45
    # ------------------------------------------------------------------
    for li, lead in enumerate(range(LEAD_START, LEAD_END + 1)):
        weather = _load_weather_for_lead(date_dir, lead, Hc, Wc)

        valid_dt = issue_dt + timedelta(days=lead)
        month = valid_dt.month

        if weather is not None:
            temp_c, dewp_c, wind_kmh, rain_mm = weather
        else:
            # Missing lead TIF — use neutral weather to advance state
            temp_c = np.full((Hc, Wc), 15.0, dtype=np.float64)
            dewp_c = np.full((Hc, Wc), 10.0, dtype=np.float64)
            wind_kmh = np.full((Hc, Wc), DEFAULT_WIND_KMH, dtype=np.float64)
            rain_mm = np.full((Hc, Wc), DEFAULT_RAIN_MM, dtype=np.float64)

        rh = dewpoint_to_rh(temp_c, dewp_c)
        fwi_result = calc.update(temp_c, rh, wind_kmh, rain_mm, month)

        # Stack 8 channels: FWI, 2t, 2d, FFMC, DMC, DC, BUI, fire_clim
        frame = np.stack([
            np.nan_to_num(fwi_result["FWI"], nan=0.0).astype(np.float32),
            temp_c.astype(np.float32),
            dewp_c.astype(np.float32),
            np.nan_to_num(fwi_result["FFMC"], nan=0.0).astype(np.float32),
            np.nan_to_num(fwi_result["DMC"], nan=0.0).astype(np.float32),
            np.nan_to_num(fwi_result["DC"], nan=0.0).astype(np.float32),
            np.nan_to_num(fwi_result["BUI"], nan=0.0).astype(np.float32),
            _fire_clim,
        ], axis=-1)  # (Hc, Wc, 8)

        # Normalize
        frame -= _meteo_means      # (8,) broadcasts over (Hc, Wc)
        frame /= _meteo_stds
        np.clip(frame, -10.0, 10.0, out=frame)

        # Patchify: (Hc, Wc, C) -> (n_patches, P*P*C)
        frame_r = frame.reshape(nph, P, npw, P, C)
        frame_r = frame_r.transpose(0, 2, 1, 3, 4)    # (nph, npw, P, P, C)
        patches = frame_r.reshape(n_patches, P * P * C)  # (n_patches, 2048)

        result[:, li, :] = patches.astype(np.float16)

    # Write directly to the cache file at the correct byte offset.
    # Using raw file I/O avoids mapping the full 5.27 TB virtual address space
    # and eliminates the large-array IPC transfer that caused OOM.
    try:
        bytes_per_date = n_patches * S2S_N_LEADS * feat_dim * 2  # float16 = 2 bytes
        offset = date_idx * bytes_per_date
        with open(out_file, 'r+b') as f:
            f.seek(offset)
            f.write(result.tobytes())
    except Exception as e:
        print(f"  [ERROR] Failed to write {date_str} (idx={date_idx}): {e}", flush=True)
        return date_str, False

    return date_str, True


def _load_weather_for_lead(date_dir, lead, Hc, Wc):
    """
    Load weather arrays for a single lead day.

    Returns (temp_c, dewp_c, wind_kmh, rain_mm) each (Hc, Wc) float64,
    or None if core TIF does not exist.
    """
    core_path = os.path.join(date_dir, f"lead{lead:02d}.tif")
    if not os.path.exists(core_path):
        return None

    try:
        with rasterio.open(core_path) as src:
            core = src.read().astype(np.float64)  # (6, H, W): 2t, 2d, tcw, sm20, st20, VPD
    except Exception as e:
        print(f"  [WARN] Failed to read {core_path}: {e}", flush=True)
        return None

    temp_c = core[0, :Hc, :Wc]   # 2t in deg C
    dewp_c = core[1, :Hc, :Wc]   # 2d in deg C

    # Try ext TIF for wind and precip
    ext_path = os.path.join(date_dir, f"lead{lead:02d}_ext.tif")
    if os.path.exists(ext_path):
        try:
            with rasterio.open(ext_path) as src:
                ext = src.read().astype(np.float64)  # (3, H, W): 10u, 10v, tp
            u10 = ext[0, :Hc, :Wc]
            v10 = ext[1, :Hc, :Wc]
            tp_m = ext[2, :Hc, :Wc]
            wind_kmh = wind_components_to_speed(u10, v10) * 3.6  # m/s -> km/h
            rain_mm = np.maximum(tp_m * 1000.0, 0.0)             # meters -> mm
        except Exception as e:
            print(f"  [WARN] Failed to read {ext_path}: {e}", flush=True)
            wind_kmh = np.full((Hc, Wc), DEFAULT_WIND_KMH, dtype=np.float64)
            rain_mm = np.full((Hc, Wc), DEFAULT_RAIN_MM, dtype=np.float64)
    else:
        wind_kmh = np.full((Hc, Wc), DEFAULT_WIND_KMH, dtype=np.float64)
        rain_mm = np.full((Hc, Wc), DEFAULT_RAIN_MM, dtype=np.float64)

    return temp_c, dewp_c, wind_kmh, rain_mm


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------
def build_cache(s2s_dir, out_file, reference_tif, fire_clim_path,
                ffmc_dir, dmc_dir, dc_dir, norm_stats_path,
                patch_size=16, workers=4, delete_after=False,
                start_year=None, end_year=None, fire_season_only=False):
    """Build the (n_dates, n_patches, 32, 2048) float16 full-patch cache."""

    # Discover date directories
    date_dirs = _parse_date_dirs(s2s_dir)

    # Apply date filters
    if start_year or end_year or fire_season_only:
        n_before = len(date_dirs)
        filtered = []
        for d in date_dirs:
            year = int(d[:4])
            month = int(d[5:7])
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue
            if fire_season_only and month not in (5, 6, 7, 8, 9, 10):
                continue
            filtered.append(d)
        date_dirs = filtered
        print(f"Date filter: {n_before} → {len(date_dirs)} dates "
              f"(years={start_year}-{end_year}, fire_season={fire_season_only})",
              flush=True)
    if not date_dirs:
        raise RuntimeError(f"No YYYY-MM-DD subdirectories found in {s2s_dir}")
    n_dates = len(date_dirs)
    print(f"Found {n_dates} date directories: {date_dirs[0]} .. {date_dirs[-1]}",
          flush=True)

    # Read grid dimensions from reference TIF
    with rasterio.open(reference_tif) as src:
        H, W = src.height, src.width
    Hc = H - H % patch_size
    Wc = W - W % patch_size
    nph = Hc // patch_size
    npw = Wc // patch_size
    n_patches = nph * npw
    feat_dim = patch_size * patch_size * N_CHANNELS
    print(f"Grid: H={H} W={W}  crop to {Hc}x{Wc}  "
          f"patches: {nph}x{npw}={n_patches}  patch_size={patch_size}",
          flush=True)
    print(f"Feature dim per lead: P^2*C = {patch_size}^2*{N_CHANNELS} = {feat_dim}",
          flush=True)

    # Estimate cache size
    cache_bytes = n_dates * n_patches * S2S_N_LEADS * feat_dim * 2  # float16
    print(f"Cache size: {cache_bytes / 1e9:.2f} GB  "
          f"shape=({n_dates}, {n_patches}, {S2S_N_LEADS}, {feat_dim}) float16",
          flush=True)

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    done_set = set()
    expected_bytes = n_dates * n_patches * S2S_N_LEADS * feat_dim * 2

    if os.path.isfile(out_file):
        actual_bytes = os.path.getsize(out_file)
        if actual_bytes == expected_bytes:
            print(f"Existing cache found ({actual_bytes / 1e9:.2f} GB). "
                  f"Opening in r+ mode for resume...", flush=True)
            cache = np.memmap(out_file, dtype='float16', mode='r+',
                              shape=(n_dates, n_patches, S2S_N_LEADS, feat_dim))
            # Detect already-processed dates by checking first patch, first lead
            for i, d in enumerate(date_dirs):
                sample = cache[i, 0, :, :16]  # small slice
                if np.any(sample != 0):
                    done_set.add(d)
            print(f"  Already processed: {len(done_set)}/{n_dates} dates",
                  flush=True)
        else:
            print(f"Existing cache has wrong size ({actual_bytes} vs "
                  f"{expected_bytes}). Recreating...", flush=True)
            cache = np.memmap(out_file, dtype='float16', mode='w+',
                              shape=(n_dates, n_patches, S2S_N_LEADS, feat_dim))
    else:
        cache = np.memmap(out_file, dtype='float16', mode='w+',
                          shape=(n_dates, n_patches, S2S_N_LEADS, feat_dim))

    # ------------------------------------------------------------------
    # Build work items — now include out_file + date_idx for direct writes
    # ------------------------------------------------------------------
    date_to_idx = {d: i for i, d in enumerate(date_dirs)}
    work_items = [
        (date_str, s2s_dir,
         patch_size, n_patches, nph, npw, H, W, Hc, Wc,
         out_file, date_to_idx[date_str], n_dates)
        for date_str in date_dirs
        if date_str not in done_set
    ]
    n_todo = len(work_items)

    if n_todo == 0:
        print(f"\nAll {n_dates} dates already processed. Nothing to do.",
              flush=True)
    else:
        print(f"\nProcessing {n_todo}/{n_dates} dates with {workers} workers "
              f"(skipping {len(done_set)} already done)...", flush=True)
        t0 = time.time()

        # Close the main-process cache mapping BEFORE forking workers.
        # This prevents workers from inheriting the 5.27 TB virtual mapping,
        # which would bloat page tables and cause OOM.
        cache.flush()
        del cache
        cache = None

        if workers <= 1:
            # Single-process mode — reopen cache for direct writes
            cache = np.memmap(out_file, dtype='float16', mode='r+',
                              shape=(n_dates, n_patches, S2S_N_LEADS, feat_dim))
            for n_done, item in enumerate(work_items):
                date_str, ok = _process_one_date(item)
                if delete_after and ok:
                    _delete_date_dir(s2s_dir, date_str)
                if (n_done + 1) % 10 == 0 or n_done + 1 == n_todo:
                    elapsed = time.time() - t0
                    eta_min = elapsed / (n_done + 1) * (n_todo - n_done - 1) / 60
                    print(f"  {n_done+1}/{n_todo}  {date_str}  "
                          f"({elapsed:.0f}s  ~{eta_min:.0f} min left)",
                          flush=True)
            cache.flush()
            del cache
        else:
            # Multi-process mode.
            # Workers write directly to the file via raw I/O — no large arrays
            # are returned through IPC, eliminating the OOM root cause.
            with Pool(
                processes=workers,
                initializer=_worker_init,
                initargs=(fire_clim_path, norm_stats_path, Hc, Wc,
                          ffmc_dir, dmc_dir, dc_dir),
            ) as pool:
                n_done = 0
                for date_str, ok in pool.imap_unordered(
                        _process_one_date, work_items, chunksize=1):
                    if delete_after and ok:
                        _delete_date_dir(s2s_dir, date_str)
                    n_done += 1
                    if n_done % 10 == 0 or n_done == n_todo:
                        elapsed = time.time() - t0
                        eta_min = elapsed / n_done * (n_todo - n_done) / 60
                        print(f"  {n_done}/{n_todo}  last={date_str}  "
                              f"({elapsed:.0f}s  ~{eta_min:.0f} min left)",
                              flush=True)
    elapsed_total = time.time() - t0 if n_todo > 0 else 0
    print(f"\nFlushed cache to {out_file}  "
          f"({os.path.getsize(out_file) / 1e9:.2f} GB)  "
          f"total={elapsed_total:.0f}s", flush=True)

    # Save companion dates array
    dates_file = out_file + ".dates.npy"
    np.save(dates_file, np.array(date_dirs, dtype=str))
    print(f"Saved dates array to {dates_file}  ({n_dates} entries)", flush=True)

    print("\nDone.", flush=True)


def _delete_date_dir(s2s_dir, date_str):
    """Remove the processed date directory to reclaim disk space."""
    date_dir = os.path.join(s2s_dir, date_str)
    try:
        shutil.rmtree(date_dir)
    except Exception as e:
        print(f"  [WARN] Could not delete {date_dir}: {e}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build S2S full-patch decoder cache (Oracle-format, 8ch)."
    )
    ap.add_argument("--s2s-dir", required=True,
                    help="Root directory containing YYYY-MM-DD subdirs "
                         "with lead{kk:02d}.tif and optional lead{kk:02d}_ext.tif")
    ap.add_argument("--out-file", required=True,
                    help="Output .dat memmap file path.")
    ap.add_argument("--reference", required=True,
                    help="Reference GeoTIFF for grid H/W.")
    ap.add_argument("--fire-clim", required=True,
                    help="Path to fire_climatology.tif (single band).")
    ap.add_argument("--ffmc-dir", required=True,
                    help="Directory with observed FFMC TIFs (ffmc_YYYYMMDD.tif).")
    ap.add_argument("--dmc-dir", required=True,
                    help="Directory with observed DMC TIFs (dmc_YYYYMMDD.tif).")
    ap.add_argument("--dc-dir", required=True,
                    help="Directory with observed DC TIFs (dc_YYYYMMDD.tif).")
    ap.add_argument("--norm-stats", required=True,
                    help="Path to norm_stats.npy with shape (2, 8): "
                         "[meteo_means, meteo_stds].")
    ap.add_argument("--patch-size", type=int, default=16,
                    help="Patch size in pixels (default: 16).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel worker processes (default: 4).")
    ap.add_argument("--delete-after", action="store_true",
                    help="Delete s2s_processed/{date}/ dirs after caching.")
    ap.add_argument("--start-year", type=int, default=None,
                    help="Only process dates >= this year (e.g. 2018).")
    ap.add_argument("--end-year", type=int, default=None,
                    help="Only process dates <= this year (e.g. 2025).")
    ap.add_argument("--fire-season-only", action="store_true",
                    help="Only process May-October dates (fire season).")
    args = ap.parse_args()

    # In single-process mode, initialize globals directly
    if args.workers <= 1:
        with rasterio.open(args.reference) as src:
            H, W = src.height, src.width
        Hc = H - H % args.patch_size
        Wc = W - W % args.patch_size
        _worker_init(args.fire_clim, args.norm_stats, Hc, Wc,
                     args.ffmc_dir, args.dmc_dir, args.dc_dir)

    build_cache(
        s2s_dir=args.s2s_dir,
        out_file=args.out_file,
        reference_tif=args.reference,
        fire_clim_path=args.fire_clim,
        ffmc_dir=args.ffmc_dir,
        dmc_dir=args.dmc_dir,
        dc_dir=args.dc_dir,
        norm_stats_path=args.norm_stats,
        patch_size=args.patch_size,
        workers=args.workers,
        delete_after=args.delete_after,
        start_year=args.start_year,
        end_year=args.end_year,
        fire_season_only=args.fire_season_only,
    )


if __name__ == "__main__":
    main()
