"""
Consolidate Encoder Source Data into Single Contiguous Array
=============================================================
Merges thousands of per-channel daily TIF files into a single memory-mapped
float32 array: (T, H, W, N_channels). After consolidation, cache building
in train_v3.py can read frames by pure index instead of 27000+ rasterio.open()
calls, reducing cache build time from ~16h to ~1-2h.

Output: encoder_consolidated_{N}ch.dat  (T × H × W × N × 4 bytes)
  - 16ch, T=2791, H=2281, W=2709: ~1.1 TB
  - 13ch: ~0.9 TB
  - 9ch: ~0.6 TB

Special handling:
  - FWI, 2t, 2d, tcw, sm20, st20, deep_soil, u10, v10, CAPE: daily TIFs
  - fire_clim: annual rolling (year Y uses fire_clim_upto_Y.tif)
  - NDVI: 16-day composites → linear interpolation to daily
  - precip_def: daily precip → rolling 30-day deficit (negative sum)
  - burn_age: annual burn_scars, year-1 to avoid temporal leakage, nodata→0
  - burn_count: annual, year-1, log1p encoded
  - population, slope: static (same for all T)

Companion: encoder_consolidated_{N}ch.dates.npy — sorted date strings

Usage:
    python -m src.data_ops.processing.consolidate_encoder_data \\
        --config configs/paths_narval.yaml \\
        --channels "FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count" \\
        --data_start 2018-05-01 --data_end 2025-12-31 \\
        --out-file /scratch/jiaqi217/meteo_cache/encoder_consolidated_16ch.dat
"""

import argparse
import glob
import os
import re
import sys
import time
from bisect import bisect_right
from collections import deque
from datetime import date, timedelta

import numpy as np
import rasterio


# ---------------------------------------------------------------------------
# Channel definitions (must match train_v3.py V3_CHANNEL_DEFS)
# ---------------------------------------------------------------------------
CHANNEL_ORDER = [
    "FWI", "2t", "fire_clim", "2d", "tcw", "sm20", "st20",
    "deep_soil", "precip_def", "u10", "v10", "CAPE", "lightning",
    "NDVI", "population", "slope", "burn_age", "burn_count",
]


def _extract_date(fname):
    """Extract date from filename like fwi_20210715.tif or 2t_20210715.tif."""
    m = re.search(r'(\d{4})(\d{2})(\d{2})', fname)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _build_date_dict(directory, prefix=None):
    """Build {date: filepath} dict from TIFs in directory."""
    result = {}
    if not directory or not os.path.isdir(directory):
        return result
    for p in sorted(glob.glob(os.path.join(directory, "*.tif"))):
        d = _extract_date(os.path.basename(p))
        if d:
            result[d] = p
    return result


def _read_tif(path, H, W, fill=0.0):
    """Read single-band TIF, mask nodata, return (H, W) float32."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[~np.isfinite(arr)] = fill
        return arr[:H, :W]
    except Exception:
        return np.full((H, W), fill, dtype=np.float32)


def _build_ndvi_index(ndvi_dir):
    """Build sorted [(date, path)] for NDVI composites."""
    result = []
    if not ndvi_dir or not os.path.isdir(ndvi_dir):
        return result
    for p in sorted(glob.glob(os.path.join(ndvi_dir, "ndvi_*.tif"))):
        d = _extract_date(os.path.basename(p))
        if d:
            result.append((d, p))
    return result


def _interpolate_ndvi(target_date, ndvi_index, cache, H, W):
    """Linearly interpolate NDVI for target_date."""
    if not ndvi_index:
        return np.zeros((H, W), dtype=np.float32)
    dates = [d for d, _ in ndvi_index]
    idx = bisect_right(dates, target_date)

    def _load(i):
        d, p = ndvi_index[i]
        if d not in cache:
            cache[d] = _read_tif(p, H, W, fill=0.0)
        return cache[d]

    if idx == 0:
        return _load(0).copy()
    if idx >= len(dates):
        return _load(len(dates) - 1).copy()
    d0, d1 = dates[idx - 1], dates[idx]
    gap = (d1 - d0).days
    if gap <= 0 or gap > 64:
        return _load(idx - 1).copy()
    w = (target_date - d0).days / gap
    return ((1 - w) * _load(idx - 1) + w * _load(idx)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Consolidate encoder TIFs → single memmap")
    ap.add_argument("--config", required=True, help="YAML config file for paths")
    ap.add_argument("--channels", required=True, help="Comma-separated channel names")
    ap.add_argument("--data_start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--data_end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--out-file", required=True, help="Output memmap .dat path")
    ap.add_argument("--precip_deficit_days", type=int, default=30)
    args = ap.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", cfg)

    ch_names = [c.strip() for c in args.channels.split(",")]
    N_CH = len(ch_names)
    start_date = date.fromisoformat(args.data_start)
    end_date = date.fromisoformat(args.data_end)

    # Reference grid
    fwi_dir = paths["fwi_dir"]
    ref_tif = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))[0]
    with rasterio.open(ref_tif) as src:
        H, W = src.height, src.width
    print(f"Grid: H={H} W={W}  Channels={N_CH}  ({', '.join(ch_names)})")

    # Build date dicts for daily channels
    fwi_dict = _build_date_dict(fwi_dir)
    obs_root = paths.get("observation_dir", paths.get("ecmwf_dir", ""))
    t2m_dict = _build_date_dict(os.path.join(obs_root, "2t"))
    if not t2m_dict:
        t2m_dict = _build_date_dict(obs_root)
    dew_dict = _build_date_dict(paths.get("dew_dir") or os.path.join(obs_root, "2d"))
    tcw_dict = _build_date_dict(paths.get("tcw_dir") or os.path.join(obs_root, "tcw"))
    sm20_dict = _build_date_dict(paths.get("sm20_dir") or os.path.join(obs_root, "sm20"))
    st20_dict = _build_date_dict(paths.get("st20_dir") or os.path.join(obs_root, "st20"))
    deep_soil_dict = _build_date_dict(paths.get("deep_soil_dir", "data/era5_deep_soil"))
    precip_dict = _build_date_dict(paths.get("precip_dir", "data/era5_precip"))
    u10_dict = _build_date_dict(paths.get("wind_u_dir", "data/era5_u10"))
    v10_dict = _build_date_dict(paths.get("wind_v_dir", "data/era5_v10"))
    cape_dict = _build_date_dict(paths.get("cape_dir", "data/era5_cape"))
    lightning_dict = _build_date_dict(paths.get("lightning_dir", "data/lightning"))

    # Align dates: require FWI + 2t
    all_dates = sorted(set(fwi_dict.keys()) & set(t2m_dict.keys()))
    all_dates = [d for d in all_dates if start_date <= d <= end_date]
    T = len(all_dates)
    print(f"Aligned dates: {T}  ({all_dates[0]} → {all_dates[-1]})")

    # Static channels
    population_arr = _read_tif(
        paths.get("population_tif", "data/population_density.tif"), H, W)
    terrain_dir = paths.get("terrain_dir", "data/terrain")
    slope_path = os.path.join(terrain_dir, "slope.tif") if terrain_dir else None
    slope_arr = _read_tif(slope_path, H, W) if slope_path and os.path.exists(slope_path) \
                else np.zeros((H, W), dtype=np.float32)

    # Annual channels
    fire_clim_dir = paths.get("fire_clim_dir", "data/fire_clim_annual")
    fire_clim_arrays = {}
    if fire_clim_dir and os.path.isdir(fire_clim_dir):
        for p in sorted(glob.glob(os.path.join(fire_clim_dir, "fire_clim_upto_*.tif"))):
            m = re.search(r'(\d{4})', os.path.basename(p))
            if m:
                yr = int(m.group(1))
                arr = _read_tif(p, H, W)
                fire_clim_arrays[yr] = np.log1p(np.maximum(arr, 0))

    burn_scars_dir = paths.get("burn_scars_dir", "data/burn_scars")
    burn_scar_raw = {}
    burn_count_arrays = {}
    if burn_scars_dir and os.path.isdir(burn_scars_dir):
        for p in sorted(glob.glob(os.path.join(burn_scars_dir, "years_since_burn_*.tif"))):
            m = re.search(r'(\d{4})', os.path.basename(p))
            if m:
                yr = int(m.group(1))
                arr = _read_tif(p, H, W)  # nodata already masked by _read_tif
                burn_scar_raw[yr] = np.maximum(arr, 0)
        for p in sorted(glob.glob(os.path.join(burn_scars_dir, "burn_count_*.tif"))):
            m = re.search(r'(\d{4})', os.path.basename(p))
            if m:
                yr = int(m.group(1))
                arr = _read_tif(p, H, W)
                burn_count_arrays[yr] = np.maximum(arr, 0)

    # NDVI
    ndvi_dir = paths.get("ndvi_dir", "data/ndvi_data")
    ndvi_index = _build_ndvi_index(ndvi_dir)
    ndvi_cache = {}

    # Create output memmap
    out_shape = (T, H, W, N_CH)
    out_bytes = T * H * W * N_CH * 4
    print(f"\nOutput: {args.out_file}")
    print(f"  shape={out_shape}  size={out_bytes / 1e9:.1f} GB")

    out = np.memmap(args.out_file, dtype=np.float32, mode='w+', shape=out_shape)

    # Precip accumulator
    precip_deque = deque(maxlen=args.precip_deficit_days)

    t0 = time.time()
    for t_idx, cur_date in enumerate(all_dates):
        for ch_idx, ch_name in enumerate(ch_names):
            if ch_name == "FWI":
                out[t_idx, :, :, ch_idx] = _read_tif(fwi_dict.get(cur_date, ""), H, W)
            elif ch_name == "2t":
                out[t_idx, :, :, ch_idx] = _read_tif(t2m_dict.get(cur_date, ""), H, W)
            elif ch_name == "2d":
                p = dew_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "tcw":
                p = tcw_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "sm20":
                p = sm20_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "st20":
                p = st20_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "deep_soil":
                p = deep_soil_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "u10":
                p = u10_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "v10":
                p = v10_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "CAPE":
                p = cape_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "lightning":
                p = lightning_dict.get(cur_date)
                if p:
                    out[t_idx, :, :, ch_idx] = _read_tif(p, H, W)
            elif ch_name == "fire_clim":
                yr = cur_date.year
                if yr in fire_clim_arrays:
                    out[t_idx, :, :, ch_idx] = fire_clim_arrays[yr]
                elif fire_clim_arrays:
                    nearest = min(fire_clim_arrays.keys(), key=lambda y: abs(y - yr))
                    out[t_idx, :, :, ch_idx] = fire_clim_arrays[nearest]
            elif ch_name == "NDVI":
                out[t_idx, :, :, ch_idx] = _interpolate_ndvi(
                    cur_date, ndvi_index, ndvi_cache, H, W)
            elif ch_name == "population":
                out[t_idx, :, :, ch_idx] = population_arr
            elif ch_name == "slope":
                out[t_idx, :, :, ch_idx] = slope_arr
            elif ch_name == "precip_def":
                p = precip_dict.get(cur_date)
                if p:
                    precip_deque.append(_read_tif(p, H, W) * 1000.0)  # m→mm
                if len(precip_deque) > 0:
                    out[t_idx, :, :, ch_idx] = -np.sum(precip_deque, axis=0)
            elif ch_name == "burn_age":
                prev_year = cur_date.year - 1
                raw = None
                if prev_year in burn_scar_raw:
                    raw = burn_scar_raw[prev_year]
                elif burn_scar_raw:
                    valid = [y for y in burn_scar_raw if y <= prev_year]
                    if valid:
                        raw = burn_scar_raw[max(valid)]
                if raw is not None:
                    out[t_idx, :, :, ch_idx] = np.log1p(raw)
            elif ch_name == "burn_count":
                prev_year = cur_date.year - 1
                if prev_year in burn_count_arrays:
                    out[t_idx, :, :, ch_idx] = np.log1p(burn_count_arrays[prev_year])
                elif burn_count_arrays:
                    valid = [y for y in burn_count_arrays if y <= prev_year]
                    if valid:
                        out[t_idx, :, :, ch_idx] = np.log1p(
                            burn_count_arrays[max(valid)])

        if (t_idx + 1) % 100 == 0 or t_idx == 0:
            elapsed = time.time() - t0
            rate = (t_idx + 1) / elapsed
            eta = (T - t_idx - 1) / rate
            print(f"  day {t_idx+1:5d}/{T}  ({elapsed:.0f}s elapsed, "
                  f"~{eta/60:.0f}m left)", flush=True)

    out.flush()
    elapsed = time.time() - t0
    print(f"\nDone: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output: {args.out_file}  ({os.path.getsize(args.out_file) / 1e9:.1f} GB)")

    # Save dates companion
    dates_path = args.out_file + ".dates.npy"
    np.save(dates_path, np.array([d.isoformat() for d in all_dates]))
    print(f"Dates: {dates_path}  ({T} entries)")


if __name__ == "__main__":
    main()
