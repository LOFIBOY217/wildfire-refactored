"""
Training-input data pipeline: channel definitions + raster loaders.
================================================================================
Primitives shared by train_v3's cache-building path (and, in future, eval /
forecast). Extracted verbatim from train_v3.py so the pipeline lives in one
place and train_v3's main() reads as orchestration.

  - V3_CHANNEL_DEFS / DECODER_CTX_CHANNELS : channel metadata registries
  - _assert_channel_quality                : sentinel/nodata sanity check
  - _load_static_channel                   : single-band GeoTIFF loader
  - _build_ndvi_index / _interpolate_ndvi  : NDVI 16-day composite interpolation

No logic changed; see train_v3.build_training_inputs() for the consumer.
"""

import os
import glob
from bisect import bisect_right

import numpy as np
import rasterio

from src.utils.date_utils import extract_date_from_filename
from src.training.train_s2s_hotspot_cwfis_v2 import _read_tif_safe
import gc
import time
from collections import deque
from datetime import datetime as dt, timedelta
from dataclasses import dataclass

from scipy.ndimage import binary_dilation

from src.config import get_path, load_config
from src.data_ops.processing.rasterize_hotspots import (
    load_hotspot_data, rasterize_hotspots_batch)
from src.training.train_s2s_hotspot_cwfis_v2 import (
    S2S_DEC_DIM, _build_file_dict, _build_flat_file_dict, _patchify_frame,
    _stream_channel_stats, _transpose_tf_to_pf)

# All available channels with metadata
V3_CHANNEL_DEFS = {
    "FWI":        {"type": "daily",   "required": True},
    "2t":         {"type": "daily",   "required": True},
    "2t_anom":    {"type": "daily",   "required": False},
    "fire_clim":  {"type": "annual",  "required": True},
    "2d":         {"type": "daily",   "required": False},
    "tcw":        {"type": "daily",   "required": False},
    "sm20":       {"type": "daily",   "required": False},
    "st20":       {"type": "daily",   "required": False},
    "lightning":  {"type": "daily",   "required": False},
    "NDVI":       {"type": "interp",  "required": False},
    "population": {"type": "static",  "required": False},
    "deep_soil":  {"type": "daily",   "required": False},
    "precip_def": {"type": "computed","required": False},
    "slope":      {"type": "static",  "required": False},
    "elevation":  {"type": "static",  "required": False},
    "aspect":     {"type": "static",  "required": False},
    "lightning_climatology": {"type": "static", "required": False},
    "burn_age":   {"type": "annual",  "required": False},
    "burn_count": {"type": "annual",  "required": False},
    "dist_recent_burn": {"type": "annual",  "required": False},
    "u10":        {"type": "daily",   "required": False},
    "v10":        {"type": "daily",   "required": False},
    "CAPE":       {"type": "daily",   "required": False},
    # V2-compatible FWI sub-components (for fair comparison experiments)
    "FFMC":       {"type": "daily",   "required": False},
    "DMC":        {"type": "daily",   "required": False},
    "DC":         {"type": "daily",   "required": False},
    "BUI":        {"type": "daily",   "required": False},
    "ISI":        {"type": "daily",   "required": False},
}


# Static channels to inject into decoder context (spatial info the decoder needs)
DECODER_CTX_CHANNELS = {"fire_clim", "population", "slope", "burn_age", "burn_count", "dist_recent_burn"}


def _assert_channel_quality(arr, name, max_frac_same_value=0.999, warn_only=True):
    """Sanity check: flag channels where >99.9% of pixels have the same value
    (typical symptom of sentinel/nodata not masked).

    Default warn-only to avoid false positives on legitimately sparse data
    (e.g. population is mostly 0). Set warn_only=False for hard fail.
    """
    total = arr.size
    if total == 0:
        return
    # Fast check: count nonzero. If almost all zero, that's fine (sparse data).
    # The dangerous case is a specific non-zero value dominating.
    vals, counts = np.unique(arr, return_counts=True)
    top_val = vals[counts.argmax()]
    top_frac = counts.max() / total
    if top_val != 0 and top_frac > max_frac_same_value:
        msg = (f"  [QUALITY] {name}: {top_frac:.1%} of pixels have value "
               f"{top_val:.3f} (likely sentinel not masked!)")
        if warn_only:
            print(f"  WARN: {msg}")
        else:
            raise ValueError(msg)


def _load_static_channel(tif_path, expected_h, expected_w, name="static"):
    """Load a static single-band GeoTIFF. Returns (H, W) float32 (zeros on failure).

    Reads the TIF's nodata value from metadata and replaces nodata pixels with 0.
    This is critical for burn_scars (nodata=9999 means 'never burned').
    """
    if tif_path is None or not os.path.exists(tif_path):
        print(f"  [WARN] {name}: file not found: {tif_path} — using zeros")
        return np.zeros((expected_h, expected_w), dtype=np.float32)
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if arr.shape != (expected_h, expected_w):
        print(f"  [WARN] {name}: shape {arr.shape} != ({expected_h},{expected_w}) — using zeros")
        return np.zeros((expected_h, expected_w), dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    # Mask nodata values (e.g. burn_scars use 9999 for 'never burned')
    if nodata is not None:
        arr[arr == nodata] = 0.0
    nonzero = int((arr > 0).sum())
    print(f"  {name}: {arr.shape}  nonzero={nonzero:,}  "
          f"max={arr.max():.3f}  mean(nz)={arr[arr>0].mean():.3f}" if nonzero else
          f"  {name}: {arr.shape}  ALL ZERO")
    # Data quality check
    _assert_channel_quality(arr, name, warn_only=True)
    return arr


def _build_ndvi_index(ndvi_dir):
    """Build sorted list of (date, path) for NDVI composites."""
    result = []
    for p in sorted(glob.glob(os.path.join(ndvi_dir, "ndvi_*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            result.append((d, p))
    return result


def _interpolate_ndvi(target_date, ndvi_index, ndvi_cache, H, W):
    """Linearly interpolate NDVI for target_date from 16-day composites.
    Returns (H, W) float32. Falls back to nearest if gap > 32 days."""
    if not ndvi_index:
        return np.zeros((H, W), dtype=np.float32)

    dates = [d for d, p in ndvi_index]
    idx = bisect_right(dates, target_date)

    # Load helper with caching
    def _load(i):
        d, p = ndvi_index[i]
        if d not in ndvi_cache:
            ndvi_cache[d] = _read_tif_safe(p, None)
            ndvi_cache[d] = np.nan_to_num(ndvi_cache[d], nan=0.0)
        return ndvi_cache[d]

    if idx == 0:
        return _load(0)
    if idx >= len(dates):
        return _load(len(dates) - 1)

    d_before = dates[idx - 1]
    d_after = dates[idx]
    gap = (d_after - d_before).days
    if gap <= 0 or gap > 32:
        # Too large gap — use nearest
        if (target_date - d_before).days <= (d_after - target_date).days:
            return _load(idx - 1)
        return _load(idx)

    w = (target_date - d_before).days / gap
    before = _load(idx - 1)
    after = _load(idx)
    return ((1 - w) * before + w * after).astype(np.float32)


@dataclass
class TrainingInputs:
    """Bundle of packaged training inputs produced by build_training_inputs().

    All fields are values train_v3.main() previously computed inline in
    STEP 1-6 and consumes from STEP 7 onward. See build_training_inputs.
    """
    meteo_patched: object
    meteo_means: object
    meteo_stds: object
    static_arrays: object
    fire_stack: object
    aligned_dates: object
    T: object
    P: object
    Hc: object
    Wc: object
    hw: object
    grid: object
    n_patches: object
    enc_dim: object
    dec_dim: object
    dec_dim_base: object
    out_dim: object
    ctx_extra_dim: object
    fusion_tag: object
    master_info: object
    meteo_mmap_gb: object


def build_training_inputs(
        args,
        CHANNEL_NAMES,
        N_CHANNELS,
        burn_scars_dir,
        ckpt_dir,
        data_start_date,
        deep_soil_dir,
        fire_clim_dir,
        fire_clim_path,
        fwi_dir,
        hotspot_csv,
        in_days,
        lead_end,
        lightning_dir,
        ndvi_dir,
        obs_root,
        paths_cfg,
        population_tif,
        precip_dir,
        pred_end_date,
        pred_start_date,
        tele_K,
        terrain_dir):
    """Build the packaged training inputs (STEP 1-6): file indices, date
    alignment, grid/stats, fire labels, and the patchified float16 meteo
    memmap. Extracted verbatim from train_v3.main(); returns TrainingInputs.
    """
    # STEP 1  Build file indices for all active channels
    # ----------------------------------------------------------------
    print(f"\n[STEP 1] Building file index ({N_CHANNELS} channels)...")

    # Required daily channels
    fwi_dict = {}
    for p in sorted(glob.glob(os.path.join(fwi_dir, "*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            fwi_dict[d] = p
    t2m_dict = _build_file_dict(obs_root, "2t")
    t2m_anom_dict = _build_flat_file_dict("data/2t_anom", "2t_anom") if "2t_anom" in CHANNEL_NAMES else {}

    # ERA5 observation channels (same directory structure as 2t)
    dew_dict = _build_file_dict(obs_root, "2d") if "2d" in CHANNEL_NAMES else {}
    tcw_dict = _build_file_dict(obs_root, "tcw") if "tcw" in CHANNEL_NAMES else {}
    sm20_dict = _build_file_dict(obs_root, "sm20") if "sm20" in CHANNEL_NAMES else {}
    st20_dict = _build_file_dict(obs_root, "st20") if "st20" in CHANNEL_NAMES else {}

    if not fwi_dict:
        raise RuntimeError(f"No FWI .tif files found in {fwi_dir}")
    if not t2m_dict:
        raise RuntimeError(f"No 2t .tif files found under {obs_root}")

    # Optional daily channels
    lightning_dict = _build_flat_file_dict(lightning_dir, "lightning") if "lightning" in CHANNEL_NAMES else {}
    deep_soil_dict = _build_flat_file_dict(deep_soil_dir, "swvl2") if "deep_soil" in CHANNEL_NAMES else {}
    if not deep_soil_dict and "deep_soil" in CHANNEL_NAMES:
        deep_soil_dict = _build_flat_file_dict(deep_soil_dir, "deep_soil")
    precip_dict = _build_flat_file_dict(precip_dir, "tp") if "precip_def" in CHANNEL_NAMES else {}

    # FWI sub-components (V2-compatible, for fair comparison experiments)
    ffmc_dir = paths_cfg.get("ffmc_dir", "data/ffmc_data")
    dmc_dir_ch = paths_cfg.get("dmc_dir", "data/dmc_data")
    dc_dir_ch = paths_cfg.get("dc_dir", "data/dc_data")
    bui_dir_ch = paths_cfg.get("bui_dir", "data/bui_data")
    isi_dir_ch = paths_cfg.get("isi_dir", "data/isi_data")
    ffmc_dict = _build_flat_file_dict(ffmc_dir, "ffmc") if "FFMC" in CHANNEL_NAMES else {}
    dmc_dict = _build_flat_file_dict(dmc_dir_ch, "dmc") if "DMC" in CHANNEL_NAMES else {}
    dc_dict = _build_flat_file_dict(dc_dir_ch, "dc") if "DC" in CHANNEL_NAMES else {}
    bui_dict = _build_flat_file_dict(bui_dir_ch, "bui") if "BUI" in CHANNEL_NAMES else {}
    isi_dict = _build_flat_file_dict(isi_dir_ch, "isi") if "ISI" in CHANNEL_NAMES else {}

    # Wind and CAPE (from ERA5 extraction — era5_to_daily.py output)
    u10_dir = paths_cfg.get("wind_u_dir", "data/era5_u10")
    v10_dir = paths_cfg.get("wind_v_dir", "data/era5_v10")
    cape_dir = paths_cfg.get("cape_dir", "data/era5_cape")
    u10_dict = _build_flat_file_dict(u10_dir, "u10") if "u10" in CHANNEL_NAMES else {}
    v10_dict = _build_flat_file_dict(v10_dir, "v10") if "v10" in CHANNEL_NAMES else {}
    cape_dict = _build_flat_file_dict(cape_dir, "cape") if "CAPE" in CHANNEL_NAMES else {}

    # NDVI index
    ndvi_index = _build_ndvi_index(ndvi_dir) if "NDVI" in CHANNEL_NAMES else []

    # Burn scars index (annual): years_since_burn + burn_count
    burn_scar_dict = {}
    if "burn_age" in CHANNEL_NAMES:
        for p in sorted(glob.glob(os.path.join(burn_scars_dir, "years_since_burn_*.tif"))):
            bn = os.path.basename(p)
            try:
                year = int(bn.replace("years_since_burn_", "").replace(".tif", ""))
                burn_scar_dict[year] = p
            except ValueError:
                pass

    burn_count_dict = {}
    if "burn_count" in CHANNEL_NAMES:
        for p in sorted(glob.glob(os.path.join(burn_scars_dir, "burn_count_*.tif"))):
            bn = os.path.basename(p)
            try:
                year = int(bn.replace("burn_count_", "").replace(".tif", ""))
                burn_count_dict[year] = p
            except ValueError:
                pass

    dist_recent_dict = {}
    if "dist_recent_burn" in CHANNEL_NAMES:
        for p in sorted(glob.glob(os.path.join("data/dist_recent_burn", "dist_recent_burn_*.tif"))):
            bn = os.path.basename(p)
            try:
                year = int(bn.replace("dist_recent_burn_", "").replace(".tif", ""))
                dist_recent_dict[year] = p
            except ValueError:
                pass

    # Annual fire climatology: fire_clim_upto_YEAR.tif (one per target year)
    fire_clim_annual_dict = {}  # year → path
    if "fire_clim" in CHANNEL_NAMES and fire_clim_dir:
        for p in sorted(glob.glob(os.path.join(fire_clim_dir, "fire_clim_upto_*.tif"))):
            bn = os.path.basename(p)
            try:
                year = int(bn.replace("fire_clim_upto_", "").replace(".tif", ""))
                fire_clim_annual_dict[year] = p
            except ValueError:
                pass
        if fire_clim_annual_dict:
            print(f"  Annual fire_clim: years {sorted(fire_clim_annual_dict.keys())}")
        else:
            print(f"  [WARN] fire_clim_dir={fire_clim_dir} has no fire_clim_upto_*.tif — "
                  f"falling back to static fire_climatology_tif")

    print(f"  FWI: {len(fwi_dict):,}  2t: {len(t2m_dict):,}")
    if dew_dict:
        print(f"  2d (dewpoint): {len(dew_dict):,}")
    if tcw_dict:
        print(f"  tcw (total column water): {len(tcw_dict):,}")
    if sm20_dict:
        print(f"  sm20 (soil moisture 0-20cm): {len(sm20_dict):,}")
    if st20_dict:
        print(f"  st20 (soil temp 0-20cm): {len(st20_dict):,}")
    if lightning_dict:
        print(f"  Lightning: {len(lightning_dict):,}")
    if ndvi_index:
        print(f"  NDVI composites: {len(ndvi_index):,}")
    if deep_soil_dict:
        print(f"  Deep soil: {len(deep_soil_dict):,}")
    if precip_dict:
        print(f"  Precip (for deficit): {len(precip_dict):,}")
    if burn_scar_dict:
        print(f"  Burn scars: years {sorted(burn_scar_dict.keys())}")
    if u10_dict:
        print(f"  u10 (wind): {len(u10_dict):,}")
    if v10_dict:
        print(f"  v10 (wind): {len(v10_dict):,}")
    if cape_dict:
        print(f"  CAPE: {len(cape_dict):,}")

    # ----------------------------------------------------------------
    # STEP 2  Align dates (require FWI + 2t)
    # ----------------------------------------------------------------
    print(f"\n[STEP 2] Aligning dates (FWI + 2t required)...")
    required_end = pred_end_date + timedelta(days=lead_end + 5)
    fwi_paths, t2m_paths = [], []
    aligned_dates = []
    cur = data_start_date
    while cur <= required_end:
        if cur in fwi_dict and cur in t2m_dict:
            fwi_paths.append(fwi_dict[cur])
            t2m_paths.append(t2m_dict[cur])
            aligned_dates.append(cur)
        cur += timedelta(days=1)

    min_needed = in_days + lead_end + 1
    if len(aligned_dates) < min_needed:
        raise RuntimeError(
            f"Only {len(aligned_dates)} aligned days, need >= {min_needed}."
        )
    T = len(aligned_dates)
    print(f"  Aligned dates: {T}  ({aligned_dates[0]} → {aligned_dates[-1]})")

    # ----------------------------------------------------------------
    # MASTER CACHE RESOLUTION
    # ----------------------------------------------------------------
    # If --master_cache_dir is set, we will reuse a wider-range cache by
    # slicing along the time axis. master_t_offset / T give the slice.
    # Channel-wise stats are RANGE-INDEPENDENT, so master stats.npy is
    # reused unchanged.
    master_info = None
    if args.master_cache_dir:
        if not args.master_data_start:
            raise RuntimeError(
                "--master_cache_dir requires --master_data_start"
            )
        master_start = dt.strptime(args.master_data_start, "%Y-%m-%d").date()
        # Compute aligned-date offset of THIS run's start within master.
        if aligned_dates[0] < master_start:
            raise RuntimeError(
                f"data_start aligned={aligned_dates[0]} < master_start={master_start}; "
                f"master cache does not cover this range."
            )
        # Master must align day-by-day with this run's aligned_dates from
        # master_t_offset onwards. We trust this: master was built with
        # the same FWI/2t/etc presence checks, so missing days are the
        # same set. master_t_offset = days from master_start to aligned_dates[0].
        master_t_offset = (aligned_dates[0] - master_start).days
        # Resolve master_T from filename pattern.
        import glob as _glob
        master_pf_glob = os.path.join(
            args.master_cache_dir,
            f"meteo_v3_p{args.patch_size}_C*_T*_{args.master_data_start}_*_pf.dat"
        )
        master_pf_matches = _glob.glob(master_pf_glob)
        if not master_pf_matches:
            raise RuntimeError(
                f"No master meteo cache found matching: {master_pf_glob}"
            )
        master_pf_path = master_pf_matches[0]
        # Parse master_T from filename:
        # meteo_v3_p16_C9_T9332_2000-05-01_2025-12-20_pf.dat
        _bn = os.path.basename(master_pf_path)
        master_T = int(_bn.split("_T")[1].split("_")[0])
        master_data_end = _bn.split("_")[5]
        master_info = dict(
            cache_dir=args.master_cache_dir,
            t_offset=master_t_offset,
            T_master=master_T,
            data_start=args.master_data_start,
            data_end=master_data_end,
        )
        print(f"\n[MASTER CACHE] Reusing wider cache:")
        print(f"  dir          : {args.master_cache_dir}")
        print(f"  master range : {args.master_data_start} → {master_data_end}  (T_master={master_T})")
        print(f"  this run     : {aligned_dates[0]} → {aligned_dates[-1]}      (T={T})")
        print(f"  t_offset     : {master_t_offset}")
        if master_t_offset + T > master_T:
            raise RuntimeError(
                f"master_T={master_T} too short: need t_offset+T={master_t_offset+T}"
            )

    # ----------------------------------------------------------------
    # STEP 3  Grid dimensions & streaming per-channel stats
    # ----------------------------------------------------------------
    with rasterio.open(fwi_paths[0]) as src:
        profile = src.profile
        H, W = src.height, src.width
    print(f"\n[STEP 3] Grid: T={T}  H={H}  W={W}  Channels={N_CHANNELS}")

    # Load static channels
    static_arrays = {}
    if "fire_clim" in CHANNEL_NAMES:
        if fire_clim_annual_dict:
            # Annual mode: load all years into memory; build mean map for stats/hard-neg
            fire_clim_arrays = {}
            for yr, p in fire_clim_annual_dict.items():
                arr = _load_static_channel(p, H, W, f"fire_clim_{yr}")
                fire_clim_arrays[yr] = arr
            # Average over all loaded maps for hard-neg mining & stats
            _stacked = np.stack(list(fire_clim_arrays.values()), axis=0)
            static_arrays["fire_clim"] = _stacked.mean(axis=0)
            print(f"  fire_clim: {len(fire_clim_arrays)} annual maps loaded "
                  f"(years {sorted(fire_clim_arrays.keys())})")
        elif fire_clim_path:
            # Fallback: single static TIF
            fire_clim_arrays = {}
            static_arrays["fire_clim"] = _load_static_channel(fire_clim_path, H, W, "fire_clim")
            print(f"  fire_clim: static fallback ({fire_clim_path})")
        else:
            fire_clim_arrays = {}
            static_arrays["fire_clim"] = np.zeros((H, W), dtype=np.float32)
            print("  [WARN] fire_clim: no source found, using zeros")
    else:
        fire_clim_arrays = {}
    if "population" in CHANNEL_NAMES:
        static_arrays["population"] = _load_static_channel(population_tif, H, W, "population")
    if "slope" in CHANNEL_NAMES:
        slope_path = os.path.join(terrain_dir, "slope.tif")
        static_arrays["slope"] = _load_static_channel(slope_path, H, W, "slope")
    if "elevation" in CHANNEL_NAMES:
        # CDEM-derived DEM altitude (metres). Distinct from 'slope' which
        # is the gradient magnitude — elevation captures the absolute
        # terrain context (boreal valley vs mountain plateau).
        elev_path = os.path.join(terrain_dir, "dem_cdem.tif")
        static_arrays["elevation"] = _load_static_channel(elev_path, H, W, "elevation")
    if "aspect" in CHANNEL_NAMES:
        # Slope direction in degrees [0, 360). Important for solar
        # exposure / drying. Encoded as raw degrees here; downstream
        # normalisation makes it usable. (For sin/cos encoding, would
        # need 2 channels — defer.)
        aspect_path = os.path.join(terrain_dir, "aspect.tif")
        static_arrays["aspect"] = _load_static_channel(aspect_path, H, W, "aspect")
    if "lightning_climatology" in CHANNEL_NAMES:
        # Per-pixel mean annual lightning strikes (log1p), built from
        # GLM 2018-onwards by scripts/build_lightning_climatology.py.
        # Lightning hot-spots are stable over decades, so a static prior
        # is a valid first-order proxy for the dominant boreal ignition
        # cause (~60% of fires).
        lc_path = "data/lightning_climatology.tif"
        static_arrays["lightning_climatology"] = _load_static_channel(
            lc_path, H, W, "lightning_climatology")

    # Train/val split index
    train_end_idx = next(
        (i for i, d in enumerate(aligned_dates) if d >= pred_start_date), None
    )
    if train_end_idx is None:
        raise RuntimeError(f"pred_start={pred_start_date} is beyond all aligned dates.")
    print(f"  Train: {train_end_idx} days | Val: {T - train_end_idx} days")

    # Check for cached stats
    P = args.patch_size
    stats_path = None
    if master_info is not None:
        # Stats are range-independent; reuse master stats.
        stats_path = os.path.join(master_info["cache_dir"],
                                  f"meteo_v3_p{P}_C{N_CHANNELS}_stats.npy")
    elif args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        stats_path = os.path.join(args.cache_dir,
                                  f"meteo_v3_p{P}_C{N_CHANNELS}_stats.npy")

    if stats_path and os.path.exists(stats_path) and not args.overwrite:
        print(f"  Loading cached stats: {stats_path}")
        _s = np.load(stats_path)
        meteo_means = _s[0].astype(np.float32)
        meteo_stds = _s[1].astype(np.float32)
        fills = meteo_means.copy()
    else:
        print(f"  Computing per-channel stats (streaming)...")
        ch_stats = []

        # FWI stats
        m, s, f = _stream_channel_stats(fwi_paths[:train_end_idx])
        ch_stats.append(("FWI", m, s, f))
        print(f"  {'FWI':12s}  mean={m:8.3f}  std={s:8.3f}")

        # 2t stats (only if 2t is an active channel; 2t_anom handled in loop)
        if "2t" in CHANNEL_NAMES:
            m, s, f = _stream_channel_stats(t2m_paths[:train_end_idx])
            ch_stats.append(("2t", m, s, f))
            print(f"  {'2t':12s}  mean={m:8.3f}  std={s:8.3f}")

        # Static channels: spatial mean/std
        for ch_name in CHANNEL_NAMES:
            if ch_name in ("FWI", "2t"):
                continue
            ch_def = V3_CHANNEL_DEFS[ch_name]
            if ch_def["type"] == "static":
                arr = static_arrays.get(ch_name, np.zeros((H, W), dtype=np.float32))
                valid = arr[(arr > -1e30) & np.isfinite(arr)]
                cm = float(valid.mean()) if valid.size else 0.0
                cs = float(valid.std()) if valid.size else 1.0
                ch_stats.append((ch_name, cm, max(cs, 1e-6), cm))
                print(f"  {ch_name:12s}  mean={cm:8.3f}  std={cs:8.3f}  (spatial)")
            elif ch_def["type"] == "daily":
                # Stream from available files
                _daily_dicts = {
                    "2t_anom": t2m_anom_dict,
                    "2d": dew_dict, "tcw": tcw_dict, "sm20": sm20_dict,
                    "st20": st20_dict, "lightning": lightning_dict,
                    "deep_soil": deep_soil_dict, "u10": u10_dict,
                    "v10": v10_dict, "CAPE": cape_dict,
                    "FFMC": ffmc_dict, "DMC": dmc_dict, "DC": dc_dict,
                    "BUI": bui_dict, "ISI": isi_dict,
                }
                ch_dict = _daily_dicts.get(ch_name, {})
                if ch_dict:
                    _paths = [ch_dict[d] for d in aligned_dates[:train_end_idx] if d in ch_dict]
                else:
                    _paths = []
                if _paths:
                    m, s, f = _stream_channel_stats(_paths)
                else:
                    m, s, f = 0.0, 1.0, 0.0
                ch_stats.append((ch_name, m, max(s, 1e-6), f))
                print(f"  {ch_name:12s}  mean={m:8.3f}  std={s:8.3f}")
            elif ch_def["type"] == "interp":
                # NDVI: sample from composites
                if ndvi_index:
                    _paths = [p for d, p in ndvi_index
                              if d <= aligned_dates[train_end_idx - 1]]
                    if _paths:
                        m, s, f = _stream_channel_stats(_paths[:50])  # sample
                    else:
                        m, s, f = 0.0, 1.0, 0.0
                else:
                    m, s, f = 0.0, 1.0, 0.0
                ch_stats.append((ch_name, m, max(s, 1e-6), f))
                print(f"  {ch_name:12s}  mean={m:8.3f}  std={s:8.3f}  (sampled)")
            elif ch_def["type"] == "computed":
                # precip_def: use precip stats as proxy (converted m→mm, ×1000)
                if precip_dict:
                    _paths = [precip_dict[d] for d in aligned_dates[:train_end_idx] if d in precip_dict]
                    if _paths:
                        m, s, f = _stream_channel_stats(_paths[:50])
                        m, s, f = m * 1000.0, s * 1000.0, f * 1000.0  # m/day → mm/day
                    else:
                        m, s, f = 0.0, 1.0, 0.0
                else:
                    m, s, f = 0.0, 1.0, 0.0
                ch_stats.append((ch_name, m, max(s, 1e-6), f))
                print(f"  {ch_name:12s}  mean={m:8.3f}  std={s:8.3f}  (proxy)")
            elif ch_def["type"] == "annual":
                if ch_name == "fire_clim":
                    # Use the mean map (average of all loaded annual TIFs)
                    arr = static_arrays.get("fire_clim", np.zeros((H, W), dtype=np.float32))
                    valid = arr[(arr > -1e30) & np.isfinite(arr)]
                    cm = float(valid.mean()) if valid.size else 0.0
                    cs = float(valid.std()) if valid.size else 1.0
                    ch_stats.append((ch_name, cm, max(cs, 1e-6), cm))
                    print(f"  {ch_name:12s}  mean={cm:8.3f}  std={cs:8.3f}  (annual-avg)")
                else:
                    # burn_age / burn_count: compute stats from actual data
                    # After nodata masking (9999→0), non-burned pixels are 0.
                    # Stats are computed on the encoded values (log1p or bucket).
                    _burn_arr = None
                    if ch_name == "burn_age" and burn_scar_dict:
                        # Sample a middle year
                        _mid_year = sorted(burn_scar_dict.keys())[len(burn_scar_dict) // 2]
                        _burn_arr = _load_static_channel(
                            burn_scar_dict[_mid_year], H, W, f"burn_age_stats_{_mid_year}")
                        _burn_arr = np.maximum(_burn_arr, 0)
                        _burn_arr = np.log1p(_burn_arr)  # same as default encoding
                    elif ch_name == "burn_count" and burn_count_dict:
                        _mid_year = sorted(burn_count_dict.keys())[len(burn_count_dict) // 2]
                        _burn_arr = _load_static_channel(
                            burn_count_dict[_mid_year], H, W, f"burn_count_stats_{_mid_year}")
                        _burn_arr = np.maximum(_burn_arr, 0)
                        _burn_arr = np.log1p(_burn_arr)
                    if _burn_arr is not None:
                        valid = _burn_arr[np.isfinite(_burn_arr)]
                        cm = float(valid.mean()) if valid.size else 0.0
                        cs = float(valid.std()) if valid.size else 1.0
                        ch_stats.append((ch_name, cm, max(cs, 1e-6), cm))
                        print(f"  {ch_name:12s}  mean={cm:8.3f}  std={cs:8.3f}  (computed from data)")
                    else:
                        # Fallback if no burn data available
                        ch_stats.append((ch_name, 0.0, 1.0, 0.0))
                        print(f"  {ch_name:12s}  mean=0.000  std=1.000  (fallback, no data)")

        meteo_means = np.array([s[1] for s in ch_stats], dtype=np.float32)
        meteo_stds = np.array([max(s[2], 1e-6) for s in ch_stats], dtype=np.float32)
        fills = np.array([s[3] for s in ch_stats], dtype=np.float32)

        if stats_path:
            np.save(stats_path, np.stack([meteo_means, meteo_stds]))

    # ----------------------------------------------------------------
    # STEP 4  Load and rasterize fire labels
    # ----------------------------------------------------------------
    # Two modes (selected by --label_fusion):
    #
    # (A) default: CWFIS satellite hotspots (as in V2 / 4y SOTA)
    #     - subject to 600x detection drift (2001 vs 2023)
    #     - per polygon-detection test 2026-04-21, misses 55-90% of NBAC
    #       polygons even in recent years
    #
    # (B) --label_fusion: NBAC polygon + NFDB ignition point (NO CWFIS)
    #     - NBAC is Landsat-based post-fire analysis (temporally stable
    #       since 1972, ~10 ha minimum size)
    #     - NFDB is agency-reported ignitions (1946+, catches <10 ha fires)
    #     - CWFIS dropped: too much drift, contributes noise in early years
    #     - Prescribed burns excluded (NBAC.PRESCRIBED=y, NFDB.CAUSE in
    #       {RE}, NFDB.FIRE_TYPE='Prescribed')
    #     - Daily granularity: NBAC polygon active range AG_SDATE..AG_EDATE
    #       all days get label=1 for pixels inside polygon (uniform within
    #       active window — matches our 32-day prediction horizon granularity)
    # ----------------------------------------------------------------
    r = args.dilate_radius
    fusion_tag = "_nbac_nfdb" if args.label_fusion else ""
    fire_cache_key = None
    fire_master_path = None
    if r > 0 and master_info is not None:
        # Master fire_dilated covers a wider date range; slice along axis 0.
        fire_master_path = os.path.join(
            master_info["cache_dir"],
            f"fire_dilated_r{r}{fusion_tag}"
            f"_{master_info['data_start']}_{master_info['data_end']}_{H}x{W}.npy")
    elif r > 0 and args.cache_dir:
        fire_cache_key = os.path.join(
            args.cache_dir,
            f"fire_dilated_r{r}{fusion_tag}"
            f"_{aligned_dates[0]}_{aligned_dates[-1]}_{H}x{W}.npy")

    if fire_master_path and os.path.exists(fire_master_path) and not args.overwrite:
        print(f"\n[STEP 4] Loading MASTER fire_stack + slicing: {fire_master_path}")
        # mmap_mode='r' avoids loading entire master into RAM; we slice + copy.
        _master_fire = np.load(fire_master_path, mmap_mode='r')
        t0 = master_info["t_offset"]
        fire_stack = np.array(_master_fire[t0:t0 + T])
        del _master_fire
        print(f"  fire_stack: {fire_stack.shape}  positive_rate={fire_stack.mean():.4%}")
    elif fire_cache_key and os.path.exists(fire_cache_key) and not args.overwrite:
        print(f"\n[STEP 4] Loading cached fire_stack: {fire_cache_key}")
        fire_stack = np.load(fire_cache_key)
        if fire_stack.shape[0] > T:
            fire_stack = fire_stack[:T]
        print(f"  fire_stack: {fire_stack.shape}  positive_rate={fire_stack.mean():.4%}")
    elif args.label_fusion:
        # Mode B: NBAC + NFDB only
        print(f"\n[STEP 4] Building NBAC + NFDB fusion labels (no CWFIS)...")
        fire_stack = np.zeros((T, H, W), dtype=np.uint8)

        # --- NBAC burn polygons (spatial + date window) ---
        try:
            from src.data_ops.processing.rasterize_burn_polygons import (
                load_nbac, rasterize_nbac_batch,
            )
            nbac_gdf = load_nbac(args.nbac_path)
            # Exclude prescribed burns (not wildfires)
            if not args.include_prescribed and "PRESCRIBED" in nbac_gdf.columns:
                _before = len(nbac_gdf)
                nbac_gdf = nbac_gdf[
                    nbac_gdf["PRESCRIBED"].isna()  # 'true' = prescribed, NaN = wildfire (confirmed via audit 2026-04-21)
                ].copy()
                print(f"  [NBAC] excluded {_before - len(nbac_gdf)} prescribed burns")
            print(f"  [NBAC] {len(nbac_gdf):,} polygons (wildfires)")
            nbac_stack = rasterize_nbac_batch(
                nbac_gdf, aligned_dates, profile,
                date_source=args.nbac_date_source)
            np.maximum(fire_stack, nbac_stack, out=fire_stack)
            del nbac_stack
            print(f"  [NBAC] positive pixels: {int(fire_stack.sum()):,}")
        except Exception as _e:
            print(f"  [NBAC] FAILED: {_e}")
            raise

        # --- NFDB ignition points ---
        try:
            from src.data_ops.processing.rasterize_hotspots import (
                load_nfdb_as_hotspot_df,
            )
            # Exclude prescribed fires (CAUSE=H-PB, CAUSE=RE) unless --include_prescribed.
            # NFDB CAUSE codes: H=human wildfire, N=natural/lightning, U=unknown,
            # H-PB=prescribed human burn, RE=reburn/managed.
            _keep_causes = None
            if not args.include_prescribed:
                _keep_causes = {"H", "N", "U"}  # wildfires only
            nfdb_df = load_nfdb_as_hotspot_df(
                args.nfdb_path,
                min_size_ha=args.nfdb_min_size_ha,
                causes=_keep_causes,
            )
            print(f"  [NFDB] {len(nfdb_df):,} fires loaded "
                  f"(size >= {args.nfdb_min_size_ha} ha, "
                  f"prescribed excluded={not args.include_prescribed})")
            _before = int(fire_stack.sum())
            nfdb_stack = rasterize_hotspots_batch(
                nfdb_df, aligned_dates, profile)
            np.maximum(fire_stack, nfdb_stack, out=fire_stack)
            del nfdb_stack
            print(f"  [NFDB] added {int(fire_stack.sum()) - _before:,} new positive pixels")
        except Exception as _e:
            print(f"  [NFDB] FAILED: {_e}")
            raise
    else:
        # Mode A: CWFIS-only (default, legacy compat)
        print(f"\n[STEP 4] Loading CWFIS hotspot records...")
        hotspot_df = load_hotspot_data(hotspot_csv)
        print(f"  Total records: {len(hotspot_df):,}")
        fire_stack = rasterize_hotspots_batch(hotspot_df, aligned_dates, profile)

    # --- Dilate + cache-save (shared by mode A and B, only when not loaded from cache) ---
    if (not fire_cache_key or not os.path.exists(fire_cache_key) or args.overwrite) and r > 0:
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2 <= r ** 2)
        print(f"  Dilating: radius={r} px...")
        for t in range(T):
            if fire_stack[t].any():
                fire_stack[t] = binary_dilation(
                    fire_stack[t], structure=disk).astype(np.uint8)
        if fire_cache_key:
            os.makedirs(args.cache_dir, exist_ok=True)
            np.save(fire_cache_key, fire_stack)
            print(f"  Saved fire_stack to {fire_cache_key}")

    print(f"  fire_stack: {fire_stack.shape}  positive_rate={fire_stack.mean():.4%}")

    # ----------------------------------------------------------------
    # STEP 5  Log stats
    # ----------------------------------------------------------------
    print(f"\n[STEP 5] Normalisation stats ({N_CHANNELS} channels):")
    for i, name in enumerate(CHANNEL_NAMES):
        # Quality sanity check on stats: warn if std is tiny (near-constant channel)
        _warn = ""
        if meteo_stds[i] < 1e-3:
            _warn = "  [WARN: std<1e-3, channel nearly constant]"
        elif abs(meteo_means[i]) > 1e4:
            _warn = "  [WARN: |mean|>1e4, possible sentinel leak]"
        print(f"  {name:12s}  mean={meteo_means[i]:8.3f}  std={meteo_stds[i]:8.3f}{_warn}")
    np.save(os.path.join(ckpt_dir, "norm_stats.npy"),
            np.stack([meteo_means, meteo_stds]))

    # ----------------------------------------------------------------
    # STEP 6  Build meteo_patched float16 memmap (streaming)
    # ----------------------------------------------------------------
    Hc, Wc = H - H % P, W - W % P
    nph, npw = Hc // P, Wc // P
    hw = (Hc, Wc)
    grid = (nph, npw)
    n_patches = nph * npw
    enc_dim = P * P * N_CHANNELS

    if args.decoder in ("oracle", "zeros", "random", "climatology"):
        dec_dim_base = enc_dim
        if args.dec_dim is not None:
            dec_dim_base = args.dec_dim
    elif args.decoder == "s2s_legacy":
        dec_dim_base = S2S_DEC_DIM
    elif args.decoder == "s2s":
        dec_dim_base = enc_dim
        if args.dec_dim is not None:
            dec_dim_base = args.dec_dim
    else:
        dec_dim_base = enc_dim
    out_dim = P * P

    # Decoder context augmentation: static channels + lead time encoding
    n_ctx_channels = 0
    ctx_extra_dim = 0
    if args.decoder_ctx:
        n_ctx_channels = sum(1 for name in CHANNEL_NAMES if name in DECODER_CTX_CHANNELS)
        # +4 lead/season sin/cos, + tele_K teleconnection scalars (0 when off).
        # Teleconnection dims are appended AFTER the lead-time encoding inside
        # _augment_decoder, so the static layout is unchanged when tele_K == 0.
        ctx_extra_dim = n_ctx_channels + 4 + tele_K
        print(f"  [decoder_ctx] {n_ctx_channels} spatial means + 4 lead/season "
              f"+ {tele_K} teleconnection dims = +{ctx_extra_dim} to dec_dim")
    dec_dim = dec_dim_base + ctx_extra_dim

    meteo_mmap_gb = T * n_patches * enc_dim * 2 / 1e9
    print(f"\n[STEP 6] Streaming meteo_patched → float16 memmap")
    print(f"  n_patches={n_patches}  enc_dim={enc_dim}  ~{meteo_mmap_gb:.1f} GB")

    mmap_path = None
    master_meteo_path = None
    if master_info is not None:
        master_meteo_path = os.path.join(
            master_info["cache_dir"],
            f"meteo_v3_p{P}_C{N_CHANNELS}_T{master_info['T_master']}"
            f"_{master_info['data_start']}_{master_info['data_end']}_pf.dat")
    elif args.cache_dir:
        mmap_key = (f"meteo_v3_p{P}_C{N_CHANNELS}_T{T}"
                    f"_{aligned_dates[0]}_{aligned_dates[-1]}_pf.dat")
        mmap_path = os.path.join(args.cache_dir, mmap_key)

    if master_meteo_path and os.path.exists(master_meteo_path) and not args.overwrite:
        print(f"  Loading MASTER memmap + slicing: {master_meteo_path}")
        T_master = master_info["T_master"]
        t0 = master_info["t_offset"]
        master_mmap = np.memmap(master_meteo_path, dtype='float16', mode='r',
                                shape=(n_patches, T_master, enc_dim))
        # Slicing returns a view; if --load_to_ram, copy the slice (much
        # smaller than the master) into RAM.
        if args.load_to_ram or args.load_train_to_ram:
            print(f"  Copying slice [{t0}:{t0+T}] to RAM...")
            meteo_patched = np.array(master_mmap[:, t0:t0 + T, :])
            del master_mmap
        else:
            meteo_patched = master_mmap[:, t0:t0 + T, :]
    elif mmap_path and os.path.exists(mmap_path) and not args.overwrite:
        print(f"  Loading cached memmap: {mmap_path}")
        meteo_patched = np.memmap(mmap_path, dtype='float16', mode='r',
                                  shape=(n_patches, T, enc_dim))
        if args.load_to_ram:
            print(f"  Copying to RAM...")
            meteo_patched = np.array(meteo_patched)
    else:
        # Build time-first, then transpose.
        # Three cases for the temp builder buffer:
        #   1. cache_dir set     → use cache_dir/*_tf.dat (persisted)
        #   2. master_cache set  → use $SLURM_TMPDIR/meteo_tf_temp_<pid>.dat
        #                          (scaling sweep path: T can be 7000+ days,
        #                           in-RAM allocation OOMs the 400GB node)
        #   3. neither           → in-RAM np.zeros (small enough for 4y data)
        tf_path = mmap_path.replace("_pf.dat", "_tf.dat") if mmap_path else None
        if tf_path is None and args.master_cache_dir and master_info is not None:
            tmp_root = os.environ.get("SLURM_TMPDIR", "/tmp")
            tf_path = os.path.join(tmp_root, f"meteo_tf_temp_{os.getpid()}.dat")
            print(f"  master_cache active without cache_dir → memmap to {tf_path} "
                  f"(in-RAM build would need {T * n_patches * enc_dim * 2 / 1e9:.0f} GB)")

        if tf_path:
            meteo_tf = np.memmap(tf_path, dtype='float16', mode='w+',
                                 shape=(T, n_patches, enc_dim))
        else:
            meteo_tf = np.zeros((T, n_patches, enc_dim), dtype=np.float16)

        # Pre-load burn scar arrays by year (raw years-since-burn, encoding applied later)
        burn_scar_raw = {}  # year → (H, W) raw years-since-burn
        for year, path in burn_scar_dict.items():
            arr = _load_static_channel(path, H, W, f"burn_{year}")
            burn_scar_raw[year] = np.maximum(arr, 0)

        # Pre-load burn count arrays by year
        burn_count_arrays = {}  # year → (H, W) uint8 count of fires
        for year, path in burn_count_dict.items():
            arr = _load_static_channel(path, H, W, f"bcount_{year}")
            burn_count_arrays[year] = np.maximum(arr, 0)

        # Pre-load dist_recent_burn arrays by year (already log1p-encoded, leak-safe)
        dist_recent_raw = {}
        for year, path in dist_recent_dict.items():
            dist_recent_raw[year] = _load_static_channel(path, H, W, f"distrb_{year}")

        def _encode_burn_age(raw_years, encoding):
            """Encode years-since-burn array based on --burn_age_encoding."""
            if encoding == "log1p":
                return np.log1p(raw_years).astype(np.float32)
            elif encoding == "bucket":
                # Categorical buckets reflecting reburn ecology:
                # 0-2yr (just burned, low fuel) → 0.25
                # 3-10yr (recovering, moderate) → 0.50
                # 11-20yr (dense regrowth, high) → 0.75
                # 20+yr (mature, very high fuel) → 1.00
                out = np.full_like(raw_years, 1.0, dtype=np.float32)
                out[raw_years <= 2] = 0.25
                out[(raw_years > 2) & (raw_years <= 10)] = 0.50
                out[(raw_years > 10) & (raw_years <= 20)] = 0.75
                # 9999 (never burned) → 1.0 (treat like mature forest)
                return out
            else:  # "multi" — caller handles separately
                return np.log1p(raw_years).astype(np.float32)

        # NDVI interpolation cache
        ndvi_cache = {}
        # Precipitation accumulator for rolling deficit
        precip_deque = deque(maxlen=args.precip_deficit_days)

        # FAST PATH: pre-consolidated encoder array (skips 27K+ TIF opens)
        _con = None
        _con_date_to_idx = None
        if args.consolidated and os.path.exists(args.consolidated):
            _con_dates_path = args.consolidated + ".dates.npy"
            if os.path.exists(_con_dates_path):
                _con_dates = np.load(_con_dates_path, allow_pickle=True)
                _con = np.memmap(args.consolidated, dtype=np.float32, mode='r',
                                 shape=(len(_con_dates), H, W, N_CHANNELS))
                _con_date_to_idx = {d: i for i, d in enumerate(_con_dates)}
                print(f"  [FAST] Consolidated: {_con.shape} "
                      f"({os.path.getsize(args.consolidated) / 1e9:.1f} GB)")

        _fallback_fwi = None
        _fallback_t2m = None
        t0_mmap = time.time()

        for t_idx in range(T):
            cur_date = aligned_dates[t_idx]
            frame = np.zeros((H, W, N_CHANNELS), dtype=np.float32)

            # FAST PATH: read entire frame from consolidated (1 memmap index)
            _used_consolidated = False
            if _con is not None:
                _d_str = (cur_date.isoformat() if hasattr(cur_date, 'isoformat')
                          else str(cur_date))
                _ci = _con_date_to_idx.get(_d_str)
                if _ci is not None:
                    frame = np.array(_con[_ci])  # (H, W, N_CH) copy
                    _used_consolidated = True

            if not _used_consolidated:
                # SLOW PATH: read individual TIF files per channel
                pass  # fall through to per-channel loop below

            for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
                if _used_consolidated:
                    continue  # already filled from consolidated array
                ch_def = V3_CHANNEL_DEFS[ch_name]

                if ch_name == "FWI":
                    arr = _read_tif_safe(fwi_paths[t_idx], _fallback_fwi)
                    _fallback_fwi = arr
                    arr = np.nan_to_num(arr, nan=float(fills[ch_idx]))
                    frame[..., ch_idx] = arr

                elif ch_name == "2t":
                    arr = _read_tif_safe(t2m_paths[t_idx], _fallback_t2m)
                    _fallback_t2m = arr
                    arr = np.nan_to_num(arr, nan=float(fills[ch_idx]))
                    frame[..., ch_idx] = arr

                elif ch_def["type"] == "static":
                    frame[..., ch_idx] = static_arrays.get(ch_name, np.zeros((H, W)))

                elif ch_name == "fire_clim":
                    # Annual fire climatology: use fire_clim_upto_{year} for current year
                    yr = cur_date.year
                    if fire_clim_arrays:
                        if yr in fire_clim_arrays:
                            frame[..., ch_idx] = fire_clim_arrays[yr]
                        else:
                            # BUG FIX 2026-04-19: must use STRICTLY PRIOR year to
                            # prevent leakage. Old code used abs(y - yr) which could
                            # pick a year > yr (includes target year's fires → leak).
                            prior_years = [y for y in fire_clim_arrays.keys() if y <= yr]
                            if prior_years:
                                nearest = max(prior_years)  # greatest year ≤ target
                                frame[..., ch_idx] = fire_clim_arrays[nearest]
                            else:
                                # No prior data → zeros (degenerate but safe)
                                frame[..., ch_idx] = 0.0
                    else:
                        # Fallback to static mean map
                        frame[..., ch_idx] = static_arrays.get("fire_clim", np.zeros((H, W)))

                elif ch_name == "NDVI":
                    frame[..., ch_idx] = _interpolate_ndvi(cur_date, ndvi_index, ndvi_cache, H, W)

                elif ch_name in ("2t_anom", "2d", "tcw", "sm20", "st20", "lightning",
                                 "deep_soil", "u10", "v10", "CAPE",
                                 "FFMC", "DMC", "DC", "BUI", "ISI"):
                    _daily_dicts = {
                        "2t_anom": t2m_anom_dict,
                        "2d": dew_dict, "tcw": tcw_dict, "sm20": sm20_dict,
                        "st20": st20_dict, "lightning": lightning_dict,
                        "deep_soil": deep_soil_dict, "u10": u10_dict,
                        "v10": v10_dict, "CAPE": cape_dict,
                        "FFMC": ffmc_dict, "DMC": dmc_dict, "DC": dc_dict,
                        "BUI": bui_dict, "ISI": isi_dict,
                    }
                    ch_dict = _daily_dicts.get(ch_name, {})
                    if cur_date in ch_dict:
                        arr = _read_tif_safe(ch_dict[cur_date], None)
                        if arr is not None:
                            frame[..., ch_idx] = np.nan_to_num(arr, nan=float(fills[ch_idx]))

                elif ch_name == "precip_def":
                    # Accumulate precipitation for rolling deficit
                    # ERA5 tp is in meters/day — convert to mm/day (* 1000)
                    if cur_date in precip_dict:
                        p_arr = _read_tif_safe(precip_dict[cur_date], None)
                        if p_arr is not None:
                            precip_deque.append(np.nan_to_num(p_arr, nan=0.0) * 1000.0)
                    if len(precip_deque) > 0:
                        # Simple deficit: negative of accumulated precip (less rain = higher deficit)
                        rolling_sum = np.sum(precip_deque, axis=0)
                        frame[..., ch_idx] = -rolling_sum  # negative = deficit

                elif ch_name == "burn_age":
                    # Use PREVIOUS year's burn scars to avoid temporal leakage.
                    # years_since_burn_YYYY.tif contains all fires from year YYYY,
                    # so a sample from 2021-07 would see Sept-Dec 2021 fires.
                    # Using year-1 ensures we only see fires strictly before this year.
                    prev_year = cur_date.year - 1
                    raw = None
                    if prev_year in burn_scar_raw:
                        raw = burn_scar_raw[prev_year]
                    elif burn_scar_raw:
                        # Fallback: use nearest year that is <= prev_year
                        valid_years = [y for y in burn_scar_raw.keys() if y <= prev_year]
                        if valid_years:
                            raw = burn_scar_raw[max(valid_years)]
                    if raw is not None:
                        frame[..., ch_idx] = _encode_burn_age(raw, args.burn_age_encoding)

                elif ch_name == "burn_count":
                    # Same logic: use previous year to avoid temporal leakage
                    prev_year = cur_date.year - 1
                    if prev_year in burn_count_arrays:
                        frame[..., ch_idx] = np.log1p(burn_count_arrays[prev_year])
                    elif burn_count_arrays:
                        valid_years = [y for y in burn_count_arrays.keys() if y <= prev_year]
                        if valid_years:
                            frame[..., ch_idx] = np.log1p(burn_count_arrays[max(valid_years)])

                elif ch_name == "dist_recent_burn":
                    # File for year Y already uses only Y-1..Y-3 burns (leak-safe),
                    # and is pre-encoded as log1p(dist_km); use cur year directly.
                    y = cur_date.year
                    if y in dist_recent_raw:
                        frame[..., ch_idx] = dist_recent_raw[y]
                    elif dist_recent_raw:
                        vy = [yy for yy in dist_recent_raw.keys() if yy <= y]
                        if vy:
                            frame[..., ch_idx] = dist_recent_raw[max(vy)]

            # Normalize and patchify
            frame -= meteo_means
            frame /= meteo_stds
            np.clip(frame, -10.0, 10.0, out=frame)
            meteo_tf[t_idx] = _patchify_frame(frame, P).astype(np.float16)

            if t_idx % 100 == 0 or t_idx == T - 1:
                elapsed = time.time() - t0_mmap
                eta_min = elapsed / max(t_idx, 1) * (T - t_idx) / 60
                print(f"  day {t_idx+1:4d}/{T}  "
                      f"({elapsed:.0f}s  ~{eta_min:.0f}m left)")

        # Clear NDVI cache
        ndvi_cache.clear()

        if mmap_path:
            meteo_tf.flush()
            del meteo_tf
            gc.collect()

            print(f"\n  Transposing to patch-first → {mmap_path}")
            _transpose_tf_to_pf(tf_path, mmap_path, T, n_patches, enc_dim,
                                chunk_patches=args.chunk_patches)
            os.remove(tf_path)
            meteo_patched = np.memmap(mmap_path, dtype='float16', mode='r',
                                      shape=(n_patches, T, enc_dim))
        elif tf_path is not None:
            # master_cache without cache_dir: meteo_tf was memmap'd to
            # SLURM_TMPDIR (case 2 above). Transpose via the SAME chunked
            # memmap path. The previous np.ascontiguousarray(transpose)
            # materialized the full (n_patches, T, enc_dim) array in RAM
            # → OOM at 383-810 GB for 10-18y ranges (the build memmap
            # fix was incomplete: it memmap'd the builder buffer but the
            # transpose still realized everything back into RAM).
            meteo_tf.flush()
            del meteo_tf
            gc.collect()
            pf_path = tf_path.replace("_tf_temp_", "_pf_temp_")
            print(f"\n  Transposing to patch-first (chunked memmap) → {pf_path}")
            _transpose_tf_to_pf(tf_path, pf_path, T, n_patches, enc_dim,
                                chunk_patches=args.chunk_patches)
            os.remove(tf_path)
            meteo_patched = np.memmap(pf_path, dtype='float16', mode='r',
                                      shape=(n_patches, T, enc_dim))
        else:
            # case 3: in-RAM np.zeros build (4y, small) — safe to realize.
            _tmp = np.ascontiguousarray(meteo_tf.transpose(1, 0, 2))
            del meteo_tf
            gc.collect()
            meteo_patched = _tmp
    return TrainingInputs(
        meteo_patched=meteo_patched,
        meteo_means=meteo_means,
        meteo_stds=meteo_stds,
        static_arrays=static_arrays,
        fire_stack=fire_stack,
        aligned_dates=aligned_dates,
        T=T,
        P=P,
        Hc=Hc,
        Wc=Wc,
        hw=hw,
        grid=grid,
        n_patches=n_patches,
        enc_dim=enc_dim,
        dec_dim=dec_dim,
        dec_dim_base=dec_dim_base,
        out_dim=out_dim,
        ctx_extra_dim=ctx_extra_dim,
        fusion_tag=fusion_tag,
        master_info=master_info,
        meteo_mmap_gb=meteo_mmap_gb,
    )
