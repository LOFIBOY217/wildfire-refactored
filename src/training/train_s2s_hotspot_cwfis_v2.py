"""
Train S2S Hotspot Transformer V2  [CWFIS Hotspot Labels, Lead 14–46 Days]
==========================================================================
V2 of the S2S model.  Architecture is identical to V1 (S2SHotspotTransformer)
but the input feature set is expanded from 3 channels to 8 channels:

  Channel  Variable    Source            Why it helps at S2S range
  -------  ---------   ------            --------------------------
  0        FWI         fwi_dir           Overall fire weather index
  1        2t          observation_dir   2 m temperature
  2        2d          observation_dir   2 m dewpoint (humidity)
  3        FFMC        ffmc_dir          Surface fuel dryness (responds in 1–2 days)
  4        DMC         dmc_dir           Mid-layer duff moisture (1–2 week lag)
  5        DC          dc_dir          ★ Deep drought code (month-long lag → S2S gold)
  6        BUI         bui_dir           Total available fuel
  7        Fire clim   fire_climatology  Static: historical fire-day frequency (log1p)

n_channels = 8   →   enc_dim = P² × 8 = 16² × 8 = 2048

When ERA5 u10, v10, precipitation, and CAPE TIF files are available (after
running download_ecmwf_reanalysis_observations.py and a GRIB→TIF step),
the script can be extended to 12 channels by increasing n_channels and adding
the extra stacks in STEP 3 / STEP 6.

Compare with V1 (train_s2s_hotspot_cwfis.py):
    V1: 3 channels (FWI, 2t, 2d)   enc_dim = 768
    V2: 8 channels (see above)      enc_dim = 2048

Output:
    outputs/s2s_hotspot_cwfis_v2_fire_prob/YYYYMMDD/fire_prob_lead{k:02d}d_YYYYMMDD.tif
    (compatible with evaluate_topk_cwfis.py — same format as V1 outputs)

Usage (quick test — 1 year training):
    python -m src.training.train_s2s_hotspot_cwfis_v2 \\
        --config configs/paths_windows.yaml \\
        --data_start 2018-05-01 \\
        --pred_start 2022-05-01 \\
        --pred_end   2024-10-31 \\
        --lead_start 14 --lead_end 46 \\
        --patch_size 16 --d_model 256 --nhead 8 \\
        --enc_layers 4 --dec_layers 4 \\
        --epochs 30 --batch_size 128 --lr 1e-4 \\
        --neg_ratio 20 --pos_weight_cap 10 \\
        --dilate_radius 14
"""

import argparse
from bisect import bisect_right
import glob
import gc
import json
import os
import sys
import time
import atexit
import threading
from datetime import date, timedelta
from datetime import datetime as dt

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

import numpy as np
import rasterio
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset, DataLoader

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    from pathlib import Path
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument

from src.utils.date_utils import extract_date_from_filename
from src.utils.raster_io import read_singleband_stack, clean_nodata
from src.utils.patch_utils import patchify, depatchify
from src.data_ops.processing.rasterize_hotspots import load_hotspot_data, rasterize_hotspots_batch
from src.models.s2s_hotspot import S2SHotspotTransformer


# ------------------------------------------------------------------ #
# V2 channel configuration
# ------------------------------------------------------------------ #

# Channel names (for logging and diagnostics)
CHANNEL_NAMES = ["FWI", "2t", "2d", "FFMC", "DMC", "DC", "BUI", "fire_clim"]
N_CHANNELS    = len(CHANNEL_NAMES)   # 8

# S2S decoder channel count (2t, 2d, tcw, sm20, st20, VPD)
S2S_N_CHANNELS = 6
# S2S decoder input dim: 6 weather channels + issue_age + is_fallback + is_missing
S2S_DEC_DIM = S2S_N_CHANNELS + 3  # 9


# ------------------------------------------------------------------ #
# Datasets  (identical to V1 — fully generic w.r.t. enc_dim)
# ------------------------------------------------------------------ #

def _make_dec_ablation(decoder_mode, x_enc, dec_days, dec_dim):
    """
    Build decoder input tensor for ablation modes.

    decoder_mode:
      "zeros"       — all zeros (shape: dec_days × dec_dim)
      "random"      — i.i.d. standard normal noise; no information leakage
      "climatology" — repeat the per-feature mean of the encoder window
                      (shape: dec_days × dec_dim); represents "no forecast,
                      just use the recent past average"
    x_enc shape: (enc_days, dec_dim)
    """
    if decoder_mode == "zeros":
        return np.zeros((dec_days, dec_dim), dtype=np.float16)
    elif decoder_mode == "random":
        return np.random.randn(dec_days, dec_dim).astype(np.float16)
    elif decoder_mode == "climatology":
        # Mean over encoder days → (dec_dim,), then tile to (dec_days, dec_dim)
        # Cast to float32 for stable mean, then back to float16
        enc_mean = x_enc.astype(np.float32).mean(axis=0, keepdims=True)  # (1, dec_dim)
        return np.repeat(enc_mean, dec_days, axis=0).astype(np.float16)  # (dec_days, dec_dim)
    else:
        raise NotImplementedError(f"decoder_mode='{decoder_mode}' not supported")


def _expand_s2s_date_mapping(s2s_dates, aligned_dates, max_lag_days):
    """
    Expand sparse S2S issue dates into a dense lookup over *aligned_dates*.

    ECMWF S2S hindcasts are often only available on issue days, so exact
    base-date lookup leaves many windows unmapped. This maps
    each aligned base date to the most recent available issue date within
    *max_lag_days*; dates with no recent issue stay unmapped.

    Returns:
        dense_map:         {YYYY-MM-DD: s2s_row_index}
        dense_map_exact:   {YYYY-MM-DD: True if exact issue-date hit else False}
        dense_map_lag:     {YYYY-MM-DD: int lag days (0 for exact)}
    """
    s2s_date_objs = [date.fromisoformat(str(d)) for d in s2s_dates]
    date_to_idx_exact = {str(d): i for i, d in enumerate(s2s_dates)}
    dense_map = {}
    dense_map_exact = {}
    dense_map_lag = {}

    for d in aligned_dates:
        d_str = str(d)
        exact_idx = date_to_idx_exact.get(d_str)
        if exact_idx is not None:
            dense_map[d_str] = exact_idx
            dense_map_exact[d_str] = True
            dense_map_lag[d_str] = 0
            continue

        pos = bisect_right(s2s_date_objs, d) - 1
        if pos < 0:
            continue
        lag_days = (d - s2s_date_objs[pos]).days
        if lag_days <= max_lag_days:
            dense_map[d_str] = pos
            dense_map_exact[d_str] = False
            dense_map_lag[d_str] = lag_days

    return dense_map, dense_map_exact, dense_map_lag


def _make_dec_s2s(s2s_cache, date_to_s2s_idx, date_str, patch_i,
                  dec_days, dec_dim, patch_size,
                  s2s_means=None, s2s_stds=None,
                  date_to_s2s_lag=None, s2s_max_lag=3):
    """
    Build decoder input from S2S forecast patch-mean cache.

    s2s_cache  : (n_dates, n_patches, 32, 6) float16 memmap or ndarray
    date_str   : YYYY-MM-DD string for the forecast issue date
    patch_i    : patch index (row-major)
    dec_days   : number of decoder lead days (= lead_end - lead_start + 1 = 33)
    dec_dim    : S2S_DEC_DIM = 9 (6 weather + issue_age + is_fallback + is_missing)
    patch_size : (unused, kept for API compatibility)
    s2s_means  : (6,) float32 per-channel means for z-score normalization (or None)
    s2s_stds   : (6,) float32 per-channel stds  for z-score normalization (or None)
    date_to_s2s_lag : {YYYY-MM-DD: int} lag days for each date (0=exact)
    s2s_max_lag     : int, max lag days for normalization of issue_age feature

    Returns (dec_days, 9) float16:
      [:, :6]  — 6-channel S2S forecast (z-score normalized; random N(0,1) if missing)
      [:, 6]   — issue_age: lag_days / s2s_max_lag, in [0, 1]
      [:, 7]   — is_fallback: 0.0 (exact) or 1.0 (fallback)
      [:, 8]   — is_missing: 1.0 if lead-day data is missing (all-zero in cache)
    """
    out = np.zeros((dec_days, S2S_DEC_DIM), dtype=np.float16)

    if date_str is None or date_to_s2s_idx is None or s2s_cache is None:
        # Entire date unmapped — random noise + is_missing=1
        # Random breaks decoder attention dependence; better than zero (see doc)
        out[:, :S2S_N_CHANNELS] = np.random.standard_normal(
            (dec_days, S2S_N_CHANNELS)).astype(np.float16)
        out[:, S2S_N_CHANNELS + 2] = np.float16(1.0)
        return out

    s2s_idx = date_to_s2s_idx.get(date_str, None)
    if s2s_idx is None:
        out[:, :S2S_N_CHANNELS] = np.random.standard_normal(
            (dec_days, S2S_N_CHANNELS)).astype(np.float16)
        out[:, S2S_N_CHANNELS + 2] = np.float16(1.0)
        return out

    # s2s_cache[s2s_idx, patch_i] has shape (32, 6)
    lead_data = s2s_cache[s2s_idx, patch_i]   # (32, 6) float16

    # Validate: dec_days must not exceed cache lead count (S2S_N_LEADS=32)
    n_cache_leads = lead_data.shape[0]  # should be 32
    if dec_days > n_cache_leads:
        raise ValueError(
            f"_make_dec_s2s: dec_days={dec_days} > cache leads={n_cache_leads}. "
            f"Ensure --lead_end <= {14 + n_cache_leads - 1} when --decoder s2s."
        )

    # Detect cache-internal missing: rows where ALL 6 channels are 0
    # (caused by missing lead tif or NaN-only patches in build_s2s_decoder_cache)
    lead_slice = lead_data[:dec_days]  # (dec_days, 6)
    missing_mask = np.all(lead_slice == 0, axis=1)  # (dec_days,) bool

    # Normalize: z-score per channel, then clip to [-10, 10]
    if s2s_means is not None and s2s_stds is not None:
        lead_data = np.array(lead_data, dtype=np.float32)  # copy for arithmetic
        lead_data = (lead_data - s2s_means) / s2s_stds
        np.clip(lead_data, -10.0, 10.0, out=lead_data)
        lead_data = lead_data.astype(np.float16)

    # Weather channels: (dec_days, 6)
    out[:dec_days, :S2S_N_CHANNELS] = lead_data[:dec_days]

    # Fill missing lead days with random noise instead of (0-mean)/std false signal
    # Random breaks decoder attention dependence; is_missing=1 tells model to ignore
    if missing_mask.any():
        n_miss = int(missing_mask.sum())
        out[missing_mask, :S2S_N_CHANNELS] = np.random.standard_normal(
            (n_miss, S2S_N_CHANNELS)).astype(np.float16)
        out[missing_mask, S2S_N_CHANNELS + 2] = np.float16(1.0)

    # Metadata features: issue_age and is_fallback (constant across all lead days)
    lag = 0
    is_fb = np.float16(0.0)
    if date_to_s2s_lag is not None:
        lag = date_to_s2s_lag.get(date_str, 0)
        is_fb = np.float16(1.0) if lag > 0 else np.float16(0.0)
    age_norm = np.float16(lag / max(s2s_max_lag, 1))  # normalize to [0, 1]
    out[:dec_days, S2S_N_CHANNELS]     = age_norm
    out[:dec_days, S2S_N_CHANNELS + 1] = is_fb

    return out


def _make_dec_s2s_full(s2s_full_cache, date_to_s2s_idx, date_str, patch_i,
                       dec_days, enc_dim):
    """
    Build decoder input from full-patch S2S cache (Oracle-format).

    s2s_full_cache: (n_dates, n_patches, 32, P²×8) float16 memmap
    Values are already z-score normalized (same as meteo_patched).
    Returns (dec_days, enc_dim) float16.
    """
    if date_str is None or date_to_s2s_idx is None or s2s_full_cache is None:
        return np.zeros((dec_days, enc_dim), dtype=np.float16)

    idx = date_to_s2s_idx.get(date_str)
    if idx is None:
        return np.zeros((dec_days, enc_dim), dtype=np.float16)

    return s2s_full_cache[idx, patch_i, :dec_days, :].copy()


class S2SHotspotDatasetMixed(Dataset):
    """
    Training dataset: pos_pairs (patches with ≥1 fire pixel in target window)
    mixed with neg_ratio × neg_pairs (pure-background patches).

    decoder_mode:
      "oracle"      — x_dec = future ERA5 obs (default, highest accuracy)
      "zeros"       — x_dec = all zeros
      "random"      — x_dec = i.i.d. standard normal noise
      "climatology" — x_dec = encoder-period mean repeated across decoder days
      "s2s"         — x_dec from ECMWF S2S forecast patch-mean cache
    """

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid, all_pairs,
                 decoder_mode="oracle", dec_dim=None,
                 s2s_cache=None, date_to_s2s_idx=None, window_dates=None,
                 patch_size=16, s2s_means=None, s2s_stds=None,
                 date_to_s2s_lag=None, s2s_max_lag=3,
                 s2s_full_cache=None):
        self.meteo          = meteo_patched
        self.fire           = fire_patched
        self.windows        = windows
        self.hw             = hw
        self.grid           = grid
        self.all_pairs      = all_pairs
        self.decoder_mode   = decoder_mode
        self.dec_dim        = dec_dim or meteo_patched.shape[2]
        self.s2s_cache      = s2s_cache        # (n_dates, n_patches, 32, 6) float16
        self.date_to_s2s_idx = date_to_s2s_idx  # {date_str: int}
        self.window_dates   = window_dates     # list[str]: aligned_dates[w[1]] (base date) for each window
        self.patch_size     = patch_size
        self.s2s_means      = s2s_means        # (6,) float32 or None
        self.s2s_stds       = s2s_stds         # (6,) float32 or None
        self.date_to_s2s_lag = date_to_s2s_lag  # {date_str: int} lag days
        self.s2s_max_lag    = s2s_max_lag       # int, for age normalization
        self.s2s_full_cache = s2s_full_cache   # (n_dates, n_patches, 32, enc_dim) float16 or None

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        win_i, patch_i = int(self.all_pairs[idx, 0]), int(self.all_pairs[idx, 1])
        hs, he, ts, te = self.windows[win_i]
        # meteo layout: (n_patches, T, enc_dim) — patch-first for sequential read
        # fire  layout: (T, n_patches, P*P)     — time-first (small, OK)
        # Keep float16 — cast to float32 happens on GPU (.to(device, dtype=torch.float32))
        # This halves the CPU copy cost and PCIe transfer bandwidth.
        x_enc = self.meteo[patch_i, hs:he, :].copy()   # float16
        if self.decoder_mode == "oracle":
            x_dec = self.meteo[patch_i, ts:te, :].copy()   # float16
        elif self.decoder_mode == "s2s":
            if self.s2s_full_cache is not None:
                x_dec = _make_dec_s2s_full(
                    self.s2s_full_cache, self.date_to_s2s_idx,
                    self.window_dates[win_i], patch_i,
                    te - ts, self.dec_dim,
                )
            else:
                x_dec = _make_dec_s2s(
                    self.s2s_cache, self.date_to_s2s_idx,
                    self.window_dates[win_i], patch_i,
                    te - ts, self.dec_dim, self.patch_size,
                    s2s_means=self.s2s_means, s2s_stds=self.s2s_stds,
                    date_to_s2s_lag=self.date_to_s2s_lag,
                    s2s_max_lag=self.s2s_max_lag,
                )
        else:
            x_dec = _make_dec_ablation(self.decoder_mode, x_enc, te - ts, self.dec_dim)
        y = self.fire[ts:te, patch_i, :].astype(np.float32)   # uint8 → float32 (needed for BCE)
        return (
            torch.from_numpy(x_enc),
            torch.from_numpy(x_dec),
            torch.from_numpy(y),
        )


class S2SHotspotDatasetUnfiltered(Dataset):
    """Unfiltered dataset for validation — all patches, no pos/neg sampling."""

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid,
                 decoder_mode="oracle", dec_dim=None,
                 s2s_cache=None, date_to_s2s_idx=None, window_dates=None,
                 patch_size=16, s2s_means=None, s2s_stds=None,
                 date_to_s2s_lag=None, s2s_max_lag=3,
                 s2s_full_cache=None):
        self.meteo          = meteo_patched
        self.fire           = fire_patched
        self.windows        = windows
        self.hw             = hw
        self.grid           = grid
        self.n_patches      = meteo_patched.shape[0]   # (n_patches, T, enc_dim)
        self.decoder_mode   = decoder_mode
        self.dec_dim        = dec_dim or meteo_patched.shape[2]
        self.s2s_cache      = s2s_cache
        self.date_to_s2s_idx = date_to_s2s_idx
        self.window_dates   = window_dates
        self.patch_size     = patch_size
        self.s2s_means      = s2s_means
        self.s2s_stds       = s2s_stds
        self.date_to_s2s_lag = date_to_s2s_lag
        self.s2s_max_lag    = s2s_max_lag
        self.s2s_full_cache = s2s_full_cache   # (n_dates, n_patches, 32, enc_dim) float16 or None

    def __len__(self):
        return len(self.windows) * self.n_patches

    def __getitem__(self, idx):
        win_i   = idx // self.n_patches
        patch_i = idx %  self.n_patches
        hs, he, ts, te = self.windows[win_i]
        # meteo layout: (n_patches, T, enc_dim) — patch-first for sequential read
        # fire  layout: (T, n_patches, P*P)     — time-first (small, OK)
        # Keep float16 — cast to float32 happens on GPU (.to(device, dtype=torch.float32))
        x_enc = self.meteo[patch_i, hs:he, :].copy()   # float16
        if self.decoder_mode == "oracle":
            x_dec = self.meteo[patch_i, ts:te, :].copy()   # float16
        elif self.decoder_mode == "s2s":
            if self.s2s_full_cache is not None:
                x_dec = _make_dec_s2s_full(
                    self.s2s_full_cache, self.date_to_s2s_idx,
                    self.window_dates[win_i], patch_i,
                    te - ts, self.dec_dim,
                )
            else:
                x_dec = _make_dec_s2s(
                    self.s2s_cache, self.date_to_s2s_idx,
                    self.window_dates[win_i], patch_i,
                    te - ts, self.dec_dim, self.patch_size,
                    s2s_means=self.s2s_means, s2s_stds=self.s2s_stds,
                    date_to_s2s_lag=self.date_to_s2s_lag,
                    s2s_max_lag=self.s2s_max_lag,
                )
        else:
            x_dec = _make_dec_ablation(self.decoder_mode, x_enc, te - ts, self.dec_dim)
        y = self.fire[ts:te, patch_i, :].astype(np.float32)   # uint8 → float32 (needed for BCE)
        return (
            torch.from_numpy(x_enc),
            torch.from_numpy(x_dec),
            torch.from_numpy(y),
        )


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _build_file_dict(directory, prefix):
    """Return {date: filepath} for tif files matching prefix_YYYYMMDD.tif.
    Looks for files in {directory}/{prefix}/{prefix}_*.tif first,
    then in {directory}/{prefix}_*.tif.
    """
    result = {}
    sub_paths = sorted(glob.glob(
        os.path.join(directory, prefix, f"{prefix}_*.tif")
    ))
    paths = sub_paths if sub_paths else sorted(glob.glob(
        os.path.join(directory, f"{prefix}_*.tif")
    ))
    for p in paths:
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            result[d] = p
    return result


def _build_flat_file_dict(directory, prefix):
    """Return {date: filepath} for FLAT dir with prefix_YYYYMMDD.tif files.
    Used for FWI sub-component dirs (ffmc, dmc, dc, bui) which store TIFs
    directly in the directory root (no subdir).
    """
    result = {}
    for p in sorted(glob.glob(os.path.join(directory, f"{prefix}_*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            result[d] = p
    # Fallback: also check subdirectory pattern (matches _build_file_dict behaviour)
    if not result:
        for p in sorted(glob.glob(os.path.join(directory, prefix, f"{prefix}_*.tif"))):
            d = extract_date_from_filename(os.path.basename(p))
            if d:
                result[d] = p
    return result


def _build_s2s_windows(n_days, in_days, lead_start, lead_end):
    """Build (enc_start, enc_end, target_start, target_end) index tuples."""
    windows = []
    for i in range(in_days, n_days - lead_end):
        windows.append((i - in_days, i, i + lead_start, i + lead_end + 1))
    return windows


# ------------------------------------------------------------------ #
# Streaming helpers for OOM-safe data loading                        #
# ------------------------------------------------------------------ #

def _read_tif_safe(path, fallback, nodata_threshold=-1e30):
    """
    Read one TIF as float32.  On error, substitute *fallback* (previous
    day's array).  Masks extreme nodata values and any nodata tag stored
    in the file.
    """
    try:
        with rasterio.open(path) as src:
            arr  = src.read(1).astype(np.float32)
            ndv  = src.nodata
    except Exception:
        if fallback is None:
            raise
        return fallback.copy()
    arr[arr < nodata_threshold] = np.nan
    arr[~np.isfinite(arr)]      = np.nan
    if ndv is not None:
        arr[arr == ndv] = np.nan
    if fallback is not None:
        # fill remaining NaN with previous day
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = fallback[mask]
    return arr


def _stream_channel_stats(paths, nodata_threshold=-1e30):
    """
    Compute (mean, std, fill_value) for one channel by loading ONE TIF
    at a time — never more than ~25 MB in RAM.

    Uses exact two-pass running sums rather than Welford to avoid
    floating-point cancellation on large arrays.
    """
    total_sum, total_sum_sq, total_count = 0.0, 0.0, 0
    fallback = None
    for p in paths:
        arr      = _read_tif_safe(p, fallback, nodata_threshold)
        fallback = arr
        valid    = arr[np.isfinite(arr)].astype(np.float64)
        if valid.size:
            total_sum    += float(valid.sum())
            total_sum_sq += float((valid ** 2).sum())
            total_count  += valid.size
    if total_count == 0:
        return 0.0, 1.0, 0.0
    mean = total_sum / total_count
    var  = max(total_sum_sq / total_count - mean ** 2, 0.0)
    std  = float(np.sqrt(var)) if var > 0 else 1.0
    return float(mean), std, float(mean)   # fill = mean


def _patchify_frame(frame_hwc, P):
    """
    Patchify a single spatial frame.

    Args:
        frame_hwc: (H, W, C) float32 numpy array
        P: patch size

    Returns:
        (n_patches, P*P*C) float32 — CROPS to P-aligned boundary.
    """
    H, W, C = frame_hwc.shape
    Hc, Wc  = H - H % P, W - W % P
    f = frame_hwc[:Hc, :Wc, :]                   # (Hc, Wc, C)
    nph, npw = Hc // P, Wc // P
    f = f.reshape(nph, P, npw, P, C)
    f = f.transpose(0, 2, 1, 3, 4)               # (nph, npw, P, P, C)
    return f.reshape(nph * npw, P * P * C)        # (n_patches, enc_dim)


def _transpose_tf_to_pf(tf_path, pf_path, T, n_patches, enc_dim,
                        chunk_patches=200):
    """
    Transpose a time-first memmap (T, n_patches, enc_dim) to patch-first
    (n_patches, T, enc_dim) on disk using chunked RAM copies.

    Peak RAM ≈ chunk_patches × T × enc_dim × 2 bytes
    (default 200 × 2427 × 2048 × 2 ≈ 2 GB per chunk).
    """
    tf = np.memmap(tf_path, dtype='float16', mode='r',
                   shape=(T, n_patches, enc_dim))
    pf = np.memmap(pf_path, dtype='float16', mode='w+',
                   shape=(n_patches, T, enc_dim))
    t0 = time.time()
    n_chunks = (n_patches + chunk_patches - 1) // chunk_patches
    for ci, p_start in enumerate(range(0, n_patches, chunk_patches)):
        p_end = min(p_start + chunk_patches, n_patches)
        # Force load to RAM → (T, chunk, enc_dim)
        chunk = np.array(tf[:, p_start:p_end, :])
        # Transpose axes 0↔1 → (chunk, T, enc_dim) and write sequentially
        pf[p_start:p_end] = chunk.transpose(1, 0, 2)
        if ci % 10 == 0 or p_end == n_patches:
            elapsed = time.time() - t0
            frac    = p_end / n_patches
            eta_min = elapsed / max(frac, 1e-9) * (1 - frac) / 60
            print(f"  Transposing {p_end:>6}/{n_patches} patches  "
                  f"({elapsed:.0f}s  ~{eta_min:.0f} min left)")
    pf.flush()
    del tf, pf
    gc.collect()   # Windows: force release file handles before caller deletes the file


class MemoryGuard(threading.Thread):
    """
    Background thread: polls system RAM every `interval` seconds.
    If usage exceeds `limit_pct`, sets self.triggered and prints a
    diagnosis report.  The training loop checks self.triggered each
    epoch and stops gracefully.

    Requires psutil (pip install psutil).  If psutil is unavailable the
    guard is created but does nothing (no AttributeError in caller code).
    """

    def __init__(self, limit_pct=90.0, interval=15,
                 meteo_gb=0.0, fire_gb=0.0, batch_size=128):
        super().__init__(daemon=True)
        self.limit_pct   = limit_pct
        self.interval    = interval
        self.meteo_gb    = meteo_gb
        self.fire_gb     = fire_gb
        self.batch_size  = batch_size
        self.triggered   = False        # set to True when threshold exceeded
        self._kill       = threading.Event()

    def run(self):
        if not _PSUTIL_OK:
            return
        while not self._kill.wait(self.interval):
            self._check()

    def _check(self):
        vm        = _psutil.virtual_memory()
        used_pct  = vm.percent
        used_gb   = vm.used  / 1e9
        total_gb  = vm.total / 1e9
        avail_gb  = vm.available / 1e9
        if used_pct >= self.limit_pct and not self.triggered:
            self.triggered = True
            sep = "=" * 70
            print(f"\n{sep}")
            print(f"  ⚠  MEMORY GUARD TRIGGERED — training will stop after this epoch.")
            print(f"  RAM: {used_gb:.1f} GB used / {total_gb:.1f} GB total "
                  f"({used_pct:.1f}%  ≥  limit {self.limit_pct:.0f}%)")
            print(f"  Available: {avail_gb:.1f} GB")
            print()
            print(f"  Likely contributors:")
            print(f"    meteo_patched  (float16 OS page-cache): up to {self.meteo_gb:.0f} GB")
            print(f"    fire_patched   (OS page-cache):         up to {self.fire_gb:.0f} GB")
            print(f"    Model + optimizer + GPU activations:    ~2–4 GB")
            print(f"    Other users on this machine:            check Task Manager")
            print()
            print(f"  Suggestions to reduce RAM:")
            print(f"    1) Lower --mem_limit_pct if you want an earlier warning")
            print(f"    2) Reboot between runs to flush OS page-cache")
            print(f"{sep}\n")

    def shutdown(self):
        self._kill.set()


def _load_fire_clim(tif_path, expected_h, expected_w):
    """
    Load the static fire-climatology GeoTIFF.

    Returns:
        numpy.ndarray of shape (H, W), float32.  Returns zeros if the file
        does not exist or has mismatched dimensions (with a warning).
    """
    if tif_path is None or not os.path.exists(tif_path):
        print(f"  [WARN] fire_climatology_tif not found: {tif_path}")
        print(f"         Using zeros for Channel 7. Run make_fire_climatology.py first.")
        return np.zeros((expected_h, expected_w), dtype=np.float32)

    with rasterio.open(tif_path) as src:
        clim = src.read(1).astype(np.float32)

    if clim.shape != (expected_h, expected_w):
        print(f"  [WARN] fire_climatology shape {clim.shape} ≠ FWI grid ({expected_h},{expected_w})")
        print(f"         Using zeros for Channel 7.")
        return np.zeros((expected_h, expected_w), dtype=np.float32)

    nonzero = int((clim > 0).sum())
    print(f"  fire_clim: {clim.shape}  nonzero={nonzero:,}  "
          f"max={clim.max():.3f}  mean(nz)={clim[clim>0].mean():.3f}")
    return clim


# ------------------------------------------------------------------ #
# Val Lift@K  (used for checkpoint selection)
# ------------------------------------------------------------------ #

def _compute_val_lift_k(model, meteo_patched, fire_patched, val_wins,
                        n_patches, k, n_sample_wins, chunk, device,
                        decoder_mode="oracle", dec_dim=None,
                        s2s_cache=None, date_to_s2s_idx=None,
                        val_win_dates=None, patch_size=16,
                        s2s_means=None, s2s_stds=None,
                        date_to_s2s_lag=None, s2s_max_lag=3,
                        s2s_full_cache=None):
    """
    Compute ranking metrics on a random sample of validation windows.

    Samples *n_sample_wins* windows from *val_wins*, runs patch-level
    inference, aggregates across lead days
      • prob  → mean  (overall fire risk across the 14–46 day window)
      • label → max   (any lead day with fire = positive pixel)
    then computes all metrics globally across all sampled pixels.

    Metrics returned (dict):
        lift_k      : Precision@K / baseline_fire_rate
        precision_k : tp / K
        recall_k    : tp / n_fires
        csi_k       : tp / (tp + fp + fn)  Critical Success Index
        ets_k       : (tp - tp_random) / (tp + fp + fn - tp_random)  Equitable Threat Score
        pr_auc      : Area under precision-recall curve (sklearn)
        n_fire      : number of fire pixels in sample
        baseline    : n_fire / n_total
    """
    from sklearn.metrics import average_precision_score

    model.eval()
    rng = np.random.default_rng(0)   # fixed seed → same sample every epoch

    if len(val_wins) > n_sample_wins:
        idx         = rng.choice(len(val_wins), size=n_sample_wins, replace=False)
        sample_idxs = sorted(idx)
        sample_wins = [val_wins[i] for i in sample_idxs]
        sample_dates = ([val_win_dates[i] for i in sample_idxs]
                        if val_win_dates is not None else [None] * len(sample_wins))
    else:
        sample_wins  = val_wins
        sample_dates = (val_win_dates if val_win_dates is not None
                        else [None] * len(val_wins))

    all_probs  = []
    all_labels = []

    _dec_dim = dec_dim or meteo_patched.shape[2]

    with torch.no_grad():
        for win_idx, (hs, he, ts, te) in enumerate(sample_wins):
            win_date = sample_dates[win_idx] if sample_dates else None
            prob_chunks = []
            for cs in range(0, n_patches, chunk):
                ce = min(cs + chunk, n_patches)
                xb_enc = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[cs:ce, hs:he, :].astype(np.float32)
                    )
                ).to(device)
                if decoder_mode == "oracle":
                    xb_dec = torch.from_numpy(
                        np.ascontiguousarray(
                            meteo_patched[cs:ce, ts:te, :].astype(np.float32)
                        )
                    ).to(device)
                elif decoder_mode == "s2s":
                    if s2s_full_cache is not None:
                        dec_list = [
                            _make_dec_s2s_full(
                                s2s_full_cache, date_to_s2s_idx,
                                win_date, cs + pi,
                                te - ts, _dec_dim,
                            ).astype(np.float32)
                            for pi in range(ce - cs)
                        ]
                    else:
                        dec_list = [
                            _make_dec_s2s(
                                s2s_cache, date_to_s2s_idx,
                                win_date, cs + pi,
                                te - ts, _dec_dim, patch_size,
                                s2s_means=s2s_means, s2s_stds=s2s_stds,
                                date_to_s2s_lag=date_to_s2s_lag,
                                s2s_max_lag=s2s_max_lag,
                            ).astype(np.float32)
                            for pi in range(ce - cs)
                        ]
                    xb_dec = torch.from_numpy(
                        np.stack(dec_list, axis=0)   # (chunk, dec_days, dec_dim)
                    ).to(device)
                else:
                    xb_enc_np = meteo_patched[cs:ce, hs:he, :].astype(np.float32)
                    dec_list = [
                        _make_dec_ablation(decoder_mode, xb_enc_np[i], te - ts, _dec_dim)
                        for i in range(ce - cs)
                    ]
                    xb_dec = torch.from_numpy(
                        np.stack(dec_list, axis=0)   # (chunk, dec_days, dec_dim)
                    ).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    logits = model(xb_enc, xb_dec)  # (chunk, dec_days, P²)
                prob_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())

            probs  = np.concatenate(prob_chunks, axis=0)  # (n_patches, dec_days, P²)
            labels = fire_patched[ts:te, :, :]            # (dec_days,  n_patches, P²)

            # Aggregate across lead days
            prob_agg  = probs.mean(axis=1)           # (n_patches, P²)
            label_agg = labels.max(axis=0)           # (n_patches, P²)  uint8

            all_probs.append(prob_agg.reshape(-1))
            all_labels.append(label_agg.reshape(-1).astype(np.float32))

    all_probs  = np.concatenate(all_probs)   # (N_total,)
    all_labels = np.concatenate(all_labels)  # (N_total,)

    n_total = len(all_probs)
    n_fire  = int(all_labels.sum())

    if n_fire == 0:
        return {"lift_k": 0.0, "precision_k": 0.0, "recall_k": 0.0,
                "csi_k": 0.0, "ets_k": 0.0, "pr_auc": 0.0,
                "n_fire": 0, "baseline": 0.0}

    k_eff       = min(k, n_total)
    top_idx     = np.argpartition(all_probs, -k_eff)[-k_eff:]
    tp          = float(all_labels[top_idx].sum())
    fp          = k_eff - tp
    fn          = n_fire - tp
    baseline    = n_fire / n_total

    precision_k = tp / k_eff
    recall_k    = tp / n_fire
    lift_k      = precision_k / baseline if baseline > 0 else 0.0
    csi_k       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    tp_random   = k_eff * baseline          # expected tp by random selection
    denom_ets   = tp + fp + fn - tp_random
    ets_k       = (tp - tp_random) / denom_ets if denom_ets > 0 else 0.0
    pr_auc      = float(average_precision_score(all_labels, all_probs))

    return {
        "lift_k":      lift_k,
        "precision_k": precision_k,
        "recall_k":    recall_k,
        "csi_k":       csi_k,
        "ets_k":       ets_k,
        "pr_auc":      pr_auc,
        "n_fire":      n_fire,
        "baseline":    baseline,
    }


# ------------------------------------------------------------------ #
# Forecast-only helper
# ------------------------------------------------------------------ #

def _run_forecast_only(args, cfg, fwi_dir, obs_root, ffmc_dir, dmc_dir, dc_dir,
                       bui_dir, fire_clim_path, output_dir, ckpt_dir):
    """
    Load the best checkpoint and generate forecast TIFs for --forecast_years.
    Decoder input matches the mode used during training (oracle / s2s / ablation).
    """
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {best_ckpt}\n"
            "Run training first (without --forecast_only)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FORECAST ONLY] Loading checkpoint: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)

    saved_args   = ckpt["args"]
    P            = saved_args["patch_size"]
    in_days      = saved_args["in_days"]
    lead_start   = saved_args["lead_start"]
    lead_end     = saved_args["lead_end"]
    decoder_days = lead_end - lead_start + 1
    meteo_means  = ckpt["meteo_means"]
    meteo_stds   = ckpt["meteo_stds"]
    patch_dim_enc = ckpt["patch_dim_enc"]
    patch_dim_dec = ckpt["patch_dim_dec"]
    patch_dim_out = ckpt["patch_dim_out"]
    s2s_means    = ckpt.get("s2s_means", None)   # (6,) float32 or None
    s2s_stds     = ckpt.get("s2s_stds", None)    # (6,) float32 or None

    n_ch = patch_dim_enc // (P * P)   # recover n_channels from enc_dim
    print(f"  n_channels={n_ch}  channels: {CHANNEL_NAMES[:n_ch]}")

    model = S2SHotspotTransformer(
        patch_dim_enc=patch_dim_enc,
        patch_dim_dec=patch_dim_dec,
        patch_dim_out=patch_dim_out,
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_encoder_layers=saved_args["enc_layers"],
        num_decoder_layers=saved_args["dec_layers"],
        dim_feedforward=saved_args["d_model"] * 4,
        dropout=saved_args.get("dropout", 0.1),
        encoder_days=in_days,
        decoder_days=decoder_days,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Device={device}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  in_days={in_days}  lead={lead_start}–{lead_end}  patch_size={P}")

    # Build full file indices
    fwi_dict  = {}
    for p in sorted(glob.glob(os.path.join(fwi_dir, "*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            fwi_dict[d] = p
    d2m_dict  = _build_file_dict(obs_root, "2d")
    t2m_dict  = _build_file_dict(obs_root, "2t")
    ffmc_dict = _build_flat_file_dict(ffmc_dir, "ffmc")
    dmc_dict  = _build_flat_file_dict(dmc_dir,  "dmc")
    dc_dict   = _build_flat_file_dict(dc_dir,   "dc")
    bui_dict  = _build_flat_file_dict(bui_dir,  "bui")

    first_fwi = sorted(fwi_dict.values())[0]
    with rasterio.open(first_fwi) as src:
        profile = src.profile
        H, W = profile["height"], profile["width"]
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    # Load static fire-climatology map
    fire_clim = _load_fire_clim(fire_clim_path, H, W)

    if args.forecast_years:
        years = [int(y.strip()) for y in args.forecast_years.split(",")]
    else:
        years = [int(args.pred_start.split("-")[0])]
    print(f"  Forecast years: {years}")

    def _clean(stack):
        stack = clean_nodata(stack.astype(np.float32))
        fill  = float(np.nanmean(stack))
        if not np.isfinite(fill):
            fill = 0.0
        return np.nan_to_num(stack, nan=fill, posinf=fill, neginf=fill)

    for year in years:
        print(f"\n{'='*60}\n  Year {year}: May 1 – Oct 31\n{'='*60}")
        pred_start   = date(year, 5, 1)
        pred_end     = date(year, 10, 31)
        data_start   = pred_start - timedelta(days=in_days + 5)
        required_end = pred_end   + timedelta(days=lead_end + 5)

        fwi_p, t2m_p, d2m_p  = [], [], []
        ffmc_p, dmc_p, dc_p, bui_p = [], [], [], []
        dates_y = []
        cur = data_start
        while cur <= required_end:
            if (cur in fwi_dict and cur in t2m_dict and cur in d2m_dict
                    and cur in ffmc_dict and cur in dmc_dict
                    and cur in dc_dict   and cur in bui_dict):
                fwi_p.append(fwi_dict[cur]);   t2m_p.append(t2m_dict[cur])
                d2m_p.append(d2m_dict[cur]);   ffmc_p.append(ffmc_dict[cur])
                dmc_p.append(dmc_dict[cur]);   dc_p.append(dc_dict[cur])
                bui_p.append(bui_dict[cur]);   dates_y.append(cur)
            cur += timedelta(days=1)

        T_y = len(dates_y)
        if T_y < in_days + lead_end + 1:
            print(f"  Only {T_y} aligned days. Skipping {year}.")
            continue
        print(f"  Aligned: {T_y} days  ({dates_y[0]} → {dates_y[-1]})")

        fwi_s  = _clean(read_singleband_stack(fwi_p))
        t2m_s  = _clean(read_singleband_stack(t2m_p))
        d2m_s  = _clean(read_singleband_stack(d2m_p))
        ffmc_s = _clean(read_singleband_stack(ffmc_p))
        dmc_s  = _clean(read_singleband_stack(dmc_p))
        dc_s   = _clean(read_singleband_stack(dc_p))
        bui_s  = _clean(read_singleband_stack(bui_p))

        meteo_y = np.empty((T_y, H, W, n_ch), dtype=np.float32)
        meteo_y[..., 0] = fwi_s;  del fwi_s
        meteo_y[..., 1] = t2m_s;  del t2m_s
        meteo_y[..., 2] = d2m_s;  del d2m_s
        meteo_y[..., 3] = ffmc_s; del ffmc_s
        meteo_y[..., 4] = dmc_s;  del dmc_s
        meteo_y[..., 5] = dc_s;   del dc_s
        meteo_y[..., 6] = bui_s;  del bui_s
        if n_ch > 7:
            meteo_y[..., 7] = fire_clim[np.newaxis, ...]  # broadcast

        meteo_y -= meteo_means
        meteo_y /= meteo_stds
        np.clip(meteo_y, -10.0, 10.0, out=meteo_y)

        date_to_idx_y = {d: i for i, d in enumerate(dates_y)}
        pred_dates_y  = [pred_start + timedelta(days=k)
                         for k in range((pred_end - pred_start).days + 1)]

        n_done = 0
        for base_date in pred_dates_y:
            if base_date not in date_to_idx_y:
                continue
            base_idx = date_to_idx_y[base_date]
            if base_idx < in_days or base_idx + lead_end + 1 > T_y:
                continue

            enc_hist = meteo_y[base_idx - in_days: base_idx]
            enc_patches, pred_hw, pred_grid = patchify(enc_hist, P)
            _decoder_mode = saved_args.get("decoder", "oracle")
            _dec_dim_saved = patch_dim_dec

            if _decoder_mode == "oracle":
                dec_fut  = meteo_y[base_idx + lead_start: base_idx + lead_end + 1]
                dec_patches, _, _ = patchify(dec_fut, P)
            else:
                dec_patches = None   # built per-chunk below

            chunk, n_p = args.pred_batch_size, enc_patches.shape[0]
            prob_list = []
            with torch.no_grad():
                for cs in range(0, n_p, chunk):
                    ce  = min(cs + chunk, n_p)
                    xb_enc = torch.from_numpy(enc_patches[cs:ce].copy()).float().to(device)
                    if _decoder_mode == "oracle":
                        xb_dec = torch.from_numpy(dec_patches[cs:ce].copy()).float().to(device)
                    elif _decoder_mode in ("zeros", "random", "climatology"):
                        enc_np = enc_patches[cs:ce]
                        dec_list = [
                            _make_dec_ablation(_decoder_mode, enc_np[i],
                                               decoder_days, _dec_dim_saved)
                            for i in range(ce - cs)
                        ]
                        xb_dec = torch.from_numpy(
                            np.stack(dec_list, axis=0).astype(np.float32)
                        ).to(device)
                    elif _decoder_mode == "s2s":
                        # Load full-patch S2S cache for inference
                        _s2s_fc_path = args.s2s_full_cache or saved_args.get("s2s_full_cache")
                        if _s2s_fc_path and os.path.exists(_s2s_fc_path):
                            _fc_dates_file = _s2s_fc_path + ".dates.npy"
                            _fc_dates = np.load(_fc_dates_file, allow_pickle=True)
                            _n_fc_dates = len(_fc_dates)
                            _fc_cache = np.memmap(_s2s_fc_path, dtype='float16', mode='r',
                                                  shape=(_n_fc_dates, n_p, 32, _dec_dim_saved))
                            _fc_idx_map, _, _ = _expand_s2s_date_mapping(
                                _fc_dates, [base_date], max_lag_days=3)
                            _bd_str = str(base_date)
                            dec_list = [
                                _make_dec_s2s_full(
                                    _fc_cache, _fc_idx_map,
                                    _bd_str, cs + pi,
                                    decoder_days, _dec_dim_saved,
                                ).astype(np.float32)
                                for pi in range(ce - cs)
                            ]
                            del _fc_cache
                        else:
                            raise FileNotFoundError(
                                f"--forecast_only with --decoder s2s requires "
                                f"--s2s_full_cache. Checkpoint s2s_full_cache path: "
                                f"{saved_args.get('s2s_full_cache', 'not saved')}")
                        xb_dec = torch.from_numpy(
                            np.stack(dec_list, axis=0).astype(np.float32)
                        ).to(device)
                    else:
                        raise NotImplementedError(
                            f"--forecast_only with --decoder {_decoder_mode} "
                            "is not supported."
                        )
                    logits = model(xb_enc, xb_dec)
                    prob_list.append(torch.sigmoid(logits).cpu().numpy())
            probs = np.concatenate(prob_list, axis=0)

            base_str = base_date.strftime("%Y%m%d")
            day_out  = os.path.join(output_dir, base_str)
            os.makedirs(day_out, exist_ok=True)

            for li, lead in enumerate(range(lead_start, lead_end + 1)):
                target_date     = base_date + timedelta(days=lead)
                target_date_str = target_date.strftime("%Y%m%d")
                out_path = os.path.join(
                    day_out, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
                )
                prob_patches_lead = probs[:, li, :]
                prob_vol = depatchify(
                    prob_patches_lead[:, np.newaxis, :],
                    pred_grid, P, pred_hw, num_channels=1
                )
                prob_map = prob_vol[0] if prob_vol.ndim == 3 else prob_vol
                if prob_map.shape != (H, W):
                    full = np.zeros((H, W), dtype=np.float32)
                    full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                    prob_map = full
                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(prob_map.astype(np.float32), 1)

            n_done += 1
            if n_done % 20 == 0 or base_date == pred_dates_y[-1]:
                print(f"  [{n_done}/{len(pred_dates_y)}] {base_date} → {decoder_days} tifs")

        del meteo_y
        print(f"  Year {year}: {n_done} base dates, {n_done * decoder_days} tifs")

    print("\n" + "=" * 70)
    print("FORECAST-ONLY COMPLETE")
    print(f"  Output: {output_dir}")
    print("=" * 70)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    run_started_at  = time.time()
    run_started_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"

    ap = argparse.ArgumentParser(
        description="Train S2S Hotspot Transformer V2 [8 channels, Lead 14–46 Days]"
    )
    add_config_argument(ap)
    ap.add_argument("--run_name", type=str, default="s2s_hotspot_cwfis_v2",
                    help="Run name — determines checkpoint and output subdirectory. "
                         "Use different names for parallel experiments.")

    # Data
    ap.add_argument("--data_start",   type=str,   default="2018-05-01",
                    help="First date for data loading.")
    ap.add_argument("--pred_start",   type=str,   default="2022-05-01",
                    help="First prediction date; also train/val split boundary.")
    ap.add_argument("--pred_end",     type=str,   default="2024-10-31",
                    help="Last prediction date (inclusive).")
    ap.add_argument("--in_days",      type=int,   default=7,
                    help="Encoder history days (default=7).")
    ap.add_argument("--lead_start",   type=int,   default=14,
                    help="First forecast lead time in days (default=14).")
    ap.add_argument("--lead_end",     type=int,   default=46,
                    help="Last forecast lead time in days (default=46).")

    # V2: fire climatology input
    ap.add_argument("--fire_climatology_tif", type=str, default=None,
                    help="Path to static fire-frequency map (from make_fire_climatology.py). "
                         "Defaults to 'fire_climatology_tif' config key. "
                         "If not found, Channel 7 is filled with zeros (with a warning).")

    # Spatial dilation
    ap.add_argument("--dilate_radius", type=int,  default=14,
                    help="Hotspot label dilation radius in pixels (default=14).")

    # Model
    ap.add_argument("--patch_size",   type=int,   default=16)
    ap.add_argument("--d_model",      type=int,   default=256)
    ap.add_argument("--nhead",        type=int,   default=8)
    ap.add_argument("--enc_layers",   type=int,   default=4)
    ap.add_argument("--dec_layers",   type=int,   default=4)
    ap.add_argument("--dropout",      type=float, default=0.1,
                    help="Dropout rate for Transformer + embedding layers (default: 0.1)")
    ap.add_argument("--weight_decay", type=float, default=0.01,
                    help="AdamW weight decay (L2 reg). Default: 0.01. Try 0.05 for more reg.")
    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Label smoothing for BCE (0=none, 0.05=light)")
    ap.add_argument("--neg_buffer",   type=int,   default=0,
                    help="Exclude negative patches within this many patches of any positive. "
                         "0=no buffer (current). 2=exclude ~32km around fires.")

    # Training
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--val_max_batches", type=int, default=500,
                    help="Max val batches per epoch (0=full). Default 500 (~30s).")
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--lr_min",       type=float, default=1e-6,
                    help="Minimum LR for CosineAnnealingLR (default: 1e-6). "
                         "Set equal to --lr to disable the scheduler.")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--neg_ratio",    type=float, default=20.0)
    ap.add_argument("--pos_weight_cap", type=float, default=10.0)
    ap.add_argument("--max_pos_pairs", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default="outputs/cache",
                    help="Directory to cache the dilated fire_stack (.npy). "
                         "Set to '' to disable caching.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Force rebuild all caches (fire_stack, meteo memmap, fire_patched).")
    ap.add_argument("--chunk_patches", type=int, default=200,
                    help="Patches per chunk during tf→pf transpose (default: 200). "
                         "Larger values (e.g. 8000) are much faster on network filesystems "
                         "but use more RAM: peak = chunk_patches × T × enc_dim × 2 bytes. "
                         "8000 chunks use ~63 GB RAM for T=2427.")
    ap.add_argument("--load_to_ram", action="store_true",
                    help="Copy meteo_patched memmap into RAM after loading (needs ~240GB RAM). "
                         "Eliminates disk IO bottleneck during training.")
    ap.add_argument("--load_train_to_ram", action="store_true",
                    help="Copy only training-period time steps into RAM (~150GB). "
                         "Val still reads from disk. Use with --fire_season_only to reduce to ~90GB.")
    ap.add_argument("--fire_season_only", action="store_true",
                    help="With --load_train_to_ram: restrict RAM copy to fire-season months only. "
                         "Also filters training windows to those whose encoder+decoder fall entirely "
                         "within fire-season months. Reduces RAM to ~90GB (fits in 1-GPU 188GB).")
    ap.add_argument("--fire_season_months", type=str, default="4,5,6,7,8,9,10",
                    help="Comma-separated months for --fire_season_only (default: 4,5,6,7,8,9,10).")
    ap.add_argument("--load_val_to_ram", action="store_true",
                    help="Copy only validation-period time steps into RAM (~40-50GB). "
                         "Eliminates disk IO during validation. Safe to combine with "
                         "--load_train_to_ram --fire_season_only.")
    ap.add_argument("--skip_val", action="store_true",
                    help="Skip validation entirely during training. A checkpoint is saved "
                         "after every epoch. Use --evaluate separately after training.")
    ap.add_argument("--skip_forecast", action="store_true",
                    help="Skip Step 10 (forecast GeoTIFF generation) after training. "
                         "Saves ~4 hours of disk IO. Run --forecast_only separately when needed.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from the latest epoch checkpoint in ckpt_dir. "
                         "Loads model, optimizer, scheduler and AMP scaler states. "
                         "If no checkpoint exists, starts from scratch.")
    ap.add_argument("--prep_only", action="store_true",
                    help="Build all data caches (meteo memmap, fire patches) then exit "
                         "without training. Use with --cache_dir to persist the memmap to "
                         "disk. Ideal for running on a CPU-only node before submitting a "
                         "GPU training job.")
    ap.add_argument("--eval_epochs", action="store_true",
                    help="Evaluate all epoch_0N.pt checkpoints in ckpt_dir on the val set "
                         "using Lift@K. Skips training. Runs after data loading (cache hit "
                         "is fast). Reports a comparison table and identifies the best epoch.")
    ap.add_argument("--eval_n_windows", type=int, default=20,
                    help="Number of val windows to sample for --eval_epochs (default: 20).")
    ap.add_argument("--mem_limit_pct", type=float, default=90.0,
                    help="Stop training when system RAM exceeds this %% (default=90). "
                         "Requires psutil (pip install psutil). Set to 0 to disable.")

    # Forecast
    ap.add_argument("--pred_batch_size", type=int, default=256)
    ap.add_argument("--forecast_only", action="store_true",
                    help="Skip training — load best checkpoint and generate forecast tifs.")
    ap.add_argument("--forecast_years", type=str, default=None,
                    help="Comma-separated years for --forecast_only, e.g. '2023,2024'.")

    # Val Lift@K checkpoint selection
    ap.add_argument("--val_lift_k", type=int, default=5000,
                    help="K for val Lift@K checkpoint selection (default=5000). "
                         "Best checkpoint = epoch with highest val Lift@K.")
    ap.add_argument("--val_lift_sample_wins", type=int, default=20,
                    help="Number of val windows sampled per epoch for Lift@K "
                         "computation (default=20). More = slower but more stable.")

    # Decoder mode
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable automatic mixed precision (AMP). "
                         "AMP is enabled by default on CUDA and uses float16 for the "
                         "forward/backward pass, typically giving 2-4× speedup on A100.")
    ap.add_argument("--decoder", type=str, default="oracle",
                    choices=["oracle", "zeros", "random", "climatology", "s2s", "s2s_legacy"],
                    help="Decoder input mode:\n"
                         "  oracle      — future ERA5 obs (default, highest accuracy, 'cheating');\n"
                         "  zeros       — all zeros (ablation);\n"
                         "  random      — i.i.d. standard normal noise (cleanest ablation);\n"
                         "  climatology — encoder-period mean repeated across decoder days;\n"
                         "  s2s         — S2S full-patch (Oracle-format, requires --s2s_full_cache);\n"
                         "  s2s_legacy  — S2S patch-mean (dec_dim=9, requires --s2s_cache).")
    ap.add_argument("--s2s_dir", type=str, default=None,
                    help="Path to processed S2S TIF directory (for --decoder s2s). "
                         "Defaults to 's2s_dir' key in config.")
    ap.add_argument("--s2s_cache", type=str, default=None,
                    help="Path to pre-built S2S decoder patch-mean cache (.dat file). "
                         "Required when --decoder s2s. "
                         "Build with: python -m src.data_ops.processing.build_s2s_decoder_cache")
    ap.add_argument("--s2s_full_cache", type=str, default=None,
                    help="Path to full-patch S2S decoder cache (.dat file, shape "
                         "(n_dates, n_patches, 32, P^2*8)). When provided with "
                         "--decoder s2s, uses Oracle-format patches (dec_dim=2048) "
                         "instead of patch-mean format (dec_dim=9). "
                         "Build with: python -m src.data_ops.processing.build_s2s_full_patch_cache")
    ap.add_argument("--s2s_max_issue_lag", type=int, default=3,
                    help="For --decoder s2s, allow base dates to reuse the most recent "
                         "available S2S issue date up to this many days back. "
                         "Default 3 captures sparse issue schedules without falling too stale.")

    # Training verbosity
    ap.add_argument("--log_interval", type=int, default=500,
                    help="Print intra-epoch progress every N batches (default=500). "
                         "Set to 0 to disable mid-epoch output.")

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------
    # Config & paths
    # ----------------------------------------------------------------
    cfg         = load_config(args.config)
    fwi_dir     = get_path(cfg, "fwi_dir")
    paths_cfg   = cfg.get("paths", {})
    obs_root    = get_path(cfg, "observation_dir") if "observation_dir" in paths_cfg \
                  else get_path(cfg, "ecmwf_dir")
    ffmc_dir    = get_path(cfg, "ffmc_dir")
    dmc_dir     = get_path(cfg, "dmc_dir")
    dc_dir      = get_path(cfg, "dc_dir")
    bui_dir     = get_path(cfg, "bui_dir")
    hotspot_csv = get_path(cfg, "hotspot_csv")
    output_dir  = os.path.join(get_path(cfg, "output_dir"),
                               f"{args.run_name}_fire_prob")
    ckpt_dir    = os.path.join(get_path(cfg, "checkpoint_dir"),
                               args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    # Resolve fire_climatology_tif path
    fire_clim_path = args.fire_climatology_tif
    if fire_clim_path is None:
        fire_clim_path = paths_cfg.get("fire_climatology_tif", None)
        if fire_clim_path is None and "fire_climatology_tif" in paths_cfg:
            fire_clim_path = get_path(cfg, "fire_climatology_tif")

    if args.forecast_only:
        _run_forecast_only(
            args, cfg, fwi_dir, obs_root, ffmc_dir, dmc_dir, dc_dir, bui_dir,
            fire_clim_path, output_dir, ckpt_dir,
        )
        return

    run_stamp     = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_meta_path = os.path.join(ckpt_dir, f"run_{run_stamp}.json")
    run_meta = {
        "run_started_at_utc": run_started_iso,
        "label_source":       "CWFIS satellite hotspots (VIIRS-M)",
        "model":              "S2SHotspotTransformer V2",
        "decoder_mode":       args.decoder,
        "n_channels":         N_CHANNELS,
        "channels":           CHANNEL_NAMES,
        "sampling":           f"mixed — pos + {args.neg_ratio}x neg patches, precomputed",
        "cli_args": vars(args),
        "status": "running",
    }

    def _flush():
        if run_meta.get("status") == "running":
            run_meta["status"] = "failed_or_interrupted"
            run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
        try:
            with open(run_meta_path, "w") as f:
                json.dump(run_meta, f, indent=2)
        except Exception:
            pass
    atexit.register(_flush)

    def _date(s):
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))

    data_start_date = _date(args.data_start)
    pred_start_date = _date(args.pred_start)
    pred_end_date   = _date(args.pred_end)
    in_days         = args.in_days
    lead_start      = args.lead_start
    lead_end        = args.lead_end
    # S2S forecasts only cover lead days 14..45 (32 days).
    # Clip lead_end to 45 to avoid a systematically zero-filled last decoder day.
    if args.decoder == "s2s" and lead_end > 45:
        print(f"  [S2S] Clipping lead_end from {lead_end} to 45 "
              f"(S2S data available for leads {lead_start}..45)")
        lead_end = 45
    decoder_days    = lead_end - lead_start + 1

    _dec_label = {
        "oracle":      "ERA5 future obs (oracle)",
        "zeros":       "zeros (ablation)",
        "random":      "random noise (ablation)",
        "climatology": "encoder-period mean (climatology ablation)",
        "s2s":         f"ECMWF S2S forecast (6 weather + age/fallback/missing, dec_dim={S2S_DEC_DIM})",
    }.get(args.decoder, args.decoder)

    print("\n" + "=" * 70)
    print("S2S HOTSPOT TRANSFORMER V2  [8 channels]")
    print("=" * 70)
    print(f"  Channels          : {N_CHANNELS} — {', '.join(CHANNEL_NAMES)}")
    print(f"  decoder           : {_dec_label}")
    print(f"  data_start        : {data_start_date}")
    print(f"  pred_start        : {pred_start_date}  (train/val split boundary)")
    print(f"  pred_end          : {pred_end_date}")
    print(f"  in_days           : {in_days}  (encoder history)")
    print(f"  lead_start–end    : {lead_start}–{lead_end}  (decoder_days={decoder_days})")
    print(f"  patch_size        : {args.patch_size}")
    print(f"  dilate_radius     : {args.dilate_radius} px  "
          f"(≈{args.dilate_radius * 2} km)")
    print(f"  neg_ratio         : {args.neg_ratio}")
    print(f"  d_model / nhead   : {args.d_model} / {args.nhead}")
    print(f"  epochs / batch    : {args.epochs} / {args.batch_size}  lr={args.lr}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # STEP 1  Build file indices
    # ----------------------------------------------------------------
    print("\n[STEP 1] Building file index (8 channels)...")

    fwi_dict  = {}
    for p in sorted(glob.glob(os.path.join(fwi_dir, "*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            fwi_dict[d] = p
    d2m_dict  = _build_file_dict(obs_root, "2d")
    t2m_dict  = _build_file_dict(obs_root, "2t")
    ffmc_dict = _build_flat_file_dict(ffmc_dir, "ffmc")
    dmc_dict  = _build_flat_file_dict(dmc_dir,  "dmc")
    dc_dict   = _build_flat_file_dict(dc_dir,   "dc")
    bui_dict  = _build_flat_file_dict(bui_dir,  "bui")

    if not fwi_dict:
        raise RuntimeError(f"No FWI .tif files found in {fwi_dir}")
    if not d2m_dict:
        raise RuntimeError(f"No 2d .tif files found under {obs_root}")
    if not t2m_dict:
        raise RuntimeError(f"No 2t .tif files found under {obs_root}")
    if not ffmc_dict:
        raise RuntimeError(f"No ffmc_*.tif files found in {ffmc_dir}")
    if not dmc_dict:
        raise RuntimeError(f"No dmc_*.tif files found in {dmc_dir}")
    if not dc_dict:
        raise RuntimeError(f"No dc_*.tif files found in {dc_dir}")
    if not bui_dict:
        raise RuntimeError(f"No bui_*.tif files found in {bui_dir}")

    print(f"  FWI : {len(fwi_dict):,} days  2t: {len(t2m_dict):,}  "
          f"2d: {len(d2m_dict):,}")
    print(f"  FFMC: {len(ffmc_dict):,} days  DMC: {len(dmc_dict):,}  "
          f"DC: {len(dc_dict):,}  BUI: {len(bui_dict):,}")
    print(f"  fire_climatology_tif: {fire_clim_path}")

    # ----------------------------------------------------------------
    # STEP 2  Align dates (require all 7 daily channels to be present)
    # ----------------------------------------------------------------
    print("\n[STEP 2] Aligning dates across all 7 daily channels...")

    required_end = pred_end_date + timedelta(days=lead_end + 5)

    fwi_paths, t2m_paths, d2m_paths = [], [], []
    ffmc_paths, dmc_paths, dc_paths, bui_paths = [], [], [], []
    aligned_dates = []
    cur = data_start_date
    while cur <= required_end:
        if (cur in fwi_dict and cur in t2m_dict and cur in d2m_dict
                and cur in ffmc_dict and cur in dmc_dict
                and cur in dc_dict   and cur in bui_dict):
            fwi_paths.append(fwi_dict[cur]);   t2m_paths.append(t2m_dict[cur])
            d2m_paths.append(d2m_dict[cur]);   ffmc_paths.append(ffmc_dict[cur])
            dmc_paths.append(dmc_dict[cur]);   dc_paths.append(dc_dict[cur])
            bui_paths.append(bui_dict[cur]);   aligned_dates.append(cur)
        cur += timedelta(days=1)

    min_needed = in_days + lead_end + 1
    if len(aligned_dates) < min_needed:
        raise RuntimeError(
            f"Only {len(aligned_dates)} aligned days, need >= {min_needed}. "
            "Check that all 7 variable directories cover the same date range."
        )
    print(f"  Aligned dates: {len(aligned_dates)}  "
          f"({aligned_dates[0]} → {aligned_dates[-1]})")
    run_meta["aligned_days"] = len(aligned_dates)

    # ----------------------------------------------------------------
    # STEP 3  Streaming per-channel stats (no full-stack load)
    # ----------------------------------------------------------------
    # Get grid dimensions from first FWI TIF header — no full array load
    with rasterio.open(fwi_paths[0]) as src:
        profile = src.profile
        H, W    = src.height, src.width
    T = len(aligned_dates)
    print(f"\n[STEP 3] Grid: T={T}  H={H}  W={W}")

    # Load static fire-climatology (Channel 7) — small single file (~25 MB)
    print(f"  Loading static fire-climatology map (Channel 7)...")
    fire_clim = _load_fire_clim(fire_clim_path, H, W)

    # Compute train/val split index from date list (no data loading)
    train_end_idx = next(
        (i for i, d in enumerate(aligned_dates) if d >= pred_start_date), None
    )
    if train_end_idx is None:
        raise RuntimeError(
            f"pred_start={pred_start_date} is beyond all aligned dates. "
            "Check --pred_start and --data_start."
        )
    print(f"  Train/val split: {aligned_dates[0]} → {aligned_dates[train_end_idx-1]} "
          f"({train_end_idx} train days) | "
          f"{aligned_dates[train_end_idx]} → {aligned_dates[-1]} "
          f"({T - train_end_idx} val days)")

    # Check if stats are already cached on disk — skip expensive streaming if so.
    # Stats only depend on the training split, not on T or date range, so we
    # first look for a canonical name, then fall back to any *_stats.npy match.
    _early_stats_path = None
    if args.cache_dir:
        # Canonical name (T-independent)
        _canonical_stats = os.path.join(args.cache_dir,
                                        f"meteo_p{args.patch_size}_C{N_CHANNELS}_stats.npy")
        # Legacy name (T-dependent, from build_meteo_cache.py)
        _early_mmap_key  = (f"meteo_p{args.patch_size}_C{N_CHANNELS}_T{T}"
                            f"_{aligned_dates[0]}_{aligned_dates[-1]}_pf.dat")
        _legacy_stats = os.path.join(args.cache_dir,
                                     _early_mmap_key.replace("_pf.dat", "_stats.npy"))
        if os.path.exists(_canonical_stats):
            _early_stats_path = _canonical_stats
        elif os.path.exists(_legacy_stats):
            _early_stats_path = _legacy_stats
        else:
            # Glob for any matching stats file from build_meteo_cache
            # Prefer files whose date range starts with our data_start
            import glob as _glob
            _candidates = sorted(_glob.glob(os.path.join(
                args.cache_dir, f"meteo_p{args.patch_size}_C{N_CHANNELS}_T*_stats.npy")))
            if _candidates:
                _data_start_str = str(aligned_dates[0])
                _matching = [c for c in _candidates if _data_start_str in c]
                _early_stats_path = _matching[-1] if _matching else _candidates[-1]

    if (not args.overwrite and
            _early_stats_path and os.path.exists(_early_stats_path)):
        print(f"\n  Loading cached stats from {_early_stats_path} (skipping STEP 3 streaming)")
        _s = np.load(_early_stats_path)
        meteo_means = _s[0].astype(np.float32)
        meteo_stds  = _s[1].astype(np.float32)
        fills       = meteo_means.copy()
        _dyn_paths_all = [fwi_paths, t2m_paths, d2m_paths,
                          ffmc_paths, dmc_paths, dc_paths, bui_paths]
        for i, name in enumerate(CHANNEL_NAMES):
            print(f"  {name:12s}  mean={meteo_means[i]:8.3f}  std={meteo_stds[i]:8.3f}")
    else:
        # Stream stats from TRAINING dates only (one TIF at a time, ~25 MB/TIF)
        print(f"\n  Computing per-channel stats (streaming, no full load)...")
        _dyn_paths_all = [fwi_paths, t2m_paths, d2m_paths,
                          ffmc_paths, dmc_paths, dc_paths, bui_paths]
        ch_stats = []
        for ch_name, ch_paths in zip(CHANNEL_NAMES[:7], _dyn_paths_all):
            m, s, f = _stream_channel_stats(ch_paths[:train_end_idx])
            ch_stats.append((m, s, f))
            print(f"  {ch_name:8s}  mean={m:8.3f}  std={s:8.3f}")

        # Channel 7: fire_clim (static map — use spatial mean/std)
        fc_valid = fire_clim[(fire_clim > -1e30) & np.isfinite(fire_clim)]
        fc_mean  = float(fc_valid.mean()) if fc_valid.size else 0.0
        fc_std   = float(fc_valid.std())  if fc_valid.size else 1.0
        ch_stats.append((fc_mean, max(fc_std, 1e-6), fc_mean))
        print(f"  {'fire_clim':8s}  mean={fc_mean:8.3f}  std={fc_std:8.3f}")

        meteo_means = np.array([s[0] for s in ch_stats], dtype=np.float32)
        meteo_stds  = np.array([max(s[1], 1e-6) for s in ch_stats], dtype=np.float32)
        fills       = np.array([s[2] for s in ch_stats], dtype=np.float32)

    # ----------------------------------------------------------------
    # STEP 4  Load and rasterize CWFIS hotspots
    # ----------------------------------------------------------------
    print("\n[STEP 4] Loading CWFIS hotspot records...")
    hotspot_df = load_hotspot_data(hotspot_csv)
    print(f"  Total records : {len(hotspot_df):,}")
    print(f"  Date range    : {hotspot_df['date'].min()} to {hotspot_df['date'].max()}")
    run_meta["hotspot_records"] = int(len(hotspot_df))

    # -- Spatial dilation --
    r = args.dilate_radius if args.dilate_radius > 0 else 0
    if r > 0 and args.cache_dir:
        cache_key  = (f"fire_dilated_r{r}"
                      f"_{aligned_dates[0]}_{aligned_dates[-1]}"
                      f"_{H}x{W}.npy")
        cache_path = os.path.join(args.cache_dir, cache_key)
        # Fuzzy match: if exact file not found, look for one with same
        # data_start, same H×W, and T >= our T (can slice first T frames)
        if not os.path.exists(cache_path):
            import glob as _glob
            _ds = str(aligned_dates[0])
            _fire_candidates = sorted(_glob.glob(os.path.join(
                args.cache_dir, f"fire_dilated_r{r}_{_ds}_*_{H}x{W}.npy")))
            for _fc in _fire_candidates:
                # Check file is large enough (T frames × H × W bytes)
                _fc_size = os.path.getsize(_fc)
                _min_size = T * H * W  # uint8, so 1 byte per element
                if _fc_size >= _min_size:
                    print(f"  [cache] Exact fire_dilated not found, "
                          f"using compatible: {os.path.basename(_fc)}")
                    cache_path = _fc
                    break
    else:
        cache_path = None

    if cache_path and os.path.exists(cache_path) and not args.overwrite:
        # Fast path: dilated cache exists — skip rasterization entirely
        print(f"\n  Found cached dilated fire_stack — skipping rasterization: {cache_path}")
        t0_load = time.time()
        fire_stack = np.load(cache_path)
        if fire_stack.shape[0] > T:
            print(f"  [cache] fire_stack has {fire_stack.shape[0]} frames, "
                  f"slicing to T={T}")
            fire_stack = fire_stack[:T]
        pos_rate_dil = fire_stack.mean()
        print(f"  Loaded in {time.time()-t0_load:.0f}s  positive_rate={pos_rate_dil:.4%}")
        run_meta["dilate_radius"]         = r
        run_meta["positive_rate_dilated"] = float(pos_rate_dil)
    else:
        # Slow path: rasterize from hotspot CSV, then dilate
        fire_stack = rasterize_hotspots_batch(hotspot_df, aligned_dates, profile)
        pos_rate   = fire_stack.mean()
        print(f"  Fire stack shape     : {fire_stack.shape}")
        print(f"  Mean fire pixel rate : {pos_rate:.6%}  (raw hotspot points)")
        run_meta["positive_rate_raw"] = float(pos_rate)

        if r > 0:
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            disk   = (xx ** 2 + yy ** 2 <= r ** 2)
            print(f"\n  Dilating fire labels: radius={r} px  "
                  f"(~{r * 2} km buffer, kernel {disk.shape[0]}×{disk.shape[1]}, "
                  f"{disk.sum()} pixels/hotspot)...")
            t0_dil = time.time()
            for t in range(T):
                if fire_stack[t].any():
                    fire_stack[t] = binary_dilation(
                        fire_stack[t], structure=disk
                    ).astype(np.uint8)
                if t % 200 == 0 or t == T - 1:
                    print(f"    dilate frame {t:4d}/{T}  ({time.time()-t0_dil:.0f}s)")
            pos_rate_dil = fire_stack.mean()
            print(f"  After dilation: positive_rate={pos_rate_dil:.4%}  "
                  f"({pos_rate_dil/pos_rate:.1f}× increase)  "
                  f"({time.time()-t0_dil:.0f}s)")
            if cache_path:
                os.makedirs(args.cache_dir, exist_ok=True)
                np.save(cache_path, fire_stack)
                print(f"  Cached to: {cache_path}  "
                      f"({os.path.getsize(cache_path)/1e9:.1f} GB)")
            run_meta["dilate_radius"]         = r
            run_meta["positive_rate_dilated"] = float(pos_rate_dil)
        else:
            run_meta["dilate_radius"]         = 0
            run_meta["positive_rate_dilated"] = float(pos_rate)

    # ----------------------------------------------------------------
    # STEP 5  Log normalisation stats (computed in STEP 3 by streaming)
    # ----------------------------------------------------------------
    print(f"\n[STEP 5] Normalisation stats ({N_CHANNELS} channels — from training split):")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name:12s}  mean={meteo_means[i]:8.3f}  std={meteo_stds[i]:8.3f}")
    run_meta["norm_stats"] = {
        "channels":    CHANNEL_NAMES,
        "meteo_means": meteo_means.tolist(),
        "meteo_stds":  meteo_stds.tolist(),
    }
    np.save(os.path.join(ckpt_dir, "norm_stats.npy"),
            np.stack([meteo_means, meteo_stds]))

    # ----------------------------------------------------------------
    # STEP 6  Build meteo_patched as float16 disk memmap (streaming)
    # ----------------------------------------------------------------
    # Geometry (crop to patch-aligned boundary, matching _patchify_frame)
    P = args.patch_size
    Hc, Wc   = H - H % P, W - W % P
    nph, npw = Hc // P, Wc // P
    hw        = (Hc, Wc)
    grid      = (nph, npw)
    n_patches = nph * npw
    enc_dim   = P * P * N_CHANNELS   # patch_dim_enc
    # dec_dim depends on --decoder mode:
    #   oracle / zeros / random / climatology → same 8 channels as encoder
    #   s2s         → full-patch Oracle-format (requires --s2s_full_cache)
    #   s2s_legacy  → old 6-channel patch-mean (requires --s2s_cache)
    if args.decoder in ("oracle", "zeros", "random", "climatology"):
        dec_dim = enc_dim                # ablation modes keep same architecture
    elif args.decoder == "s2s":
        # ── Full-patch S2S cache (Oracle-format, pre-normalized) ──
        if not args.s2s_full_cache:
            raise ValueError(
                "--decoder s2s requires --s2s_full_cache <path>.\n"
                "Build with: python -m src.data_ops.processing.build_s2s_full_patch_cache\n"
                "For the old patch-mean format (dec_dim=9), use --decoder s2s_legacy.")
        dec_dim = enc_dim  # 2048 = P²×8, same as Oracle encoder
        print(f"\n[S2S decoder — full-patch] dec_dim={dec_dim}  (Oracle-format, pre-normalized)")
        s2s_full_cache_path = args.s2s_full_cache
        if not os.path.exists(s2s_full_cache_path):
            raise FileNotFoundError(f"S2S full cache not found: {s2s_full_cache_path}\n"
                                    "Run: python -m src.data_ops.processing.build_s2s_full_patch_cache")
        dates_file = s2s_full_cache_path + ".dates.npy"
        if not os.path.exists(dates_file):
            raise FileNotFoundError(f"S2S dates companion file not found: {dates_file}")
        s2s_dates = np.load(dates_file, allow_pickle=True)
        s2s_n_dates = len(s2s_dates)
        print(f"  S2S full cache dates: {s2s_n_dates}  ({s2s_dates[0]} .. {s2s_dates[-1]})")
        s2s_full_cache = np.memmap(s2s_full_cache_path, dtype='float16', mode='r',
                                   shape=(s2s_n_dates, n_patches, 32, enc_dim))
        print(f"  S2S full cache shape: {s2s_full_cache.shape}  "
              f"({os.path.getsize(s2s_full_cache_path)/1e9:.2f} GB)")
        s2s_cache = None
        s2s_means = None
        s2s_stds  = None
        date_to_s2s_idx, date_to_s2s_exact, date_to_s2s_lag = _expand_s2s_date_mapping(
            s2s_dates, aligned_dates, max_lag_days=args.s2s_max_issue_lag
        )
        _n_exact = sum(1 for d in aligned_dates if date_to_s2s_exact.get(str(d), False))
        _n_fallback = sum(
            1 for d in aligned_dates
            if str(d) in date_to_s2s_idx and not date_to_s2s_exact.get(str(d), False)
        )
        _n_miss = len(aligned_dates) - _n_exact - _n_fallback
        print(f"  S2S date mapping: exact={_n_exact}  fallback={_n_fallback}  "
              f"miss={_n_miss}  (max_lag={args.s2s_max_issue_lag}d)")
    elif args.decoder == "s2s_legacy":
        # ── Legacy: patch-mean S2S cache (dec_dim=9) ──
        s2s_full_cache = None
        dec_dim = S2S_DEC_DIM   # 9 — 6 weather + issue_age + is_fallback + is_missing
        print(f"\n[S2S decoder — legacy patch-mean] dec_dim={dec_dim}")
        s2s_cache_path = args.s2s_cache
        if not s2s_cache_path:
            raise ValueError("--decoder s2s_legacy requires --s2s_cache <path to .dat file>")
        if not os.path.exists(s2s_cache_path):
            raise FileNotFoundError(f"S2S cache not found: {s2s_cache_path}")
        dates_file = s2s_cache_path + ".dates.npy"
        if not os.path.exists(dates_file):
            raise FileNotFoundError(f"S2S dates companion file not found: {dates_file}")
        s2s_dates = np.load(dates_file, allow_pickle=True)
        s2s_n_dates = len(s2s_dates)
        print(f"  S2S cache dates: {s2s_n_dates}  ({s2s_dates[0]} .. {s2s_dates[-1]})")
        s2s_cache = np.memmap(s2s_cache_path, dtype='float16', mode='r',
                              shape=(s2s_n_dates, n_patches, 32, S2S_N_CHANNELS))
        print(f"  S2S cache shape: {s2s_cache.shape}  "
              f"({os.path.getsize(s2s_cache_path)/1e9:.2f} GB)")
        date_to_s2s_idx, date_to_s2s_exact, date_to_s2s_lag = _expand_s2s_date_mapping(
            s2s_dates, aligned_dates, max_lag_days=args.s2s_max_issue_lag
        )
        _n_exact = sum(1 for d in aligned_dates if date_to_s2s_exact.get(str(d), False))
        _n_fallback = sum(
            1 for d in aligned_dates
            if str(d) in date_to_s2s_idx and not date_to_s2s_exact.get(str(d), False)
        )
        _n_miss = len(aligned_dates) - _n_exact - _n_fallback
        print(f"  S2S date mapping: exact={_n_exact}  fallback={_n_fallback}  "
              f"miss={_n_miss}  (max_lag={args.s2s_max_issue_lag}d)")

        # ── Compute S2S per-channel normalization stats ──
        from datetime import date as _date_cls
        _pred_start = _date_cls.fromisoformat(str(args.pred_start))
        _s2s_train_rows = [
            i for i, d in enumerate(s2s_dates)
            if _date_cls.fromisoformat(str(d)) < _pred_start
        ]
        _S2S_CH_NAMES = ["2t", "2d", "tcw", "sm20", "st20", "VPD"]
        if _s2s_train_rows:
            _rng_s2s = np.random.default_rng(42)
            _sample_patches = _rng_s2s.choice(n_patches, size=min(2000, n_patches), replace=False)
            _ch_sums  = np.zeros(S2S_N_CHANNELS, dtype=np.float64)
            _ch_sqsums = np.zeros(S2S_N_CHANNELS, dtype=np.float64)
            _ch_counts = np.zeros(S2S_N_CHANNELS, dtype=np.int64)
            _ch_mins = np.full(S2S_N_CHANNELS, np.inf)
            _ch_maxs = np.full(S2S_N_CHANNELS, -np.inf)
            _n_nonzero_rows = 0
            _n_total_rows = 0
            for _row_i in _s2s_train_rows:
                _block = np.array(s2s_cache[_row_i, _sample_patches, :, :],
                                  dtype=np.float32)
                _flat = _block.reshape(-1, S2S_N_CHANNELS)
                _nz_mask = np.any(_flat != 0, axis=1)
                _n_nonzero_rows += _nz_mask.sum()
                _n_total_rows += len(_nz_mask)
                _valid = _flat[_nz_mask]
                if len(_valid) > 0:
                    _ch_sums += _valid.sum(axis=0)
                    _ch_sqsums += (_valid ** 2).sum(axis=0)
                    _ch_counts += len(_valid)
                    _ch_mins = np.minimum(_ch_mins, _valid.min(axis=0))
                    _ch_maxs = np.maximum(_ch_maxs, _valid.max(axis=0))
            s2s_means = np.zeros(S2S_N_CHANNELS, dtype=np.float32)
            s2s_stds  = np.ones(S2S_N_CHANNELS, dtype=np.float32)
            for _ch in range(S2S_N_CHANNELS):
                if _ch_counts[_ch] > 0:
                    s2s_means[_ch] = _ch_sums[_ch] / _ch_counts[_ch]
                    _var = _ch_sqsums[_ch] / _ch_counts[_ch] - s2s_means[_ch] ** 2
                    s2s_stds[_ch] = max(np.sqrt(max(_var, 0)), 1e-6)
            print(f"\n  S2S normalization stats ({len(_s2s_train_rows)} train rows, "
                  f"{len(_sample_patches)} sampled patches):")
            for _ch in range(S2S_N_CHANNELS):
                print(f"    ch{_ch} ({_S2S_CH_NAMES[_ch]:>4s}):  "
                      f"mean={s2s_means[_ch]:10.4f}  std={s2s_stds[_ch]:10.4f}")
        else:
            print("  WARNING: no S2S training-period rows found — skipping normalization")
            s2s_means = None
            s2s_stds  = None
    else:
        raise ValueError(f"Unknown --decoder: {args.decoder}")

    # Null out S2S variables for non-S2S modes
    if args.decoder not in ("s2s", "s2s_legacy"):
        s2s_cache        = None
        date_to_s2s_idx  = None
        date_to_s2s_exact = None
        date_to_s2s_lag  = None
        s2s_means        = None
        s2s_stds         = None
        s2s_full_cache   = None

    out_dim   = P * P                # patch_dim_out

    meteo_mmap_gb = T * n_patches * enc_dim * 2 / 1e9   # float16
    fire_ram_gb   = T * n_patches * out_dim     / 1e9   # uint8
    print(f"\n[STEP 6] Streaming meteo_patched → float16 memmap")
    print(f"  n_patches={n_patches}  enc_dim={enc_dim}  grid={nph}×{npw}  "
          f"crop={Hc}×{Wc}")
    print(f"  Disk needed: ~{meteo_mmap_gb:.1f} GB  "
          f"(fire_patched in RAM: ~{fire_ram_gb:.1f} GB uint8)")

    # Build or load from disk cache
    # Layout: patch-first (n_patches, T, enc_dim) — sequential reads in __getitem__
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        mmap_key   = (f"meteo_p{P}_C{N_CHANNELS}_T{T}"
                      f"_{aligned_dates[0]}_{aligned_dates[-1]}_pf.dat")
        mmap_path  = os.path.join(args.cache_dir, mmap_key)

        # Fuzzy match: if exact pf.dat not found, look for one with same
        # data_start and T >= our T (can slice the first T timesteps).
        _cache_T = T  # the T in the cache file (may differ from training T)
        if not os.path.exists(mmap_path):
            import glob as _glob
            _data_start_str = str(aligned_dates[0])
            _pf_candidates = sorted(_glob.glob(os.path.join(
                args.cache_dir,
                f"meteo_p{P}_C{N_CHANNELS}_T*_{_data_start_str}_*_pf.dat")))
            for _pf_cand in _pf_candidates:
                # Extract T from filename
                import re as _re
                _m = _re.search(r'_T(\d+)_', os.path.basename(_pf_cand))
                if _m:
                    _cand_T = int(_m.group(1))
                    if _cand_T >= T:
                        print(f"  [cache] Exact pf.dat not found (T={T}), "
                              f"using compatible cache (T={_cand_T}): "
                              f"{os.path.basename(_pf_cand)}")
                        mmap_path = _pf_cand
                        _cache_T = _cand_T
                        break

        # Stats: look for canonical, legacy, then glob (prefer same data_start)
        _canon_sp = os.path.join(args.cache_dir,
                                 f"meteo_p{P}_C{N_CHANNELS}_stats.npy")
        _legacy_sp = mmap_path.replace("_pf.dat", "_stats.npy")
        if os.path.exists(_canon_sp):
            stats_path = _canon_sp
        elif os.path.exists(_legacy_sp):
            stats_path = _legacy_sp
        else:
            import glob as _glob
            _data_start_str = str(aligned_dates[0])
            _cands = sorted(_glob.glob(os.path.join(
                args.cache_dir, f"meteo_p{P}_C{N_CHANNELS}_T*_stats.npy")))
            _matching = [c for c in _cands if _data_start_str in c]
            stats_path = _matching[-1] if _matching else (_cands[-1] if _cands else _legacy_sp)
    else:
        mmap_path  = None
        stats_path = None
        _cache_T   = T

    _stats_ok = stats_path and os.path.exists(stats_path)

    if mmap_path and os.path.exists(mmap_path) and _stats_ok:
        # ── Fast path: load existing float16 patch-first cache ───────
        print(f"  Loading cached memmap (float16 patch-first): {mmap_path}")
        if _cache_T > T:
            # Cache has more timesteps than needed — load full, slice later
            print(f"  [cache] Cache has T={_cache_T}, training needs T={T}, "
                  f"loading full cache (will slice in Dataset)")
        meteo_patched = np.memmap(mmap_path, dtype='float16', mode='r',
                                  shape=(n_patches, _cache_T, enc_dim))
        _saved_stats  = np.load(stats_path)
        meteo_means   = _saved_stats[0]
        meteo_stds    = _saved_stats[1]
        print(f"  shape={meteo_patched.shape}  "
              f"({os.path.getsize(mmap_path)/1e9:.1f} GB on disk)")
        if args.load_to_ram:
            import psutil
            ram = psutil.virtual_memory()
            needed_gb = os.path.getsize(mmap_path) / 1e9
            avail_gb  = ram.available / 1e9
            total_gb  = ram.total / 1e9
            print(f"  [--load_to_ram] RAM before copy: {avail_gb:.1f}GB available / {total_gb:.1f}GB total")
            if avail_gb < needed_gb + 20:
                print(f"  WARNING: available RAM ({avail_gb:.1f}GB) may be insufficient for {needed_gb:.1f}GB copy (+20GB buffer)")
            print(f"  [--load_to_ram] Copying meteo_patched into RAM (~{needed_gb:.1f} GB)...")
            t0 = time.time()
            meteo_patched = np.array(meteo_patched)
            ram_after = psutil.virtual_memory()
            print(f"  [--load_to_ram] Copy complete in {time.time()-t0:.0f}s")
            print(f"  [--load_to_ram] RAM after copy: {ram_after.available/1e9:.1f}GB available / {ram_after.total/1e9:.1f}GB total")
            print(f"  [--load_to_ram] meteo_patched is now type={type(meteo_patched).__name__} (in RAM: {meteo_patched.nbytes/1e9:.1f}GB)")
    else:
        # ── Build: stream day-by-day into time-first temp file,
        #          then transpose to patch-first final file ──────────
        if mmap_path:
            tf_path = mmap_path.replace("_pf.dat", "_tf.dat")
        else:
            tf_path = None

        # ── Resume path: tf.dat exists on disk but pf.dat does not ──
        # This happens when a previous job was cancelled mid-transpose.
        # We skip the expensive day-by-day streaming and go straight to
        # the transpose step using the already-complete tf.dat file.
        _tf_exists = (tf_path and os.path.exists(tf_path) and
                      _stats_ok and
                      os.path.getsize(tf_path) > 0)

        if _tf_exists:
            expected_bytes = T * n_patches * enc_dim * 2  # float16
            actual_bytes   = os.path.getsize(tf_path)
            if actual_bytes < expected_bytes * 0.99:
                print(f"  WARNING: tf.dat size {actual_bytes/1e9:.1f} GB < "
                      f"expected {expected_bytes/1e9:.1f} GB — rebuilding from scratch.")
                _tf_exists = False

        if _tf_exists:
            print(f"  Found complete time-first file ({os.path.getsize(tf_path)/1e9:.1f} GB). "
                  f"Skipping day-by-day streaming, resuming transpose.")
            _saved_stats  = np.load(stats_path)
            meteo_means   = _saved_stats[0]
            meteo_stds    = _saved_stats[1]
        else:
            if tf_path:
                print(f"  Creating time-first temp file: {tf_path}")
                meteo_tf = np.memmap(tf_path, dtype='float16', mode='w+',
                                     shape=(T, n_patches, enc_dim))
            else:
                print(f"  cache_dir disabled — using in-memory float16 array "
                      f"({meteo_mmap_gb:.1f} GB)")
                meteo_tf = np.zeros((T, n_patches, enc_dim), dtype=np.float16)

            _dyn_paths_all = [fwi_paths, t2m_paths, d2m_paths,
                              ffmc_paths, dmc_paths, dc_paths, bui_paths]
            _fallbacks = [None] * 7
            t0_mmap    = time.time()

            for t_idx in range(T):
                frame = np.empty((H, W, N_CHANNELS), dtype=np.float32)
                for ch_idx, ch_paths in enumerate(_dyn_paths_all):
                    arr = _read_tif_safe(ch_paths[t_idx], _fallbacks[ch_idx])
                    _fallbacks[ch_idx] = arr
                    arr = np.nan_to_num(arr, nan=float(fills[ch_idx]),
                                        posinf=float(fills[ch_idx]),
                                        neginf=float(fills[ch_idx]))
                    frame[..., ch_idx] = arr
                frame[..., 7] = fire_clim   # static channel
                frame -= meteo_means
                frame /= meteo_stds
                np.clip(frame, -10.0, 10.0, out=frame)
                meteo_tf[t_idx] = _patchify_frame(frame, P).astype(np.float16)

                if t_idx % 100 == 0 or t_idx == T - 1:
                    elapsed = time.time() - t0_mmap
                    eta_min = elapsed / max(t_idx, 1) * (T - t_idx) / 60
                    print(f"  day {t_idx+1:4d}/{T}  "
                          f"({elapsed:.0f}s elapsed  ~{eta_min:.0f} min left)")

            if tf_path:
                meteo_tf.flush()
                del meteo_tf   # Windows: release write handle before transpose
                gc.collect()
                np.save(stats_path, np.stack([meteo_means, meteo_stds]))
                print(f"  Saved time-first: {tf_path}  "
                      f"({os.path.getsize(tf_path)/1e9:.1f} GB)")

        if mmap_path:
            # ── Transpose to patch-first ─────────────────────────────
            print(f"\n  Transposing to patch-first layout → {mmap_path}")
            _transpose_tf_to_pf(tf_path, mmap_path, T, n_patches, enc_dim,
                                 chunk_patches=args.chunk_patches)
            print(f"  Transpose complete. Deleting temp file: {tf_path}")
            os.remove(tf_path)
            print(f"  Saved patch-first: {mmap_path}  "
                  f"({os.path.getsize(mmap_path)/1e9:.1f} GB)")
            meteo_patched = np.memmap(mmap_path, dtype='float16', mode='r',
                                      shape=(n_patches, T, enc_dim))
        else:
            # In-memory: transpose to patch-first float16
            _tmp = np.ascontiguousarray(meteo_tf.transpose(1, 0, 2))
            del meteo_tf; gc.collect()
            meteo_patched = _tmp

    # ----------------------------------------------------------------
    # STEP 7  Patchify fire labels (uint8 disk memmap)
    # ----------------------------------------------------------------
    print("\n[STEP 7] Pre-computing fire patches (uint8)...")
    fire_gb = T * n_patches * out_dim / 1e9
    print(f"  fire_patched: ({T}, {n_patches}, {out_dim})  ≈ {fire_gb:.1f} GB uint8")
    t0_fire = time.time()

    fire_cache_path = None
    if args.cache_dir:
        fire_cache_key  = (f"fire_patched_r{args.dilate_radius}"
                           f"_{aligned_dates[0]}_{aligned_dates[-1]}"
                           f"_{T}x{n_patches}x{out_dim}.dat")
        fire_cache_path = os.path.join(args.cache_dir, fire_cache_key)

    if fire_cache_path and os.path.exists(fire_cache_path) and not args.overwrite:
        fire_patched = np.memmap(fire_cache_path, dtype='uint8', mode='r',
                                 shape=(T, n_patches, out_dim))
        print(f"  Loaded cached fire_patched: {fire_cache_path}  ({time.time()-t0_fire:.0f}s)")
    else:
        if fire_cache_path:
            fire_patched = np.memmap(fire_cache_path, dtype='uint8', mode='w+',
                                     shape=(T, n_patches, out_dim))
        else:
            fire_patched = np.empty((T, n_patches, out_dim), dtype=np.uint8)
        for t_idx in range(T):
            frame_f = fire_stack[t_idx, :Hc, :Wc, np.newaxis].astype(np.float32)
            fire_patched[t_idx] = _patchify_frame(frame_f, P).astype(np.uint8)
            if t_idx % 500 == 0 or t_idx == T - 1:
                print(f"  fire frame {t_idx:4d}/{T}  ({time.time()-t0_fire:.0f}s)")
        if fire_cache_path:
            fire_patched.flush()
        print(f"  fire_patched: {fire_patched.shape}  dtype=uint8  "
              f"{fire_gb:.1f} GB  ({time.time()-t0_fire:.0f}s)")
    del fire_stack

    # ----------------------------------------------------------------
    # --prep_only: exit here — all caches built, no need for STEP 7b+
    # ----------------------------------------------------------------
    if args.prep_only:
        print("\n[--prep_only] All caches built. Exiting before training.")
        print(f"  meteo memmap : {mmap_path}  ({os.path.getsize(mmap_path)/1e9:.1f} GB)")
        if fire_cache_path:
            print(f"  fire_patched : {fire_cache_path}  ({os.path.getsize(fire_cache_path)/1e9:.1f} GB)")
        print("  Re-run without --prep_only to train using the cached files.")
        return

    # Build S2S windows
    all_windows = _build_s2s_windows(T, in_days, lead_start, lead_end)
    # Split by TARGET END date for train (no label leakage into val period)
    # and by BASE DATE for val (no gap at the start of the val period).
    # w = (hs, he, ts, te) where he = base_date_index, te = exclusive target end
    train_wins  = [w for w in all_windows
                   if aligned_dates[w[3] - 1] < pred_start_date]
    val_wins    = [w for w in all_windows
                   if aligned_dates[w[1]] >= pred_start_date]
    n_gap = len(all_windows) - len(train_wins) - len(val_wins)
    print(f"\n  S2S windows built (enc_days={in_days}, gap={lead_start-1}, "
          f"target_days={decoder_days})")
    print(f"  Total: {len(all_windows)}  train: {len(train_wins)}  "
          f"val: {len(val_wins)}  buffer_gap: {n_gap}")

    # Build window_dates: one date string per window (the last encoder day),
    # used as the S2S forecast issue date lookup key.
    # For S2S cache lookup we need the BASE DATE (= aligned_dates[w[1]]),
    # which is the first day of the prediction window / S2S forecast issue date.
    # Step 10 inference also uses base_date, so this keeps them consistent.
    all_train_window_dates = [
        str(aligned_dates[w[1]]) for w in train_wins
    ]
    all_val_window_dates = [
        str(aligned_dates[w[1]]) for w in val_wins
    ]

    # ── S2S cache coverage diagnostic ────────────────────────────────
    if args.decoder == "s2s" and date_to_s2s_idx is not None:
        _train_exact = sum(1 for d in all_train_window_dates
                           if date_to_s2s_exact.get(d, False))
        _train_total = sum(1 for d in all_train_window_dates if d in date_to_s2s_idx)
        _train_fb    = _train_total - _train_exact
        _train_miss  = len(all_train_window_dates) - _train_total
        _val_exact   = sum(1 for d in all_val_window_dates
                           if date_to_s2s_exact.get(d, False))
        _val_total   = sum(1 for d in all_val_window_dates if d in date_to_s2s_idx)
        _val_fb      = _val_total - _val_exact
        _val_miss    = len(all_val_window_dates) - _val_total
        print(f"\n  S2S cache coverage:")
        print(f"    train: {_train_exact}/{len(all_train_window_dates)} exact  "
              f"{_train_fb} fallback  "
              f"{_train_miss} miss  "
              f"({100*_train_total/max(len(all_train_window_dates),1):.1f}% usable)")
        print(f"    val  : {_val_exact}/{len(all_val_window_dates)} exact  "
              f"{_val_fb} fallback  "
              f"{_val_miss} miss  "
              f"({100*_val_total/max(len(all_val_window_dates),1):.1f}% usable)")

        # ── S2S cache internal missing-value diagnostic ─────────────
        # Sample random (date, patch) pairs to estimate % of all-zero lead rows
        _rng_diag = np.random.RandomState(42)
        _n_sample_dates = min(200, s2s_cache.shape[0])
        _n_sample_patches = min(50, s2s_cache.shape[1])
        _diag_date_idxs = _rng_diag.choice(s2s_cache.shape[0], _n_sample_dates, replace=False)
        _diag_patch_idxs = _rng_diag.choice(s2s_cache.shape[1], _n_sample_patches, replace=False)
        _total_leads = 0
        _missing_leads = 0
        for _di in _diag_date_idxs:
            for _pi in _diag_patch_idxs:
                _row = s2s_cache[_di, _pi]  # (32, 6)
                _allzero = np.all(_row == 0, axis=1)  # (32,) bool
                _total_leads += _row.shape[0]
                _missing_leads += int(_allzero.sum())
        _miss_pct = 100.0 * _missing_leads / max(_total_leads, 1)
        print(f"\n  S2S cache internal missing (sampled {_n_sample_dates} dates × "
              f"{_n_sample_patches} patches):")
        print(f"    all-zero lead-day rows: {_missing_leads}/{_total_leads} ({_miss_pct:.1f}%)")
        print(f"    → these will get is_missing=1 + random N(0,1) weather channels")

    # Do not filter train/val windows by S2S coverage here.
    # _make_dec_s2s already handles unmapped base dates by injecting
    # random weather channels and setting is_missing=1, which preserves
    # the full training history when the S2S archive starts late.

    # ----------------------------------------------------------------
    # STEP 7b  Build positive pairs; sample negative pairs
    # ----------------------------------------------------------------
    print("\n[STEP 7b] Filtering positive patches + sampling negative patches...")
    t0_filter = time.time()

    pos_pairs = []
    for win_i, (hs, he, ts, te) in enumerate(train_wins):
        for patch_i in range(n_patches):
            if fire_patched[ts:te, patch_i, :].max() > 0:
                pos_pairs.append((win_i, patch_i))
        if win_i % 200 == 0 or win_i == len(train_wins) - 1:
            print(f"  scanned {win_i+1}/{len(train_wins)} windows  "
                  f"pos_pairs: {len(pos_pairs):,}  ({time.time()-t0_filter:.0f}s)")

    total_pairs = len(train_wins) * n_patches
    pct = 100.0 * len(pos_pairs) / max(total_pairs, 1)
    print(f"  Positive pairs: {len(pos_pairs):,} / {total_pairs:,}  ({pct:.3f}%)")

    if len(pos_pairs) == 0:
        raise RuntimeError(
            "No positive (window, patch) pairs found. "
            "Check hotspot_csv and date alignment."
        )

    rng = np.random.default_rng(args.seed)
    if args.max_pos_pairs > 0 and len(pos_pairs) > args.max_pos_pairs:
        idx_cap  = rng.choice(len(pos_pairs), size=args.max_pos_pairs, replace=False)
        pos_pairs = [pos_pairs[i] for i in idx_cap]
        print(f"  Capped pos_pairs → {len(pos_pairs):,}")

    pos_flat = np.array([w * n_patches + p for w, p in pos_pairs], dtype=np.int64)
    pos_mask = np.zeros(total_pairs, dtype=bool)
    pos_mask[pos_flat] = True

    # Spatial buffer: exclude patches near positive patches from negative pool
    if args.neg_buffer > 0:
        _buf = args.neg_buffer
        _struct = np.ones((2*_buf+1, 2*_buf+1), dtype=bool)
        _nrow, _ncol = grid  # (142, 169)
        _n_excluded = 0
        for win_i in range(len(train_wins)):
            _win_offset = win_i * n_patches
            _win_pos = pos_mask[_win_offset:_win_offset + n_patches]
            if not _win_pos.any():
                continue
            _pos_grid = _win_pos.reshape(_nrow, _ncol)
            _buf_grid = binary_dilation(_pos_grid, structure=_struct)
            _buf_flat = _buf_grid.reshape(-1)
            # Mark buffer patches as excluded (treat as positive → can't be neg)
            _newly_excluded = _buf_flat & ~_win_pos
            pos_mask[_win_offset:_win_offset + n_patches] |= _buf_flat
            _n_excluded += _newly_excluded.sum()
        print(f"  neg_buffer={_buf}: excluded {_n_excluded:,} additional buffer patches")

    neg_flat          = np.where(~pos_mask)[0]
    max_available_neg = int(neg_flat.shape[0])
    n_neg_target      = min(int(len(pos_pairs) * args.neg_ratio), max_available_neg)
    print(f"  neg_target={n_neg_target:,}  max_available_neg={max_available_neg:,}")
    chosen      = rng.choice(neg_flat, size=n_neg_target, replace=False)
    neg_wins    = (chosen // n_patches).astype(np.int32)
    neg_patches = (chosen %  n_patches).astype(np.int32)
    # Build all_pairs as a compact numpy (N, 2) int32 array instead of a
    # Python list of tuples.  12.7M Python tuples ≈ 1.4 GB of Python objects
    # and are very slow to pickle when spawning DataLoader workers on Windows.
    # A numpy (N, 2) int32 array is ≈ 100 MB and pickles as a single binary blob.
    pos_arr  = np.array(pos_pairs,   dtype=np.int32)             # (pos_count, 2)
    neg_arr  = np.column_stack([neg_wins, neg_patches]).astype(np.int32)  # (neg_count, 2)
    all_pairs = np.vstack([pos_arr, neg_arr])                    # (N, 2)  ~100 MB
    rng.shuffle(all_pairs)                                       # shuffle rows in-place
    del pos_arr, neg_arr
    gc.collect()
    n_neg = int(neg_wins.shape[0])
    print(f"  Neg pairs sampled: {n_neg:,}  (neg_ratio={args.neg_ratio})")
    print(f"  Total train pairs: {len(all_pairs):,}  "
          f"(numpy int32, {all_pairs.nbytes/1e6:.0f} MB — fast pickle)")
    print(f"  Sampling time: {time.time()-t0_filter:.0f}s")
    run_meta["pos_pairs"]   = len(pos_pairs)
    run_meta["neg_pairs"]   = n_neg
    run_meta["total_pairs"] = len(all_pairs)

    # ----------------------------------------------------------------
    # Memory Guard  (background thread — starts now, checked each epoch)
    # ----------------------------------------------------------------
    _meteo_guard_gb = (n_patches * T * enc_dim * 2) / 1e9   # float16
    _fire_guard_gb  = (T * n_patches * out_dim) / 1e9        # uint8
    mem_guard = MemoryGuard(
        limit_pct   = args.mem_limit_pct,
        interval    = 15,
        meteo_gb    = _meteo_guard_gb,
        fire_gb     = _fire_guard_gb,
        batch_size  = args.batch_size,
    )
    if _PSUTIL_OK and args.mem_limit_pct > 0:
        mem_guard.start()
        vm0 = _psutil.virtual_memory()
        print(f"\n  MemoryGuard active: limit={args.mem_limit_pct:.0f}%  "
              f"current={vm0.percent:.1f}%  "
              f"({vm0.used/1e9:.1f}/{vm0.total/1e9:.1f} GB)")
    else:
        if not _PSUTIL_OK:
            print("\n  MemoryGuard disabled (psutil not installed — run: pip install psutil)")
        else:
            print("\n  MemoryGuard disabled (--mem_limit_pct=0)")

    # ----------------------------------------------------------------
    # STEP 8  Compute pos_weight; build BCE criterion
    # ----------------------------------------------------------------
    print("\n[STEP 8] Computing BCE pos_weight from pixel counts...")
    pos_pixels        = 0
    neg_pixels_in_pos = 0
    for win_i, patch_i in pos_pairs:
        hs, he, ts, te = train_wins[win_i]
        pf = fire_patched[ts:te, patch_i, :]
        p  = int(pf.sum())
        pos_pixels        += p
        neg_pixels_in_pos += pf.size - p
    neg_pixels_total = neg_pixels_in_pos + n_neg * out_dim * decoder_days
    raw_ratio        = neg_pixels_total / max(pos_pixels, 1)
    pos_weight_val   = min(raw_ratio, args.pos_weight_cap)
    print(f"  pos_pixels : {pos_pixels:,}")
    print(f"  neg_pixels : {neg_pixels_total:,}")
    print(f"  raw ratio  : {raw_ratio:.1f}   "
          f"pos_weight (capped {args.pos_weight_cap}) = {pos_weight_val:.2f}")
    run_meta["pos_weight"] = pos_weight_val

    # ----------------------------------------------------------------
    # Optionally copy only training time steps into RAM
    # ----------------------------------------------------------------
    meteo_train           = meteo_patched        # default: training reads from disk
    train_wins_eff        = train_wins           # effective windows passed to train_ds
    all_pairs_eff         = all_pairs            # effective pairs passed to train_ds
    train_window_dates_eff = all_train_window_dates  # effective window_dates for train_ds

    if args.load_train_to_ram and not args.load_to_ram:
        import psutil
        train_T_max = max(te for hs, he, ts, te in train_wins)

        if args.fire_season_only:
            fire_months = set(int(m) for m in args.fire_season_months.split(","))

            # Filter windows: keep only those where every T in encoder+decoder is fire-season
            valid_mask = []
            for hs, he, ts, te in train_wins:
                ok = all(aligned_dates[t].month in fire_months for t in range(hs, he)) and \
                     all(aligned_dates[t].month in fire_months for t in range(ts, te))
                valid_mask.append(ok)
            valid_idxs   = [i for i, v in enumerate(valid_mask) if v]
            filtered_wins = [train_wins[i] for i in valid_idxs]
            print(f"\n[--load_train_to_ram --fire_season_only] "
                  f"{len(filtered_wins)}/{len(train_wins)} windows kept (months {sorted(fire_months)})")

            # Collect union of all T indices needed by filtered windows
            t_needed = set()
            for hs, he, ts, te in filtered_wins:
                t_needed.update(range(hs, he))
                t_needed.update(range(ts, te))
            t_indices = np.array(sorted(t_needed), dtype=np.int32)

            # Build T remapping: original T → compact index
            t_remap = np.full(train_T_max + 2, -1, dtype=np.int32)
            for new_t, orig_t in enumerate(t_indices):
                t_remap[orig_t] = new_t

            # Remap windows.
            # he/te are exclusive ends; they are NOT in t_indices, so t_remap[he]==-1.
            # Correct: new_exclusive_end = t_remap[last_included] + 1
            train_wins_eff = [(int(t_remap[hs]),    int(t_remap[he - 1] + 1),
                               int(t_remap[ts]),    int(t_remap[te - 1] + 1))
                              for hs, he, ts, te in filtered_wins]

            # Remap window_dates to match filtered windows
            train_window_dates_eff = [all_train_window_dates[i] for i in valid_idxs]

            # Remap all_pairs: filter to valid windows, remap win_i
            win_remap = np.full(len(train_wins), -1, dtype=np.int32)
            for new_i, old_i in enumerate(valid_idxs):
                win_remap[old_i] = new_i
            keep = np.isin(all_pairs[:, 0], valid_idxs)
            all_pairs_eff = all_pairs[keep].copy()
            all_pairs_eff[:, 0] = win_remap[all_pairs_eff[:, 0]]

            needed_gb = len(t_indices) * n_patches * enc_dim * 2 / 1e9
            print(f"  T indices to load: {len(t_indices)}  (~{needed_gb:.1f}GB)")
        else:
            needed_gb = (train_T_max + 1) * n_patches * enc_dim * 2 / 1e9
            t_indices = np.arange(train_T_max + 1, dtype=np.int32)
            print(f"\n[--load_train_to_ram] Training T slice 0:{train_T_max+1}  (~{needed_gb:.1f}GB)")

        ram = psutil.virtual_memory()
        print(f"  RAM available: {ram.available/1e9:.1f}GB / {ram.total/1e9:.1f}GB total")
        if ram.available / 1e9 < needed_gb + 20:
            print(f"  WARNING: available RAM may be insufficient (need ~{needed_gb+20:.0f}GB)")
        t0 = time.time()
        print(f"  Copying into RAM...")
        meteo_train = np.array(meteo_patched[:, t_indices, :])
        ram_after = psutil.virtual_memory()
        print(f"  [OK] Copy complete in {time.time()-t0:.0f}s  "
              f"({meteo_train.nbytes/1e9:.1f}GB in RAM, "
              f"{ram_after.available/1e9:.1f}GB RAM remaining)")

        # Drop memmap page cache NOW (before val copy) to avoid OOM peak.
        # The train numpy array is safely in RAM; evicting the disk-cache
        # pages frees ~90GB so the subsequent val copy doesn't spike over budget.
        if hasattr(meteo_patched, '_mmap') and meteo_patched._mmap is not None:
            try:
                import mmap as _mmap
                meteo_patched._mmap.madvise(_mmap.MADV_DONTNEED)
                print("  [RAM] Dropped memmap page cache before val copy (MADV_DONTNEED)")
            except Exception as _e:
                print(f"  [RAM] madvise not available: {_e}")

    # ----------------------------------------------------------------
    # Optionally copy only validation time steps into RAM
    # ----------------------------------------------------------------
    meteo_val            = meteo_patched   # default: val reads from disk
    val_wins_eff         = val_wins
    val_window_dates_eff = all_val_window_dates

    if args.load_val_to_ram and not args.load_to_ram and val_wins:
        import psutil
        t_needed_val = set()

        # When --fire_season_only is set, filter val windows to those where every T
        # in encoder+decoder range is a fire-season month (same logic as train).
        # NOTE: val windows CAN span non-fire-season months because lead_end=46 days
        # means a window starting in late September may target into November.
        val_wins_for_ram        = val_wins
        val_dates_for_ram       = all_val_window_dates
        if args.fire_season_only:
            fire_months = set(int(m) for m in args.fire_season_months.split(","))
            _val_filtered = [
                (w, d) for w, d in zip(val_wins, all_val_window_dates)
                if all(aligned_dates[t].month in fire_months for t in range(w[0], w[1]))
                and all(aligned_dates[t].month in fire_months for t in range(w[2], w[3]))
            ]
            val_wins_for_ram  = [x[0] for x in _val_filtered]
            val_dates_for_ram = [x[1] for x in _val_filtered]
            print(f"\n[--load_val_to_ram --fire_season_only] "
                  f"{len(val_wins_for_ram)}/{len(val_wins)} val windows kept "
                  f"(months {sorted(fire_months)})")

        for hs, he, ts, te in val_wins_for_ram:
            t_needed_val.update(range(hs, he))
            t_needed_val.update(range(ts, te))
        t_indices_val = np.array(sorted(t_needed_val), dtype=np.int32)

        val_T_max = max(t_indices_val)
        t_remap_val = np.full(val_T_max + 2, -1, dtype=np.int32)
        for new_t, orig_t in enumerate(t_indices_val):
            t_remap_val[orig_t] = new_t

        val_wins_eff = [(int(t_remap_val[hs]),    int(t_remap_val[he - 1] + 1),
                         int(t_remap_val[ts]),    int(t_remap_val[te - 1] + 1))
                        for hs, he, ts, te in val_wins_for_ram]
        val_window_dates_eff = val_dates_for_ram

        needed_gb_val = len(t_indices_val) * n_patches * enc_dim * 2 / 1e9
        ram = psutil.virtual_memory()
        print(f"\n[--load_val_to_ram] Val T indices: {len(t_indices_val)}  (~{needed_gb_val:.1f}GB)")
        print(f"  RAM available: {ram.available/1e9:.1f}GB / {ram.total/1e9:.1f}GB total")
        if ram.available / 1e9 < needed_gb_val + 10:
            print(f"  WARNING: available RAM may be insufficient (need ~{needed_gb_val+10:.0f}GB)")
        t0 = time.time()
        print(f"  Copying val data into RAM...")
        meteo_val = np.array(meteo_patched[:, t_indices_val, :])
        ram_after = psutil.virtual_memory()
        print(f"  [OK] Copy complete in {time.time()-t0:.0f}s  "
              f"({meteo_val.nbytes/1e9:.1f}GB in RAM, "
              f"{ram_after.available/1e9:.1f}GB RAM remaining)")

    # Final page cache drop (covers val-copy pages and the --load_to_ram path).
    if (args.load_train_to_ram or args.load_to_ram) and \
            hasattr(meteo_patched, '_mmap') and meteo_patched._mmap is not None:
        try:
            import mmap as _mmap
            meteo_patched._mmap.madvise(_mmap.MADV_DONTNEED)
            print("  [RAM] Final memmap page cache drop (MADV_DONTNEED)")
        except Exception as _e:
            print(f"  [RAM] madvise not available: {_e}")

    # ----------------------------------------------------------------
    # Build datasets and dataloaders
    # ----------------------------------------------------------------
    train_ds = S2SHotspotDatasetMixed(
        meteo_train, fire_patched, train_wins_eff, hw, grid, all_pairs_eff,
        decoder_mode=args.decoder, dec_dim=dec_dim,
        s2s_cache=s2s_cache, date_to_s2s_idx=date_to_s2s_idx,
        window_dates=train_window_dates_eff, patch_size=P,
        s2s_means=s2s_means, s2s_stds=s2s_stds,
        date_to_s2s_lag=date_to_s2s_lag, s2s_max_lag=args.s2s_max_issue_lag,
        s2s_full_cache=s2s_full_cache,
    )
    if val_wins:
        val_ds = S2SHotspotDatasetUnfiltered(
            meteo_val, fire_patched, val_wins_eff, hw, grid,
            decoder_mode=args.decoder, dec_dim=dec_dim,
            s2s_cache=s2s_cache, date_to_s2s_idx=date_to_s2s_idx,
            window_dates=val_window_dates_eff, patch_size=P,
            s2s_means=s2s_means, s2s_stds=s2s_stds,
            date_to_s2s_lag=date_to_s2s_lag, s2s_max_lag=args.s2s_max_issue_lag,
            s2s_full_cache=s2s_full_cache,
        )
    else:
        val_ds = train_ds
        print("  WARNING: no val windows — using train set as val proxy")

    patch_dim_enc = enc_dim
    patch_dim_dec = dec_dim
    patch_dim_out = out_dim

    print(f"\n  Train samples (mixed pos+neg): {len(train_ds):,}")
    print(f"  Val   samples (unfiltered)  : {len(val_ds):,}")
    print(f"  Grid: {grid[0]}×{grid[1]} patches/frame  "
          f"(enc_dim={patch_dim_enc}  dec_dim={patch_dim_dec}  out_dim={patch_dim_out})")

    _prefetch = 4 if args.num_workers > 0 else None
    _persistent = args.num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          pin_memory=True, num_workers=args.num_workers,
                          persistent_workers=_persistent,
                          prefetch_factor=_prefetch)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          pin_memory=True, num_workers=args.num_workers,
                          persistent_workers=_persistent,
                          prefetch_factor=_prefetch)

    # Pre-compute fire-season val windows for Lift@K (avoids 0.00x from winter windows)
    _lift_fire_months = {4, 5, 6, 7, 8, 9, 10}
    _lift_filtered = [
        (w, d) for w, d in zip(val_wins, all_val_window_dates)
        if all(aligned_dates[t].month in _lift_fire_months for t in range(w[0], w[1]))
        and all(aligned_dates[t].month in _lift_fire_months for t in range(w[2], w[3]))
    ]
    val_wins_lift       = [x[0] for x in _lift_filtered]
    val_wins_lift_dates = [x[1] for x in _lift_filtered]
    print(f"  Val Lift@K windows (fire season only): {len(val_wins_lift)}/{len(val_wins)}")

    # ----------------------------------------------------------------
    # EVAL EPOCHS MODE  (--eval_epochs)
    # ----------------------------------------------------------------
    if args.eval_epochs:
        print("\n[EVAL EPOCHS] Evaluating all epoch checkpoints on val set...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device  : {device}")
        print(f"  Val wins: {len(val_wins)}  (sampling {args.eval_n_windows})")
        print(f"  K       : {args.val_lift_k}")

        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        if not ckpt_files:
            print("  No epoch_*.pt checkpoints found. Run training with --skip_val first.")
            return

        results = []
        for ckpt_path in ckpt_files:
            epoch_name = os.path.basename(ckpt_path).replace(".pt", "")
            print(f"\n  Loading {epoch_name}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            _ckpt_args = ckpt.get("args", {})
            _ckpt_dropout = _ckpt_args.get("dropout", args.dropout)
            model_eval = S2SHotspotTransformer(
                patch_dim_enc=patch_dim_enc,
                patch_dim_dec=patch_dim_dec,
                patch_dim_out=patch_dim_out,
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.enc_layers,
                num_decoder_layers=args.dec_layers,
                dim_feedforward=args.d_model * 4,
                dropout=_ckpt_dropout,
                encoder_days=in_days,
                decoder_days=decoder_days,
            ).to(device)
            model_eval.load_state_dict(ckpt["model_state"])
            m = _compute_val_lift_k(
                model_eval, meteo_patched, fire_patched, val_wins,
                n_patches, k=args.val_lift_k,
                n_sample_wins=args.eval_n_windows,
                chunk=256, device=device,
                decoder_mode=args.decoder, dec_dim=dec_dim,
                s2s_cache=s2s_cache, date_to_s2s_idx=date_to_s2s_idx,
                val_win_dates=all_val_window_dates, patch_size=P,
                s2s_means=s2s_means, s2s_stds=s2s_stds,
                date_to_s2s_lag=date_to_s2s_lag, s2s_max_lag=args.s2s_max_issue_lag,
                s2s_full_cache=s2s_full_cache,
            )
            print(f"    Lift@{args.val_lift_k}={m['lift_k']:.2f}x  "
                  f"Prec={m['precision_k']:.4f}  Recall={m['recall_k']:.4f}  "
                  f"CSI={m['csi_k']:.4f}  ETS={m['ets_k']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
                  f"(n_fire={m['n_fire']:,}  baseline={m['baseline']:.6f})")
            results.append((epoch_name, m))
            del model_eval
            torch.cuda.empty_cache()

        K = args.val_lift_k
        print(f"\n{'─'*90}")
        print(f"  {'Epoch':<12}  {'Lift@K':>8}  {'Prec@K':>8}  {'Recall@K':>9}  "
              f"{'CSI@K':>7}  {'ETS@K':>7}  {'PR-AUC':>8}")
        print(f"{'─'*90}")
        best_name = max(results, key=lambda x: x[1]["lift_k"])[0]
        for name, m in results:
            marker = " ★" if name == best_name else ""
            print(f"  {name:<12}  {m['lift_k']:>8.2f}x  {m['precision_k']:>8.4f}  "
                  f"{m['recall_k']:>9.4f}  {m['csi_k']:>7.4f}  {m['ets_k']:>7.4f}  "
                  f"{m['pr_auc']:>8.4f}{marker}")
        print(f"{'─'*90}")
        best_m = next(m for n, m in results if n == best_name)
        print(f"  Best: {best_name}  Lift@{K}={best_m['lift_k']:.2f}x  "
              f"ETS={best_m['ets_k']:.4f}  PR-AUC={best_m['pr_auc']:.4f}")
        return

    # ----------------------------------------------------------------
    # STEP 9  Build model & train
    # ----------------------------------------------------------------
    print("\n[STEP 9] Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = S2SHotspotTransformer(
        patch_dim_enc=patch_dim_enc,
        patch_dim_dec=patch_dim_dec,
        patch_dim_out=patch_dim_out,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        encoder_days=in_days,
        decoder_days=decoder_days,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Regularization: dropout={args.dropout}  weight_decay={args.weight_decay}  "
          f"label_smoothing={args.label_smoothing}  neg_buffer={args.neg_buffer}")
    run_meta["n_params"] = n_params

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # AMP (Automatic Mixed Precision) — enabled by default on CUDA
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    print(f"  AMP: {'enabled (float16 forward/backward)' if amp_enabled else 'disabled'}")

    best_val_loss  = float("inf")
    best_val_lift_k = -1.0          # checkpoint selected by Lift@K, not val_loss
    best_ckpt      = os.path.join(ckpt_dir, "best_model.pt")
    start_epoch    = 1

    # ── Resume from latest epoch checkpoint ──────────────────────────
    if args.resume:
        import re as _re
        _epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        if _epoch_ckpts:
            _latest_ckpt = _epoch_ckpts[-1]
            print(f"  Resuming from: {_latest_ckpt}")
            _ckpt = torch.load(_latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(_ckpt["model_state"])
            if "optimizer_state" in _ckpt:
                optimizer.load_state_dict(_ckpt["optimizer_state"])
            if "scheduler_state" in _ckpt:
                scheduler.load_state_dict(_ckpt["scheduler_state"])
            if "scaler_state" in _ckpt and amp_enabled:
                scaler.load_state_dict(_ckpt["scaler_state"])
                # Guard: if scaler scale collapsed to 0 (e.g. from excessive overflows
                # in a previous run with too-high lr), reset to default to avoid
                # div-by-zero in unscale_() → all-NaN gradients.
                if hasattr(scaler, '_scale') and scaler._scale is not None:
                    if scaler._scale.item() <= 0:
                        print(f"  WARNING: loaded scaler scale={scaler._scale.item():.1f} → "
                              f"resetting to 2^16 (previous run likely had gradient overflow)")
                        scaler._scale = torch.tensor(2**16, dtype=torch.float32,
                                                      device=scaler._scale.device)
            start_epoch = _ckpt["epoch"] + 1
            best_val_lift_k = _ckpt.get("best_val_lift_k_global", -1.0)
            print(f"  Resumed: starting at epoch {start_epoch}, "
                  f"best_val_lift_k={best_val_lift_k:.2f}")
            del _ckpt
        else:
            print("  --resume: no epoch checkpoints found, starting from scratch.")

    n_batches_train = len(train_dl)
    train_started_at = time.time()
    print(f"\n  Starting training: epochs {start_epoch}→{args.epochs}  "
          f"{n_batches_train:,} batches/epoch  "
          f"log_interval={args.log_interval}  lr={args.lr}")

    for epoch in range(start_epoch, args.epochs + 1):
        # -- Memory guard check (background thread sets .triggered) --
        if mem_guard.triggered:
            print(f"  Epoch {epoch}: MemoryGuard triggered — stopping training early.")
            break

        # -- Training --
        model.train()
        t0_epoch      = time.time()
        train_loss    = 0.0
        train_samples = 0
        _gnorm_sum    = 0.0
        _gnorm_max    = 0.0
        _gnorm_count  = 0

        for batch_idx, (xb_enc, xb_dec, yb) in enumerate(train_dl):
            # Cast float16→float32 on GPU (2× faster transfer than CPU cast)
            xb_enc = xb_enc.to(device, dtype=torch.float32, non_blocking=True)
            xb_dec = xb_dec.to(device, dtype=torch.float32, non_blocking=True)
            yb     = yb.to(device, non_blocking=True)
            if args.label_smoothing > 0:
                yb = yb * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=amp_enabled):
                logits = model(xb_enc, xb_dec)
                loss   = criterion(logits, yb)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(_gnorm):
                optimizer.zero_grad()
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            train_loss    += loss.item() * xb_enc.size(0)
            train_samples += xb_enc.size(0)
            _gv = _gnorm.item()
            _gnorm_sum   += _gv
            _gnorm_count += 1
            _gnorm_max    = max(_gnorm_max, _gv)

            # ── Intra-epoch progress print ─────────────────────────────
            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                elapsed_b   = time.time() - t0_epoch
                pct         = (batch_idx + 1) / n_batches_train
                eta_min     = elapsed_b / pct * (1 - pct) / 60
                running_loss = train_loss / max(train_samples, 1)
                spd          = (batch_idx + 1) / max(elapsed_b, 1e-9)
                print(f"    ep{epoch}  [{batch_idx+1:5d}/{n_batches_train}  "
                      f"{pct*100:4.1f}%]  "
                      f"loss={running_loss:.4f}  "
                      f"grad={_gv:.3f}  "
                      f"{spd:.1f}b/s  "
                      f"ETA~{eta_min:.0f}m")

        if train_samples == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs}  *** ALL TRAIN BATCHES NaN — stopping.")
            break
        train_loss /= train_samples
        epoch_train_time = time.time() - t0_epoch
        avg_gnorm = _gnorm_sum / max(_gnorm_count, 1)

        # -- Validation --
        val_loss, val_lift_k, val_prec_k = float("nan"), 0.0, 0.0
        _lfire_sum, _lfire_n = 0.0, 0
        _lbg_sum,   _lbg_n   = 0.0, 0

        if not args.skip_val:
            model.eval()
            val_loss    = 0.0
            val_samples = 0
            with torch.no_grad():
                for _val_bi, (xb_enc, xb_dec, yb) in enumerate(val_dl):
                    if args.val_max_batches > 0 and _val_bi >= args.val_max_batches:
                        break
                    xb_enc = xb_enc.to(device, dtype=torch.float32, non_blocking=True)
                    xb_dec = xb_dec.to(device, dtype=torch.float32, non_blocking=True)
                    yb     = yb.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=amp_enabled):
                        logits_v = model(xb_enc, xb_dec)
                    vl = criterion(logits_v, yb)
                    if torch.isfinite(vl):
                        val_loss    += vl.item() * xb_enc.size(0)
                        val_samples += xb_enc.size(0)
                    _lv = logits_v.detach().float()
                    _yb = yb.detach().bool()
                    _f  = _lv[_yb]
                    _bg = _lv[~_yb]
                    if _f.numel() > 0:
                        _lfire_sum += _f.sum().item()
                        _lfire_n   += _f.numel()
                    _lbg_sum += _bg.sum().item()
                    _lbg_n   += _bg.numel()

            if val_samples == 0:
                print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  "
                      f"val=NaN — stopping early.")
                break
            val_loss /= val_samples

            # Lift@K on fire-season val windows
            if val_wins_lift:
                _m = _compute_val_lift_k(
                    model, meteo_patched, fire_patched, val_wins_lift,
                    n_patches, k=args.val_lift_k,
                    n_sample_wins=args.val_lift_sample_wins,
                    chunk=256, device=device,
                    decoder_mode=args.decoder, dec_dim=dec_dim,
                    s2s_cache=s2s_cache, date_to_s2s_idx=date_to_s2s_idx,
                    val_win_dates=val_wins_lift_dates, patch_size=P,
                    s2s_means=s2s_means, s2s_stds=s2s_stds,
                    date_to_s2s_lag=date_to_s2s_lag, s2s_max_lag=args.s2s_max_issue_lag,
                    s2s_full_cache=s2s_full_cache,
                )
                val_lift_k = _m["lift_k"]
                val_prec_k = _m["precision_k"]
            model.train()

        # ── Epoch summary ──────────────────────────────────────────────
        elapsed_total = time.time() - train_started_at
        _epochs_done  = epoch - start_epoch + 1
        eta_total_min = elapsed_total / _epochs_done * (args.epochs - epoch) / 60
        epoch_min     = epoch_train_time / 60

        # GPU memory
        if torch.cuda.is_available():
            _gpu_alloc = torch.cuda.memory_allocated() / 1e9
            _gpu_res   = torch.cuda.memory_reserved()  / 1e9
            gpu_str    = f"  GPU {_gpu_alloc:.1f}/{_gpu_res:.1f} GB"
        else:
            gpu_str = "  GPU n/a (CPU mode)"

        # System RAM
        if _PSUTIL_OK:
            _vm     = _psutil.virtual_memory()
            ram_str = (f"  RAM {_vm.used/1e9:.0f}/{_vm.total/1e9:.0f} GB "
                       f"({_vm.percent:.0f}%)")
        else:
            ram_str = ""

        print(f"\n  ── Epoch {epoch:3d}/{args.epochs} ───────────────────────────")
        if args.skip_val:
            print(f"  loss   train={train_loss:.6f}  val=skipped")
        else:
            print(f"  loss   train={train_loss:.6f}  val={val_loss:.6f}")
            print(f"  metric Lift@{args.val_lift_k}={val_lift_k:.2f}x  "
                  f"prec@{args.val_lift_k}={val_prec_k:.4f}")
        if _lfire_n > 0:
            print(f"  logits fire={_lfire_sum/_lfire_n:+.3f}  "
                  f"bg={_lbg_sum/_lbg_n:+.3f}  "
                  f"gap={(_lfire_sum/_lfire_n - _lbg_sum/_lbg_n):.3f}  "
                  f"({_lfire_n:,} fire px / {_lbg_n:,} bg px)")
        print(f"  grad   avg={avg_gnorm:.4f}  max={_gnorm_max:.4f}  "
              f"({'CLIPPED' if _gnorm_max >= 0.99 else 'ok'})")
        print(f"  lr     current={scheduler.get_last_lr()[0]:.2e}  "
              f"→ next={args.lr_min + 0.5*(args.lr - args.lr_min)*(1 + __import__('math').cos(__import__('math').pi*(epoch)/args.epochs)):.2e}")
        print(f"  time   epoch={epoch_min:.1f}m  "
              f"elapsed={elapsed_total/60:.0f}m  "
              f"ETA~{eta_total_min:.0f}m")
        print(f"  sys   {gpu_str}{ram_str}")
        print()

        scheduler.step()

        # Update best_val_lift_k BEFORE building checkpoint payload
        # so that epoch_XX.pt always contains the correct global best when resumed.
        is_new_best = (not args.skip_val) and (val_lift_k > best_val_lift_k)
        if is_new_best:
            best_val_lift_k = val_lift_k
            best_val_loss   = val_loss

        # Save checkpoint: every epoch when --skip_val, otherwise best val_loss
        ckpt_payload = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_val_lift_k_global": best_val_lift_k,
            "meteo_means":     meteo_means,
            "meteo_stds":      meteo_stds,
            "patch_dim_enc":   patch_dim_enc,
            "patch_dim_dec":   patch_dim_dec,
            "patch_dim_out":   patch_dim_out,
            "hw":              hw,
            "grid":            grid,
            "args":            vars(args),
            "channel_names":   CHANNEL_NAMES,
            "s2s_means":       s2s_means,      # (6,) float32 or None
            "s2s_stds":        s2s_stds,        # (6,) float32 or None
            "n_channels":      N_CHANNELS,
            "best_val_lift_k": val_lift_k,
        }
        # Always save epoch checkpoint for resume
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pt")
        torch.save(ckpt_payload, epoch_ckpt)
        print(f"           → epoch_{epoch:02d}.pt saved")

        if args.skip_val:
            torch.save(ckpt_payload, best_ckpt)
        else:
            if is_new_best:
                torch.save(ckpt_payload, best_ckpt)
                print(f"           ★ New best  Lift@{args.val_lift_k}={val_lift_k:.2f}x  "
                      f"val_loss={val_loss:.6f}  → best_model.pt saved")

    print(f"\n  Best val Lift@{args.val_lift_k}: {best_val_lift_k:.2f}x  "
          f"(val_loss at that epoch: {best_val_loss:.6f})  saved → {best_ckpt}")
    run_meta["best_val_lift_k"] = best_val_lift_k
    run_meta["best_val_loss"]   = best_val_loss
    mem_guard.shutdown()   # stop background memory monitor

    # ----------------------------------------------------------------
    # STEP 10  Generate forecast GeoTIFFs
    # ----------------------------------------------------------------
    if args.skip_forecast:
        print("\n[STEP 10] Skipped (--skip_forecast). "
              "Run with --forecast_only to generate tifs later.")
        return

    print("\n[STEP 10] Generating forecast tifs...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    pred_dates  = []
    cur = pred_start_date
    while cur <= pred_end_date:
        pred_dates.append(cur)
        cur += timedelta(days=1)

    date_to_idx = {d: i for i, d in enumerate(aligned_dates)}

    n_done = 0
    for base_date in pred_dates:
        if base_date not in date_to_idx:
            continue
        base_idx = date_to_idx[base_date]
        enc_start = base_idx - in_days
        enc_end   = base_idx
        dec_start = base_idx + lead_start
        dec_end   = base_idx + lead_end + 1

        if enc_start < 0 or dec_end > T:
            continue

        chunk = args.pred_batch_size
        n_p   = n_patches
        _dec_days_pred = dec_end - dec_start
        _base_date_str = str(base_date)
        prob_list = []
        with torch.no_grad():
            for cs in range(0, n_p, chunk):
                ce = min(cs + chunk, n_p)
                xb_enc = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[cs:ce, enc_start:enc_end, :].astype(np.float32)
                    )
                ).to(device)
                # Decoder input must match training mode
                if args.decoder == "oracle":
                    xb_dec = torch.from_numpy(
                        np.ascontiguousarray(
                            meteo_patched[cs:ce, dec_start:dec_end, :].astype(np.float32)
                        )
                    ).to(device)
                elif args.decoder == "s2s":
                    if s2s_full_cache is not None:
                        dec_list = [
                            _make_dec_s2s_full(
                                s2s_full_cache, date_to_s2s_idx,
                                _base_date_str, cs + pi,
                                _dec_days_pred, dec_dim,
                            ).astype(np.float32)
                            for pi in range(ce - cs)
                        ]
                    else:
                        dec_list = [
                            _make_dec_s2s(
                                s2s_cache, date_to_s2s_idx,
                                _base_date_str, cs + pi,
                                _dec_days_pred, dec_dim, P,
                                s2s_means=s2s_means, s2s_stds=s2s_stds,
                                date_to_s2s_lag=date_to_s2s_lag,
                                s2s_max_lag=args.s2s_max_issue_lag,
                            ).astype(np.float32)
                            for pi in range(ce - cs)
                        ]
                    xb_dec = torch.from_numpy(
                        np.stack(dec_list, axis=0)
                    ).to(device)
                else:
                    # Ablation modes: zeros / random / climatology
                    enc_np = meteo_patched[cs:ce, enc_start:enc_end, :].astype(np.float32)
                    dec_list = [
                        _make_dec_ablation(args.decoder, enc_np[i],
                                           _dec_days_pred, dec_dim)
                        for i in range(ce - cs)
                    ]
                    xb_dec = torch.from_numpy(
                        np.stack(dec_list, axis=0).astype(np.float32)
                    ).to(device)
                logits = model(xb_enc, xb_dec)
                prob_list.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(prob_list, axis=0)

        base_str = base_date.strftime("%Y%m%d")
        day_out  = os.path.join(output_dir, base_str)
        os.makedirs(day_out, exist_ok=True)

        for li, lead in enumerate(range(lead_start, lead_end + 1)):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            out_path        = os.path.join(
                day_out, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
            )
            prob_patches_lead = probs[:, li, :]
            prob_vol = depatchify(
                prob_patches_lead[:, np.newaxis, :],
                grid, P, hw, num_channels=1
            )
            prob_map = prob_vol[0] if prob_vol.ndim == 3 else prob_vol
            if prob_map.shape != (H, W):
                full = np.zeros((H, W), dtype=np.float32)
                full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                prob_map = full
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(prob_map.astype(np.float32), 1)

        n_done += 1
        if n_done % 20 == 0 or base_date == pred_dates[-1]:
            print(f"  [{n_done}/{len(pred_dates)}] {base_date} → {decoder_days} lead tifs "
                  f"(lead {lead_start}–{lead_end})")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE  [S2S Hotspot V2, ERA5 Oracle Decoder]")
    print(f"  Channels   : {N_CHANNELS} — {', '.join(CHANNEL_NAMES)}")
    print(f"  Forecasts  : {output_dir}")
    print(f"  Checkpoint : {best_ckpt}")
    print(f"  Lead range : {lead_start}–{lead_end} days")
    print("=" * 70)

    run_meta["status"]           = "success"
    run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run log: {run_meta_path}")


if __name__ == "__main__":
    main()
