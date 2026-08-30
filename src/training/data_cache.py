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
