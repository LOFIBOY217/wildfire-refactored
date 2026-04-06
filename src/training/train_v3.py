"""
Train S2S Hotspot Transformer V3  [10-Channel, Focal Loss, Hard Negative Mining]
================================================================================
V3 of the S2S model.  Builds on V2 architecture (S2SHotspotTransformer) with
three major improvements:

  1. EXPANDED FEATURE SET (10 channels, configurable)
     Drops redundant FWI sub-components (FFMC/DMC/DC/BUI/2d), adds:
       - Lightning flash count (GOES GLM) — ~50% of boreal fires
       - NDVI (MODIS, 16-day interpolated) — vegetation dryness/fuel state
       - Population density (WorldPop, static) — human ignition proxy
       - Deep soil moisture (ERA5 swvl2, 7-28cm) — long-term drought
       - 30-day precipitation deficit (computed) — cumulative drought
       - Terrain slope (SRTM, static) — fire spread factor
       - Years since last burn (NBAC, annual) — fuel availability

  2. FOCAL LOSS + OPTIONAL RANKING LOSS
     FocalBCELoss replaces BCEWithLogitsLoss — down-weights trivially easy
     negatives (ocean, tundra).  Optional ApproxNDCG ranking component for
     direct Top-K optimisation.

  3. HARD NEGATIVE MINING
     Negative patches sampled proportional to fire_clim (geographic fire
     frequency) rather than uniform random.  Focuses training on the
     difficult cases: areas that are fire-prone but didn't burn.

  4. CLUSTER-LEVEL EVALUATION
     Fire pixels merged into spatial clusters via connected components.
     Each cluster = 1 positive event.  Reduces inflation from spatial
     autocorrelation in Top-K metrics.

Channel table:
  Index  Variable        Source               Type
  -----  ----------      ------               ----
  0      FWI             fwi_dir              daily
  1      2t              observation_dir       daily
  2      fire_clim       fire_climatology_tif  static
  3      lightning       lightning_dir         daily
  4      NDVI            ndvi_dir              16-day→daily
  5      population      population_tif        static
  6      deep_soil       deep_soil_dir         daily
  7      precip_def      precip_dir (computed) daily
  8      slope           terrain_dir/slope.tif static
  9      burn_age        burn_scars_dir        annual

n_channels = 10  →  enc_dim = P² × 10 = 16² × 10 = 2560

Usage:
    python -m src.training.train_v3 \\
        --config configs/paths_narval.yaml \\
        --loss_fn focal --hard_neg_fraction 0.5 \\
        --cluster_eval --epochs 8 --batch_size 4096
"""

import argparse
from bisect import bisect_right
from collections import deque
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
from scipy.ndimage import binary_dilation, label as ndimage_label
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
from src.training.losses import FocalBCELoss, ApproxNDCGLoss, HybridLoss

# Reuse helpers from V2
from src.training.train_s2s_hotspot_cwfis_v2 import (
    _make_dec_ablation,
    _make_dec_s2s,
    _make_dec_s2s_full,
    _expand_s2s_date_mapping,
    S2SHotspotDatasetMixed,
    S2SHotspotDatasetUnfiltered,
    _build_file_dict,
    _build_flat_file_dict,
    _build_s2s_windows,
    _read_tif_safe,
    _stream_channel_stats,
    _patchify_frame,
    _transpose_tf_to_pf,
    MemoryGuard,
    _compute_val_lift_k,
    S2S_N_CHANNELS,
    S2S_DEC_DIM,
)


# ------------------------------------------------------------------ #
# V3 channel configuration
# ------------------------------------------------------------------ #

# All available channels with metadata
V3_CHANNEL_DEFS = {
    "FWI":        {"type": "daily",   "required": True},
    "2t":         {"type": "daily",   "required": True},
    "fire_clim":  {"type": "static",  "required": True},
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
    "burn_age":   {"type": "annual",  "required": False},
    "burn_count": {"type": "annual",  "required": False},
    "u10":        {"type": "daily",   "required": False},
    "v10":        {"type": "daily",   "required": False},
    "CAPE":       {"type": "daily",   "required": False},
}

DEFAULT_CHANNELS = "FWI,2t,fire_clim,lightning,NDVI,population,deep_soil,precip_def,slope,burn_age,burn_count,u10,v10,CAPE"

# Static channels to inject into decoder context (spatial info the decoder needs)
DECODER_CTX_CHANNELS = {"fire_clim", "population", "slope", "burn_age", "burn_count"}


# ------------------------------------------------------------------ #
# Static channel loader
# ------------------------------------------------------------------ #

def _load_static_channel(tif_path, expected_h, expected_w, name="static"):
    """Load a static single-band GeoTIFF. Returns (H, W) float32 (zeros on failure)."""
    if tif_path is None or not os.path.exists(tif_path):
        print(f"  [WARN] {name}: file not found: {tif_path} — using zeros")
        return np.zeros((expected_h, expected_w), dtype=np.float32)
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
    if arr.shape != (expected_h, expected_w):
        print(f"  [WARN] {name}: shape {arr.shape} != ({expected_h},{expected_w}) — using zeros")
        return np.zeros((expected_h, expected_w), dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    nonzero = int((arr > 0).sum())
    print(f"  {name}: {arr.shape}  nonzero={nonzero:,}  "
          f"max={arr.max():.3f}  mean(nz)={arr[arr>0].mean():.3f}" if nonzero else
          f"  {name}: {arr.shape}  ALL ZERO")
    return arr


# ------------------------------------------------------------------ #
# Decoder context augmentation
# ------------------------------------------------------------------ #

def _build_decoder_ctx_static(meteo_patched, channel_names, P2):
    """
    Extract static channels from meteo_patched and return per-patch context.

    Args:
        meteo_patched: (n_patches, T, enc_dim) — only need T=0 slice
        channel_names: list of channel name strings
        P2: pixels per patch (P*P = 256)

    Returns:
        static_ctx: (n_patches, n_static * P2) float16
        static_channel_indices: list of int (which channels are static)
    """
    indices = [i for i, name in enumerate(channel_names)
               if name in DECODER_CTX_CHANNELS]
    if not indices:
        return None, []

    # Extract from first timestep (static channels are same across all T)
    # meteo_patched layout: each timestep is [ch0_p0..p255, ch1_p0..p255, ...]
    # Channel k occupies positions [k*P2 : (k+1)*P2]
    parts = []
    for ch_idx in indices:
        # (n_patches, P2) — one static map per patch
        ch_data = meteo_patched[:, 0, ch_idx * P2: (ch_idx + 1) * P2]
        parts.append(ch_data)

    # (n_patches, n_static * P2)
    static_ctx = np.concatenate(parts, axis=1)
    return static_ctx, indices


def _build_lead_time_encoding(dec_days, lead_start, base_doy=None, device=None):
    """
    Build lead time + seasonal encoding for decoder.

    Returns (dec_days, 4) tensor:
      [:, 0] = sin(2π * lead_day / 60)   — lead time cycle (~60 day period)
      [:, 1] = cos(2π * lead_day / 60)
      [:, 2] = sin(2π * doy / 365)       — season cycle
      [:, 3] = cos(2π * doy / 365)
    """
    import math
    leads = torch.arange(lead_start, lead_start + dec_days, dtype=torch.float32)
    lead_enc = torch.stack([
        torch.sin(2 * math.pi * leads / 60),
        torch.cos(2 * math.pi * leads / 60),
    ], dim=-1)  # (dec_days, 2)

    if base_doy is not None:
        doys = torch.arange(base_doy + lead_start,
                            base_doy + lead_start + dec_days,
                            dtype=torch.float32)
    else:
        doys = leads + 180  # rough summer default
    season_enc = torch.stack([
        torch.sin(2 * math.pi * doys / 365),
        torch.cos(2 * math.pi * doys / 365),
    ], dim=-1)  # (dec_days, 2)

    enc = torch.cat([lead_enc, season_enc], dim=-1)  # (dec_days, 4)
    if device is not None:
        enc = enc.to(device)
    return enc


def _augment_decoder(xb_dec, static_ctx_tensor, lead_time_enc):
    """
    Concatenate static spatial context + lead time encoding to decoder input.

    Args:
        xb_dec: (B, dec_days, dec_dim_base)
        static_ctx_tensor: (B, ctx_dim) — static features per patch
        lead_time_enc: (dec_days, 4) — lead time + season encoding

    Returns:
        (B, dec_days, dec_dim_base + ctx_dim + 4)
    """
    B, D, _ = xb_dec.shape

    # Expand static context: (B, ctx_dim) → (B, dec_days, ctx_dim)
    ctx_expanded = static_ctx_tensor.unsqueeze(1).expand(B, D, -1)

    # Expand lead time: (dec_days, 4) → (B, dec_days, 4)
    lt_expanded = lead_time_enc.unsqueeze(0).expand(B, -1, -1)

    return torch.cat([xb_dec, ctx_expanded, lt_expanded], dim=-1)


# ------------------------------------------------------------------ #
# NDVI interpolation helper
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Hard negative mining
# ------------------------------------------------------------------ #

def _sample_hard_negatives(neg_flat, fire_clim_per_patch, n_patches,
                           hard_frac, n_target, rng):
    """
    Sample negatives with fire-climatology-weighted hard mining.

    Args:
        neg_flat: 1D array of flat indices (win_i * n_patches + patch_i) for candidates
        fire_clim_per_patch: (n_patches,) mean fire_clim value per patch
        n_patches: number of spatial patches
        hard_frac: fraction of negatives to sample as "hard" (clim-weighted)
        n_target: total negatives to sample
        rng: numpy random generator

    Returns:
        chosen: 1D array of selected flat indices
    """
    if hard_frac <= 0 or len(neg_flat) == 0:
        # Uniform sampling (v2 behavior)
        return rng.choice(neg_flat, size=min(n_target, len(neg_flat)), replace=False)

    n_hard = min(int(n_target * hard_frac), len(neg_flat))
    n_easy = min(n_target - n_hard, len(neg_flat) - n_hard)

    # Compute weights for hard sampling: proportional to fire_clim of each patch
    patch_ids = neg_flat % n_patches
    weights = fire_clim_per_patch[patch_ids].astype(np.float64)
    weights = np.maximum(weights, 0.0)  # ensure non-negative
    # Add small floor so zero-clim patches can still be sampled
    # (avoids "Fewer non-zero entries in p than size" error)
    weights += 1e-8
    weights /= weights.sum()

    # Sample hard negatives (weighted by fire climatology)
    hard_idx = rng.choice(len(neg_flat), size=n_hard, replace=False, p=weights)
    hard_chosen = neg_flat[hard_idx]

    # Sample easy negatives (uniform from remainder)
    remaining_mask = np.ones(len(neg_flat), dtype=bool)
    remaining_mask[hard_idx] = False
    remaining = neg_flat[remaining_mask]
    if n_easy > 0 and len(remaining) > 0:
        easy_chosen = rng.choice(remaining, size=min(n_easy, len(remaining)), replace=False)
        return np.concatenate([hard_chosen, easy_chosen])

    return hard_chosen


# ------------------------------------------------------------------ #
# Cluster-level evaluation
# ------------------------------------------------------------------ #

def _compute_cluster_lift_k(all_probs_2d, all_labels_2d, k, min_cluster_size=1):
    """
    Compute Lift@K with fire-cluster de-duplication.

    Instead of counting each fire pixel independently, merge adjacent fire
    pixels into clusters.  Each cluster gets a single prediction score
    (max probability over its pixels).  Background pixels remain individual.

    Args:
        all_probs_2d: (H, W) float32 — predicted probabilities
        all_labels_2d: (H, W) uint8 — binary fire labels
        k: top-K cutoff
        min_cluster_size: ignore clusters smaller than this

    Returns:
        dict with lift_k, precision_k, n_clusters, etc.
    """
    # Label connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=bool)
    cluster_map, n_clusters_raw = ndimage_label(all_labels_2d, structure=structure)

    # Compute per-cluster prediction score = max prob over cluster pixels
    cluster_scores = []
    cluster_sizes = []
    for c_id in range(1, n_clusters_raw + 1):
        mask = cluster_map == c_id
        size = mask.sum()
        if size < min_cluster_size:
            continue
        score = float(all_probs_2d[mask].max())
        cluster_scores.append(score)
        cluster_sizes.append(size)

    n_clusters = len(cluster_scores)
    if n_clusters == 0:
        return {"lift_k": 0.0, "precision_k": 0.0, "n_clusters": 0,
                "n_clusters_raw": n_clusters_raw}

    # Background scores: all non-fire pixels
    bg_mask = all_labels_2d == 0
    bg_probs = all_probs_2d[bg_mask]

    # Combine: each cluster = 1 item (label=1), each bg pixel = 1 item (label=0)
    all_scores = np.concatenate([np.array(cluster_scores), bg_probs])
    all_labels = np.concatenate([np.ones(n_clusters), np.zeros(len(bg_probs))])

    n_total = len(all_scores)
    k_eff = min(k, n_total)
    top_idx = np.argpartition(all_scores, -k_eff)[-k_eff:]
    tp = float(all_labels[top_idx].sum())
    baseline = n_clusters / n_total

    precision_k = tp / k_eff
    lift_k = precision_k / baseline if baseline > 0 else 0.0
    recall_k = tp / n_clusters if n_clusters > 0 else 0.0

    return {
        "lift_k": lift_k,
        "precision_k": precision_k,
        "recall_k": recall_k,
        "n_clusters": n_clusters,
        "n_clusters_raw": n_clusters_raw,
        "baseline": baseline,
    }


# ------------------------------------------------------------------ #
# Extended validation with cluster-level and per-lead metrics
# ------------------------------------------------------------------ #

def _compute_val_lift_k_v3(model, meteo_patched, fire_patched, val_wins,
                           n_patches, k, n_sample_wins, chunk, device,
                           decoder_mode="oracle", dec_dim=None,
                           val_win_dates=None, patch_size=16,
                           s2s_cache=None, date_to_s2s_idx=None,
                           s2s_means=None, s2s_stds=None,
                           date_to_s2s_lag=None, s2s_max_lag=3,
                           s2s_full_cache=None, use_patch_embed=False,
                           random_encoder=False,
                           cluster_eval=False, cluster_min_size=1,
                           hw=None, grid=None, full_val=False,
                           per_lead_eval=False):
    """V3 validation: standard pixel-level metrics + optional cluster/per-lead."""
    # Pixel-level metrics via V2 function
    _n_wins = len(val_wins) if full_val else n_sample_wins
    pixel_metrics = _compute_val_lift_k(
        model, meteo_patched, fire_patched, val_wins,
        n_patches, k=k, n_sample_wins=_n_wins,
        chunk=chunk, device=device,
        decoder_mode=decoder_mode, dec_dim=dec_dim,
        val_win_dates=val_win_dates, patch_size=patch_size,
        s2s_means=s2s_means, s2s_stds=s2s_stds,
        date_to_s2s_lag=date_to_s2s_lag, s2s_max_lag=s2s_max_lag,
        s2s_full_cache=s2s_full_cache, use_patch_embed=use_patch_embed,
        random_encoder=random_encoder,
        s2s_cache=s2s_cache,
    )

    result = dict(pixel_metrics)

    # Cluster-level metrics (optional, more expensive)
    if cluster_eval and hw is not None and grid is not None:
        P = patch_size
        Hc, Wc = hw
        result["cluster_lift_k"] = 0.0
        result["cluster_precision_k"] = 0.0
        result["n_clusters"] = 0

        # Quick cluster eval on a single sampled window
        rng = np.random.default_rng(0)
        if len(val_wins) > 0:
            win_idx = rng.choice(len(val_wins))
            hs, he, ts, te = val_wins[win_idx]

            # Get predictions for this window
            model.eval()
            prob_chunks = []
            with torch.no_grad():
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
                    else:
                        enc_np = meteo_patched[cs:ce, hs:he, :].astype(np.float32)
                        dec_list = [
                            _make_dec_ablation(decoder_mode, enc_np[i], te - ts, dec_dim or meteo_patched.shape[2])
                            for i in range(ce - cs)
                        ]
                        xb_dec = torch.from_numpy(
                            np.stack(dec_list, axis=0)
                        ).to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=(device.type == "cuda")):
                        logits = model(xb_enc, xb_dec)
                    prob_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())

            probs = np.concatenate(prob_chunks, axis=0)  # (n_patches, dec_days, P*P)
            labels = fire_patched[ts:te, :, :]  # (dec_days, n_patches, P*P)

            # Aggregate across lead days
            prob_agg = probs.max(axis=1)   # (n_patches, P*P)  max risk over window
            label_agg = labels.max(axis=0)  # (n_patches, P*P)

            # Depatchify to 2D
            prob_2d = depatchify(
                prob_agg[:, np.newaxis, :], grid, P, hw, num_channels=1
            )[0]
            label_2d = depatchify(
                label_agg[:, np.newaxis, :].astype(np.float32), grid, P, hw, num_channels=1
            )[0].astype(np.uint8)

            cm = _compute_cluster_lift_k(prob_2d, label_2d, k, cluster_min_size)
            result["cluster_lift_k"] = cm["lift_k"]
            result["cluster_precision_k"] = cm["precision_k"]
            result["n_clusters"] = cm["n_clusters"]

    # Per-lead-day metrics (optional)
    if per_lead_eval:
        _n_wins = len(val_wins) if full_val else n_sample_wins
        rng = np.random.default_rng(0)
        if len(val_wins) > _n_wins:
            _idx = rng.choice(len(val_wins), size=_n_wins, replace=False)
            _sample_idxs = sorted(_idx)
            _sample_wins = [val_wins[i] for i in _sample_idxs]
            _sample_dates = ([val_win_dates[i] for i in _sample_idxs]
                             if val_win_dates is not None else [None] * len(_sample_wins))
        else:
            _sample_wins = val_wins
            _sample_dates = (val_win_dates if val_win_dates is not None
                             else [None] * len(val_wins))

        _dec_dim = dec_dim or meteo_patched.shape[2]
        # Collect per-lead probs and labels: dict[lead_day] -> (probs_list, labels_list)
        n_lead_days = None
        per_lead_probs = {}   # lead_day -> list of 1-D arrays
        per_lead_labels = {}

        model.eval()
        with torch.no_grad():
            for win_i, (hs, he, ts, te) in enumerate(_sample_wins):
                win_date = _sample_dates[win_i] if _sample_dates else None
                prob_chunks = []
                for cs in range(0, n_patches, chunk):
                    ce = min(cs + chunk, n_patches)
                    xb_enc = torch.from_numpy(
                        np.ascontiguousarray(
                            meteo_patched[cs:ce, hs:he, :].astype(np.float32)
                        )
                    ).to(device)
                    if random_encoder:
                        xb_enc = torch.randn_like(xb_enc)
                    if decoder_mode == "oracle":
                        xb_dec = torch.from_numpy(
                            np.ascontiguousarray(
                                meteo_patched[cs:ce, ts:te, :].astype(np.float32)
                            )
                        ).to(device)
                    else:
                        enc_np = meteo_patched[cs:ce, hs:he, :].astype(np.float32)
                        dec_list = [
                            _make_dec_ablation(decoder_mode, enc_np[i], te - ts, _dec_dim)
                            for i in range(ce - cs)
                        ]
                        xb_dec = torch.from_numpy(
                            np.stack(dec_list, axis=0)
                        ).to(device)
                    _chunk_patch_ids = (torch.arange(cs, ce, device=device)
                                        if use_patch_embed else None)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=(device.type == "cuda")):
                        logits = model(xb_enc, xb_dec, _chunk_patch_ids)
                    prob_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())

                probs = np.concatenate(prob_chunks, axis=0)   # (n_patches, dec_days, P²)
                labels = fire_patched[ts:te, :, :]            # (dec_days, n_patches, P²)
                dec_days = probs.shape[1]
                if n_lead_days is None:
                    n_lead_days = dec_days

                for ld in range(dec_days):
                    p = probs[:, ld, :].reshape(-1)             # (n_patches * P²,)
                    l = labels[ld, :, :].reshape(-1).astype(np.float32)
                    per_lead_probs.setdefault(ld, []).append(p)
                    per_lead_labels.setdefault(ld, []).append(l)

        # Compute Lift@K and Precision@K for each lead day
        per_lead_lift = []
        per_lead_precision = []
        for ld in range(n_lead_days or 0):
            all_p = np.concatenate(per_lead_probs[ld])
            all_l = np.concatenate(per_lead_labels[ld])
            n_total = len(all_p)
            n_fire = int(all_l.sum())
            if n_fire == 0 or n_total == 0:
                per_lead_lift.append(0.0)
                per_lead_precision.append(0.0)
                continue
            k_eff = min(k, n_total)
            top_idx = np.argpartition(all_p, -k_eff)[-k_eff:]
            tp = float(all_l[top_idx].sum())
            baseline = n_fire / n_total
            prec = tp / k_eff
            lift = prec / baseline if baseline > 0 else 0.0
            per_lead_lift.append(lift)
            per_lead_precision.append(prec)

        result["per_lead_lift"] = per_lead_lift
        result["per_lead_precision"] = per_lead_precision

    return result


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    run_started_at = time.time()
    run_started_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"

    ap = argparse.ArgumentParser(
        description="Train S2S Hotspot Transformer V3 [10 channels, Focal Loss, Hard Negatives]"
    )
    add_config_argument(ap)
    ap.add_argument("--run_name", type=str, default="s2s_hotspot_v3",
                    help="Run name for checkpoint/output subdirectory.")

    # Data
    ap.add_argument("--data_start", type=str, default="2018-05-01")
    ap.add_argument("--pred_start", type=str, default="2022-05-01")
    ap.add_argument("--pred_end", type=str, default="2024-10-31")
    ap.add_argument("--in_days", type=int, default=7)
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=46)
    ap.add_argument("--fire_climatology_tif", type=str, default=None)
    ap.add_argument("--dilate_radius", type=int, default=14)

    # V3: channel selection
    ap.add_argument("--channels", type=str, default=DEFAULT_CHANNELS,
                    help="Comma-separated channel names (default: all 10).")
    ap.add_argument("--precip_deficit_days", type=int, default=30,
                    help="Rolling window days for precip deficit (default: 30).")
    ap.add_argument("--deep_soil_dir", type=str, default=None)
    ap.add_argument("--population_tif", type=str, default=None)
    ap.add_argument("--burn_age_encoding", type=str, default="log1p",
                    choices=["log1p", "bucket"],
                    help="How to encode burn_age channel:\n"
                         "  log1p  — log1p(years), continuous (default)\n"
                         "  bucket — categorical buckets reflecting reburn ecology:\n"
                         "           0-2yr=0.25 (just burned, low fuel)\n"
                         "           3-10yr=0.50 (recovering shrubs)\n"
                         "           11-20yr=0.75 (dense regrowth)\n"
                         "           20+yr/never=1.0 (mature forest, high fuel)\n"
                         "  TODO: multi mode (3 sub-channels) for future work")

    # V3: loss function
    ap.add_argument("--loss_fn", type=str, default="focal",
                    choices=["bce", "focal", "ranking", "hybrid"],
                    help="Loss function (default: focal).")
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--rank_weight", type=float, default=0.3)
    ap.add_argument("--rank_temperature", type=float, default=1.0)
    ap.add_argument("--rank_subsample", type=int, default=50000)

    # V3: hard negative mining
    ap.add_argument("--hard_neg_fraction", type=float, default=0.5,
                    help="Fraction of negatives sampled proportional to fire_clim (0=uniform).")

    # V3: decoder context augmentation
    ap.add_argument("--decoder_ctx", action="store_true",
                    help="Augment decoder input with static spatial context + lead time.\n"
                         "Appends to every decoder timestep:\n"
                         "  - Static channels from encoder (fire_clim, population, slope,\n"
                         "    burn_age, burn_count) — tells decoder WHERE it's predicting\n"
                         "  - Lead time sin/cos encoding — tells decoder WHEN (day 14 vs 46)\n"
                         "  - Day-of-year sin/cos — tells decoder WHAT SEASON\n"
                         "Increases dec_dim by n_static*P² + 4.")

    # V3: evaluation
    ap.add_argument("--cluster_eval", action="store_true",
                    help="Enable cluster-level Top-K metrics.")
    ap.add_argument("--cluster_min_size", type=int, default=1)
    ap.add_argument("--per_lead_eval", action="store_true")
    ap.add_argument("--full_val", action="store_true",
                    help="Evaluate ALL val windows (slow but stable metrics).")

    # Model
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--neg_buffer", type=int, default=0)

    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_max_batches", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_ratio", type=float, default=20.0)
    ap.add_argument("--pos_weight_cap", type=float, default=10.0)
    ap.add_argument("--max_pos_pairs", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default="outputs/cache_v3")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--chunk_patches", type=int, default=200)
    ap.add_argument("--load_to_ram", action="store_true")
    ap.add_argument("--load_train_to_ram", action="store_true")
    ap.add_argument("--fire_season_only", action="store_true")
    ap.add_argument("--fire_season_months", type=str, default="4,5,6,7,8,9,10")
    ap.add_argument("--load_val_to_ram", action="store_true")
    ap.add_argument("--skip_val", action="store_true")
    ap.add_argument("--skip_forecast", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prep_only", action="store_true")
    ap.add_argument("--eval_epochs", action="store_true")
    ap.add_argument("--eval_n_windows", type=int, default=20)
    ap.add_argument("--mem_limit_pct", type=float, default=90.0)
    ap.add_argument("--log_interval", type=int, default=500)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--pred_batch_size", type=int, default=256)
    ap.add_argument("--forecast_only", action="store_true")
    ap.add_argument("--forecast_years", type=str, default=None)
    ap.add_argument("--val_lift_k", type=int, default=5000)
    ap.add_argument("--val_lift_sample_wins", type=int, default=20)

    # Decoder mode
    ap.add_argument("--decoder", type=str, default="oracle",
                    choices=["oracle", "zeros", "random", "climatology", "s2s", "s2s_legacy"])
    ap.add_argument("--s2s_cache", type=str, default=None)
    ap.add_argument("--s2s_full_cache", type=str, default=None)
    ap.add_argument("--dec_dim", type=int, default=None)
    ap.add_argument("--s2s_max_issue_lag", type=int, default=3)

    # Architecture
    ap.add_argument("--use_patch_embed", action="store_true")
    ap.add_argument("--mlp_dec_embed", action="store_true")
    ap.add_argument("--random_encoder", action="store_true")

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------
    # Parse active channels
    # ----------------------------------------------------------------
    CHANNEL_NAMES = [c.strip() for c in args.channels.split(",")]
    N_CHANNELS = len(CHANNEL_NAMES)
    for ch in CHANNEL_NAMES:
        if ch not in V3_CHANNEL_DEFS:
            raise ValueError(f"Unknown channel: {ch}. Available: {list(V3_CHANNEL_DEFS.keys())}")

    # ----------------------------------------------------------------
    # Config & paths
    # ----------------------------------------------------------------
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})

    fwi_dir = get_path(cfg, "fwi_dir")
    obs_root = get_path(cfg, "observation_dir") if "observation_dir" in paths_cfg \
               else get_path(cfg, "ecmwf_dir")
    hotspot_csv = get_path(cfg, "hotspot_csv")
    output_dir = os.path.join(get_path(cfg, "output_dir"), f"{args.run_name}_fire_prob")
    ckpt_dir = os.path.join(get_path(cfg, "checkpoint_dir"), args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    fire_clim_path = args.fire_climatology_tif or paths_cfg.get("fire_climatology_tif")
    lightning_dir = paths_cfg.get("lightning_dir", "data/lightning")
    ndvi_dir = paths_cfg.get("ndvi_dir", "data/ndvi_data")
    terrain_dir = paths_cfg.get("terrain_dir", "data/terrain")
    burn_scars_dir = paths_cfg.get("burn_scars_dir", "data/burn_scars")
    precip_dir = paths_cfg.get("precip_dir", "data/era5_precip")
    deep_soil_dir = args.deep_soil_dir or paths_cfg.get("deep_soil_dir", "data/era5_deep_soil")
    population_tif = args.population_tif or paths_cfg.get("population_tif", "data/population_density.tif")

    def _date(s):
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))

    data_start_date = _date(args.data_start)
    pred_start_date = _date(args.pred_start)
    pred_end_date = _date(args.pred_end)
    in_days = args.in_days
    lead_start = args.lead_start
    lead_end = args.lead_end
    if args.decoder == "s2s" and lead_end > 45:
        lead_end = 45
    decoder_days = lead_end - lead_start + 1

    print("\n" + "=" * 70)
    print("S2S HOTSPOT TRANSFORMER V3  [Focal Loss, Hard Negatives]")
    print("=" * 70)
    print(f"  Channels          : {N_CHANNELS} — {', '.join(CHANNEL_NAMES)}")
    print(f"  Loss function     : {args.loss_fn}"
          + (f" (alpha={args.focal_alpha}, gamma={args.focal_gamma})" if "focal" in args.loss_fn else ""))
    print(f"  Hard neg fraction : {args.hard_neg_fraction}")
    print(f"  Cluster eval      : {args.cluster_eval}")
    print(f"  decoder           : {args.decoder}")
    print(f"  data_start        : {data_start_date}")
    print(f"  pred_start        : {pred_start_date}")
    print(f"  pred_end          : {pred_end_date}")
    print(f"  lead range        : {lead_start}–{lead_end} (decoder_days={decoder_days})")
    print(f"  epochs / batch    : {args.epochs} / {args.batch_size}  lr={args.lr}")
    print("=" * 70)

    # ----------------------------------------------------------------
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
    deep_soil_dict = _build_flat_file_dict(deep_soil_dir, "deep_soil") if "deep_soil" in CHANNEL_NAMES else {}
    precip_dict = _build_flat_file_dict(precip_dir, "tp") if "precip_def" in CHANNEL_NAMES else {}

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
    # STEP 3  Grid dimensions & streaming per-channel stats
    # ----------------------------------------------------------------
    with rasterio.open(fwi_paths[0]) as src:
        profile = src.profile
        H, W = src.height, src.width
    print(f"\n[STEP 3] Grid: T={T}  H={H}  W={W}  Channels={N_CHANNELS}")

    # Load static channels
    static_arrays = {}
    if "fire_clim" in CHANNEL_NAMES:
        static_arrays["fire_clim"] = _load_static_channel(fire_clim_path, H, W, "fire_clim")
    if "population" in CHANNEL_NAMES:
        static_arrays["population"] = _load_static_channel(population_tif, H, W, "population")
    if "slope" in CHANNEL_NAMES:
        slope_path = os.path.join(terrain_dir, "slope.tif")
        static_arrays["slope"] = _load_static_channel(slope_path, H, W, "slope")

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
    if args.cache_dir:
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

        # 2t stats
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
                    "2d": dew_dict, "tcw": tcw_dict, "sm20": sm20_dict,
                    "st20": st20_dict, "lightning": lightning_dict,
                    "deep_soil": deep_soil_dict, "u10": u10_dict,
                    "v10": v10_dict, "CAPE": cape_dict,
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
                # precip_def: use precip stats as proxy
                if precip_dict:
                    _paths = [precip_dict[d] for d in aligned_dates[:train_end_idx] if d in precip_dict]
                    if _paths:
                        m, s, f = _stream_channel_stats(_paths[:50])
                    else:
                        m, s, f = 0.0, 1.0, 0.0
                else:
                    m, s, f = 0.0, 1.0, 0.0
                ch_stats.append((ch_name, m, max(s, 1e-6), f))
                print(f"  {ch_name:12s}  mean={m:8.3f}  std={s:8.3f}  (proxy)")
            elif ch_def["type"] == "annual":
                # burn_age: log1p transform, typical range 0-4
                ch_stats.append((ch_name, 1.5, 1.0, 1.5))
                print(f"  {ch_name:12s}  mean=1.500  std=1.000  (default)")

        meteo_means = np.array([s[1] for s in ch_stats], dtype=np.float32)
        meteo_stds = np.array([max(s[2], 1e-6) for s in ch_stats], dtype=np.float32)
        fills = np.array([s[3] for s in ch_stats], dtype=np.float32)

        if stats_path:
            np.save(stats_path, np.stack([meteo_means, meteo_stds]))

    # ----------------------------------------------------------------
    # STEP 4  Load and rasterize CWFIS hotspots (same as V2)
    # ----------------------------------------------------------------
    print(f"\n[STEP 4] Loading CWFIS hotspot records...")
    hotspot_df = load_hotspot_data(hotspot_csv)
    print(f"  Total records: {len(hotspot_df):,}")

    r = args.dilate_radius
    fire_cache_key = None
    if r > 0 and args.cache_dir:
        fire_cache_key = os.path.join(args.cache_dir,
                                      f"fire_dilated_r{r}_{aligned_dates[0]}_{aligned_dates[-1]}_{H}x{W}.npy")

    if fire_cache_key and os.path.exists(fire_cache_key) and not args.overwrite:
        print(f"  Loading cached fire_stack: {fire_cache_key}")
        fire_stack = np.load(fire_cache_key)
        if fire_stack.shape[0] > T:
            fire_stack = fire_stack[:T]
    else:
        fire_stack = rasterize_hotspots_batch(hotspot_df, aligned_dates, profile)
        if r > 0:
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            disk = (xx ** 2 + yy ** 2 <= r ** 2)
            print(f"  Dilating: radius={r} px...")
            for t in range(T):
                if fire_stack[t].any():
                    fire_stack[t] = binary_dilation(fire_stack[t], structure=disk).astype(np.uint8)
            if fire_cache_key:
                os.makedirs(args.cache_dir, exist_ok=True)
                np.save(fire_cache_key, fire_stack)

    print(f"  fire_stack: {fire_stack.shape}  positive_rate={fire_stack.mean():.4%}")

    # ----------------------------------------------------------------
    # STEP 5  Log stats
    # ----------------------------------------------------------------
    print(f"\n[STEP 5] Normalisation stats ({N_CHANNELS} channels):")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name:12s}  mean={meteo_means[i]:8.3f}  std={meteo_stds[i]:8.3f}")
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
    else:
        dec_dim_base = enc_dim
    out_dim = P * P

    # Decoder context augmentation: static channels + lead time encoding
    n_ctx_channels = 0
    ctx_extra_dim = 0
    if args.decoder_ctx:
        n_ctx_channels = sum(1 for name in CHANNEL_NAMES if name in DECODER_CTX_CHANNELS)
        ctx_extra_dim = n_ctx_channels * out_dim + 4  # static patches + lead/season sin/cos
        print(f"  [decoder_ctx] {n_ctx_channels} static channels + 4 lead/season dims "
              f"= +{ctx_extra_dim} to dec_dim")
    dec_dim = dec_dim_base + ctx_extra_dim

    meteo_mmap_gb = T * n_patches * enc_dim * 2 / 1e9
    print(f"\n[STEP 6] Streaming meteo_patched → float16 memmap")
    print(f"  n_patches={n_patches}  enc_dim={enc_dim}  ~{meteo_mmap_gb:.1f} GB")

    mmap_path = None
    if args.cache_dir:
        mmap_key = (f"meteo_v3_p{P}_C{N_CHANNELS}_T{T}"
                    f"_{aligned_dates[0]}_{aligned_dates[-1]}_pf.dat")
        mmap_path = os.path.join(args.cache_dir, mmap_key)

    if mmap_path and os.path.exists(mmap_path) and not args.overwrite:
        print(f"  Loading cached memmap: {mmap_path}")
        meteo_patched = np.memmap(mmap_path, dtype='float16', mode='r',
                                  shape=(n_patches, T, enc_dim))
        if args.load_to_ram:
            print(f"  Copying to RAM...")
            meteo_patched = np.array(meteo_patched)
    else:
        # Build time-first, then transpose
        tf_path = mmap_path.replace("_pf.dat", "_tf.dat") if mmap_path else None

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

        _fallback_fwi = None
        _fallback_t2m = None
        t0_mmap = time.time()

        for t_idx in range(T):
            cur_date = aligned_dates[t_idx]
            frame = np.zeros((H, W, N_CHANNELS), dtype=np.float32)

            for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
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

                elif ch_name == "NDVI":
                    frame[..., ch_idx] = _interpolate_ndvi(cur_date, ndvi_index, ndvi_cache, H, W)

                elif ch_name in ("2d", "tcw", "sm20", "st20", "lightning",
                                 "deep_soil", "u10", "v10", "CAPE"):
                    _daily_dicts = {
                        "2d": dew_dict, "tcw": tcw_dict, "sm20": sm20_dict,
                        "st20": st20_dict, "lightning": lightning_dict,
                        "deep_soil": deep_soil_dict, "u10": u10_dict,
                        "v10": v10_dict, "CAPE": cape_dict,
                    }
                    ch_dict = _daily_dicts.get(ch_name, {})
                    if cur_date in ch_dict:
                        arr = _read_tif_safe(ch_dict[cur_date], None)
                        if arr is not None:
                            frame[..., ch_idx] = np.nan_to_num(arr, nan=float(fills[ch_idx]))

                elif ch_name == "precip_def":
                    # Accumulate precipitation for rolling deficit
                    if cur_date in precip_dict:
                        p_arr = _read_tif_safe(precip_dict[cur_date], None)
                        if p_arr is not None:
                            precip_deque.append(np.nan_to_num(p_arr, nan=0.0))
                    if len(precip_deque) > 0:
                        # Simple deficit: negative of accumulated precip (less rain = higher deficit)
                        rolling_sum = np.sum(precip_deque, axis=0)
                        frame[..., ch_idx] = -rolling_sum  # negative = deficit

                elif ch_name == "burn_age":
                    year = cur_date.year
                    raw = None
                    if year in burn_scar_raw:
                        raw = burn_scar_raw[year]
                    elif burn_scar_raw:
                        nearest = min(burn_scar_raw.keys(), key=lambda y: abs(y - year))
                        raw = burn_scar_raw[nearest]
                    if raw is not None:
                        frame[..., ch_idx] = _encode_burn_age(raw, args.burn_age_encoding)

                elif ch_name == "burn_count":
                    year = cur_date.year
                    if year in burn_count_arrays:
                        # log1p to compress range (0-10+ fires → 0-2.4)
                        frame[..., ch_idx] = np.log1p(burn_count_arrays[year])
                    elif burn_count_arrays:
                        nearest = min(burn_count_arrays.keys(), key=lambda y: abs(y - year))
                        frame[..., ch_idx] = np.log1p(burn_count_arrays[nearest])

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
        else:
            _tmp = np.ascontiguousarray(meteo_tf.transpose(1, 0, 2))
            del meteo_tf
            gc.collect()
            meteo_patched = _tmp

    # ----------------------------------------------------------------
    # STEP 7  Patchify fire labels
    # ----------------------------------------------------------------
    print(f"\n[STEP 7] Pre-computing fire patches...")
    fire_gb = T * n_patches * out_dim / 1e9
    fire_cache_path = None
    if args.cache_dir:
        fire_cache_path = os.path.join(args.cache_dir,
                                       f"fire_patched_v3_r{args.dilate_radius}"
                                       f"_{aligned_dates[0]}_{aligned_dates[-1]}"
                                       f"_{T}x{n_patches}x{out_dim}.dat")

    if fire_cache_path and os.path.exists(fire_cache_path) and not args.overwrite:
        _fp_T = os.path.getsize(fire_cache_path) // (n_patches * out_dim)
        fire_patched = np.memmap(fire_cache_path, dtype='uint8', mode='r',
                                 shape=(_fp_T, n_patches, out_dim))
        if _fp_T > T:
            fire_patched = fire_patched[:T]
        print(f"  Loaded cached fire_patched: {fire_cache_path}")
    else:
        if fire_cache_path:
            fire_patched = np.memmap(fire_cache_path, dtype='uint8', mode='w+',
                                     shape=(T, n_patches, out_dim))
        else:
            fire_patched = np.empty((T, n_patches, out_dim), dtype=np.uint8)
        for t_idx in range(T):
            frame_f = fire_stack[t_idx, :Hc, :Wc, np.newaxis].astype(np.float32)
            fire_patched[t_idx] = _patchify_frame(frame_f, P).astype(np.uint8)
        if fire_cache_path:
            fire_patched.flush()
        print(f"  fire_patched: {fire_patched.shape}  ~{fire_gb:.1f} GB")
    del fire_stack

    if args.prep_only:
        print("\n[--prep_only] All caches built. Exiting.")
        return

    # Build windows
    all_windows = _build_s2s_windows(T, in_days, lead_start, lead_end)
    train_wins = [w for w in all_windows if aligned_dates[w[3] - 1] < pred_start_date]
    val_wins = [w for w in all_windows if aligned_dates[w[1]] >= pred_start_date]
    print(f"\n  Windows: train={len(train_wins)}  val={len(val_wins)}")

    all_train_window_dates = [str(aligned_dates[w[1]]) for w in train_wins]
    all_val_window_dates = [str(aligned_dates[w[1]]) for w in val_wins]

    # S2S cache setup (delegate to v2 if needed)
    s2s_cache = None
    s2s_full_cache = None
    date_to_s2s_idx = None
    date_to_s2s_lag = None
    s2s_means = None
    s2s_stds = None

    # ----------------------------------------------------------------
    # STEP 7b  Positive pairs + HARD NEGATIVE MINING
    # ----------------------------------------------------------------
    print(f"\n[STEP 7b] Building pos/neg pairs (hard_neg_fraction={args.hard_neg_fraction})...")
    t0_filter = time.time()

    pos_pairs = []
    for win_i, (hs, he, ts, te) in enumerate(train_wins):
        # Vectorized: read entire window at once, max over (time, pixels) per patch
        # fire_patched shape: (T, n_patches, P²), read [ts:te] slice = (dec_days, n_patches, P²)
        win_fire = np.array(fire_patched[ts:te, :, :])   # load to RAM once
        has_fire = win_fire.max(axis=(0, 2)) > 0          # (n_patches,) bool
        for patch_i in np.where(has_fire)[0]:
            pos_pairs.append((win_i, int(patch_i)))
        if win_i % 200 == 0 or win_i == len(train_wins) - 1:
            print(f"  scanned {win_i+1}/{len(train_wins)} windows  "
                  f"pos: {len(pos_pairs):,}  ({time.time()-t0_filter:.0f}s)")

    if len(pos_pairs) == 0:
        raise RuntimeError("No positive pairs found. Check hotspot_csv and date alignment.")

    rng = np.random.default_rng(args.seed)
    if args.max_pos_pairs > 0 and len(pos_pairs) > args.max_pos_pairs:
        idx_cap = rng.choice(len(pos_pairs), size=args.max_pos_pairs, replace=False)
        pos_pairs = [pos_pairs[i] for i in idx_cap]

    total_pairs = len(train_wins) * n_patches
    pos_flat = np.array([w * n_patches + p for w, p in pos_pairs], dtype=np.int64)
    pos_mask = np.zeros(total_pairs, dtype=bool)
    pos_mask[pos_flat] = True

    # Spatial buffer
    if args.neg_buffer > 0:
        _buf = args.neg_buffer
        _struct = np.ones((2 * _buf + 1, 2 * _buf + 1), dtype=bool)
        _nrow, _ncol = grid
        for win_i in range(len(train_wins)):
            _win_offset = win_i * n_patches
            _win_pos = pos_mask[_win_offset:_win_offset + n_patches]
            if not _win_pos.any():
                continue
            _pos_grid = _win_pos.reshape(_nrow, _ncol)
            _buf_grid = binary_dilation(_pos_grid, structure=_struct)
            pos_mask[_win_offset:_win_offset + n_patches] |= _buf_grid.reshape(-1)

    neg_flat = np.where(~pos_mask)[0]
    n_neg_target = min(int(len(pos_pairs) * args.neg_ratio), len(neg_flat))

    # Compute fire_clim per patch for hard negative mining
    fire_clim_per_patch = np.zeros(n_patches, dtype=np.float32)
    if "fire_clim" in CHANNEL_NAMES and "fire_clim" in static_arrays:
        fc = static_arrays["fire_clim"][:Hc, :Wc]
        fc_patched = _patchify_frame(fc[:, :, np.newaxis], P)  # (n_patches, P*P)
        fire_clim_per_patch = fc_patched.mean(axis=1)

    # Hard negative mining
    chosen = _sample_hard_negatives(
        neg_flat, fire_clim_per_patch, n_patches,
        args.hard_neg_fraction, n_neg_target, rng
    )
    neg_wins = (chosen // n_patches).astype(np.int32)
    neg_patches = (chosen % n_patches).astype(np.int32)

    pos_arr = np.array(pos_pairs, dtype=np.int32)
    neg_arr = np.column_stack([neg_wins, neg_patches]).astype(np.int32)
    all_pairs = np.vstack([pos_arr, neg_arr])
    rng.shuffle(all_pairs)

    print(f"  Pos: {len(pos_pairs):,}  Neg: {len(chosen):,}  "
          f"Total: {len(all_pairs):,}")

    # MemoryGuard
    mem_guard = MemoryGuard(limit_pct=args.mem_limit_pct, interval=15,
                            meteo_gb=meteo_mmap_gb, fire_gb=fire_gb,
                            batch_size=args.batch_size)
    if _PSUTIL_OK and args.mem_limit_pct > 0:
        mem_guard.start()

    # ----------------------------------------------------------------
    # STEP 8  Build loss criterion
    # ----------------------------------------------------------------
    print(f"\n[STEP 8] Building loss: {args.loss_fn}")
    # Compute pos_weight
    pos_pixels = 0
    neg_pixels_in_pos = 0
    for win_i, patch_i in pos_pairs:
        hs, he, ts, te = train_wins[win_i]
        pf = fire_patched[ts:te, patch_i, :]
        pos_pixels += int(pf.sum())
        neg_pixels_in_pos += pf.size - int(pf.sum())
    neg_pixels_total = neg_pixels_in_pos + len(chosen) * out_dim * decoder_days
    raw_ratio = neg_pixels_total / max(pos_pixels, 1)
    pos_weight_val = min(raw_ratio, args.pos_weight_cap)
    print(f"  pos_weight={pos_weight_val:.2f} (raw={raw_ratio:.1f}, cap={args.pos_weight_cap})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pw_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    if args.loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    elif args.loss_fn == "focal":
        criterion = FocalBCELoss(alpha=args.focal_alpha, gamma=args.focal_gamma,
                                 pos_weight=pw_tensor)
    elif args.loss_fn == "ranking":
        criterion = ApproxNDCGLoss(temperature=args.rank_temperature,
                                   subsample=args.rank_subsample)
    elif args.loss_fn == "hybrid":
        criterion = HybridLoss(rank_weight=args.rank_weight,
                               focal_alpha=args.focal_alpha,
                               focal_gamma=args.focal_gamma,
                               pos_weight=pw_tensor,
                               rank_temperature=args.rank_temperature,
                               rank_subsample=args.rank_subsample)
    print(f"  Loss: {criterion.__class__.__name__}")

    # ----------------------------------------------------------------
    # Build datasets & dataloaders
    # ----------------------------------------------------------------
    meteo_train = meteo_patched
    train_wins_eff = train_wins
    all_pairs_eff = all_pairs
    train_window_dates_eff = all_train_window_dates

    # Optionally load to RAM (same logic as v2, simplified)
    if args.load_train_to_ram and not args.load_to_ram:
        train_T_max = max(te for hs, he, ts, te in train_wins)
        if args.fire_season_only:
            fire_months = set(int(m) for m in args.fire_season_months.split(","))
            valid_mask = []
            for hs, he, ts, te in train_wins:
                ok = all(aligned_dates[t].month in fire_months for t in range(hs, he)) and \
                     all(aligned_dates[t].month in fire_months for t in range(ts, te))
                valid_mask.append(ok)
            valid_idxs = [i for i, v in enumerate(valid_mask) if v]
            filtered_wins = [train_wins[i] for i in valid_idxs]
            t_needed = set()
            for hs, he, ts, te in filtered_wins:
                t_needed.update(range(hs, he))
                t_needed.update(range(ts, te))
            t_indices = np.array(sorted(t_needed), dtype=np.int32)
            t_remap = np.full(train_T_max + 2, -1, dtype=np.int32)
            for new_t, orig_t in enumerate(t_indices):
                t_remap[orig_t] = new_t
            train_wins_eff = [(int(t_remap[hs]), int(t_remap[he - 1] + 1),
                               int(t_remap[ts]), int(t_remap[te - 1] + 1))
                              for hs, he, ts, te in filtered_wins]
            train_window_dates_eff = [all_train_window_dates[i] for i in valid_idxs]
            win_remap = np.full(len(train_wins), -1, dtype=np.int32)
            for new_i, old_i in enumerate(valid_idxs):
                win_remap[old_i] = new_i
            keep = np.isin(all_pairs[:, 0], valid_idxs)
            all_pairs_eff = all_pairs[keep].copy()
            all_pairs_eff[:, 0] = win_remap[all_pairs_eff[:, 0]]
            print(f"\n  [fire_season_only] {len(filtered_wins)}/{len(train_wins)} windows, "
                  f"{len(t_indices)} T indices")
        else:
            t_indices = np.arange(train_T_max + 1, dtype=np.int32)

        print(f"  Copying train data to RAM...")
        meteo_train = np.array(meteo_patched[:, t_indices, :])
        print(f"  [OK] {meteo_train.nbytes/1e9:.1f} GB in RAM")

    train_ds = S2SHotspotDatasetMixed(
        meteo_train, fire_patched, train_wins_eff, hw, grid, all_pairs_eff,
        decoder_mode=args.decoder, dec_dim=dec_dim,
        s2s_cache=s2s_cache, date_to_s2s_idx=date_to_s2s_idx,
        window_dates=train_window_dates_eff, patch_size=P,
        s2s_means=s2s_means, s2s_stds=s2s_stds,
        date_to_s2s_lag=date_to_s2s_lag, s2s_max_lag=args.s2s_max_issue_lag,
        s2s_full_cache=s2s_full_cache,
    )

    _prefetch = 4 if args.num_workers > 0 else None
    _persistent = args.num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          pin_memory=True, num_workers=args.num_workers,
                          persistent_workers=_persistent, prefetch_factor=_prefetch)

    # Val lift windows (fire season only)
    _lift_months = {4, 5, 6, 7, 8, 9, 10}
    _lift_filtered = [
        (w, d) for w, d in zip(val_wins, all_val_window_dates)
        if all(aligned_dates[t].month in _lift_months for t in range(w[0], w[1]))
        and all(aligned_dates[t].month in _lift_months for t in range(w[2], w[3]))
    ]
    val_wins_lift = [x[0] for x in _lift_filtered]
    val_wins_lift_dates = [x[1] for x in _lift_filtered]
    print(f"  Val Lift windows: {len(val_wins_lift)}/{len(val_wins)} (fire season)")

    # ----------------------------------------------------------------
    # STEP 9  Build model & train
    # ----------------------------------------------------------------
    print(f"\n[STEP 9] Training on {device}...")
    patch_dim_enc = enc_dim
    patch_dim_dec = dec_dim
    patch_dim_out = out_dim

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
        n_patches=(n_patches if args.use_patch_embed else 0),
        mlp_dec_embed=args.mlp_dec_embed,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Train samples: {len(train_ds):,}  batches/epoch: {len(train_dl):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_lift_k = -1.0
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    start_epoch = 1

    if args.resume:
        _epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        if _epoch_ckpts:
            _latest = _epoch_ckpts[-1]
            print(f"  Resuming from: {_latest}")
            _ckpt = torch.load(_latest, map_location=device, weights_only=False)
            model.load_state_dict(_ckpt["model_state"])
            if "optimizer_state" in _ckpt:
                optimizer.load_state_dict(_ckpt["optimizer_state"])
            if "scheduler_state" in _ckpt:
                scheduler.load_state_dict(_ckpt["scheduler_state"])
            if "scaler_state" in _ckpt and amp_enabled:
                scaler.load_state_dict(_ckpt["scaler_state"])
            start_epoch = _ckpt["epoch"] + 1
            best_val_lift_k = _ckpt.get("best_val_lift_k_global", -1.0)
            print(f"  Resumed: epoch {start_epoch}, best_lift={best_val_lift_k:.2f}")
            del _ckpt

    # Pre-compute decoder context if enabled
    _dec_ctx_np = None      # (n_patches, n_static * P²) float16 or None
    _lead_time_enc = None   # (dec_days, 4) tensor or None
    if args.decoder_ctx:
        _dec_ctx_np, _ctx_indices = _build_decoder_ctx_static(
            meteo_patched, CHANNEL_NAMES, out_dim)
        if _dec_ctx_np is not None:
            print(f"  [decoder_ctx] Built static context: {_dec_ctx_np.shape} "
                  f"(channels: {[CHANNEL_NAMES[i] for i in _ctx_indices]})")
        _lead_time_enc = _build_lead_time_encoding(
            decoder_days, lead_start, device=device)
        print(f"  [decoder_ctx] Lead time encoding: {_lead_time_enc.shape}")

    n_batches = len(train_dl)
    train_started = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        if mem_guard.triggered:
            print(f"  MemoryGuard triggered — stopping.")
            break

        model.train()
        t0_ep = time.time()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, (xb_enc, xb_dec, yb, patch_ids) in enumerate(train_dl):
            xb_enc = xb_enc.to(device, dtype=torch.float32, non_blocking=True)
            xb_dec = xb_dec.to(device, dtype=torch.float32, non_blocking=True)
            if args.random_encoder:
                xb_enc = torch.randn_like(xb_enc)
            yb = yb.to(device, non_blocking=True)
            if args.label_smoothing > 0:
                yb = yb * (1 - args.label_smoothing) + 0.5 * args.label_smoothing

            # Augment decoder with static context + lead time
            if args.decoder_ctx and _dec_ctx_np is not None:
                _batch_pids = patch_ids.numpy()
                _ctx_batch = torch.from_numpy(
                    _dec_ctx_np[_batch_pids].astype(np.float32)
                ).to(device)
                xb_dec = _augment_decoder(xb_dec, _ctx_batch, _lead_time_enc)

            _pids = patch_ids.to(device) if args.use_patch_embed else None
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp_enabled):
                logits = model(xb_enc, xb_dec, _pids)
                loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                optimizer.zero_grad()
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb_enc.size(0)
            train_samples += xb_enc.size(0)

            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                pct = (batch_idx + 1) / n_batches
                elapsed = time.time() - t0_ep
                print(f"    ep{epoch} [{batch_idx+1:5d}/{n_batches} {pct*100:4.1f}%]  "
                      f"loss={train_loss/max(train_samples,1):.4f}  "
                      f"{(batch_idx+1)/elapsed:.1f}b/s")

        if train_samples == 0:
            print(f"  Epoch {epoch}: all NaN — stopping.")
            break
        train_loss /= train_samples

        # Validation
        val_lift_k = 0.0
        val_prec_k = 0.0
        if not args.skip_val and val_wins_lift:
            _m = _compute_val_lift_k_v3(
                model, meteo_patched, fire_patched, val_wins_lift,
                n_patches, k=args.val_lift_k,
                n_sample_wins=args.val_lift_sample_wins,
                chunk=256, device=device,
                decoder_mode=args.decoder, dec_dim=dec_dim,
                val_win_dates=val_wins_lift_dates, patch_size=P,
                use_patch_embed=args.use_patch_embed,
                random_encoder=args.random_encoder,
                cluster_eval=args.cluster_eval,
                cluster_min_size=args.cluster_min_size,
                hw=hw, grid=grid, full_val=args.full_val,
                per_lead_eval=args.per_lead_eval,
            )
            val_lift_k = _m["lift_k"]
            val_prec_k = _m["precision_k"]
            val_roc_auc = _m.get("roc_auc", 0.0)
            val_brier = _m.get("brier", 0.0)
            cluster_str = ""
            if args.cluster_eval and "cluster_lift_k" in _m:
                cluster_str = (f"  cluster: Lift={_m['cluster_lift_k']:.2f}x  "
                               f"n_clusters={_m.get('n_clusters', 0)}")
            if args.per_lead_eval and "per_lead_lift" in _m:
                _pl = _m["per_lead_lift"]
                _pp = _m["per_lead_precision"]
                print(f"  per-lead Lift@{args.val_lift_k}: "
                      + "  ".join(f"d{i}={v:.1f}x" for i, v in enumerate(_pl)))
                print(f"  per-lead Prec@{args.val_lift_k}: "
                      + "  ".join(f"d{i}={v:.3f}" for i, v in enumerate(_pp)))

        epoch_time = time.time() - t0_ep
        print(f"\n  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.6f}  "
              f"Lift@{args.val_lift_k}={val_lift_k:.2f}x  "
              f"prec={val_prec_k:.4f}  "
              f"ROC-AUC={val_roc_auc:.4f}  "
              f"Brier={val_brier:.6f}  "
              f"({epoch_time/60:.1f}m)")
        if cluster_str:
            print(f"  {cluster_str}")

        scheduler.step()

        is_new_best = (not args.skip_val) and (val_lift_k > best_val_lift_k)
        if is_new_best:
            best_val_lift_k = val_lift_k

        ckpt_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_lift_k_global": best_val_lift_k,
            "meteo_means": meteo_means,
            "meteo_stds": meteo_stds,
            "patch_dim_enc": patch_dim_enc,
            "patch_dim_dec": patch_dim_dec,
            "patch_dim_out": patch_dim_out,
            "hw": hw,
            "grid": grid,
            "args": vars(args),
            "channel_names": CHANNEL_NAMES,
            "n_channels": N_CHANNELS,
            "s2s_means": s2s_means,
            "s2s_stds": s2s_stds,
        }
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pt")
        torch.save(ckpt_payload, epoch_ckpt)
        if is_new_best:
            torch.save(ckpt_payload, best_ckpt)
            print(f"  ★ New best Lift@{args.val_lift_k}={val_lift_k:.2f}x → best_model.pt")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE  [V3, {N_CHANNELS} channels, {args.loss_fn} loss]")
    print(f"  Best Lift@{args.val_lift_k}: {best_val_lift_k:.2f}x")
    print(f"  Checkpoint: {best_ckpt}")
    print(f"{'='*70}")

    mem_guard.shutdown()


if __name__ == "__main__":
    main()
