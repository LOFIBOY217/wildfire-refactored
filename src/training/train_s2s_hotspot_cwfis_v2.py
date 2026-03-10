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
METEO_SCALE   = 12.7   # int8 quantization: float16 [-10,10] → int8 [-127,127]


# ------------------------------------------------------------------ #
# Datasets  (identical to V1 — fully generic w.r.t. enc_dim)
# ------------------------------------------------------------------ #

class S2SHotspotDatasetMixed(Dataset):
    """
    Training dataset: pos_pairs (patches with ≥1 fire pixel in target window)
    mixed with neg_ratio × neg_pairs (pure-background patches).
    """

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid, all_pairs):
        self.meteo     = meteo_patched
        self.fire      = fire_patched
        self.windows   = windows
        self.hw        = hw
        self.grid      = grid
        self.all_pairs = all_pairs

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        win_i, patch_i = self.all_pairs[idx]
        hs, he, ts, te = self.windows[win_i]
        # meteo layout: (n_patches, T, enc_dim) — patch-first for sequential read
        # fire  layout: (T, n_patches, P*P)     — time-first (small, OK)
        x_enc = self.meteo[patch_i, hs:he, :].astype(np.float32) * (1.0 / METEO_SCALE)
        x_dec = self.meteo[patch_i, ts:te, :].astype(np.float32) * (1.0 / METEO_SCALE)
        y     = self.fire[ts:te, patch_i, :].astype(np.float32)
        return (
            torch.from_numpy(x_enc),
            torch.from_numpy(x_dec),
            torch.from_numpy(y),
        )


class S2SHotspotDatasetUnfiltered(Dataset):
    """Unfiltered dataset for validation — all patches, no pos/neg sampling."""

    def __init__(self, meteo_patched, fire_patched, windows, hw, grid):
        self.meteo     = meteo_patched
        self.fire      = fire_patched
        self.windows   = windows
        self.hw        = hw
        self.grid      = grid
        self.n_patches = meteo_patched.shape[0]   # (n_patches, T, enc_dim)

    def __len__(self):
        return len(self.windows) * self.n_patches

    def __getitem__(self, idx):
        win_i   = idx // self.n_patches
        patch_i = idx %  self.n_patches
        hs, he, ts, te = self.windows[win_i]
        # meteo layout: (n_patches, T, enc_dim) — patch-first for sequential read
        # fire  layout: (T, n_patches, P*P)     — time-first (small, OK)
        x_enc = self.meteo[patch_i, hs:he, :].astype(np.float32) * (1.0 / METEO_SCALE)
        x_dec = self.meteo[patch_i, ts:te, :].astype(np.float32) * (1.0 / METEO_SCALE)
        y     = self.fire[ts:te, patch_i, :].astype(np.float32)
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


def _convert_pf_to_int8(pf_path, p8_path, n_patches, T, enc_dim, chunk=500):
    """
    Convert an existing float16 patch-first memmap to int8 in-place on disk.
    Reads _pf.dat in chunks, quantizes with METEO_SCALE, writes _p8.dat.
    No TIF re-loading needed — ~30–60 min vs 5 hours from scratch.
    """
    print(f"  Converting float16 → int8  ({n_patches} patches × {T} × {enc_dim})")
    src = np.memmap(pf_path, dtype='float16', mode='r',  shape=(n_patches, T, enc_dim))
    dst = np.memmap(p8_path, dtype='int8',    mode='w+', shape=(n_patches, T, enc_dim))
    t0  = time.time()
    for i in range(0, n_patches, chunk):
        j       = min(i + chunk, n_patches)
        dst[i:j] = np.clip(
            np.round(np.array(src[i:j]).astype(np.float32) * METEO_SCALE),
            -128, 127
        ).astype(np.int8)
        if i % 5000 == 0 or j == n_patches:
            pct = j / n_patches * 100
            print(f"  int8 convert {j:6d}/{n_patches}  ({pct:.0f}%  {time.time()-t0:.0f}s)")
    dst.flush()
    del src, dst
    gc.collect()
    print(f"  int8 conversion done in {time.time()-t0:.0f}s  →  {p8_path}")


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
                 meteo_gb=0.0, fire_gb=0.0, num_workers=0, batch_size=128):
        super().__init__(daemon=True)
        self.limit_pct   = limit_pct
        self.interval    = interval
        self.meteo_gb    = meteo_gb
        self.fire_gb     = fire_gb
        self.num_workers = num_workers
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
            w   = self.num_workers
            print(f"\n{sep}")
            print(f"  ⚠  MEMORY GUARD TRIGGERED — training will stop after this epoch.")
            print(f"  RAM: {used_gb:.1f} GB used / {total_gb:.1f} GB total "
                  f"({used_pct:.1f}%  ≥  limit {self.limit_pct:.0f}%)")
            print(f"  Available: {avail_gb:.1f} GB")
            print()
            print(f"  Likely contributors:")
            print(f"    meteo_patched  (int8 OS page-cache):  up to {self.meteo_gb:.0f} GB")
            print(f"    fire_patched   (OS page-cache):       up to {self.fire_gb:.0f} GB")
            if w > 0:
                print(f"    DataLoader workers × {w}:             "
                      f"~{w * (self.meteo_gb + self.fire_gb) * 0.03:.0f} GB  "
                      f"(shared OS pages — should be small)")
            print(f"    Model + optimizer + GPU activations:  ~2–4 GB")
            print()
            print(f"  Suggestions to reduce RAM:")
            print(f"    1) Lower --mem_limit_pct if you want an earlier warning")
            print(f"    2) Drop --num_workers (currently {w}) to 0 or 2")
            print(f"    3) Reboot between runs to flush OS page-cache")
            print(f"       (_pf.dat is now auto-deleted after int8 conversion)")
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
                        n_patches, k, n_sample_wins, chunk, device):
    """
    Compute Lift@K on a random sample of validation windows.

    Samples *n_sample_wins* windows from *val_wins*, runs patch-level
    inference, aggregates across lead days
      • prob  → mean  (overall fire risk across the 14–46 day window)
      • label → max   (any lead day with fire = positive pixel)
    then computes precision@K and Lift@K globally across all sampled pixels.

    This is cheap: for n_sample_wins=20 and n_patches≈24 310 it runs
    ~20 × 95 GPU batches ≈ 10–30 seconds.

    Returns:
        (lift_k: float, precision_at_k: float)
        lift_k = precision@K / baseline_fire_rate
        Returns (0.0, 0.0) if no fire pixels found in sample.
    """
    model.eval()
    rng = np.random.default_rng(0)   # fixed seed → same sample every epoch

    if len(val_wins) > n_sample_wins:
        idx         = rng.choice(len(val_wins), size=n_sample_wins, replace=False)
        sample_wins = [val_wins[i] for i in sorted(idx)]
    else:
        sample_wins = val_wins

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for hs, he, ts, te in sample_wins:
            prob_chunks = []
            for cs in range(0, n_patches, chunk):
                ce = min(cs + chunk, n_patches)
                xb_enc = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[cs:ce, hs:he, :].astype(np.float32) * (1.0 / METEO_SCALE)
                    )
                ).to(device)
                xb_dec = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[cs:ce, ts:te, :].astype(np.float32) * (1.0 / METEO_SCALE)
                    )
                ).to(device)
                logits = model(xb_enc, xb_dec)  # (chunk, dec_days, P²)
                prob_chunks.append(torch.sigmoid(logits).cpu().numpy())

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
        return 0.0, 0.0

    k_eff        = min(k, n_total)
    top_idx      = np.argpartition(all_probs, -k_eff)[-k_eff:]
    precision_k  = float(all_labels[top_idx].sum()) / k_eff
    baseline     = n_fire / n_total
    lift_k       = precision_k / baseline if baseline > 0 else 0.0

    return lift_k, precision_k


# ------------------------------------------------------------------ #
# Forecast-only helper
# ------------------------------------------------------------------ #

def _run_forecast_only(args, cfg, fwi_dir, obs_root, ffmc_dir, dmc_dir, dc_dir,
                       bui_dir, fire_clim_path, output_dir, ckpt_dir):
    """
    Load the best checkpoint and generate forecast TIFs for --forecast_years.
    Uses ERA5 oracle decoder (precomputed future meteo) — same normalisation
    stats as training.
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
            dec_fut  = meteo_y[base_idx + lead_start: base_idx + lead_end + 1]

            enc_patches, pred_hw, pred_grid = patchify(enc_hist, P)
            dec_patches, _, _               = patchify(dec_fut,  P)

            chunk, n_p = args.pred_batch_size, enc_patches.shape[0]
            prob_list = []
            with torch.no_grad():
                for cs in range(0, n_p, chunk):
                    ce  = min(cs + chunk, n_p)
                    xb_enc = torch.from_numpy(enc_patches[cs:ce].copy()).float().to(device)
                    xb_dec = torch.from_numpy(dec_patches[cs:ce].copy()).float().to(device)
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

    # Training
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--neg_ratio",    type=float, default=20.0)
    ap.add_argument("--pos_weight_cap", type=float, default=10.0)
    ap.add_argument("--max_pos_pairs", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default="outputs/cache",
                    help="Directory to cache the dilated fire_stack (.npy). "
                         "Set to '' to disable caching.")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader worker processes (default=4). "
                         "Set to 0 to disable multiprocessing.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Force rebuild all caches (fire_stack, meteo memmap, fire_patched).")
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
                               "s2s_hotspot_cwfis_v2_fire_prob")
    ckpt_dir    = os.path.join(get_path(cfg, "checkpoint_dir"),
                               "s2s_hotspot_cwfis_v2")
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
        "decoder_mode":       "era5_oracle",
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
    decoder_days    = lead_end - lead_start + 1

    print("\n" + "=" * 70)
    print("S2S HOTSPOT TRANSFORMER V2  [8 channels + ERA5 Oracle Decoder]")
    print("=" * 70)
    print(f"  Channels          : {N_CHANNELS} — {', '.join(CHANNEL_NAMES)}")
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

    fire_stack = rasterize_hotspots_batch(hotspot_df, aligned_dates, profile)
    pos_rate   = fire_stack.mean()
    print(f"  Fire stack shape     : {fire_stack.shape}")
    print(f"  Mean fire pixel rate : {pos_rate:.6%}  (raw hotspot points)")
    run_meta["positive_rate_raw"] = float(pos_rate)

    # -- Spatial dilation --
    if args.dilate_radius > 0:
        r = args.dilate_radius
        cache_key  = (f"fire_dilated_r{r}"
                      f"_{aligned_dates[0]}_{aligned_dates[-1]}"
                      f"_{H}x{W}.npy")
        cache_path = (os.path.join(args.cache_dir, cache_key)
                      if args.cache_dir else None)

        if cache_path and os.path.exists(cache_path):
            print(f"\n  Loading cached dilated fire_stack: {cache_path}")
            t0_load = time.time()
            fire_stack = np.load(cache_path)
            pos_rate_dil = fire_stack.mean()
            print(f"  Loaded in {time.time()-t0_load:.0f}s  "
                  f"positive_rate={pos_rate_dil:.4%}")
        else:
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
    dec_dim   = enc_dim              # ERA5 oracle: same channels for decoder
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
        p8_path    = mmap_path.replace("_pf.dat", "_p8.dat")   # int8 version
        stats_path = mmap_path.replace("_pf.dat", "_stats.npy")
    else:
        mmap_path  = None
        p8_path    = None
        stats_path = None

    _stats_ok = stats_path and os.path.exists(stats_path)

    if p8_path and os.path.exists(p8_path) and _stats_ok:
        # ── Load existing int8 patch-first cache (preferred) ─────────
        print(f"  Loading cached memmap (int8 patch-first): {p8_path}")
        meteo_patched = np.memmap(p8_path, dtype='int8', mode='r',
                                  shape=(n_patches, T, enc_dim))
        _saved_stats  = np.load(stats_path)
        meteo_means   = _saved_stats[0]
        meteo_stds    = _saved_stats[1]
        print(f"  shape={meteo_patched.shape}  "
              f"({os.path.getsize(p8_path)/1e9:.1f} GB on disk)")
    elif (mmap_path and os.path.exists(mmap_path) and _stats_ok):
        if p8_path and not os.path.exists(p8_path) and not args.overwrite:
            # ── float16 exists but int8 does not → convert ───────────
            _convert_pf_to_int8(mmap_path, p8_path, n_patches, T, enc_dim)
            meteo_patched = np.memmap(p8_path, dtype='int8', mode='r',
                                      shape=(n_patches, T, enc_dim))
            # ── Delete _pf.dat: no longer needed; frees 238 GB RAM ───
            try:
                os.remove(mmap_path)
                gc.collect()
                _freed = n_patches * T * enc_dim * 2 / 1e9
                print(f"  Deleted float16 cache ({_freed:.0f} GB freed): "
                      f"{os.path.basename(mmap_path)}")
            except OSError as _e:
                print(f"  [WARN] Could not delete float16 cache: {_e}")
        else:
            # ── Load float16 cache (overwrite mode or no p8_path) ────
            print(f"  Loading cached memmap (patch-first): {mmap_path}")
            meteo_patched = np.memmap(mmap_path, dtype='float16', mode='r',
                                      shape=(n_patches, T, enc_dim))
        _saved_stats  = np.load(stats_path)
        meteo_means   = _saved_stats[0]
        meteo_stds    = _saved_stats[1]
        _src = p8_path if (p8_path and os.path.exists(p8_path)) else mmap_path
        print(f"  shape={meteo_patched.shape}  "
              f"({os.path.getsize(_src)/1e9:.1f} GB on disk)")
    else:
        # ── Build: stream day-by-day into time-first temp file,
        #          then transpose to patch-first final file ──────────
        if mmap_path:
            tf_path = mmap_path.replace("_pf.dat", "_tf.dat")
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

        if mmap_path:
            meteo_tf.flush()
            del meteo_tf   # Windows: release write handle before transpose/delete
            gc.collect()
            np.save(stats_path, np.stack([meteo_means, meteo_stds]))
            print(f"  Saved time-first: {tf_path}  "
                  f"({os.path.getsize(tf_path)/1e9:.1f} GB)")
            # ── Transpose to patch-first ─────────────────────────────
            print(f"\n  Transposing to patch-first layout → {mmap_path}")
            _transpose_tf_to_pf(tf_path, mmap_path, T, n_patches, enc_dim)
            print(f"  Transpose complete. Deleting temp file: {tf_path}")
            os.remove(tf_path)
            print(f"  Saved patch-first: {mmap_path}  "
                  f"({os.path.getsize(mmap_path)/1e9:.1f} GB)")
            # ── Immediately convert to int8 so Dataset dequantization is correct ──
            if p8_path:
                _convert_pf_to_int8(mmap_path, p8_path, n_patches, T, enc_dim)
                meteo_patched = np.memmap(p8_path, dtype='int8', mode='r',
                                          shape=(n_patches, T, enc_dim))
                # ── Delete _pf.dat: no longer needed; frees ~238 GB RAM ──
                # Deleting the file causes Windows to immediately reclaim
                # the 238 GB of standby page-cache pages, preventing OOM
                # when DataLoader workers start reading _p8.dat.
                try:
                    os.remove(mmap_path)
                    gc.collect()
                    _freed = n_patches * T * enc_dim * 2 / 1e9
                    print(f"  Deleted float16 cache ({_freed:.0f} GB freed): "
                          f"{os.path.basename(mmap_path)}")
                except OSError as _e:
                    print(f"  [WARN] Could not delete float16 cache: {_e}")
            else:
                meteo_patched = np.memmap(mmap_path, dtype='float16', mode='r',
                                          shape=(n_patches, T, enc_dim))
        else:
            # In-memory: transpose then quantize to int8 (consistent with Dataset dequantization)
            _tmp = np.ascontiguousarray(meteo_tf.transpose(1, 0, 2))
            del meteo_tf; gc.collect()
            meteo_patched = np.clip(
                np.round(_tmp.astype(np.float32) * METEO_SCALE), -128, 127
            ).astype(np.int8)
            del _tmp; gc.collect()

    # ----------------------------------------------------------------
    # STEP 7  Patchify fire labels (uint8 disk memmap)
    # ----------------------------------------------------------------
    # Using memmap instead of np.empty avoids duplicating 15 GB into every
    # DataLoader worker process (Windows spawns separate processes; memmap
    # workers share OS file pages → ~60 GB less RAM with num_workers=4).
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

    # Build S2S windows
    all_windows = _build_s2s_windows(T, in_days, lead_start, lead_end)
    train_wins  = [w for w in all_windows
                   if aligned_dates[w[1] - 1] < pred_start_date]
    val_wins    = [w for w in all_windows
                   if aligned_dates[w[0]] >= pred_start_date]
    print(f"\n  S2S windows built (enc_days={in_days}, gap={lead_start-1}, "
          f"target_days={decoder_days})")
    print(f"  Total: {len(all_windows)}  train: {len(train_wins)}  val: {len(val_wins)}")

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

    max_available_neg = total_pairs - len(pos_pairs)
    n_neg_target      = min(int(len(pos_pairs) * args.neg_ratio), max_available_neg)
    print(f"  neg_target={n_neg_target:,}  max_available_neg={max_available_neg:,}")

    pos_flat = np.array([w * n_patches + p for w, p in pos_pairs], dtype=np.int64)
    pos_mask = np.zeros(total_pairs, dtype=bool)
    pos_mask[pos_flat] = True
    neg_flat    = np.where(~pos_mask)[0]
    chosen      = rng.choice(neg_flat, size=n_neg_target, replace=False)
    neg_wins    = (chosen // n_patches).astype(np.int32)
    neg_patches = (chosen %  n_patches).astype(np.int32)
    neg_pairs   = list(zip(neg_wins.tolist(), neg_patches.tolist()))

    all_pairs = list(pos_pairs) + neg_pairs
    rng.shuffle(all_pairs)
    print(f"  Neg pairs sampled: {len(neg_pairs):,}  (neg_ratio={args.neg_ratio})")
    print(f"  Total train pairs: {len(all_pairs):,}")
    print(f"  Sampling time: {time.time()-t0_filter:.0f}s")
    run_meta["pos_pairs"]   = len(pos_pairs)
    run_meta["neg_pairs"]   = len(neg_pairs)
    run_meta["total_pairs"] = len(all_pairs)

    # ----------------------------------------------------------------
    # Memory Guard  (background thread — starts now, checked each epoch)
    # ----------------------------------------------------------------
    _meteo_guard_gb = (n_patches * T * enc_dim) / 1e9   # int8
    _fire_guard_gb  = (T * n_patches * out_dim) / 1e9   # uint8
    mem_guard = MemoryGuard(
        limit_pct   = args.mem_limit_pct,
        interval    = 15,
        meteo_gb    = _meteo_guard_gb,
        fire_gb     = _fire_guard_gb,
        num_workers = args.num_workers,
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
    neg_pixels_total = neg_pixels_in_pos + len(neg_pairs) * out_dim * decoder_days
    raw_ratio        = neg_pixels_total / max(pos_pixels, 1)
    pos_weight_val   = min(raw_ratio, args.pos_weight_cap)
    print(f"  pos_pixels : {pos_pixels:,}")
    print(f"  neg_pixels : {neg_pixels_total:,}")
    print(f"  raw ratio  : {raw_ratio:.1f}   "
          f"pos_weight (capped {args.pos_weight_cap}) = {pos_weight_val:.2f}")
    run_meta["pos_weight"] = pos_weight_val

    # ----------------------------------------------------------------
    # Build datasets and dataloaders
    # ----------------------------------------------------------------
    train_ds = S2SHotspotDatasetMixed(
        meteo_patched, fire_patched, train_wins, hw, grid, all_pairs
    )
    if val_wins:
        val_ds = S2SHotspotDatasetUnfiltered(
            meteo_patched, fire_patched, val_wins, hw, grid
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

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=(args.num_workers > 0))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=(args.num_workers > 0))

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
        encoder_days=in_days,
        decoder_days=decoder_days,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    run_meta["n_params"] = n_params

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss  = float("inf")
    best_val_lift_k = -1.0          # checkpoint selected by Lift@K, not val_loss
    best_ckpt      = os.path.join(ckpt_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        # -- Memory guard check (background thread sets .triggered) --
        if mem_guard.triggered:
            print(f"  Epoch {epoch}: MemoryGuard triggered — stopping training early.")
            break

        # -- Training --
        model.train()
        train_loss    = 0.0
        train_samples = 0
        for xb_enc, xb_dec, yb in train_dl:
            xb_enc, xb_dec, yb = xb_enc.to(device), xb_dec.to(device), yb.to(device)
            logits = model(xb_enc, xb_dec)
            loss   = criterion(logits, yb)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            _gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(_gnorm):
                optimizer.zero_grad()
                continue
            optimizer.step()
            train_loss    += loss.item() * xb_enc.size(0)
            train_samples += xb_enc.size(0)

        if train_samples == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs}  *** ALL TRAIN BATCHES NaN — stopping.")
            break
        train_loss /= train_samples

        # -- Validation --
        model.eval()
        val_loss    = 0.0
        val_samples = 0
        _lfire_sum, _lfire_n = 0.0, 0
        _lbg_sum,   _lbg_n   = 0.0, 0
        with torch.no_grad():
            for xb_enc, xb_dec, yb in val_dl:
                xb_enc, xb_dec, yb = xb_enc.to(device), xb_dec.to(device), yb.to(device)
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

        # -- Val Lift@K  (checkpoint selection criterion) --
        val_lift_k, val_prec_k = _compute_val_lift_k(
            model, meteo_patched, fire_patched, val_wins,
            n_patches,
            k=args.val_lift_k,
            n_sample_wins=args.val_lift_sample_wins,
            chunk=args.pred_batch_size,
            device=device,
        )

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"Lift@{args.val_lift_k}={val_lift_k:.2f}x  prec={val_prec_k:.4f}")
        if _lfire_n > 0:
            print(f"           logit  fire={_lfire_sum/_lfire_n:+.3f}  "
                  f"bg={_lbg_sum/_lbg_n:+.3f}  "
                  f"({_lfire_n:,} fire px / {_lbg_n:,} bg px)")

        # Save best checkpoint by val Lift@K  (aligned with final evaluation metric)
        if val_lift_k > best_val_lift_k:
            best_val_lift_k = val_lift_k
            best_val_loss   = val_loss   # track for reference only
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "meteo_means":    meteo_means,
                "meteo_stds":     meteo_stds,
                "patch_dim_enc":  patch_dim_enc,
                "patch_dim_dec":  patch_dim_dec,
                "patch_dim_out":  patch_dim_out,
                "hw":             hw,
                "grid":           grid,
                "args":           vars(args),
                "channel_names":  CHANNEL_NAMES,
                "n_channels":     N_CHANNELS,
                "best_val_lift_k": best_val_lift_k,
            }, best_ckpt)
            print(f"           ★ New best  Lift@{args.val_lift_k}={best_val_lift_k:.2f}x  "
                  f"→ checkpoint saved")

    print(f"\n  Best val Lift@{args.val_lift_k}: {best_val_lift_k:.2f}x  "
          f"(val_loss at that epoch: {best_val_loss:.6f})  saved → {best_ckpt}")
    run_meta["best_val_lift_k"] = best_val_lift_k
    run_meta["best_val_loss"]   = best_val_loss
    mem_guard.shutdown()   # stop background memory monitor

    # ----------------------------------------------------------------
    # STEP 10  Generate forecast GeoTIFFs
    # ----------------------------------------------------------------
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
        prob_list = []
        with torch.no_grad():
            for cs in range(0, n_p, chunk):
                ce = min(cs + chunk, n_p)
                xb_enc = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[cs:ce, enc_start:enc_end, :].astype(np.float32) * (1.0 / METEO_SCALE)
                    )
                ).to(device)
                xb_dec = torch.from_numpy(
                    np.ascontiguousarray(
                        meteo_patched[cs:ce, dec_start:dec_end, :].astype(np.float32) * (1.0 / METEO_SCALE)
                    )
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
