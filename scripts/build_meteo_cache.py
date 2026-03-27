"""
Build Meteo + Fire Patch Cache (standalone)
============================================
Pre-builds the large memmap caches that the training script (Step 3/6/7) needs:
  1. meteo_p16_C8_T{T}_{start}_{end}_pf.dat   (~287 GB, float16, patch-first)
  2. meteo_p16_C8_T{T}_{start}_{end}_stats.npy (normalisation stats)
  3. fire_patched_r14_{start}_{end}_{T}x{n}x256.dat (~18 GB, uint8)
  4. fire_dilated_r14_{start}_{end}_{H}x{W}.npy (fire stack)

Run this on a CPU node BEFORE submitting GPU training jobs.

Usage:
    python scripts/build_meteo_cache.py \
        --config configs/paths_narval.yaml \
        --cache-dir /scratch/jiaqi217/meteo_cache \
        --data-start 2018-01-01 \
        --pred-end 2025-12-31 \
        --lead-end 45 \
        --patch-size 16 \
        --dilate-radius 14 \
        --chunk-patches 500
"""

import argparse
import gc
import glob
import os
import re
import sys
import time
from datetime import date, timedelta

import numpy as np
import rasterio
import yaml

# ── Constants matching training script ──
N_CHANNELS = 8   # 7 dynamic + 1 static (fire_clim)
CHANNEL_NAMES = ["FWI", "2t", "2d", "FFMC", "DMC", "DC", "BUI", "fire_clim"]


def extract_date_from_filename(filename):
    match = re.search(r'(\d{8})', os.path.basename(filename))
    if match:
        try:
            from datetime import datetime
            return datetime.strptime(match.group(1), '%Y%m%d').date()
        except ValueError:
            return None
    return None


def _build_file_dict(directory, prefix):
    """Build {date: filepath} dict. Flat first, then subdirectory."""
    result = {}
    if not os.path.isdir(directory):
        return result
    # Flat
    files = glob.glob(os.path.join(directory, f"{prefix}_*.tif"))
    # Subdirectory fallback
    if not files:
        files = glob.glob(os.path.join(directory, prefix, f"{prefix}_*.tif"))
    # FWI special: *.tif
    if not files and prefix == "fwi":
        files = glob.glob(os.path.join(directory, "*.tif"))
    for f in files:
        d = extract_date_from_filename(f)
        if d is not None:
            result[d] = f
    return result


def _build_era5_dict(directory, prefix):
    """Build {date: filepath} for ERA5 (tif or grib)."""
    result = {}
    if not os.path.isdir(directory):
        return result
    # TIF
    for pat in [os.path.join(directory, prefix, f"{prefix}_*.tif"),
                os.path.join(directory, f"{prefix}_*.tif")]:
        files = glob.glob(pat)
        if files:
            for f in files:
                d = extract_date_from_filename(f)
                if d:
                    result[d] = f
            return result
    # GRIB: era5_sfc_YYYY_MM_DD.grib
    for f in glob.glob(os.path.join(directory, "era5_sfc_*.grib")):
        m = re.search(r'era5_sfc_(\d{4})_(\d{2})_(\d{2})\.grib', os.path.basename(f))
        if m:
            try:
                result[date(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = f
            except ValueError:
                pass
    return result


def _read_tif_safe(path, fallback_arr=None):
    """Read single-band TIF, return 2D array. On failure, return fallback."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        return arr
    except Exception as e:
        print(f"  [WARN] Failed to read {path}: {e}")
        if fallback_arr is not None:
            return fallback_arr.copy()
        return None


def _read_era5_grib(path, variable, H, W, fallback_arr=None):
    """Read ERA5 grib file, reproject/resample to target grid."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        # If shape matches target, return directly
        if arr.shape == (H, W):
            return arr
        # Otherwise need resampling — for now just warn
        print(f"  [WARN] ERA5 shape {arr.shape} != target ({H},{W}), skipping: {path}")
        return fallback_arr.copy() if fallback_arr is not None else np.zeros((H, W), np.float32)
    except Exception as e:
        print(f"  [WARN] Failed to read {path}: {e}")
        return fallback_arr.copy() if fallback_arr is not None else np.zeros((H, W), np.float32)


def _patchify_frame(frame, P):
    """(H, W, C) -> (n_patches, P*P*C) float32. H,W must be divisible by P."""
    H, W, C = frame.shape
    nph = H // P
    npw = W // P
    return (frame
            .reshape(nph, P, npw, P, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(nph * npw, P * P * C))


def _transpose_tf_to_pf(tf_path, pf_path, T, n_patches, enc_dim, chunk=500):
    """Transpose (T, n_patches, enc_dim) -> (n_patches, T, enc_dim) on disk."""
    tf = np.memmap(tf_path, dtype='float16', mode='r', shape=(T, n_patches, enc_dim))
    pf = np.memmap(pf_path, dtype='float16', mode='w+', shape=(n_patches, T, enc_dim))
    t0 = time.time()
    for p0 in range(0, n_patches, chunk):
        p1 = min(p0 + chunk, n_patches)
        pf[p0:p1] = tf[:, p0:p1, :].transpose(1, 0, 2).copy()
        if (p0 // chunk) % 10 == 0 or p1 == n_patches:
            elapsed = time.time() - t0
            pct = p1 / n_patches * 100
            eta = elapsed / max(p1, 1) * (n_patches - p1) / 60
            print(f"    transpose: {p1}/{n_patches} patches ({pct:.0f}%)  "
                  f"{elapsed:.0f}s  ~{eta:.0f} min left", flush=True)
    pf.flush()
    del tf, pf


def main():
    ap = argparse.ArgumentParser(description="Pre-build meteo + fire caches for training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-start", default="2018-01-01")
    ap.add_argument("--pred-end", default="2025-12-31")
    ap.add_argument("--lead-end", type=int, default=45)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--dilate-radius", type=int, default=14)
    ap.add_argument("--chunk-patches", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_start = date.fromisoformat(args.data_start)
    pred_end = date.fromisoformat(args.pred_end)
    required_end = pred_end + timedelta(days=args.lead_end + 5)
    P = args.patch_size

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", cfg)

    fwi_dir = paths["fwi_dir"]
    obs_dir = paths.get("observation_dir", paths.get("ecmwf_dir", ""))
    ffmc_dir = paths["ffmc_dir"]
    dmc_dir = paths["dmc_dir"]
    dc_dir = paths["dc_dir"]
    bui_dir = paths.get("bui_dir", "")
    hotspot_csv = paths["hotspot_csv"]
    fire_clim_tif = paths.get("fire_climatology_tif", "")

    print("=" * 70)
    print("METEO + FIRE CACHE BUILDER")
    print(f"  data_start   : {data_start}")
    print(f"  pred_end     : {pred_end}")
    print(f"  required_end : {required_end} (pred_end + {args.lead_end + 5} days)")
    print(f"  cache_dir    : {args.cache_dir}")
    print(f"  patch_size   : {P}")
    print("=" * 70)

    # ── STEP 1: Build file index ──
    print("\n[STEP 1] Building file index...", flush=True)
    fwi_dict = _build_file_dict(fwi_dir, "fwi")
    t2m_dict = _build_era5_dict(obs_dir, "2t")
    d2m_dict = _build_era5_dict(obs_dir, "2d")
    ffmc_dict = _build_file_dict(ffmc_dir, "ffmc")
    dmc_dict = _build_file_dict(dmc_dir, "dmc")
    dc_dict = _build_file_dict(dc_dir, "dc")
    bui_dict = _build_file_dict(bui_dir, "bui")

    for name, d in [("FWI", fwi_dict), ("2t", t2m_dict), ("2d", d2m_dict),
                    ("FFMC", ffmc_dict), ("DMC", dmc_dict), ("DC", dc_dict),
                    ("BUI", bui_dict)]:
        print(f"  {name:6s}: {len(d):,d} days")

    # ── STEP 2: Align dates ──
    print("\n[STEP 2] Aligning dates...", flush=True)
    fwi_paths, t2m_paths, d2m_paths = [], [], []
    ffmc_paths, dmc_paths, dc_paths, bui_paths = [], [], [], []
    aligned_dates = []

    cur = data_start
    while cur <= required_end:
        if (cur in fwi_dict and cur in t2m_dict and cur in d2m_dict
                and cur in ffmc_dict and cur in dmc_dict
                and cur in dc_dict and cur in bui_dict):
            fwi_paths.append(fwi_dict[cur])
            t2m_paths.append(t2m_dict[cur])
            d2m_paths.append(d2m_dict[cur])
            ffmc_paths.append(ffmc_dict[cur])
            dmc_paths.append(dmc_dict[cur])
            dc_paths.append(dc_dict[cur])
            bui_paths.append(bui_dict[cur])
            aligned_dates.append(cur)
        cur += timedelta(days=1)

    T = len(aligned_dates)
    print(f"  Aligned dates: {T}  ({aligned_dates[0]} → {aligned_dates[-1]})")

    # Grid dimensions from first TIF
    with rasterio.open(fwi_paths[0]) as src:
        H, W = src.height, src.width
    Hc = H - H % P
    Wc = W - W % P
    nph = Hc // P
    npw = Wc // P
    n_patches = nph * npw
    enc_dim = P * P * N_CHANNELS
    out_dim = P * P

    print(f"  Grid: {H}×{W}  crop: {Hc}×{Wc}  patches: {nph}×{npw}={n_patches}")
    print(f"  enc_dim={enc_dim}  out_dim={out_dim}")

    # ── Cache file paths ──
    os.makedirs(args.cache_dir, exist_ok=True)
    mmap_key = f"meteo_p{P}_C{N_CHANNELS}_T{T}_{aligned_dates[0]}_{aligned_dates[-1]}"
    pf_path = os.path.join(args.cache_dir, mmap_key + "_pf.dat")
    tf_path = os.path.join(args.cache_dir, mmap_key + "_tf.dat")
    stats_path = os.path.join(args.cache_dir, mmap_key + "_stats.npy")

    meteo_gb = T * n_patches * enc_dim * 2 / 1e9
    print(f"\n  Meteo cache: {pf_path}")
    print(f"  Size: ~{meteo_gb:.1f} GB")

    # ── STEP 3: Compute normalisation stats (training split only) ──
    # Use same split point as training: pred_start_date default = 2022-05-01
    pred_start_date = date(2022, 5, 1)
    train_end_idx = 0
    for i, d in enumerate(aligned_dates):
        if d < pred_start_date:
            train_end_idx = i + 1

    if os.path.exists(stats_path) and not args.overwrite:
        print(f"\n[STEP 3] Loading cached stats: {stats_path}", flush=True)
        _s = np.load(stats_path)
        meteo_means = _s[0].astype(np.float32)
        meteo_stds = _s[1].astype(np.float32)
    else:
        print(f"\n[STEP 3] Computing per-channel stats (train split: {aligned_dates[0]} → "
              f"{aligned_dates[train_end_idx - 1]}, {train_end_idx} days)...", flush=True)

        _dyn_paths_all = [fwi_paths, t2m_paths, d2m_paths,
                          ffmc_paths, dmc_paths, dc_paths, bui_paths]
        ch_sums = np.zeros(7, dtype=np.float64)
        ch_sq_sums = np.zeros(7, dtype=np.float64)
        ch_counts = np.zeros(7, dtype=np.float64)

        for t_idx in range(train_end_idx):
            for ch_idx, ch_paths in enumerate(_dyn_paths_all):
                arr = _read_tif_safe(ch_paths[t_idx])
                if arr is None:
                    continue
                valid = np.isfinite(arr)
                ch_sums[ch_idx] += np.sum(arr[valid])
                ch_sq_sums[ch_idx] += np.sum(arr[valid] ** 2)
                ch_counts[ch_idx] += np.sum(valid)
            if t_idx % 200 == 0 or t_idx == train_end_idx - 1:
                print(f"  stats: day {t_idx+1}/{train_end_idx}", flush=True)

        meteo_means = np.zeros(N_CHANNELS, dtype=np.float32)
        meteo_stds = np.ones(N_CHANNELS, dtype=np.float32)
        for i in range(7):
            if ch_counts[i] > 0:
                meteo_means[i] = ch_sums[i] / ch_counts[i]
                var = ch_sq_sums[i] / ch_counts[i] - meteo_means[i] ** 2
                meteo_stds[i] = max(np.sqrt(max(var, 0)), 1e-6)

        # Fire climatology stats
        if fire_clim_tif and os.path.exists(fire_clim_tif):
            with rasterio.open(fire_clim_tif) as src:
                fc = src.read(1).astype(np.float32)
            fc = np.nan_to_num(fc, nan=0.0)
            meteo_means[7] = np.mean(fc)
            meteo_stds[7] = max(np.std(fc), 1e-6)

        np.save(stats_path, np.stack([meteo_means, meteo_stds]))
        print(f"  Saved stats: {stats_path}")

    fills = meteo_means.copy()
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name:12s}  mean={meteo_means[i]:8.3f}  std={meteo_stds[i]:8.3f}")

    # ── Load fire climatology ──
    fire_clim = np.zeros((H, W), dtype=np.float32)
    if fire_clim_tif and os.path.exists(fire_clim_tif):
        with rasterio.open(fire_clim_tif) as src:
            fire_clim = src.read(1).astype(np.float32)
        fire_clim = np.nan_to_num(fire_clim, nan=0.0)
        print(f"  fire_clim loaded: {fire_clim.shape}  max={fire_clim.max():.3f}")

    # ── STEP 4: Check if pf.dat already exists ──
    if os.path.exists(pf_path) and not args.overwrite:
        expected = T * n_patches * enc_dim * 2
        actual = os.path.getsize(pf_path)
        if actual >= expected * 0.99:
            print(f"\n[STEP 4] Meteo cache already exists: {pf_path} ({actual/1e9:.1f} GB)")
            print(f"  Skipping meteo build. Use --overwrite to rebuild.")
        else:
            print(f"\n[STEP 4] Meteo cache exists but incomplete ({actual/1e9:.1f}/{expected/1e9:.1f} GB). Rebuilding.")
            os.remove(pf_path)

    if not os.path.exists(pf_path):
        # ── Check if tf.dat already exists (resume from interrupted transpose) ──
        tf_exists = False
        if os.path.exists(tf_path):
            expected_tf = T * n_patches * enc_dim * 2
            actual_tf = os.path.getsize(tf_path)
            if actual_tf >= expected_tf * 0.99:
                tf_exists = True
                print(f"\n[STEP 4] Found complete time-first file: {tf_path} ({actual_tf/1e9:.1f} GB)")
                print(f"  Skipping streaming, going straight to transpose.")

        if not tf_exists:
            # ── Stream day-by-day into time-first memmap ──
            print(f"\n[STEP 4] Streaming {T} days → time-first memmap: {tf_path}", flush=True)
            print(f"  Size: ~{meteo_gb:.1f} GB", flush=True)

            meteo_tf = np.memmap(tf_path, dtype='float16', mode='w+',
                                 shape=(T, n_patches, enc_dim))

            _dyn_paths_all = [fwi_paths, t2m_paths, d2m_paths,
                              ffmc_paths, dmc_paths, dc_paths, bui_paths]
            _fallbacks = [None] * 7
            t0 = time.time()

            for t_idx in range(T):
                frame = np.empty((Hc, Wc, N_CHANNELS), dtype=np.float32)
                for ch_idx, ch_paths in enumerate(_dyn_paths_all):
                    arr = _read_tif_safe(ch_paths[t_idx], _fallbacks[ch_idx])
                    if arr is None:
                        arr = np.full((H, W), fills[ch_idx], dtype=np.float32)
                    _fallbacks[ch_idx] = arr
                    arr = np.nan_to_num(arr, nan=float(fills[ch_idx]),
                                        posinf=float(fills[ch_idx]),
                                        neginf=float(fills[ch_idx]))
                    frame[..., ch_idx] = arr[:Hc, :Wc]
                frame[..., 7] = fire_clim[:Hc, :Wc]
                frame -= meteo_means
                frame /= meteo_stds
                np.clip(frame, -10.0, 10.0, out=frame)
                meteo_tf[t_idx] = _patchify_frame(frame, P).astype(np.float16)

                if t_idx % 100 == 0 or t_idx == T - 1:
                    elapsed = time.time() - t0
                    eta_min = elapsed / max(t_idx, 1) * (T - t_idx) / 60
                    print(f"  day {t_idx+1:4d}/{T}  "
                          f"({elapsed:.0f}s elapsed  ~{eta_min:.0f} min left)", flush=True)

            meteo_tf.flush()
            del meteo_tf
            gc.collect()
            print(f"  Time-first complete: {tf_path} ({os.path.getsize(tf_path)/1e9:.1f} GB)")

        # ── Transpose to patch-first ──
        print(f"\n[STEP 5] Transposing to patch-first: {pf_path}", flush=True)
        _transpose_tf_to_pf(tf_path, pf_path, T, n_patches, enc_dim,
                             chunk=args.chunk_patches)
        print(f"  Transpose complete. Deleting temp file.")
        os.remove(tf_path)
        print(f"  Saved: {pf_path} ({os.path.getsize(pf_path)/1e9:.1f} GB)")

    # ── STEP 6: Build fire_patched cache ──
    print(f"\n[STEP 6] Building fire patches...", flush=True)

    fire_cache_key = (f"fire_patched_r{args.dilate_radius}"
                      f"_{aligned_dates[0]}_{aligned_dates[-1]}"
                      f"_{T}x{n_patches}x{out_dim}.dat")
    fire_cache_path = os.path.join(args.cache_dir, fire_cache_key)
    fire_gb = T * n_patches * out_dim / 1e9

    # Also need the fire_dilated stack
    fire_stack_key = (f"fire_dilated_r{args.dilate_radius}"
                      f"_{aligned_dates[0]}_{aligned_dates[-1]}"
                      f"_{H}x{W}.npy")
    fire_stack_path = os.path.join(args.cache_dir, fire_stack_key)

    if os.path.exists(fire_cache_path) and not args.overwrite:
        print(f"  Fire cache already exists: {fire_cache_path}")
        print(f"  Skipping. Use --overwrite to rebuild.")
    else:
        # Need fire_stack (dilated binary labels)
        if os.path.exists(fire_stack_path) and not args.overwrite:
            print(f"  Loading cached fire_stack: {fire_stack_path}", flush=True)
            fire_stack = np.load(fire_stack_path, mmap_mode='r')
            print(f"  Loaded: {fire_stack.shape}")
        else:
            print(f"  Building fire_stack from hotspot CSV...", flush=True)
            print(f"  Hotspot CSV: {hotspot_csv}")
            import pandas as pd
            from scipy.ndimage import binary_dilation

            # Read hotspot CSV
            df = pd.read_csv(hotspot_csv)
            print(f"  Total hotspot records: {len(df):,d}")

            # Get CRS and transform from reference TIF
            with rasterio.open(fwi_paths[0]) as src:
                transform = src.transform
                crs = src.crs

            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

            # Build date index
            date_to_idx = {d: i for i, d in enumerate(aligned_dates)}

            fire_stack = np.zeros((T, H, W), dtype=np.uint8)
            n_mapped = 0

            for _, row in df.iterrows():
                try:
                    lat = float(row.get('lat', row.get('latitude', 0)))
                    lon = float(row.get('lon', row.get('longitude', 0)))
                    date_str = str(row.get('rep_date', row.get('date', '')))
                    d = date.fromisoformat(date_str[:10])
                except (ValueError, TypeError):
                    continue

                if d not in date_to_idx:
                    continue

                x, y = transformer.transform(lon, lat)
                col = int((x - transform.c) / transform.a)
                row_idx = int((y - transform.f) / transform.e)

                if 0 <= row_idx < H and 0 <= col < W:
                    fire_stack[date_to_idx[d], row_idx, col] = 1
                    n_mapped += 1

            print(f"  Mapped {n_mapped:,d} hotspot points to grid")

            # Dilate
            if args.dilate_radius > 0:
                print(f"  Dilating with radius={args.dilate_radius}...", flush=True)
                r = args.dilate_radius
                yy, xx = np.ogrid[-r:r+1, -r:r+1]
                kernel = (xx**2 + yy**2 <= r**2).astype(np.uint8)
                t0 = time.time()
                for t_idx in range(T):
                    if fire_stack[t_idx].any():
                        fire_stack[t_idx] = binary_dilation(
                            fire_stack[t_idx], structure=kernel).astype(np.uint8)
                    if t_idx % 500 == 0 or t_idx == T - 1:
                        print(f"    dilate {t_idx+1}/{T}  ({time.time()-t0:.0f}s)", flush=True)

            pos_rate = fire_stack.sum() / fire_stack.size * 100
            print(f"  fire_stack: {fire_stack.shape}  positive_rate={pos_rate:.4f}%")

            np.save(fire_stack_path, fire_stack)
            print(f"  Saved: {fire_stack_path}")

        # Patchify fire labels
        print(f"  Patchifying fire labels → {fire_cache_path}", flush=True)
        print(f"  Size: ~{fire_gb:.1f} GB", flush=True)
        fire_patched = np.memmap(fire_cache_path, dtype='uint8', mode='w+',
                                 shape=(T, n_patches, out_dim))
        t0 = time.time()
        for t_idx in range(T):
            frame_f = fire_stack[t_idx, :Hc, :Wc, np.newaxis].astype(np.float32)
            fire_patched[t_idx] = _patchify_frame(frame_f, P).astype(np.uint8)
            if t_idx % 500 == 0 or t_idx == T - 1:
                print(f"    fire patch {t_idx+1}/{T}  ({time.time()-t0:.0f}s)", flush=True)
        fire_patched.flush()
        del fire_patched
        print(f"  Saved: {fire_cache_path} ({os.path.getsize(fire_cache_path)/1e9:.1f} GB)")

    print(f"\n{'=' * 70}")
    print("ALL CACHES BUILT SUCCESSFULLY")
    print(f"  meteo pf : {pf_path}")
    print(f"  stats    : {stats_path}")
    print(f"  fire     : {fire_cache_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
