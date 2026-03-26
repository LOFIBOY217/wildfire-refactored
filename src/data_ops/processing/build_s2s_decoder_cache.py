"""
Build S2S Decoder Cache
=======================
Converts per-date S2S forecast TIF files into a flat float16 memmap cache
suitable for fast random access during model training.

Output layout: (n_dates, n_patches, 32, 6) float16
  - n_dates   : number of YYYY-MM-DD subdirectories found in --s2s-dir
  - n_patches : (H//patch_size * W//patch_size) using cropped grid
  - 32        : lead days 14..45 (one value per lead day)
  - 6         : S2S channels (2t, 2d, tcw, sm20, st20, VPD)

Each entry is the patch MEAN of the corresponding S2S TIF band, broadcast to
all 256 pixels during training (S2S is ~28 km resolution, essentially uniform
within a 32 km patch).

Companion file: {out_file}.dates.npy — 1-D array of date strings (sorted),
used by the training script to map date strings to cache row indices.

Grid defaults (patch_size=16):
  H=2281, W=2709  ->  crop to 2272 x 2704  ->  142 x 169 = 23998 patches

Usage:
    python -m src.data_ops.processing.build_s2s_decoder_cache \\
        --s2s-dir data/s2s_processed \\
        --out-file /scratch/jiaqi217/meteo_cache/s2s_decoder_cache.dat \\
        --reference data/fwi_data/fwi_20250615.tif \\
        --patch-size 16 \\
        --workers 8
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import rasterio

S2S_N_LEADS = 32        # lead14 .. lead45
S2S_N_CHANNELS = 6     # 2t, 2d, tcw, sm20, st20, VPD
LEAD_START = 14
LEAD_END = 45


def _parse_date_dirs(s2s_dir):
    """Return sorted list of YYYY-MM-DD subdirectory names that exist in s2s_dir."""
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


def _extract_patch_means(args_tuple):
    """
    Worker function: read all 32 lead-day TIFs for one date directory,
    compute per-patch mean for each of the 6 channels.

    Returns:
        (date_str, result) where result is (32, n_patches, 6) float16
        or (date_str, None) on error.
    """
    date_str, s2s_dir, patch_size, n_patches, nph, npw, H, W = args_tuple
    Hc = H - H % patch_size
    Wc = W - W % patch_size
    date_dir = os.path.join(s2s_dir, date_str)

    result = np.zeros((S2S_N_LEADS, n_patches, S2S_N_CHANNELS), dtype=np.float32)

    for li, lead in enumerate(range(LEAD_START, LEAD_END + 1)):
        tif_path = os.path.join(date_dir, f"lead{lead:02d}.tif")
        if not os.path.exists(tif_path):
            # Leave as zeros — will be treated as missing
            continue
        try:
            with rasterio.open(tif_path) as src:
                # Read all 6 bands: shape (6, H, W)
                arr = src.read().astype(np.float32)
        except Exception as e:
            print(f"  [WARN] Failed to read {tif_path}: {e}", flush=True)
            continue

        # arr shape: (6, H_file, W_file) — use the spatial dims from file
        _, Hf, Wf = arr.shape
        Hc_eff = min(Hf, H) - (min(Hf, H) % patch_size)
        Wc_eff = min(Wf, W) - (min(Wf, W) % patch_size)
        nph_eff = Hc_eff // patch_size
        npw_eff = Wc_eff // patch_size

        # Crop and compute patch means for each channel (nanmean to handle NaN/nodata)
        for ch in range(S2S_N_CHANNELS):
            band = arr[ch, :Hc_eff, :Wc_eff]     # (Hc_eff, Wc_eff)
            # Reshape to (nph_eff, P, npw_eff, P) -> mean over patch dims
            band_r = band.reshape(nph_eff, patch_size, npw_eff, patch_size)
            patch_means = np.nanmean(band_r, axis=(1, 3))  # (nph_eff, npw_eff)
            # Fill any remaining NaN (entire-patch missing) with 0
            np.nan_to_num(patch_means, copy=False, nan=0.0)
            # Flatten to patch index
            flat = patch_means.reshape(-1)           # (n_patches_eff,)
            n_fill = min(len(flat), n_patches)
            result[li, :n_fill, ch] = flat[:n_fill]

    return date_str, result.astype(np.float16)


def build_cache(s2s_dir, out_file, reference_tif, patch_size=16, workers=8):
    """Build the (n_dates, n_patches, 32, 6) float16 cache."""

    # Discover date directories
    date_dirs = _parse_date_dirs(s2s_dir)
    if not date_dirs:
        raise RuntimeError(f"No YYYY-MM-DD subdirectories found in {s2s_dir}")
    n_dates = len(date_dirs)
    print(f"Found {n_dates} date directories: {date_dirs[0]} .. {date_dirs[-1]}", flush=True)

    # Read grid dimensions from reference TIF
    with rasterio.open(reference_tif) as src:
        H, W = src.height, src.width
    Hc = H - H % patch_size
    Wc = W - W % patch_size
    nph = Hc // patch_size
    npw = Wc // patch_size
    n_patches = nph * npw
    print(f"Grid: H={H} W={W}  crop to {Hc}x{Wc}  "
          f"patches: {nph}x{npw}={n_patches}  patch_size={patch_size}", flush=True)

    # Estimate cache size
    cache_bytes = n_dates * n_patches * S2S_N_LEADS * S2S_N_CHANNELS * 2  # float16
    print(f"Cache size: {cache_bytes / 1e9:.2f} GB  "
          f"shape=({n_dates}, {n_patches}, {S2S_N_LEADS}, {S2S_N_CHANNELS}) float16",
          flush=True)

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    # Create memmap: layout (n_dates, n_patches, 32, 6)
    cache = np.memmap(out_file, dtype='float16', mode='w+',
                      shape=(n_dates, n_patches, S2S_N_LEADS, S2S_N_CHANNELS))

    # Build work items
    work_items = [
        (date_str, s2s_dir, patch_size, n_patches, nph, npw, H, W)
        for date_str in date_dirs
    ]

    # Build date → index mapping for lookup
    date_to_idx = {d: i for i, d in enumerate(date_dirs)}

    print(f"\nProcessing {n_dates} dates with {workers} workers...", flush=True)
    t0 = __import__('time').time()

    if workers <= 1:
        # Single-process mode (easier to debug)
        for n_done, item in enumerate(work_items):
            date_str, patch_data = _extract_patch_means(item)
            idx = date_to_idx[date_str]
            cache[idx] = patch_data.transpose(1, 0, 2)  # (32,n_patches,6) -> (n_patches,32,6)
            if (n_done + 1) % 50 == 0 or n_done + 1 == n_dates:
                elapsed = __import__('time').time() - t0
                eta_min = elapsed / (n_done + 1) * (n_dates - n_done - 1) / 60
                print(f"  {n_done+1}/{n_dates}  {date_str}  "
                      f"({elapsed:.0f}s  ~{eta_min:.0f} min left)", flush=True)
    else:
        # Multi-process mode
        with Pool(processes=workers) as pool:
            n_done = 0
            for date_str, patch_data in pool.imap_unordered(
                    _extract_patch_means, work_items, chunksize=4):
                idx = date_to_idx[date_str]
                cache[idx] = patch_data.transpose(1, 0, 2)  # (32,n_patches,6) -> (n_patches,32,6)
                n_done += 1
                if n_done % 50 == 0 or n_done == n_dates:
                    elapsed = __import__('time').time() - t0
                    eta_min = elapsed / n_done * (n_dates - n_done) / 60
                    print(f"  {n_done}/{n_dates}  last={date_str}  "
                          f"({elapsed:.0f}s  ~{eta_min:.0f} min left)", flush=True)

    cache.flush()
    del cache
    elapsed_total = __import__('time').time() - t0
    print(f"\nFlushed cache to {out_file}  ({os.path.getsize(out_file)/1e9:.2f} GB)  "
          f"total={elapsed_total:.0f}s", flush=True)

    # Save companion dates array
    dates_file = out_file + ".dates.npy"
    np.save(dates_file, np.array(date_dirs, dtype=str))
    print(f"Saved dates array to {dates_file}  ({n_dates} entries)", flush=True)

    print("\nDone.", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Build S2S decoder patch-mean cache for training."
    )
    ap.add_argument("--s2s-dir", required=True,
                    help="Root directory containing YYYY-MM-DD subdirs with lead{kk:02d}.tif")
    ap.add_argument("--out-file", required=True,
                    help="Output .dat memmap file path.")
    ap.add_argument("--reference", required=True,
                    help="Reference GeoTIFF (e.g. fwi_20250615.tif) to read grid H/W.")
    ap.add_argument("--patch-size", type=int, default=16,
                    help="Patch size in pixels (default: 16).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel worker processes (default: 8).")
    args = ap.parse_args()

    build_cache(
        s2s_dir=args.s2s_dir,
        out_file=args.out_file,
        reference_tif=args.reference,
        patch_size=args.patch_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
