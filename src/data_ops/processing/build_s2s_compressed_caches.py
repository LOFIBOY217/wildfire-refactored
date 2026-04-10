"""
Build Compressed S2S Decoder Caches from Full-Patch Cache
==========================================================
Reads the existing s2s_full_patch_cache.dat (n_dates, 23998, 32, 2048) float16
and produces compressed versions suitable for decoder experiments.

Three modes:

  multi_stat   — per-channel mean/std/max → 24 dims   (~40G)
  subpatch_4x4 — 4×4 sub-block means     → 128 dims  (~260G)
  pca          — PCA projection           → 128 dims  (~260G)

The full-patch layout is: 2048 = P*P*C = 16*16*8, stored as
  [ch0_px0, ch0_px1, ..., ch0_px255, ch1_px0, ..., ch7_px255]

Usage:
    python -m src.data_ops.processing.build_s2s_compressed_caches \\
        --full-cache data/s2s_full_patch_cache.dat \\
        --out-file /scratch/jiaqi217/meteo_cache/s2s_multistat_cache.dat \\
        --mode multi_stat

    python -m src.data_ops.processing.build_s2s_compressed_caches \\
        --full-cache data/s2s_full_patch_cache.dat \\
        --out-file /scratch/jiaqi217/meteo_cache/s2s_subpatch4x4_cache.dat \\
        --mode subpatch_4x4

    python -m src.data_ops.processing.build_s2s_compressed_caches \\
        --full-cache data/s2s_full_patch_cache.dat \\
        --out-file /scratch/jiaqi217/meteo_cache/s2s_pca128_cache.dat \\
        --mode pca --pca-components 128 --pca-samples 1000000
"""

import argparse
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Full-patch cache constants (must match build_s2s_full_patch_cache.py)
# ---------------------------------------------------------------------------
N_CHANNELS = 8       # FWI, 2t, 2d, FFMC, DMC, DC, BUI, fire_clim
PATCH_SIZE = 16
PIXELS_PER_PATCH = PATCH_SIZE * PATCH_SIZE  # 256
FULL_DIM = PIXELS_PER_PATCH * N_CHANNELS    # 2048
N_LEADS = 32


def _reshape_to_spatial(flat, n_channels=N_CHANNELS, P=PATCH_SIZE):
    """Reshape (..., P*P*C) → (..., C, P, P).

    Full-patch layout: [ch0_px0..px255, ch1_px0..px255, ...].
    """
    *batch, D = flat.shape
    assert D == P * P * n_channels, f"Expected dim={P*P*n_channels}, got {D}"
    # (..., C*P*P) → (..., C, P*P) → (..., C, P, P)
    out = flat.reshape(*batch, n_channels, P * P)
    return out.reshape(*batch, n_channels, P, P)


# ---------------------------------------------------------------------------
# Mode 1: multi_stat — mean/std/max per channel → 24 dims
# ---------------------------------------------------------------------------
def compress_multi_stat(chunk):
    """
    Args:
        chunk: (n_patches, 32, 2048) float16 — one date slice

    Returns:
        (n_patches, 32, 24) float32
    """
    spatial = _reshape_to_spatial(chunk.astype(np.float32))
    # spatial: (n_patches, 32, 8, 16, 16)

    # Flatten spatial dims for stats: (n_patches, 32, 8, 256)
    NP, NL, C, H, W = spatial.shape
    flat = spatial.reshape(NP, NL, C, H * W)

    ch_mean = flat.mean(axis=-1)    # (NP, NL, 8)
    ch_std = flat.std(axis=-1)      # (NP, NL, 8)
    ch_max = flat.max(axis=-1)      # (NP, NL, 8)

    # Concatenate: (NP, NL, 24)
    return np.concatenate([ch_mean, ch_std, ch_max], axis=-1)


# ---------------------------------------------------------------------------
# Mode 2: subpatch_4x4 — 16 sub-block means × 8 channels → 128 dims
# ---------------------------------------------------------------------------
def compress_subpatch_4x4(chunk):
    """
    Args:
        chunk: (n_patches, 32, 2048) float16 — one date slice

    Returns:
        (n_patches, 32, 128) float32
    """
    spatial = _reshape_to_spatial(chunk.astype(np.float32))
    # spatial: (n_patches, 32, 8, 16, 16)

    NP, NL, C, H, W = spatial.shape
    BLOCK = 4  # 16/4 = 4 sub-blocks per axis
    nH, nW = H // BLOCK, W // BLOCK  # 4, 4

    # Reshape to sub-blocks: (NP, NL, C, nH, BLOCK, nW, BLOCK)
    blocked = spatial.reshape(NP, NL, C, nH, BLOCK, nW, BLOCK)
    # Mean over pixel dims (axes 4, 6): (NP, NL, C, nH, nW)
    sub_means = blocked.mean(axis=(4, 6))
    # Flatten: (NP, NL, C * nH * nW) = (NP, NL, 8*4*4) = (NP, NL, 128)
    return sub_means.reshape(NP, NL, C * nH * nW)


# ---------------------------------------------------------------------------
# Mode 3: PCA — fit on sampled data, project all
# ---------------------------------------------------------------------------
def fit_pca(full_cache, n_dates, n_patches, n_components=128,
            n_samples=1_000_000, seed=42):
    """
    Sample random (date, patch, lead) vectors from full-patch cache,
    fit PCA, return (components, mean).

    Args:
        full_cache: memmap (n_dates, n_patches, 32, 2048)
        n_components: number of PCA components

    Returns:
        components: (n_components, 2048) float32
        mean_vec: (2048,) float32
    """
    rng = np.random.default_rng(seed)
    print(f"  [PCA] Sampling {n_samples:,} vectors for PCA fitting...")
    t0 = time.time()

    # Sample random indices
    date_idx = rng.integers(0, n_dates, size=n_samples)
    patch_idx = rng.integers(0, n_patches, size=n_samples)
    lead_idx = rng.integers(0, N_LEADS, size=n_samples)

    # Gather samples (read sequentially by date for cache efficiency)
    samples = np.empty((n_samples, FULL_DIM), dtype=np.float32)
    # Sort by date for sequential access
    order = np.argsort(date_idx)
    for i, idx in enumerate(order):
        samples[i] = full_cache[date_idx[idx], patch_idx[idx],
                                lead_idx[idx], :].astype(np.float32)
        if (i + 1) % 200_000 == 0:
            print(f"    sampled {i+1:,}/{n_samples:,}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"  [PCA] Fitting PCA via covariance method...", flush=True)
    # Memory-efficient PCA: compute covariance matrix (2048x2048 = 16MB)
    # instead of doing SVD on (1M x 2048) sample matrix (which needs ~30GB).
    mean_vec = samples.mean(axis=0).astype(np.float32)
    # In-place centering to save memory
    samples -= mean_vec
    # Covariance: (2048, 2048) float64 for numerical precision
    print(f"  [PCA] Computing covariance (2048x2048)...", flush=True)
    cov = (samples.astype(np.float64).T @ samples.astype(np.float64)) / (n_samples - 1)
    # Free the sample matrix now
    del samples
    import gc; gc.collect()

    print(f"  [PCA] Eigendecomposition of covariance...", flush=True)
    # Eigendecomposition (symmetric matrix, fast)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending, flip to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    components = eigvecs[:, :n_components].T  # (n_components, 2048)

    # Report variance explained
    total_var = eigvals.sum()
    var_explained = eigvals[:n_components].sum() / total_var
    print(f"  [PCA] Variance explained by {n_components} components: "
          f"{var_explained:.4f} ({var_explained*100:.1f}%)")
    print(f"  [PCA] Fitting done ({time.time()-t0:.0f}s)", flush=True)

    return components.astype(np.float32), mean_vec.astype(np.float32)


def compress_pca(chunk, components, mean_vec):
    """
    Args:
        chunk: (n_patches, 32, 2048) float16
        components: (n_components, 2048) float32
        mean_vec: (2048,) float32

    Returns:
        (n_patches, 32, n_components) float32
    """
    x = chunk.astype(np.float32) - mean_vec  # (NP, 32, 2048)
    # Project: (NP, 32, 2048) @ (2048, n_components) → (NP, 32, n_components)
    return x @ components.T


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------
def build_compressed_cache(full_cache_path, out_file, mode,
                           pca_components=128, pca_samples=1_000_000):
    """Build a compressed S2S decoder cache."""

    # Load dates companion
    dates_path = full_cache_path + ".dates.npy"
    if not os.path.exists(dates_path):
        print(f"ERROR: dates file not found: {dates_path}", file=sys.stderr)
        sys.exit(1)
    dates = np.load(dates_path, allow_pickle=True)
    n_dates = len(dates)

    # Open full-patch cache as memmap
    print(f"Opening full-patch cache: {full_cache_path}")
    # Infer n_patches from file size
    file_size = os.path.getsize(full_cache_path)
    expected_per_date = N_LEADS * FULL_DIM * 2  # float16 = 2 bytes
    # file_size = n_dates * n_patches * expected_per_date
    n_patches = file_size // (n_dates * expected_per_date)
    print(f"  n_dates={n_dates}  n_patches={n_patches}  "
          f"n_leads={N_LEADS}  full_dim={FULL_DIM}")
    print(f"  file_size={file_size / 1e12:.2f} TB")

    full_cache = np.memmap(full_cache_path, dtype=np.float16, mode='r',
                           shape=(n_dates, n_patches, N_LEADS, FULL_DIM))

    # Determine output dim
    if mode == "multi_stat":
        out_dim = N_CHANNELS * 3  # mean + std + max = 24
        compress_fn = compress_multi_stat
    elif mode == "subpatch_4x4":
        out_dim = N_CHANNELS * 4 * 4  # 8 * 16 = 128
        compress_fn = compress_subpatch_4x4
    elif mode == "pca":
        out_dim = pca_components
        components, mean_vec = fit_pca(
            full_cache, n_dates, n_patches,
            n_components=pca_components, n_samples=pca_samples)
        # Save PCA basis for reconstruction / future use
        pca_path = out_file + ".pca_basis.npz"
        np.savez(pca_path, components=components, mean=mean_vec)
        print(f"  Saved PCA basis to {pca_path}")
        compress_fn = lambda chunk: compress_pca(chunk, components, mean_vec)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create output memmap
    out_shape = (n_dates, n_patches, N_LEADS, out_dim)
    out_bytes = n_dates * n_patches * N_LEADS * out_dim * 2  # float16
    print(f"\nOutput: {out_file}")
    print(f"  mode={mode}  out_dim={out_dim}")
    print(f"  shape={out_shape}  size={out_bytes / 1e9:.1f} GB")

    out_cache = np.memmap(out_file, dtype=np.float16, mode='w+',
                          shape=out_shape)

    # Process date by date
    t0 = time.time()
    for i in range(n_dates):
        chunk = full_cache[i]  # (n_patches, 32, 2048) float16
        compressed = compress_fn(chunk)  # (n_patches, 32, out_dim) float32
        out_cache[i] = compressed.astype(np.float16)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_dates - i - 1) / rate
            print(f"  date {i+1:4d}/{n_dates}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    out_cache.flush()
    elapsed = time.time() - t0

    # Copy dates companion
    out_dates = out_file + ".dates.npy"
    np.save(out_dates, dates)

    print(f"\nDone: {mode} cache built in {elapsed:.0f}s")
    print(f"  Output: {out_file}  ({os.path.getsize(out_file) / 1e9:.1f} GB)")
    print(f"  Dates:  {out_dates}  ({n_dates} entries)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build compressed S2S decoder cache from full-patch cache.")
    ap.add_argument("--full-cache", required=True,
                    help="Path to s2s_full_patch_cache.dat (4.9TB memmap).")
    ap.add_argument("--out-file", required=True,
                    help="Output .dat memmap file path.")
    ap.add_argument("--mode", required=True,
                    choices=["multi_stat", "subpatch_4x4", "pca"],
                    help="Compression mode.")
    ap.add_argument("--pca-components", type=int, default=128,
                    help="Number of PCA components (default: 128). "
                         "Only used in pca mode.")
    ap.add_argument("--pca-samples", type=int, default=1_000_000,
                    help="Number of samples for PCA fitting (default: 1M). "
                         "Only used in pca mode.")
    args = ap.parse_args()

    build_compressed_cache(
        full_cache_path=args.full_cache,
        out_file=args.out_file,
        mode=args.mode,
        pca_components=args.pca_components,
        pca_samples=args.pca_samples,
    )


if __name__ == "__main__":
    main()
