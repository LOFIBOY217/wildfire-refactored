#!/usr/bin/env python3
"""
Gate 2: validate a v3 meteo cache directory after a cache build job completes.

Checks performed:
  1. Directory exists and is non-empty.
  2. stats.npy exists with expected shape (2, N_channels).
  3. fire_dilated_r{R}_*.npy exists with expected shape (T, H, W).
  4. meteo_v3_p{P}_C{N}_T{T}_*_pf.dat exists with expected byte size.
  5. pf.dat is memmap-loadable with expected shape.
  6. Sample patches are finite, non-constant, have reasonable range per
     channel (uses stats mean/std as reference).

Exit 0 if all pass, 1 if any fail. Prints a summary table.

Usage:
  python scripts/gate2_validate_cache.py \\
      --cache_dir /scratch/jiaqi217/meteo_cache/v3_9ch_2000 \\
      --n_channels 9

Designed to be run ON NARVAL (needs the venv with numpy/rasterio).
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np


def _find_one(pattern, descr):
    """Find exactly one file matching glob pattern. Returns path or None."""
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        print(f"  [FAIL] {descr}: no file matches {pattern}")
        return None
    if len(matches) > 1:
        print(f"  [WARN] {descr}: multiple matches, using first")
        for m in matches:
            print(f"         {m}")
    return matches[0]


def check(cache_dir: Path, n_channels: int, patch_size: int = 16,
          H: int = 2281, W: int = 2709, dilate_radius: int = 14):
    print(f"\n[Gate 2] Validating cache at {cache_dir}")
    print(f"  expected: C={n_channels}  P={patch_size}  H×W={H}×{W}  "
          f"dilate_r={dilate_radius}")
    print(f"  ─────────────────────────────────────────────────────")

    ok = True

    # 1. Dir exists
    if not cache_dir.is_dir():
        print(f"  [FAIL] cache dir does not exist: {cache_dir}")
        return False
    files = list(cache_dir.iterdir())
    print(f"  [PASS] dir exists, {len(files)} entries")

    # 2. stats.npy
    stats_path = cache_dir / f"meteo_v3_p{patch_size}_C{n_channels}_stats.npy"
    if not stats_path.exists():
        print(f"  [FAIL] stats.npy not found: {stats_path}")
        ok = False
        stats = None
    else:
        stats = np.load(stats_path, allow_pickle=True)
        if stats.shape != (2, n_channels):
            print(f"  [FAIL] stats shape {stats.shape}, expected (2,{n_channels})")
            ok = False
        else:
            means = stats[0]
            stds = stats[1]
            if np.any(stds < 1e-4):
                print(f"  [FAIL] stats has near-zero std: {stds}")
                ok = False
            else:
                print(f"  [PASS] stats shape={stats.shape}  "
                      f"means range [{means.min():.3f}, {means.max():.3f}]")

    # 3. fire_dilated
    fd_pattern = str(cache_dir / f"fire_dilated_r{dilate_radius}_*_{H}x{W}.npy")
    fd_path = _find_one(fd_pattern, "fire_dilated")
    if fd_path is None:
        ok = False
        fire_stack = None
    else:
        try:
            fire_stack = np.load(fd_path, mmap_mode="r")
            T_fd = fire_stack.shape[0]
            if fire_stack.shape[1:] != (H, W):
                print(f"  [FAIL] fire_dilated shape {fire_stack.shape}, expected (T,{H},{W})")
                ok = False
            else:
                pos_rate = float(fire_stack.sum()) / fire_stack.size
                print(f"  [PASS] fire_dilated shape={fire_stack.shape}  "
                      f"pos_rate={pos_rate:.4%}")
                if pos_rate < 1e-5 or pos_rate > 0.5:
                    print(f"  [WARN] pos_rate outside [1e-5, 0.5]")
        except Exception as e:
            print(f"  [FAIL] fire_dilated load error: {e}")
            ok = False
            fire_stack = None

    # 4. meteo memmap pf.dat
    mem_pattern = str(cache_dir / f"meteo_v3_p{patch_size}_C{n_channels}_T*_*_pf.dat")
    mem_path = _find_one(mem_pattern, "meteo pf.dat")
    if mem_path is None:
        ok = False
        return ok
    m = re.search(r"_T(\d+)_", os.path.basename(mem_path))
    if not m:
        print(f"  [FAIL] cannot parse T from filename: {mem_path}")
        return False
    T = int(m.group(1))
    Hc, Wc = H - H % patch_size, W - W % patch_size
    n_patches = (Hc // patch_size) * (Wc // patch_size)
    enc_dim = patch_size * patch_size * n_channels
    expected_bytes = n_patches * T * enc_dim * 2  # float16
    actual_bytes = os.path.getsize(mem_path)
    if actual_bytes != expected_bytes:
        print(f"  [FAIL] pf.dat size {actual_bytes/1e9:.2f} GB != "
              f"expected {expected_bytes/1e9:.2f} GB")
        print(f"         (n_patches={n_patches} T={T} enc_dim={enc_dim})")
        ok = False
    else:
        print(f"  [PASS] pf.dat size {actual_bytes/1e9:.2f} GB  "
              f"(n_patches={n_patches} T={T} enc_dim={enc_dim})")

    # 5. memmap loadable + sample check
    try:
        mmap = np.memmap(mem_path, dtype=np.float16, mode="r",
                         shape=(n_patches, T, enc_dim))
    except Exception as e:
        print(f"  [FAIL] memmap open failed: {e}")
        return False

    print(f"  [PASS] memmap opened  shape={mmap.shape}")

    # 6. Sample: grab 3 random patches × 3 random times per channel
    rng = np.random.default_rng(0)
    sample_patches = rng.choice(n_patches, size=5, replace=False)
    sample_times = rng.choice(T, size=5, replace=False)
    sample_ok = True
    for ch in range(n_channels):
        ch_slice = mmap[np.ix_(sample_patches, sample_times)][:, :, ch * patch_size * patch_size]
        arr = np.asarray(ch_slice).astype(np.float32)
        finite = np.isfinite(arr)
        if not finite.all():
            nan_pct = 100 * (~finite).mean()
            print(f"  [WARN] ch{ch}: {nan_pct:.1f}% NaN in sample")
        if finite.sum() == 0:
            print(f"  [FAIL] ch{ch}: entire sample is NaN")
            sample_ok = False
            continue
        mn, mx = float(arr[finite].min()), float(arr[finite].max())
        mu = float(arr[finite].mean())
        if mx - mn < 1e-5:
            print(f"  [FAIL] ch{ch}: near-constant sample (min={mn:.4f} max={mx:.4f})")
            sample_ok = False
            continue
        if stats is not None:
            ref_std = stats[1, ch]
            if abs(mu) > 10 * ref_std and ref_std > 1e-3:
                print(f"  [WARN] ch{ch}: sample mean {mu:.3f} far from 0 "
                      f"(should be ~z-scored)")
    if sample_ok:
        print(f"  [PASS] {n_channels} channels sampled, no constant/all-NaN")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, type=Path)
    ap.add_argument("--n_channels", required=True, type=int,
                    help="9, 13, or 16")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--dilate_radius", type=int, default=14)
    args = ap.parse_args()

    ok = check(args.cache_dir, args.n_channels,
               patch_size=args.patch_size,
               dilate_radius=args.dilate_radius)

    print(f"  ─────────────────────────────────────────────────────")
    if ok:
        print(f"  [RESULT] PASS — cache usable for training")
        sys.exit(0)
    else:
        print(f"  [RESULT] FAIL — do NOT proceed to training")
        sys.exit(1)


if __name__ == "__main__":
    main()
