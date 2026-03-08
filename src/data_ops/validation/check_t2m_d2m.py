"""
Diagnostic script: check whether 2t and 2d TIF files are identical.
Run from project root:
    python -m src.data_ops.validation.check_t2m_d2m --config configs/paths_windows.yaml
"""
import argparse
import glob
import os
import sys
import numpy as np
import rasterio

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    from pathlib import Path
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument

# ── Load config ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(
    description="Check whether 2t and 2d TIF files contain identical data."
)
add_config_argument(ap)
args = ap.parse_args()

cfg      = load_config(args.config)
obs_root = get_path(cfg, "observation_dir")
print(f"obs_root : {obs_root}\n")

# ── Find files ─────────────────────────────────────────────────────────────────
def find_files(directory, prefix):
    sub = sorted(glob.glob(os.path.join(directory, prefix, f"{prefix}_*.tif")))
    flat = sorted(glob.glob(os.path.join(directory, f"{prefix}_*.tif")))
    found = sub if sub else flat
    print(f"  [{prefix}] subdirectory pattern: {len(sub)} files")
    print(f"  [{prefix}] flat pattern        : {len(flat)} files")
    print(f"  [{prefix}] using              : {'subdirectory' if sub else 'flat'}")
    return found

print("=== File search ===")
t2m_files = find_files(obs_root, "2t")
print()
d2m_files = find_files(obs_root, "2d")
print()

# ── Basic counts ───────────────────────────────────────────────────────────────
print("=== File counts ===")
print(f"  2t files : {len(t2m_files)}")
print(f"  2d files : {len(d2m_files)}")
if not t2m_files:
    print("  [ERROR] No 2t files found!")
if not d2m_files:
    print("  [ERROR] No 2d files found!")
print()

# ── Path comparison ────────────────────────────────────────────────────────────
if t2m_files and d2m_files:
    print("=== First file paths ===")
    print(f"  t2m[0] : {t2m_files[0]}")
    print(f"  d2m[0] : {d2m_files[0]}")
    print(f"  Same path? {t2m_files[0] == d2m_files[0]}")
    print()

    # ── Content comparison (first file) ───────────────────────────────────────
    print("=== Content comparison (first file) ===")
    with rasterio.open(t2m_files[0]) as src:
        a = src.read(1).astype(np.float32)
        print(f"  t2m nodata tag : {src.nodata}")
    with rasterio.open(d2m_files[0]) as src:
        b = src.read(1).astype(np.float32)
        print(f"  d2m nodata tag : {src.nodata}")

    print(f"  t2m mean : {np.nanmean(a):.4f}   std : {np.nanstd(a):.4f}")
    print(f"  d2m mean : {np.nanmean(b):.4f}   std : {np.nanstd(b):.4f}")
    print(f"  Pixel-identical? {np.allclose(a, b, equal_nan=True, rtol=0, atol=0)}")
    print()

    # ── Sample a few more dates to confirm ────────────────────────────────────
    print("=== Content comparison (5 random files) ===")
    n_check = min(5, len(t2m_files), len(d2m_files))
    indices = np.linspace(0, min(len(t2m_files), len(d2m_files)) - 1,
                          n_check, dtype=int)
    all_identical = True
    for i in indices:
        with rasterio.open(t2m_files[i]) as src:
            a = src.read(1).astype(np.float32)
        with rasterio.open(d2m_files[i]) as src:
            b = src.read(1).astype(np.float32)
        identical = np.allclose(a, b, equal_nan=True, rtol=0, atol=0)
        all_identical = all_identical and identical
        fname_t = os.path.basename(t2m_files[i])
        fname_d = os.path.basename(d2m_files[i])
        print(f"  {fname_t} vs {fname_d}  →  identical={identical}"
              f"  t2m_mean={np.nanmean(a):.3f}  d2m_mean={np.nanmean(b):.3f}")

    print()
    if all_identical:
        print("CONCLUSION: 2t and 2d files contain IDENTICAL data.")
        print("  → The dewpoint channel is a duplicate of temperature.")
        print("  → Model is effectively seeing 7 unique channels, not 8.")
        print("  → Need to regenerate 2d_*.tif from ERA5 GRIB with d2m variable.")
    else:
        print("CONCLUSION: 2t and 2d files contain DIFFERENT data. (expected)")
        print("  → The identical statistics in training log have a different cause.")
