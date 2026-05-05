"""
Build lightning_climatology.tif from GLM raw daily TIFs.

For each pixel on the EPSG:3978 2 km grid: mean annual strike count
(or strike rate, depending on glm_raw value units), log1p-transformed
to control the heavy tail.

Usage:
  python -m scripts.build_lightning_climatology \\
      --input_dir data/lightning_raw \\
      --reference data/fwi_data/fwi_20250615.tif \\
      --output data/lightning_climatology.tif

Output: a single (H, W) float32 GeoTIFF on the canonical 2 km
EPSG:3978 grid; treat as a 'static' channel like population/slope.
"""
from __future__ import annotations
import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/lightning_raw",
                    help="Directory of glm_raw_YYYYMMDD.tif files")
    ap.add_argument("--reference", default="data/fwi_data/fwi_20250615.tif",
                    help="Reference TIF defining EPSG:3978 grid")
    ap.add_argument("--output", default="data/lightning_climatology.tif")
    ap.add_argument("--apply_log1p", action="store_true", default=True,
                    help="Apply log1p to compress heavy tail")
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: only process first N files")
    args = ap.parse_args()

    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
    except ImportError as e:
        print(f"[ERROR] missing dep: {e}")
        sys.exit(1)

    files = sorted(glob.glob(str(Path(args.input_dir) / "glm_raw_*.tif")))
    if not files:
        print(f"[ERROR] no glm_raw_*.tif in {args.input_dir}")
        sys.exit(1)
    if args.limit > 0:
        files = files[:args.limit]
    print(f"Found {len(files)} GLM daily files")

    # Reference grid
    with rasterio.open(args.reference) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        H, W = ref.height, ref.width
    print(f"Reference: {ref_crs}, shape=({H}, {W})")

    # Pre-scan: distinct years covered
    date_pat = re.compile(r"glm_raw_(\d{8})\.tif$")
    years = set()
    for f in files:
        m = date_pat.search(f)
        if m:
            years.add(int(m.group(1)[:4]))
    n_years = max(len(years), 1)
    print(f"Distinct years covered: {sorted(years)}  (n={n_years})")

    # Reproject + accumulate
    strike_sum = np.zeros((H, W), dtype=np.float64)
    n_processed = 0
    n_failed = 0
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            with rasterio.open(f) as src:
                src_data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    src_data[src_data == src.nodata] = 0.0
                src_data = np.nan_to_num(src_data, nan=0.0, posinf=0.0, neginf=0.0)
                src_t = src.transform
                src_c = src.crs
            dst_data = np.zeros((H, W), dtype=np.float32)
            reproject(src_data, dst_data,
                      src_transform=src_t, src_crs=src_c,
                      dst_transform=ref_transform, dst_crs=ref_crs,
                      resampling=Resampling.bilinear,
                      src_nodata=0, dst_nodata=0)
            strike_sum += dst_data
            n_processed += 1
        except Exception as e:
            print(f"  [WARN] {Path(f).name}: {e}")
            n_failed += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.0f}s — "
                  f"sum max={strike_sum.max():.1f}")

    print(f"\nProcessed: {n_processed}/{len(files)}  Failed: {n_failed}")
    print(f"Strike sum: min={strike_sum.min():.2f} max={strike_sum.max():.2f} "
          f"mean={strike_sum.mean():.4f}")

    # Per-year mean
    strike_per_year = strike_sum / float(n_years)
    print(f"Per-year mean: max={strike_per_year.max():.2f}")

    # Log1p transform for stability
    if args.apply_log1p:
        out = np.log1p(strike_per_year).astype(np.float32)
        print(f"After log1p: max={out.max():.4f}")
    else:
        out = strike_per_year.astype(np.float32)

    # Write
    out_profile = ref_profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0,
                       compress="DEFLATE", tiled=True,
                       blockxsize=256, blockysize=256)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output, "w", **out_profile) as dst:
        dst.write(out, 1)
    print(f"\nWrote {args.output}  size={Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
