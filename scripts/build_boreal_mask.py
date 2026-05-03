"""
Build a boreal-belt mask from leak-free fire climatology.

Mask pixels where climatology >= percentile threshold across Canada land.
The threshold defines the "high-fire region" — typically top 30% of
land pixels by climatological fire density.

Output: (H, W) uint8 npy file (1 = inside boreal belt, 0 = outside).

Usage:
  python -m scripts.build_boreal_mask \\
      --climatology_tif data/fire_clim_annual_nbac/fire_clim_upto_2022.tif \\
      --top_frac 0.30 \\
      --output data/masks/boreal_belt_top30pct.npy
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--climatology_tif", required=True)
    ap.add_argument("--top_frac", type=float, default=0.30,
                    help="Mask = top X fraction of LAND pixels by climatology")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import rasterio
    with rasterio.open(args.climatology_tif) as src:
        clim = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        clim[clim == nodata] = np.nan

    land_mask = np.isfinite(clim)
    land_pixels = clim[land_mask]
    if len(land_pixels) == 0:
        raise RuntimeError("No land pixels found")

    # Threshold = (1-top_frac) percentile of land pixels
    pct = (1.0 - args.top_frac) * 100.0
    thresh = np.percentile(land_pixels, pct)
    mask = (clim >= thresh) & land_mask
    out = mask.astype(np.uint8)

    print(f"  climatology range: [{land_pixels.min():.4f}, {land_pixels.max():.4f}]")
    print(f"  threshold (top {args.top_frac*100:.0f}% of land): {thresh:.4f}")
    print(f"  mask area: {out.sum():,} / {out.size:,} pixels "
          f"({100 * out.sum() / out.size:.2f}% of canvas, "
          f"{100 * out.sum() / max(land_mask.sum(), 1):.2f}% of land)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
