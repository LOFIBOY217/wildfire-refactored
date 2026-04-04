#!/usr/bin/env python3
"""
Download WorldPop 2020 global population density and reproject to FWI grid.

Source: WorldPop 2020 Constrained Individual Countries UN Adjusted
  https://data.worldpop.org/

Pipeline:
  1. Download global mosaic (ppp_2020_1km_Aggregated.tif, ~2.4 GB)
  2. Clip to Canada bounding box
  3. Reproject to EPSG:3978 (2709x2281, bilinear)
  4. Apply log1p transform (population spans orders of magnitude)
  5. Save as data/population_density.tif

Output semantics:
  Each pixel = log1p(people per km^2) at ~2km resolution.
  0 = uninhabited.  Higher = more populated.

Usage:
    python -m src.data_ops.download.download_population
    python -m src.data_ops.download.download_population --config configs/paths_narval.yaml

Prerequisites:
    pip install requests  (numpy, rasterio already in environment)
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# FWI grid constants
# ------------------------------------------------------------------ #

FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)

# Canada bounding box (WGS84)
CANADA_BBOX_WGS84 = (-141.0, 41.0, -52.0, 84.0)  # (W, S, E, N)

# WorldPop data URL
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
    "2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif"
)


def _download_file(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024):
    """Stream-download a large file with progress."""
    import requests

    print(f"[DOWNLOAD] {url}")
    print(f"  -> {dest}")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)",
                      end="", flush=True)
    print()
    print(f"[DONE] {dest.stat().st_size / 1e9:.2f} GB")


def _clip_and_reproject(src_path: Path, dst_path: Path):
    """Read source population raster, clip to Canada, reproject to FWI grid."""
    dst_transform = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    dst_crs = CRS.from_string(FWI_CRS)

    print(f"[REPROJECT] {src_path.name} -> EPSG:3978 ({FWI_WIDTH}x{FWI_HEIGHT})")

    with rasterio.open(src_path) as src:
        # Read windowed to Canada bbox to avoid loading entire global raster
        from rasterio.windows import from_bounds as window_from_bounds
        try:
            window = window_from_bounds(
                *CANADA_BBOX_WGS84, transform=src.transform
            )
            # Expand window slightly to ensure full coverage after reprojection
            window = rasterio.windows.Window(
                max(0, int(window.col_off) - 10),
                max(0, int(window.row_off) - 10),
                min(int(window.width) + 20, src.width),
                min(int(window.height) + 20, src.height),
            )
            data = src.read(1, window=window).astype(np.float32)
            win_transform = src.window_transform(window)
        except Exception:
            # Fallback: read entire raster
            print("  [WARN] Windowed read failed, reading full raster")
            data = src.read(1).astype(np.float32)
            win_transform = src.transform

        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[~np.isfinite(data)] = 0.0
        data[data < 0] = 0.0

        # Prepare destination array
        dst_data = np.zeros((FWI_HEIGHT, FWI_WIDTH), dtype=np.float32)

        reproject(
            source=data,
            destination=dst_data,
            src_transform=win_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    # log1p transform (population spans many orders of magnitude)
    dst_data = np.where(np.isfinite(dst_data), dst_data, 0.0)
    dst_data = np.maximum(dst_data, 0.0)
    dst_data = np.log1p(dst_data).astype(np.float32)

    # Write output GeoTIFF
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": FWI_WIDTH,
        "height": FWI_HEIGHT,
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dst_data, 1)
        dst.update_tags(
            variable="population_density",
            units="log1p(people_per_km2)",
            source="WorldPop 2020",
        )

    valid = np.isfinite(dst_data) & (dst_data > 0)
    print(f"[SAVED] {dst_path}")
    print(f"  Shape: {dst_data.shape}, Valid pixels: {valid.sum():,}")
    print(f"  Range: [{dst_data[valid].min():.2f}, {dst_data[valid].max():.2f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Download WorldPop population density -> FWI grid"
    )
    add_config_argument(parser)
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path (default: from config)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, assume raw file exists in /tmp")
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    if args.output:
        out_path = Path(args.output)
    else:
        try:
            out_path = Path(get_path(cfg, "population_tif"))
        except (KeyError, TypeError):
            out_path = Path("data/population_density.tif")

    if out_path.exists():
        print(f"[SKIP] Output already exists: {out_path}")
        print("  Delete it and re-run to rebuild.")
        return

    # Download to temp location
    tmp_dir = Path(tempfile.gettempdir())
    raw_path = tmp_dir / "worldpop_2020_1km.tif"

    if not args.skip_download and not raw_path.exists():
        _download_file(WORLDPOP_URL, raw_path)
    elif not raw_path.exists():
        print(f"[ERROR] Raw file not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    _clip_and_reproject(raw_path, out_path)
    print("[COMPLETE]")


if __name__ == "__main__":
    main()
