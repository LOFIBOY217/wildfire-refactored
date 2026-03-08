#!/usr/bin/env python3
"""
Build a static fire-frequency climatology map from CWFIS hotspot records.

Rasterises all CWFIS/VIIRS-M hotspot detections from the *training* period
(default 2018–2021, fire season May–Oct) onto the FWI grid, counting how many
days each pixel was detected as on fire.  The raw count is log1p-normalised
and saved as a single-band GeoTIFF.

This static map is used as Channel 7 in the V2 S2S model
(train_s2s_hotspot_cwfis_v2.py).  It encodes the spatial persistence of fire
risk — pixels that burned often in the past tend to burn often in the future —
which is the single strongest zero-download signal available at subseasonal range.

Usage:
    python -m src.data_ops.processing.make_fire_climatology \\
        --config configs/paths_windows.yaml

    # Custom year/month range
    python -m src.data_ops.processing.make_fire_climatology \\
        --config configs/paths_windows.yaml \\
        --years 2018-2021 \\
        --months 5-10

    # Save to custom path
    python -m src.data_ops.processing.make_fire_climatology \\
        --config configs/paths_windows.yaml \\
        --output data/fire_climatology.tif

Output:
    <fire_climatology_tif>   (from config) or --output path
    Single-band float32 GeoTIFF on the FWI grid.
    Values: log1p(number of fire-detection days) per pixel.
    Background pixels = 0.0.
"""

import argparse
import glob
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument

from src.data_ops.processing.rasterize_hotspots import load_hotspot_data


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _rasterize_to_count(lats, lons, transformer, transform, height, width, count_map):
    """
    Add hotspot detections to *count_map* (in-place, float32).

    Vectorised: projects all points at once, increments by 1 at each valid pixel.
    A pixel can accumulate multiple counts if multiple hotspots hit the same cell.
    """
    if len(lats) == 0:
        return

    xs, ys = transformer.transform(lons, lats)
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.intp)
    cols = np.asarray(cols, dtype=np.intp)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    np.add.at(count_map, (rows[valid], cols[valid]), 1)


def _iter_dates(years, months):
    """Yield every date in the requested year×month combinations."""
    for year in years:
        for month in months:
            if month == 12:
                first = date(year, month, 1)
                last  = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                first = date(year, month, 1)
                last  = date(year, month + 1, 1) - timedelta(days=1)
            cur = first
            while cur <= last:
                yield cur
                cur += timedelta(days=1)


def build_fire_climatology(
    hotspot_csv,
    fwi_reference_tif,
    years,
    months,
    output_path,
    log_transform=True,
    count_mode="days",
):
    """
    Build and save a fire-frequency climatology map.

    Args:
        hotspot_csv:       Path to CWFIS hotspot CSV (latitude, longitude, acq_date).
        fwi_reference_tif: Path to any FWI GeoTIFF (used to obtain grid/CRS/transform).
        years:             List of ints, e.g. [2018, 2019, 2020, 2021].
        months:            List of ints, e.g. [5, 6, 7, 8, 9, 10] (fire season).
        output_path:       Output file path for the GeoTIFF.
        log_transform:     If True (default), apply log1p to the count map.
        count_mode:        "days"  → count days with ≥1 detection per pixel (binary/day)
                           "detections" → count total detection events (may exceed 1/day
                           if multiple sensors see the same pixel on the same day)
    Returns:
        Path: path to the saved GeoTIFF.
    """
    print(f"[CLIMATOLOGY] Loading hotspot data …")
    df = load_hotspot_data(hotspot_csv)
    print(f"  Total records : {len(df):,}")
    print(f"  Full range    : {df['date'].min()} – {df['date'].max()}")

    # Filter to the requested training period
    df_train = df[df['date'].apply(lambda d: d.year in years and d.month in months)]
    print(f"  Training subset (years={years}, months={months}): {len(df_train):,} records")

    if len(df_train) == 0:
        raise ValueError(
            f"No hotspot records found for years={years}, months={months}. "
            "Check that hotspot_csv covers these years."
        )

    # Read reference grid parameters
    with rasterio.open(fwi_reference_tif) as src:
        profile   = src.profile.copy()
        height    = src.height
        width     = src.width
        transform = src.transform
        crs       = src.crs

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Accumulate fire counts
    count_map = np.zeros((height, width), dtype=np.float32)
    all_dates = list(_iter_dates(years, months))
    print(f"  Processing {len(all_dates)} dates …")

    n_reported = 0
    for i, d in enumerate(all_dates):
        day_df = df_train[df_train['date'] == d]
        if len(day_df) == 0:
            continue

        if count_mode == "days":
            # Count each pixel at most once per day (binary raster per day)
            day_count = np.zeros((height, width), dtype=np.float32)
            _rasterize_to_count(
                day_df['field_latitude'].values,
                day_df['field_longitude'].values,
                transformer, transform, height, width, day_count,
            )
            count_map += (day_count > 0).astype(np.float32)
        else:
            # Count every detection event
            _rasterize_to_count(
                day_df['field_latitude'].values,
                day_df['field_longitude'].values,
                transformer, transform, height, width, count_map,
            )

        n_reported += 1
        if i % 200 == 0 or i == len(all_dates) - 1:
            pct = 100.0 * (i + 1) / len(all_dates)
            print(f"  {i+1:4d}/{len(all_dates)}  {d}  "
                  f"days_with_fires={n_reported}  max_count={count_map.max():.0f}  "
                  f"({pct:.1f}%)")

    # Summary statistics
    n_fire_pixels = int((count_map > 0).sum())
    pct_fire      = 100.0 * n_fire_pixels / (height * width)
    print(f"\n  Pixels with ≥1 fire day : {n_fire_pixels:,} / {height*width:,} "
          f"({pct_fire:.2f}%)")
    print(f"  Max fire-days at one pixel : {count_map.max():.0f}")
    if n_fire_pixels > 0:
        print(f"  Mean count (nonzero pixels): {count_map[count_map > 0].mean():.2f}")

    # Optional log1p transform (reduces skew from hotspot-dense cells)
    if log_transform:
        out_map = np.log1p(count_map)
        print(f"  After log1p: max={out_map.max():.3f}  "
              f"mean(nonzero)={out_map[out_map > 0].mean():.3f}")
    else:
        out_map = count_map

    # Write output GeoTIFF
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=None,
        photometric=None,   # remove photometric tag (not meaningful for float data)
    )
    # Remove photometric key if it would cause a conflict
    out_profile.pop("photometric", None)

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(out_map.astype(np.float32), 1)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[CLIMATOLOGY] Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"  Grid: {height}×{width}  CRS: {crs}")
    print(f"  Values: log1p(fire-days)  range [{out_map.min():.3f}, {out_map.max():.3f}]")

    return out_path


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_range(s):
    """
    Parse a range string into a list of ints.

    '2018-2021' → [2018, 2019, 2020, 2021]
    '5-10'      → [5, 6, 7, 8, 9, 10]
    '2020'      → [2020]
    """
    parts = s.split("-")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(p) for p in parts]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a static fire-frequency climatology map (V2 model input)"
    )
    add_config_argument(parser)

    parser.add_argument(
        "--years", type=str, default="2018-2021",
        help="Training year range, e.g. '2018-2021' (default: 2018-2021). "
             "Use only the years that are in the TRAINING set to avoid data leakage.",
    )
    parser.add_argument(
        "--months", type=str, default="5-10",
        help="Month range within each year, e.g. '5-10' for May–Oct (default: 5-10).",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Path to a reference FWI GeoTIFF (auto-detected from fwi_dir if omitted).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output GeoTIFF path. Defaults to the 'fire_climatology_tif' config key, "
             "or <output_dir>/fire_climatology.tif.",
    )
    parser.add_argument(
        "--no_log", action="store_true",
        help="Skip log1p transform — output raw fire-day counts.",
    )
    parser.add_argument(
        "--detections", action="store_true",
        help="Count total detection events instead of unique fire-days per pixel.",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    hotspot_csv = get_path(cfg, "hotspot_csv")
    fwi_dir     = get_path(cfg, "fwi_dir")

    # Auto-detect reference FWI TIF
    ref_tif = args.reference
    if ref_tif is None:
        candidates = sorted(glob.glob(str(Path(fwi_dir) / "fwi_*.tif")))
        if not candidates:
            raise RuntimeError(
                f"No FWI TIF files found in {fwi_dir}. "
                "Provide --reference to specify a reference grid."
            )
        ref_tif = candidates[0]
        print(f"[CLIMATOLOGY] Auto-selected reference grid: {ref_tif}")

    # Resolve output path
    out_path = args.output
    if out_path is None:
        paths_cfg = cfg.get("paths", {})
        if "fire_climatology_tif" in paths_cfg:
            out_path = get_path(cfg, "fire_climatology_tif")
        else:
            out_path = str(Path(get_path(cfg, "output_dir")) / "fire_climatology.tif")

    years  = _parse_range(args.years)
    months = _parse_range(args.months)

    print("=" * 60)
    print("FIRE FREQUENCY CLIMATOLOGY  [V2 static channel]")
    print("=" * 60)
    print(f"  Training years  : {years}")
    print(f"  Months          : {months}  (fire season)")
    print(f"  Reference grid  : {ref_tif}")
    print(f"  Hotspot CSV     : {hotspot_csv}")
    print(f"  Output          : {out_path}")
    print(f"  Log1p transform : {not args.no_log}")
    print(f"  Count mode      : {'detections' if args.detections else 'days'}")
    print("=" * 60)
    print()

    build_fire_climatology(
        hotspot_csv=hotspot_csv,
        fwi_reference_tif=ref_tif,
        years=years,
        months=months,
        output_path=out_path,
        log_transform=not args.no_log,
        count_mode="detections" if args.detections else "days",
    )


if __name__ == "__main__":
    main()
