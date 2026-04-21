#!/usr/bin/env python3
"""
Rasterize NBAC (National Burned Area Composite) burn polygons to daily binary
masks matching the EPSG:3978 2281×2709 training grid.

Purpose (anti-drift label fusion):
    CWFIS hotspots have strong temporal drift (MODIS 2000-2011 vs MODIS+VIIRS
    2012+; VIIRS detects 3-10× more small fires). NBAC is built from Landsat
    dNBR imagery processed AFTER fire season, so it is temporally stable and
    catches small fires that MODIS missed. Rasterizing NBAC polygons and
    OR-ing into the CWFIS hotspot label mask restores these missed fires.

Approach:
    For each NBAC polygon, fill its spatial extent at every day between its
    active date range. Two date-range choices:
        - AG_SDATE..AG_EDATE: agency-reported (earlier, can predate satellite
          detection by weeks — best for catching early ignitions)
        - HS_SDATE..HS_EDATE: satellite hotspot-observed range (narrower,
          more conservative)
    Default: AG_SDATE..AG_EDATE.

Output:
    Binary (T, H, W) uint8 stack, 1 = NBAC polygon active that day, 0 = not.

Performance:
    For 22y × 30k polygons with ~30-day active span each, uses a single-pass
    per-polygon rasterize + OR into the relevant date slice. Expected runtime
    ~5-10 min on a single CPU.

Public API (mirrors rasterize_hotspots.py structure):
    load_nbac(shp_path) -> gpd.GeoDataFrame
    rasterize_nbac_batch(gdf, date_list, profile, date_source="AG") -> (T,H,W) uint8

Usage as CLI (smoke-test):
    python -m src.data_ops.processing.rasterize_burn_polygons \\
        --nbac data/burn_scars_raw/NBAC_1972to2024_shp.zip \\
        --reference data/fwi_data/fwi_20230801.tif \\
        --start 2023-07-01 --end 2023-07-31 \\
        --output /tmp/nbac_test.npy
"""

import argparse
import glob
import os
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_zip_if_needed(shp_or_zip_path: Path) -> Path:
    """If given a .zip, extract to a sibling dir and return the .shp path."""
    p = Path(shp_or_zip_path)
    if p.suffix.lower() == ".shp":
        return p
    if p.suffix.lower() != ".zip":
        raise ValueError(f"Expected .shp or .zip, got: {p}")

    extract_dir = p.parent / (p.stem + "_extract")
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Find .shp already extracted
    shps = list(extract_dir.glob("*.shp"))
    if shps:
        return shps[0]

    with zipfile.ZipFile(p) as zf:
        # Only extract shapefile sidecars (.shp/.shx/.dbf/.prj/.cpg)
        for name in zf.namelist():
            if name.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                zf.extract(name, extract_dir)

    shps = list(extract_dir.rglob("*.shp"))
    if not shps:
        raise RuntimeError(f"No .shp found after extracting {p}")
    return shps[0]


def load_nbac(shp_path):
    """Load NBAC polygon layer. Accepts .shp path or .zip archive path.

    Returns GeoDataFrame with at least: YEAR, HS_SDATE, HS_EDATE, AG_SDATE,
    AG_EDATE, POLY_HA, FIRECAUS, geometry.
    """
    shp = _extract_zip_if_needed(Path(shp_path))
    gdf = gpd.read_file(shp)
    # Ensure date columns are datetime
    for col in ("HS_SDATE", "HS_EDATE", "AG_SDATE", "AG_EDATE"):
        if col in gdf.columns:
            gdf[col] = pd.to_datetime(gdf[col], errors="coerce")
    return gdf


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_nbac_batch(gdf, date_list, profile, date_source="AG",
                        progress_every=5000):
    """Rasterize NBAC polygons to daily binary masks.

    Args:
        gdf: GeoDataFrame from load_nbac().
        date_list: sorted list of datetime.date / pandas.Timestamp (length T).
        profile: rasterio profile (from reference TIF). Must include
                 'transform', 'crs', 'height', 'width'.
        date_source: "AG" (agency-reported SDATE/EDATE; recommended) or "HS"
                     (satellite hotspot-observed SDATE/EDATE; narrower).
        progress_every: print progress every N polygons.

    Returns:
        (T, H, W) np.ndarray uint8, 1 where any NBAC polygon active on that day.
    """
    if date_source not in ("AG", "HS"):
        raise ValueError(f"date_source must be 'AG' or 'HS', got {date_source!r}")
    sdate_col = f"{date_source}_SDATE"
    edate_col = f"{date_source}_EDATE"

    H, W = int(profile["height"]), int(profile["width"])
    T = len(date_list)
    transform = profile["transform"]
    target_crs = profile["crs"]

    # Normalize input dates to pandas timestamps (for comparison against gdf)
    dates_pd = pd.to_datetime([pd.Timestamp(d) for d in date_list])
    date_to_idx = {d.date(): i for i, d in enumerate(dates_pd)}
    first_date = dates_pd.min()
    last_date = dates_pd.max()

    # Reproject polygons to target CRS if needed
    if str(gdf.crs).upper() != str(target_crs).upper():
        print(f"  [NBAC] reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # Filter polygons to overlap our date range (any-day overlap)
    mask = (
        gdf[sdate_col].notna()
        & gdf[edate_col].notna()
        & (gdf[edate_col] >= first_date)
        & (gdf[sdate_col] <= last_date)
    )
    gdf_active = gdf.loc[mask].copy()
    print(f"  [NBAC] {len(gdf_active):,} / {len(gdf):,} polygons overlap "
          f"date range {first_date.date()}..{last_date.date()} "
          f"(using {date_source}_SDATE/EDATE)")

    out = np.zeros((T, H, W), dtype=np.uint8)

    t0 = time.time()
    n_ok = 0
    for idx, row in enumerate(gdf_active.itertuples(index=False)):
        sdate = getattr(row, sdate_col)
        edate = getattr(row, edate_col)
        # Clip to our date range
        sdate = max(sdate, first_date)
        edate = min(edate, last_date)
        # Find (ts, te) slice indices
        cur = sdate.date()
        end = edate.date()
        if cur > end:
            continue
        ts = date_to_idx.get(cur)
        te = date_to_idx.get(end)
        if ts is None or te is None:
            # Find nearest index (dates may skip)
            # date_to_idx keys are dates in date_list; for any cur not in it,
            # find first date >= cur
            all_dates = [d.date() for d in dates_pd]
            ts = next((i for i, d in enumerate(all_dates) if d >= cur), None)
            te = next((i for i, d in enumerate(all_dates) if d > end), T) - 1
            if ts is None or ts > te:
                continue

        # Rasterize this polygon once to a (H, W) mask
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            poly_mask = rio_rasterize(
                [(geom, 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )
        except Exception as e:
            print(f"  [NBAC] skip polygon idx={idx}: {e}")
            continue

        # OR into every day's layer in [ts, te]
        np.maximum(out[ts:te + 1], poly_mask, out=out[ts:te + 1])
        n_ok += 1

        if (idx + 1) % progress_every == 0:
            dt = time.time() - t0
            rate = (idx + 1) / max(dt, 1e-3)
            print(f"  [NBAC] {idx+1:,}/{len(gdf_active):,} polygons "
                  f"rasterized  ({dt:.0f}s, {rate:.0f}/s)")

    dt = time.time() - t0
    pos = int(out.sum())
    print(f"  [NBAC] done: {n_ok:,} polygons used  ({dt:.0f}s)  "
          f"positive pixels total={pos:,}  "
          f"pos_rate/day_avg={100 * pos / out.size:.4f}%")
    return out


# ---------------------------------------------------------------------------
# CLI (smoke-test)
# ---------------------------------------------------------------------------

def _smoke_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbac", required=True,
                    help="Path to NBAC shapefile or zip")
    ap.add_argument("--reference", required=True,
                    help="Reference TIF (profile source, e.g. fwi_YYYYMMDD.tif)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--date_source", default="AG", choices=("AG", "HS"))
    ap.add_argument("--output", default=None, help="Save (T,H,W) .npy")
    args = ap.parse_args()

    with rasterio.open(args.reference) as src:
        profile = src.profile
    H, W = int(profile["height"]), int(profile["width"])

    sdate = date.fromisoformat(args.start)
    edate = date.fromisoformat(args.end)
    date_list = []
    cur = sdate
    while cur <= edate:
        date_list.append(cur)
        cur += timedelta(days=1)
    print(f"Grid: {H}×{W} CRS={profile['crs']}  Dates: {len(date_list)} days")

    print(f"Loading NBAC: {args.nbac}")
    gdf = load_nbac(args.nbac)
    print(f"  loaded {len(gdf):,} polygons, CRS={gdf.crs}")

    stack = rasterize_nbac_batch(gdf, date_list, profile,
                                 date_source=args.date_source)
    print(f"\nResult: shape={stack.shape} dtype={stack.dtype}")
    print(f"  total positive: {int(stack.sum()):,}")
    print(f"  daily pos count: min={int(stack.sum(axis=(1,2)).min())} "
          f"max={int(stack.sum(axis=(1,2)).max())} "
          f"mean={stack.sum(axis=(1,2)).mean():.0f}")

    if args.output:
        np.save(args.output, stack)
        print(f"Saved to {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    _smoke_cli()
