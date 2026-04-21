#!/usr/bin/env python3
"""
Build standalone fire-label (T,H,W) uint8 stack files with provenance.

Two schemes produced (run once for each, they coexist):
    - CWFIS-only:   legacy label (matches 4y SOTA v4 eval)
    - NBAC+NFDB:    new label (drift-free, per LABEL_DECISION_2026_04_21.md)

Each produces a .npy (the label array) + sidecar .json (provenance):
    fire_labels_{scheme}_{start}_{end}_{H}x{W}_r{r}.npy
    fire_labels_{scheme}_{start}_{end}_{H}x{W}_r{r}.json

This file is the **ground-truth artifact** for all downstream training /
eval. train_v3.py reads these on cache hit (same filename pattern as
before); this script lets you pre-build labels outside the training
pipeline for transparency and A/B comparison.

Usage:
    # Build CWFIS-only label (matches old behavior)
    python scripts/build_fire_labels.py --scheme cwfis \\
        --start 2000-05-01 --end 2025-12-21 --dilate_radius 14 \\
        --output_dir data/fire_labels

    # Build NBAC+NFDB label (new default)
    python scripts/build_fire_labels.py --scheme nbac_nfdb \\
        --start 2000-05-01 --end 2025-12-21 --dilate_radius 14 \\
        --nfdb_min_size_ha 1.0 --exclude_prescribed \\
        --output_dir data/fire_labels
"""

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

# Ensure src/ is importable when script is run directly
for _p in Path(__file__).resolve().parents:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break

from src.data_ops.processing.rasterize_hotspots import (
    load_hotspot_data, rasterize_hotspots_batch, load_nfdb_as_hotspot_df,
)
from src.data_ops.processing.rasterize_burn_polygons import (
    load_nbac, rasterize_nbac_batch,
)


def build_date_list(start: str, end: str):
    sd = date.fromisoformat(start)
    ed = date.fromisoformat(end)
    dates = []
    cur = sd
    while cur <= ed:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def build_cwfis(hotspot_csv: str, dates, profile):
    df = load_hotspot_data(hotspot_csv)
    print(f"  CWFIS: {len(df):,} hotspot records loaded")
    return rasterize_hotspots_batch(df, dates, profile)


def build_nbac_nfdb(nbac_path: str, nfdb_path: str,
                    nbac_date_source: str,
                    nfdb_min_size_ha: float,
                    exclude_prescribed: bool,
                    dates, profile):
    H, W = int(profile["height"]), int(profile["width"])
    stack = np.zeros((len(dates), H, W), dtype=np.uint8)

    # --- NBAC polygons ---
    nbac = load_nbac(nbac_path)
    print(f"  NBAC: {len(nbac):,} polygons loaded")
    if exclude_prescribed and "PRESCRIBED" in nbac.columns:
        _before = len(nbac)
        # NBAC PRESCRIBED: 'true' = prescribed burn, NaN = wildfire (audit 2026-04-21).
        nbac = nbac[nbac["PRESCRIBED"].isna()].copy()
        print(f"  NBAC: dropped {_before - len(nbac)} prescribed polygons "
              f"({len(nbac):,} remain)")
    nbac_stack = rasterize_nbac_batch(nbac, dates, profile,
                                      date_source=nbac_date_source)
    np.maximum(stack, nbac_stack, out=stack)
    nbac_pos = int(stack.sum())
    print(f"  after NBAC: {nbac_pos:,} positive pixels")

    # --- NFDB points ---
    keep_causes = {"H", "N", "U"} if exclude_prescribed else None
    nfdb = load_nfdb_as_hotspot_df(
        nfdb_path,
        min_size_ha=nfdb_min_size_ha,
        causes=keep_causes,
    )
    print(f"  NFDB: {len(nfdb):,} fires loaded "
          f"(size >= {nfdb_min_size_ha} ha, excl prescribed={exclude_prescribed})")
    nfdb_stack = rasterize_hotspots_batch(nfdb, dates, profile)
    before = int(stack.sum())
    np.maximum(stack, nfdb_stack, out=stack)
    added = int(stack.sum()) - before
    print(f"  after NFDB: +{added:,} pixels")

    return stack, {"nbac_positive": nbac_pos, "nfdb_added": added}


def dilate_stack(stack: np.ndarray, r: int):
    if r <= 0:
        return stack
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= r ** 2)
    T = stack.shape[0]
    out = np.zeros_like(stack)
    print(f"  Dilating {T} frames with r={r} px disk...")
    t0 = time.time()
    for t in range(T):
        if stack[t].any():
            out[t] = binary_dilation(stack[t], structure=disk).astype(np.uint8)
        if (t + 1) % 1000 == 0:
            print(f"    {t+1}/{T}  ({time.time()-t0:.0f}s)")
    return out


def per_year_positive_count(stack, dates):
    """Total positive pixels per year (sum across days)."""
    import pandas as pd
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    s = pd.Series(stack.reshape(stack.shape[0], -1).sum(axis=1),
                  index=idx)
    return s.groupby(idx.year).sum().astype(int).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=("cwfis", "nbac_nfdb"),
                    help="Label-source scheme")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--dilate_radius", type=int, default=14)
    ap.add_argument("--reference",
                    default="data/fwi_data/fwi_20230801.tif",
                    help="Reference TIF for H/W/CRS/transform")
    ap.add_argument("--hotspot",
                    default="data/hotspot/hotspot_2000_2025.csv")
    ap.add_argument("--nbac",
                    default="data/burn_scars_raw/NBAC_1972to2024_shp.zip")
    ap.add_argument("--nfdb",
                    default="data/nfdb/NFDB_point.zip")
    ap.add_argument("--nbac_date_source", default="AG", choices=("AG", "HS"))
    ap.add_argument("--nfdb_min_size_ha", type=float, default=1.0)
    ap.add_argument("--exclude_prescribed", action="store_true", default=True)
    ap.add_argument("--include_prescribed", action="store_true",
                    help="Override --exclude_prescribed (include managed burns)")
    ap.add_argument("--output_dir", default="data/fire_labels")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.include_prescribed:
        args.exclude_prescribed = False

    # Reference grid
    with rasterio.open(args.reference) as src:
        profile = src.profile
    H, W = int(profile["height"]), int(profile["width"])
    print(f"Reference grid: {H}×{W} CRS={profile['crs']}")

    # Date list
    dates = build_date_list(args.start, args.end)
    T = len(dates)
    print(f"Date range: {dates[0]}..{dates[-1]}  ({T} days)")

    # Output paths
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = (f"fire_labels_{args.scheme}_{args.start}_{args.end}_"
            f"{H}x{W}_r{args.dilate_radius}")
    npy_path = outdir / f"{stem}.npy"
    json_path = outdir / f"{stem}.json"

    if npy_path.exists() and not args.overwrite:
        print(f"[SKIP] {npy_path} already exists. Use --overwrite.")
        return

    t_start = time.time()

    # Build raw stack
    print(f"\n=== Building raw (undilated) stack for scheme={args.scheme} ===")
    extra_provenance = {}
    if args.scheme == "cwfis":
        stack = build_cwfis(args.hotspot, dates, profile)
    else:  # nbac_nfdb
        stack, extra_provenance = build_nbac_nfdb(
            args.nbac, args.nfdb,
            nbac_date_source=args.nbac_date_source,
            nfdb_min_size_ha=args.nfdb_min_size_ha,
            exclude_prescribed=args.exclude_prescribed,
            dates=dates, profile=profile)

    raw_positive = int(stack.sum())
    print(f"  raw positive: {raw_positive:,}")

    # Dilate
    stack = dilate_stack(stack, args.dilate_radius)
    dilated_positive = int(stack.sum())
    print(f"  dilated positive: {dilated_positive:,}  "
          f"(expansion {dilated_positive / max(raw_positive, 1):.1f}x)")

    # Per-year breakdown
    yearly = per_year_positive_count(stack, dates)

    # Save artifact + provenance
    np.save(npy_path, stack)
    print(f"\nSaved label array: {npy_path} ({npy_path.stat().st_size/1e9:.2f} GB)")

    provenance = {
        "scheme": args.scheme,
        "date_range": [args.start, args.end],
        "T": T,
        "H": H,
        "W": W,
        "dilate_radius": args.dilate_radius,
        "reference": args.reference,
        "crs": str(profile["crs"]),
        "raw_positive": raw_positive,
        "dilated_positive": dilated_positive,
        "positive_rate": float(stack.mean()),
        "yearly_positive": yearly,
        "build_time_sec": round(time.time() - t_start, 1),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "sources": {
            "cwfis_csv": args.hotspot if args.scheme == "cwfis" else None,
            "nbac_shp": args.nbac if args.scheme == "nbac_nfdb" else None,
            "nfdb_shp": args.nfdb if args.scheme == "nbac_nfdb" else None,
        },
        "filters": {
            "nfdb_min_size_ha": args.nfdb_min_size_ha
                if args.scheme == "nbac_nfdb" else None,
            "nbac_date_source": args.nbac_date_source
                if args.scheme == "nbac_nfdb" else None,
            "exclude_prescribed": args.exclude_prescribed
                if args.scheme == "nbac_nfdb" else None,
        },
        **extra_provenance,
    }
    with open(json_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"Saved provenance:    {json_path}")

    print(f"\nTotal build time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
