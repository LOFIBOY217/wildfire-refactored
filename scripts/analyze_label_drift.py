#!/usr/bin/env python3
"""
Cross-validate CWFIS hotspots, NFDB (reported fires), and NBAC (post-fire
Landsat burn polygons) by year to quantify detection drift.

Hypothesis: CWFIS hotspot COUNTS are heavily drift-biased (MODIS 2000-2011
vs MODIS+VIIRS 2012+). NFDB reported fire COUNTS and NBAC burned AREA are
much more temporally stable because they don't depend on satellite
active-fire detection sensitivity.

Output: per-year CSV + stdout table showing:
    - CWFIS hotspot records      (count, sum of positive pixels)
    - NFDB reported fires        (count, sum of SIZE_HA)
    - NBAC burn polygons         (count, sum of POLY_HA)
    - Ratios vs 2023 baseline    (exposes drift)

Usage:
    python scripts/analyze_label_drift.py \\
        --hotspot data/hotspot/hotspot_2000_2025.csv \\
        --nfdb data/nfdb/NFDB_point.zip \\
        --nbac data/burn_scars_raw/NBAC_1972to2024_shp.zip \\
        --out data/audit/label_drift.csv \\
        --start 2000 --end 2024
"""

import argparse
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _extract_shp(zip_path: Path) -> Path:
    extract_dir = zip_path.parent / (zip_path.stem + "_extract")
    extract_dir.mkdir(parents=True, exist_ok=True)
    shps = list(extract_dir.glob("*.shp"))
    if shps:
        return shps[0]
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                zf.extract(name, extract_dir)
    return next(extract_dir.rglob("*.shp"))


def load_hotspot_yearly(csv_path):
    """CWFIS/FIRMS hotspot CSV → per-year count."""
    df = pd.read_csv(csv_path, usecols=["acq_date"])
    df["year"] = pd.to_datetime(df["acq_date"]).dt.year
    return df.groupby("year").size().rename("cwfis_hotspot_count")


def load_nfdb_yearly(shp_or_zip_path):
    """NFDB point → per-year (count, sum SIZE_HA)."""
    import geopandas as gpd
    p = Path(shp_or_zip_path)
    shp = _extract_shp(p) if p.suffix.lower() == ".zip" else p
    gdf = gpd.read_file(shp)
    if "REP_DATE" in gdf.columns:
        gdf["year"] = pd.to_datetime(gdf["REP_DATE"], errors="coerce").dt.year
    else:
        gdf["year"] = gdf["YEAR"]
    gdf = gdf[gdf["year"].notna()]
    gdf["year"] = gdf["year"].astype(int)
    counts = gdf.groupby("year").size().rename("nfdb_fire_count")
    size_ha = gdf.groupby("year")["SIZE_HA"].sum().rename("nfdb_sum_ha")
    return pd.concat([counts, size_ha], axis=1)


def load_nbac_yearly(shp_or_zip_path):
    """NBAC polygon → per-year (count, sum POLY_HA)."""
    import geopandas as gpd
    p = Path(shp_or_zip_path)
    shp = _extract_shp(p) if p.suffix.lower() == ".zip" else p
    gdf = gpd.read_file(shp)
    gdf = gdf[gdf["YEAR"].notna()]
    gdf["year"] = gdf["YEAR"].astype(int)
    counts = gdf.groupby("year").size().rename("nbac_polygon_count")
    poly_ha = gdf.groupby("year")["POLY_HA"].sum().rename("nbac_sum_ha")
    return pd.concat([counts, poly_ha], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotspot", default="data/hotspot/hotspot_2000_2025.csv")
    ap.add_argument("--nfdb", default="data/nfdb/NFDB_point.zip")
    ap.add_argument("--nbac", default="data/burn_scars_raw/NBAC_1972to2024_shp.zip")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--out", default="data/audit/label_drift.csv")
    ap.add_argument("--baseline-year", type=int, default=2023,
                    help="Normalize ratios relative to this year")
    args = ap.parse_args()

    print(f"[drift] Loading CWFIS hotspots from {args.hotspot} ...")
    cwfis = load_hotspot_yearly(args.hotspot)

    print(f"[drift] Loading NFDB fires from {args.nfdb} ...")
    nfdb = load_nfdb_yearly(args.nfdb)

    print(f"[drift] Loading NBAC polygons from {args.nbac} ...")
    nbac = load_nbac_yearly(args.nbac)

    df = pd.concat([cwfis, nfdb, nbac], axis=1).sort_index()
    df = df.loc[args.start:args.end].fillna(0).astype({
        "cwfis_hotspot_count": int,
        "nfdb_fire_count": int,
        "nfdb_sum_ha": float,
        "nbac_polygon_count": int,
        "nbac_sum_ha": float,
    })

    # Baseline year ratios
    base = df.loc[args.baseline_year]
    for col in df.columns:
        if base[col] > 0:
            df[col + f"_vs_{args.baseline_year}"] = df[col] / base[col]
        else:
            df[col + f"_vs_{args.baseline_year}"] = np.nan

    print("\n=== YEARLY SUMMARY ===")
    cols = ["cwfis_hotspot_count", "nfdb_fire_count",
            "nbac_polygon_count", "nbac_sum_ha"]
    print(df[cols].to_string())

    print(f"\n=== RATIOS (vs {args.baseline_year}) — drift indicator ===")
    ratio_cols = [c + f"_vs_{args.baseline_year}" for c in cols]
    print(df[ratio_cols].to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\n=== DRIFT VERDICT ===")
    # If detection drift exists, CWFIS ratio(2001/2023) << NFDB/NBAC ratio(2001/2023)
    if args.start in df.index:
        c_ratio = df.at[args.start, "cwfis_hotspot_count"] / max(base["cwfis_hotspot_count"], 1)
        n_ratio = df.at[args.start, "nfdb_fire_count"] / max(base["nfdb_fire_count"], 1)
        b_ratio = df.at[args.start, "nbac_sum_ha"] / max(base["nbac_sum_ha"], 1)
        print(f"  CWFIS hotspot count      {args.start}/{args.baseline_year} ratio: {c_ratio:.4f}")
        print(f"  NFDB fire count          {args.start}/{args.baseline_year} ratio: {n_ratio:.4f}")
        print(f"  NBAC burned area (ha)    {args.start}/{args.baseline_year} ratio: {b_ratio:.4f}")
        if c_ratio > 0 and n_ratio / max(c_ratio, 1e-6) > 5:
            print(f"\n  ⚠ CWFIS drift CONFIRMED: NFDB ratio > 5x CWFIS ratio")
            print(f"     → Pre-{args.baseline_year} CWFIS hotspot labels are "
                  f"underreported by ~{n_ratio / max(c_ratio, 1e-6):.1f}x")
            print(f"     → --label_fusion (NBAC + NFDB) recommended")
        else:
            print(f"\n  CWFIS drift modest — label_fusion gain may be small")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\n[drift] Saved to {out_path}")


if __name__ == "__main__":
    main()
