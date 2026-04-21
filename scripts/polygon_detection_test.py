#!/usr/bin/env python3
"""
Raw-geometry NBAC polygon detection test.

For each year and each NBAC burn polygon (a "ground-truth fire event"):
  - Count how many CWFIS hotspots fall INSIDE the polygon during its
    active date range (HS_SDATE..HS_EDATE or AG_SDATE..AG_EDATE).
  - Count how many NFDB ignition points fall INSIDE the polygon in
    the same year.

Then per year compute:
  - CWFIS detection rate: % of NBAC polygons with at least one CWFIS hit
  - NFDB detection rate:  % of NBAC polygons with at least one NFDB hit
  - Joint rate:           % detected by both
  - Miss rate:            % detected by neither ← pure drift indicator

Answers the question: "From what year can CWFIS detect all recorded fires?"
That's the year where CWFIS detection rate > 95%.

Optional stratification: report detection rate by polygon size bucket
(<10ha, 10-100ha, 100-1000ha, 1000+ha). Drift is expected to be worst
for small fires pre-VIIRS.

NO rasterization, NO dilation — pure GeoPandas spatial join on raw
geometries.

Usage:
    python scripts/polygon_detection_test.py --start 2000 --end 2024
"""

import argparse
import sys
import time
import zipfile
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotspot", default="data/hotspot/hotspot_2000_2025.csv")
    ap.add_argument("--nfdb", default="data/nfdb/NFDB_point.zip")
    ap.add_argument("--nbac", default="data/burn_scars_raw/NBAC_1972to2024_shp.zip")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--date_source", default="HS", choices=("HS", "AG", "YEAR"),
                    help="NBAC date-range source for CWFIS match. HS=hotspot "
                         "observation window (narrow, strict), AG=agency window "
                         "(wider), YEAR=entire year (loose).")
    ap.add_argument("--out", default="data/audit/polygon_detection.csv")
    args = ap.parse_args()

    import geopandas as gpd
    from shapely.geometry import Point

    # ------------------------------------------------------------
    # Load three sources in their NATIVE geometry (no rasterization)
    # ------------------------------------------------------------
    print(f"[polygon-detect] Loading NBAC polygons from {args.nbac}...")
    nbac_shp = _extract_shp(Path(args.nbac))
    nbac = gpd.read_file(nbac_shp)
    print(f"  loaded {len(nbac):,} polygons")
    # Parse dates
    for col in ("HS_SDATE", "HS_EDATE", "AG_SDATE", "AG_EDATE"):
        if col in nbac.columns:
            nbac[col] = pd.to_datetime(nbac[col], errors="coerce")
    nbac["year"] = nbac["YEAR"].astype(int)

    # Always project to EPSG:3978 (Canada Atlas Lambert) — both CWFIS and
    # NFDB points will be projected into this CRS for clean within-test.
    target_crs = "EPSG:3978"
    if str(nbac.crs).upper() != target_crs:
        nbac = nbac.to_crs(target_crs)

    print(f"[polygon-detect] Loading CWFIS hotspots from {args.hotspot}...")
    cwfis_df = pd.read_csv(args.hotspot,
                           usecols=["latitude", "longitude", "acq_date"])
    cwfis_df["date"] = pd.to_datetime(cwfis_df["acq_date"], errors="coerce")
    cwfis_df["year"] = cwfis_df["date"].dt.year
    cwfis_gdf = gpd.GeoDataFrame(
        cwfis_df,
        geometry=gpd.points_from_xy(cwfis_df.longitude, cwfis_df.latitude),
        crs="EPSG:4326").to_crs(target_crs)
    print(f"  {len(cwfis_gdf):,} total CWFIS hotspots")

    print(f"[polygon-detect] Loading NFDB fires from {args.nfdb}...")
    nfdb_shp = _extract_shp(Path(args.nfdb))
    nfdb_gdf = gpd.read_file(nfdb_shp)
    if str(nfdb_gdf.crs).upper() != target_crs:
        nfdb_gdf = nfdb_gdf.to_crs(target_crs)
    if "REP_DATE" in nfdb_gdf.columns:
        nfdb_gdf["date"] = pd.to_datetime(nfdb_gdf["REP_DATE"], errors="coerce")
        nfdb_gdf["year"] = nfdb_gdf["date"].dt.year
    else:
        nfdb_gdf["year"] = nfdb_gdf["YEAR"].astype(int)
    print(f"  {len(nfdb_gdf):,} total NFDB fires")

    # ------------------------------------------------------------
    # Per-year spatial join
    # ------------------------------------------------------------
    stats = []
    for yr in range(args.start, args.end + 1):
        t0 = time.time()
        print(f"\n[polygon-detect] Year {yr}...", flush=True)

        nbac_yr = nbac[nbac["year"] == yr].copy()
        if len(nbac_yr) == 0:
            print(f"    no NBAC polygons — skip")
            continue

        # Filter CWFIS / NFDB to this year first
        cwfis_yr = cwfis_gdf[cwfis_gdf["year"] == yr]
        nfdb_yr = nfdb_gdf[nfdb_gdf["year"] == yr]
        print(f"    NBAC polygons: {len(nbac_yr):,}")
        print(f"    CWFIS hotspots: {len(cwfis_yr):,}   NFDB fires: {len(nfdb_yr):,}")

        # --- CWFIS: spatial join (within = point inside polygon) ---
        # Give each polygon a unique id for grouping
        nbac_yr = nbac_yr.reset_index(drop=True)
        nbac_yr["_pid"] = np.arange(len(nbac_yr))
        if len(cwfis_yr) > 0:
            j = gpd.sjoin(
                cwfis_yr[["date", "geometry"]],
                nbac_yr[["_pid", "HS_SDATE", "HS_EDATE",
                         "AG_SDATE", "AG_EDATE", "geometry"]],
                how="inner", predicate="within")
            # Apply date-range filter per polygon
            if args.date_source != "YEAR" and len(j) > 0:
                sd_col = f"{args.date_source}_SDATE"
                ed_col = f"{args.date_source}_EDATE"
                mask = (j["date"] >= j[sd_col]) & (j["date"] <= j[ed_col])
                j = j[mask]
            cwfis_hits_per_poly = j.groupby("_pid").size()
        else:
            cwfis_hits_per_poly = pd.Series(dtype=int)

        # --- NFDB: spatial join (within; just year filter) ---
        if len(nfdb_yr) > 0:
            jn = gpd.sjoin(
                nfdb_yr[["geometry"]],
                nbac_yr[["_pid", "geometry"]],
                how="inner", predicate="within")
            nfdb_hits_per_poly = jn.groupby("_pid").size()
        else:
            nfdb_hits_per_poly = pd.Series(dtype=int)

        # --- Aggregate detection per polygon ---
        nbac_yr["cwfis_hits"] = nbac_yr["_pid"].map(cwfis_hits_per_poly).fillna(0).astype(int)
        nbac_yr["nfdb_hits"] = nbac_yr["_pid"].map(nfdb_hits_per_poly).fillna(0).astype(int)
        nbac_yr["cwfis_det"] = nbac_yr["cwfis_hits"] > 0
        nbac_yr["nfdb_det"] = nbac_yr["nfdb_hits"] > 0
        nbac_yr["both_det"] = nbac_yr["cwfis_det"] & nbac_yr["nfdb_det"]
        nbac_yr["none_det"] = ~nbac_yr["cwfis_det"] & ~nbac_yr["nfdb_det"]

        n = len(nbac_yr)
        row = {
            "year": yr,
            "n_polygons": n,
            "cwfis_det": int(nbac_yr["cwfis_det"].sum()),
            "nfdb_det":  int(nbac_yr["nfdb_det"].sum()),
            "both_det":  int(nbac_yr["both_det"].sum()),
            "none_det":  int(nbac_yr["none_det"].sum()),
            "cwfis_det_rate": round(nbac_yr["cwfis_det"].mean(), 3),
            "nfdb_det_rate":  round(nbac_yr["nfdb_det"].mean(), 3),
            "both_det_rate":  round(nbac_yr["both_det"].mean(), 3),
            "none_det_rate":  round(nbac_yr["none_det"].mean(), 3),
        }

        # Size-stratified CWFIS detection rate (key for drift)
        if "POLY_HA" in nbac_yr.columns:
            for lo, hi, label in [(0, 10, "lt10"), (10, 100, "10_100"),
                                   (100, 1000, "100_1k"),
                                   (1000, 1e12, "gt1k")]:
                m = (nbac_yr["POLY_HA"] >= lo) & (nbac_yr["POLY_HA"] < hi)
                n_bucket = int(m.sum())
                if n_bucket > 0:
                    rate = nbac_yr.loc[m, "cwfis_det"].mean()
                else:
                    rate = np.nan
                row[f"cwfis_det_rate_{label}"] = round(float(rate), 3) if not np.isnan(rate) else np.nan
                row[f"n_poly_{label}"] = n_bucket

        dt = time.time() - t0
        print(f"    {dt:.1f}s  CWFIS det = {row['cwfis_det_rate']*100:.1f}%  "
              f"NFDB det = {row['nfdb_det_rate']*100:.1f}%  "
              f"neither = {row['none_det_rate']*100:.1f}%",
              flush=True)
        stats.append(row)

    df = pd.DataFrame(stats).set_index("year")

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    print("\n=== YEARLY NBAC POLYGON DETECTION RATES (raw geometry) ===")
    display = ["n_polygons", "cwfis_det_rate", "nfdb_det_rate",
               "both_det_rate", "none_det_rate"]
    print(df[display].to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== CWFIS DETECTION BY POLYGON SIZE ===")
    size_cols = [c for c in df.columns if c.startswith("cwfis_det_rate_")]
    if size_cols:
        print(df[size_cols].to_string(float_format=lambda x: f"{x:.3f}"
                                       if pd.notna(x) else "NA"))

    print("\n=== CONVERGENCE YEAR ===")
    convergence_thresholds = [0.80, 0.90, 0.95]
    for th in convergence_thresholds:
        mask = df["cwfis_det_rate"] >= th
        first_yr = df.index[mask].min() if mask.any() else None
        if first_yr is not None:
            print(f"  first year CWFIS_det_rate >= {th*100:.0f}%: "
                  f"{int(first_yr)}")
        else:
            print(f"  first year CWFIS_det_rate >= {th*100:.0f}%: "
                  f"NEVER (max = {df['cwfis_det_rate'].max():.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"\n[polygon-detect] saved to {args.out}")


if __name__ == "__main__":
    main()
