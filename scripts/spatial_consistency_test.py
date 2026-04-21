#!/usr/bin/env python3
"""
Spatial consistency test across years: CWFIS vs NFDB vs NBAC.

For each year Y, rasterize the annual fire footprint from all three
sources (union over days/polygons within year) and compute per-pixel
Venn-diagram statistics:

    cwfis_only   = pixels only in CWFIS (satellite hotspot, not reported, not burn polygon)
    nfdb_only    = pixels only in NFDB (human-reported ignition, not seen by satellite, not polygon)
    nbac_only    = pixels only in NBAC (Landsat burn scar, CWFIS/NFDB missed)
    two_source   = pixels in exactly two sources
    all_three    = pixels in all three (high-confidence)

Hypothesis (to validate):
    - Early years (2000-2011, MODIS-only era):
        nbac_only + nfdb_only should be HIGH (drift — sources complement CWFIS)
    - Recent years (2012+, MODIS+VIIRS era):
        all_three + two_source should dominate (sources agree)

Point sources (CWFIS, NFDB) are dilated by r=14 pixels (~28 km) to make
them spatially comparable to NBAC's full burn-polygon footprint — same
dilation radius used in training labels.

Output: per-year CSV + formatted stdout table.

Usage:
    python scripts/spatial_consistency_test.py --start 2000 --end 2024
"""

import argparse
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import binary_dilation


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


def _rasterize_points(lat, lon, src_crs, profile):
    """Rasterize point coords (WGS84) onto (H, W) EPSG:3978 grid."""
    from pyproj import Transformer
    H, W = int(profile["height"]), int(profile["width"])
    transform = profile["transform"]
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(lat) == 0:
        return mask
    tr = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    xs, ys = tr.transform(lon, lat)
    from rasterio.transform import rowcol
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    mask[rows[valid], cols[valid]] = 1
    return mask


def _dilate(mask, r):
    if r <= 0:
        return mask
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= r ** 2)
    return binary_dilation(mask, structure=disk).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotspot", default="data/hotspot/hotspot_2000_2025.csv")
    ap.add_argument("--nfdb", default="data/nfdb/NFDB_point.zip")
    ap.add_argument("--nbac", default="data/burn_scars_raw/NBAC_1972to2024_shp.zip")
    ap.add_argument("--reference", default="data/fwi_data/fwi_20230801.tif",
                    help="Reference TIF for grid/CRS (EPSG:3978 2281x2709)")
    ap.add_argument("--dilate_radius", type=int, default=14,
                    help="Dilation radius for CWFIS+NFDB points (matches training)")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--out", default="data/audit/spatial_consistency.csv")
    args = ap.parse_args()

    # --- Load once, group by year ---
    print(f"[consistency] Loading CWFIS hotspots from {args.hotspot}...")
    cwfis = pd.read_csv(args.hotspot)
    cwfis["year"] = pd.to_datetime(cwfis["acq_date"]).dt.year
    cwfis_by_year = {int(y): g for y, g in cwfis.groupby("year")}
    print(f"  {len(cwfis):,} total hotspot rows")

    print(f"[consistency] Loading NFDB points from {args.nfdb}...")
    import geopandas as gpd
    nfdb_shp = _extract_shp(Path(args.nfdb))
    nfdb = gpd.read_file(nfdb_shp)
    if nfdb.crs != "EPSG:4326":
        nfdb = nfdb.to_crs("EPSG:4326")
    if "REP_DATE" in nfdb.columns:
        nfdb["year"] = pd.to_datetime(nfdb["REP_DATE"], errors="coerce").dt.year
    else:
        nfdb["year"] = nfdb["YEAR"]
    nfdb = nfdb[nfdb["year"].notna()].copy()
    nfdb["year"] = nfdb["year"].astype(int)
    nfdb_by_year = {int(y): g for y, g in nfdb.groupby("year")}
    print(f"  {len(nfdb):,} total NFDB fires")

    print(f"[consistency] Loading NBAC polygons from {args.nbac}...")
    nbac_shp = _extract_shp(Path(args.nbac))
    nbac = gpd.read_file(nbac_shp)
    nbac = nbac[nbac["YEAR"].notna()].copy()
    nbac["year"] = nbac["YEAR"].astype(int)
    nbac_by_year = {int(y): g for y, g in nbac.groupby("year")}
    print(f"  {len(nbac):,} total NBAC polygons")

    # --- Reference grid profile ---
    with rasterio.open(args.reference) as src:
        profile = src.profile
    H, W = int(profile["height"]), int(profile["width"])
    target_crs = profile["crs"]
    transform = profile["transform"]

    # Reproject NBAC to target CRS (stays in gpd for rasterize)
    if str(nbac.crs).upper() != str(target_crs).upper():
        nbac = nbac.to_crs(target_crs)
    nbac_by_year = {int(y): g for y, g in nbac.groupby("year")}

    # --- Per-year stats ---
    r = args.dilate_radius
    stats = []
    for yr in range(args.start, args.end + 1):
        t0 = time.time()
        print(f"[consistency] Year {yr}...", flush=True)

        # CWFIS mask (dilated)
        cwg = cwfis_by_year.get(yr, pd.DataFrame(columns=["latitude", "longitude"]))
        cw_mask = _rasterize_points(cwg["latitude"].values, cwg["longitude"].values,
                                    "EPSG:4326", profile)
        cw_mask = _dilate(cw_mask, r)

        # NFDB mask (dilated)
        nfg = nfdb_by_year.get(yr, None)
        if nfg is not None and len(nfg) > 0:
            lat = nfg.geometry.y.values
            lon = nfg.geometry.x.values
            nf_mask = _rasterize_points(lat, lon, "EPSG:4326", profile)
        else:
            nf_mask = np.zeros((H, W), dtype=np.uint8)
        nf_mask = _dilate(nf_mask, r)

        # NBAC mask (polygon — no dilation needed)
        nbg = nbac_by_year.get(yr, None)
        if nbg is not None and len(nbg) > 0:
            shapes = [(g, 1) for g in nbg.geometry if g is not None and not g.is_empty]
            if shapes:
                nb_mask = rio_rasterize(
                    shapes, out_shape=(H, W), transform=transform,
                    fill=0, all_touched=True, dtype="uint8")
            else:
                nb_mask = np.zeros((H, W), dtype=np.uint8)
        else:
            nb_mask = np.zeros((H, W), dtype=np.uint8)

        # Venn counts
        cw, nf, nb = cw_mask.astype(bool), nf_mask.astype(bool), nb_mask.astype(bool)
        any_src = cw | nf | nb
        all3 = cw & nf & nb
        cw_nf = cw & nf & ~nb
        cw_nb = cw & nb & ~nf
        nf_nb = nf & nb & ~cw
        cw_only = cw & ~nf & ~nb
        nf_only = nf & ~cw & ~nb
        nb_only = nb & ~cw & ~nf

        row = {
            "year": yr,
            "cwfis_total":   int(cw.sum()),
            "nfdb_total":    int(nf.sum()),
            "nbac_total":    int(nb.sum()),
            "any_source":    int(any_src.sum()),
            "all_three":     int(all3.sum()),
            "cwfis_nfdb":    int(cw_nf.sum()),
            "cwfis_nbac":    int(cw_nb.sum()),
            "nfdb_nbac":     int(nf_nb.sum()),
            "cwfis_only":    int(cw_only.sum()),
            "nfdb_only":     int(nf_only.sum()),
            "nbac_only":     int(nb_only.sum()),
        }
        # Fractions
        tot = max(row["any_source"], 1)
        row["frac_all_three"] = round(row["all_three"] / tot, 3)
        row["frac_cwfis_only"] = round(row["cwfis_only"] / tot, 3)
        row["frac_nfdb_only"] = round(row["nfdb_only"] / tot, 3)
        row["frac_nbac_only"] = round(row["nbac_only"] / tot, 3)
        row["frac_any_two"] = round(
            (row["cwfis_nfdb"] + row["cwfis_nbac"] + row["nfdb_nbac"]) / tot, 3)

        dt = time.time() - t0
        print(f"    {dt:.1f}s  any={row['any_source']:>9,}  "
              f"all3={row['frac_all_three']:.3f}  "
              f"nbac_only={row['frac_nbac_only']:.3f}  "
              f"nfdb_only={row['frac_nfdb_only']:.3f}  "
              f"cwfis_only={row['frac_cwfis_only']:.3f}",
              flush=True)
        stats.append(row)

    df = pd.DataFrame(stats).set_index("year")

    # --- Report ---
    print("\n=== PER-YEAR SPATIAL CONSISTENCY (dilated r=14) ===")
    display_cols = ["any_source", "frac_all_three", "frac_any_two",
                    "frac_cwfis_only", "frac_nfdb_only", "frac_nbac_only"]
    print(df[display_cols].to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== HYPOTHESIS TEST ===")
    # Pre-VIIRS era: high nbac_only / nfdb_only (sources complement CWFIS)
    early = df.loc[args.start:2011]
    late = df.loc[2012:args.end]
    if len(early) > 0 and len(late) > 0:
        print(f"  Early era (MODIS only, {early.index.min()}-2011):")
        print(f"    mean frac_nbac_only:  {early['frac_nbac_only'].mean():.3f}")
        print(f"    mean frac_nfdb_only:  {early['frac_nfdb_only'].mean():.3f}")
        print(f"    mean frac_all_three:  {early['frac_all_three'].mean():.3f}")
        print(f"  Late era (MODIS+VIIRS, 2012-{late.index.max()}):")
        print(f"    mean frac_nbac_only:  {late['frac_nbac_only'].mean():.3f}")
        print(f"    mean frac_nfdb_only:  {late['frac_nfdb_only'].mean():.3f}")
        print(f"    mean frac_all_three:  {late['frac_all_three'].mean():.3f}")

        nbac_delta = early['frac_nbac_only'].mean() - late['frac_nbac_only'].mean()
        all3_delta = late['frac_all_three'].mean() - early['frac_all_three'].mean()

        print(f"\n  Complementarity drop early→late (NBAC-only):  {nbac_delta:+.3f}")
        print(f"  Agreement increase early→late (all_three):    {all3_delta:+.3f}")

        if nbac_delta > 0.05:
            print(f"\n  ✓ HYPOTHESIS CONFIRMED: NBAC complements CWFIS more in "
                  f"early era, sources converge in VIIRS era.")
            print(f"  → Label fusion most beneficial for 2000-2011 training years.")
        else:
            print(f"\n  Hypothesis NOT strongly confirmed — label fusion may be "
                  f"less impactful than expected.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"\n[consistency] saved to {args.out}")


if __name__ == "__main__":
    main()
