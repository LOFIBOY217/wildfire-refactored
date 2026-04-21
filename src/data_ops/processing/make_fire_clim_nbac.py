#!/usr/bin/env python3
"""
Build NBAC+NFDB-based fire climatology (fire_clim_nbac_upto_Y.tif).

Drop-in replacement for the CWFIS-based `fire_clim_upto_Y.tif` files
(made by make_fire_clim_annual.py). Addresses CWFIS drift documented
in docs/LABEL_DECISION_2026_04_21.md:

    CWFIS fire_clim over-emphasises 2012+ (VIIRS era) because the
    raw hotspot density is 200-300x higher there than in 2000-2011.
    NBAC+NFDB is temporally stable (Landsat + human reports back
    to 1972/1946), producing a fire_clim that truthfully reflects
    real historical burn density.

For each target year Y, builds the map from NBAC polygons with YEAR < Y
(fire perimeters) + NFDB points with REP_DATE.year < Y (ignitions),
filtered to fire-season months (5-10 by default), log1p-normalized.

No dilation — fire_clim is a feature channel representing "historical
density"; dilation would blur real signal.

Output:
    {output_dir}/fire_clim_nbac_upto_{Y}.tif  for Y in [start, end]

Usage:
    python -m src.data_ops.processing.make_fire_clim_nbac \\
        --start_year 2000 --end_year 2025 \\
        --output_dir data/fire_clim_annual_nbac
"""

import argparse
import glob
import sys
import zipfile
from pathlib import Path

import numpy as np
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


def load_nbac(nbac_path):
    import geopandas as gpd
    p = Path(nbac_path)
    shp = _extract_shp(p) if p.suffix.lower() == ".zip" else p
    gdf = gpd.read_file(shp)
    for col in ("HS_SDATE", "HS_EDATE", "AG_SDATE", "AG_EDATE"):
        if col in gdf.columns:
            gdf[col] = pd.to_datetime(gdf[col], errors="coerce")
    gdf["year"] = gdf["YEAR"].astype(int)
    return gdf


def load_nfdb(nfdb_path):
    import geopandas as gpd
    p = Path(nfdb_path)
    shp = _extract_shp(p) if p.suffix.lower() == ".zip" else p
    gdf = gpd.read_file(shp)
    if "REP_DATE" in gdf.columns:
        gdf["date"] = pd.to_datetime(gdf["REP_DATE"], errors="coerce")
        gdf["year"] = gdf["date"].dt.year
    else:
        gdf["year"] = gdf["YEAR"].astype(int)
    gdf = gdf[gdf["year"].notna()].copy()
    gdf["year"] = gdf["year"].astype(int)
    return gdf


def build_fire_clim_year(nbac_gdf, nfdb_gdf, target_year, months,
                         profile, log_transform=True,
                         nbac_date_source="AG",
                         exclude_prescribed=True,
                         nfdb_min_size_ha=1.0):
    """
    Build fire_clim map "upto_{target_year}" using years < target_year only.

    NBAC: accumulate polygon FOOTPRINT over relevant years
         (each polygon fills its polygon area × number of season-months it was active)
    NFDB: accumulate ignition POINT (each point contributes 1 to its pixel)

    Sum the two sources, log1p-normalize.
    """
    H, W = int(profile["height"]), int(profile["width"])
    transform = profile["transform"]
    target_crs = profile["crs"]

    count_map = np.zeros((H, W), dtype=np.float32)

    # ---- NBAC contribution ----
    nbac = nbac_gdf[nbac_gdf["year"] < target_year].copy()
    if exclude_prescribed and "PRESCRIBED" in nbac.columns:
        _before = len(nbac)
        nbac = nbac[nbac["PRESCRIBED"].isna()]
        _dropped = _before - len(nbac)
        if _dropped:
            print(f"    [NBAC] dropped {_dropped} prescribed polygons")

    # Filter to fire-season: polygon must have AG_SDATE/EDATE overlap with
    # any month in `months`
    sdate_col = f"{nbac_date_source}_SDATE"
    edate_col = f"{nbac_date_source}_EDATE"
    if sdate_col in nbac.columns and edate_col in nbac.columns:
        def _has_season_overlap(row):
            sd, ed = row[sdate_col], row[edate_col]
            if pd.isna(sd) or pd.isna(ed):
                return False
            # any month in our fire season covered?
            cur = sd
            while cur <= ed:
                if cur.month in months:
                    return True
                cur += pd.Timedelta(days=1)
                if (cur - sd).days > 400:
                    break
            return False
        nbac = nbac[nbac.apply(_has_season_overlap, axis=1)]

    # Reproject + rasterize each polygon as weight = #seasonal days active
    if str(nbac.crs).upper() != str(target_crs).upper():
        nbac = nbac.to_crs(target_crs)

    print(f"    [NBAC] {len(nbac):,} polygons overlap fire-season")
    for _, row in nbac.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Weight = 1 per polygon (binary "burned in that year")
        try:
            m = rio_rasterize([(geom, 1)], out_shape=(H, W),
                              transform=transform, fill=0,
                              all_touched=True, dtype="uint8")
            count_map += m.astype(np.float32)
        except Exception:
            continue

    nbac_sum = float(count_map.sum())

    # ---- NFDB contribution ----
    nfdb = nfdb_gdf[nfdb_gdf["year"] < target_year].copy()
    if "SIZE_HA" in nfdb.columns:
        nfdb = nfdb[nfdb["SIZE_HA"].fillna(0) >= nfdb_min_size_ha]
    if exclude_prescribed and "CAUSE" in nfdb.columns:
        nfdb = nfdb[nfdb["CAUSE"].isin({"H", "N", "U"})]
    if "date" in nfdb.columns:
        nfdb = nfdb[nfdb["date"].dt.month.isin(months)]

    if str(nfdb.crs).upper() != str(target_crs).upper():
        nfdb = nfdb.to_crs(target_crs)

    print(f"    [NFDB] {len(nfdb):,} ignition points (size>={nfdb_min_size_ha}ha, "
          f"in fire season, excl prescribed)")

    if len(nfdb) > 0:
        # Rasterize points as 1 per ignition
        try:
            from rasterio.transform import rowcol
            xs = nfdb.geometry.x.values
            ys = nfdb.geometry.y.values
            rows, cols = rowcol(transform, xs, ys)
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            for r, c in zip(rows[valid], cols[valid]):
                count_map[r, c] += 1
        except Exception as e:
            print(f"    [NFDB] rasterize failed: {e}")

    total_sum = float(count_map.sum())
    print(f"    pre-log1p: NBAC contribution = {nbac_sum:.0f}, "
          f"NFDB added = {total_sum - nbac_sum:.0f}")

    if log_transform:
        count_map = np.log1p(count_map)
    return count_map


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--months", type=str, default="5-10")
    parser.add_argument("--nbac_path", default="data/burn_scars_raw/NBAC_1972to2024_shp.zip")
    parser.add_argument("--nfdb_path", default="data/nfdb/NFDB_point.zip")
    parser.add_argument("--nbac_date_source", default="AG", choices=("AG", "HS"))
    parser.add_argument("--nfdb_min_size_ha", type=float, default=1.0)
    parser.add_argument("--include_prescribed", action="store_true")
    parser.add_argument("--reference", default=None,
                        help="Reference FWI TIF for grid")
    parser.add_argument("--output_dir", default="data/fire_clim_annual_nbac")
    parser.add_argument("--no_log_transform", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    fwi_dir = get_path(cfg, "fwi_dir")
    ref_tif = args.reference
    if ref_tif is None:
        candidates = sorted(glob.glob(str(Path(fwi_dir) / "fwi_*.tif")))
        ref_tif = candidates[0]
        print(f"Reference grid: {ref_tif}")
    with rasterio.open(ref_tif) as src:
        profile = src.profile
    H, W = int(profile["height"]), int(profile["width"])

    # Parse months
    if "-" in args.months:
        lo, hi = [int(x) for x in args.months.split("-")]
        months = list(range(lo, hi + 1))
    else:
        months = [int(x) for x in args.months.split(",")]
    print(f"Fire-season months: {months}")

    # Load sources once
    print(f"\nLoading NBAC: {args.nbac_path}")
    nbac_gdf = load_nbac(args.nbac_path)
    print(f"  {len(nbac_gdf):,} polygons, years {int(nbac_gdf.year.min())}..{int(nbac_gdf.year.max())}")

    print(f"\nLoading NFDB: {args.nfdb_path}")
    nfdb_gdf = load_nfdb(args.nfdb_path)
    print(f"  {len(nfdb_gdf):,} fires, years {int(nfdb_gdf.year.min())}..{int(nfdb_gdf.year.max())}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="lzw",
                       nodata=None)
    out_profile.pop("photometric", None)

    # Build per target year
    import time
    for Y in range(args.start_year, args.end_year + 1):
        t0 = time.time()
        # Use same base filename as legacy CWFIS files (fire_clim_upto_Y.tif).
        # The output directory is what identifies this as NBAC-based, so
        # downstream train_v3.py / benchmark_baselines.py can point
        # --fire_clim_dir at data/fire_clim_annual_nbac/ and transparently
        # pick up NBAC-source climatology files.
        out_path = out_dir / f"fire_clim_upto_{Y}.tif"
        if out_path.exists():
            print(f"\n[SKIP] {out_path} exists")
            continue
        print(f"\n[upto_{Y}] Building from years < {Y} ...")
        clim = build_fire_clim_year(
            nbac_gdf, nfdb_gdf, target_year=Y, months=months,
            profile=profile, log_transform=not args.no_log_transform,
            nbac_date_source=args.nbac_date_source,
            exclude_prescribed=not args.include_prescribed,
            nfdb_min_size_ha=args.nfdb_min_size_ha)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(clim.astype(np.float32), 1)
        nz = int((clim > 0).sum())
        print(f"    → {out_path}  "
              f"nonzero={nz:,}  max={float(clim.max()):.3f}  "
              f"mean(nz)={float(clim[clim>0].mean()) if nz>0 else 0:.3f}  "
              f"({time.time()-t0:.0f}s)")

    print("\nAll years built.")


if __name__ == "__main__":
    main()
