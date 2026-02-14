#!/usr/bin/env python3
"""
Wildfire dataset downloader
============================
Downloads NFDB point + large fires, FWI (live, batch, archive), NBAC 30m,
and FBP fuel types from the Canadian Forest Service.

Replaces Makefile targets:
- NFDB point + large fires
- FWI (live, batch, archive)
- NBAC 30m
- FBP fuel types

Usage:
    python -m src.data_ops.download.wildfire_data nfdb
    python -m src.data_ops.download.wildfire_data fwi-live 20250906
    python -m src.data_ops.download.wildfire_data fwi-batch 20250904 20250905 20250906
    python -m src.data_ops.download.wildfire_data fwi-archive 20250906
    python -m src.data_ops.download.wildfire_data nbac
    python -m src.data_ops.download.wildfire_data fbp
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download(url: str, out: Path, fail_ok: bool = False):
    try:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, out)
        print(f"Saved to {out}")
    except Exception as e:
        if fail_ok:
            print(f"Skipped {out.name} ({e})")
        else:
            raise


def unzip(zip_path: Path, dest: Path):
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


# ------------------------------------------------------------------ #
# NFDB
# ------------------------------------------------------------------ #

POINT_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point.zip"
LARGE_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_large_fires.zip"


def fetch_nfdb(base_dir: Path):
    point_dir = base_dir / "NFDB_point"
    large_dir = base_dir / "NFDB_point_large_fires"

    point_zip = point_dir / "NFDB_point.zip"
    large_zip = large_dir / "NFDB_point_large_fires.zip"

    mkdir(point_dir)
    mkdir(large_dir)

    download(POINT_URL, point_zip)
    unzip(point_zip, point_dir)

    download(LARGE_URL, large_zip)
    unzip(large_zip, large_dir)


def clean_nfdb(base_dir: Path):
    point_zip = base_dir / "NFDB_point" / "NFDB_point.zip"
    large_zip = base_dir / "NFDB_point_large_fires" / "NFDB_point_large_fires.zip"
    point_zip.unlink(missing_ok=True)
    large_zip.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# FWI (LIVE)
# ------------------------------------------------------------------ #

FWI_BASE_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/cffdrs"


def fetch_fwi_live(fwi_dir: Path, date: str):
    mkdir(fwi_dir)
    fname = f"fwi_scribe_{date}.tif"
    out = fwi_dir / fname
    download(f"{FWI_BASE_URL}/{fname}", out)


def fetch_fwi_batch(fwi_dir: Path, dates):
    mkdir(fwi_dir)
    for d in dates:
        fname = f"fwi_scribe_{d}.tif"
        out = fwi_dir / fname
        download(f"{FWI_BASE_URL}/{fname}", out, fail_ok=True)


# ------------------------------------------------------------------ #
# FWI (ARCHIVE via WCS)
# ------------------------------------------------------------------ #

BASE_WCS = (
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs"
    "?service=WCS&version=1.0.0&request=GetCoverage"
)

BBOX = "-2378164,-707617,3039835,3854382"
WIDTH = 2709
HEIGHT = 2281
CRS = "EPSG:3978"
FORMAT = "geotiff"


def fetch_fwi_archive(fwi_dir: Path, date: str):
    mkdir(fwi_dir)
    out = fwi_dir / f"fwi_{date}.tif"

    url = (
        f"{BASE_WCS}"
        f"&coverage=public:fwi{date}"
        f"&BBOX={BBOX}"
        f"&WIDTH={WIDTH}"
        f"&HEIGHT={HEIGHT}"
        f"&CRS={CRS}"
        f"&FORMAT={FORMAT}"
    )

    download(url, out)


# ------------------------------------------------------------------ #
# NBAC 30m
# ------------------------------------------------------------------ #

NBAC_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/NBAC_MRB_1972to2024_30m.tif.zip"


def fetch_nbac_30m(nbac_dir: Path):
    mkdir(nbac_dir)
    nbac_zip = nbac_dir / "NBAC_MRB_1972to2024_30m.tif.zip"
    download(NBAC_URL, nbac_zip)
    unzip(nbac_zip, nbac_dir)


def clean_nbac(nbac_dir: Path):
    nbac_zip = nbac_dir / "NBAC_MRB_1972to2024_30m.tif.zip"
    nbac_zip.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# FBP Fuel Types
# ------------------------------------------------------------------ #

FBP_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/current/FBP_fueltypes_Canada_100m_EPSG3978_20240527.tif"


def fetch_fbp_fueltypes(fbp_dir: Path):
    mkdir(fbp_dir)
    fbp_file = fbp_dir / "FBP_fueltypes_Canada_100m_EPSG3978_20240527.tif"
    download(FBP_URL, fbp_file)


def clean_fbp(fbp_dir: Path):
    fbp_file = fbp_dir / "FBP_fueltypes_Canada_100m_EPSG3978_20240527.tif"
    fbp_file.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Download wildfire-related datasets (NFDB, FWI, NBAC, FBP)"
    )
    add_config_argument(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("nfdb", help="Download NFDB point + large fire shapefiles")
    sub.add_parser("nfdb-clean", help="Remove NFDB zip files")

    p_live = sub.add_parser("fwi-live", help="Download a single FWI live raster")
    p_live.add_argument("date", help="Date string, e.g. 20250906")

    p_batch = sub.add_parser("fwi-batch", help="Download multiple FWI live rasters")
    p_batch.add_argument("dates", nargs="+", help="Date strings, e.g. 20250904 20250905")

    p_arch = sub.add_parser("fwi-archive", help="Download FWI archive via WCS")
    p_arch.add_argument("date", help="Date string, e.g. 20250906")

    sub.add_parser("nbac", help="Download NBAC 30m fire perimeters")
    sub.add_parser("nbac-clean", help="Remove NBAC zip file")
    sub.add_parser("fbp", help="Download FBP fuel type raster")
    sub.add_parser("fbp-clean", help="Remove FBP raster file")

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Resolve paths from config (with fallback defaults)
    raw_base = Path(get_path(cfg, "fwi_dir")).parent  # data/
    nfdb_dir = raw_base / "nfdb"
    fwi_dir = Path(get_path(cfg, "fwi_dir"))
    nbac_dir = raw_base / "nbac" / "NBAC_MRB_1972to2024_30m"
    fbp_dir = raw_base / "fuel" / "FBP_fueltypes_100m"

    cmd = args.command

    if cmd == "nfdb":
        fetch_nfdb(nfdb_dir)
    elif cmd == "nfdb-clean":
        clean_nfdb(nfdb_dir)
    elif cmd == "fwi-live":
        fetch_fwi_live(fwi_dir, args.date)
    elif cmd == "fwi-batch":
        fetch_fwi_batch(fwi_dir, args.dates)
    elif cmd == "fwi-archive":
        fetch_fwi_archive(fwi_dir, args.date)
    elif cmd == "nbac":
        fetch_nbac_30m(nbac_dir)
    elif cmd == "nbac-clean":
        clean_nbac(nbac_dir)
    elif cmd == "fbp":
        fetch_fbp_fueltypes(fbp_dir)
    elif cmd == "fbp-clean":
        clean_fbp(fbp_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
