#!/usr/bin/env python3
"""
check_s2s_data.py — Diagnose ECMWF S2S hindcast data availability.

Checks whether the S2S data needed for the decoder of S2SHotspotTransformer
has been downloaded.  Reports:
  - Where it looked
  - What GRIB/NC files were found
  - Date coverage vs training range (2018-05-01 – 2024-10-31)
  - What variables are inside each file (if cfgrib / netCDF4 available)
  - A clear "YES / NO" answer on whether data is ready

Usage:
    python -m src.data_ops.check_s2s_data
    python -m src.data_ops.check_s2s_data --config configs/paths_windows.yaml
    python -m src.data_ops.check_s2s_data --config configs/paths_windows.yaml \\
        --extra_dirs "D:/data/s2s" "E:/ecmwf"
"""

import argparse
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# ------------------------------------------------------------------ #
# Config loading (same pattern as all project scripts)
# ------------------------------------------------------------------ #
try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

# Fire-season training range used by train_s2s_hotspot_cwfis_v2.py
TRAIN_START = date(2018, 5, 1)
TRAIN_END   = date(2024, 10, 31)

# Variables that s2s_ecmwf.py downloads (ECMWF param codes → human names)
S2S_PARAMS = {
    "136":    "tcw  (Total Column Water)",
    "167":    "2t   (2m Temperature)",
    "168":    "2d   (2m Dewpoint)",
    "228086": "sm20 (20cm Soil Moisture)",
    "228095": "st20 (20cm Soil Temperature)",
}

# Filename pattern produced by s2s_ecmwf.py
S2S_GRIB_RE = re.compile(
    r"s2s_ecmf_cf_(\d{4}-\d{2}-\d{2})(?:_to_(\d{4}-\d{2}-\d{2}))?\.grib$",
    re.IGNORECASE,
)

# Candidate sub-directory names (relative to project root or data root)
S2S_CANDIDATE_SUBDIRS = [
    "data/ecmwf_s2s",
    "data/ecmwf_hindcast",
    "data/s2s",
    "data/ecmwf_observation",   # may have been used as output dir
    "data/ecmwf",
]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1e9:
        return f"{n_bytes/1e9:.2f} GB"
    elif n_bytes >= 1e6:
        return f"{n_bytes/1e6:.1f} MB"
    else:
        return f"{n_bytes/1e3:.0f} KB"


def _extract_date_from_filename(fname: str):
    """Return (start_date, end_date) parsed from a s2s_ecmf_cf_*.grib filename."""
    m = S2S_GRIB_RE.search(fname)
    if not m:
        return None, None
    try:
        d1 = date.fromisoformat(m.group(1))
        d2 = date.fromisoformat(m.group(2)) if m.group(2) else d1
        return d1, d2
    except ValueError:
        return None, None


def _inspect_grib(path: Path) -> dict:
    """
    Try to read a GRIB file and extract variable / date metadata.
    Returns a dict with keys: variables, forecast_dates, error.
    Requires cfgrib or eccodes; gracefully degrades if not installed.
    """
    result = {"variables": [], "forecast_dates": [], "error": None}
    try:
        import cfgrib
        import xarray as xr
        ds_list = cfgrib.open_datasets(str(path))
        for ds in ds_list:
            for var in ds.data_vars:
                result["variables"].append(var)
            if "valid_time" in ds.coords:
                times = ds.coords["valid_time"].values
                import numpy as np
                for t in times.flat:
                    try:
                        import pandas as pd
                        d = pd.Timestamp(t).date()
                        result["forecast_dates"].append(d)
                    except Exception:
                        pass
    except ImportError:
        result["error"] = "cfgrib not installed (pip install cfgrib)"
    except Exception as e:
        result["error"] = str(e)
    return result


def _inspect_nc(path: Path) -> dict:
    """Try to read a NetCDF file and extract basic metadata."""
    result = {"variables": [], "time_range": None, "error": None}
    try:
        import netCDF4 as nc4
        with nc4.Dataset(str(path)) as ds:
            result["variables"] = list(ds.variables.keys())
            if "time" in ds.variables:
                t = ds["time"][:]
                result["time_range"] = (float(t.min()), float(t.max()))
    except ImportError:
        result["error"] = "netCDF4 not installed"
    except Exception as e:
        result["error"] = str(e)
    return result


def _scan_directory(d: Path, verbose: bool = False) -> list[dict]:
    """
    Recursively scan a directory for .grib and .nc files.
    Returns list of file-info dicts.
    """
    if not d.exists():
        return []
    files = []
    for ext in ("*.grib", "*.grb", "*.grib2", "*.nc", "*.nc4"):
        for fp in sorted(d.rglob(ext)):
            info = {
                "path":      fp,
                "size":      fp.stat().st_size,
                "ext":       fp.suffix.lower(),
                "date_from": None,
                "date_to":   None,
            }
            # Try to extract date from filename
            d1, d2 = _extract_date_from_filename(fp.name)
            info["date_from"] = d1
            info["date_to"]   = d2

            # Optional deep inspection
            if verbose:
                if info["ext"] in (".grib", ".grb", ".grib2"):
                    info["inspect"] = _inspect_grib(fp)
                elif info["ext"] in (".nc", ".nc4"):
                    info["inspect"] = _inspect_nc(fp)

            files.append(info)
    return files


def _compute_coverage(file_infos: list[dict]) -> tuple[date | None, date | None, int]:
    """Return (earliest_date, latest_date, n_files_with_dates)."""
    dates = []
    for fi in file_infos:
        if fi["date_from"]:
            dates.append(fi["date_from"])
        if fi["date_to"]:
            dates.append(fi["date_to"])
    if not dates:
        return None, None, 0
    return min(dates), max(dates), len([f for f in file_infos if f["date_from"]])


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)
    parser.add_argument(
        "--extra_dirs", nargs="*", default=[],
        help="Additional directories to scan (absolute paths).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Open each GRIB/NC and report variables inside (requires cfgrib/netCDF4).",
    )
    args = parser.parse_args()
    cfg  = load_config(args.config)

    # Determine project root
    try:
        project_root = Path(get_path(cfg, "project_root"))
    except Exception:
        project_root = Path(__file__).resolve().parents[2]

    print("=" * 60)
    print("  ECMWF S2S Hindcast Data Check")
    print("=" * 60)
    print(f"  Project root : {project_root}")
    print(f"  Config       : {getattr(args, 'config', 'default')}")
    print(f"  Training range to check: {TRAIN_START} → {TRAIN_END}")
    print()

    # Build list of directories to scan
    candidate_dirs: list[Path] = []

    # 1. From config (if ecmwf_s2s_dir or similar exists)
    for key in ("ecmwf_s2s_dir", "s2s_dir", "hindcast_dir"):
        try:
            p = Path(get_path(cfg, key))
            candidate_dirs.append(p)
            print(f"  [config] Found '{key}' → {p}")
        except Exception:
            pass  # Key not in config — expected

    # 2. Well-known sub-directories relative to project root
    for sub in S2S_CANDIDATE_SUBDIRS:
        p = project_root / sub
        if p not in candidate_dirs:
            candidate_dirs.append(p)

    # 3. User-supplied extra dirs
    for extra in args.extra_dirs:
        candidate_dirs.append(Path(extra))

    # ─ Scan ──────────────────────────────────────────────────────────
    all_files: list[dict] = []
    print("  Scanning directories...")
    print()

    for d in candidate_dirs:
        print(f"  [{d}]")
        if not d.exists():
            print("    ✗ Directory does not exist")
            print()
            continue

        files = _scan_directory(d, verbose=args.verbose)
        if not files:
            print("    ✗ No GRIB / NC files found")
        else:
            total_size = sum(f["size"] for f in files)
            print(f"    ✓ {len(files)} file(s)  ({_fmt_size(total_size)} total)")
            for fi in files[:10]:            # show first 10
                tag = ""
                if fi["date_from"]:
                    tag = f"  [{fi['date_from']} → {fi['date_to'] or fi['date_from']}]"
                print(f"      {fi['path'].name}  "
                      f"({_fmt_size(fi['size'])}){tag}")
            if len(files) > 10:
                print(f"      … and {len(files)-10} more")

            # Verbose: show variables
            if args.verbose:
                for fi in files[:5]:
                    insp = fi.get("inspect", {})
                    if insp.get("variables"):
                        print(f"      Variables in {fi['path'].name}: "
                              f"{', '.join(insp['variables'][:8])}")
                    if insp.get("error"):
                        print(f"      [inspect error] {insp['error']}")

        all_files.extend(files)
        print()

    # ─ Coverage summary ──────────────────────────────────────────────
    earliest, latest, n_dated = _compute_coverage(all_files)
    grib_files = [f for f in all_files
                  if f["ext"] in (".grib", ".grb", ".grib2")]
    s2s_grib   = [f for f in grib_files if S2S_GRIB_RE.search(f["path"].name)]

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total GRIB/NC files found   : {len(all_files)}")
    print(f"  Files matching s2s_ecmf_cf_*: {len(s2s_grib)}")

    if earliest:
        print(f"  Date coverage (from filenames): {earliest} → {latest}")
        # Check vs training range
        covers_start = earliest <= TRAIN_START
        covers_end   = latest   >= TRAIN_END
        if covers_start and covers_end:
            print(f"  ✓ Coverage spans full training range "
                  f"({TRAIN_START} – {TRAIN_END})")
        else:
            print(f"  ✗ Coverage INCOMPLETE for training range "
                  f"({TRAIN_START} – {TRAIN_END})")
            if not covers_start:
                print(f"    Missing: {TRAIN_START} → {earliest - timedelta(days=1)}")
            if not covers_end:
                print(f"    Missing: {latest + timedelta(days=1)} → {TRAIN_END}")
    else:
        print("  ✗ No date information in filenames")

    print()
    # ─ Verdict ───────────────────────────────────────────────────────
    if s2s_grib and earliest and latest and earliest <= TRAIN_START and latest >= TRAIN_END:
        print("  ✅  S2S hindcast data is AVAILABLE and covers the training range.")
        print("      → Next step: integrate into decoder pipeline.")
    elif s2s_grib:
        print("  ⚠️  S2S data found but INCOMPLETE for training range.")
        print("      → Run s2s_ecmwf.py to fill the gaps.")
    else:
        print("  ❌  S2S hindcast data NOT FOUND.")
        print()
        print("  To download (requires ECMWF API key at ~/.ecmwfapirc):")
        print("    python -m src.data_ops.download.s2s_ecmwf \\")
        print(f"        {TRAIN_START} {TRAIN_END}")
        print()
        print("  The decoder currently uses ERA5 oracle (real future weather).")
        print("  Training can proceed as-is; switch to S2S after data is ready.")

    print()
    print("  Expected S2S variables (when downloaded):")
    for code, name in S2S_PARAMS.items():
        print(f"    param {code:>6} → {name}")
    print()


if __name__ == "__main__":
    main()
