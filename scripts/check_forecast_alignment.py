#!/usr/bin/env python3
"""
Check spatial alignment between fire_prob_lead forecast GeoTIFFs and FWI reference grid.
=========================================================================================
Verifies that every fire_prob_lead*.tif produced by train_logistic.py shares the
same CRS, resolution, dimensions, and spatial extent as the FWI source files.

Checks performed per forecast file:
    1. CRS matches reference
    2. Pixel size (res_x, res_y) matches reference (within tolerance)
    3. Grid dimensions (height, width) match reference
    4. Bounding box matches reference (within tolerance)
    5. Origin (top-left corner) matches reference (within tolerance)
    6. dtype is float32
    7. No all-NaN or all-zero bands

Usage:
    # Auto-detect paths from default config
    python scripts/check_forecast_alignment.py

    # Explicit paths
    python scripts/check_forecast_alignment.py \\
        --fwi_dir   data/fwi_data \\
        --prob_dir  outputs/logreg_fire_prob_7day_forecast

    # With a different config (e.g. Windows paths)
    python scripts/check_forecast_alignment.py --config configs/paths_windows.yaml

    # Limit to one forecast date folder
    python scripts/check_forecast_alignment.py --date 20250801

    # Save report to file
    python scripts/check_forecast_alignment.py --output_csv forecast_alignment_report.csv

Output:
    Prints a per-file pass/fail table and a summary.
    Optionally writes a CSV report for post-processing on HPC.
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Resolve src on the path so the script works from any working directory
# ---------------------------------------------------------------------------
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "config.py").exists():
        sys.path.insert(0, str(_parent))
        break

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    # Graceful fallback: config loading is optional if paths are given explicitly.
    load_config = None
    get_path = None
    add_config_argument = None


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
RES_TOL_M   = 1.0      # metres  – pixel size tolerance
ORIGIN_TOL  = 1.0      # metres  – top-left corner tolerance
BOUNDS_TOL  = 10.0     # metres  – bounding-box corner tolerance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_from(path):
    with rasterio.open(path) as src:
        return {
            "crs":    src.crs.to_epsg() if src.crs else None,
            "crs_wkt": src.crs.to_wkt() if src.crs else "",
            "height": src.height,
            "width":  src.width,
            "res_x":  abs(src.transform.a),
            "res_y":  abs(src.transform.e),
            "origin_x": src.transform.c,
            "origin_y": src.transform.f,
            "left":   src.bounds.left,
            "bottom": src.bounds.bottom,
            "right":  src.bounds.right,
            "top":    src.bounds.top,
            "dtype":  src.dtypes[0],
            "nodata": src.nodata,
            "count":  src.count,
        }


def _read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def check_file(prob_path, ref):
    """
    Compare a single forecast file against the reference profile.

    Returns:
        (passed: bool, issues: list[str], stats: dict)
    """
    issues = []
    stats  = {}

    try:
        p = _profile_from(prob_path)
    except Exception as exc:
        return False, [f"Cannot open file: {exc}"], {}

    # 1. CRS
    if p["crs"] != ref["crs"]:
        issues.append(
            f"CRS mismatch: forecast EPSG:{p['crs']}  ref EPSG:{ref['crs']}"
        )

    # 2. Pixel size
    dx = abs(p["res_x"] - ref["res_x"])
    dy = abs(p["res_y"] - ref["res_y"])
    if dx > RES_TOL_M or dy > RES_TOL_M:
        issues.append(
            f"Pixel size mismatch: forecast ({p['res_x']:.2f}, {p['res_y']:.2f})  "
            f"ref ({ref['res_x']:.2f}, {ref['res_y']:.2f})"
        )

    # 3. Dimensions
    if p["height"] != ref["height"] or p["width"] != ref["width"]:
        issues.append(
            f"Dimension mismatch: forecast {p['height']}×{p['width']}  "
            f"ref {ref['height']}×{ref['width']}"
        )

    # 4. Bounding box
    for corner, pv, rv in [
        ("left",   p["left"],   ref["left"]),
        ("bottom", p["bottom"], ref["bottom"]),
        ("right",  p["right"],  ref["right"]),
        ("top",    p["top"],    ref["top"]),
    ]:
        if abs(pv - rv) > BOUNDS_TOL:
            issues.append(
                f"Bounds[{corner}] mismatch: forecast {pv:.1f}  ref {rv:.1f}"
            )

    # 5. Origin (top-left corner of transform)
    if abs(p["origin_x"] - ref["origin_x"]) > ORIGIN_TOL or \
       abs(p["origin_y"] - ref["origin_y"]) > ORIGIN_TOL:
        issues.append(
            f"Origin mismatch: forecast ({p['origin_x']:.1f}, {p['origin_y']:.1f})  "
            f"ref ({ref['origin_x']:.1f}, {ref['origin_y']:.1f})"
        )

    # 6. dtype
    if p["dtype"] != "float32":
        issues.append(f"dtype is {p['dtype']}, expected float32")

    # 7. Band content sanity check  (only if dimensions match, so we can read)
    if not any("Dimension" in i or "Cannot open" in i for i in issues):
        try:
            arr = _read_band(prob_path)
            finite = arr[np.isfinite(arr)]
            stats["n_pixels"]  = int(arr.size)
            stats["n_nan"]     = int(np.sum(~np.isfinite(arr)))
            stats["n_zero"]    = int(np.sum(arr == 0))
            stats["min"]       = float(np.nanmin(arr)) if finite.size else float("nan")
            stats["max"]       = float(np.nanmax(arr)) if finite.size else float("nan")
            stats["mean"]      = float(np.nanmean(arr)) if finite.size else float("nan")

            if finite.size == 0:
                issues.append("Band is entirely NaN/nodata")
            elif np.all(arr == 0):
                issues.append("Band is all zeros (suspicious)")
        except Exception as exc:
            issues.append(f"Could not read band data: {exc}")

    passed = len(issues) == 0
    return passed, issues, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check spatial alignment of fire_prob_lead GeoTIFFs vs FWI reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    if add_config_argument:
        add_config_argument(parser)
    else:
        parser.add_argument("--config", default=None, help="Path to YAML config")

    parser.add_argument(
        "--fwi_dir", default=None,
        help="Directory containing FWI reference *.tif files (overrides config)",
    )
    parser.add_argument(
        "--prob_dir", default=None,
        help="Root directory of forecast output, i.e. outputs/logreg_fire_prob_7day_forecast "
             "(overrides config)",
    )
    parser.add_argument(
        "--date", default=None,
        help="Only check one forecast date folder, e.g. 20250801",
    )
    parser.add_argument(
        "--output_csv", default=None,
        help="Optional path to write a CSV report (useful on HPC)",
    )
    parser.add_argument(
        "--max_files", type=int, default=0,
        help="Limit total files checked (0 = unlimited, handy for quick smoke-tests)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve directories
    # -----------------------------------------------------------------------
    fwi_dir  = args.fwi_dir
    prob_dir = args.prob_dir

    if (fwi_dir is None or prob_dir is None) and load_config is not None:
        cfg = load_config(args.config)
        if fwi_dir is None:
            fwi_dir = get_path(cfg, "fwi_dir")
        if prob_dir is None:
            prob_dir = os.path.join(
                get_path(cfg, "output_dir"),
                "logreg_fire_prob_7day_forecast",
            )

    if not fwi_dir or not os.path.isdir(fwi_dir):
        print(f"ERROR: fwi_dir not found or not a directory: {fwi_dir}")
        print("       Pass --fwi_dir explicitly or check your config file.")
        sys.exit(1)

    if not prob_dir or not os.path.isdir(prob_dir):
        print(f"ERROR: prob_dir not found or not a directory: {prob_dir}")
        print("       Pass --prob_dir explicitly or check your config file.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Pick reference FWI file (first chronologically)
    # -----------------------------------------------------------------------
    fwi_files = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
    if not fwi_files:
        print(f"ERROR: No *.tif files found in fwi_dir: {fwi_dir}")
        sys.exit(1)

    ref_path = fwi_files[0]
    ref      = _profile_from(ref_path)

    print()
    print("=" * 72)
    print("FORECAST ALIGNMENT CHECKER")
    print("=" * 72)
    print(f"  Reference FWI file : {ref_path}")
    print(f"  Grid (H × W)       : {ref['height']} × {ref['width']}")
    print(f"  Pixel size         : {ref['res_x']:.2f} m × {ref['res_y']:.2f} m")
    print(f"  CRS                : EPSG:{ref['crs']}")
    print(f"  Bounds (L,B,R,T)   : {ref['left']:.0f}, {ref['bottom']:.0f}, "
          f"{ref['right']:.0f}, {ref['top']:.0f}")
    print(f"  FWI dir            : {fwi_dir}  ({len(fwi_files)} files)")
    print(f"  Forecast dir       : {prob_dir}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Collect forecast files
    # -----------------------------------------------------------------------
    if args.date:
        pattern = os.path.join(prob_dir, args.date, "fire_prob_lead*.tif")
    else:
        pattern = os.path.join(prob_dir, "**", "fire_prob_lead*.tif")

    prob_files = sorted(glob.glob(pattern, recursive=True))

    if not prob_files:
        print(f"\nNo fire_prob_lead*.tif files found under: {prob_dir}")
        print("  (Have you run train_logistic.py yet?)")
        sys.exit(0)

    if args.max_files and len(prob_files) > args.max_files:
        print(f"  Limiting to first {args.max_files} of {len(prob_files)} files (--max_files)")
        prob_files = prob_files[: args.max_files]

    print(f"\nChecking {len(prob_files)} forecast file(s)...\n")

    # -----------------------------------------------------------------------
    # Check each file
    # -----------------------------------------------------------------------
    rows        = []   # for CSV
    n_pass      = 0
    n_fail      = 0
    fail_detail = []

    COL_W = 60   # display column width for filename

    for i, fpath in enumerate(prob_files):
        rel = os.path.relpath(fpath, prob_dir)
        passed, issues, stats = check_file(fpath, ref)

        status = "PASS" if passed else "FAIL"
        pad    = max(1, COL_W - len(rel))
        print(f"  [{status}]  {rel}{' ' * pad}", end="")

        if stats:
            print(
                f"  min={stats.get('min', float('nan')):5.3f}  "
                f"max={stats.get('max', float('nan')):5.3f}  "
                f"nan={stats.get('n_nan', '?'):>8,}"
            )
        else:
            print()

        if not passed:
            n_fail += 1
            fail_detail.append((rel, issues))
            for iss in issues:
                print(f"           !! {iss}")
        else:
            n_pass += 1

        rows.append({
            "file":    rel,
            "status":  status,
            "issues":  " | ".join(issues),
            **{k: str(v) for k, v in stats.items()},
        })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"  Total checked : {len(prob_files)}")
    print(f"  PASS          : {n_pass}")
    print(f"  FAIL          : {n_fail}")
    print("=" * 72)

    if fail_detail:
        print(f"\nFailed files ({len(fail_detail)}):")
        for rel, issues in fail_detail:
            print(f"  {rel}")
            for iss in issues:
                print(f"      -> {iss}")

    # -----------------------------------------------------------------------
    # Optional CSV report
    # -----------------------------------------------------------------------
    if args.output_csv:
        fieldnames = ["file", "status", "issues",
                      "n_pixels", "n_nan", "n_zero", "min", "max", "mean"]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nReport saved: {args.output_csv}")

    print()
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
