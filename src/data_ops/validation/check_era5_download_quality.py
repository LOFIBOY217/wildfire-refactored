#!/usr/bin/env python3
"""
Validate downloaded ERA5 GRIB files for empty/zero/invalid content.

Checks per date:
1) file exists and is non-empty
2) GRIB can be decoded by cfgrib/xarray
3) key variables contain finite, non-zero values

Usage:
    python -m src.data_ops.validation.check_era5_download_quality \
        --start 2018-01-01 --end 2025-12-31

    python -m src.data_ops.validation.check_era5_download_quality \
        --config configs/default.yaml --output-csv era5_quality_report.csv
"""

import argparse
import csv
import os
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import cfgrib

# Silence cfgrib/xarray future-warning noise during validation output.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"cfgrib\..*")

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    import sys
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument


EXPECTED_VARS = ("t2m", "d2m", "tcw", "swvl1", "stl1")
DATE_RE = re.compile(r"era5_sfc_(\d{4})_(\d{2})_(\d{2})\.grib$")


def _parse_filename_date(path: Path):
    m = DATE_RE.search(path.name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _iter_dates(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def _check_var(arr):
    arr = np.asarray(arr, dtype=np.float64)
    size = arr.size
    finite_mask = np.isfinite(arr)
    finite_count = int(finite_mask.sum())
    if size == 0 or finite_count == 0:
        return {
            "size": size,
            "finite_count": finite_count,
            "nonzero_count": 0,
            "all_nan": 1,
            "all_zero": 1,
            "constant": 1,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    v = arr[finite_mask]
    nonzero_count = int((np.abs(v) > 1e-12).sum())
    return {
        "size": size,
        "finite_count": finite_count,
        "nonzero_count": nonzero_count,
        "all_nan": 0,
        "all_zero": int(nonzero_count == 0),
        "constant": int(float(np.nanstd(v)) == 0.0),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
    }


def _check_file(path: Path, expected_vars):
    result = {
        "date": _parse_filename_date(path) or "",
        "file": str(path),
        "exists": int(path.exists()),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "decode_ok": 0,
        "missing_vars": "",
        "has_empty_var": 0,
        "has_all_zero_var": 0,
        "has_constant_var": 0,
        "error": "",
    }

    if not path.exists():
        result["error"] = "missing_file"
        return result
    if result["bytes"] == 0:
        result["error"] = "empty_file"
        return result

    try:
        datasets = cfgrib.open_datasets(str(path))
        found = {}
        for ds in datasets:
            for v in ds.data_vars:
                if v not in found:
                    found[v] = ds[v].values
            ds.close()

        result["decode_ok"] = 1
        missing = [v for v in expected_vars if v not in found]
        result["missing_vars"] = ",".join(missing)

        for v in expected_vars:
            if v not in found:
                continue
            st = _check_var(found[v])
            if st["all_nan"] == 1:
                result["has_empty_var"] = 1
            if st["all_zero"] == 1:
                result["has_all_zero_var"] = 1
            if st["constant"] == 1:
                result["has_constant_var"] = 1

    except Exception as e:
        result["error"] = f"decode_error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check ERA5 GRIB download quality (empty/zero/invalid)."
    )
    add_config_argument(parser)
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--input-dir", default=None, help="Directory containing era5_sfc_*.grib")
    parser.add_argument("--output-csv", default=None, help="Optional CSV report path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(get_path(cfg, "era5_dir"))

    dates = list(_iter_dates(args.start, args.end))
    rows = []

    print(f"Checking {len(dates)} dates in: {input_dir}")
    for i, d in enumerate(dates, 1):
        fn = input_dir / f"era5_sfc_{d.replace('-', '_')}.grib"
        r = _check_file(fn, EXPECTED_VARS)
        r["date"] = d
        rows.append(r)
        if i % 50 == 0 or i == len(dates):
            print(f"  Progress: {i}/{len(dates)}")

    missing = sum(1 for r in rows if r["error"] == "missing_file")
    empty_files = sum(1 for r in rows if r["error"] == "empty_file")
    decode_fail = sum(1 for r in rows if r["decode_ok"] == 0 and r["error"] not in ("missing_file", "empty_file"))
    missing_vars = sum(1 for r in rows if r["decode_ok"] == 1 and r["missing_vars"])
    empty_vars = sum(1 for r in rows if r["has_empty_var"] == 1)
    zero_vars = sum(1 for r in rows if r["has_all_zero_var"] == 1)
    constant_vars = sum(1 for r in rows if r["has_constant_var"] == 1)

    print("\n" + "=" * 70)
    print("ERA5 DOWNLOAD QUALITY SUMMARY")
    print("=" * 70)
    print(f"Total dates:         {len(rows)}")
    print(f"Missing files:       {missing}")
    print(f"Empty files:         {empty_files}")
    print(f"Decode failures:     {decode_fail}")
    print(f"Missing key vars:    {missing_vars}")
    print(f"Has empty var:       {empty_vars}")
    print(f"Has all-zero var:    {zero_vars}")
    print(f"Has constant var:    {constant_vars}")
    print("=" * 70)

    bad = [
        r for r in rows
        if r["error"] or r["missing_vars"] or r["has_empty_var"] or r["has_all_zero_var"]
    ]
    if bad:
        print("\nSample problematic dates:")
        for r in bad[:20]:
            print(f"  {r['date']} | error={r['error']} | missing_vars={r['missing_vars']} | all_zero={r['has_all_zero_var']}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "date", "file", "exists", "bytes", "decode_ok", "missing_vars",
                    "has_empty_var", "has_all_zero_var", "has_constant_var", "error",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
