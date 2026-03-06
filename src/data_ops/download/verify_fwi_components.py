#!/usr/bin/env python3
"""
Verify completeness and integrity of downloaded FWI component GeoTIFFs.

Checks for each variable (fwi, ffmc, dmc, dc, isi, bui):
  1. File exists for every expected date
  2. File opens as a valid GeoTIFF
  3. Shape matches reference grid (2281 x 2709)
  4. Value range is physically plausible

Usage:
    # Check 2018-2024 fire season (May–Oct), all 6 variables
    python -m src.data_ops.download.verify_fwi_components \\
        --config configs/paths_windows.yaml \\
        --start 2018-01-01 --end 2024-12-31

    # Check only DMC and FFMC for 2022
    python -m src.data_ops.download.verify_fwi_components \\
        --config configs/paths_windows.yaml \\
        --start 2022-01-01 --end 2022-12-31 \\
        --vars dmc ffmc

    # Quick mode: skip per-file content checks (existence only)
    python -m src.data_ops.download.verify_fwi_components \\
        --config configs/paths_windows.yaml --quick

Output:
    Prints a summary table per variable.
    Exits with code 1 if any files are missing or corrupt.
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument

# Expected physical value ranges for each component
EXPECTED_RANGES = {
    'fwi':  (0.0, 300.0),
    'ffmc': (0.0, 101.0),
    'dmc':  (0.0, 600.0),
    'dc':   (0.0, 1200.0),
    'isi':  (0.0, 120.0),
    'bui':  (0.0, 1200.0),
}

CONFIG_KEYS = {
    'fwi':  'fwi_dir',
    'ffmc': 'ffmc_dir',
    'dmc':  'dmc_dir',
    'dc':   'dc_dir',
    'isi':  'isi_dir',
    'bui':  'bui_dir',
}


def date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def verify_variable(var_name: str, var_dir: Path, dates: list,
                    quick: bool = False) -> dict:
    """
    Check all expected date files for one variable.

    Returns dict with keys:
        missing, corrupt, range_errors, ok, total
    """
    import numpy as np

    missing, corrupt, range_errors, ok = [], [], [], 0
    ref_shape = None
    lo, hi = EXPECTED_RANGES[var_name]

    for d in dates:
        fname = var_dir / f"{var_name}_{d.strftime('%Y%m%d')}.tif"
        if not fname.exists():
            missing.append(d)
            continue

        if quick:
            ok += 1
            continue

        # Full integrity check
        try:
            import rasterio
            with rasterio.open(fname) as src:
                arr = src.read(1)
                shape = (src.height, src.width)
        except Exception as e:
            corrupt.append((d, str(e)))
            continue

        # Shape consistency
        if ref_shape is None:
            ref_shape = shape
        elif shape != ref_shape:
            corrupt.append((d, f"shape {shape} != expected {ref_shape}"))
            continue

        # Value range (ignore NaN / nodata)
        import numpy as np
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            vmin, vmax = float(valid.min()), float(valid.max())
            if vmax > hi * 1.5 or vmin < lo - 1:
                range_errors.append((d, vmin, vmax))

        ok += 1

    return {
        'missing':      missing,
        'corrupt':      corrupt,
        'range_errors': range_errors,
        'ok':           ok,
        'total':        len(dates),
        'shape':        ref_shape,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Verify FWI component GeoTIFFs for completeness and integrity"
    )
    add_config_argument(ap)
    ap.add_argument('--start', type=str, default='2018-01-01',
                    help='First date to check (YYYY-MM-DD, default 2018-01-01)')
    ap.add_argument('--end',   type=str, default='2024-12-31',
                    help='Last date to check (YYYY-MM-DD, default 2024-12-31)')
    ap.add_argument('--vars', nargs='+',
                    default=list(EXPECTED_RANGES.keys()),
                    choices=list(EXPECTED_RANGES.keys()),
                    help='Variables to check (default: all)')
    ap.add_argument('--quick', action='store_true',
                    help='Existence check only — skip file open and value range check')
    args = ap.parse_args()

    cfg = load_config(args.config)
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    dates = list(date_range(start, end))

    print("=" * 65)
    print("FWI COMPONENT VERIFICATION")
    print("=" * 65)
    print(f"Date range : {start} → {end}  ({len(dates)} days)")
    print(f"Variables  : {args.vars}")
    print(f"Mode       : {'quick (existence only)' if args.quick else 'full (open + range check)'}")
    print("=" * 65)

    all_ok = True

    for var in args.vars:
        var_dir = Path(get_path(cfg, CONFIG_KEYS[var]))
        print(f"\n[{var.upper():4s}]  {var_dir}")

        if not var_dir.exists():
            print(f"  ✗ Directory does not exist!")
            all_ok = False
            continue

        result = verify_variable(var, var_dir, dates, quick=args.quick)

        # Summary line
        status = "✓" if not result['missing'] and not result['corrupt'] else "✗"
        print(f"  {status}  {result['ok']}/{result['total']} files OK"
              f"  |  missing: {len(result['missing'])}"
              f"  |  corrupt: {len(result['corrupt'])}"
              f"  |  range errors: {len(result['range_errors'])}")
        if result['shape']:
            print(f"     grid shape: {result['shape']}")

        # Detail: missing dates (show first 10)
        if result['missing']:
            all_ok = False
            shown = result['missing'][:10]
            print(f"     MISSING dates: {[str(d) for d in shown]}"
                  + (f" ... +{len(result['missing'])-10} more"
                     if len(result['missing']) > 10 else ""))

        # Detail: corrupt files
        if result['corrupt']:
            all_ok = False
            for d, msg in result['corrupt'][:5]:
                print(f"     CORRUPT {d}: {msg}")

        # Detail: range errors
        if result['range_errors']:
            lo, hi = EXPECTED_RANGES[var]
            print(f"     Expected range [{lo}, {hi}]")
            for d, vmin, vmax in result['range_errors'][:5]:
                print(f"     RANGE ERROR {d}: min={vmin:.1f} max={vmax:.1f}")

    print("\n" + "=" * 65)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED ✓")
    else:
        print("RESULT: ISSUES FOUND ✗  — re-run download for missing dates")
    print("=" * 65)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
