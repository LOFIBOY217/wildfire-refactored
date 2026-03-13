"""
check_v2_data.py -- V2 training data availability check
=========================================================
Scans all inputs required by train_s2s_hotspot_cwfis_v2 and prints a
year x channel coverage table so you can immediately see what is present
and what is still missing.

V2 inputs:
  ch 0  FWI        fwi_dir          fwi_YYYYMMDD.tif      (flat dir)
  ch 1  2t         observation_dir  2t/2t_YYYYMMDD.tif    (subdir)
  ch 2  2d         observation_dir  2d/2d_YYYYMMDD.tif    (subdir)
  ch 3  FFMC       ffmc_dir         ffmc_YYYYMMDD.tif     (flat dir)
  ch 4  DMC        dmc_dir          dmc_YYYYMMDD.tif      (flat dir)
  ch 5  DC         dc_dir           dc_YYYYMMDD.tif       (flat dir)
  ch 6  BUI        bui_dir          bui_YYYYMMDD.tif      (flat dir)
  static  fire_clim  fire_clim_tif  fire_climatology.tif
  labels  hotspot    hotspot_csv    hotspot_*.csv

Usage:
    python -m src.data_ops.validation.check_v2_data \\
        --config configs/paths_trillium.yaml

    # Custom date range:
    python -m src.data_ops.validation.check_v2_data \\
        --config configs/paths_trillium.yaml \\
        --data_start 2018-01-01 \\
        --data_end   2025-12-31

    # Show every missing date range (verbose):
    python -m src.data_ops.validation.check_v2_data \\
        --config configs/paths_trillium.yaml \\
        --show_missing

Exit codes:
    0 -- data complete (fully-aligned coverage >= 95%)
    1 -- issues found
"""

import argparse
import glob
import os
import sys
from datetime import date, datetime, timedelta

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    from pathlib import Path
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument

try:
    from src.utils.date_utils import extract_date_from_filename
except ImportError:
    import re as _re
    def extract_date_from_filename(fname):
        m = _re.search(r'(\d{4})(\d{2})(\d{2})', fname)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
        return None

# ---------------------------------------------------------------------------
# Terminal colours (disabled on non-TTY or plain Windows cmd)
# ---------------------------------------------------------------------------
_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"

def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

OK   = lambda s: _c(s, "32")
WARN = lambda s: _c(s, "33")
ERR  = lambda s: _c(s, "31")
BOLD = lambda s: _c(s, "1")
DIM  = lambda s: _c(s, "2")

def _icon(pct):
    """Single-character coverage icon for the year x channel table."""
    if pct >= 95: return OK("v")   # full
    if pct >= 10: return WARN("~") # partial
    return ERR("x")                # missing

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
_DATE_FMTS = [
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
    "%Y%m%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y",
]

def _parse_date_str(s):
    """Try multiple formats; return a date object or None."""
    s = s.strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

def _cli_date(s):
    d = _parse_date_str(s)
    if d is None:
        raise argparse.ArgumentTypeError(f"Cannot parse date: {s!r}")
    return d

# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def _scan_flat(directory, prefix):
    """Scan flat directory for <prefix>_YYYYMMDD.tif. Returns {date: path}."""
    found = {}
    for p in glob.glob(os.path.join(directory, f"{prefix}_*.tif")):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            found[d] = p
    return found

def _scan_subdir(parent, prefix):
    """
    Scan <parent>/<prefix>/<prefix>_YYYYMMDD.tif.
    Falls back to flat layout if the subdirectory is empty.
    Returns {date: path}.
    """
    found = {}
    for p in glob.glob(os.path.join(parent, prefix, f"{prefix}_*.tif")):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            found[d] = p
    if not found:
        for p in glob.glob(os.path.join(parent, f"{prefix}_*.tif")):
            d = extract_date_from_filename(os.path.basename(p))
            if d:
                found[d] = p
    return found

def _group_ranges(dates):
    """Compress a sorted date list into human-readable range strings."""
    if not dates:
        return []
    ranges, start, prev = [], dates[0], dates[0]
    for d in dates[1:]:
        if (d - prev).days == 1:
            prev = d
        else:
            ranges.append(str(start) if start == prev else f"{start} -> {prev}")
            start = prev = d
    ranges.append(str(start) if start == prev else f"{start} -> {prev}")
    return ranges

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="V2 training data availability check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_argument(ap)
    ap.add_argument("--data_start", type=_cli_date, default="2018-01-01",
                    help="Start of checked date range.")
    ap.add_argument("--data_end",   type=_cli_date, default="2025-12-31",
                    help="End of checked date range. "
                         "Set to pred_end + lead_end + 5 to match training exactly.")
    ap.add_argument("--show_missing", action="store_true",
                    help="Print every missing date range (can be long).")
    args = ap.parse_args()

    data_start = args.data_start
    data_end   = args.data_end

    # Build the full set of expected dates (no month filter --
    # the training script scans every calendar day in the range)
    expected = []
    cur = data_start
    while cur <= data_end:
        expected.append(cur)
        cur += timedelta(days=1)
    expected_set = set(expected)
    years = sorted(set(d.year for d in expected))

    # ----------------------------------------------------------------
    # Load config & resolve paths
    # ----------------------------------------------------------------
    cfg   = load_config(args.config)
    paths = cfg.get("paths", {})

    def _path(key):
        try:
            return get_path(cfg, key)
        except Exception:
            return None

    fwi_dir     = _path("fwi_dir")
    obs_dir     = _path("observation_dir") or _path("ecmwf_dir")
    ffmc_dir    = _path("ffmc_dir")
    dmc_dir     = _path("dmc_dir")
    dc_dir      = _path("dc_dir")
    bui_dir     = _path("bui_dir")
    fire_clim   = paths.get("fire_climatology_tif") or _path("fire_climatology_tif")
    hotspot_csv = _path("hotspot_csv")

    # ----------------------------------------------------------------
    # Scan each channel
    # ----------------------------------------------------------------
    # (name, layout, directory, file_prefix, channel_label)
    channels = [
        ("FWI",  "flat",   fwi_dir,  "fwi",  "ch 0"),
        ("2t",   "subdir", obs_dir,  "2t",   "ch 1"),
        ("2d",   "subdir", obs_dir,  "2d",   "ch 2"),
        ("FFMC", "flat",   ffmc_dir, "ffmc", "ch 3"),
        ("DMC",  "flat",   dmc_dir,  "dmc",  "ch 4"),
        ("DC",   "flat",   dc_dir,   "dc",   "ch 5"),
        ("BUI",  "flat",   bui_dir,  "bui",  "ch 6"),
    ]

    ch_dicts = {}
    for name, layout, directory, prefix, _ in channels:
        if not directory or not os.path.isdir(directory):
            ch_dicts[name] = {}
            continue
        ch_dicts[name] = (
            _scan_flat(directory, prefix) if layout == "flat"
            else _scan_subdir(directory, prefix)
        )

    # Fully-aligned dates: all 7 channels present simultaneously
    non_empty = [set(d.keys()) for d in ch_dicts.values() if d]
    aligned_set = (
        set.intersection(*non_empty) if len(non_empty) == 7 else set()
    )

    # ----------------------------------------------------------------
    # Print: year x channel coverage table
    # ----------------------------------------------------------------
    YEAR_W = 6   # column width per year
    NAME_W = 14  # channel name column width

    print()
    print(BOLD("=" * 72))
    print(BOLD(f"  V2 DATA AVAILABILITY   {data_start}  ->  {data_end}"))
    print(BOLD("=" * 72))
    print()

    # Header row
    hdr = f"  {'Channel':<{NAME_W}}"
    for yr in years:
        hdr += f"  {yr:>{YEAR_W - 2}}"
    hdr += f"  {'Present / Total':>16}"
    print(BOLD(hdr))
    print("  " + "-" * (NAME_W + len(years) * YEAR_W + 18))

    all_missing = {}
    has_problems = False

    def _year_pct(fdict, yr):
        yr_exp = [d for d in expected if d.year == yr]
        if not yr_exp:
            return 0.0
        return 100.0 * sum(1 for d in yr_exp if d in fdict) / len(yr_exp)

    for name, _, _, _, ch_label in channels:
        fdict = ch_dicts[name]
        row   = f"  {name + ' (' + ch_label + ')':<{NAME_W}}"

        for yr in years:
            icon = _icon(_year_pct(fdict, yr))
            row += f"  {icon:>{YEAR_W - 2}}"

        n_found    = sum(1 for d in expected if d in fdict)
        n_expected = len(expected)
        row += f"  {n_found:>7,} / {n_expected:<7,}"
        print(row)

        missing = [d for d in expected if d not in fdict]
        all_missing[name] = missing
        if missing:
            has_problems = True

    # Separator + fully-aligned summary row
    print("  " + "-" * (NAME_W + len(years) * YEAR_W + 18))
    row = f"  {'All 7 aligned':<{NAME_W}}"
    for yr in years:
        yr_exp = [d for d in expected if d.year == yr]
        pct    = (100.0 * sum(1 for d in yr_exp if d in aligned_set)
                  / max(len(yr_exp), 1))
        row   += f"  {_icon(pct):>{YEAR_W - 2}}"
    n_aligned  = sum(1 for d in expected if d in aligned_set)
    row       += f"  {n_aligned:>7,} / {len(expected):<7,}"
    print(row)

    print()
    print(f"  Legend:  {OK('v')} >= 95%   {WARN('~')} partial   {ERR('x')} < 10%")
    print()

    # ----------------------------------------------------------------
    # Static files & label CSV
    # ----------------------------------------------------------------
    print(BOLD("-- Static files & labels " + "-" * 46))
    print()

    # fire_climatology.tif
    if fire_clim and os.path.isfile(fire_clim):
        size_mb = os.path.getsize(fire_clim) / 1e6
        print(f"  fire_climatology.tif   [{OK('OK')}]   {size_mb:.1f} MB   {DIM(fire_clim)}")
    else:
        print(f"  fire_climatology.tif   [{WARN('MISSING')}]   "
              f"{fire_clim or '(not set in config)'}   "
              f"<- channel 7 will be zeros; run make_fire_climatology.py")
        has_problems = True

    # hotspot CSV
    if hotspot_csv and os.path.isfile(hotspot_csv):
        size_mb   = os.path.getsize(hotspot_csv) / 1e6
        row_count = 0
        csv_min   = None
        csv_max   = None
        date_col  = None
        bad_rows  = 0
        try:
            import csv as _csv
            with open(hotspot_csv, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        for col in ("date", "DATE", "acq_date", "REP_DATE",
                                    "hotspot_date", "FIRE_DATE", "fire_date",
                                    "detection_date"):
                            if col in row:
                                date_col = col
                                break
                    row_count += 1
                    if date_col:
                        d = _parse_date_str(row.get(date_col, ""))
                        if d is None:
                            bad_rows += 1
                        else:
                            if csv_min is None or d < csv_min: csv_min = d
                            if csv_max is None or d > csv_max: csv_max = d
        except Exception as e:
            print(f"  hotspot CSV            [{WARN('READ ERROR')}]   {e}")
            csv_min = csv_max = None

        if csv_min and csv_max:
            covers_start = csv_min <= data_start
            covers_end   = csv_max >= data_end
            tag = OK("OK") if (covers_start and covers_end) else WARN("PARTIAL")
            print(f"  hotspot CSV            [{tag}]   "
                  f"{row_count:,} rows   {csv_min} -> {csv_max}   {size_mb:.1f} MB")
            if not covers_start:
                print(f"    -> CSV starts at {csv_min}, need <= {data_start}")
            if not covers_end:
                print(f"    -> CSV ends at {csv_max}, need >= {data_end}")
            if bad_rows:
                print(f"    -> {WARN(str(bad_rows))} rows with unparseable dates")
        else:
            print(f"  hotspot CSV            [{OK('OK')}]   "
                  f"{row_count:,} rows   {size_mb:.1f} MB   (date column not detected)")
    else:
        print(f"  hotspot CSV            [{ERR('MISSING')}]   "
              f"{hotspot_csv or '(not set in config)'}   <- required training labels")
        has_problems = True

    print()

    # ----------------------------------------------------------------
    # Missing data details
    # ----------------------------------------------------------------
    any_missing = any(v for v in all_missing.values())
    if any_missing:
        print(BOLD("-- Missing data details " + "-" * 47))
        print()
        for name, missing in all_missing.items():
            if not missing:
                continue
            ranges = _group_ranges(missing)
            print(f"  {name}:  {len(missing):,} days missing  ({len(ranges)} gap(s))")
            limit = None if args.show_missing else 4
            for r in (ranges if limit is None else ranges[:limit]):
                print(f"    {r}")
            if limit and len(ranges) > limit:
                print(f"    ... +{len(ranges) - limit} more  "
                      f"(re-run with --show_missing to see all)")
            print()

        print(BOLD("  Fix commands:"))
        print()
        print("  # FWI / FFMC / DMC / DC / BUI")
        print("  python -m src.data_ops.download.fwi_historical \\")
        print(f"      --start {data_start.year} --end {data_end.year} \\")
        print("      --months 4 5 6 7 8 9 10 \\")
        print("      --reference data/fwi_data/fwi_20250615.tif")
        print()
        print("  # 2t / 2d  (ERA5 reanalysis observations)")
        print("  python -m src.data_ops.download.download_ecmwf_reanalysis_observations \\")
        print(f"      {data_start} {data_end} --workers 2")
        print()
        print("  # fire_climatology.tif")
        print("  python -m src.data_ops.processing.make_fire_climatology \\")
        print("      --reference data/fwi_data/fwi_20250615.tif")
        print()
        print("  # hotspot CSV")
        print("  python -m src.data_ops.download.download_hotspots")
        print()

    # ----------------------------------------------------------------
    # Final verdict
    # ----------------------------------------------------------------
    print(BOLD("=" * 72))
    if not has_problems and not any_missing:
        print(BOLD(OK("  OK  All data present. Ready to train.")))
    else:
        print(BOLD(ERR("  INCOMPLETE  Missing data found. See details above.")))
    print(BOLD("=" * 72))
    print()

    sys.exit(0 if (not has_problems and not any_missing) else 1)


if __name__ == "__main__":
    main()
