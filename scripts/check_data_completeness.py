"""
Data Completeness Check
========================
Scans all 7 daily channels (FWI, 2t, 2d, FFMC, DMC, DC, BUI) and reports:
  - Per-channel date coverage (first/last date, total files)
  - Aligned date count (days where ALL 7 channels are present)
  - Monthly gap report showing which months are missing per channel
  - Summary: whether the target range (e.g. 2018-01-01 ~ 2025-12-31) is fully covered

Usage:
    python scripts/check_data_completeness.py \
        --config configs/paths_narval.yaml \
        --target-start 2018-01-01 \
        --target-end   2025-12-31
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta

import yaml


def extract_date_from_filename(filename: str):
    """Extract date from filename containing YYYYMMDD pattern."""
    match = re.search(r'(\d{8})', os.path.basename(filename))
    if match:
        try:
            from datetime import datetime
            return datetime.strptime(match.group(1), '%Y%m%d').date()
        except ValueError:
            return None
    return None


def build_file_dict(directory, prefix):
    """Build {date: filepath} dict. Try flat first, then subdirectory."""
    result = {}
    if not os.path.isdir(directory):
        return result

    # Try flat: {dir}/{prefix}_*.tif
    pattern_flat = os.path.join(directory, f"{prefix}_*.tif")
    files = glob.glob(pattern_flat)

    # Try subdirectory: {dir}/{prefix}/{prefix}_*.tif
    if not files:
        pattern_sub = os.path.join(directory, prefix, f"{prefix}_*.tif")
        files = glob.glob(pattern_sub)

    # For FWI dir, also try *.tif without prefix
    if not files and prefix == "fwi":
        pattern_all = os.path.join(directory, "*.tif")
        files = glob.glob(pattern_all)

    for f in files:
        d = extract_date_from_filename(f)
        if d is not None:
            result[d] = f
    return result


def build_era5_dict(directory, prefix):
    """Build {date: filepath} dict for ERA5 grib or tif files."""
    result = {}
    if not os.path.isdir(directory):
        return result

    # Try TIF first: {dir}/{prefix}/{prefix}_*.tif or {dir}/{prefix}_*.tif
    for pattern in [
        os.path.join(directory, prefix, f"{prefix}_*.tif"),
        os.path.join(directory, f"{prefix}_*.tif"),
    ]:
        files = glob.glob(pattern)
        if files:
            for f in files:
                d = extract_date_from_filename(f)
                if d is not None:
                    result[d] = f
            return result

    # Try GRIB: era5_sfc_YYYY_MM_DD.grib → need to extract date differently
    grib_pattern = os.path.join(directory, "era5_sfc_*.grib")
    files = glob.glob(grib_pattern)
    for f in files:
        basename = os.path.basename(f)
        # era5_sfc_2019_04_01.grib → extract 20190401
        match = re.search(r'era5_sfc_(\d{4})_(\d{2})_(\d{2})\.grib', basename)
        if match:
            try:
                d = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                result[d] = f
            except ValueError:
                continue
    return result


def monthly_coverage(dates_set, start_date, end_date):
    """Return dict: (year, month) -> count of days present."""
    coverage = defaultdict(int)
    for d in dates_set:
        if start_date <= d <= end_date:
            coverage[(d.year, d.month)] += 1
    return coverage


def print_monthly_grid(channel_dicts, start_date, end_date):
    """Print a monthly coverage grid for all channels."""
    import calendar

    channels = list(channel_dicts.keys())
    # Header
    print(f"\n{'Month':>10}", end="")
    for ch in channels:
        print(f" {ch:>6}", end="")
    print(f" {'ALIGN':>6}")
    print("-" * (11 + 7 * (len(channels) + 1)))

    # Compute aligned dates
    all_date_sets = [set(d.keys()) for d in channel_dicts.values()]

    cur_year = start_date.year
    cur_month = start_date.month
    end_year = end_date.year
    end_month = end_date.month

    total_missing_months = defaultdict(list)

    while (cur_year, cur_month) <= (end_year, end_month):
        days_in_month = calendar.monthrange(cur_year, cur_month)[1]
        month_start = date(cur_year, cur_month, 1)
        month_end = date(cur_year, cur_month, days_in_month)
        # Clamp to target range
        eff_start = max(month_start, start_date)
        eff_end = min(month_end, end_date)
        expected = (eff_end - eff_start).days + 1

        label = f"{cur_year}-{cur_month:02d}"
        print(f"{label:>10}", end="")

        month_aligned = 0
        for ch_name, ch_dict in channel_dicts.items():
            count = sum(1 for d in ch_dict if eff_start <= d <= eff_end)
            if count == 0:
                print(f"  {'---':>4}", end="")
                total_missing_months[ch_name].append(label)
            elif count < expected:
                print(f" {count:>3}*", end="")  # partial
            else:
                print(f" {count:>4}", end="")

        # Aligned count for this month
        aligned_count = 0
        d = eff_start
        while d <= eff_end:
            if all(d in s for s in all_date_sets):
                aligned_count += 1
            d += timedelta(days=1)
        if aligned_count == 0:
            print(f"  {'---':>4}", end="")
        elif aligned_count < expected:
            print(f" {aligned_count:>3}*", end="")
        else:
            print(f" {aligned_count:>4}", end="")

        print()

        # Next month
        if cur_month == 12:
            cur_year += 1
            cur_month = 1
        else:
            cur_month += 1

    return total_missing_months


def main():
    ap = argparse.ArgumentParser(description="Check data completeness for training.")
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/paths_narval.yaml)")
    ap.add_argument("--target-start", default="2018-01-01", help="Target start date (YYYY-MM-DD)")
    ap.add_argument("--target-end", default="2025-12-31", help="Target end date (YYYY-MM-DD)")
    args = ap.parse_args()

    target_start = date.fromisoformat(args.target_start)
    target_end = date.fromisoformat(args.target_end)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", cfg)

    fwi_dir = paths["fwi_dir"]
    obs_dir = paths.get("observation_dir", paths.get("ecmwf_dir", ""))
    ffmc_dir = paths["ffmc_dir"]
    dmc_dir = paths["dmc_dir"]
    dc_dir = paths["dc_dir"]
    bui_dir = paths.get("bui_dir", "")

    print("=" * 70)
    print(f"DATA COMPLETENESS CHECK")
    print(f"Target range: {target_start} → {target_end}")
    print(f"Lead buffer:  +46 days → need data to {target_end + timedelta(days=46)}")
    print("=" * 70)

    # Build file dicts
    print("\nScanning directories...", flush=True)
    channel_dicts = {}

    channel_dicts["FWI"] = build_file_dict(fwi_dir, "fwi")
    channel_dicts["FFMC"] = build_file_dict(ffmc_dir, "ffmc")
    channel_dicts["DMC"] = build_file_dict(dmc_dir, "dmc")
    channel_dicts["DC"] = build_file_dict(dc_dir, "dc")
    channel_dicts["BUI"] = build_file_dict(bui_dir, "bui")
    channel_dicts["2t"] = build_era5_dict(obs_dir, "2t")
    channel_dicts["2d"] = build_era5_dict(obs_dir, "2d")

    # Per-channel summary
    print(f"\n{'Channel':>8}  {'Files':>6}  {'First':>12}  {'Last':>12}  {'In Range':>8}")
    print("-" * 60)
    for ch_name, ch_dict in channel_dicts.items():
        dates = sorted(ch_dict.keys())
        n_total = len(dates)
        if n_total == 0:
            print(f"{ch_name:>8}  {'0':>6}  {'N/A':>12}  {'N/A':>12}  {'0':>8}")
            continue
        first = dates[0]
        last = dates[-1]
        in_range = sum(1 for d in dates if target_start <= d <= target_end)
        print(f"{ch_name:>8}  {n_total:>6}  {str(first):>12}  {str(last):>12}  {in_range:>8}")

    # Extended range (target_end + 46 days for lead)
    extended_end = target_end + timedelta(days=46)
    print(f"\n{'Channel':>8}  {'Extended (→' + str(extended_end) + ')':>20}")
    print("-" * 40)
    for ch_name, ch_dict in channel_dicts.items():
        dates = sorted(ch_dict.keys())
        in_ext = sum(1 for d in dates if target_start <= d <= extended_end)
        print(f"{ch_name:>8}  {in_ext:>20}")

    # Aligned dates (all 7 channels present)
    all_date_sets = [set(d.keys()) for d in channel_dicts.values()]
    all_dates_range = set()
    d = target_start
    while d <= extended_end:
        all_dates_range.add(d)
        d += timedelta(days=1)

    aligned = set.intersection(*all_date_sets) & all_dates_range
    aligned_sorted = sorted(aligned)

    print(f"\n{'=' * 70}")
    print(f"ALIGNED DATES (all 7 channels present, within extended range):")
    if aligned_sorted:
        print(f"  Count : {len(aligned_sorted)}")
        print(f"  First : {aligned_sorted[0]}")
        print(f"  Last  : {aligned_sorted[-1]}")
    else:
        print(f"  Count : 0  *** NO ALIGNED DATES ***")
    print(f"{'=' * 70}")

    # Monthly grid
    print("\nMONTHLY COVERAGE (extended range):")
    print("  Numbers = days with data.  --- = no data.  * = partial month.")
    missing = print_monthly_grid(channel_dicts, target_start, extended_end)

    # Gap analysis
    print(f"\n{'=' * 70}")
    print("GAP SUMMARY — months with ZERO data per channel:")
    print("-" * 70)
    has_gaps = False
    for ch_name, months in missing.items():
        if months:
            has_gaps = True
            print(f"  {ch_name:>6}: {len(months)} months missing")
            # Group by year
            by_year = defaultdict(list)
            for m in months:
                y, mo = m.split("-")
                by_year[y].append(mo)
            for y in sorted(by_year):
                print(f"         {y}: months {', '.join(by_year[y])}")
    if not has_gaps:
        print("  No gaps! All channels have data for every month in range.")

    # Final verdict
    print(f"\n{'=' * 70}")
    total_days = (extended_end - target_start).days + 1
    coverage_pct = len(aligned_sorted) / total_days * 100 if total_days > 0 else 0
    print(f"VERDICT:")
    print(f"  Target range (with lead): {target_start} → {extended_end} ({total_days} days)")
    print(f"  Aligned days available  : {len(aligned_sorted)} ({coverage_pct:.1f}%)")

    if coverage_pct >= 95:
        print(f"  ✓ Coverage is good (≥95%). Ready for cache rebuild.")
    elif coverage_pct >= 70:
        print(f"  △ Coverage is partial ({coverage_pct:.1f}%). Winter months likely missing.")
        print(f"    Training will work but skip missing days. Consider downloading FWI winter months.")
    else:
        print(f"  ✗ Coverage is low ({coverage_pct:.1f}%). Check gap summary above.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
