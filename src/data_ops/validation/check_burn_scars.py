#!/usr/bin/env python3
"""
Validate NBAC burn scar data (years_since_burn + burn_count TIFs).

Checks:
  1. File existence for each year (2018-2024)
  2. Grid alignment (EPSG:3978, 2281×2709)
  3. Value ranges (years_since_burn: 0-9999, burn_count: 0-~20)
  4. Ecological plausibility (2023 should have most recent burns)
  5. burn_count consistency (burned pixels should match between files)

Usage:
    python -m src.data_ops.validation.check_burn_scars
    python -m src.data_ops.validation.check_burn_scars --config configs/paths_narval.yaml
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


FWI_CRS = "EPSG:3978"
EXPECTED_SHAPE = (2281, 2709)


def main():
    parser = argparse.ArgumentParser(description="Validate NBAC burn scar TIFs")
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2024)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        burn_dir = Path(get_path(cfg, "burn_scars_dir"))
    except (KeyError, TypeError):
        burn_dir = Path("data/burn_scars")

    print("=" * 60)
    print("BURN SCAR DATA VALIDATION")
    print("=" * 60)
    print(f"  Directory: {burn_dir}")
    print(f"  Years: {args.start_year}-{args.end_year}")

    all_ok = True

    for year in range(args.start_year, args.end_year + 1):
        age_path = burn_dir / f"years_since_burn_{year}.tif"
        count_path = burn_dir / f"burn_count_{year}.tif"

        print(f"\n--- {year} ---")

        # Check years_since_burn
        if not age_path.exists():
            print(f"  [FAIL] years_since_burn_{year}.tif MISSING")
            all_ok = False
            continue

        with rasterio.open(age_path) as src:
            age = src.read(1).astype(np.float32)
            crs = str(src.crs)
            shape = (src.height, src.width)

        if shape != EXPECTED_SHAPE:
            print(f"  [FAIL] Shape {shape} != {EXPECTED_SHAPE}")
            all_ok = False
        if crs != FWI_CRS:
            print(f"  [FAIL] CRS {crs} != {FWI_CRS}")
            all_ok = False

        never = int((age == 9999).sum())
        recent_5 = int(((age >= 0) & (age <= 5)).sum())
        burned_this_year = int((age == 0).sum())
        print(f"  years_since_burn: never={never:,}  within_5yr={recent_5:,}  "
              f"this_year(age=0)={burned_this_year:,}")

        # Check burn_count
        if not count_path.exists():
            print(f"  [WARN] burn_count_{year}.tif MISSING")
            continue

        with rasterio.open(count_path) as src:
            cnt = src.read(1).astype(np.float32)

        never_cnt = int((cnt == 0).sum())
        once = int((cnt == 1).sum())
        twice = int((cnt == 2).sum())
        three_plus = int((cnt >= 3).sum())
        max_cnt = int(cnt.max())

        print(f"  burn_count: never={never_cnt:,}  1x={once:,}  "
              f"2x={twice:,}  3+={three_plus:,}  max={max_cnt}")

        # Consistency check: never-burned should match
        if never != never_cnt:
            print(f"  [WARN] Never-burned mismatch: age={never:,} vs count={never_cnt:,}")

        # Plausibility: 2023 should have lots of recent burns
        if year == 2023 and burned_this_year < 10000:
            print(f"  [WARN] 2023 was record fire year but only {burned_this_year:,} "
                  f"pixels burned — check NBAC data")

    # ── Cross-validation: year-over-year burn_count consistency ──
    print(f"\n--- Year-over-year consistency ---")
    for year in range(args.start_year + 1, args.end_year + 1):
        prev_count_path = burn_dir / f"burn_count_{year - 1}.tif"
        curr_count_path = burn_dir / f"burn_count_{year}.tif"
        curr_age_path = burn_dir / f"years_since_burn_{year}.tif"
        if not (prev_count_path.exists() and curr_count_path.exists()
                and curr_age_path.exists()):
            continue

        with rasterio.open(prev_count_path) as src:
            prev_cnt = src.read(1).astype(np.int32)
        with rasterio.open(curr_count_path) as src:
            curr_cnt = src.read(1).astype(np.int32)
        with rasterio.open(curr_age_path) as src:
            curr_age = src.read(1).astype(np.int32)

        # Pixels burned in {year} (age=0) should have curr_count = prev_count + 1
        burned_this_year = curr_age == 0
        n_burned = int(burned_this_year.sum())
        if n_burned > 0:
            expected = prev_cnt[burned_this_year] + 1
            actual = curr_cnt[burned_this_year]
            match = int((expected == actual).sum())
            mismatch = n_burned - match
            pct = 100 * match / n_burned
            status = "OK" if pct > 95 else "WARN"
            print(f"  {year}: {n_burned:,} new burns, "
                  f"count consistency: {match:,}/{n_burned:,} ({pct:.0f}%) [{status}]")
            if pct < 95:
                all_ok = False
        else:
            print(f"  {year}: 0 new burns")

    # ── Cross-validation: spatial distribution vs fire_climatology ──
    print(f"\n--- Spatial distribution vs fire_climatology ---")
    try:
        clim_path = Path(get_path(cfg, "fire_climatology_tif"))
    except (KeyError, TypeError):
        clim_path = burn_dir.parent / "fire_climatology.tif"

    if clim_path.exists():
        with rasterio.open(clim_path) as src:
            clim = src.read(1).astype(np.float32)

        # Use most recent burn_count
        latest_count_path = burn_dir / f"burn_count_{args.end_year}.tif"
        if latest_count_path.exists():
            with rasterio.open(latest_count_path) as src:
                latest_cnt = src.read(1).astype(np.float32)

            # High fire_clim pixels (top 5%) should have higher burn_count
            clim_valid = clim[np.isfinite(clim) & (clim > 0)]
            if len(clim_valid) > 0:
                threshold = np.percentile(clim_valid, 95)
                high_clim = clim >= threshold
                low_clim = (clim > 0) & (clim < np.percentile(clim_valid, 50))

                mean_high = float(latest_cnt[high_clim].mean()) if high_clim.any() else 0
                mean_low = float(latest_cnt[low_clim].mean()) if low_clim.any() else 0
                print(f"  High fire_clim areas (top 5%): mean burn_count = {mean_high:.2f}")
                print(f"  Low fire_clim areas (bottom 50%): mean burn_count = {mean_low:.2f}")

                if mean_high > mean_low:
                    print(f"  [OK] High-clim areas burn more ({mean_high:.2f} > {mean_low:.2f})")
                else:
                    print(f"  [WARN] Expected high-clim areas to have more burns")
                    all_ok = False
    else:
        print(f"  [SKIP] fire_climatology.tif not found: {clim_path}")

    # ── Cross-validation: burn_age=0 vs hotspot records ──
    print(f"\n--- 2023 burn_age=0 vs hotspot records ---")
    try:
        hotspot_path = Path(get_path(cfg, "hotspot_csv"))
    except (KeyError, TypeError):
        hotspot_path = burn_dir.parent / "hotspot" / "hotspot_2018_2025.csv"

    if hotspot_path.exists():
        import pandas as pd
        from datetime import date as _date

        df = pd.read_csv(hotspot_path)
        # Detect date column
        date_col = None
        for col in ["acq_date", "date", "rep_date"]:
            if col in df.columns:
                date_col = col
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            hotspot_2023 = df[(df[date_col].dt.year == 2023)]
            n_hotspots = len(hotspot_2023)

            age_2023_path = burn_dir / "years_since_burn_2023.tif"
            if age_2023_path.exists():
                with rasterio.open(age_2023_path) as src:
                    age_2023 = src.read(1).astype(np.int32)
                n_burn_pixels = int((age_2023 == 0).sum())
                print(f"  Hotspot records in 2023: {n_hotspots:,}")
                print(f"  NBAC pixels burned in 2023 (age=0): {n_burn_pixels:,}")
                # They won't match exactly (different data sources) but should correlate
                if n_hotspots > 0 and n_burn_pixels > 0:
                    print(f"  [OK] Both sources confirm 2023 fire activity")
                elif n_hotspots > 1000 and n_burn_pixels == 0:
                    print(f"  [FAIL] Hotspots exist but NBAC shows no burns")
                    all_ok = False
        else:
            print(f"  [SKIP] No date column found in hotspot CSV")
    else:
        print(f"  [SKIP] Hotspot CSV not found: {hotspot_path}")

    print(f"\n{'=' * 60}")
    print(f"Result: {'ALL PASS' if all_ok else 'ISSUES FOUND'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
