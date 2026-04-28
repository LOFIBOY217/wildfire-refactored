#!/usr/bin/env python3
"""
Fire-recurrence analysis from NBAC+NFDB label stack.
Pure numpy, no model dependencies. ~5-8 min on the 22y label.

Phase 1 — answers 6 questions:
  A1. Fire return interval (FRI) distribution
  A2. % of land burned ≥1, ≥2, ≥5 times in 26 years
  A4. Has FRI shortened over time (2000-2012 vs 2013-2025)?
  B4. % of land NEVER burned
  D1. Persistence baseline: how much does last year's burn predict this year's?
  D3. What % of each year's fires happen in pixels never burned before?

Output:
  outputs/fire_recurrence_summary.json  (machine-readable)
  outputs/fire_recurrence_per_year.csv   (per-year stats for D1/D3 + plotting)
  console: human-readable summary

Usage:
  python scripts/analyze_fire_recurrence.py \
      --label_npy data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy \
      --land_ref data/fwi_data/fwi_20230801.tif \
      --out_json outputs/fire_recurrence_summary.json \
      --out_csv outputs/fire_recurrence_per_year.csv
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_land_mask(ref_path):
    """Use any reference TIF; non-NaN pixels = Canadian land."""
    import rasterio
    with rasterio.open(ref_path) as src:
        arr = src.read(1)
    return ~np.isnan(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_npy", required=True)
    ap.add_argument("--land_ref", required=True,
                    help="Reference TIF for land mask (e.g. fwi_20230801.tif)")
    ap.add_argument("--label_start", default="2000-05-01")
    ap.add_argument("--out_json", default="outputs/fire_recurrence_summary.json")
    ap.add_argument("--out_csv", default="outputs/fire_recurrence_per_year.csv")
    args = ap.parse_args()

    label_start = date.fromisoformat(args.label_start)
    print(f"  Loading label: {args.label_npy}")
    fire = np.load(args.label_npy, mmap_mode="r")
    T, H, W = fire.shape
    print(f"    shape: {fire.shape}, label_start={label_start}")

    land = load_land_mask(args.land_ref)
    if land.shape != (H, W):
        print(f"    cropping land mask {land.shape} to label {(H, W)}")
        land = land[:H, :W]
    n_land = int(land.sum())
    print(f"    land pixels: {n_land:,} ({100 * n_land / (H * W):.1f}% of grid)")

    # ── Build per-pixel annual burn mask (years × H × W bool) ──────────
    # For each year, "burned in year Y" = ANY day in that year had fire=1
    print()
    print(f"  Building per-year annual burn mask ...")
    t0 = time.time()
    end_date = label_start + timedelta(days=T - 1)
    years = list(range(label_start.year, end_date.year + 1))
    n_years = len(years)
    annual = np.zeros((n_years, H, W), dtype=bool)

    for yi, yr in enumerate(years):
        # Fire-season window: Apr 1 — Oct 31 (covers all real wildfire dates)
        y_start = max(0, (date(yr, 1, 1) - label_start).days)
        y_end = min(T, (date(yr, 12, 31) - label_start).days + 1)
        if y_end <= y_start:
            continue
        # Stream chunks to avoid loading 9366×2281×2709 at once
        chunk_size = 90  # ~3 months at a time
        any_burn = np.zeros((H, W), dtype=bool)
        for c0 in range(y_start, y_end, chunk_size):
            c1 = min(c0 + chunk_size, y_end)
            sub = np.asarray(fire[c0:c1])
            any_burn |= sub.any(axis=0)
        annual[yi] = any_burn
        n_pix = int(any_burn.sum())
        n_land_burned = int((any_burn & land).sum())
        print(f"    {yr}: {n_pix:>9,d} burned px ({n_land_burned:>9,d} on land)")
    print(f"    annual mask built in {time.time() - t0:.0f}s")

    # ── A1, A2: per-pixel burn count + FRI distribution ──────────────
    print()
    print("=" * 70)
    print("A1 / A2 — Per-pixel burn count and Fire Return Interval (FRI)")
    print("=" * 70)
    burn_count = annual.sum(axis=0)  # (H, W) int — # years burned
    burn_count_land = burn_count[land]

    # Distribution of burn counts
    n_burned_ge1 = int((burn_count_land >= 1).sum())
    n_burned_ge2 = int((burn_count_land >= 2).sum())
    n_burned_ge3 = int((burn_count_land >= 3).sum())
    n_burned_ge5 = int((burn_count_land >= 5).sum())
    n_never = int((burn_count_land == 0).sum())

    print(f"  Of {n_land:,} land pixels (over {n_years} years):")
    print(f"    NEVER burned    : {n_never:>11,d}  ({100*n_never/n_land:>5.1f}%)")
    print(f"    burned ≥ 1 year : {n_burned_ge1:>11,d}  ({100*n_burned_ge1/n_land:>5.1f}%)")
    print(f"    burned ≥ 2 years: {n_burned_ge2:>11,d}  ({100*n_burned_ge2/n_land:>5.1f}%)")
    print(f"    burned ≥ 3 years: {n_burned_ge3:>11,d}  ({100*n_burned_ge3/n_land:>5.1f}%)")
    print(f"    burned ≥ 5 years: {n_burned_ge5:>11,d}  ({100*n_burned_ge5/n_land:>5.1f}%)")
    print(f"    max burn count: {int(burn_count_land.max())} years")

    # FRI: average gap between consecutive burn years per pixel
    # Only compute for pixels with ≥ 2 burns
    print()
    print("  FRI (average gap between consecutive burn YEARS, per pixel):")
    fri_values = []
    repeat_pixel_idx = np.argwhere(burn_count >= 2)
    print(f"    computing FRI for {len(repeat_pixel_idx):,} repeat-burn pixels ...")
    t0 = time.time()
    # Vectorize: for each pixel, find which years it burned, then diff
    # Process in chunks of pixels to manage memory
    pix_chunk = 100000
    for c0 in range(0, len(repeat_pixel_idx), pix_chunk):
        c1 = min(c0 + pix_chunk, len(repeat_pixel_idx))
        idx = repeat_pixel_idx[c0:c1]
        rows = idx[:, 0]
        cols = idx[:, 1]
        # (n_years, n_pixels_in_chunk) — which year-pixel pairs burned
        burned_mat = annual[:, rows, cols]   # (n_years, n_pix_chunk) bool
        for j in range(burned_mat.shape[1]):
            yrs_burned = np.where(burned_mat[:, j])[0]
            if len(yrs_burned) >= 2:
                gaps = np.diff(yrs_burned)
                fri_values.extend(gaps.tolist())
    fri_arr = np.array(fri_values, dtype=np.int32)
    print(f"    FRI samples: {len(fri_arr):,} (computed in {time.time() - t0:.0f}s)")
    print(f"    FRI mean   : {fri_arr.mean():.2f} years")
    print(f"    FRI median : {np.median(fri_arr):.0f} years")
    print(f"    FRI 25th % : {np.percentile(fri_arr, 25):.0f} years")
    print(f"    FRI 75th % : {np.percentile(fri_arr, 75):.0f} years")
    print(f"    FRI 90th % : {np.percentile(fri_arr, 90):.0f} years")
    print(f"    FRI min/max: {fri_arr.min()} / {fri_arr.max()}")
    pct_under_5 = 100 * (fri_arr < 5).sum() / len(fri_arr)
    pct_under_10 = 100 * (fri_arr < 10).sum() / len(fri_arr)
    print(f"    % FRI < 5 yr : {pct_under_5:.1f}%")
    print(f"    % FRI < 10 yr: {pct_under_10:.1f}%")

    # ── A4: has FRI shortened over time? ─────────────────────────────
    # Compare: pixels with first burn 2000-2012 vs 2013-2025.
    # For each, FRI = next burn year - first burn year (only first interval)
    print()
    print("=" * 70)
    print("A4 — Has FRI shortened? (2000-2012 vs 2013-2025 first-interval)")
    print("=" * 70)
    split_year = 2013  # threshold
    split_idx = years.index(split_year) if split_year in years else n_years // 2
    # For each pixel, find years it burned. If first-burn-year < split_year and
    # has another burn, the first-interval = year2 - year1.
    early_intervals = []
    late_intervals = []
    t0 = time.time()
    for c0 in range(0, len(repeat_pixel_idx), pix_chunk):
        c1 = min(c0 + pix_chunk, len(repeat_pixel_idx))
        idx = repeat_pixel_idx[c0:c1]
        burned_mat = annual[:, idx[:, 0], idx[:, 1]]
        for j in range(burned_mat.shape[1]):
            ys = np.where(burned_mat[:, j])[0]
            if len(ys) >= 2:
                first_interval = int(ys[1] - ys[0])
                if ys[0] < split_idx:
                    early_intervals.append(first_interval)
                else:
                    late_intervals.append(first_interval)
    print(f"    computed in {time.time() - t0:.0f}s")
    if early_intervals:
        ea = np.array(early_intervals)
        print(f"    Early period (first burn < {split_year}): "
              f"n={len(ea):,}  mean={ea.mean():.2f}  median={np.median(ea):.0f}")
    if late_intervals:
        la = np.array(late_intervals)
        print(f"    Late period  (first burn ≥ {split_year}): "
              f"n={len(la):,}  mean={la.mean():.2f}  median={np.median(la):.0f}")
    if early_intervals and late_intervals:
        delta = la.mean() - ea.mean()
        print(f"    Δ mean FRI : {delta:+.2f} years (negative = shortened)")
        # KS test
        try:
            from scipy.stats import ks_2samp
            d, p = ks_2samp(ea, la)
            print(f"    KS test    : D={d:.3f}, p={p:.2e}")
        except ImportError:
            print(f"    (scipy unavailable, skipping KS test)")

    # ── B4: % land never burned (already computed in A2, restate) ─────
    print()
    print("=" * 70)
    print(f"B4 — Of {n_land:,} land pixels, {n_never:,} ({100*n_never/n_land:.1f}%) NEVER burned in {n_years}y")
    print("=" * 70)

    # ── D1, D3: predictability from history ──────────────────────────
    print()
    print("=" * 70)
    print("D1 / D3 — Predictability from history (per-year)")
    print("=" * 70)

    # D1: persistence baseline = "did year Y-1 fires predict year Y fires?"
    # For each year Y (2001..2025), recall = |Y ∩ Y-1| / |Y|
    # D3: % of year Y fires that fall in pixels NEVER burned 2000..Y-1
    per_year_rows = []
    print(f"  {'year':>5} {'n_burn_Y':>10} {'recall_Y-1':>12} "
          f"{'recall_Y-5':>12} {'%_in_unburned':>15}")
    cumulative_burn = np.zeros((H, W), dtype=bool)
    for yi, yr in enumerate(years):
        burn_y = annual[yi] & land
        n_y = int(burn_y.sum())
        if n_y == 0:
            continue

        # D1: recall from previous year
        if yi >= 1:
            prev1 = annual[yi - 1] & land
            recall_1 = float((burn_y & prev1).sum()) / n_y
        else:
            recall_1 = float("nan")

        # D1 extended: recall from union of past 5 years
        if yi >= 5:
            prev5 = (annual[yi - 5: yi].any(axis=0)) & land
            recall_5 = float((burn_y & prev5).sum()) / n_y
        else:
            recall_5 = float("nan")

        # D3: % of year-Y fire pixels that fall in NEVER-burned area
        #     (cumulative burn up through year Y-1)
        novel_mask = ~cumulative_burn & land
        n_in_novel = int((burn_y & novel_mask).sum())
        pct_novel = 100.0 * n_in_novel / n_y if n_y > 0 else 0.0

        per_year_rows.append({
            "year": yr,
            "n_burn": n_y,
            "recall_lookback_1y": recall_1,
            "recall_lookback_5y": recall_5,
            "pct_in_never_burned": pct_novel,
        })
        print(f"  {yr:>5} {n_y:>10,d} "
              f"{recall_1:>12.3f} " if not np.isnan(recall_1) else f"  {yr:>5} {n_y:>10,d} {'-':>12} ",
              end="")
        if not np.isnan(recall_5):
            print(f"{recall_5:>12.3f} {pct_novel:>14.1f}%")
        else:
            print(f"{'-':>12} {pct_novel:>14.1f}%")

        cumulative_burn |= burn_y

    # ── Write outputs ────────────────────────────────────────────────
    summary = {
        "label_npy": args.label_npy,
        "n_land_pixels": int(n_land),
        "n_years": n_years,
        "year_range": [years[0], years[-1]],
        "A2_burn_count_pct": {
            "never_burned": 100 * n_never / n_land,
            "ge_1_year": 100 * n_burned_ge1 / n_land,
            "ge_2_years": 100 * n_burned_ge2 / n_land,
            "ge_3_years": 100 * n_burned_ge3 / n_land,
            "ge_5_years": 100 * n_burned_ge5 / n_land,
        },
        "A1_FRI": {
            "n_samples": int(len(fri_arr)),
            "mean": float(fri_arr.mean()),
            "median": float(np.median(fri_arr)),
            "p25": float(np.percentile(fri_arr, 25)),
            "p75": float(np.percentile(fri_arr, 75)),
            "p90": float(np.percentile(fri_arr, 90)),
            "min": int(fri_arr.min()),
            "max": int(fri_arr.max()),
            "pct_under_5y": float((fri_arr < 5).sum() / len(fri_arr)),
            "pct_under_10y": float((fri_arr < 10).sum() / len(fri_arr)),
        },
        "A4_FRI_shift": {
            "split_year": split_year,
            "early_n": len(early_intervals),
            "early_mean": float(np.mean(early_intervals)) if early_intervals else None,
            "late_n": len(late_intervals),
            "late_mean": float(np.mean(late_intervals)) if late_intervals else None,
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print()
    print(f"  → wrote {args.out_json}")

    fieldnames = ["year", "n_burn", "recall_lookback_1y",
                  "recall_lookback_5y", "pct_in_never_burned"]
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in per_year_rows:
            w.writerow(r)
    print(f"  → wrote {args.out_csv} ({len(per_year_rows)} rows)")


if __name__ == "__main__":
    main()
