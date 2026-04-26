#!/usr/bin/env python3
"""
Per-year analysis of fire activity + climate to test our claim that
2000-2017 vs 2018-2024 are non-i.i.d. (justifying why 22y model
overfits). Doesn't rely on outside published numbers — computes from
our own NBAC labels + FWI rasters.

Outputs:
    outputs/climate_drift_per_year.csv
    outputs/climate_drift_summary.md

Tests reported:
  - Annual fire pixels (NBAC)
  - Annual mean FWI (sampled, fire season May-Sep)
  - Annual fraction of high-FWI days (FWI > 30, fire season)
  - Mann-Kendall trend test on each
  - KS test 2000-2017 vs 2018-2024 (are these from same distribution?)
  - Year-similarity ranking: which past years are most like 2024?
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import rasterio


def load_nbac_per_year(npy_path, sidecar_json_path):
    """Sum NBAC positive pixels per calendar year from the dilated label stack."""
    with open(sidecar_json_path) as f:
        prov = json.load(f)
    label_start = date.fromisoformat(prov["date_range"][0])
    print(f"  NBAC start date: {label_start}")
    fire = np.load(npy_path, mmap_mode="r")  # (T, H, W) uint8
    T = fire.shape[0]

    by_year = defaultdict(lambda: 0)
    by_year_days = defaultdict(lambda: 0)
    fire_season_months = {5, 6, 7, 8, 9, 10}
    for t in range(T):
        d = label_start + (date.fromordinal(label_start.toordinal() + t)
                           - label_start)
        if d.month not in fire_season_months:
            continue
        by_year[d.year] += int(fire[t].sum())
        by_year_days[d.year] += 1
    return dict(by_year), dict(by_year_days)


def sample_fwi_per_year(fwi_dir, start_year, end_year, sample_dates_per_year=12):
    """For each year, sample N dates in fire season (May-Sep), read FWI,
    compute mean + count of high-FWI pixels. Avoid reading 9000+ files."""
    fire_season_months = [5, 6, 7, 8, 9]
    pattern = re.compile(r"fwi_(\d{4})(\d{2})(\d{2})\.tif")

    # Index files by date
    by_date = {}
    for p in glob.glob(os.path.join(fwi_dir, "fwi_*.tif")):
        m = pattern.search(os.path.basename(p))
        if not m:
            continue
        y, mo, dy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            by_date[date(y, mo, dy)] = p
        except ValueError:
            continue
    print(f"  FWI files indexed: {len(by_date)}")

    # Sample N dates per year evenly across fire season
    rng = np.random.default_rng(0)
    out = {}
    for year in range(start_year, end_year + 1):
        candidates = [d for d in by_date.keys()
                      if d.year == year and d.month in fire_season_months]
        if len(candidates) < 5:
            continue
        sample = sorted(rng.choice(candidates,
                                    size=min(sample_dates_per_year, len(candidates)),
                                    replace=False))
        means = []
        max_means = []
        high_fracs = []
        for d in sample:
            try:
                with rasterio.open(by_date[d]) as src:
                    arr = src.read(1).astype(np.float32)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                valid = arr > 0
                if not valid.any():
                    continue
                means.append(float(arr[valid].mean()))
                max_means.append(float(arr[valid].max()))
                high_fracs.append(float((arr > 30).sum() / valid.sum()))
            except Exception as e:
                print(f"    [skip] {d}: {e}")
        if means:
            out[year] = dict(
                fwi_mean=float(np.mean(means)),
                fwi_max=float(np.mean(max_means)),
                high_fwi_frac=float(np.mean(high_fracs)),
                n_dates_sampled=len(means),
            )
            print(f"    {year}: FWI mean={out[year]['fwi_mean']:.2f}, "
                  f"high-FWI frac={out[year]['high_fwi_frac']:.3%}")
    return out


def mann_kendall_simple(values):
    """Simple Mann-Kendall trend test. Returns S statistic + Z + p-value."""
    n = len(values)
    if n < 5:
        return None
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(values[j] - values[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    # Two-sided p-value from standard normal
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return dict(S=s, Z=float(z), p=float(p),
                trend="increasing" if z > 0 else "decreasing" if z < 0 else "none")


def ks_test_simple(a, b):
    """KS test scratch implementation. Returns D statistic + approximate p."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    n, m = len(a), len(b)
    en = np.sqrt(n * m / (n + m))
    # Asymptotic p
    p = 2 * np.exp(-2 * (en * D) ** 2)
    p = max(0.0, min(1.0, p))
    return dict(D=D, p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--fire_label_json", required=True)
    ap.add_argument("--fwi_dir", required=True)
    ap.add_argument("--start_year", type=int, default=2000)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--split_year", type=int, default=2018,
                    help="Cutoff: years <split are 'old period', >=split are 'new'")
    ap.add_argument("--out_csv", default="outputs/climate_drift_per_year.csv")
    ap.add_argument("--out_md", default="outputs/climate_drift_summary.md")
    args = ap.parse_args()

    print("=" * 70)
    print("CLIMATE-DRIFT ANALYSIS — testing if 2000-2017 vs 2018-2024 are i.i.d.")
    print("=" * 70)

    print("\n[1] Annual fire pixels (NBAC + NFDB)")
    fire_per_year, days_per_year = load_nbac_per_year(
        args.fire_label_npy, args.fire_label_json)

    print("\n[2] Annual FWI summary (sampled fire-season days)")
    fwi_per_year = sample_fwi_per_year(
        args.fwi_dir, args.start_year, args.end_year)

    # Combine into per-year table
    rows = []
    for year in range(args.start_year, args.end_year + 1):
        rec = dict(
            year=year,
            fire_pixels=fire_per_year.get(year, 0),
            fire_season_days=days_per_year.get(year, 0),
            fwi_mean=fwi_per_year.get(year, {}).get("fwi_mean", float('nan')),
            fwi_max=fwi_per_year.get(year, {}).get("fwi_max", float('nan')),
            high_fwi_frac=fwi_per_year.get(year, {}).get("high_fwi_frac", float('nan')),
            n_fwi_dates=fwi_per_year.get(year, {}).get("n_dates_sampled", 0),
        )
        rows.append(rec)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n  per-year CSV -> {args.out_csv}")

    # Statistics: trend tests + 2-period KS
    print("\n[3] Trend tests (Mann-Kendall, two-sided)")
    metrics = ["fire_pixels", "fwi_mean", "fwi_max", "high_fwi_frac"]
    trends = {}
    for m in metrics:
        vals = [r[m] for r in rows if not np.isnan(r[m]) and r[m] > 0]
        years = [r["year"] for r in rows if not np.isnan(r[m]) and r[m] > 0]
        mk = mann_kendall_simple(vals)
        if mk is None:
            continue
        trends[m] = mk
        print(f"  {m:20s}  Z={mk['Z']:+.2f}  p={mk['p']:.4f}  "
              f"trend={mk['trend']}  n={len(vals)}")

    print(f"\n[4] KS test: 2000..{args.split_year-1} vs {args.split_year}..2024")
    ks_results = {}
    for m in metrics:
        old = [r[m] for r in rows
               if r["year"] < args.split_year and not np.isnan(r[m]) and r[m] > 0]
        new = [r[m] for r in rows
               if r["year"] >= args.split_year and not np.isnan(r[m]) and r[m] > 0]
        if len(old) < 3 or len(new) < 3:
            continue
        ks = ks_test_simple(old, new)
        ks_results[m] = ks
        verdict = "DIFFERENT" if ks["p"] < 0.05 else "same"
        print(f"  {m:20s}  D={ks['D']:.3f}  p={ks['p']:.4f}  -> {verdict}")
        print(f"    old (n={len(old)}) mean={np.mean(old):.4g}  "
              f"new (n={len(new)}) mean={np.mean(new):.4g}  "
              f"ratio={np.mean(new)/max(np.mean(old), 1e-9):.2f}x")

    # Find similar years to 2024
    print(f"\n[5] Similarity to 2024 (Euclidean in normalized FWI features)")
    feat_cols = ["fwi_mean", "fwi_max", "high_fwi_frac"]
    valid_rows = [r for r in rows
                  if all(not np.isnan(r[c]) and r[c] > 0 for c in feat_cols)]
    if len(valid_rows) > 5:
        feat_arr = np.array([[r[c] for c in feat_cols] for r in valid_rows])
        means = feat_arr.mean(axis=0)
        stds = feat_arr.std(axis=0) + 1e-9
        feat_z = (feat_arr - means) / stds
        target_idx = next((i for i, r in enumerate(valid_rows) if r["year"] == 2024), None)
        if target_idx is not None:
            target = feat_z[target_idx]
            distances = np.linalg.norm(feat_z - target, axis=1)
            ranked = sorted(zip(distances, [r["year"] for r in valid_rows]))
            print("  Years most similar to 2024 by FWI fingerprint (lower=closer):")
            for d, y in ranked[:8]:
                if y == 2024:
                    continue
                print(f"    {y}: distance={d:.3f}")

    # Summary markdown
    md = ["# Climate-drift analysis (computed from project data)\n"]
    md.append("## Per-year fire + FWI table\n")
    md.append("| year | fire_pixels | FWI_mean | high_FWI_frac |")
    md.append("|---|---|---|---|")
    for r in rows:
        if r["fire_pixels"] > 0:
            md.append(f"| {r['year']} | {r['fire_pixels']:,} | "
                      f"{r['fwi_mean']:.2f} | {r['high_fwi_frac']:.3%} |")

    md.append("\n## Trend tests (Mann-Kendall)\n")
    md.append("| Metric | Z | p | Trend |")
    md.append("|---|---|---|---|")
    for m, t in trends.items():
        md.append(f"| {m} | {t['Z']:+.2f} | {t['p']:.4f} | {t['trend']} |")

    md.append(f"\n## KS test 2000-{args.split_year-1} vs {args.split_year}-2024\n")
    md.append("| Metric | D | p | Verdict |")
    md.append("|---|---|---|---|")
    for m, k in ks_results.items():
        verdict = "DIFFERENT (p<0.05)" if k["p"] < 0.05 else "same"
        md.append(f"| {m} | {k['D']:.3f} | {k['p']:.4f} | {verdict} |")

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))
    print(f"\n  markdown summary -> {args.out_md}")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
