#!/usr/bin/env python3
"""
Comprehensive statistical analysis of our NBAC + NFDB fire labels.

Validates claims I made in the writeup:
  - Mega-fires really contain thousands of pixels
  - Distribution is heavy-tailed (top X% accounts for Y% of area)
  - Fire activity is non-uniform across years and months

Computes:
  1. Per-fire stats from NBAC raw shapefile (each polygon = 1 fire event)
     - Total count, total area
     - Size distribution: percentiles, histogram (log-binned)
     - Pareto plot: cumulative area vs sorted fire rank
     - Top 20 largest fires (date, size, location)

  2. Per-year + per-month
     - Fire count per year (2000-2024)
     - Total burned area per year
     - Largest fire per year
     - Mega-fire count per year (>100k ha)
     - Per-month aggregation

  3. NFDB ignition-level stats
     - Total ignitions
     - Cause distribution (lightning vs human)
     - Per-year ignition count

  4. Quadrant analysis: size × time
     - Size buckets: tiny (<100 ha), small (100-1k), medium (1k-10k),
                     large (10k-100k), mega (>100k)
     - Time buckets: 4y-train (2018-2022), 12y-train (2014-2022),
                     22y-train (2000-2022), val (2022-2024)
     - Cross-tab: count + area per cell

  5. Pixel-level conversion check
     - For each fire size class, what's the equivalent pixel count
       at our 2km × 2km grid?
     - With dilation r=14, what's the effective dilated pixel area?
     - Validate the "thousands of pixels" claim quantitatively.

Outputs:
  outputs/fire_label_stats/
    fire_per_event.csv         per-fire (one row per NBAC polygon)
    fire_per_year.csv          year-level summary
    fire_per_month.csv
    fire_quadrants.csv         size × time cross-tab
    nfdb_per_year.csv
    summary.md                 human-readable headlines
"""
import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── helpers ───────────────────────────────────────────────────────────────

def parse_nbac_date(date_val):
    """NBAC AG_SDATE/EDATE can be string 'YYYYMMDD' or datetime; return date or None."""
    if date_val is None:
        return None
    if hasattr(date_val, 'year'):  # datetime-like
        return date(date_val.year, date_val.month, date_val.day)
    s = str(date_val).strip()
    if not s or s in ('nan', 'None', '0', '00000000'):
        return None
    try:
        if len(s) >= 8 and s[:8].isdigit():
            y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
            return date(y, m, d)
    except (ValueError, IndexError):
        pass
    return None


def size_bucket(ha):
    if ha < 100:
        return "tiny (<100 ha)"
    if ha < 1000:
        return "small (100-1k ha)"
    if ha < 10000:
        return "medium (1k-10k ha)"
    if ha < 100000:
        return "large (10k-100k ha)"
    return "mega (>100k ha)"


def time_bucket(yr):
    if yr < 2014:
        return "early (2000-2013)"
    if yr < 2018:
        return "mid (2014-2017)"
    if yr < 2022:
        return "late-train (2018-2021)"
    if yr < 2025:
        return "val (2022-2024)"
    return "future (2025+)"


# ── Section 1: NBAC per-fire ──────────────────────────────────────────────

def analyze_nbac(nbac_shapefile, exclude_prescribed=True):
    """Read NBAC shapefile and return list of per-fire dicts."""
    import geopandas as gpd
    print(f"\n[NBAC] reading {nbac_shapefile} ...")
    nbac = gpd.read_file(nbac_shapefile)
    print(f"  total rows: {len(nbac)}")
    print(f"  columns: {list(nbac.columns)[:15]}...")

    # Exclude prescribed burns if column exists
    if exclude_prescribed and "PRESCRIBED" in nbac.columns:
        before = len(nbac)
        mask = nbac["PRESCRIBED"].astype(str).str.lower().isin(["true", "1", "yes"])
        nbac = nbac[~mask].copy()
        print(f"  excluded {before - len(nbac)} prescribed burns")

    # Find size column
    size_col = None
    for c in ("POLY_HA", "AREA_HA", "POLYAREA_HA", "AG_HA"):
        if c in nbac.columns:
            size_col = c
            break
    if size_col is None:
        print(f"  [warn] no size column; computing from geometry")
        nbac_eq = nbac.to_crs("EPSG:3978")
        nbac["_area_ha"] = nbac_eq.geometry.area / 10000
        size_col = "_area_ha"
    print(f"  size column: {size_col}")

    # Find date column
    date_col = None
    for c in ("AG_SDATE", "AG_EDATE", "AFSDATE", "STARTDATE"):
        if c in nbac.columns:
            date_col = c
            break
    print(f"  date column: {date_col}")

    # Find jurisdiction column (Section 1: per-province)
    juris_col = None
    for c in ("AGENCY", "JURISDIC", "JURISDICT", "PROV", "JURISDICTION"):
        if c in nbac.columns:
            juris_col = c
            break
    print(f"  jurisdiction column: {juris_col}")

    rows = []
    for _, r in nbac.iterrows():
        ha = float(r[size_col]) if r[size_col] is not None else 0.0
        if ha <= 0:
            continue
        d = parse_nbac_date(r[date_col]) if date_col else None
        juris = str(r[juris_col]).strip() if juris_col and r[juris_col] is not None else "?"
        rows.append({
            "ha": ha,
            "year": d.year if d else None,
            "month": d.month if d else None,
            "jurisdiction": juris,
        })
    print(f"  parsed {len(rows)} valid fires")
    return rows


# ── Section 1: per-province ───────────────────────────────────────────────

# NBAC AGENCY codes → readable province names (per CFS documentation)
# https://cwfis.cfs.nrcan.gc.ca/datamart
AGENCY_MAP = {
    "BCA": "BC (British Columbia)",
    "AB":  "AB (Alberta)",
    "ABT": "AB (Alberta)",
    "SK":  "SK (Saskatchewan)",
    "SKA": "SK (Saskatchewan)",
    "MB":  "MB (Manitoba)",
    "MBA": "MB (Manitoba)",
    "ON":  "ON (Ontario)",
    "ONA": "ON (Ontario)",
    "QC":  "QC (Quebec)",
    "QCA": "QC (Quebec)",
    "NB":  "NB (New Brunswick)",
    "NBA": "NB (New Brunswick)",
    "NS":  "NS (Nova Scotia)",
    "NSA": "NS (Nova Scotia)",
    "PE":  "PE (PEI)",
    "PEA": "PE (PEI)",
    "NL":  "NL (Newfoundland)",
    "NLA": "NL (Newfoundland)",
    "YT":  "YT (Yukon)",
    "YTA": "YT (Yukon)",
    "NT":  "NT (NWT)",
    "NTA": "NT (NWT)",
    "NU":  "NU (Nunavut)",
    "NUA": "NU (Nunavut)",
    "PC":  "PC (Parks Canada)",
    "PCA": "PC (Parks Canada)",
}


def per_province(fires):
    by_juris = {}
    for f in fires:
        j = AGENCY_MAP.get(f["jurisdiction"], f["jurisdiction"])
        d = by_juris.setdefault(j, {"count": 0, "total_ha": 0.0,
                                     "max_ha": 0.0, "mega_count": 0})
        d["count"] += 1
        d["total_ha"] += f["ha"]
        if f["ha"] > d["max_ha"]:
            d["max_ha"] = f["ha"]
        if f["ha"] >= 100000:
            d["mega_count"] += 1
    return by_juris


# ── Section 3: Power-law fit ─────────────────────────────────────────────

def fit_powerlaw_mle(values, x_min=100.0):
    """Continuous power-law MLE: P(x) ~ x^(-α) for x >= x_min.
    Returns alpha estimate + standard error + KS goodness-of-fit."""
    arr = np.asarray(values, dtype=float)
    arr = arr[arr >= x_min]
    n = len(arr)
    if n < 50:
        return {"alpha": float('nan'), "sigma": float('nan'),
                "x_min": x_min, "n": n, "ks_d": float('nan')}
    alpha = 1 + n / np.sum(np.log(arr / x_min))
    sigma = (alpha - 1) / np.sqrt(n)
    # KS distance: empirical CDF vs theoretical power-law CDF
    sorted_arr = np.sort(arr)
    emp_cdf = np.arange(1, n + 1) / n
    theo_cdf = 1 - (sorted_arr / x_min) ** (1 - alpha)
    ks_d = float(np.max(np.abs(emp_cdf - theo_cdf)))
    return {"alpha": float(alpha), "sigma": float(sigma),
            "x_min": x_min, "n": n, "ks_d": ks_d}


# ── Section 4: Train vs Val KS test ───────────────────────────────────────

def ks_two_sample(a, b):
    """KS test on two samples. Returns D + asymptotic p-value."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return float('nan'), float('nan')
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    p = 2 * np.exp(-2 * (en * D) ** 2)
    return D, max(0.0, min(1.0, p))


def train_val_ks(fires, splits):
    """For each train range, KS test fire-size distribution vs val (2022-2024).
    splits = {'4y': (2018, 2021), '12y': (2014, 2021), '22y': (2000, 2021)}.
    Val = 2022-2024 always."""
    val_sizes = [f["ha"] for f in fires
                  if f["year"] is not None and 2022 <= f["year"] <= 2024]
    out = {}
    for name, (y0, y1) in splits.items():
        train_sizes = [f["ha"] for f in fires
                        if f["year"] is not None and y0 <= f["year"] <= y1]
        D, p = ks_two_sample(train_sizes, val_sizes)
        out[name] = {
            "n_train": len(train_sizes), "n_val": len(val_sizes),
            "train_mean": float(np.mean(train_sizes)) if train_sizes else 0,
            "val_mean": float(np.mean(val_sizes)) if val_sizes else 0,
            "D": D, "p": p,
            "verdict": "DIFFERENT (p<0.05)" if p < 0.05 else "same",
        }
    return out


def percentile_table(values, pcts=(50, 75, 90, 95, 99, 99.5, 99.9)):
    arr = np.asarray(values)
    return {p: float(np.percentile(arr, p)) for p in pcts}


def pareto_table(values):
    """For sorted-descending values, return cumulative area fraction at
    {1, 2, 5, 10, 20, 50}% top-fire-rank cutoffs."""
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    cum = np.cumsum(arr)
    total = cum[-1]
    out = {}
    for p in (1, 2, 5, 10, 20, 50):
        n = max(1, int(len(arr) * p / 100))
        out[f"top_{p}pct"] = {
            "n_fires": n,
            "fraction_of_total_area": float(cum[n - 1] / total),
        }
    return out


# ── Section 2: per-year / per-month ───────────────────────────────────────

def aggregate_per_year(fires):
    by_yr = {}
    for f in fires:
        if f["year"] is None:
            continue
        d = by_yr.setdefault(f["year"], {"count": 0, "total_ha": 0.0,
                                          "max_ha": 0.0, "mega_count": 0})
        d["count"] += 1
        d["total_ha"] += f["ha"]
        if f["ha"] > d["max_ha"]:
            d["max_ha"] = f["ha"]
        if f["ha"] >= 100000:
            d["mega_count"] += 1
    return by_yr


def aggregate_per_month(fires):
    by_m = {}
    for f in fires:
        if f["month"] is None:
            continue
        d = by_m.setdefault(f["month"], {"count": 0, "total_ha": 0.0})
        d["count"] += 1
        d["total_ha"] += f["ha"]
    return by_m


# ── Section 3: NFDB ───────────────────────────────────────────────────────

def analyze_nfdb(nfdb_csv, min_size_ha=1.0, exclude_prescribed=True):
    import pandas as pd
    print(f"\n[NFDB] reading {nfdb_csv}")
    df = pd.read_csv(nfdb_csv, low_memory=False)
    print(f"  total rows: {len(df)}")
    if "SIZE_HA" in df.columns:
        before = len(df)
        df = df[df["SIZE_HA"] >= min_size_ha]
        print(f"  filtered SIZE_HA >= {min_size_ha}: kept {len(df)} of {before}")
    if exclude_prescribed and "FIRECAUS" in df.columns:
        before = len(df)
        df = df[df["FIRECAUS"].astype(str).str.upper() != "PR"]
        print(f"  excluded prescribed (FIRECAUS=Pr): kept {len(df)} of {before}")
    out = []
    for _, r in df.iterrows():
        yr = None
        if "YEAR" in df.columns and r["YEAR"]:
            try:
                yr = int(r["YEAR"])
            except (ValueError, TypeError):
                pass
        ha = 0.0
        if "SIZE_HA" in df.columns:
            try:
                ha = float(r["SIZE_HA"])
            except (ValueError, TypeError):
                pass
        cause = str(r["FIRECAUS"]).upper() if "FIRECAUS" in df.columns else "?"
        out.append({"year": yr, "ha": ha, "cause": cause})
    return out


# ── Section 4: pixel-level conversion ─────────────────────────────────────

def pixel_conversion_check(fires, pixel_km=2.0, dilate_radius_px=14):
    """For each size bucket, compute equivalent raw + dilated pixel count."""
    pixel_ha = (pixel_km * 1000) ** 2 / 10000  # 2 km px = 400 ha
    dilation_factor = np.pi * dilate_radius_px ** 2  # ~615 px per source px
    print(f"\n[Pixel check] 1 px = {pixel_ha:.0f} ha (at 2 km)")
    print(f"  dilation r={dilate_radius_px} → ~{dilation_factor:.0f} px per source px")

    buckets = ["tiny (<100 ha)", "small (100-1k ha)", "medium (1k-10k ha)",
               "large (10k-100k ha)", "mega (>100k ha)"]
    by_bucket = {b: [] for b in buckets}
    for f in fires:
        by_bucket[size_bucket(f["ha"])].append(f["ha"])

    rows = []
    for b in buckets:
        sizes = by_bucket[b]
        if not sizes:
            continue
        median_ha = float(np.median(sizes))
        max_ha = float(max(sizes))
        median_raw_px = median_ha / pixel_ha
        max_raw_px = max_ha / pixel_ha
        rows.append({
            "size_bucket": b,
            "n_fires": len(sizes),
            "median_ha": median_ha,
            "max_ha": max_ha,
            "median_raw_px": median_raw_px,
            "max_raw_px": max_raw_px,
            "median_dilated_px": median_raw_px * dilation_factor,
            "max_dilated_px": max_raw_px * dilation_factor,
        })
    return rows


# ── Section 5: quadrant ───────────────────────────────────────────────────

def quadrant_analysis(fires):
    """size × time cross-tab."""
    out = {}
    for f in fires:
        if f["year"] is None:
            continue
        s = size_bucket(f["ha"])
        t = time_bucket(f["year"])
        cell = out.setdefault((s, t), {"count": 0, "total_ha": 0.0})
        cell["count"] += 1
        cell["total_ha"] += f["ha"]
    return out


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbac", default="data/burn_scars_raw/NBAC_1972to2024_shp.zip")
    ap.add_argument("--nfdb", default="data/burn_scars_raw/NFDB_point.csv")
    ap.add_argument("--output_dir", default="outputs/fire_label_stats")
    ap.add_argument("--start_year", type=int, default=2000)
    ap.add_argument("--end_year", type=int, default=2024)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    md = ["# Fire Label Distribution Analysis\n"]

    # ── NBAC ──────────────────────────────────────────────────────────
    nbac_fires = analyze_nbac(args.nbac)
    nbac_fires = [f for f in nbac_fires
                   if f["year"] is None or args.start_year <= f["year"] <= args.end_year]
    sizes = [f["ha"] for f in nbac_fires]
    print(f"\n[NBAC analysis] {len(nbac_fires)} fires in {args.start_year}-{args.end_year}")
    md.append(f"## 1. NBAC fire-event statistics ({args.start_year}-{args.end_year})\n")
    md.append(f"- Total NBAC fires: **{len(nbac_fires):,}**")
    md.append(f"- Total area burned: **{sum(sizes):,.0f} ha** "
              f"({sum(sizes)/1e6:.2f} M ha)")
    md.append(f"- Mean fire size: {np.mean(sizes):,.0f} ha")
    md.append(f"- Median fire size: {np.median(sizes):,.0f} ha")
    md.append(f"- Max fire size: {max(sizes):,.0f} ha\n")

    # Percentiles
    pcts = percentile_table(sizes)
    md.append("### Size percentiles (ha)\n")
    md.append("| Percentile | Size (ha) |")
    md.append("|---|---|")
    for p, v in pcts.items():
        md.append(f"| {p}% | {v:,.0f} |")
    md.append("")

    # Pareto
    pareto = pareto_table(sizes)
    md.append("### Pareto: top X% fires account for Y% of area\n")
    md.append("| Top % of fires | n_fires | % of total burned area |")
    md.append("|---|---|---|")
    for k, v in pareto.items():
        md.append(f"| {k} | {v['n_fires']:,} | "
                  f"{v['fraction_of_total_area']*100:.1f}% |")
    md.append("")

    # Per-year
    by_yr = aggregate_per_year(nbac_fires)
    md.append("### Per-year (NBAC)\n")
    md.append("| Year | n_fires | total_ha (M) | max_fire_ha | mega_count (>100k) |")
    md.append("|---|---|---|---|---|")
    rows_year = []
    for y in sorted(by_yr):
        d = by_yr[y]
        md.append(f"| {y} | {d['count']:,} | {d['total_ha']/1e6:.2f} | "
                  f"{d['max_ha']:,.0f} | {d['mega_count']} |")
        rows_year.append({"year": y, **d})
    md.append("")
    with open(os.path.join(args.output_dir, "fire_per_year.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=["year", "count", "total_ha", "max_ha", "mega_count"])
        wr.writeheader()
        for r in rows_year:
            wr.writerow(r)

    # Per-month
    by_m = aggregate_per_month(nbac_fires)
    md.append("### Per-month (NBAC, all years pooled)\n")
    md.append("| Month | n_fires | total_ha (M) |")
    md.append("|---|---|---|")
    for m in sorted(by_m):
        d = by_m[m]
        md.append(f"| {m} | {d['count']:,} | {d['total_ha']/1e6:.2f} |")
    md.append("")
    with open(os.path.join(args.output_dir, "fire_per_month.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=["month", "count", "total_ha"])
        wr.writeheader()
        for m in sorted(by_m):
            wr.writerow({"month": m, **by_m[m]})

    # Pixel check
    pixel_rows = pixel_conversion_check(nbac_fires)
    md.append("### Pixel-level conversion (validates 'mega-fire = thousands of pixels' claim)\n")
    md.append("| Bucket | n_fires | median_ha | max_ha | median_raw_px (2km) | "
              "median_dilated_px (r=14) | max_dilated_px |")
    md.append("|---|---|---|---|---|---|---|")
    for r in pixel_rows:
        md.append(f"| {r['size_bucket']} | {r['n_fires']:,} | "
                  f"{r['median_ha']:,.0f} | {r['max_ha']:,.0f} | "
                  f"{r['median_raw_px']:.1f} | {r['median_dilated_px']:.0f} | "
                  f"{r['max_dilated_px']:.0f} |")
    md.append("")
    with open(os.path.join(args.output_dir, "pixel_conversion.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(pixel_rows[0].keys()))
        wr.writeheader()
        for r in pixel_rows:
            wr.writerow(r)

    # ── Section 1: Per-province ───────────────────────────────────────
    prov = per_province(nbac_fires)
    md.append("### Per-province / per-jurisdiction (NBAC)\n")
    md.append("| Jurisdiction | n_fires | total_ha (M) | max_fire_ha | mega_count |")
    md.append("|---|---|---|---|---|")
    prov_rows = []
    for j in sorted(prov, key=lambda x: -prov[x]["total_ha"]):
        d = prov[j]
        md.append(f"| {j} | {d['count']:,} | {d['total_ha']/1e6:.2f} | "
                  f"{d['max_ha']:,.0f} | {d['mega_count']} |")
        prov_rows.append({"jurisdiction": j, **d})
    md.append("")
    with open(os.path.join(args.output_dir, "fire_per_province.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=["jurisdiction", "count", "total_ha", "max_ha", "mega_count"])
        wr.writeheader()
        for r in prov_rows:
            wr.writerow(r)

    # ── Section 3: Power-law fit ──────────────────────────────────────
    md.append("### Power-law (Pareto) fit on fire size distribution\n")
    md.append("Tests claim that fire sizes are heavy-tailed: P(size > x) ~ x^(-α).\n")
    md.append("| x_min (ha) | n fires used | α (exponent) | std err | KS-D vs power-law |")
    md.append("|---|---|---|---|---|")
    pl_rows = []
    for x_min in (10, 100, 1000, 10000):
        pl = fit_powerlaw_mle(sizes, x_min=x_min)
        md.append(f"| {x_min:,} | {pl['n']:,} | {pl['alpha']:.3f} | "
                  f"{pl['sigma']:.3f} | {pl['ks_d']:.3f} |")
        pl_rows.append({"x_min": x_min, **pl})
    md.append("\nInterpretation:")
    md.append("- α ≈ 1.5–2.5 = typical heavy-tailed wildfire distribution (power law)")
    md.append("- α > 3   = lighter tail (less extreme)")
    md.append("- KS-D < 0.05 indicates good fit to power-law\n")
    with open(os.path.join(args.output_dir, "fire_powerlaw_fit.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=["x_min", "alpha", "sigma", "n", "ks_d"])
        wr.writeheader()
        for r in pl_rows:
            wr.writerow(r)

    # ── Section 4: Train vs Val KS test ──────────────────────────────
    md.append("### Train vs Val distribution (KS test on fire sizes)\n")
    md.append("Tests whether the 4y/12y/22y train data and 2022-2024 val data come from the same distribution.\n")
    splits = {"4y (2018-2021)": (2018, 2021),
              "12y (2014-2021)": (2014, 2021),
              "22y (2000-2021)": (2000, 2021)}
    ks_results = train_val_ks(nbac_fires, splits)
    md.append("| Train range | n_train | n_val | mean_train (ha) | mean_val (ha) | KS-D | p-value | Verdict |")
    md.append("|---|---|---|---|---|---|---|---|")
    ks_rows = []
    for name, r in ks_results.items():
        md.append(f"| {name} | {r['n_train']:,} | {r['n_val']:,} | "
                  f"{r['train_mean']:,.0f} | {r['val_mean']:,.0f} | "
                  f"{r['D']:.3f} | {r['p']:.4f} | {r['verdict']} |")
        ks_rows.append({"train_range": name, **r})
    md.append("")
    with open(os.path.join(args.output_dir, "train_val_ks.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=["train_range", "n_train", "n_val",
                                            "train_mean", "val_mean", "D", "p", "verdict"])
        wr.writeheader()
        for r in ks_rows:
            wr.writerow(r)

    # Quadrant (existing)
    quad = quadrant_analysis(nbac_fires)
    md.append("### Quadrant: size × time\n")
    sizes_order = ["tiny (<100 ha)", "small (100-1k ha)", "medium (1k-10k ha)",
                   "large (10k-100k ha)", "mega (>100k ha)"]
    times_order = ["early (2000-2013)", "mid (2014-2017)",
                   "late-train (2018-2021)", "val (2022-2024)"]
    md.append("| size \\ time | " + " | ".join(times_order) + " |")
    md.append("|" + "---|" * (len(times_order) + 1))
    quad_rows = []
    for s in sizes_order:
        cells = []
        for t in times_order:
            d = quad.get((s, t), {"count": 0, "total_ha": 0})
            cells.append(f"{d['count']:,} fires / {d['total_ha']/1e6:.2f}M ha")
            quad_rows.append({"size": s, "time": t, "count": d["count"],
                               "total_ha": d["total_ha"]})
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md.append("")
    with open(os.path.join(args.output_dir, "fire_quadrants.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=["size", "time", "count", "total_ha"])
        wr.writeheader()
        for r in quad_rows:
            wr.writerow(r)

    # ── NFDB ──────────────────────────────────────────────────────────
    if os.path.exists(args.nfdb):
        nfdb_fires = analyze_nfdb(args.nfdb)
        nfdb_fires = [f for f in nfdb_fires
                       if f["year"] and args.start_year <= f["year"] <= args.end_year]
        md.append(f"## 2. NFDB ignition statistics ({args.start_year}-{args.end_year})\n")
        md.append(f"- Total NFDB ignitions (size>=1ha, excl prescribed): "
                  f"**{len(nfdb_fires):,}**")
        # Cause distribution
        causes = {}
        for f in nfdb_fires:
            causes[f["cause"]] = causes.get(f["cause"], 0) + 1
        md.append("\n### Cause distribution (NFDB)\n")
        for c in sorted(causes, key=lambda x: -causes[x])[:8]:
            md.append(f"- {c}: {causes[c]:,} ({causes[c]/len(nfdb_fires)*100:.1f}%)")
        md.append("")

    # ── Save markdown ──────────────────────────────────────────────────
    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"\n=== output ===")
    print(f"  {md_path}")
    print(f"  {args.output_dir}/*.csv")


if __name__ == "__main__":
    main()
