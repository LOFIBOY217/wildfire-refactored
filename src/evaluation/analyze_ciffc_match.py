"""
analyze_ciffc_match.py
======================
Deep analysis of compare_ciffc_hotspot.py output, answering three questions:

  1. Overall detection rate: did the satellite see the fire?
  2. Error quantification: how large is the time gap? the spatial gap?
  3. Undetected profile: what do undetected fires have in common?

Usage:
    python -m src.evaluation.analyze_ciffc_match ^
        --match_csv ciffc_hotspot_match.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, total: float = 100.0, width: int = 25) -> str:
    """ASCII progress bar: maps value/total onto <width> characters."""
    filled = int(round(width * value / total)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    return f"{n:>7,}  ({100*n/total:>5.1f}%)" if total > 0 else f"{n:>7,}  ( N/A )"


def _section(title: str) -> None:
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _subsection(title: str) -> None:
    print()
    print(f"  ── {title}")
    print(f"  {'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis sections
# ─────────────────────────────────────────────────────────────────────────────

def section1_detection_rate(df: pd.DataFrame) -> None:
    """Section 1: Overall detection rate — did the satellite see the fire?"""
    _section("1. Overall Detection Rate")

    n          = len(df)
    detected   = df[~df["no_hotspots_in_window"]]   # hotspot found in window
    undetected = df[df["no_hotspots_in_window"]]     # no hotspot in window
    n_det  = len(detected)
    n_und  = len(undetected)

    print(f"\n  Total CIFFC records          : {n:>7,}")
    print(f"  ✅ Satellite detected (±7d)  : {_pct(n_det, n)}   {_bar(n_det, n)}")
    print(f"  ❌ No satellite detection    : {_pct(n_und, n)}   {_bar(n_und, n)}")

    same_day = (~df["same_day_nearest_km"].isna()).sum()
    print(f"\n  Of detected: same-day hotspot: {_pct(same_day, n)}")


def section2_time_error(df: pd.DataFrame) -> None:
    """Section 2: Time error — how many days apart are the satellite and human records?"""
    _section("2. Time Error  (records with satellite detection)")

    det     = df[~df["no_hotspots_in_window"]].copy()
    offsets = det["window_nearest_day_offset"].dropna()

    if len(offsets) == 0:
        print("\n  No data.")
        return

    print(f"\n  Sample size : {len(offsets):,} records (those with a hotspot in window)")
    print(f"\n  Offset = hotspot_date − CIFFC_date  (negative = satellite ahead of report)\n")

    print(f"  {'Offset':>8}  {'Count':>8}  {'Pct':>6}  Distribution")
    print(f"  {'─'*55}")
    for d in range(-7, 8):
        cnt = int((offsets == d).sum())
        pct = 100 * cnt / len(offsets)
        bar = _bar(pct, 20.0, 20)
        tag = "  ← same day" if d == 0 else ""
        print(f"  {d:>+7}d  {cnt:>8,}  {pct:>5.1f}%  {bar}{tag}")

    print(f"\n  Percentile summary:")
    for q, label in [(0, "Min"), (0.25, "P25"), (0.5, "Median"), (0.75, "P75"), (1.0, "Max")]:
        print(f"    {label:<8} : {offsets.quantile(q):>+.0f} days")


def section3_distance_error(df: pd.DataFrame) -> None:
    """Section 3: Distance error — grouped by fire size."""
    _section("3. Distance Error  (grouped by fire size)")

    det = df[~df["no_hotspots_in_window"]].copy()
    det = det[det["window_nearest_km"].notna()]

    if len(det) == 0:
        print("\n  No data.")
        return

    # Remove obvious outliers (coordinate errors)
    n_outlier = (det["window_nearest_km"] > 5000).sum()
    if n_outlier > 0:
        print(f"\n  ⚠  Removed {n_outlier} records with distance >5000 km (likely bad coordinates)")
        det = det[det["window_nearest_km"] <= 5000]

    print(f"\n  Note: larger fires naturally show greater distance between the CIFFC")
    print(f"  report point and the nearest satellite pixel (point vs. burn-area).\n")

    bins   = [0, 1, 10, 100, 1_000, 10_000, float("inf")]
    labels = ["Tiny  <1ha", "Small  1–10ha", "Medium 10–100ha",
              "Large 100–1000ha", "XLarge 1k–10kha", "Huge  >10kha"]

    if "field_fire_size" not in det.columns:
        print("  (field_fire_size column missing — skipping grouped analysis)")
        _print_dist(det["window_nearest_km"], "All records")
        return

    det["_size_grp"] = pd.cut(
        det["field_fire_size"].fillna(0).clip(lower=0),
        bins=bins, labels=labels, right=False
    )

    print(f"  {'Size group':<20} {'N':>6}  {'Median dist':>11}  "
          f"{'<10km':>6}  {'<50km':>7}  {'<100km':>7}")
    print(f"  {'─'*70}")

    for grp in labels:
        sub = det[det["_size_grp"] == grp]["window_nearest_km"]
        if len(sub) == 0:
            continue
        med   = sub.median()
        lt10  = 100 * (sub < 10).mean()
        lt50  = 100 * (sub < 50).mean()
        lt100 = 100 * (sub < 100).mean()
        print(f"  {grp:<20} {len(sub):>6,}  {med:>10.1f}km  "
              f"{lt10:>5.1f}%  {lt50:>6.1f}%  {lt100:>6.1f}%")

    print(f"\n  Overall distance percentiles (after outlier removal, n={len(det):,}):")
    for q, label in [(0,   "P0  (min)"),  (0.1,  "P10"),
                     (0.25,"P25"),         (0.5,  "P50 (median)"),
                     (0.75,"P75"),         (0.9,  "P90"),
                     (1.0, "P100 (max)")]:
        print(f"    {label:<14} : {det['window_nearest_km'].quantile(q):>8.2f} km")


def _print_dist(series: pd.Series, label: str) -> None:
    print(f"\n  {label} distance distribution:")
    for q, ql in [(0, "P0"), (0.25, "P25"), (0.5, "P50"), (0.75, "P75"), (1.0, "P100")]:
        print(f"    {ql}: {series.quantile(q):.2f} km")


def section4_undetected_profile(df: pd.DataFrame) -> None:
    """Section 4: Profile of undetected fires — what do they have in common?"""
    _section("4. Undetected Fire Profile")

    det   = df[~df["no_hotspots_in_window"]]
    und   = df[df["no_hotspots_in_window"]]
    n_und = len(und)
    n_det = len(det)

    if n_und == 0:
        print("\n  All records have satellite detection — nothing to analyse.")
        return

    print(f"\n  Analysing {n_und:,} records with no satellite detection in window\n")

    # ── 4a. Fire size ─────────────────────────────────────────────────
    _subsection("a. Fire size (hectares)")

    if "field_fire_size" in df.columns:
        und_sz = und["field_fire_size"].dropna()
        det_sz = det["field_fire_size"].dropna()

        print(f"  {'Statistic':<20} {'Undetected':>12}  {'Detected':>12}")
        print(f"  {'─'*48}")
        for label, q in [("Min", 0), ("P10", 0.1), ("Median", 0.5),
                          ("P90", 0.9), ("Max", 1.0)]:
            u = und_sz.quantile(q) if len(und_sz) > 0 else float("nan")
            d = det_sz.quantile(q) if len(det_sz) > 0 else float("nan")
            print(f"  {label:<20} {u:>12.2f}  {d:>12.2f}")

        u_tiny = 100 * (und_sz < 1).mean() if len(und_sz) > 0 else 0
        d_tiny = 100 * (det_sz < 1).mean() if len(det_sz) > 0 else 0
        print(f"\n  Size < 1 ha:  undetected {u_tiny:.1f}%  vs  detected {d_tiny:.1f}%")
        print(f"  → Small fires are below satellite detection threshold (~375m pixel)")

    # ── 4b. Control status ────────────────────────────────────────────
    _subsection("b. Stage of control (is the fire still burning?)")

    status_col = "field_stage_of_control_status"
    if status_col in df.columns:
        status_map = {
            "OUT": "OUT (extinguished)",
            "BH":  "BH  (being held)",
            "UC":  "UC  (out of control)",
            "OC":  "OC  (under control)",
            "H":   "H   (held)",
        }
        print(f"\n  {'Status':<22} {'Undet. n':>9}  {'Undet. %':>9}  "
              f"{'Det. n':>9}  {'Det. %':>8}")
        print(f"  {'─'*65}")

        for s in sorted(df[status_col].dropna().unique()):
            u   = (und[status_col] == s).sum()
            d   = (det[status_col] == s).sum()
            u_p = 100 * u / n_und if n_und > 0 else 0
            d_p = 100 * d / n_det if n_det > 0 else 0
            name = status_map.get(s, s)
            print(f"  {name:<22} {u:>9,}  {u_p:>8.1f}%  {d:>9,}  {d_p:>7.1f}%")

        out_und = 100 * (und[status_col] == "OUT").sum() / n_und if n_und > 0 else 0
        out_det = 100 * (det[status_col] == "OUT").sum() / n_det if n_det > 0 else 0
        print(f"\n  → OUT share: undetected {out_und:.1f}%  vs  detected {out_det:.1f}%")
        print(f"     OUT = already extinguished → no heat → satellite cannot see it (expected)")

    # ── 4c. Fire cause ────────────────────────────────────────────────
    _subsection("c. Fire cause")

    cause_col = "field_system_fire_cause"
    if cause_col in df.columns:
        cause_map = {
            "N": "N  (natural / lightning)",
            "H": "H  (human)",
            "U": "U  (unknown)",
        }
        print(f"\n  {'Cause':<26} {'Undetected %':>13}  {'Detected %':>11}")
        print(f"  {'─'*55}")
        for c in sorted(df[cause_col].dropna().unique()):
            u_p  = 100 * (und[cause_col] == c).sum() / n_und if n_und > 0 else 0
            d_p  = 100 * (det[cause_col] == c).sum() / n_det if n_det > 0 else 0
            name = cause_map.get(c, c)
            print(f"  {name:<26} {u_p:>12.1f}%  {d_p:>10.1f}%")

    # ── 4d. Province / agency ─────────────────────────────────────────
    _subsection("d. Province / agency  (top 10 by undetected rate)")

    agency_col = "field_agency_code"
    if agency_col in df.columns:
        grp = df.groupby(agency_col).agg(
            total = (agency_col, "count"),
            undet = ("no_hotspots_in_window", "sum"),
        )
        grp["undet_pct"] = 100 * grp["undet"] / grp["total"]
        grp = grp.sort_values("undet_pct", ascending=False)

        print(f"\n  {'Agency':>6}  {'Total':>8}  {'Undet.':>8}  {'Undet. %':>9}")
        print(f"  {'─'*40}")
        for agency, row in grp.head(10).iterrows():
            print(f"  {str(agency):>6}  {int(row['total']):>8,}  "
                  f"{int(row['undet']):>8,}  {row['undet_pct']:>8.1f}%")

    # ── 4e. Monthly distribution ──────────────────────────────────────
    _subsection("e. Monthly distribution  (which months have the most undetected?)")

    if "ciffc_date" in und.columns or "field_situation_report_date" in und.columns:
        und_copy = und.copy()
        # Prefer the original timestamp column (reliable format); fall back to ciffc_date
        date_col = ("field_situation_report_date"
                    if "field_situation_report_date" in und_copy.columns
                    else "ciffc_date")
        und_copy["_month"] = pd.to_datetime(
            und_copy[date_col], errors="coerce"
        ).dt.month.astype("Int64")   # nullable Int64 so NaN rows don't corrupt dtype
        month_counts = und_copy.dropna(subset=["_month"]).groupby("_month").size()
        month_names  = {5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct"}
        total_und    = month_counts.sum()

        print(f"\n  {'Month':<6} {'Count':>7}  {'Pct':>6}  Distribution")
        print(f"  {'─'*45}")
        for m in range(5, 11):
            cnt = int(month_counts.get(m, month_counts.get(np.int64(m), 0)))
            pct = 100 * cnt / total_und if total_und > 0 else 0
            bar = _bar(pct, 30.0, 20)
            print(f"  {month_names.get(m, str(m)):<6} {cnt:>7,}  {pct:>5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Analyse CIFFC vs hotspot match results (output of compare_ciffc_hotspot.py)"
    )
    ap.add_argument("--match_csv", type=str, default="ciffc_hotspot_match.csv",
                    help="CSV produced by compare_ciffc_hotspot.py (default: ciffc_hotspot_match.csv)")
    ap.add_argument("--window_days", type=int, default=7,
                    help="Time window used when running the match (display only, default: 7)")
    args = ap.parse_args(argv)

    csv_path = args.match_csv
    if not Path(csv_path).exists():
        sys.exit(f"[ERROR] File not found: {csv_path}\n"
                 f"Run compare_ciffc_hotspot.py first to generate the match file.")

    print(f"\n{'='*65}")
    print(f"  CIFFC vs Hotspot — Deep Analysis Report")
    print(f"  Input : {csv_path}")
    print(f"  Window: ±{args.window_days} days")
    print(f"{'='*65}")

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded {len(df):,} records")

    section1_detection_rate(df)
    section2_time_error(df)
    section3_distance_error(df)
    section4_undetected_profile(df)

    print(f"\n{'='*65}")
    print(f"  Analysis complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
