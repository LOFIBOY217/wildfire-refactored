#!/usr/bin/env python3
"""
Post-hoc stratified analysis of per-window val metrics.

Takes the JSON dump from `train_v3.py --save_per_window_json` and
breaks down Lift, BSS, etc. by:
    - year (of val window date)
    - month (fire season phase: early/mid/late)
    - fire-count bucket (small / medium / large val windows)

Optional: if --nbac is passed, also stratify by whether the val-window
primarily contained fires of certain sizes (requires spatial lookup).

Usage:
    python scripts/analyze_per_window_eval.py \\
        --input outputs/v3_9ch_enc21/per_window_val.json \\
        --out data/audit/per_window_analysis_enc21.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="per_window_val.json from train_v3 --save_per_window_json")
    ap.add_argument("--out", default=None,
                    help="Save CSV; default: <input>.stratified.csv")
    args = ap.parse_args()

    with open(args.input) as f:
        blob = json.load(f)

    per_win = blob["per_window"]
    df = pd.DataFrame(per_win)
    print(f"Loaded {len(df)} per-window records  "
          f"(total windows with fire)")
    print(f"Global summary: {blob['summary']}")

    # Parse date → year / month
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["fire_bucket"] = pd.cut(
        df["n_fire"],
        bins=[0, 10, 100, 1000, 10_000, np.inf],
        labels=["0-10", "10-100", "100-1k", "1k-10k", "10k+"])

    print("\n" + "=" * 70)
    print("PER-YEAR STRATIFICATION")
    print("=" * 70)
    cols = ["lift_k", "lift_coarse", "bss", "brier", "recall_k",
            "precision_k", "f2", "mcc"]
    yr_stats = df.groupby("year").agg({
        "n_fire": "sum",
        "baseline": "mean",
        **{c: "mean" for c in cols},
    })
    yr_stats["n_windows"] = df.groupby("year").size()
    print(yr_stats[["n_windows", "n_fire", "lift_k", "lift_coarse",
                    "bss", "recall_k"]].to_string(
                        float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("PER-MONTH (fire-season phase)")
    print("=" * 70)
    mo_stats = df.groupby("month").agg({
        "n_fire": "sum",
        **{c: "mean" for c in cols},
    })
    mo_stats["n_windows"] = df.groupby("month").size()
    print(mo_stats[["n_windows", "n_fire", "lift_k", "lift_coarse",
                    "bss"]].to_string(
                        float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("PER-FIRE-BUCKET (size of val window by fire pixel count)")
    print("=" * 70)
    fb_stats = df.groupby("fire_bucket", observed=True).agg({
        "n_fire": "sum",
        **{c: "mean" for c in cols},
    })
    fb_stats["n_windows"] = df.groupby("fire_bucket", observed=True).size()
    print(fb_stats[["n_windows", "n_fire", "lift_k", "lift_coarse",
                    "bss"]].to_string(
                        float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("TOP-5 BEST / WORST WINDOWS (by Lift@K)")
    print("=" * 70)
    df_sorted = df.sort_values("lift_k", ascending=False)
    print("\nBest 5:")
    print(df_sorted[["date", "n_fire", "lift_k", "lift_coarse",
                     "bss"]].head(5).to_string(
                         float_format=lambda x: f"{x:.3f}"))
    print("\nWorst 5:")
    print(df_sorted[["date", "n_fire", "lift_k", "lift_coarse",
                     "bss"]].tail(5).to_string(
                         float_format=lambda x: f"{x:.3f}"))

    # Save CSV (per-year)
    out = args.out or str(args.input).replace(".json", ".stratified.csv")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([
        yr_stats.assign(stratum_type="year"),
        mo_stats.assign(stratum_type="month"),
        fb_stats.assign(stratum_type="fire_bucket"),
    ])
    combined.to_csv(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
