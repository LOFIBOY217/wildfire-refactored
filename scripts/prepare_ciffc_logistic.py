#!/usr/bin/env python3
"""
Merge and preprocess CIFFC files for logistic training.

Output filename includes "logistic" and the final date range:
    ciffc_logistic_YYYYMMDD_YYYYMMDD.csv
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Prepare merged CIFFC file for logistic training")
    parser.add_argument("--input-dir", default="data/ciffc", help="Directory containing CIFFC CSV files")
    parser.add_argument("--output-dir", default="data/ciffc", help="Directory for merged output")
    parser.add_argument(
        "--min-date",
        default="2018-01-01",
        help="Earliest acceptable report date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-date",
        default=None,
        help="Latest acceptable report date (YYYY-MM-DD). Default: today (UTC).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    min_dt = pd.Timestamp(args.min_date, tz="UTC")
    max_dt = (
        pd.Timestamp(args.max_date, tz="UTC")
        if args.max_date
        else pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")
    )

    frames = []
    for f in files:
        df = pd.read_csv(f)
        # Handle BOM if present.
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    before_rows = len(merged)

    required = ["field_situation_report_date", "field_longitude", "field_latitude"]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    merged["report_dt"] = pd.to_datetime(merged["field_situation_report_date"], errors="coerce", utc=True)
    merged = merged[merged["report_dt"].notna()].copy()

    # Keep only plausible date range for training and remove future outliers.
    merged = merged[(merged["report_dt"] >= min_dt) & (merged["report_dt"] <= max_dt)].copy()

    # Basic coordinate sanity filter.
    merged["field_longitude"] = pd.to_numeric(merged["field_longitude"], errors="coerce")
    merged["field_latitude"] = pd.to_numeric(merged["field_latitude"], errors="coerce")
    merged = merged[
        merged["field_longitude"].between(-180, 180, inclusive="both")
        & merged["field_latitude"].between(-90, 90, inclusive="both")
    ].copy()

    # De-duplicate exact repeated records.
    dedup_keys = [
        "field_agency_fire_id",
        "field_situation_report_date",
        "field_longitude",
        "field_latitude",
    ]
    dedup_keys = [k for k in dedup_keys if k in merged.columns]
    if dedup_keys:
        merged = merged.drop_duplicates(subset=dedup_keys)

    if merged.empty:
        raise SystemExit("Merged dataset is empty after preprocessing.")

    start = merged["report_dt"].min().date()
    end = merged["report_dt"].max().date()
    out_name = f"ciffc_logistic_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    out_path = output_dir / out_name

    merged = merged.drop(columns=["report_dt"])
    merged.to_csv(out_path, index=False)

    print(f"Input files: {len(files)}")
    print(f"Rows before: {before_rows}")
    print(f"Rows after:  {len(merged)}")
    print(f"Date range:  {start} -> {end}")
    print(f"Output:      {out_path}")


if __name__ == "__main__":
    main()
