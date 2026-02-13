#!/usr/bin/env python3
"""
CIFFC Fire Data Quality Checker.

Standalone script that validates CIFFC wildfire data before training.
Checks field presence, missing values, date ranges, coordinate validity,
and provides train/test split recommendations.

Refactored from: data/check_data_quality.py
  - Added --config option to resolve ciffc_csv path from YAML config
  - Kept as mostly standalone (no heavy src.utils dependencies)

Usage:
  python -m src.data.validation.check_data_quality data/ciffc.csv
  python -m src.data.validation.check_data_quality --config configs/default.yaml
  python -m src.data.validation.check_data_quality  # auto-detects file
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import argparse

from src.config import load_config, get_path, add_config_argument


def check_data_quality(filepath):
    """
    Check CIFFC wildfire data quality.

    Args:
        filepath: Path to CSV or JSON data file.

    Returns:
        bool: True if data quality is acceptable for training.
    """
    print("=" * 70)
    print("CIFFC DATA QUALITY CHECK")
    print("=" * 70)

    # 1. Check file exists
    if not os.path.exists(filepath):
        print(f"[FAIL] File not found: {filepath}")
        return False

    print(f"[OK] File exists: {filepath}")
    print(f"     File size: {os.path.getsize(filepath) / 1024:.2f} KB")

    # 2. Read data
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'rows' in data:
                df = pd.DataFrame(data['rows'])
            else:
                df = pd.DataFrame(data)
        else:
            print("[FAIL] Unsupported file format (need .csv or .json)")
            return False

        print("[OK] Data loaded successfully")
    except Exception as e:
        print(f"[FAIL] Cannot read file: {str(e)}")
        return False

    # 3. Basic info
    print("\n" + "=" * 70)
    print("BASIC INFO")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")

    # 4. Required fields check
    print("\n" + "=" * 70)
    print("REQUIRED FIELDS")
    print("=" * 70)

    required_fields = {
        'field_situation_report_date': 'Date',
        'field_latitude': 'Latitude',
        'field_longitude': 'Longitude'
    }

    all_fields_present = True
    for field, description in required_fields.items():
        if field in df.columns:
            print(f"[OK]   {description} ({field}): present")
        else:
            print(f"[FAIL] {description} ({field}): missing")
            all_fields_present = False

    if not all_fields_present:
        print("\n  WARNING: Required fields missing, training notebook may fail")
        print("  Ensure CSV contains these columns:")
        for field in required_fields.keys():
            print(f"    - {field}")
        return False

    # 5. Missing values check
    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = df[list(required_fields.keys())].isnull().sum()
    if missing.sum() == 0:
        print("[OK] No missing values")
    else:
        print("WARNING: Missing values found:")
        for field, count in missing.items():
            if count > 0:
                print(f"  - {field}: {count} records ({count / len(df) * 100:.2f}%)")

    # 6. Date range check
    print("\n" + "=" * 70)
    print("DATE RANGE")
    print("=" * 70)

    try:
        df['date_parsed'] = pd.to_datetime(df['field_situation_report_date'])
        min_date = df['date_parsed'].min()
        max_date = df['date_parsed'].max()
        date_span = (max_date - min_date).days + 1
        unique_dates = df['date_parsed'].dt.date.nunique()

        print(f"Earliest date: {min_date.date()}")
        print(f"Latest date:   {max_date.date()}")
        print(f"Date span:     {date_span} days")
        print(f"Unique dates:  {unique_dates} days")
        print(f"Coverage:      {unique_dates / date_span * 100:.1f}%")

        # Check date continuity
        all_dates = pd.date_range(min_date, max_date, freq='D')
        missing_dates = set(all_dates.date) - set(df['date_parsed'].dt.date)

        if len(missing_dates) == 0:
            print("[OK] Dates are continuous, no gaps")
        else:
            print(f"WARNING: {len(missing_dates)} days have no fire records")
            if len(missing_dates) <= 10:
                print(f"  Missing dates: {sorted(missing_dates)}")

    except Exception as e:
        print(f"[FAIL] Date parsing error: {str(e)}")
        return False

    # 7. Coordinate range check
    print("\n" + "=" * 70)
    print("COORDINATE RANGE")
    print("=" * 70)

    lat_min, lat_max = df['field_latitude'].min(), df['field_latitude'].max()
    lon_min, lon_max = df['field_longitude'].min(), df['field_longitude'].max()

    print(f"Latitude range:  {lat_min:.4f} to {lat_max:.4f}")
    print(f"Longitude range: {lon_min:.4f} to {lon_max:.4f}")

    # Canada reasonable range check
    canada_lat_range = (41.7, 83.1)
    canada_lon_range = (-141.0, -52.6)

    lat_ok = canada_lat_range[0] <= lat_min and lat_max <= canada_lat_range[1]
    lon_ok = canada_lon_range[0] <= lon_min and lon_max <= canada_lon_range[1]

    if lat_ok and lon_ok:
        print("[OK] Coordinates within Canada range")
    else:
        print("WARNING: Some coordinates outside Canada range")
        if not lat_ok:
            print(f"  Latitude outside {canada_lat_range}")
        if not lon_ok:
            print(f"  Longitude outside {canada_lon_range}")

    # Check obviously invalid coordinates
    invalid_coords = (
        (df['field_latitude'].abs() > 90) |
        (df['field_longitude'].abs() > 180)
    )
    if invalid_coords.sum() > 0:
        print(f"[FAIL] Found {invalid_coords.sum()} invalid coordinates (outside Earth range)")

    # 8. Daily record statistics
    print("\n" + "=" * 70)
    print("DAILY RECORD STATS")
    print("=" * 70)

    daily_counts = df.groupby(df['date_parsed'].dt.date).size()

    print(f"Average per day: {daily_counts.mean():.1f} records")
    print(f"Median:          {daily_counts.median():.1f} records")
    print(f"Minimum:         {daily_counts.min()} records")
    print(f"Maximum:         {daily_counts.max()} records")
    print(f"Std dev:         {daily_counts.std():.1f}")

    # 9. Data volume assessment
    print("\n" + "=" * 70)
    print("DATA VOLUME ASSESSMENT")
    print("=" * 70)

    total_records = len(df)

    if total_records < 500:
        status = "[FAIL] Insufficient"
        recommendation = "Too few records, model may not train. Need at least 500."
    elif total_records < 2000:
        status = "[WARN] Low"
        recommendation = "Low record count, model performance may be limited. Recommend 2000+."
    elif total_records < 10000:
        status = "[OK]   Good"
        recommendation = "Sufficient data, ready to start training."
    else:
        status = "[OK]   Excellent"
        recommendation = "Ample data, expect good model performance."

    print(f"Total records: {total_records} - {status}")
    print(f"Recommendation: {recommendation}")

    # 10. Train/test split recommendation
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT RECOMMENDATION")
    print("=" * 70)

    # Calculate 80/20 split point
    split_date = min_date + timedelta(days=int(date_span * 0.8))
    train_records = len(df[df['date_parsed'] <= split_date])
    test_records = len(df[df['date_parsed'] > split_date])

    print(f"Recommended split date: {split_date.date()}")
    print(f"Train set: {min_date.date()} to {split_date.date()} ({train_records} records)")
    print(f"Test set:  {split_date.date() + timedelta(days=1)} to {max_date.date()} ({test_records} records)")

    # 11. Sample data
    print("\n" + "=" * 70)
    print("FIRST 5 RECORDS")
    print("=" * 70)
    print(df[list(required_fields.keys())].head().to_string())

    # 12. Final assessment
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    issues = []
    if not all_fields_present:
        issues.append("Required fields missing")
    if missing.sum() > 0:
        issues.append("Missing values found")
    if total_records < 500:
        issues.append("Insufficient data volume")
    if not (lat_ok and lon_ok):
        issues.append("Coordinate range anomalies")

    if len(issues) == 0:
        print("[OK] Data quality is good, ready for training!")
        print("\nNext steps:")
        print("1. Ensure FWI and ECMWF data are prepared")
        print("2. Update date configuration in notebook:")
        print(f"   DATA_START_DATE = date({min_date.year}, {min_date.month}, {min_date.day})")
        print(f"   TRAIN_END_DATE = date({split_date.year}, {split_date.month}, {split_date.day})")
        print("3. Run training notebook")
        return True
    else:
        print("WARNING: Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nRecommend resolving these issues before training.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Check CIFFC wildfire data quality',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_config_argument(parser)

    parser.add_argument(
        'filepath',
        nargs='?',
        default=None,
        help='Path to CIFFC data file (.csv or .json)'
    )

    args = parser.parse_args()

    # Determine file path
    if args.filepath:
        filepath = args.filepath
    else:
        # Try config
        cfg = load_config(args.config)
        try:
            filepath = get_path(cfg, 'ciffc_csv')
        except KeyError:
            filepath = None

        if filepath and os.path.exists(filepath):
            print(f"Using path from config: {filepath}\n")
        else:
            # Auto-detect default files
            default_files = [
                "ciffc_wildfires_training.csv",
                "ciffc_wildfires_training.json",
                "ciffc_wildfires_20251107.csv",
                "ciffc_wildfires_20251107.json"
            ]

            filepath = None
            for filename in default_files:
                if os.path.exists(filename):
                    print(f"Found data file: {filename}\n")
                    filepath = filename
                    break

            if filepath is None:
                print("No data file found.")
                print("Usage: python -m src.data.validation.check_data_quality <data_file_path>")
                print(f"\nSupported filenames: {', '.join(default_files)}")
                sys.exit(1)

    check_data_quality(filepath)


if __name__ == "__main__":
    main()
