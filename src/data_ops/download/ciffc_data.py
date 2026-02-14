#!/usr/bin/env python3
"""
CIFFC fire data downloader

Downloads historical fire data from Canadian sources for training fire
prediction models.

Data sources (tried in order):
    1. CWFIS (Canadian Wildland Fire Information System) -- recommended
    2. NASA MODIS Active Fire archive (no API key required)
    3. Sample/test data generation (for development only)

NASA FIRMS is also supported but requires a separate API key registration.

Usage:
    python -m src.data_ops.download.ciffc_data
    python -m src.data_ops.download.ciffc_data --start 2025-05-01 --end 2025-10-31
    python -m src.data_ops.download.ciffc_data --sample
    python -m src.data_ops.download.ciffc_data --config configs/custom.yaml
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

# Bounding box for Canada: [min_lon, min_lat, max_lon, max_lat]
BBOX_CANADA = [-141.0, 41.7, -52.6, 83.1]

# API endpoints
CWFIS_API = "https://cwfis.cfs.nrcan.gc.ca/geoserver/cwfis/ows"
NASA_FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"


# ------------------------------------------------------------------ #
# Download functions
# ------------------------------------------------------------------ #

def download_cwfis_data(start_date, end_date, bbox=None):
    """
    Download fire data from CWFIS.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]

    Returns:
        pandas.DataFrame with fire records, or None on failure.
    """
    if bbox is None:
        bbox = BBOX_CANADA

    print("=" * 60)
    print("Method 1: Downloading from CWFIS...")
    print("=" * 60)

    # WFS (Web Feature Service) request parameters
    params = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeName': 'cwfis:hotspots',
        'outputFormat': 'json',
        'srsName': 'EPSG:4326',
        'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
    }

    # CQL filter for time range
    cql_filter = f"rep_date >= '{start_date}' AND rep_date <= '{end_date}'"
    params['CQL_FILTER'] = cql_filter

    try:
        print(f"Request URL: {CWFIS_API}")
        print(f"Date range:  {start_date} to {end_date}")
        print(f"Bounding box: {bbox}")
        print("\nDownloading... (may take a few minutes)")

        response = requests.get(CWFIS_API, params=params, timeout=300)
        response.raise_for_status()

        data = response.json()

        if 'features' not in data or len(data['features']) == 0:
            print("WARNING: CWFIS returned empty data")
            return None

        # Parse GeoJSON
        records = []
        for feature in data['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates']

            record = {
                'field_situation_report_date': props.get('rep_date', ''),
                'field_latitude': coords[1],
                'field_longitude': coords[0],
                'fire_id': props.get('fire_id', ''),
                'agency': props.get('agency', ''),
                'fire_size_ha': props.get('size_ha', 0),
                'fire_type': props.get('fire_type', ''),
                'source': 'CWFIS',
            }
            records.append(record)

        df = pd.DataFrame(records)
        print(f"[OK] Downloaded {len(df)} CWFIS fire records")
        return df

    except Exception as e:
        print(f"[FAIL] CWFIS download failed: {e}")
        return None


def download_nasa_firms_data(start_date, end_date, country="CAN"):
    """
    Print instructions for downloading from NASA FIRMS (requires API key).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        country: Country code (CAN = Canada)

    Returns:
        None (informational only).
    """
    print("\n" + "=" * 60)
    print("Method 2: NASA FIRMS (requires API key)...")
    print("=" * 60)

    print("WARNING: NASA FIRMS requires an API key.")
    print("Visit: https://firms.modaps.eosdis.nasa.gov/api/")
    print("1. Register an account")
    print("2. Obtain your MAP_KEY")
    print("3. Download using:")
    print(f"\n  MAP_KEY='your_key_here'")
    print(f"  URL='https://firms.modaps.eosdis.nasa.gov/api/country/csv/"
          f"$MAP_KEY/VIIRS_SNPP_NRT/{country}/1/{start_date}'")
    print(f"  curl -o firms_data.csv $URL\n")

    return None


def download_modis_active_fire(start_date, end_date, bbox=None):
    """
    Download active fire data from NASA MODIS Collection 6.1 archive.
    No API key required but data granularity may be coarser.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]

    Returns:
        pandas.DataFrame with fire records, or None on failure.
    """
    if bbox is None:
        bbox = BBOX_CANADA

    print("\n" + "=" * 60)
    print("Method 3: Downloading from NASA MODIS Archive...")
    print("=" * 60)

    base_url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_records = []
    current = start

    while current <= end:
        filename = f"MODIS_C6_1_Global_24h_{current.strftime('%Y_%m_%d')}.csv"
        url = f"{base_url}{filename}"

        try:
            print(f"Downloading {current.strftime('%Y-%m-%d')}...", end=" ")
            response = requests.get(url, timeout=60)

            if response.status_code == 200:
                df_day = pd.read_csv(StringIO(response.text))

                # Filter to Canada bounding box
                df_canada = df_day[
                    (df_day['latitude'] >= bbox[1]) &
                    (df_day['latitude'] <= bbox[3]) &
                    (df_day['longitude'] >= bbox[0]) &
                    (df_day['longitude'] <= bbox[2])
                ]

                if len(df_canada) > 0:
                    all_records.append(df_canada)
                    print(f"[OK] {len(df_canada)} records")
                else:
                    print("no data")
            else:
                print(f"[FAIL] HTTP {response.status_code}")

        except Exception as e:
            print(f"[FAIL] {e}")

        current += timedelta(days=1)
        time.sleep(0.5)  # Rate limiting

    if len(all_records) == 0:
        print("WARNING: No data downloaded")
        return None

    # Merge all daily data
    df = pd.concat(all_records, ignore_index=True)

    # Convert to CIFFC format
    df_formatted = pd.DataFrame({
        'field_situation_report_date': df['acq_date'],
        'field_latitude': df['latitude'],
        'field_longitude': df['longitude'],
        'fire_id': (
            df['latitude'].astype(str) + '_'
            + df['longitude'].astype(str) + '_'
            + df['acq_date'].astype(str)
        ),
        'confidence': df['confidence'],
        'brightness': df['bright_ti4'],
        'frp': df['frp'],  # Fire Radiative Power
        'source': 'MODIS',
    })

    print(f"\n[OK] Downloaded {len(df_formatted)} MODIS fire records total")
    return df_formatted


def create_sample_data(start_date, end_date, n_samples=1000):
    """
    Create random sample data for testing (NOT real fire data).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        n_samples: Number of random records to generate

    Returns:
        pandas.DataFrame with fake fire records.
    """
    import numpy as np

    print("\n" + "=" * 60)
    print("Method 4: Generating sample data (testing only)")
    print("=" * 60)
    print("WARNING: This is randomly generated fake data, not suitable for training!")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1

    records = []
    for idx in range(n_samples):
        random_days = np.random.randint(0, days)
        fire_date = start + timedelta(days=int(random_days))

        lat = np.random.uniform(49.0, 70.0)   # Fire-prone latitudes
        lon = np.random.uniform(-130.0, -60.0)

        record = {
            'field_situation_report_date': fire_date.strftime('%Y-%m-%d'),
            'field_latitude': lat,
            'field_longitude': lon,
            'fire_id': f'TEST_{fire_date.strftime("%Y%m%d")}_{idx}',
            'source': 'SAMPLE',
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} sample records")
    return df


# ------------------------------------------------------------------ #
# Save helpers
# ------------------------------------------------------------------ #

def save_data(df, csv_path, json_path):
    """
    Save DataFrame to CSV and JSON.

    Args:
        df: pandas.DataFrame with fire records
        csv_path: Output CSV path
        json_path: Output JSON path

    Returns:
        True on success, False if no data.
    """
    if df is None or len(df) == 0:
        print("[FAIL] No data to save")
        return False

    # Ensure parent directories exist
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] CSV saved to: {csv_path}")

    # Save JSON
    json_data = {
        'metadata': {
            'total_records': len(df),
            'date_range': {
                'start': str(df['field_situation_report_date'].min()),
                'end': str(df['field_situation_report_date'].max()),
            },
            'generated_at': datetime.now().isoformat(),
        },
        'rows': df.to_dict('records'),
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] JSON saved to: {json_path}")

    # Statistics
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"Total records:  {len(df)}")
    print(f"Date range:     {df['field_situation_report_date'].min()} "
          f"to {df['field_situation_report_date'].max()}")
    print(f"Unique dates:   {df['field_situation_report_date'].nunique()}")

    if 'source' in df.columns:
        print(f"\nSource distribution:")
        print(df['source'].value_counts().to_string())

    print(f"\nFirst 5 records:")
    print(df.head().to_string())

    return True


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Download CIFFC fire data from CWFIS / MODIS / sample"
    )
    add_config_argument(parser)

    parser.add_argument(
        "--start", type=str, default="2025-01-18",
        help="Start date YYYY-MM-DD (default: 2025-01-18)",
    )
    parser.add_argument(
        "--end", type=str, default="2025-10-31",
        help="End date YYYY-MM-DD (default: 2025-10-31)",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Output CSV path (default: from config, ciffc_csv)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Output JSON path (default: derived from CSV path)",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate sample data for testing instead of downloading",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of sample records to generate (default: 1000)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Load config ----
    cfg = load_config(args.config)

    # ---- Resolve output paths ----
    if args.output_csv:
        csv_path = args.output_csv
    else:
        csv_path = get_path(cfg, "ciffc_csv")

    if args.output_json:
        json_path = args.output_json
    else:
        json_path = str(Path(csv_path).with_suffix('.json'))

    start_date = args.start
    end_date = args.end

    print("\n" + "=" * 60)
    print("CIFFC Fire Data Downloader")
    print("=" * 60)
    print(f"  Date range:  {start_date} to {end_date}")
    print(f"  Output CSV:  {csv_path}")
    print(f"  Output JSON: {json_path}")

    df = None

    # ---- Sample mode ----
    if args.sample:
        df = create_sample_data(start_date, end_date, n_samples=args.n_samples)
        save_data(df, csv_path, json_path)
        return

    # ---- Method 1: CWFIS (recommended) ----
    try:
        df = download_cwfis_data(start_date, end_date)
        if df is not None and len(df) > 0:
            save_data(df, csv_path, json_path)
            return
    except Exception as e:
        print(f"CWFIS method failed: {e}")

    # ---- Method 2: NASA FIRMS (informational only) ----
    print("\nCWFIS failed, trying other methods...")
    download_nasa_firms_data(start_date, end_date)

    # ---- Method 3: MODIS Archive (no API key) ----
    try:
        df = download_modis_active_fire(start_date, end_date)
        if df is not None and len(df) > 0:
            save_data(df, csv_path, json_path)
            return
    except Exception as e:
        print(f"MODIS method failed: {e}")

    # ---- All methods failed ----
    print("\n" + "=" * 60)
    print("All download methods failed")
    print("=" * 60)
    print("\nTo generate sample data for testing, re-run with --sample flag:")
    print(f"  python -m src.data_ops.download.ciffc_data --sample "
          f"--start {start_date} --end {end_date}")
    print("\nOtherwise, try:")
    print("1. Check your network connection")
    print("2. Visit https://cwfis.cfs.nrcan.gc.ca/ to download manually")
    print("3. Register for NASA FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/")


if __name__ == "__main__":
    main()
