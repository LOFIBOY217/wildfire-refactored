"""
Validate downloaded CWFIS hotspot data against NASA FIRMS source.

Two-part validation:

  PART A — Basic quality checks on the downloaded CSV (no API key needed):
    - Total record count
    - Date range coverage
    - Null checks
    - Coordinates within Canada bounding box
    - Per-year record counts (2023 should be the highest — big fire year)

  PART B — Cross-check against NASA FIRMS raw data (requires free MAP key):
    - Downloads a fixed validation week from NASA FIRMS VIIRS
    - Compares record count, spatial overlap, daily correlation

Get a free NASA FIRMS MAP key at:
    https://firms.modaps.eosdis.nasa.gov/api/area/

Usage:
    # Part A only
    python -m src.data_ops.validation.check_hotspot_data --config configs/default.yaml

    # Part A + B (with NASA comparison)
    python -m src.data_ops.validation.check_hotspot_data --config configs/default.yaml --map_key YOUR_KEY

    # Custom validation window
    python -m src.data_ops.validation.check_hotspot_data --map_key YOUR_KEY --val_start 2023-07-10 --val_end 2023-07-16
"""

import argparse
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

# Canada bounding box [west, south, east, north]
BBOX_CANADA = (-141.0, 41.7, -52.6, 83.1)

# Default validation window (2023 peak fire week)
DEFAULT_VAL_START = "2023-07-10"
DEFAULT_VAL_END   = "2023-07-16"

# NASA FIRMS API
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_DATASET = "VIIRS_SNPP_NRT"   # S-NPP VIIRS 375m


# ------------------------------------------------------------------ #
# PART A — Basic quality checks
# ------------------------------------------------------------------ #

def check_basic(df: pd.DataFrame, csv_path: str) -> bool:
    """
    Run basic quality checks on the hotspot DataFrame.

    Returns:
        True if all checks pass, False if any critical check fails.
    """
    print(f"\n{'='*55}")
    print(f"PART A — Basic Quality Checks")
    print(f"  File: {csv_path}")
    print(f"{'='*55}")

    all_ok = True

    # 1. Total record count
    n = len(df)
    min_expected = 500_000
    ok = n >= min_expected
    all_ok &= ok
    _print_check(
        "Total records",
        f"{n:,}",
        f">= {min_expected:,}",
        ok,
    )

    # 2. Required columns
    required = {"latitude", "longitude", "acq_date"}
    missing = required - set(df.columns)
    ok = len(missing) == 0
    all_ok &= ok
    _print_check("Required columns", str(set(df.columns) & required), "all present", ok)
    if not ok:
        print(f"  [FATAL] Missing columns: {missing}")
        return False

    # 3. Null values
    nulls = df[["latitude", "longitude", "acq_date"]].isnull().sum()
    ok = int(nulls.sum()) == 0
    all_ok &= ok
    _print_check("Null values", str(nulls.to_dict()), "0 nulls", ok)

    # 4. Date range
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    date_min = df["acq_date"].min()
    date_max = df["acq_date"].max()
    ok = (str(date_min) <= "2018-05-31") and (str(date_max) >= "2025-10-01")
    all_ok &= ok
    _print_check(
        "Date range",
        f"{date_min} → {date_max}",
        "2018-05 → 2025-10",
        ok,
    )

    # 5. Coordinates within Canada bounding box
    w, s, e, n_lat = BBOX_CANADA
    in_bbox = (
        (df["latitude"]  >= s) & (df["latitude"]  <= n_lat) &
        (df["longitude"] >= w) & (df["longitude"] <= e)
    )
    pct = in_bbox.mean() * 100
    ok = pct >= 99.0
    all_ok &= ok
    _print_check("In Canada bbox", f"{pct:.2f}%", ">= 99%", ok)

    # 6. Per-year record counts (2023 should be the peak fire year)
    df["year"] = pd.to_datetime(df["acq_date"]).dt.year
    yearly = df.groupby("year").size().sort_index()
    print(f"\n  Records per year:")
    peak_year = int(yearly.idxmax())
    for yr, cnt in yearly.items():
        marker = " ← peak" if yr == peak_year else ""
        print(f"    {yr}: {cnt:>10,}{marker}")

    ok = peak_year in (2023, 2024)   # 2023 was Canada's worst fire year on record
    all_ok &= ok
    _print_check(
        "Peak fire year",
        str(peak_year),
        "2023 or 2024 (historically bad years)",
        ok,
    )

    return all_ok


# ------------------------------------------------------------------ #
# PART B — Cross-check against NASA FIRMS
# ------------------------------------------------------------------ #

def _download_firms_sample(map_key: str, val_start: str, val_end: str) -> pd.DataFrame | None:
    """
    Download a validation-week sample from NASA FIRMS VIIRS for Canada.

    Args:
        map_key:   NASA FIRMS MAP key.
        val_start: Start date 'YYYY-MM-DD'.
        val_end:   End date 'YYYY-MM-DD'.

    Returns:
        DataFrame with columns [latitude, longitude, acq_date] or None on failure.
    """
    from datetime import date as _date
    start = _date.fromisoformat(val_start)
    end   = _date.fromisoformat(val_end)
    day_range = (end - start).days + 1

    # FIRMS area API: /csv/{key}/{dataset}/{west,south,east,north}/{days}/{date}
    w, s, e, n = BBOX_CANADA
    area = f"{w},{s},{e},{n}"
    url  = f"{FIRMS_API}/{map_key}/{FIRMS_DATASET}/{area}/{day_range}/{val_start}"

    print(f"  GET {url}")
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            if df.empty or "latitude" not in df.columns:
                print(f"  [WARN] Empty or unexpected response from FIRMS API")
                return None
            df = df[["latitude", "longitude", "acq_date"]].copy()
            df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
            print(f"  -> {len(df):,} records downloaded from NASA FIRMS")
            return df
        else:
            print(f"  [FAIL] HTTP {r.status_code}: {r.text[:300]}")
            return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def _spatial_overlap(df_nasa: pd.DataFrame, df_cwfis: pd.DataFrame,
                     threshold_km: float = 5.0) -> float:
    """
    For each NASA hotspot, find the nearest CWFIS hotspot.
    Return the fraction of NASA hotspots that have a CWFIS match within threshold_km.

    Uses a fast approximate grid-based approach (no scipy required).
    1 degree lat ≈ 111 km; 1 degree lon ≈ 111 * cos(lat) km.
    """
    if df_nasa.empty or df_cwfis.empty:
        return 0.0

    # Convert threshold to approximate degree distance at mean latitude
    mean_lat   = float(df_nasa["latitude"].mean())
    cos_lat    = np.cos(np.radians(mean_lat))
    thresh_deg = threshold_km / 111.0

    nasa_lats  = df_nasa["latitude"].values
    nasa_lons  = df_nasa["longitude"].values
    cwfis_lats = df_cwfis["latitude"].values
    cwfis_lons = df_cwfis["longitude"].values

    matched = 0
    # Process in chunks to avoid memory issues (don't build full N×M matrix)
    chunk_size = 5000
    for i in range(0, len(nasa_lats), chunk_size):
        lat_chunk = nasa_lats[i : i + chunk_size, np.newaxis]   # (C, 1)
        lon_chunk = nasa_lons[i : i + chunk_size, np.newaxis]

        dlat = np.abs(lat_chunk - cwfis_lats)                   # (C, M)
        dlon = np.abs(lon_chunk - cwfis_lons) * cos_lat

        dist_approx = np.sqrt(dlat**2 + dlon**2)
        matched += int((dist_approx.min(axis=1) <= thresh_deg).sum())

    return matched / len(nasa_lats)


def check_nasa_comparison(df_cwfis: pd.DataFrame, map_key: str,
                          val_start: str, val_end: str) -> bool:
    """
    Download NASA FIRMS sample and cross-check against CWFIS data.

    Returns:
        True if cross-check passes, False otherwise.
    """
    print(f"\n{'='*55}")
    print(f"PART B — NASA FIRMS Cross-Check ({val_start} to {val_end})")
    print(f"{'='*55}")

    # Download NASA sample
    print(f"\n  Downloading from NASA FIRMS ({FIRMS_DATASET})...")
    df_nasa = _download_firms_sample(map_key, val_start, val_end)
    if df_nasa is None:
        print("  [SKIP] Could not download NASA data — skipping Part B.")
        return False

    # Filter CWFIS to same period
    mask = (
        (df_cwfis["acq_date"] >= pd.to_datetime(val_start).date()) &
        (df_cwfis["acq_date"] <= pd.to_datetime(val_end).date())
    )
    df_cwfis_period = df_cwfis[mask].copy()
    print(f"  CWFIS records for same period: {len(df_cwfis_period):,}")

    all_ok = True

    # 1. Record count comparison
    n_nasa  = len(df_nasa)
    n_cwfis = len(df_cwfis_period)
    diff_pct = abs(n_nasa - n_cwfis) / max(n_nasa, 1) * 100
    ok = diff_pct <= 50.0   # Allow up to 50% difference (different processing)
    all_ok &= ok
    _print_check(
        "Record count",
        f"NASA={n_nasa:,}  CWFIS={n_cwfis:,}  diff={diff_pct:.1f}%",
        "within 50%",
        ok,
    )

    # 2. Spatial overlap
    print(f"\n  Computing spatial overlap (< 5km)... ", end="", flush=True)
    overlap = _spatial_overlap(df_nasa, df_cwfis_period, threshold_km=5.0)
    print(f"{overlap*100:.1f}%")
    ok = overlap >= 0.60
    all_ok &= ok
    _print_check("Spatial overlap", f"{overlap*100:.1f}%", ">= 60%", ok)

    # 3. Daily count correlation
    df_nasa["acq_date"]       = pd.to_datetime(df_nasa["acq_date"]).dt.date
    df_cwfis_period["acq_date"] = pd.to_datetime(df_cwfis_period["acq_date"]).dt.date

    daily_nasa  = df_nasa.groupby("acq_date").size()
    daily_cwfis = df_cwfis_period.groupby("acq_date").size()

    common_dates = daily_nasa.index.intersection(daily_cwfis.index)
    if len(common_dates) >= 3:
        r = float(np.corrcoef(
            daily_nasa[common_dates].values,
            daily_cwfis[common_dates].values,
        )[0, 1])
        ok = r >= 0.70
        all_ok &= ok
        _print_check("Daily count Pearson R", f"{r:.3f}", ">= 0.70", ok)
    else:
        print("  [SKIP] Not enough common dates for correlation check.")

    return all_ok


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #

def _print_check(label: str, value: str, expected: str, ok: bool) -> None:
    status = "✅" if ok else "❌"
    print(f"  {status} {label:30s}: {value}  (expected {expected})")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _build_parser():
    p = argparse.ArgumentParser(
        description="Validate downloaded CWFIS hotspot data against NASA FIRMS."
    )
    add_config_argument(p)

    p.add_argument("--hotspot_csv", type=str, default=None,
                   help="Path to hotspot CSV (default: from config hotspot_csv).")
    p.add_argument("--map_key", type=str, default=None,
                   help="NASA FIRMS MAP key for Part B comparison. "
                        "Get free key at: firms.modaps.eosdis.nasa.gov/api/area/")
    p.add_argument("--val_start", type=str, default=DEFAULT_VAL_START,
                   help=f"Validation window start (default: {DEFAULT_VAL_START}).")
    p.add_argument("--val_end",   type=str, default=DEFAULT_VAL_END,
                   help=f"Validation window end (default: {DEFAULT_VAL_END}).")
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Resolve CSV path
    if args.hotspot_csv:
        csv_path = args.hotspot_csv
    else:
        try:
            csv_path = get_path(cfg, "hotspot_csv")
        except Exception:
            csv_path = "data/hotspot/hotspot_2018_2025.csv"

    print("\n" + "="*55)
    print("Hotspot Data Validator")
    print("="*55)

    # Load CSV
    print(f"\nLoading {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        print("Run download_hotspots.py first.")
        sys.exit(1)

    # Part A
    part_a_ok = check_basic(df, csv_path)

    # Part B (optional, requires MAP key)
    part_b_ok = None
    if args.map_key:
        df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
        part_b_ok = check_nasa_comparison(
            df_cwfis=df,
            map_key=args.map_key,
            val_start=args.val_start,
            val_end=args.val_end,
        )
    else:
        print(f"\n  [INFO] Skipping Part B (no --map_key provided).")
        print(f"         Get a free MAP key at: https://firms.modaps.eosdis.nasa.gov/api/area/")

    # Final verdict
    print(f"\n{'='*55}")
    if part_b_ok is None:
        verdict = "PASS (Part A only)" if part_a_ok else "FAIL"
    else:
        verdict = "PASS" if (part_a_ok and part_b_ok) else "FAIL"

    symbol = "✅" if "PASS" in verdict else "❌"
    print(f"  {symbol} Overall result: {verdict}")
    print(f"{'='*55}\n")

    sys.exit(0 if "PASS" in verdict else 1)


if __name__ == "__main__":
    main()
