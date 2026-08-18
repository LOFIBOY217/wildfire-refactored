"""
compare_ciffc_hotspot.py
========================
For every fire record in CIFFC, find the closest spatio-temporal match in the
CWFIS satellite hotspot data and quantify the spatial distance error between
the two.

Data sources
------------
CIFFC (Canadian Interagency Forest Fire Centre):
    Manually reported fire status records. Each record = one status update
    (a single lat/lon point stands for the whole fire). Rich attributes
    including fire size (hectares), control status, and fire cause, with
    timestamps precise to the second.

Hotspot (CWFIS / VIIRS satellite thermal pixels):
    Satellite-detected surface thermal-anomaly pixels (~375 m resolution).
    Each record = one detected pixel, with coordinates + date only, no area
    or status information. Far more numerous per day than CIFFC
    (~3,000/day vs ~26/day).

Key differences:
    1. Granularity  CIFFC row = a whole fire; hotspot row = one 375 m pixel
    2. Area         CIFFC has field_fire_size (ha); hotspot has none
    3. Time precision  CIFFC is second-precise; hotspot is date-only
    4. Count        Hotspot outnumbers CIFFC by ~100x
    5. Origin       CIFFC is manual; hotspot is automated satellite detection

Usage
-----
# Explicit paths
python -m src.evaluation.compare_ciffc_hotspot \\
    --ciffc_csv    path/to/ciffc.csv \\
    --hotspot_csv  path/to/hotspot.csv \\
    --output_csv   ciffc_hotspot_match.csv \\
    --window_days  7 \\
    --match_km     10

# Using config (reads paths.ciffc_csv / paths.hotspot_csv)
python -m src.evaluation.compare_ciffc_hotspot \\
    --config configs/default.yaml \\
    --window_days 7 --match_km 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project-internal helpers (optional; the script also works without config,
#    taking paths directly from the CLI) ──────────────────────────────────────
try:
    from src.config import add_config_argument, get_path, load_config
    from src.data_ops.processing.rasterize_fires import load_ciffc_data
    _HAS_PROJECT = True
except ImportError:
    _HAS_PROJECT = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(
    lat1: float,
    lon1: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Haversine great-circle distance in km from one point (lat1, lon1) to an
    array of points (lats, lons).

    Parameters
    ----------
    lat1, lon1 : float
        Reference point (CIFFC fire location).
    lats, lons : np.ndarray  shape (N,)
        Candidate hotspot coordinate arrays.

    Returns
    -------
    np.ndarray  shape (N,)  distance (km) from each candidate to the reference.
    """
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _load_hotspot_raw(hotspot_path: str) -> pd.DataFrame:
    """
    Load the hotspot CSV, keeping the original column names.
    Returns columns: latitude, longitude, acq_date (date object).
    """
    df = pd.read_csv(hotspot_path)
    df = df.dropna(subset=["acq_date"])
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    return df


def _load_ciffc_raw(ciffc_path: str) -> pd.DataFrame:
    """
    Load the CIFFC CSV and parse the date column.
    Reuse load_ciffc_data() if the project environment is available, otherwise
    parse it directly.
    """
    if _HAS_PROJECT:
        return load_ciffc_data(ciffc_path)
    df = pd.read_csv(ciffc_path)
    df["date"] = pd.to_datetime(df["field_situation_report_date"]).dt.date
    return df


def _build_hotspot_index(
    hotspot_df: pd.DataFrame,
) -> dict[date, np.ndarray]:
    """
    Pre-build the hotspot DataFrame into a date-indexed dictionary.

    Returns
    -------
    dict  {date -> np.ndarray shape (N, 2)}   each element = [[lat, lon], ...]
    """
    print("  Building hotspot date index (one-time)...", flush=True)
    idx: dict[date, np.ndarray] = {}
    for d, g in hotspot_df.groupby("acq_date"):
        idx[d] = g[["latitude", "longitude"]].values.astype(np.float64)
    print(f"  Index built: {len(idx):,} distinct dates", flush=True)
    return idx


def _print_data_summary(ciffc_df: pd.DataFrame, hotspot_df: pd.DataFrame) -> None:
    """Print a content-comparison summary of the two data sources."""
    cif_dates = pd.to_datetime(ciffc_df["field_situation_report_date"])
    hot_dates = hotspot_df["acq_date"]

    print()
    print("=" * 65)
    print("  Data source comparison: CIFFC vs CWFIS Hotspot")
    print("=" * 65)

    print()
    print("[CIFFC - manually reported fire records]")
    print(f"  Records    : {len(ciffc_df):>10,}")
    print(f"  Date range : {cif_dates.min().date()} -> {cif_dates.max().date()}")
    print(f"  Columns(12): field_agency_fire_id, field_agency_code,")
    print(f"               field_situation_report_date (second-precise),")
    print(f"               field_stage_of_control_status, field_system_fire_cause,")
    print(f"               field_response_type,")
    print(f"               field_fire_size (hectares), field_latitude, field_longitude")
    if "field_fire_size" in ciffc_df.columns:
        sz = ciffc_df["field_fire_size"].dropna()
        print(f"  fire_size  : min={sz.min():.1f} ha, median={sz.median():.1f} ha, "
              f"max={sz.max():,.0f} ha")
    status_str = ", ".join(
        f"{k}:{v}"
        for k, v in ciffc_df["field_stage_of_control_status"].value_counts().items()
    ) if "field_stage_of_control_status" in ciffc_df.columns else "N/A"
    print(f"  Status     : {status_str}")
    print(f"  Row meaning: one fire's status report on a given day; a single "
          f"lat/lon stands for the whole fire")

    print()
    print("[Hotspot - CWFIS satellite thermal pixels]")
    print(f"  Records    : {len(hotspot_df):>10,}")
    print(f"  Date range : {hot_dates.min()} -> {hot_dates.max()}")
    print(f"  Columns(3) : latitude, longitude, acq_date (date only, no time)")
    lat_r = f"{hotspot_df['latitude'].min():.2f} -> {hotspot_df['latitude'].max():.2f}"
    lon_r = f"{hotspot_df['longitude'].min():.2f} -> {hotspot_df['longitude'].max():.2f}"
    print(f"  lat range  : {lat_r}")
    print(f"  lon range  : {lon_r}")
    print(f"  Row meaning: a single thermal pixel detected by satellite "
          f"(VIIRS/MODIS ~375 m)")

    print()
    print("[Key differences]")
    rows = [
        ("Source",      "Manual report (prov./federal agency)", "Automated satellite (CWFIS/VIIRS)"),
        ("Per record",  "One status report for a whole fire",   "One 375 m satellite thermal pixel"),
        ("Area info",   "Yes (field_fire_size, hectares)",      "No"),
        ("Status attr", "Yes (OUT/UC/OC/BH/H)",                 "No"),
        ("Time prec.",  "Second (ISO 8601 timestamp)",          "Day (YYYY-MM-DD)"),
        ("Records/day", "~26 (active fires only)",              "~3,000 (all thermal anomalies)"),
        ("Magnitude",   f"{len(ciffc_df):,} (2 yrs)",           f"{len(hotspot_df):,}"),
    ]
    col1_w = max(len(r[0]) for r in rows) + 2
    for label, cif_val, hot_val in rows:
        print(f"  {label:<{col1_w}} CIFFC: {cif_val}")
        print(f"  {'':<{col1_w}}         Hotspot: {hot_val}")

    # Temporal overlap check
    cif_date_set = set(
        str(pd.to_datetime(ciffc_df["field_situation_report_date"]).dt.date)
        if False else  # pragma: no cover
        pd.to_datetime(ciffc_df["field_situation_report_date"]).dt.date.astype(str)
    )
    hot_date_set = set(hotspot_df["acq_date"].astype(str))
    overlap = cif_date_set & hot_date_set
    print()
    if overlap:
        print(f"  [OK] Temporal overlap: {len(overlap):,} shared dates (matchable)")
    else:
        print(f"  [WARN] No temporal overlap: CIFFC {cif_dates.min().year}-{cif_dates.max().year}, "
              f"Hotspot {hot_dates.min()}-{hot_dates.max()}")
        print(f"     -> Run with a full hotspot file covering 2023-2025 to see the "
              f"real match rate")
    print("=" * 65)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Core matching function
# ─────────────────────────────────────────────────────────────────────────────

def match_ciffc_to_hotspot(
    ciffc_df: pd.DataFrame,
    hotspot_idx: dict[date, np.ndarray],
    window_days: int,
    match_km: float,
    date_field: str = "situation",
) -> pd.DataFrame:
    """
    For every row of ciffc_df, find the nearest hotspot in hotspot_idx.

    Parameters
    ----------
    ciffc_df     : CIFFC DataFrame (with date / field_latitude / field_longitude)
    hotspot_idx  : {date -> np.ndarray(N,2)} date index
    window_days  : time window of +/-N days
    match_km     : distance threshold (km) for declaring a match
    date_field   : "situation"=field_situation_report_date, "status"=field_status_date

    Returns
    -------
    DataFrame: original CIFFC columns + new match columns.
    """
    # Decide which date column to use
    if date_field == "status" and "field_status_date" in ciffc_df.columns:
        ciffc_df = ciffc_df.copy()
        ciffc_df["date"] = pd.to_datetime(
            ciffc_df["field_status_date"]
        ).dt.date
    # If load_ciffc_data already parsed `date`, use it directly
    dates = ciffc_df["date"].values
    lats  = ciffc_df["field_latitude"].values.astype(float)
    lons  = ciffc_df["field_longitude"].values.astype(float)

    n = len(ciffc_df)
    # Result columns (pre-allocated to NaN)
    same_day_nearest_km        = np.full(n, np.nan)
    window_nearest_km          = np.full(n, np.nan)
    window_nearest_day_offset  = np.full(n, np.nan)
    nearest_hotspot_lat        = np.full(n, np.nan)
    nearest_hotspot_lon        = np.full(n, np.nan)
    no_hotspots_in_window      = np.ones(n, dtype=bool)

    print(f"[matching] {n:,} CIFFC records, window +/-{window_days} days, "
          f"threshold {match_km} km ...", flush=True)

    for i in range(n):
        if i % 500 == 0 and i > 0:
            pct = 100 * i / n
            print(f"  {i:,}/{n:,} ({pct:.0f}%)...", flush=True)

        ciffc_date = dates[i]
        lat1 = lats[i]
        lon1 = lons[i]

        # --- Collect all hotspot points within the window ---
        window_pts_list:  list[np.ndarray] = []
        day_offsets_list: list[np.ndarray] = []

        for offset in range(-window_days, window_days + 1):
            d = ciffc_date + timedelta(days=offset)
            pts = hotspot_idx.get(d)
            if pts is not None and len(pts) > 0:
                window_pts_list.append(pts)
                day_offsets_list.append(np.full(len(pts), offset, dtype=np.int32))

        if not window_pts_list:
            # No hotspot at all within the window (file does not cover this period)
            continue

        no_hotspots_in_window[i] = False

        # --- Same-day match ---
        same_day_pts = hotspot_idx.get(ciffc_date)
        if same_day_pts is not None and len(same_day_pts) > 0:
            dists = _haversine_km(lat1, lon1, same_day_pts[:, 0], same_day_pts[:, 1])
            same_day_nearest_km[i] = float(dists.min())

        # --- Nearest match within the window ---
        combined  = np.vstack(window_pts_list)    # (M, 2)
        offsets   = np.concatenate(day_offsets_list)  # (M,)
        dists_all = _haversine_km(lat1, lon1, combined[:, 0], combined[:, 1])
        best_idx  = int(dists_all.argmin())
        window_nearest_km[i]         = float(dists_all[best_idx])
        window_nearest_day_offset[i] = int(offsets[best_idx])
        nearest_hotspot_lat[i]       = float(combined[best_idx, 0])
        nearest_hotspot_lon[i]       = float(combined[best_idx, 1])

    print(f"  Matching done.", flush=True)

    # Assemble results
    out = ciffc_df.copy()
    out["ciffc_date"]                = [str(d) for d in dates]
    out["no_hotspots_in_window"]     = no_hotspots_in_window
    out["same_day_nearest_km"]       = same_day_nearest_km
    out["window_nearest_km"]         = window_nearest_km
    out["window_nearest_day_offset"] = window_nearest_day_offset  # float; NaN where no hotspot in window
    out["nearest_hotspot_lat"]       = nearest_hotspot_lat
    out["nearest_hotspot_lon"]       = nearest_hotspot_lon
    out["matched_same_day"]          = (
        ~np.isnan(same_day_nearest_km) & (same_day_nearest_km < match_km)
    )
    out["matched_window"]            = (
        ~np.isnan(window_nearest_km) & (window_nearest_km < match_km)
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_match_summary(out: pd.DataFrame, window_days: int, match_km: float) -> None:
    n = len(out)
    no_win  = out["no_hotspots_in_window"].sum()
    has_win = n - no_win

    same_day_has = out["same_day_nearest_km"].notna().sum()
    matched_sd   = out["matched_same_day"].sum()
    matched_win  = out["matched_window"].sum()

    print()
    print("=" * 65)
    print(f"  Match statistics (window=+/-{window_days}d, threshold={match_km}km)")
    print("=" * 65)
    print(f"  Total CIFFC records      : {n:>8,}")
    print(f"  No hotspot in window     : {no_win:>8,}  ({100*no_win/n:.1f}%)")
    print(f"  Hotspot in window        : {has_win:>8,}  ({100*has_win/n:.1f}%)")
    print(f"  Hotspot same day         : {same_day_has:>8,}  ({100*same_day_has/n:.1f}%)")
    print(f"  Same-day match (<{match_km:.0f}km)  : {matched_sd:>8,}  ({100*matched_sd/n:.1f}%)")
    print(f"  Window match (<{match_km:.0f}km)    : {matched_win:>8,}  ({100*matched_win/n:.1f}%)")

    # Distance distribution (only printed when data is present)
    wkm = out["window_nearest_km"].dropna()
    if len(wkm) > 0:
        print()
        print("  Nearest-hotspot-in-window distance distribution (km):")
        for pct, val in [
            (0,   wkm.min()),
            (10,  wkm.quantile(0.10)),
            (25,  wkm.quantile(0.25)),
            (50,  wkm.median()),
            (75,  wkm.quantile(0.75)),
            (90,  wkm.quantile(0.90)),
            (100, wkm.max()),
        ]:
            print(f"    P{pct:>3}  {val:>10.2f} km")

    doff = out["window_nearest_day_offset"].dropna()
    if len(doff) > 0:
        print()
        print("  Nearest-hotspot day-offset distribution (days, negative = hotspot before CIFFC):")
        for pct, val in [
            (0,   doff.min()),
            (25,  doff.quantile(0.25)),
            (50,  doff.median()),
            (75,  doff.quantile(0.75)),
            (100, doff.max()),
        ]:
            print(f"    P{pct:>3}  {val:>+8.0f} days")

    if no_win == n:
        print()
        print("  [WARN] Every CIFFC record falls outside the hotspot file's date range.")
        print("     -> Re-run on the server with a full hotspot file covering 2023-2025.")
    print("=" * 65)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="For each CIFFC record, find the nearest CWFIS hotspot match",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    if _HAS_PROJECT:
        add_config_argument(ap)
    ap.add_argument("--ciffc_csv",   type=str, default=None,
                    help="Path to CIFFC CSV (overrides ciffc_csv in config)")
    ap.add_argument("--hotspot_csv", type=str, default=None,
                    help="Path to hotspot CSV (overrides hotspot_csv in config)")
    ap.add_argument("--output_csv",  type=str, default="ciffc_hotspot_match.csv",
                    help="Output CSV path (default ciffc_hotspot_match.csv)")
    ap.add_argument("--window_days", type=int, default=7,
                    help="Time window of +/-N days (default 7)")
    ap.add_argument("--match_km",    type=float, default=10.0,
                    help="Max distance in km to count as a match (default 10.0)")
    ap.add_argument("--date_field",  choices=["situation", "status"],
                    default="situation",
                    help="Use field_situation_report_date(situation) or "
                         "field_status_date(status) (default situation)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    ciffc_path   = args.ciffc_csv
    hotspot_path = args.hotspot_csv

    if _HAS_PROJECT and hasattr(args, "config"):
        cfg = load_config(args.config)
        if ciffc_path is None:
            ciffc_path = get_path(cfg, "ciffc_csv")
        if hotspot_path is None:
            hotspot_path = get_path(cfg, "hotspot_csv")

    if not ciffc_path or not hotspot_path:
        sys.exit(
            "Error: specify data paths via --ciffc_csv / --hotspot_csv or --config."
        )

    ciffc_path   = str(ciffc_path)
    hotspot_path = str(hotspot_path)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n[STEP 1] Load CIFFC  <- {ciffc_path}")
    ciffc_df = _load_ciffc_raw(ciffc_path)
    print(f"  {len(ciffc_df):,} records")

    print(f"\n[STEP 2] Load Hotspot <- {hotspot_path}")
    hotspot_df = _load_hotspot_raw(hotspot_path)
    print(f"  {len(hotspot_df):,} records")

    # ── Data comparison summary ──────────────────────────────────────────────
    _print_data_summary(ciffc_df, hotspot_df)

    # ── Build hotspot date index ─────────────────────────────────────────────
    print("[STEP 3] Build hotspot date index...")
    hotspot_idx = _build_hotspot_index(hotspot_df)

    # ── Match record by record ───────────────────────────────────────────────
    print("\n[STEP 4] Start matching...")
    out_df = match_ciffc_to_hotspot(
        ciffc_df     = ciffc_df,
        hotspot_idx  = hotspot_idx,
        window_days  = args.window_days,
        match_km     = args.match_km,
        date_field   = args.date_field,
    )

    # ── Statistical summary ──────────────────────────────────────────────────
    _print_match_summary(out_df, args.window_days, args.match_km)

    # ── Save output CSV ──────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    out_df.to_csv(out_path, index=False)
    print(f"[done] Results saved to: {out_path.resolve()}")
    print(f"  {len(out_df):,} rows, new columns:")
    new_cols = [
        "ciffc_date", "no_hotspots_in_window",
        "same_day_nearest_km", "window_nearest_km",
        "window_nearest_day_offset",
        "nearest_hotspot_lat", "nearest_hotspot_lon",
        "matched_same_day", "matched_window",
    ]
    for c in new_cols:
        print(f"    {c}")
    print()


if __name__ == "__main__":
    main()
