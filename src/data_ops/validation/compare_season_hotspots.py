"""
compare_season_hotspots.py
==========================
Compare fire-season (May–Oct) vs off-season (Jan–Apr, Nov–Dec) hotspot
record counts to verify whether off-season fires are truly negligible.

Step 1 — Download off-season data (run on server first):

    python -m src.data_ops.download.download_hotspots ^
        --config configs/paths_windows.yaml ^
        --start_month 1 --end_month 4 ^
        --output data/hotspot/hotspot_offseason_jan_apr.csv

    python -m src.data_ops.download.download_hotspots ^
        --config configs/paths_windows.yaml ^
        --start_month 11 --end_month 12 ^
        --output data/hotspot/hotspot_offseason_nov_dec.csv

Step 2 — Run this script:

    python -m src.data_ops.validation.compare_season_hotspots ^
        --config configs/paths_windows.yaml

    # or with explicit paths:
    python -m src.data_ops.validation.compare_season_hotspots ^
        --fire_csv   data/hotspot/hotspot_2018_2025.csv ^
        --off1_csv   data/hotspot/hotspot_offseason_jan_apr.csv ^
        --off2_csv   data/hotspot/hotspot_offseason_nov_dec.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import load_config, get_path, add_config_argument
    _HAS_PROJECT = True
except ImportError:
    _HAS_PROJECT = False


MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar",  4: "Apr",
    5: "May", 6: "Jun", 7: "Jul",  8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
FIRE_SEASON   = {5, 6, 7, 8, 9, 10}
OFF_SEASON_1  = {1, 2, 3, 4}          # Jan–Apr
OFF_SEASON_2  = {11, 12}              # Nov–Dec


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, total: float, width: int = 30) -> str:
    filled = int(round(width * value / total)) if total > 0 else 0
    return "#" * filled + "." * (width - filled)


def _load(path: str, label: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {label}: file not found — {path}")
        return None
    df = pd.read_csv(path, usecols=["acq_date"])
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"])
    df["month"] = df["acq_date"].dt.month
    df["year"]  = df["acq_date"].dt.year
    print(f"  Loaded {len(df):>10,} records  ← {label}  ({path})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    fire_path: str,
    off1_path: str | None,
    off2_path: str | None,
) -> None:
    print(f"\n{'='*65}")
    print(f"  Fire Season vs Off-Season Hotspot Comparison")
    print(f"{'='*65}\n")

    # ── Load files ────────────────────────────────────────────────────
    print("Loading files...")
    df_fire = _load(fire_path, "Fire season (May–Oct)")
    df_off1 = _load(off1_path, "Off-season (Jan–Apr)") if off1_path else None
    df_off2 = _load(off2_path, "Off-season (Nov–Dec)") if off2_path else None

    have_off = df_off1 is not None or df_off2 is not None
    if df_fire is None:
        sys.exit("\n[ERROR] Fire-season CSV not found. Cannot continue.")
    if not have_off:
        sys.exit(
            "\n[ERROR] No off-season CSV found.\n"
            "Run the download commands shown in this script's docstring first."
        )

    frames = [df for df in [df_fire, df_off1, df_off2] if df is not None]
    all_df = pd.concat(frames, ignore_index=True)
    total  = len(all_df)

    # ── Monthly breakdown ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Monthly Record Counts  (all years combined)")
    print(f"{'='*65}")
    print(f"\n  {'Month':<6} {'Count':>10}  {'% of year':>10}  Distribution          Season")
    print(f"  {'─'*65}")

    monthly = all_df.groupby("month").size().reindex(range(1, 13), fill_value=0)
    for m in range(1, 13):
        cnt  = int(monthly[m])
        pct  = 100 * cnt / total if total > 0 else 0
        bar  = _bar(cnt, total, 30)
        if m in FIRE_SEASON:
            season = "← fire season"
        elif m in OFF_SEASON_1:
            season = "  off-season (Jan–Apr)"
        else:
            season = "  off-season (Nov–Dec)"
        print(f"  {MONTH_NAMES[m]:<6} {cnt:>10,}  {pct:>9.2f}%  {bar}  {season}")

    # ── Season summary ────────────────────────────────────────────────
    fire_n = int(monthly[list(FIRE_SEASON)].sum())
    off1_n = int(monthly[list(OFF_SEASON_1)].sum())
    off2_n = int(monthly[list(OFF_SEASON_2)].sum())
    off_n  = off1_n + off2_n

    print(f"\n{'='*65}")
    print(f"  Season Summary")
    print(f"{'='*65}")
    print(f"\n  {'Season':<25} {'Records':>10}  {'% of year':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Fire season (May–Oct)':<25} {fire_n:>10,}  {100*fire_n/total:>9.2f}%")
    print(f"  {'Off-season Jan–Apr':<25} {off1_n:>10,}  {100*off1_n/total:>9.2f}%")
    print(f"  {'Off-season Nov–Dec':<25} {off2_n:>10,}  {100*off2_n/total:>9.2f}%")
    print(f"  {'─'*50}")
    print(f"  {'Total off-season':<25} {off_n:>10,}  {100*off_n/total:>9.2f}%")
    print(f"  {'TOTAL':<25} {total:>10,}  {'100.00%':>10}")

    # ── Year-by-year off-season breakdown ─────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Off-Season Records by Year")
    print(f"{'='*65}")

    off_df = pd.concat(
        [df for df in [df_off1, df_off2] if df is not None],
        ignore_index=True,
    )
    if len(off_df) > 0:
        yearly_off  = off_df.groupby("year").size()
        yearly_fire = df_fire.groupby("year").size()
        all_years   = sorted(set(yearly_off.index) | set(yearly_fire.index))

        print(f"\n  {'Year':<6} {'Off-season':>11}  {'Fire season':>12}  {'Off/Fire ratio':>15}")
        print(f"  {'─'*52}")
        for y in all_years:
            o = int(yearly_off.get(y, 0))
            f = int(yearly_fire.get(y, 0))
            ratio = f"{o/f*100:.1f}%" if f > 0 else "N/A"
            print(f"  {y:<6} {o:>11,}  {f:>12,}  {ratio:>15}")

    # ── Verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    off_pct = 100 * off_n / total if total > 0 else 0
    if off_pct < 3.0:
        verdict = (f"[PASS] Off-season fires are NEGLIGIBLE ({off_pct:.2f}% of all hotspots).\n"
                   f"   Limiting training data to May-Oct is justified.")
    elif off_pct < 10.0:
        verdict = (f"[WARN] Off-season fires are MINOR ({off_pct:.2f}% of all hotspots).\n"
                   f"   May-Oct captures the vast majority; consider your use case.")
    else:
        verdict = (f"[FAIL] Off-season fires are SIGNIFICANT ({off_pct:.2f}% of all hotspots).\n"
                   f"   Consider extending the download to include off-season months.")
    print(f"\n  {verdict}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="Compare fire-season vs off-season hotspot record counts."
    )
    if _HAS_PROJECT:
        add_config_argument(p)
    p.add_argument("--fire_csv", type=str, default=None,
                   help="Fire-season hotspot CSV (default: from config hotspot_csv)")
    p.add_argument("--off1_csv", type=str, default=None,
                   help="Off-season Jan–Apr CSV (default: data/hotspot/hotspot_offseason_jan_apr.csv)")
    p.add_argument("--off2_csv", type=str, default=None,
                   help="Off-season Nov–Dec CSV (default: data/hotspot/hotspot_offseason_nov_dec.csv)")
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Resolve fire-season path
    fire_path = args.fire_csv
    if fire_path is None:
        if _HAS_PROJECT and hasattr(args, "config"):
            cfg = load_config(args.config)
            try:
                fire_path = get_path(cfg, "hotspot_csv")
            except Exception:
                pass
        if fire_path is None:
            fire_path = "data/hotspot/hotspot_2018_2025.csv"

    # Resolve off-season paths (default next to fire CSV)
    fire_dir  = str(Path(fire_path).parent)
    off1_path = args.off1_csv or str(Path(fire_dir) / "hotspot_offseason_jan_apr.csv")
    off2_path = args.off2_csv or str(Path(fire_dir) / "hotspot_offseason_nov_dec.csv")

    run_comparison(
        fire_path=str(fire_path),
        off1_path=off1_path,
        off2_path=off2_path,
    )


if __name__ == "__main__":
    main()
