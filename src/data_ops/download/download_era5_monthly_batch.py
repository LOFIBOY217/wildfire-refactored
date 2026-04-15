"""
ERA5 MONTHLY batch downloader — one CDS request per month (vs per day).

Motivation: CDS queue is currently 1-2h per request. With 9 years × 365 days
× 2 workers = 1643 serial requests → months of wall time. Batching to
1 request per month = 108 requests → tractable even with slow CDS queue.

Each monthly request returns 1 GRIB containing 30 days × 24 hours × N vars.
This is then split into daily GRIBs matching the existing per-day format
so that era5_to_daily.py can process them unchanged.

Usage:
    python -m src.data_ops.download.download_era5_monthly_batch 2009 2017 --workers 4 --variant main
    python -m src.data_ops.download.download_era5_monthly_batch 2000 2017 --workers 4 --variant deep_soil
"""

import argparse
import calendar
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from src.config import load_config, get_path
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path


DEFAULT_AREA = [83, -141, 41, -52]  # Canada [N, W, S, E]

VARIANTS = {
    "main": {
        "variables": [
            '2m_temperature',
            '2m_dewpoint_temperature',
            'total_column_water',
            'volumetric_soil_water_layer_1',
            'soil_temperature_level_1',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation',
            'convective_available_potential_energy',
        ],
        "output_pattern": "era5_sfc_{year}_{month:02d}.grib",
        "output_dir_key": "era5_on_fwi_grid",
    },
    "deep_soil": {
        "variables": ['volumetric_soil_water_layer_2'],
        "output_pattern": "era5_swvl2_{year}_{month:02d}.grib",
        "output_dir_key": "era5_deep_soil",
    },
}


def _make_cds_client(api_key=None):
    import cdsapi
    if api_key is None:
        api_key = os.environ.get("CDS_API_KEY", "5d9b9a0f-cb8f-4773-884d-5ecaa33e6c39")
    return cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=api_key,
        quiet=True,
    )


def download_month(year, month, variant_cfg, out_dir, area, skip_existing=True):
    """Download one monthly GRIB (all days, 24 hours, all vars)."""
    out_path = Path(out_dir) / variant_cfg["output_pattern"].format(year=year, month=month)

    if skip_existing and out_path.exists() and out_path.stat().st_size > 1e6:
        return (year, month, "skip")

    # All days of this month
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]

    req = {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": variant_cfg["variables"],
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
    }

    try:
        client = _make_cds_client()
        client.retrieve("reanalysis-era5-single-levels", req, str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        return (year, month, f"ok ({size_mb:.0f}MB)")
    except Exception as e:
        return (year, month, f"fail: {type(e).__name__}: {e}")


def split_monthly_grib_to_daily(monthly_grib, daily_dir, variant):
    """Split a monthly GRIB into daily GRIBs using eccodes grib_copy.

    Uses shell command: grib_copy -w time=0000 monthly.grib "daily_{dataDate}.grib"
    But we want all times. Use: grib_copy monthly.grib "out_[dataDate].grib"
    """
    import subprocess
    monthly_grib = Path(monthly_grib)
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)

    # grib_copy splits by dataDate (YYYYMMDD)
    # output pattern: era5_sfc_YYYY_MM_DD.grib or era5_swvl2_YYYY_MM_DD.grib
    prefix = "era5_sfc" if variant == "main" else "era5_swvl2"
    # grib_copy uses named keys: [key] in output pattern
    tmp_pattern = str(daily_dir / f"{prefix}_tmp_[dataDate].grib")

    try:
        subprocess.run(
            ["grib_copy", str(monthly_grib), tmp_pattern],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return f"grib_copy failed: {e.stderr.decode()[:200]}"

    # Rename tmp files: era5_sfc_tmp_20090101.grib → era5_sfc_2009_01_01.grib
    renamed = 0
    for tmp_file in daily_dir.glob(f"{prefix}_tmp_*.grib"):
        name = tmp_file.stem.replace(f"{prefix}_tmp_", "")
        if len(name) == 8 and name.isdigit():
            y, m, d = name[:4], name[4:6], name[6:8]
            final = daily_dir / f"{prefix}_{y}_{m}_{d}.grib"
            if not final.exists():
                tmp_file.rename(final)
                renamed += 1
            else:
                tmp_file.unlink()
    return f"split {renamed} daily files"


def main():
    ap = argparse.ArgumentParser(description="ERA5 monthly batch downloader")
    ap.add_argument("start_year", type=int)
    ap.add_argument("end_year", type=int)
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), default="main")
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--monthly-dir", type=str, default=None,
                    help="Where to store monthly GRIBs (default: {output_dir}/_monthly)")
    ap.add_argument("--daily-dir", type=str, default=None,
                    help="Where to split daily GRIBs (default: from config)")
    ap.add_argument("--config", type=str, default="configs/paths_narval.yaml")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-split", action="store_true",
                    help="Only download monthly, don't split to daily")
    args = ap.parse_args()

    variant_cfg = VARIANTS[args.variant]

    # Resolve paths
    cfg = load_config(args.config)
    paths = cfg.get("paths", cfg)
    daily_dir = args.daily_dir or get_path(cfg, variant_cfg["output_dir_key"])
    monthly_dir = args.monthly_dir or os.path.join(daily_dir, "_monthly")
    os.makedirs(monthly_dir, exist_ok=True)
    os.makedirs(daily_dir, exist_ok=True)

    # Build job list
    jobs = []
    for year in range(args.start_year, args.end_year + 1):
        for month in args.months:
            jobs.append((year, month))

    print(f"=== ERA5 Monthly Batch Downloader ===")
    print(f"  Variant : {args.variant}")
    print(f"  Months  : {len(jobs)} (years {args.start_year}-{args.end_year})")
    print(f"  Workers : {args.workers}")
    print(f"  Monthly : {monthly_dir}")
    print(f"  Daily   : {daily_dir}")

    t0 = time.time()
    ok, skipped, failed = 0, 0, 0

    def _work(ym):
        y, m = ym
        return download_month(y, m, variant_cfg, monthly_dir, DEFAULT_AREA, args.skip_existing)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for year, month, status in pool.map(_work, jobs):
            elapsed = time.time() - t0
            print(f"  [{year}-{month:02d}] {status}  ({elapsed:.0f}s total)", flush=True)
            if status.startswith("ok"):
                ok += 1
                # Split immediately for this month
                if not args.no_split:
                    monthly_grib = Path(monthly_dir) / variant_cfg["output_pattern"].format(year=year, month=month)
                    if monthly_grib.exists():
                        split_status = split_monthly_grib_to_daily(monthly_grib, daily_dir, args.variant)
                        print(f"    → {split_status}", flush=True)
            elif status == "skip":
                skipped += 1
            else:
                failed += 1

    print(f"\n=== Done: {ok} ok, {skipped} skipped, {failed} failed ({time.time()-t0:.0f}s) ===")


if __name__ == "__main__":
    main()
