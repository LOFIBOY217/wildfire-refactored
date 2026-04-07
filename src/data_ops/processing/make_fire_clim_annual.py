#!/usr/bin/env python3
"""
Generate annual rolling fire climatology maps for train_v3.

For each target year Y, builds fire_clim_upto_Y.tif using only hotspot
records from years BEFORE Y (i.e. 2018 … Y-1).  This prevents data leakage:
the model never sees future fire events in its prior.

Output files:
    {fire_clim_dir}/fire_clim_upto_2018.tif  — uses 2017 (requires 2017 hotspot data in CSV)
    {fire_clim_dir}/fire_clim_upto_2019.tif  — uses 2017-2018
    {fire_clim_dir}/fire_clim_upto_2020.tif  — uses 2017-2019
    ...
    {fire_clim_dir}/fire_clim_upto_2025.tif  — uses 2017-2024

Requires: 2017 hotspot data appended to hotspot_csv (via slurm/download_hotspot_2017_narval.sh)

Usage:
    python -m src.data_ops.processing.make_fire_clim_annual \
        --config configs/paths_narval.yaml \
        --start_year 2018 \
        --end_year 2025 \
        --output_dir data/fire_clim_annual
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import rasterio

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument

from src.data_ops.processing.make_fire_climatology import build_fire_climatology


def make_zero_tif(reference_tif, output_path):
    """Write an all-zero fire climatology (used when no prior data exists)."""
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
    profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=None)
    profile.pop("photometric", None)
    out = np.zeros((height, width), dtype=np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)
    print(f"  [zero prior] {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate annual rolling fire climatology TIFs for train_v3"
    )
    add_config_argument(parser)
    parser.add_argument("--start_year", type=int, default=2018,
                        help="First target year (default: 2018)")
    parser.add_argument("--end_year",   type=int, default=2025,
                        help="Last target year (default: 2025)")
    parser.add_argument("--data_start_year", type=int, default=2017,
                        help="First year of available hotspot data (default: 2017)")
    parser.add_argument("--months", type=str, default="5-10",
                        help="Fire season months, e.g. '5-10' (default: May-Oct)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/fire_clim_annual)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference FWI TIF for grid (auto-detected if omitted)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})
    hotspot_csv = get_path(cfg, "hotspot_csv")
    fwi_dir     = get_path(cfg, "fwi_dir")

    # Auto-detect reference TIF
    ref_tif = args.reference
    if ref_tif is None:
        candidates = sorted(glob.glob(str(Path(fwi_dir) / "fwi_*.tif")))
        if not candidates:
            raise RuntimeError(f"No FWI TIFs found in {fwi_dir}")
        ref_tif = candidates[0]
        print(f"Reference grid: {ref_tif}")

    # Output directory
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = str(Path(get_path(cfg, "output_dir")).parent / "data" / "fire_clim_annual")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Parse months
    parts = args.months.split("-")
    months = list(range(int(parts[0]), int(parts[-1]) + 1))

    print("=" * 60)
    print("ANNUAL ROLLING FIRE CLIMATOLOGY")
    print("=" * 60)
    print(f"  Target years   : {args.start_year} – {args.end_year}")
    print(f"  Months         : {months}")
    print(f"  Data available : from {args.data_start_year}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Rule           : fire_clim_upto_Y = hotspots from [{args.data_start_year}, Y-1]")
    print("=" * 60)

    for target_year in range(args.start_year, args.end_year + 1):
        out_path = str(Path(out_dir) / f"fire_clim_upto_{target_year}.tif")

        if Path(out_path).exists():
            print(f"\n[{target_year}] Already exists — skipping: {out_path}")
            continue

        prior_years = list(range(args.data_start_year, target_year))  # strictly before Y

        print(f"\n[{target_year}] fire_clim_upto_{target_year}.tif")
        if not prior_years:
            print(f"  No prior years available (data_start_year={args.data_start_year} == {target_year})")
            print(f"  → writing zero map")
            make_zero_tif(ref_tif, out_path)
        else:
            print(f"  Using years: {prior_years}")
            build_fire_climatology(
                hotspot_csv=hotspot_csv,
                fwi_reference_tif=ref_tif,
                years=prior_years,
                months=months,
                output_path=out_path,
                log_transform=True,
                count_mode="days",
            )

    print("\n=== ALL DONE ===")
    print(f"Generated TIFs in: {out_dir}")
    import glob as _glob
    tifs = sorted(_glob.glob(str(Path(out_dir) / "fire_clim_upto_*.tif")))
    for t in tifs:
        size_kb = Path(t).stat().st_size / 1e3
        print(f"  {Path(t).name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
