"""
Compute FWI from ECMWF S2S forecast variables.

For each S2S issue date, runs Van Wagner equations forward across lead days
using S2S forecast weather (2t, 2d, 10u, 10v, tp) and initializing
FFMC/DMC/DC from observed values at the issue date.

Input:
  - S2S forecast TIFs: data/s2s_processed/{YYYY-MM-DD}/lead{NN}.tif
    Band layout: [2t, 2d, tcw, sm20, st20, VPD] (core set, existing)
    Plus extended TIFs: lead{NN}_ext.tif with [10u, 10v, tp]
  - Observed FFMC/DMC/DC TIFs for initialization

Output:
  - Per-issue-date FWI grids for lead 14-45

Usage:
  python -m src.data_ops.processing.compute_s2s_fwi \
      --config configs/paths_narval.yaml \
      --s2s_dir data/s2s_processed \
      --output_dir data/s2s_fwi \
      --lead_start 14 --lead_end 45
"""

import argparse
import glob
import os
import re
import time
from datetime import date, timedelta

import numpy as np
import rasterio
import yaml

from src.data_ops.processing.fwi_calculator import (
    FWICalculator,
    dewpoint_to_rh,
    wind_components_to_speed,
)


LEAD_START_DEFAULT = 14
LEAD_END_DEFAULT = 45


def _parse_date(s):
    parts = s.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _extract_date_from_filename(fname):
    m = re.search(r"(\d{8})", fname)
    if m:
        ds = m.group(1)
        return date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
    return None


def _build_file_index(directory):
    """Map date -> filepath."""
    index = {}
    for f in glob.glob(os.path.join(directory, "*.tif")):
        d = _extract_date_from_filename(os.path.basename(f))
        if d:
            index[d] = f
    return index


def _read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _read_tif_bands(path, bands=None):
    """Read multi-band TIF. Returns (n_bands, H, W) float32."""
    with rasterio.open(path) as src:
        if bands is None:
            return src.read().astype(np.float32)
        return src.read(bands).astype(np.float32)


def load_s2s_weather(s2s_dir, issue_date_str, lead, use_extended=True):
    """
    Load S2S forecast weather for one issue date and lead day.

    Returns dict with keys: temp_c, rh, wind_kmh, rain_mm
    or None if data is missing.
    """
    date_dir = os.path.join(s2s_dir, issue_date_str)
    if not os.path.isdir(date_dir):
        return None

    # Core TIF: [2t, 2d, tcw, sm20, st20, VPD]
    core_path = os.path.join(date_dir, f"lead{lead:02d}.tif")
    if not os.path.exists(core_path):
        return None

    core = _read_tif_bands(core_path)  # (6, H, W)
    temp_c = core[0]   # 2t in Celsius
    dewp_c = core[1]   # 2d in Celsius

    # Compute RH from temperature and dewpoint
    rh = dewpoint_to_rh(temp_c, dewp_c)

    # Extended TIF: [10u, 10v, tp]
    if use_extended:
        ext_path = os.path.join(date_dir, f"lead{lead:02d}_ext.tif")
        if os.path.exists(ext_path):
            ext = _read_tif_bands(ext_path)  # (3, H, W)
            u10 = ext[0]  # m/s
            v10 = ext[1]  # m/s
            tp = ext[2]   # total precipitation (m) -> convert to mm
            wind_ms = wind_components_to_speed(u10, v10)
            wind_kmh = wind_ms * 3.6  # m/s -> km/h
            rain_mm = np.maximum(tp * 1000.0, 0.0)  # m -> mm, clip negative
        else:
            return None
    else:
        # Fallback: no wind/precip available, use defaults
        wind_kmh = np.full_like(temp_c, 12.0)  # ~moderate wind
        rain_mm = np.zeros_like(temp_c)

    return {
        "temp_c": temp_c,
        "rh": rh,
        "wind_kmh": wind_kmh,
        "rain_mm": rain_mm,
    }


def compute_s2s_fwi_for_date(s2s_dir, issue_date, lead_start, lead_end,
                             ffmc_init, dmc_init, dc_init,
                             use_extended=True):
    """
    Compute FWI for all lead days of one S2S issue date.

    Args:
        s2s_dir: path to s2s_processed directory
        issue_date: date object
        lead_start, lead_end: lead day range (e.g., 14-45)
        ffmc_init, dmc_init, dc_init: (H, W) arrays from observations
        use_extended: whether to use wind/precip from extended param set

    Returns:
        fwi_stack: (n_leads, H, W) float32, or None if insufficient data
    """
    issue_str = issue_date.strftime("%Y-%m-%d")

    # We need to run FWI from lead 0 to lead_end to properly evolve
    # the moisture codes. But if we only have lead_start..lead_end data,
    # we initialize at lead_start and run forward.
    H, W = ffmc_init.shape
    calc = FWICalculator(shape=(H, W))
    calc.set_state(ffmc_init, dmc_init, dc_init)

    n_leads = lead_end - lead_start + 1
    fwi_stack = np.full((n_leads, H, W), np.nan, dtype=np.float32)

    # If we have lead 0..lead_start-1, use them to spin up moisture codes
    for lead in range(0, lead_start):
        target_date = issue_date + timedelta(days=lead)
        month = target_date.month
        weather = load_s2s_weather(s2s_dir, issue_str, lead, use_extended)
        if weather is not None:
            calc.update(weather["temp_c"], weather["rh"],
                       weather["wind_kmh"], weather["rain_mm"], month)

    # Now compute FWI for lead_start..lead_end
    for li, lead in enumerate(range(lead_start, lead_end + 1)):
        target_date = issue_date + timedelta(days=lead)
        month = target_date.month
        weather = load_s2s_weather(s2s_dir, issue_str, lead, use_extended)
        if weather is None:
            # Missing lead day — keep NaN
            continue
        result = calc.update(weather["temp_c"], weather["rh"],
                           weather["wind_kmh"], weather["rain_mm"], month)
        fwi_stack[li] = result["FWI"].astype(np.float32)

    n_valid = np.sum(~np.isnan(fwi_stack[:, 0, 0]))
    if n_valid == 0:
        return None

    return fwi_stack


def main():
    parser = argparse.ArgumentParser(
        description="Compute FWI from ECMWF S2S forecasts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--s2s_dir", default=None,
                        help="S2S processed directory (override config)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for S2S-FWI TIFs")
    parser.add_argument("--lead_start", type=int, default=LEAD_START_DEFAULT)
    parser.add_argument("--lead_end", type=int, default=LEAD_END_DEFAULT)
    parser.add_argument("--no_extended", action="store_true",
                        help="Skip extended params (wind/precip), use defaults")
    parser.add_argument("--issue_dates", nargs="*", default=None,
                        help="Specific issue dates to process (YYYY-MM-DD)")
    parser.add_argument("--start_date", default=None,
                        help="Skip issue dates before this (YYYY-MM-DD)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["paths"]

    s2s_dir = args.s2s_dir or cfg.get("s2s_processed", cfg.get("s2s_dir"))
    output_dir = args.output_dir or os.path.join(cfg["project_root"], "data", "s2s_fwi")
    os.makedirs(output_dir, exist_ok=True)

    # Build indices for observed FFMC/DMC/DC (for initialization)
    ffmc_index = _build_file_index(cfg["ffmc_dir"])
    dmc_index = _build_file_index(cfg["dmc_dir"])
    dc_index = _build_file_index(cfg["dc_dir"])
    print(f"Observed indices: FFMC={len(ffmc_index)} DMC={len(dmc_index)} DC={len(dc_index)}")

    # Discover S2S issue dates
    if args.issue_dates:
        issue_dates = [_parse_date(d) for d in args.issue_dates]
    else:
        issue_dates = []
        for entry in sorted(os.listdir(s2s_dir)):
            try:
                issue_dates.append(_parse_date(entry))
            except Exception:
                continue
    if args.start_date:
        cutoff = _parse_date(args.start_date)
        issue_dates = [d for d in issue_dates if d >= cutoff]
    print(f"S2S issue dates: {len(issue_dates)}")

    # Reference TIF for writing output
    ref_path = cfg.get("fwi_reference_tif")

    n_success = 0
    n_skip = 0
    t0 = time.time()

    for di, issue_date in enumerate(issue_dates):
        # Find observed FFMC/DMC/DC on or near the issue date
        init_date = issue_date
        ffmc_path = ffmc_index.get(init_date)
        dmc_path = dmc_index.get(init_date)
        dc_path = dc_index.get(init_date)

        # Try day before if exact date not available
        if not all([ffmc_path, dmc_path, dc_path]):
            init_date = issue_date - timedelta(days=1)
            ffmc_path = ffmc_path or ffmc_index.get(init_date)
            dmc_path = dmc_path or dmc_index.get(init_date)
            dc_path = dc_path or dc_index.get(init_date)

        if not all([ffmc_path, dmc_path, dc_path]):
            n_skip += 1
            if n_skip <= 5:
                print(f"  SKIP {issue_date}: missing observed FFMC/DMC/DC for init")
            continue

        ffmc_init = _read_tif(ffmc_path)
        dmc_init = _read_tif(dmc_path)
        dc_init = _read_tif(dc_path)

        # Compute S2S-FWI
        fwi_stack = compute_s2s_fwi_for_date(
            s2s_dir, issue_date, args.lead_start, args.lead_end,
            ffmc_init, dmc_init, dc_init,
            use_extended=not args.no_extended,
        )

        if fwi_stack is None:
            n_skip += 1
            continue

        # Save output
        out_date_dir = os.path.join(output_dir, issue_date.strftime("%Y-%m-%d"))
        os.makedirs(out_date_dir, exist_ok=True)

        # Write mean FWI across all leads as summary
        fwi_mean = np.nanmean(fwi_stack, axis=0)
        fwi_max = np.nanmax(fwi_stack, axis=0)

        # Save per-lead and summary TIFs
        if ref_path and os.path.exists(ref_path):
            with rasterio.open(ref_path) as ref:
                profile = ref.profile.copy()
                profile.update(dtype="float32", count=1, compress="lzw")

            # Summary files
            for name, arr in [("fwi_mean", fwi_mean), ("fwi_max", fwi_max)]:
                out_path = os.path.join(out_date_dir, f"s2s_{name}.tif")
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(np.nan_to_num(arr, nan=0.0)[np.newaxis])

        n_success += 1
        if (di + 1) % 10 == 0 or di == len(issue_dates) - 1:
            elapsed = time.time() - t0
            print(f"  [{di+1}/{len(issue_dates)}] success={n_success} "
                  f"skip={n_skip} ({elapsed:.0f}s)")

    print(f"\nDone: {n_success} dates processed, {n_skip} skipped "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
