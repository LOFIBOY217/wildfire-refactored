#!/usr/bin/env python3
"""
Cross-validate SRTM slope vs CDEM slope in their overlap zone (<60°N).

Both datasets are on the FWI grid (EPSG:3978, 2709×2281). The overlap zone
is where both have finite values — this is roughly Canada south of 60°N.

Metrics computed:
  1. Pixel-wise correlation (R², Pearson r)
  2. Error statistics (RMSE, MAE, bias = CDEM - SRTM)
  3. Per-latitude-band comparison (5 bands from south to north)
  4. Distribution comparison (percentiles)
  5. Fire-relevant zone check (where fire_climatology > 0)

Pass criteria:
  - R² > 0.90 in overlap zone
  - MAE < 2° (slope degrees)
  - |bias| < 0.5°

Usage:
    python -m src.data_ops.validation.cross_validate_slope
    python -m src.data_ops.validation.cross_validate_slope --config configs/paths_narval.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from pyproj import Transformer

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.config import load_config, get_path, add_config_argument


FWI_CRS    = "EPSG:3978"
FWI_WIDTH  = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)


def _pixel_latitudes():
    """Compute latitude (WGS84) for each row of the FWI grid."""
    tf = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
    # Centre x of grid (we only need latitude, x doesn't matter much)
    cx = FWI_BOUNDS[0] + (FWI_BOUNDS[2] - FWI_BOUNDS[0]) / 2
    ys = np.array([tf.f + tf.e * (r + 0.5) for r in range(FWI_HEIGHT)])
    xs = np.full_like(ys, cx)
    proj = Transformer.from_crs(FWI_CRS, "EPSG:4326", always_xy=True)
    _, lats = proj.transform(xs, ys)
    return lats  # shape (FWI_HEIGHT,)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate SRTM vs CDEM slope in overlap zone"
    )
    add_config_argument(parser)
    args = parser.parse_args()

    cfg = load_config(args.config) if hasattr(args, "config") and args.config else {}
    try:
        terrain_dir = Path(get_path(cfg, "terrain_dir"))
    except (KeyError, TypeError):
        terrain_dir = Path("data/terrain")

    # Load SRTM slope (original slope.tif from SRTM, or slope_srtm.tif)
    srtm_path = terrain_dir / "slope_srtm.tif"
    if not srtm_path.exists():
        # Fall back to original slope.tif (before CDEM replaced it)
        srtm_path = terrain_dir / "slope.tif"
    cdem_path = terrain_dir / "slope_cdem.tif"

    # Also try fire_climatology for fire-relevant zone analysis
    try:
        fire_clim_path = Path(get_path(cfg, "fire_climatology_tif"))
    except (KeyError, TypeError):
        fire_clim_path = Path("data/fire_climatology.tif")

    print("=" * 70)
    print("SRTM vs CDEM SLOPE — Cross-Validation")
    print("=" * 70)

    # Check file existence
    for name, path in [("SRTM slope", srtm_path), ("CDEM slope", cdem_path)]:
        if not path.exists():
            print(f"  [FAIL] {name} not found: {path}")
            print("  Both slope files are required for cross-validation.")
            sys.exit(1)
        print(f"  {name}: {path}")

    # Load data
    with rasterio.open(srtm_path) as src:
        srtm = src.read(1).astype(np.float32)
    with rasterio.open(cdem_path) as src:
        cdem = src.read(1).astype(np.float32)

    print(f"\n  Grid shape: SRTM={srtm.shape}  CDEM={cdem.shape}")
    assert srtm.shape == (FWI_HEIGHT, FWI_WIDTH), "SRTM shape mismatch"
    assert cdem.shape == (FWI_HEIGHT, FWI_WIDTH), "CDEM shape mismatch"

    srtm_valid = np.isfinite(srtm)
    cdem_valid = np.isfinite(cdem)
    overlap = srtm_valid & cdem_valid

    srtm_pct = 100 * srtm_valid.sum() / srtm.size
    cdem_pct = 100 * cdem_valid.sum() / cdem.size
    overlap_pct = 100 * overlap.sum() / srtm.size
    cdem_only_pct = 100 * (cdem_valid & ~srtm_valid).sum() / cdem.size

    print(f"  SRTM coverage:   {srtm_pct:.1f}%  ({srtm_valid.sum():,} pixels)")
    print(f"  CDEM coverage:   {cdem_pct:.1f}%  ({cdem_valid.sum():,} pixels)")
    print(f"  Overlap zone:    {overlap_pct:.1f}%  ({overlap.sum():,} pixels)")
    print(f"  CDEM-only (>60°N): {cdem_only_pct:.1f}%  ({(cdem_valid & ~srtm_valid).sum():,} pixels)")

    if overlap.sum() < 1000:
        print("  [FAIL] Too few overlap pixels for comparison")
        sys.exit(1)

    # ── Section 1: Overall statistics ──
    s = srtm[overlap]
    c = cdem[overlap]
    diff = c - s

    bias = np.mean(diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    r = np.corrcoef(s, c)[0, 1]
    r2 = r ** 2

    print(f"\n{'─' * 60}")
    print(f"  OVERLAP ZONE STATISTICS ({overlap.sum():,} pixels)")
    print(f"{'─' * 60}")
    print(f"  Pearson r:  {r:.4f}")
    print(f"  R²:         {r2:.4f}")
    print(f"  Bias:       {bias:+.3f}° (CDEM - SRTM)")
    print(f"  MAE:        {mae:.3f}°")
    print(f"  RMSE:       {rmse:.3f}°")
    print(f"  SRTM mean:  {s.mean():.2f}°  std: {s.std():.2f}°")
    print(f"  CDEM mean:  {c.mean():.2f}°  std: {c.std():.2f}°")

    # ── Section 2: Percentile comparison ──
    pcts = [5, 25, 50, 75, 90, 95, 99]
    s_pct = np.percentile(s, pcts)
    c_pct = np.percentile(c, pcts)

    print(f"\n{'─' * 60}")
    print(f"  PERCENTILE COMPARISON")
    print(f"{'─' * 60}")
    print(f"  {'Pct':>5s}  {'SRTM':>8s}  {'CDEM':>8s}  {'Diff':>8s}")
    for p, sv, cv in zip(pcts, s_pct, c_pct):
        print(f"  {p:5d}  {sv:8.2f}  {cv:8.2f}  {cv - sv:+8.3f}")

    # ── Section 3: Per-latitude-band comparison ──
    print(f"\n{'─' * 60}")
    print(f"  PER-LATITUDE-BAND COMPARISON")
    print(f"{'─' * 60}")

    lats = _pixel_latitudes()  # (FWI_HEIGHT,)
    lat_bands = [(41, 46), (46, 50), (50, 54), (54, 58), (58, 61)]

    print(f"  {'Band':>10s}  {'N_pixels':>10s}  {'R²':>6s}  {'Bias':>8s}  "
          f"{'MAE':>6s}  {'RMSE':>6s}")
    for lat_lo, lat_hi in lat_bands:
        row_mask = (lats >= lat_lo) & (lats < lat_hi)
        band_mask = overlap.copy()
        band_mask[~row_mask[:, None].repeat(FWI_WIDTH, axis=1)] = False
        n = band_mask.sum()
        if n < 100:
            print(f"  {lat_lo}-{lat_hi}°N  {n:>10d}  (too few)")
            continue
        sb = srtm[band_mask]
        cb = cdem[band_mask]
        db = cb - sb
        br = np.corrcoef(sb, cb)[0, 1] ** 2
        print(f"  {lat_lo}-{lat_hi}°N  {n:>10,d}  {br:6.3f}  {db.mean():+8.3f}  "
              f"{np.abs(db).mean():6.3f}  {np.sqrt((db**2).mean()):6.3f}")

    # ── Section 4: CDEM-only zone (>60°N) statistics ──
    cdem_only = cdem_valid & ~srtm_valid
    if cdem_only.sum() > 100:
        north_slope = cdem[cdem_only]
        print(f"\n{'─' * 60}")
        print(f"  CDEM-ONLY ZONE (>60°N, {cdem_only.sum():,} pixels)")
        print(f"{'─' * 60}")
        print(f"  Mean slope:  {north_slope.mean():.2f}°")
        print(f"  Std:         {north_slope.std():.2f}°")
        print(f"  Range:       [{north_slope.min():.2f}°, {north_slope.max():.2f}°]")
        print(f"  Median:      {np.median(north_slope):.2f}°")

    # ── Section 5: Fire-relevant zone ──
    if fire_clim_path.exists():
        with rasterio.open(fire_clim_path) as src:
            fire_clim = src.read(1).astype(np.float32)
        fire_zone = (fire_clim > 0) & overlap
        if fire_zone.sum() > 100:
            sf = srtm[fire_zone]
            cf = cdem[fire_zone]
            df = cf - sf
            fr = np.corrcoef(sf, cf)[0, 1] ** 2
            print(f"\n{'─' * 60}")
            print(f"  FIRE-RELEVANT ZONE (fire_clim > 0, {fire_zone.sum():,} pixels)")
            print(f"{'─' * 60}")
            print(f"  R²:    {fr:.4f}")
            print(f"  Bias:  {df.mean():+.3f}°")
            print(f"  MAE:   {np.abs(df).mean():.3f}°")
            print(f"  RMSE:  {np.sqrt((df**2).mean()):.3f}°")
    else:
        print(f"\n  [INFO] fire_climatology.tif not found, skipping fire-zone analysis")

    # ── Section 6: Error distribution ──
    print(f"\n{'─' * 60}")
    print(f"  ERROR DISTRIBUTION (|CDEM - SRTM|)")
    print(f"{'─' * 60}")
    abs_diff = np.abs(diff)
    thresholds = [0.5, 1.0, 2.0, 5.0, 10.0]
    for t in thresholds:
        pct = 100 * (abs_diff <= t).sum() / len(abs_diff)
        print(f"  |diff| ≤ {t:5.1f}°: {pct:6.1f}%")

    # ── Final verdict ──
    print(f"\n{'=' * 70}")
    all_ok = True
    checks = [
        ("R² > 0.90", r2 > 0.90, f"R²={r2:.4f}"),
        ("MAE < 2.0°", mae < 2.0, f"MAE={mae:.3f}°"),
        ("|Bias| < 0.5°", abs(bias) < 0.5, f"Bias={bias:+.3f}°"),
        ("CDEM covers >60°N", cdem_only.sum() > 0,
         f"{cdem_only.sum():,} CDEM-only pixels"),
    ]
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  [{status}] {desc}: {detail}")

    print(f"\n  OVERALL: {'PASS — CDEM is a reliable replacement for SRTM' if all_ok else 'ISSUES FOUND — investigate before using CDEM'}")
    print(f"{'=' * 70}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
