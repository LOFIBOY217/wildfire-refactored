"""
Temperature Scaling Calibration for Wildfire Forecast
======================================================
Finds the optimal temperature parameter T such that

    calibrated_prob = sigmoid(logit / T)

minimises negative log-likelihood (NLL) on the validation period.

Why this is needed
------------------
The model is trained on balanced mini-batches (1 positive patch : 3 negative
patches) and uses a high pos_weight in BCEWithLogitsLoss.  During evaluation
the full spatial grid is used, where fires occupy <0.1 % of pixels.  The model
therefore outputs much higher probabilities than the true fire rate, causing
BSS << 0 and Brier >> BS_climatology.

Temperature scaling (Guo et al., 2017) fixes this with a single scalar T > 1
that "cools down" the model's confidence without changing the rank order.
AUC and AP are unchanged; BSS and Brier improve significantly.

Usage
-----
    python -m src.evaluation.calibrate_forecast \\
        --config configs/paths_windows.yaml \\
        --forecast_dir outputs/transformer7d_cwfis_fire_prob \\
        --eval_start 2024-05-01 \\
        --eval_end   2024-10-31 \\
        --output_T   checkpoints/temperature.json

    # Then re-evaluate with calibration applied:
    python -m src.evaluation.evaluate_forecast_cwfis \\
        --config configs/paths_windows.yaml \\
        --calibration_file checkpoints/temperature.json
"""

import argparse
import glob
import json
import os
from datetime import datetime, timedelta

import numpy as np
import rasterio
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_dilation

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

from src.data_ops.processing.rasterize_hotspots import (
    load_hotspot_data,
    _build_transformer,
    _rasterize_points,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _nll(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy (negative log-likelihood)."""
    p = np.clip(y_pred, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def _bss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    clim = float(y_true.mean())
    bs_clim = clim * (1.0 - clim)
    bs_raw  = _brier(y_true, y_pred)
    return (1.0 - bs_raw / bs_clim) if bs_clim > 1e-12 else float("nan")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute temperature scaling parameter T for forecast calibration"
    )
    add_config_argument(ap)
    ap.add_argument("--forecast_dir", type=str, default=None,
                    help="Directory containing forecast GeoTIFFs "
                         "(default: outputs/transformer7d_cwfis_fire_prob)")
    ap.add_argument("--eval_start", type=str, default="2024-05-01")
    ap.add_argument("--eval_end",   type=str, default="2024-10-31")
    ap.add_argument("--forecast_horizon", type=int, default=7)
    ap.add_argument("--dilate_radius", type=int, default=0)
    ap.add_argument("--output_T", type=str, default=None,
                    help="Path to save temperature JSON (default: checkpoints/temperature.json)")
    ap.add_argument("--max_neg_per_day", type=int, default=50_000,
                    help="Max negative pixels to sample per (date, lead) pair "
                         "to keep memory bounded. Default: 50000.")
    args = ap.parse_args()

    def _parse_date(s):
        parts = s.split("-")
        return datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()

    cfg          = load_config(args.config)
    forecast_dir = args.forecast_dir or os.path.join(
        get_path(cfg, "output_dir"), "transformer7d_cwfis_fire_prob"
    )
    hotspot_csv = get_path(cfg, "hotspot_csv")
    output_T    = args.output_T or os.path.join(
        get_path(cfg, "checkpoint_dir", default="checkpoints"), "temperature.json"
    )
    eval_start  = _parse_date(args.eval_start)
    eval_end    = _parse_date(args.eval_end)

    print("=" * 70)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 70)
    print(f"Forecast dir  : {forecast_dir}")
    print(f"Period        : {eval_start} to {eval_end}")
    print(f"Output T file : {output_T}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load hotspot data
    # ------------------------------------------------------------------
    print("\nLoading CWFIS hotspot records...")
    hotspot_df = load_hotspot_data(hotspot_csv)
    hotspot_by_date: dict = {}
    for d, g in hotspot_df.groupby("date"):
        hotspot_by_date[d] = (
            g["field_latitude"].values,
            g["field_longitude"].values,
        )
    print(f"  Unique fire dates: {len(hotspot_by_date):,}")

    # ------------------------------------------------------------------
    # Get raster profile
    # ------------------------------------------------------------------
    fwi_dir   = get_path(cfg, "fwi_dir")
    fwi_files = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
    with rasterio.open(fwi_files[0]) as src:
        profile = src.profile
        H, W    = profile["height"], profile["width"]
        nodata  = profile.get("nodata") or -9999
    transformer = _build_transformer(profile)
    raster_tf   = profile["transform"]

    if args.dilate_radius > 0:
        r = args.dilate_radius
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk   = (xx ** 2 + yy ** 2 <= r ** 2)
    else:
        disk = None

    # ------------------------------------------------------------------
    # Collect logits + labels from all validation predictions
    # ------------------------------------------------------------------
    print("\nCollecting predictions for calibration...")
    all_logits = []
    all_labels = []
    total_fire_pixels  = 0   # true positive count across all (date, lead) pairs
    total_valid_pixels = 0   # total valid pixel count across all (date, lead) pairs

    eval_dates = []
    cur = eval_start
    while cur <= eval_end:
        eval_dates.append(cur)
        cur += timedelta(days=1)

    rng = np.random.default_rng(seed=42)
    n_files_found = 0

    for base_date in eval_dates:
        base_dir = os.path.join(forecast_dir, base_date.strftime("%Y%m%d"))
        if not os.path.exists(base_dir):
            continue

        for lead in range(1, args.forecast_horizon + 1):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            pred_file = os.path.join(
                base_dir, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
            )
            if not os.path.exists(pred_file):
                continue
            n_files_found += 1

            with rasterio.open(pred_file) as src:
                y_pred_raw = src.read(1).flatten()

            lats, lons = hotspot_by_date.get(
                target_date, (np.array([]), np.array([]))
            )
            fire_raster = _rasterize_points(
                lats, lons, transformer, raster_tf, H, W
            )
            if disk is not None:
                fire_raster = binary_dilation(fire_raster, structure=disk).astype(np.uint8)
            y_true_raw = fire_raster.flatten()

            valid  = (y_pred_raw != nodata) & np.isfinite(y_pred_raw)
            y_pred = y_pred_raw[valid].astype(np.float32)
            y_true = y_true_raw[valid].astype(np.float32)

            # Track true positive rate across the FULL grid (before sampling)
            total_fire_pixels  += int(y_true.sum())
            total_valid_pixels += int(valid.sum())

            # Convert probability -> logit
            p      = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
            logits = np.log(p / (1.0 - p))

            # Sample: all positives + up to max_neg negatives
            pos_idx = np.where(y_true == 1)[0]
            neg_idx = np.where(y_true == 0)[0]
            if len(neg_idx) > args.max_neg_per_day:
                neg_idx = rng.choice(neg_idx, size=args.max_neg_per_day, replace=False)
            idx = np.concatenate([pos_idx, neg_idx])

            all_logits.append(logits[idx])
            all_labels.append(y_true[idx])

        if base_date.day == 1:  # progress update monthly
            print(f"  {base_date} (files found so far: {n_files_found})", flush=True)

    if not all_logits:
        print("No prediction files found. Check --forecast_dir and date range.")
        return

    logits_all = np.concatenate(all_logits, dtype=np.float32)
    labels_all = np.concatenate(all_labels, dtype=np.float32)

    # True positive rate on the full grid (no sampling bias)
    true_rate   = total_fire_pixels / max(total_valid_pixels, 1)
    sample_rate = float(labels_all.mean())

    print(f"\n  Total samples collected : {len(logits_all):,}")
    print(f"  Positive (fire) samples : {int(labels_all.sum()):,}  "
          f"({sample_rate * 100:.3f}%  — oversampled for calibration)")
    print(f"  True positive rate      : {true_rate * 100:.4f}%  "
          f"(from {total_valid_pixels:,} total grid pixels)")
    print(f"  Oversampling factor     : {sample_rate / true_rate:.1f}x")
    print(f"  Files processed         : {n_files_found}")

    # ------------------------------------------------------------------
    # Importance weights: correct for positive oversampling
    # The sample has `sample_rate` positives but the true grid has `true_rate`.
    # Weight each sample so the effective distribution matches the true grid.
    # ------------------------------------------------------------------
    w_pos = true_rate   / max(sample_rate,        1e-12)
    w_neg = (1.0 - true_rate) / max(1.0 - sample_rate, 1e-12)
    weights = np.where(labels_all == 1, w_pos, w_neg).astype(np.float32)
    # Normalise so mean weight = 1 (keeps NLL on the same scale)
    weights /= weights.mean()

    print(f"\n  Importance weights: w_pos={w_pos:.4f}, w_neg={w_neg:.4f}")
    print("  (positives are downweighted to match true grid frequency)")

    # ------------------------------------------------------------------
    # Evaluate uncalibrated model
    # ------------------------------------------------------------------
    prob_uncalib = _sigmoid(logits_all)
    nll_before   = _nll(labels_all, prob_uncalib)
    bss_before   = _bss(labels_all, prob_uncalib)
    brier_before = _brier(labels_all, prob_uncalib)
    print(f"\nBefore calibration (on sample, unweighted):")
    print(f"  NLL   : {nll_before:.6f}")
    print(f"  Brier : {brier_before:.6f}")
    print(f"  BSS   : {bss_before:.2f}  (at {sample_rate*100:.3f}% positive rate)")

    # ------------------------------------------------------------------
    # Optimise T using importance-weighted NLL
    # This finds T calibrated to the TRUE positive rate (~0.01-0.1%)
    # rather than the oversampled rate (~1.7%).
    # ------------------------------------------------------------------
    print(f"\nOptimising T with importance-weighted NLL "
          f"(target rate: {true_rate*100:.4f}%)...")

    def objective_weighted(T: float) -> float:
        if T <= 0:
            return 1e9
        prob = _sigmoid(logits_all / T)
        p    = np.clip(prob, 1e-9, 1.0 - 1e-9)
        nll  = -(labels_all * np.log(p) + (1.0 - labels_all) * np.log(1.0 - p))
        return float(np.mean(weights * nll))

    result = minimize_scalar(objective_weighted, bounds=(0.01, 200.0), method="bounded")
    T_opt  = float(result.x)
    print(f"  Optimal T : {T_opt:.4f}")
    if T_opt > 1:
        print(f"  (T > 1: model outputs are too high — calibration scales them down)")
    else:
        print(f"  (T < 1: unusual — model may need retraining rather than calibration)")

    # ------------------------------------------------------------------
    # Evaluate calibrated model
    # ------------------------------------------------------------------
    prob_calib = _sigmoid(logits_all / T_opt)
    nll_after  = _nll(labels_all, prob_calib)
    bss_after  = _bss(labels_all, prob_calib)
    brier_after = _brier(labels_all, prob_calib)

    print(f"\nAfter calibration (T={T_opt:.4f}, evaluated on sample):")
    print(f"  NLL   : {nll_after:.6f}  (was {nll_before:.6f})")
    print(f"  Brier : {brier_after:.6f}  (was {brier_before:.6f})")
    print(f"  BSS   : {bss_after:.4f}  (was {bss_before:.2f})")
    print()
    print("  NOTE: Sample BSS uses oversampled positive rate "
          f"({sample_rate*100:.3f}%), not the true rate ({true_rate*100:.4f}%).")
    print("  Run evaluate_forecast_cwfis.py --calibration_file to see BSS on the full grid.")

    # ------------------------------------------------------------------
    # Save T
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_T)), exist_ok=True)
    calib_data = {
        "temperature":       T_opt,
        "eval_start":        str(eval_start),
        "eval_end":          str(eval_end),
        "n_samples":         int(len(logits_all)),
        "true_positive_rate": true_rate,
        "sample_positive_rate": sample_rate,
        "nll_before":        nll_before,
        "nll_after":         nll_after,
        "bss_before":        bss_before,
        "bss_after":         bss_after,
    }
    with open(output_T, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\nSaved: {output_T}")
    print("\nNext step — re-evaluate with calibration:")
    print(f"  python -m src.evaluation.evaluate_forecast_cwfis \\")
    print(f"      --config <your_config.yaml> \\")
    print(f"      --calibration_file {output_T}")
    print("=" * 70)


if __name__ == "__main__":
    main()
