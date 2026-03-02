"""
Forecast Evaluation Pipeline (CWFIS Hotspot Ground Truth)
==========================================================
Evaluates model predictions against CWFIS satellite hotspot records
instead of CIFFC fire-perimeter data.

Why this matters
----------------
Models trained on CWFIS hotspot labels (dilated or raw) should be
evaluated against the same data source.  Using CIFFC perimeters as
ground truth creates a train/eval mismatch that causes FAR=1.0 and
CSI=0 even when the model has genuinely learned fire patterns.

Additional metrics vs evaluate_forecast.py
------------------------------------------
- AUC-PR (Average Precision):  area under the precision-recall curve.
  More informative than AUC-ROC for severely imbalanced datasets.
- BSS (Brier Skill Score):     1 - BS / BS_clim.
  Measures improvement over a naive climatological baseline.
  BSS > 0 → model beats climatology; BSS < 0 → worse than climatology.

Usage (Windows server):
    python -m src.evaluation.evaluate_forecast_cwfis \\
        --config configs/default.yaml \\
        --forecast_dir outputs/transformer7d_cwfis_fire_prob \\
        --eval_start 2022-05-01 \\
        --eval_end   2024-10-31 \\
        --run_name   transformer7d_cwfis_hotspot

Optional dilation of ground truth (for sensitivity analysis):
    ... --dilate_radius 5
"""

import argparse
import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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
from src.evaluation.metrics import compute_confusion_metrics
from src.evaluation.visualize import (
    plot_metrics_vs_threshold,
    plot_performance_by_lead,
    plot_confusion_matrix_heatmap,
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate Wildfire Forecast vs CWFIS Hotspots")
    add_config_argument(ap)
    ap.add_argument("--forecast_dir", type=str, default=None,
                    help="Directory containing forecast GeoTIFFs "
                         "(default: outputs/transformer7d_cwfis_fire_prob)")
    ap.add_argument("--eval_start", type=str, default="2022-05-01",
                    help="First evaluation date (inclusive). Default: 2022-05-01")
    ap.add_argument("--eval_end",   type=str, default="2024-10-31",
                    help="Last evaluation date (inclusive). Default: 2024-10-31")
    ap.add_argument("--forecast_horizon", type=int, default=7)
    ap.add_argument("--run_name", type=str, default=None,
                    help="Sub-directory name for outputs. Defaults to basename of --forecast_dir.")
    ap.add_argument("--dilate_radius", type=int, default=0,
                    help="Dilate CWFIS hotspot ground truth by N pixels before evaluation. "
                         "0 = raw single-pixel hotspots (default). "
                         "Set to 5 to match the training-time dilation (sensitivity test).")
    args = ap.parse_args()

    cfg          = load_config(args.config)
    forecast_dir = args.forecast_dir or os.path.join(
        get_path(cfg, "output_dir"), "transformer7d_cwfis_fire_prob"
    )
    hotspot_csv  = get_path(cfg, "hotspot_csv")
    fwi_dir      = get_path(cfg, "fwi_dir")
    model_name   = args.run_name or os.path.basename(forecast_dir.rstrip("/\\"))
    output_dir   = os.path.join(
        get_path(cfg, "output_dir"), "evaluation_confusion_matrix", model_name
    )

    thresholds = cfg.get("evaluation", {}).get(
        "thresholds",
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    )

    def _parse_date(s):
        parts = s.split("-")
        return datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()

    eval_start = _parse_date(args.eval_start)
    eval_end   = _parse_date(args.eval_end)

    print("=" * 70)
    print("WILDFIRE FORECAST EVALUATION  [CWFIS Hotspot Ground Truth]")
    print("=" * 70)
    print(f"Evaluation period  : {eval_start} to {eval_end}")
    print(f"Forecast horizon   : {args.forecast_horizon} days")
    print(f"Thresholds         : {thresholds}")
    print(f"Ground truth       : CWFIS hotspots (dilate_radius={args.dilate_radius})")
    print(f"Output dir         : {output_dir}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # STEP 1  Load CWFIS hotspot records
    # ----------------------------------------------------------------
    print("\n[STEP 1] Loading CWFIS hotspot records...")
    hotspot_df = load_hotspot_data(hotspot_csv)
    print(f"  Total records: {len(hotspot_df):,}")
    print(f"  Date range   : {hotspot_df['date'].min()} to {hotspot_df['date'].max()}")

    # ----------------------------------------------------------------
    # STEP 2  Get raster profile from FWI grid
    # ----------------------------------------------------------------
    print("\n[STEP 2] Getting raster profile...")
    fwi_files = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
    with rasterio.open(fwi_files[0]) as src:
        profile = src.profile
        H, W    = profile["height"], profile["width"]
        nodata_raw = profile.get("nodata")
        nodata     = nodata_raw if nodata_raw is not None else -9999
    print(f"  Raster: {H} x {W}  nodata={nodata}")

    # Pre-build dilation kernel if needed
    if args.dilate_radius > 0:
        r = args.dilate_radius
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk   = (xx ** 2 + yy ** 2 <= r ** 2)
        print(f"  Dilation kernel: radius={r}  size={disk.shape}  pixels={disk.sum()}")
    else:
        disk = None

    # ------------------------------------------------------------------
    # Performance optimisation: pre-group hotspot records by date and
    # build the coordinate transformer ONCE — avoids O(9.2M) pandas scan
    # and repeated pyproj initialisation inside the evaluation loop.
    # ------------------------------------------------------------------
    print("  Pre-grouping hotspot data by date (one-time cost)...")
    hotspot_by_date: dict = {}
    for d, g in hotspot_df.groupby("date"):
        hotspot_by_date[d] = (
            g["field_latitude"].values,
            g["field_longitude"].values,
        )
    print(f"  Unique fire dates indexed: {len(hotspot_by_date):,}")

    transformer  = _build_transformer(profile)   # built once, reused for every call
    raster_tf    = profile["transform"]
    raster_H     = profile["height"]
    raster_W     = profile["width"]

    # ----------------------------------------------------------------
    # STEP 3  Evaluate predictions
    # ----------------------------------------------------------------
    print("\n[STEP 3] Evaluating predictions...")
    all_results = []

    eval_dates = []
    cur = eval_start
    while cur <= eval_end:
        eval_dates.append(cur)
        cur += timedelta(days=1)

    for base_date in eval_dates:
        base_date_str = base_date.strftime("%Y%m%d")
        base_dir = os.path.join(forecast_dir, base_date_str)
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

            with rasterio.open(pred_file) as src:
                y_pred = src.read(1).flatten()

            # Build ground truth: O(1) dict lookup + pre-built transformer
            lats, lons = hotspot_by_date.get(
                target_date, (np.array([]), np.array([]))
            )
            fire_raster = _rasterize_points(
                lats, lons, transformer, raster_tf, raster_H, raster_W
            )
            if disk is not None:
                fire_raster = binary_dilation(fire_raster, structure=disk).astype(np.uint8)
            y_true = fire_raster.flatten()

            valid_mask = (y_pred != nodata) & np.isfinite(y_pred)

            # --- Pre-masked arrays (computed once, reused for every threshold) ---
            # Avoids repeating expensive fancy-indexing inside compute_confusion_metrics.
            y_true_v = y_true[valid_mask].astype(np.float32)
            y_pred_v = y_pred[valid_mask].astype(np.float32)

            # --- Sub-sample for AUC / AUC-PR (threshold-independent) ---
            # Sorting 5.9 M floats per call × 7 calls/pair × 1288 pairs = 8–12 h.
            # Sub-sampling to (all positives + 100 k random negatives) gives an
            # essentially identical ranking metric in < 1 ms instead of 3–5 s.
            _AUC_MAX_NEG = 100_000
            _pos = np.where(y_true_v == 1)[0]
            _neg = np.where(y_true_v == 0)[0]
            if len(_neg) > _AUC_MAX_NEG:
                _rng = np.random.default_rng(seed=base_date.toordinal() + lead)
                _neg = _rng.choice(_neg, size=_AUC_MAX_NEG, replace=False)
            _sub = np.concatenate([_pos, _neg])
            y_true_s = y_true_v[_sub]
            y_pred_s = y_pred_v[_sub]

            if len(np.unique(y_true_s)) > 1:
                ap      = float(average_precision_score(y_true_s, y_pred_s))
                auc_pre = float(roc_auc_score(y_true_s, y_pred_s))
            else:
                ap = auc_pre = float("nan")

            # BSS uses full valid arrays (just mean + MSE, O(n), no sort needed)
            clim_prob = float(y_true_v.mean())
            bs_raw    = float(brier_score_loss(y_true_v, y_pred_v))
            bs_clim   = clim_prob * (1.0 - clim_prob)   # BS of constant-climatology forecast
            bss = (1.0 - bs_raw / bs_clim) if bs_clim > 0 else float("nan")

            # --- Threshold-dependent metrics ---
            # Pass pre-masked arrays directly (no nodata_mask) and skip_auc=True
            # so compute_confusion_metrics does NOT call roc_auc_score internally.
            for threshold in thresholds:
                metrics = compute_confusion_metrics(
                    y_true_v, y_pred_v, threshold, skip_auc=True
                )
                if metrics is not None:
                    metrics["auc"]         = auc_pre   # sub-sampled value
                    metrics["base_date"]   = base_date
                    metrics["target_date"] = target_date
                    metrics["lead_time"]   = lead
                    metrics["ap"]          = ap
                    metrics["bss"]         = bss
                    all_results.append(metrics)

        # Per-base_date progress line so operator knows the script is alive
        print(f"  {base_date}: done — total evaluations so far: {len(all_results):,}",
              flush=True)

    print(f"  Total evaluations: {len(all_results):,}")

    if not all_results:
        print("\nNo evaluations completed. Check that forecast tifs exist "
              f"for {eval_start}–{eval_end} under {forecast_dir}")
        return

    # ----------------------------------------------------------------
    # STEP 4  Summary statistics
    # ----------------------------------------------------------------
    print("\n[STEP 4] Computing summary statistics...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    threshold_summary = results_df.groupby("threshold").agg({
        "pod":  "mean",
        "far":  "mean",
        "csi":  "mean",
        "bias": "mean",
        "precision": "mean",
        "f1":   "mean",
        "brier": "mean",
        "auc":  "mean",
        "ap":   "mean",
        "bss":  "mean",
        "tp":   "sum",
        "fp":   "sum",
        "tn":   "sum",
        "fn":   "sum",
    }).round(4)

    print("\n" + "=" * 70)
    print("PERFORMANCE BY THRESHOLD")
    print("=" * 70)
    print(threshold_summary[
        ["pod", "far", "csi", "bias", "auc", "ap", "bss", "brier"]
    ].to_string())

    threshold_summary.to_csv(os.path.join(output_dir, "metrics_by_threshold.csv"))

    best_threshold = threshold_summary["csi"].idxmax()
    print(f"\nBest threshold by CSI: {best_threshold}")

    lead_summary = (
        results_df[results_df["threshold"] == best_threshold]
        .groupby("lead_time")
        .agg({
            "pod":   "mean",
            "far":   "mean",
            "csi":   "mean",
            "bias":  "mean",
            "brier": "mean",
            "auc":   "mean",
            "ap":    "mean",
            "bss":   "mean",
        })
        .round(4)
    )

    print("\n" + "=" * 70)
    print(f"PERFORMANCE BY LEAD TIME (threshold={best_threshold})")
    print("=" * 70)
    print(lead_summary.to_string())

    lead_summary.to_csv(os.path.join(output_dir, "metrics_by_lead.csv"))

    # ----------------------------------------------------------------
    # STEP 5  Visualizations
    # ----------------------------------------------------------------
    print("\n[STEP 5] Creating visualizations...")
    plot_metrics_vs_threshold(
        threshold_summary,
        os.path.join(output_dir, "metrics_vs_threshold.png")
    )
    plot_performance_by_lead(
        lead_summary,
        os.path.join(output_dir, "performance_by_lead.png")
    )

    best_results = results_df[results_df["threshold"] == best_threshold]
    plot_confusion_matrix_heatmap(
        tp=int(best_results["tp"].sum()),
        fp=int(best_results["fp"].sum()),
        tn=int(best_results["tn"].sum()),
        fn=int(best_results["fn"].sum()),
        threshold=best_threshold,
        output_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
