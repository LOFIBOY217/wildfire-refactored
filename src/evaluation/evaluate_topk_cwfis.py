"""
Top-K Evaluation Pipeline (CWFIS Hotspot Ground Truth)
=======================================================
Evaluates model predictions using **Top-K ranking** instead of fixed probability
thresholds.  This sidesteps two fundamental problems with the threshold approach
on Canada's rare-fire landscape:

1. **FAR inflation**: Canada has ~6.2 M valid pixels; even at threshold=0.3 the
   number of false alarms is 1000× the number of real fires (FAR ≈ 0.999).
2. **BSS collapse**: BSS = 1 - BS/BS_clim.  When the fire rate is 0.014%,
   BS_clim = p*(1-p) ≈ 7e-5.  Any small absolute calibration error makes BSS
   extremely negative.

Top-K evaluation asks: "Among the K pixels with the highest predicted
probability, how many actually caught fire?"  It depends only on the model's
ranking ability (measured by AUC), not on probability calibration.

Metrics
-------
- **Precision@K** : tp / K  — fraction of flagged pixels that were real fires
- **Recall@K**    : tp / n_fires  — fraction of fires captured in the top-K
- **Lift@K**      : Precision@K / baseline  — improvement over random guessing
  * Lift=1  → same as random; Lift=100 → 100× better than random
- **CSI@K**       : tp / (tp + fp + fn)  — Critical Success Index
  * Penalises both false alarms (fp = K-tp) and misses (fn = n_fires-tp)
  * Standard metric in meteorological and wildfire forecasting
  * CSI=1 perfect, CSI=0 useless
- **ETS@K**       : (tp - tp_random) / (tp + fp + fn - tp_random)
                    where tp_random = K × n_fires / n_valid
  * Equitable Threat Score (Gilbert Skill Score) — WMO-recommended NWP metric
  * Subtracts the hits expected by random selection; answers "how much better
    than random is our CSI?"
  * ETS=0 → random skill;  ETS=1 → perfect;  ETS<0 → worse than random
  * Relationship to CSI:  CSI = F1 / (2 - F1),  ETS ≠ CSI (ETS adds chance correction)
- **PR-AUC**      : Area under the full precision-recall curve
  * Random baseline ≈ fire_rate ≈ 0.00014  (dilate_radius=0)
  * A well-calibrated model with AUC=0.85 should reach PR-AUC ≈ 0.01–0.05
  * Much more sensitive than ROC-AUC for severely imbalanced data

Usage (Windows server):
    python -m src.evaluation.evaluate_topk_cwfis \\
        --config configs/paths_windows.yaml \\
        --forecast_dir outputs/transformer7d_cwfis_fire_prob \\
        --eval_start 2024-05-01 \\
        --eval_end   2024-10-31 \\
        --k_values   100,500,1000,2500,5000,10000,25000,50000,100000 \\
        --dilate_radius 0

Output files (written to <output_dir>/evaluation_topk/<model_name>/):
    all_results_topk.csv     — long format: one row per (date, lead, k)
    summary_by_k.csv         — mean metrics for each K value
    summary_by_lead.csv      — mean metrics per lead time  (K = display_k)
    summary_by_month.csv     — mean metrics per calendar month (K = display_k)
"""

import argparse
import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation
from sklearn.metrics import average_precision_score

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

# ---------------------------------------------------------------------------
# Month name helper
# ---------------------------------------------------------------------------
_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def main():
    # ----------------------------------------------------------------
    # CLI
    # ----------------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Top-K Wildfire Forecast Evaluation (CWFIS Hotspot Ground Truth)"
    )
    add_config_argument(ap)
    ap.add_argument(
        "--forecast_dir", type=str, default=None,
        help="Directory containing forecast GeoTIFFs "
             "(default: outputs/transformer7d_cwfis_fire_prob)",
    )
    ap.add_argument(
        "--eval_start", type=str, default="2022-05-01",
        help="First evaluation date (inclusive).  Default: 2022-05-01",
    )
    ap.add_argument(
        "--eval_end", type=str, default="2024-10-31",
        help="Last evaluation date (inclusive).  Default: 2024-10-31",
    )
    ap.add_argument(
        "--forecast_horizon", type=int, default=7,
        help="Number of lead days to evaluate.  Default: 7",
    )
    ap.add_argument(
        "--k_values", type=str,
        default="100,500,1000,2500,5000,10000,25000,50000,100000",
        help="Comma-separated list of K values.  "
             "Default: 100,500,1000,2500,5000,10000,25000,50000,100000",
    )
    ap.add_argument(
        "--display_k", type=int, default=5000,
        help="K value used for the lead-time and month breakdowns.  Default: 5000",
    )
    ap.add_argument(
        "--dilate_radius", type=int, default=0,
        help="Dilate CWFIS hotspot ground truth by N pixels before evaluation.  "
             "0 = raw single-pixel hotspots (default).  "
             "Set to 5 to match the training-time dilation (sensitivity test).",
    )
    ap.add_argument(
        "--run_name", type=str, default=None,
        help="Sub-directory name for outputs.  Defaults to basename of --forecast_dir.",
    )
    args = ap.parse_args()

    # Parse K values
    k_values = sorted(set(int(k.strip()) for k in args.k_values.split(",")))
    display_k = args.display_k
    if display_k not in k_values:
        k_values = sorted(set(k_values) | {display_k})
        print(f"[INFO] display_k={display_k} added to k_values list automatically.")

    # Parse config / paths
    cfg          = load_config(args.config)
    forecast_dir = args.forecast_dir or os.path.join(
        get_path(cfg, "output_dir"), "transformer7d_cwfis_fire_prob"
    )
    hotspot_csv  = get_path(cfg, "hotspot_csv")
    fwi_dir      = get_path(cfg, "fwi_dir")
    model_name   = args.run_name or os.path.basename(forecast_dir.rstrip("/\\"))
    output_dir   = os.path.join(
        get_path(cfg, "output_dir"), "evaluation_topk", model_name
    )

    def _parse_date(s):
        parts = s.split("-")
        return datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()

    eval_start = _parse_date(args.eval_start)
    eval_end   = _parse_date(args.eval_end)

    print("=" * 70)
    print("WILDFIRE FORECAST — TOP-K EVALUATION  [CWFIS Hotspot Ground Truth]")
    print("=" * 70)
    print(f"Evaluation period  : {eval_start} to {eval_end}")
    print(f"Forecast horizon   : {args.forecast_horizon} days")
    print(f"K values           : {k_values}")
    print(f"Display K          : {display_k}")
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

    # Pre-group by date for O(1) lookup inside the loop
    hotspot_by_date: dict = {}
    for d, g in hotspot_df.groupby("date"):
        hotspot_by_date[d] = (
            g["field_latitude"].values,
            g["field_longitude"].values,
        )
    print(f"  Unique fire dates indexed: {len(hotspot_by_date):,}")

    # ----------------------------------------------------------------
    # STEP 2  Raster profile from FWI grid
    # ----------------------------------------------------------------
    print("\n[STEP 2] Getting raster profile from FWI grid...")
    fwi_files = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
    if not fwi_files:
        raise FileNotFoundError(f"No .tif files found in fwi_dir: {fwi_dir}")
    with rasterio.open(fwi_files[0]) as src:
        profile    = src.profile
        H, W       = profile["height"], profile["width"]
        nodata_raw = profile.get("nodata")
        nodata     = nodata_raw if nodata_raw is not None else -9999
        tf         = profile["transform"]
        # Pixel area in km².  CRS is EPSG:3978 (NAD83/Canada Atlas Lambert),
        # a projected CRS with units in METRES.  tf.a = pixel width in metres,
        # tf.e = pixel height in metres (negative).  Divide by 1e6 to convert m² → km².
        # For the FWI grid: 2000 m × 2000 m = 4 km² per pixel.
        pixel_km2 = abs(tf.a * tf.e) / 1_000_000.0
    print(f"  Raster shape   : {H} × {W}  (nodata={nodata})")
    print(f"  Pixel area     : ~{pixel_km2:.2f} km²  (rough estimate)")

    # Build optional dilation kernel
    if args.dilate_radius > 0:
        r = args.dilate_radius
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk   = (xx ** 2 + yy ** 2 <= r ** 2)
        print(f"  Dilation kernel: radius={r}  size={disk.shape}  pixels={int(disk.sum())}")
    else:
        disk = None

    transformer = _build_transformer(profile)   # coordinate transformer, built once
    raster_tf   = profile["transform"]
    raster_H    = profile["height"]
    raster_W    = profile["width"]

    # ----------------------------------------------------------------
    # STEP 3  Evaluation loop
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

        any_lead_done = False
        for lead in range(1, args.forecast_horizon + 1):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            pred_file = os.path.join(
                base_dir, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif"
            )
            if not os.path.exists(pred_file):
                continue

            # --- Load prediction raster ---
            with rasterio.open(pred_file) as src:
                y_pred_2d = src.read(1)

            # --- Build ground truth raster ---
            lats, lons = hotspot_by_date.get(
                target_date, (np.array([]), np.array([]))
            )
            fire_raster = _rasterize_points(
                lats, lons, transformer, raster_tf, raster_H, raster_W
            )
            if disk is not None:
                fire_raster = binary_dilation(fire_raster, structure=disk).astype(np.uint8)

            # --- Flatten and mask nodata / non-finite ---
            y_pred = y_pred_2d.flatten()
            y_true = fire_raster.flatten()

            valid_mask = (y_pred != nodata) & np.isfinite(y_pred)
            probs  = y_pred[valid_mask].astype(np.float32)
            labels = y_true[valid_mask].astype(np.int8)

            n_valid = int(valid_mask.sum())
            n_fires = int(labels.sum())
            baseline = n_fires / n_valid if n_valid > 0 else 0.0

            # --- PR-AUC (threshold-independent, one value per date×lead) ---
            if n_fires > 0 and len(np.unique(labels)) > 1:
                pr_auc = float(average_precision_score(labels, probs))
            else:
                pr_auc = float("nan")

            # --- Top-K ranking ---
            # Sort pixels by descending predicted probability once, then use
            # cumulative TP for O(n_valid) computation across all K values.
            order         = np.argsort(probs)[::-1]
            sorted_labels = labels[order]
            cum_tp        = np.cumsum(sorted_labels.astype(np.int64))

            for k in k_values:
                if k > n_valid:
                    continue
                tp_k = int(cum_tp[k - 1])
                fp_k = k - tp_k
                fn_k = n_fires - tp_k

                precision_k = tp_k / k
                recall_k    = tp_k / max(n_fires, 1)
                lift_k      = precision_k / max(baseline, 1e-12)
                denom_csi   = tp_k + fp_k + fn_k
                csi_k       = tp_k / denom_csi if denom_csi > 0 else 0.0

                # ETS (Equitable Threat Score / Gilbert Skill Score)
                # tp_random: hits expected by a random forecast that flags K pixels
                tp_random  = k * n_fires / n_valid if n_valid > 0 else 0.0
                denom_ets  = denom_csi - tp_random   # = tp + fp + fn - tp_random
                ets_k      = (tp_k - tp_random) / denom_ets if abs(denom_ets) > 1e-9 else float("nan")

                all_results.append({
                    "base_date":   base_date,
                    "target_date": target_date,
                    "lead_time":   lead,
                    "month":       target_date.month,
                    "k":           k,
                    "n_valid":     n_valid,
                    "n_fires":     n_fires,
                    "tp":          tp_k,
                    "fp":          fp_k,
                    "fn":          fn_k,
                    "precision":   precision_k,
                    "recall":      recall_k,
                    "lift":        lift_k,
                    "csi":         csi_k,
                    "ets":         ets_k,
                    "baseline":    baseline,
                    "pr_auc":      pr_auc,
                })
            any_lead_done = True

        if any_lead_done:
            n_rows = len(all_results)
            print(f"  {base_date}: done — total rows so far: {n_rows:,}", flush=True)

    print(f"\n  Total rows written: {len(all_results):,}")

    if not all_results:
        print("\nNo evaluations completed.  Check that forecast TIFs exist "
              f"for {eval_start}–{eval_end} under {forecast_dir}")
        return

    # ----------------------------------------------------------------
    # Save raw CSV
    # ----------------------------------------------------------------
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "all_results_topk.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ----------------------------------------------------------------
    # STEP 4  Summary: mean over all (date × lead) pairs, per K
    # ----------------------------------------------------------------
    print("\n[STEP 4] Summary by K value...")

    k_summary = (
        df.groupby("k")
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            lift=("lift", "mean"),
            csi=("csi", "mean"),
            ets=("ets", "mean"),
            pr_auc=("pr_auc", "mean"),
            n_pairs=("base_date", "count"),
        )
        .reset_index()
    )
    k_summary["area_km2"] = k_summary["k"] * pixel_km2

    # Global PR-AUC (average over unique date×lead pairs)
    prl_df = df.drop_duplicates(subset=["base_date", "lead_time"])
    mean_pr_auc = prl_df["pr_auc"].mean(skipna=True)
    fire_rate_approx = df["baseline"].mean()

    print()
    print("=" * 80)
    print("TOP-K EVALUATION SUMMARY  (mean over all date × lead pairs)")
    print("=" * 80)
    print(f"  mean PR-AUC = {mean_pr_auc:.5f}   "
          f"(random baseline ≈ {fire_rate_approx:.5f})")
    print("=" * 80)
    print(f"  {'K':>8}  {'Precision@K':>12}  {'Recall@K':>10}  "
          f"{'Lift@K':>9}  {'CSI@K':>7}  {'ETS@K':>7}  {'area_km2':>14}")
    print("  " + "-" * 76)
    for _, row in k_summary.iterrows():
        k_int = int(row["k"])
        print(
            f"  {k_int:>8,}  "
            f"{row['precision'] * 100:>10.3f}%  "
            f"{row['recall'] * 100:>9.3f}%  "
            f"{row['lift']:>9.1f}x  "
            f"{row['csi']:>7.4f}  "
            f"{row['ets']:>7.4f}  "
            f"{row['area_km2']:>11,.0f} km²"
        )
    print()

    k_summary.to_csv(os.path.join(output_dir, "summary_by_k.csv"), index=False)

    # ----------------------------------------------------------------
    # STEP 5  Breakdown by lead time  (at display_k)
    # ----------------------------------------------------------------
    print(f"[STEP 5] Breakdown by lead time  (K={display_k:,})")
    lead_df = df[df["k"] == display_k].copy()

    lead_summary = (
        lead_df.groupby("lead_time")
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            lift=("lift", "mean"),
            csi=("csi", "mean"),
            ets=("ets", "mean"),
            pr_auc=("pr_auc", "mean"),
            n_pairs=("base_date", "count"),
        )
        .reset_index()
    )

    print()
    print(f"  {'lead':>5}  {'Precision@K':>12}  {'Recall@K':>10}  "
          f"{'Lift@K':>9}  {'CSI@K':>7}  {'ETS@K':>7}  {'PR-AUC':>9}")
    print("  " + "-" * 68)
    for _, row in lead_summary.iterrows():
        print(
            f"  {int(row['lead_time']):>5}  "
            f"{row['precision'] * 100:>10.2f}%  "
            f"{row['recall'] * 100:>9.2f}%  "
            f"{row['lift']:>9.1f}x  "
            f"{row['csi']:>7.4f}  "
            f"{row['ets']:>7.4f}  "
            f"{row['pr_auc']:>9.5f}"
        )
    print()

    lead_summary.to_csv(os.path.join(output_dir, "summary_by_lead.csv"), index=False)

    # ----------------------------------------------------------------
    # STEP 6  Breakdown by calendar month  (at display_k)
    # ----------------------------------------------------------------
    print(f"[STEP 6] Breakdown by calendar month  (K={display_k:,})")
    month_summary = (
        lead_df.groupby("month")
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            lift=("lift", "mean"),
            csi=("csi", "mean"),
            ets=("ets", "mean"),
            pr_auc=("pr_auc", "mean"),
            n_pairs=("base_date", "count"),
        )
        .reset_index()
    )

    print()
    print(f"  {'month':>6}  {'Precision@K':>12}  {'Recall@K':>10}  "
          f"{'Lift@K':>9}  {'CSI@K':>7}  {'ETS@K':>7}  {'PR-AUC':>9}  {'n_pairs':>8}")
    print("  " + "-" * 78)
    for _, row in month_summary.iterrows():
        m = int(row["month"])
        print(
            f"  {_MONTH_NAMES.get(m, str(m)):>6}  "
            f"{row['precision'] * 100:>10.2f}%  "
            f"{row['recall'] * 100:>9.2f}%  "
            f"{row['lift']:>9.1f}x  "
            f"{row['csi']:>7.4f}  "
            f"{row['ets']:>7.4f}  "
            f"{row['pr_auc']:>9.5f}  "
            f"{int(row['n_pairs']):>8,}"
        )
    print()

    month_summary.to_csv(os.path.join(output_dir, "summary_by_month.csv"), index=False)

    # ----------------------------------------------------------------
    # Sanity check hints
    # ----------------------------------------------------------------
    print("=" * 72)
    print("SANITY CHECKS")
    print("=" * 72)
    max_k = max(k_values)
    max_k_row = k_summary[k_summary["k"] == max_k].iloc[0]
    print(f"  Recall@{max_k:,}  = {max_k_row['recall'] * 100:.2f}%   "
          f"(should be < 100% unless max_k >= n_valid)")
    print(f"  Lift@{k_values[0]:,}   = {k_summary[k_summary['k'] == k_values[0]].iloc[0]['lift']:.1f}x  "
          f"(should be highest — top pixels should be most accurate)")

    # Note: Lift and ETS are expected to be monotone decreasing only when each K
    # yields reliable statistics (i.e. avg TP_k >> 1).  For rare events
    # (fire_rate ~ 0.01%), very small K values (e.g. K=100 → avg TP ≈ 0.05)
    # produce noisy estimates dominated by chance hits, so Lift/ETS may
    # *increase* with K until statistics stabilise.  This is expected behaviour,
    # not a bug.  A rough threshold: K should satisfy K * baseline > 5.
    reliable_k_threshold = 5.0 / max(fire_rate_approx, 1e-12)
    is_monotone_lift = all(
        k_summary["lift"].iloc[i] >= k_summary["lift"].iloc[i + 1]
        for i in range(len(k_summary) - 1)
    )
    lift_note = "YES ✓" if is_monotone_lift else f"NO (expected for K < {reliable_k_threshold:,.0f} with rare events)"
    print(f"  Lift monotone decreasing: {lift_note}")

    is_monotone_ets = all(
        k_summary["ets"].iloc[i] >= k_summary["ets"].iloc[i + 1]
        for i in range(len(k_summary) - 1)
    )
    ets_note = "YES ✓" if is_monotone_ets else f"NO (expected for K < {reliable_k_threshold:,.0f} with rare events)"
    print(f"  ETS  monotone decreasing: {ets_note}")
    min_k_ets = k_summary[k_summary["k"] == k_values[0]].iloc[0]["ets"]
    print(f"  ETS@{k_values[0]:,}  = {min_k_ets:.4f}   (>0 → better than random; 0 → same as random)")
    print(f"  Reliable K threshold     : ~{reliable_k_threshold:,.0f}  (avg TP > 5 per date×lead)")

    print()
    print("=" * 72)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
