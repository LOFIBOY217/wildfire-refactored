"""
Forecast Evaluation Pipeline
==============================
Model-agnostic evaluation: loads predictions, compares with CIFFC ground truth,
computes metrics at multiple thresholds, and generates reports.

Usage:
    python -m src.evaluation.evaluate_forecast --config configs/default.yaml
    python -m src.evaluation.evaluate_forecast --config configs/paths_windows.yaml

Based on evaluate_with_confusion_matrix.py.
"""

import argparse
import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio

from src.config import load_config, get_path, add_config_argument
from src.data.processing.rasterize_fires import load_ciffc_data, rasterize_fires_single
from src.evaluation.metrics import compute_confusion_metrics
from src.evaluation.visualize import (
    plot_metrics_vs_threshold,
    plot_performance_by_lead,
    plot_confusion_matrix_heatmap
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate Wildfire Forecast")
    add_config_argument(ap)
    ap.add_argument("--forecast_dir", type=str, default=None,
                    help="Directory containing forecast GeoTIFFs")
    ap.add_argument("--eval_start", type=str, default="2025-08-01")
    ap.add_argument("--eval_end", type=str, default="2025-12-31")
    ap.add_argument("--forecast_horizon", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    forecast_dir = args.forecast_dir or os.path.join(
        get_path(cfg, 'output_dir'), 'logreg_fire_prob_7day_forecast'
    )
    ciffc_csv = get_path(cfg, 'ciffc_csv')
    fwi_dir = get_path(cfg, 'fwi_dir')
    output_dir = os.path.join(get_path(cfg, 'output_dir'), 'evaluation_confusion_matrix')

    thresholds = cfg.get('evaluation', {}).get('thresholds', [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])

    # Parse dates
    parts = args.eval_start.split('-')
    eval_start = datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()
    parts = args.eval_end.split('-')
    eval_end = datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()

    print("=" * 70)
    print("WILDFIRE FORECAST EVALUATION")
    print("=" * 70)
    print(f"Evaluation period: {eval_start} to {eval_end}")
    print(f"Forecast horizon: {args.forecast_horizon} days")
    print(f"Thresholds: {thresholds}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n[STEP 1] Loading fire records...")
    ciffc_df = load_ciffc_data(ciffc_csv)
    print(f"  Total records: {len(ciffc_df)}")

    print("\n[STEP 2] Getting raster profile...")
    fwi_files = sorted(glob.glob(os.path.join(fwi_dir, "*.tif")))
    with rasterio.open(fwi_files[0]) as src:
        profile = src.profile
        H, W = profile['height'], profile['width']
        nodata = profile.get('nodata', -9999)
    print(f"  Raster: {H} x {W}")

    # Evaluate predictions
    print("\n[STEP 3] Evaluating predictions...")
    all_results = []

    eval_dates = []
    current = eval_start
    while current <= eval_end:
        eval_dates.append(current)
        current += timedelta(days=1)

    for base_date in eval_dates:
        base_date_str = base_date.strftime("%Y%m%d")
        base_dir = os.path.join(forecast_dir, base_date_str)
        if not os.path.exists(base_dir):
            continue

        for lead in range(1, args.forecast_horizon + 1):
            target_date = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            pred_file = os.path.join(base_dir, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif")

            if not os.path.exists(pred_file):
                continue

            with rasterio.open(pred_file) as src:
                y_pred = src.read(1).flatten()

            fire_raster = rasterize_fires_single(ciffc_df, target_date, profile)
            y_true = fire_raster.flatten()

            valid_mask = (y_pred != nodata) & np.isfinite(y_pred)

            for threshold in thresholds:
                metrics = compute_confusion_metrics(y_true, y_pred, threshold, valid_mask)
                if metrics is not None:
                    metrics['base_date'] = base_date
                    metrics['target_date'] = target_date
                    metrics['lead_time'] = lead
                    all_results.append(metrics)

        if len(all_results) % 50 == 0 and len(all_results) > 0:
            print(f"  Processed {len(all_results)} evaluations...")

    print(f"  Total evaluations: {len(all_results)}")

    # Summarize
    print("\n[STEP 4] Computing summary statistics...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)

    # Group by threshold
    threshold_summary = results_df.groupby('threshold').agg({
        'pod': 'mean', 'far': 'mean', 'csi': 'mean', 'bias': 'mean',
        'precision': 'mean', 'f1': 'mean', 'brier': 'mean', 'auc': 'mean',
        'tp': 'sum', 'fp': 'sum', 'tn': 'sum', 'fn': 'sum'
    }).round(4)

    print("\n" + "=" * 70)
    print("PERFORMANCE BY THRESHOLD")
    print("=" * 70)
    print(threshold_summary.to_string())

    threshold_summary.to_csv(os.path.join(output_dir, 'metrics_by_threshold.csv'))

    # Best threshold by CSI
    best_threshold = threshold_summary['csi'].idxmax()
    print(f"\nBest threshold by CSI: {best_threshold}")

    # Group by lead time
    lead_summary = results_df[results_df['threshold'] == best_threshold].groupby('lead_time').agg({
        'pod': 'mean', 'far': 'mean', 'csi': 'mean',
        'bias': 'mean', 'brier': 'mean', 'auc': 'mean'
    }).round(4)

    print("\n" + "=" * 70)
    print(f"PERFORMANCE BY LEAD TIME (threshold={best_threshold})")
    print("=" * 70)
    print(lead_summary.to_string())

    lead_summary.to_csv(os.path.join(output_dir, 'metrics_by_lead.csv'))

    # Visualizations
    print("\n[STEP 5] Creating visualizations...")
    plot_metrics_vs_threshold(threshold_summary, os.path.join(output_dir, 'metrics_vs_threshold.png'))
    plot_performance_by_lead(lead_summary, os.path.join(output_dir, 'performance_by_lead.png'))

    best_results = results_df[results_df['threshold'] == best_threshold]
    plot_confusion_matrix_heatmap(
        tp=int(best_results['tp'].sum()),
        fp=int(best_results['fp'].sum()),
        tn=int(best_results['tn'].sum()),
        fn=int(best_results['fn'].sum()),
        threshold=best_threshold,
        output_path=os.path.join(output_dir, 'confusion_matrix.png')
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
