"""
Evaluate 7-Day FWI Regression Forecast
=======================================
Computes regression metrics (MAE, RMSE, Pearson R, Bias) for predicted FWI
GeoTIFFs against actual FWI observations, broken down by lead day.

Also computes a **persistence baseline**: "predict tomorrow's FWI = today's FWI"
This is the minimum sensible benchmark for any FWI forecasting model.

Expected prediction directory structure:
    pred_dir/
        YYYYMMDD/                          <- base date (day forecast was issued)
            fwi_pred_lead01d_YYYYMMDD.tif
            fwi_pred_lead02d_YYYYMMDD.tif
            ...
            fwi_pred_lead07d_YYYYMMDD.tif

Actual FWI directory:
    fwi_dir/
        fwi_YYYYMMDD.tif  (or similar naming, auto-detected)

Usage:
    python -m src.evaluation.evaluate_fwi_forecast \\
        --pred_dir  outputs/transformer7d_fwi_pred \\
        --fwi_dir   /path/to/fwi_data \\
        --output_dir outputs/evaluation_fwi
"""

import argparse
import glob
import os
import sys
from datetime import date, timedelta
from datetime import datetime as dt

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    from src.utils.date_utils import extract_date_from_filename
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    from pathlib import Path
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "config.py").exists():
            sys.path.insert(0, str(_p))
            break
    from src.utils.date_utils import extract_date_from_filename
    from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _build_fwi_dict(fwi_dir):
    """Map date → file path for all FWI GeoTIFFs."""
    result = {}
    for p in sorted(glob.glob(os.path.join(fwi_dir, "**", "*.tif"), recursive=True)):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            result[d] = p
    return result


def _read_raster(path):
    """Read a GeoTIFF as a float32 numpy array, returning (data, nodata)."""
    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata
    return data, nodata


def _valid_mask(arr, nodata=None):
    """Boolean mask of valid (non-nodata, finite) pixels."""
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= (arr != nodata)
    return mask


def _regression_metrics(y_true_flat, y_pred_flat):
    """Compute MAE, RMSE, Pearson R, and Bias from 1-D arrays."""
    if len(y_true_flat) < 2:
        return dict(mae=np.nan, rmse=np.nan, pearson_r=np.nan, bias=np.nan, n=0)

    diff   = y_pred_flat - y_true_flat
    mae    = float(np.mean(np.abs(diff)))
    rmse   = float(np.sqrt(np.mean(diff ** 2)))
    bias   = float(np.mean(diff))
    try:
        r, _ = pearsonr(y_true_flat, y_pred_flat)
        r = float(r)
    except Exception:
        r = np.nan

    return dict(mae=mae, rmse=rmse, pearson_r=r, bias=bias, n=int(len(y_true_flat)))


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate FWI regression forecasts (MAE / RMSE / Pearson R)"
    )
    add_config_argument(ap)
    ap.add_argument("--pred_dir",    type=str, required=True,
                    help="Root directory of predicted FWI GeoTIFFs "
                         "(sub-folders named YYYYMMDD).")
    ap.add_argument("--fwi_dir",     type=str, default=None,
                    help="Directory containing actual FWI GeoTIFFs. "
                         "If omitted, read from --config.")
    ap.add_argument("--output_dir",  type=str, default="outputs/evaluation_fwi",
                    help="Where to save CSV and plots.")
    ap.add_argument("--max_lead",    type=int, default=7,
                    help="Maximum lead day to evaluate (default=7).")
    ap.add_argument("--sample_frac", type=float, default=1.0,
                    help="Fraction of valid pixels to sample per raster (default=1.0). "
                         "Use <1.0 to speed up evaluation on large grids.")
    args = ap.parse_args()

    # Resolve fwi_dir from config if not provided directly
    if args.fwi_dir is None:
        if args.config is None:
            ap.error("--fwi_dir is required unless --config is provided.")
        cfg = load_config(args.config)
        args.fwi_dir = get_path(cfg, "fwi_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("FWI FORECAST EVALUATION")
    print("=" * 65)
    print(f"  pred_dir   : {args.pred_dir}")
    print(f"  fwi_dir    : {args.fwi_dir}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  max_lead   : {args.max_lead}")
    print("=" * 65)

    # ----------------------------------------------------------------
    # Build FWI observation index
    # ----------------------------------------------------------------
    print("\n[1] Building FWI observation index...")
    fwi_dict = _build_fwi_dict(args.fwi_dir)
    print(f"  Found {len(fwi_dict)} FWI observation files.")
    if not fwi_dict:
        raise RuntimeError(f"No FWI .tif files found in {args.fwi_dir}")

    # ----------------------------------------------------------------
    # Discover prediction base dates
    # ----------------------------------------------------------------
    print("\n[2] Discovering prediction base dates...")
    base_dirs = sorted([
        d for d in glob.glob(os.path.join(args.pred_dir, "????????"))
        if os.path.isdir(d)
    ])
    if not base_dirs:
        raise RuntimeError(f"No YYYYMMDD subdirectories found under {args.pred_dir}")
    print(f"  Found {len(base_dirs)} base date directories.")

    def _parse_dir_date(path):
        name = os.path.basename(path)
        try:
            return date(int(name[:4]), int(name[4:6]), int(name[6:8]))
        except Exception:
            return None

    # ----------------------------------------------------------------
    # Collect metrics per (base_date, lead)
    # ----------------------------------------------------------------
    print("\n[3] Computing metrics...")
    records = []
    rng     = np.random.default_rng(42)

    for base_dir in base_dirs:
        base_date = _parse_dir_date(base_dir)
        if base_date is None:
            continue

        for lead in range(1, args.max_lead + 1):
            target_date     = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")

            # -- predicted FWI --
            pred_path = os.path.join(
                base_dir, f"fwi_pred_lead{lead:02d}d_{target_date_str}.tif"
            )
            if not os.path.exists(pred_path):
                continue

            # -- actual FWI --
            if target_date not in fwi_dict:
                continue

            y_pred, pred_nodata = _read_raster(pred_path)
            y_true, true_nodata = _read_raster(fwi_dict[target_date])

            if y_pred.shape != y_true.shape:
                # Resize if needed (shouldn't happen, but safety)
                continue

            mask = _valid_mask(y_pred, pred_nodata) & _valid_mask(y_true, true_nodata)
            if mask.sum() == 0:
                continue

            y_pred_v = y_pred[mask]
            y_true_v = y_true[mask]

            # Optional pixel subsampling for speed
            if args.sample_frac < 1.0:
                n_keep = max(1, int(len(y_pred_v) * args.sample_frac))
                idx    = rng.choice(len(y_pred_v), size=n_keep, replace=False)
                y_pred_v = y_pred_v[idx]
                y_true_v = y_true_v[idx]

            m = _regression_metrics(y_true_v, y_pred_v)
            m["base_date"]   = base_date.isoformat()
            m["target_date"] = target_date.isoformat()
            m["lead_day"]    = lead
            m["model"]       = "transformer_fwi"
            records.append(m)

        # -- persistence baseline: predict lead-k FWI = base_date FWI --
        if base_date not in fwi_dict:
            continue
        y_base, base_nodata = _read_raster(fwi_dict[base_date])

        for lead in range(1, args.max_lead + 1):
            target_date = base_date + timedelta(days=lead)
            if target_date not in fwi_dict:
                continue

            y_true, true_nodata = _read_raster(fwi_dict[target_date])
            if y_base.shape != y_true.shape:
                continue

            mask = _valid_mask(y_base, base_nodata) & _valid_mask(y_true, true_nodata)
            if mask.sum() == 0:
                continue

            y_base_v = y_base[mask]
            y_true_v = y_true[mask]

            if args.sample_frac < 1.0:
                n_keep   = max(1, int(len(y_base_v) * args.sample_frac))
                idx      = rng.choice(len(y_base_v), size=n_keep, replace=False)
                y_base_v = y_base_v[idx]
                y_true_v = y_true_v[idx]

            m = _regression_metrics(y_true_v, y_base_v)
            m["base_date"]   = base_date.isoformat()
            m["target_date"] = target_date.isoformat()
            m["lead_day"]    = lead
            m["model"]       = "persistence"
            records.append(m)

        print(f"  Processed {base_date}")

    if not records:
        raise RuntimeError("No matching prediction/observation pairs found.")

    df = pd.DataFrame(records)

    # ----------------------------------------------------------------
    # Save detailed CSV
    # ----------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(df)} rows)")

    # ----------------------------------------------------------------
    # Summary table per lead day
    # ----------------------------------------------------------------
    summary = (
        df.groupby(["model", "lead_day"])
          .agg(mae=("mae", "mean"), rmse=("rmse", "mean"),
               pearson_r=("pearson_r", "mean"), bias=("bias", "mean"),
               n_dates=("base_date", "count"))
          .reset_index()
    )
    summary_path = os.path.join(args.output_dir, "summary_by_lead.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("MEAN METRICS BY LEAD DAY")
    print("=" * 65)
    for model_name in summary["model"].unique():
        sub = summary[summary["model"] == model_name]
        print(f"\n  Model: {model_name}")
        print(f"  {'Lead':>5}  {'MAE':>8}  {'RMSE':>8}  {'Pearson R':>10}  {'Bias':>8}")
        for _, row in sub.iterrows():
            print(f"  {int(row.lead_day):>5}  {row.mae:>8.3f}  {row.rmse:>8.3f}  "
                  f"{row.pearson_r:>10.4f}  {row.bias:>8.3f}")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    print("\n[4] Generating plots...")
    leads = sorted(summary["lead_day"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics   = ["rmse", "pearson_r", "mae"]
    titles    = ["RMSE (FWI units)", "Pearson R", "MAE (FWI units)"]
    colors    = {"transformer_fwi": "#E74C3C", "persistence": "#95A5A6"}
    markers   = {"transformer_fwi": "o", "persistence": "s"}

    for ax, metric, title in zip(axes, metrics, titles):
        for model_name in summary["model"].unique():
            sub = summary[summary["model"] == model_name].sort_values("lead_day")
            ax.plot(
                sub["lead_day"], sub[metric],
                label=model_name,
                color=colors.get(model_name, "steelblue"),
                marker=markers.get(model_name, "^"),
                linewidth=2, markersize=6
            )
        ax.set_xlabel("Lead Day")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(leads)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric == "pearson_r":
            ax.set_ylim(0, 1.05)

    plt.suptitle("FWI Forecast Evaluation: Transformer vs Persistence Baseline",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "metrics_vs_lead.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    # ----------------------------------------------------------------
    # Mean AUC equivalent: mean Pearson R across all leads
    # ----------------------------------------------------------------
    for model_name in summary["model"].unique():
        sub       = summary[summary["model"] == model_name]
        mean_r    = sub["pearson_r"].mean()
        mean_rmse = sub["rmse"].mean()
        print(f"\n  {model_name}  |  mean Pearson R = {mean_r:.4f}  |  mean RMSE = {mean_rmse:.3f}")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print(f"  Results: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
