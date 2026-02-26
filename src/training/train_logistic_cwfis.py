"""
Train Logistic Regression Baseline for Wildfire Prediction (CWFIS Hotspot Labels)
===================================================================================
Identical workflow to train_logistic.py, but uses CWFIS satellite hotspot records
(VIIRS-M detections, 2018–2025) as binary fire labels instead of CIFFC reports.

Advantages over CIFFC:
  - 7 years of training data (2018–2025 vs 2023–2025 for CIFFC)
  - Satellite-derived labels: no human reporting delays or missed fires
  - Higher density: ~10-100x more fire pixels per day

Usage:
    python -m src.training.train_logistic_cwfis --config configs/default.yaml
    python -m src.training.train_logistic_cwfis --config configs/default.yaml \\
        --data_start 2018-05-01 --pred_start 2024-05-01 --pred_end 2024-10-31

Output:
    outputs/logreg_cwfis_fire_prob_7day_forecast/
        YYYYMMDD/
            fire_prob_lead01d_YYYYMMDD.tif
            ...
"""

import argparse
import os
import glob
import json
import time
import atexit
from datetime import date, timedelta
from datetime import datetime as dt

import numpy as np
import rasterio

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
from src.utils.date_utils import extract_date_from_filename
from src.utils.raster_io import read_singleband_stack
from src.data_ops.processing.rasterize_hotspots import load_hotspot_data, rasterize_hotspots_batch
from src.models.logistic_baseline import compute_features, sample_training_data, build_logistic_model


def main():
    run_started_at = time.time()
    run_started_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"

    ap = argparse.ArgumentParser(description="Train Logistic Regression Wildfire Forecast (CWFIS Hotspot Labels)")
    add_config_argument(ap)
    ap.add_argument("--data_start", type=str, default="2018-05-01",
                    help="Start of training data (default: 2018-05-01, hotspot archive start)")
    ap.add_argument("--pred_start", type=str, default="2025-05-01")
    ap.add_argument("--pred_end",   type=str, default="2025-10-31")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fwi_dir = get_path(cfg, 'fwi_dir')
    paths_cfg = cfg.get('paths', {})
    if 'observation_dir' in paths_cfg:
        observation_root = get_path(cfg, 'observation_dir')
    else:
        observation_root = get_path(cfg, 'ecmwf_dir')
    hotspot_csv = get_path(cfg, 'hotspot_csv')
    output_dir = os.path.join(get_path(cfg, 'output_dir'), 'logreg_cwfis_fire_prob_7day_forecast')
    os.makedirs(output_dir, exist_ok=True)

    run_stamp = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_meta_path = os.path.join(output_dir, f"run_logistic_cwfis_{run_stamp}.json")
    run_meta = {
        "run_started_at_utc": run_started_iso,
        "label_source": "CWFIS satellite hotspots (VIIRS-M)",
        "cli_args": {
            "data_start": args.data_start,
            "pred_start": args.pred_start,
            "pred_end":   args.pred_end,
            "config":     args.config,
        },
        "resolved_paths": {
            "fwi_dir":          fwi_dir,
            "observation_root": observation_root,
            "hotspot_csv":      hotspot_csv,
            "output_dir":       output_dir,
        },
        "training_config": {
            "forecast_horizon":  cfg.get('training', {}).get('forecast_horizon', 7),
            "n_samples_per_day": cfg.get('training', {}).get('n_samples_per_day', 14285),
        },
        "status": "running",
    }

    print("\n" + "=" * 70)
    print("RUN METADATA  [CWFIS Hotspot Labels]")
    print("=" * 70)
    print(f"Run started (UTC): {run_started_iso}")
    print(f"Config file:        {args.config if args.config else 'configs/default.yaml'}")
    print(f"data_start:         {args.data_start}")
    print(f"pred_start:         {args.pred_start}")
    print(f"pred_end:           {args.pred_end}")
    print(f"FWI dir:            {fwi_dir}")
    print(f"Observation root:   {observation_root}")
    print(f"Hotspot CSV:        {hotspot_csv}")
    print(f"Run log path:       {run_meta_path}")
    print("=" * 70)

    def _flush_run_meta():
        if run_meta.get("status") == "running":
            run_meta["status"] = "failed_or_interrupted"
            run_meta["run_finished_at_utc"] = dt.utcnow().isoformat(timespec="seconds") + "Z"
            run_meta["duration_seconds"] = round(time.time() - run_started_at, 3)
        try:
            with open(run_meta_path, "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
        except Exception:
            pass

    atexit.register(_flush_run_meta)

    # Parse dates
    def _parse_date(s):
        parts = s.split('-')
        return date(int(parts[0]), int(parts[1]), int(parts[2]))

    data_start_date = _parse_date(args.data_start)
    pred_start_date = _parse_date(args.pred_start)
    pred_end_date   = _parse_date(args.pred_end)

    # Training config from YAML
    seq_len          = cfg.get('training', {}).get('forecast_horizon', 7)
    forecast_horizon = cfg.get('training', {}).get('forecast_horizon', 7)
    n_samples_per_day = cfg.get('training', {}).get('n_samples_per_day', 14285)

    print("=" * 70)
    print("LOGISTIC REGRESSION 7-DAY WILDFIRE FORECAST  [CWFIS Hotspot Labels]")
    print("=" * 70)
    print(f"  Data range:       {data_start_date} to {pred_end_date}")
    print(f"  Prediction range: {pred_start_date} to {pred_end_date}")
    print(f"  History window:   {seq_len} days")
    print(f"  Forecast horizon: {forecast_horizon} days")
    print("=" * 70)

    # Step 1: Build date -> filepath mappings
    print("\n[STEP 1] Building file index...")

    fwi_dict = {}
    for path in sorted(glob.glob(os.path.join(fwi_dir, "*.tif"))):
        date_obj = extract_date_from_filename(os.path.basename(path))
        if date_obj:
            fwi_dict[date_obj] = path

    d2m_dict = {}
    d2m_dir = os.path.join(observation_root, "2d")
    d2m_pattern = os.path.join(d2m_dir, "2d_*.tif")
    if not glob.glob(d2m_pattern):
        d2m_pattern = os.path.join(observation_root, "2d_*.tif")
    for path in sorted(glob.glob(d2m_pattern)):
        date_obj = extract_date_from_filename(os.path.basename(path))
        if date_obj:
            d2m_dict[date_obj] = path

    t2m_dict = {}
    t2m_dir = os.path.join(observation_root, "2t")
    t2m_pattern = os.path.join(t2m_dir, "2t_*.tif")
    if not glob.glob(t2m_pattern):
        t2m_pattern = os.path.join(observation_root, "2t_*.tif")
    for path in sorted(glob.glob(t2m_pattern)):
        date_obj = extract_date_from_filename(os.path.basename(path))
        if date_obj:
            t2m_dict[date_obj] = path

    if not fwi_dict:
        raise RuntimeError(f"No FWI files found in {fwi_dir}")
    if not d2m_dict:
        raise RuntimeError(f"No 2d files found under {observation_root}")
    if not t2m_dict:
        raise RuntimeError(f"No 2t files found under {observation_root}")

    print(f"  FWI: {len(fwi_dict)} days, 2d: {len(d2m_dict)} days, 2t: {len(t2m_dict)} days")
    run_meta["inventory"] = {
        "fwi_days": len(fwi_dict),
        "d2m_days": len(d2m_dict),
        "t2m_days": len(t2m_dict),
    }

    # Step 2: Generate aligned date sequence
    print("\n[STEP 2] Aligning files by date...")
    required_end = pred_end_date + timedelta(days=forecast_horizon)
    date_sequence = []
    current_date = data_start_date
    while current_date <= required_end:
        date_sequence.append(current_date)
        current_date += timedelta(days=1)

    fwi_paths, d2m_paths, t2m_paths, aligned_dates = [], [], [], []
    for date_obj in date_sequence:
        if date_obj in fwi_dict and date_obj in d2m_dict and date_obj in t2m_dict:
            fwi_paths.append(fwi_dict[date_obj])
            d2m_paths.append(d2m_dict[date_obj])
            t2m_paths.append(t2m_dict[date_obj])
            aligned_dates.append(date_obj)

    print(f"  Complete dates: {len(aligned_dates)}")
    run_meta["aligned_days"] = len(aligned_dates)
    if len(fwi_paths) < seq_len:
        raise RuntimeError(f"Insufficient data! Need at least {seq_len} days.")

    # Step 3: Load raster data
    print("\n[STEP 3] Loading raster data...")
    fwi_stack = read_singleband_stack(fwi_paths)
    t2m_stack = read_singleband_stack(t2m_paths)
    d2m_stack = read_singleband_stack(d2m_paths)
    print(f"  FWI shape: {fwi_stack.shape}")

    with rasterio.open(fwi_paths[0]) as src:
        profile = src.profile
        H, W = profile['height'], profile['width']

    # Step 4: Load and rasterize hotspots
    print("\n[STEP 4] Loading CWFIS hotspot records...")
    hotspot_df = load_hotspot_data(hotspot_csv)
    print(f"  Total hotspot records : {len(hotspot_df):,}")
    print(f"  Date range            : {hotspot_df['date'].min()} to {hotspot_df['date'].max()}")
    run_meta["hotspot_records"] = int(len(hotspot_df))
    fire_stack = rasterize_hotspots_batch(hotspot_df, aligned_dates, profile)
    print(f"  Fire stack shape      : {fire_stack.shape}")
    print(f"  Mean fire pixel rate  : {fire_stack.mean():.4%}")

    # Step 5: Build training dataset
    print("\n[STEP 5] Building training dataset...")
    train_end_idx = None
    for i, d in enumerate(aligned_dates):
        if d >= pred_start_date:
            train_end_idx = i
            break

    if train_end_idx is None or train_end_idx < seq_len:
        raise RuntimeError(f"Not enough training data before {pred_start_date}")

    X_train_list, y_train_list = [], []
    for idx in range(seq_len - 1, train_end_idx):
        ws = idx - seq_len + 1
        we = idx + 1
        features = compute_features(
            fwi_stack[ws:we], t2m_stack[ws:we], d2m_stack[ws:we], fire_stack[ws:we]
        )
        label = fire_stack[idx]
        X_sample, y_sample = sample_training_data(features, label, n_samples_per_day)
        if X_sample is not None:
            X_train_list.append(X_sample)
            y_train_list.append(y_sample)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    print(f"  Training samples: {len(X_train):,}, Positive rate: {y_train.mean():.4f}")
    run_meta["train_dataset"] = {
        "num_samples":   int(len(X_train)),
        "positive_rate": float(y_train.mean()),
    }

    # Step 6: Train
    print("\n[STEP 6] Training logistic regression...")
    model = build_logistic_model()
    model.fit(X_train, y_train)
    print(f"  Coefficients: {model.coef_[0]}")
    print(f"  Intercept:    {model.intercept_[0]}")
    run_meta["model"] = {
        "coefficients": [float(v) for v in model.coef_[0]],
        "intercept":    float(model.intercept_[0]),
    }

    # Step 7: Generate predictions
    print("\n[STEP 7] Generating forecasts...")
    pred_dates = []
    current = pred_start_date
    while current <= pred_end_date:
        pred_dates.append(current)
        current += timedelta(days=1)

    chunk_size = 250000

    for base_date in pred_dates:
        try:
            base_idx = aligned_dates.index(base_date)
        except ValueError:
            continue

        window_start = base_idx - seq_len
        if window_start < 0:
            continue

        features = compute_features(
            fwi_stack[window_start:base_idx],
            t2m_stack[window_start:base_idx],
            d2m_stack[window_start:base_idx],
            fire_stack[window_start:base_idx]
        )

        base_date_str = base_date.strftime("%Y%m%d")
        out_dir = os.path.join(output_dir, base_date_str)
        os.makedirs(out_dir, exist_ok=True)

        for lead in range(1, forecast_horizon + 1):
            target_date = base_date + timedelta(days=lead)
            target_date_str = target_date.strftime("%Y%m%d")
            output_path = os.path.join(out_dir, f"fire_prob_lead{lead:02d}d_{target_date_str}.tif")

            X_pred = features.reshape(-1, 3)
            prob_map = np.zeros(H * W, dtype=np.float32)
            for i in range(0, len(X_pred), chunk_size):
                chunk = X_pred[i:i + chunk_size]
                prob_map[i:i + chunk_size] = model.predict_proba(chunk)[:, 1]
            prob_map = prob_map.reshape(H, W)

            out_profile = profile.copy()
            out_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
            with rasterio.open(output_path, 'w', **out_profile) as dst:
                dst.write(prob_map, 1)

        print(f"  [DONE] {base_date} -> {forecast_horizon} forecasts")

    print("\n" + "=" * 70)
    print("FORECAST GENERATION COMPLETE!  [CWFIS Hotspot Labels]")
    print(f"Output: {output_dir}")
    print("=" * 70)

    run_ended_at = time.time()
    finished_iso = dt.utcnow().isoformat(timespec="seconds") + "Z"
    run_meta["run_finished_at_utc"] = finished_iso
    run_meta["duration_seconds"] = round(run_ended_at - run_started_at, 3)
    run_meta["status"] = "success"
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run log saved: {run_meta_path}")
    print(f"Run finished (UTC): {finished_iso}")
    print(f"Total duration:     {run_meta['duration_seconds']}s")


if __name__ == "__main__":
    main()
