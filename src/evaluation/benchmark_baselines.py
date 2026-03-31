#!/usr/bin/env python
"""
Compute Lift@K for baseline methods on the same validation windows
used in S2S Transformer v2 training.

Baselines:
  fwi_oracle    — rank pixels by observed FWI (perfect weather knowledge)
  climatology   — rank pixels by static fire_climatology.tif
  fwi_max       — rank by max FWI over lead window (instead of mean)

Usage:
  python -m src.evaluation.benchmark_baselines \
      --config configs/paths_narval.yaml \
      --baseline fwi_oracle climatology \
      --k_values 1000 2500 5000 10000 \
      --pred_start 2022-05-01 --pred_end 2025-10-31

Outputs a summary table to stdout and optionally saves CSV.
"""

import argparse
import glob
import os
import time

import numpy as np
import rasterio
import yaml

# ------------------------------------------------------------------ #
# Constants — must match train_s2s_hotspot_cwfis_v2.py
# ------------------------------------------------------------------ #
CHANNEL_NAMES = ["FWI", "2t", "2d", "FFMC", "DMC", "DC", "BUI", "fire_clim"]
FWI_CHANNEL_IDX = 0  # FWI is channel 0 in the meteo stack


def _parse_date(s):
    """Parse YYYY-MM-DD string to date object."""
    from datetime import date
    parts = s.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _extract_date_from_filename(fname):
    """Extract YYYYMMDD -> date from filenames like fwi_20180501.tif."""
    import re
    from datetime import date
    m = re.search(r"(\d{8})", fname)
    if m:
        ds = m.group(1)
        return date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
    return None


def _build_file_index(directory, ext=".tif"):
    """Map date -> filepath for all TIFs in a directory."""
    index = {}
    for f in glob.glob(os.path.join(directory, f"*{ext}")):
        d = _extract_date_from_filename(os.path.basename(f))
        if d:
            index[d] = f
    return index


def _read_tif(path):
    """Read single-band TIF as float32."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


def _build_s2s_windows(n_days, in_days, lead_start, lead_end):
    """Build (enc_start, enc_end, target_start, target_end) index tuples."""
    windows = []
    for i in range(in_days, n_days - lead_end):
        windows.append((i - in_days, i, i + lead_start, i + lead_end + 1))
    return windows


def _patchify_frame(frame, patch_size):
    """
    (H, W, C) -> (n_patches, C * P * P)
    Same as training script.
    """
    H, W, C = frame.shape
    P = patch_size
    nph = H // P
    npw = W // P
    # (nph, P, npw, P, C) -> (nph, npw, P, P, C) -> (n_patches, P*P*C)
    patches = frame[:nph * P, :npw * P, :].reshape(nph, P, npw, P, C)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(nph * npw, P * P * C)
    return patches


def load_data(config_path, pred_start_str, pred_end_str, in_days, lead_start,
              lead_end, patch_size, dilate_radius):
    """
    Load and align FWI data + fire ground truth.
    Returns: fwi_patched, fire_patched, aligned_dates, val_wins, grid_info
    """
    from datetime import date
    from scipy.ndimage import binary_dilation

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["paths"]

    pred_start = _parse_date(pred_start_str)
    pred_end = _parse_date(pred_end_str)

    print("[1] Building file index...")
    fwi_index = _build_file_index(cfg["fwi_dir"])
    print(f"  FWI: {len(fwi_index)} days")

    # Also load other channels for alignment
    obs_dir = cfg.get("observation_dir") or cfg.get("ecmwf_dir")
    obs_index = {}
    if obs_dir and os.path.isdir(obs_dir):
        for f in glob.glob(os.path.join(obs_dir, "*.tif")):
            d = _extract_date_from_filename(os.path.basename(f))
            if d:
                obs_index[d] = f

    # Align dates: need FWI available
    all_dates = sorted(fwi_index.keys())
    if pred_end:
        all_dates = [d for d in all_dates if d <= pred_end]
    print(f"  Aligned dates: {len(all_dates)}  "
          f"({all_dates[0]} -> {all_dates[-1]})")

    # Read grid shape from first FWI file
    sample_path = fwi_index[all_dates[0]]
    with rasterio.open(sample_path) as src:
        H, W = src.height, src.width
    P = patch_size
    Hc, Wc = H - H % P, W - W % P
    nph, npw = Hc // P, Wc // P
    n_patches = nph * npw
    out_dim = P * P
    print(f"  Grid: H={H} W={W}  crop=({Hc},{Wc})  "
          f"patches={nph}x{npw}={n_patches}")

    T = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Load FWI data -> patched
    print(f"\n[2] Loading FWI data ({T} days)...")
    t0 = time.time()
    fwi_patched = np.zeros((n_patches, T, out_dim), dtype=np.float16)
    for t_idx, d in enumerate(all_dates):
        arr = _read_tif(fwi_index[d])
        frame = arr[:Hc, :Wc, np.newaxis]  # (Hc, Wc, 1)
        fwi_patched[:, t_idx, :] = _patchify_frame(frame, P).astype(np.float16)
        if t_idx % 500 == 0 or t_idx == T - 1:
            print(f"  day {t_idx:4d}/{T}  ({time.time()-t0:.0f}s)")

    # Load fire climatology
    clim_path = cfg.get("fire_climatology_tif")
    clim_patched = None
    if clim_path and os.path.exists(clim_path):
        print(f"\n[3] Loading fire climatology...")
        clim_arr = _read_tif(clim_path)
        clim_frame = clim_arr[:Hc, :Wc, np.newaxis]
        clim_patched = _patchify_frame(clim_frame, P).astype(np.float16)
        # clim_patched: (n_patches, P*P)
        print(f"  shape={clim_patched.shape}  "
              f"nonzero={np.count_nonzero(clim_patched)}")

    # Load fire ground truth — reuse the same rasterization as training
    print(f"\n[4] Loading hotspot data...")
    hotspot_csv = cfg["hotspot_csv"]
    from src.data_ops.processing.rasterize_hotspots import (
        load_hotspot_data, rasterize_hotspots_batch,
    )
    hotspot_df = load_hotspot_data(hotspot_csv)
    print(f"  Records: {len(hotspot_df):,}")

    # Get rasterio profile from reference TIF
    with rasterio.open(sample_path) as src:
        profile = src.profile

    fire_stack = rasterize_hotspots_batch(hotspot_df, all_dates, profile)
    n_rasterized = int(fire_stack.sum())
    print(f"  Rasterized: {n_rasterized:,} hotspot pixels")

    # Dilate fire labels
    if dilate_radius > 0:
        print(f"  Dilating fire labels: radius={dilate_radius} px...")
        from scipy.ndimage import generate_binary_structure
        # Create circular kernel
        y_grid, x_grid = np.ogrid[-dilate_radius:dilate_radius + 1,
                                  -dilate_radius:dilate_radius + 1]
        kernel = (x_grid ** 2 + y_grid ** 2) <= dilate_radius ** 2
        for t_idx in range(T):
            if fire_stack[t_idx].any():
                fire_stack[t_idx] = binary_dilation(
                    fire_stack[t_idx], structure=kernel).astype(np.uint8)
            if t_idx % 500 == 0:
                print(f"    frame {t_idx}/{T}")

    # Patchify fire
    print(f"  Patchifying fire labels...")
    # fire_patched: (T, n_patches, P*P) to match training layout
    fire_patched = np.zeros((T, n_patches, out_dim), dtype=np.uint8)
    for t_idx in range(T):
        frame_f = fire_stack[t_idx, :Hc, :Wc, np.newaxis].astype(np.float32)
        fire_patched[t_idx] = _patchify_frame(frame_f, P).astype(np.uint8)
    del fire_stack
    print(f"  fire_patched: {fire_patched.shape}")

    # Build val windows
    all_windows = _build_s2s_windows(T, in_days, lead_start, lead_end)
    val_wins = [w for w in all_windows
                if all_dates[w[1]] >= pred_start]
    print(f"\n[5] Val windows: {len(val_wins)}  "
          f"(from {len(all_windows)} total)")

    return fwi_patched, fire_patched, clim_patched, all_dates, val_wins, n_patches


def compute_baseline_lift_k(score_fn, fire_patched, val_wins, n_patches,
                            k_values, n_sample_wins, name):
    """
    Compute Lift@K for a baseline scoring function.

    score_fn(win):  given (hs, he, ts, te) returns (n_patches, P²) float scores
                    higher score = higher predicted fire risk
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)  # same seed as training

    if len(val_wins) > n_sample_wins:
        idx = rng.choice(len(val_wins), size=n_sample_wins, replace=False)
        sample_wins = [val_wins[i] for i in sorted(idx)]
    else:
        sample_wins = val_wins

    all_scores = []
    all_labels = []

    t0 = time.time()
    for wi, (hs, he, ts, te) in enumerate(sample_wins):
        scores = score_fn((hs, he, ts, te))  # (n_patches, P²)
        labels = fire_patched[ts:te, :, :]   # (dec_days, n_patches, P²)

        # Aggregate: scores already aggregated by score_fn; labels = max over leads
        label_agg = labels.max(axis=0)  # (n_patches, P²) uint8

        all_scores.append(scores.reshape(-1))
        all_labels.append(label_agg.reshape(-1).astype(np.float32))

        if (wi + 1) % 5 == 0 or wi == len(sample_wins) - 1:
            print(f"  [{name}] window {wi+1}/{len(sample_wins)}  "
                  f"({time.time()-t0:.0f}s)")

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    n_total = len(all_scores)
    n_fire = int(all_labels.sum())
    baseline = n_fire / n_total if n_total > 0 else 0.0

    print(f"  [{name}] n_total={n_total:,}  n_fire={n_fire:,}  "
          f"baseline={baseline:.6f}")

    if n_fire == 0:
        return {k: {"lift_k": 0.0, "precision_k": 0.0, "recall_k": 0.0,
                     "n_fire": 0, "baseline": 0.0}
                for k in k_values}

    # Compute PR-AUC
    try:
        pr_auc = float(average_precision_score(all_labels, all_scores))
    except Exception:
        pr_auc = 0.0

    results = {}
    for k in k_values:
        k_eff = min(k, n_total)
        top_idx = np.argpartition(all_scores, -k_eff)[-k_eff:]
        tp = float(all_labels[top_idx].sum())
        fp = k_eff - tp
        fn = n_fire - tp

        precision_k = tp / k_eff
        recall_k = tp / n_fire
        lift_k = precision_k / baseline if baseline > 0 else 0.0
        csi_k = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        tp_random = k_eff * baseline
        denom_ets = tp + fp + fn - tp_random
        ets_k = (tp - tp_random) / denom_ets if denom_ets > 0 else 0.0

        results[k] = {
            "lift_k": lift_k,
            "precision_k": precision_k,
            "recall_k": recall_k,
            "csi_k": csi_k,
            "ets_k": ets_k,
            "pr_auc": pr_auc,
            "n_fire": n_fire,
            "baseline": baseline,
            "tp": int(tp),
            "k_eff": k_eff,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute Lift@K for baseline fire prediction methods")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--baseline", nargs="+",
                        choices=["fwi_oracle", "fwi_max", "climatology", "s2s_fwi"],
                        default=["fwi_oracle", "climatology"],
                        help="Baselines to evaluate")
    parser.add_argument("--s2s_fwi_dir", default=None,
                        help="Directory with pre-computed S2S-FWI TIFs (for s2s_fwi baseline)")
    parser.add_argument("--pred_start", default="2022-05-01")
    parser.add_argument("--pred_end", default="2025-10-31")
    parser.add_argument("--in_days", type=int, default=7)
    parser.add_argument("--lead_start", type=int, default=14)
    parser.add_argument("--lead_end", type=int, default=45)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dilate_radius", type=int, default=14,
                        help="Fire label dilation radius in pixels (0=no dilation)")
    parser.add_argument("--k_values", nargs="+", type=int,
                        default=[1000, 2500, 5000, 10000, 25000],
                        help="K values for Lift@K")
    parser.add_argument("--n_sample_wins", type=int, default=20,
                        help="Number of val windows to sample (seed=0)")
    parser.add_argument("--output_csv", default=None,
                        help="Save results to CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("BASELINE BENCHMARK: Lift@K Evaluation")
    print("=" * 70)
    print(f"  Baselines   : {args.baseline}")
    print(f"  Val period  : {args.pred_start} -> {args.pred_end}")
    print(f"  Lead range  : {args.lead_start}-{args.lead_end} days")
    print(f"  K values    : {args.k_values}")
    print(f"  Sample wins : {args.n_sample_wins}")
    print(f"  Dilate      : {args.dilate_radius} px")
    print("=" * 70)

    fwi_patched, fire_patched, clim_patched, aligned_dates, val_wins, n_patches = \
        load_data(args.config, args.pred_start, args.pred_end,
                  args.in_days, args.lead_start, args.lead_end,
                  args.patch_size, args.dilate_radius)

    all_results = {}

    # --- FWI Oracle: mean observed FWI over lead window ---
    if "fwi_oracle" in args.baseline:
        print(f"\n{'='*40}")
        print("Baseline: FWI Oracle (mean observed FWI)")
        print(f"{'='*40}")

        def fwi_oracle_score(win):
            hs, he, ts, te = win
            # fwi_patched: (n_patches, T, P²)
            # Mean FWI over target window
            fwi_window = fwi_patched[:, ts:te, :]  # (n_patches, dec_days, P²)
            return fwi_window.mean(axis=1).astype(np.float32)  # (n_patches, P²)

        all_results["fwi_oracle"] = compute_baseline_lift_k(
            fwi_oracle_score, fire_patched, val_wins, n_patches,
            args.k_values, args.n_sample_wins, "fwi_oracle")

    # --- FWI Max: max observed FWI over lead window ---
    if "fwi_max" in args.baseline:
        print(f"\n{'='*40}")
        print("Baseline: FWI Max (max observed FWI)")
        print(f"{'='*40}")

        def fwi_max_score(win):
            hs, he, ts, te = win
            fwi_window = fwi_patched[:, ts:te, :]
            return fwi_window.max(axis=1).astype(np.float32)

        all_results["fwi_max"] = compute_baseline_lift_k(
            fwi_max_score, fire_patched, val_wins, n_patches,
            args.k_values, args.n_sample_wins, "fwi_max")

    # --- Fire Climatology: static historical fire frequency ---
    if "climatology" in args.baseline:
        print(f"\n{'='*40}")
        print("Baseline: Fire Climatology (static)")
        print(f"{'='*40}")

        if clim_patched is None:
            print("  SKIP: fire_climatology.tif not found")
        else:
            def clim_score(win):
                # Same score for every window — static map
                return clim_patched.astype(np.float32)  # (n_patches, P²)

            all_results["climatology"] = compute_baseline_lift_k(
                clim_score, fire_patched, val_wins, n_patches,
                args.k_values, args.n_sample_wins, "climatology")

    # --- S2S-FWI: FWI computed from S2S forecast weather ---
    if "s2s_fwi" in args.baseline:
        print(f"\n{'='*40}")
        print("Baseline: S2S-FWI (forecast-derived FWI)")
        print(f"{'='*40}")

        s2s_fwi_dir = args.s2s_fwi_dir
        if s2s_fwi_dir is None:
            with open(args.config) as f:
                _cfg = yaml.safe_load(f)["paths"]
            s2s_fwi_dir = os.path.join(_cfg["project_root"], "data", "s2s_fwi")

        if not os.path.isdir(s2s_fwi_dir):
            print(f"  SKIP: {s2s_fwi_dir} not found.")
            print(f"  Run 'python -m src.data_ops.processing.compute_s2s_fwi' first.")
        else:
            # Build date index for S2S-FWI mean TIFs
            s2s_fwi_dates = {}
            for entry in sorted(os.listdir(s2s_fwi_dir)):
                mean_path = os.path.join(s2s_fwi_dir, entry, "s2s_fwi_mean.tif")
                if os.path.exists(mean_path):
                    try:
                        d = _parse_date(entry)
                        s2s_fwi_dates[d] = mean_path
                    except Exception:
                        continue
            print(f"  Found {len(s2s_fwi_dates)} S2S-FWI issue dates")

            if len(s2s_fwi_dates) == 0:
                print("  SKIP: no S2S-FWI TIFs found")
            else:
                P = args.patch_size
                # Pre-load S2S-FWI into patched format indexed by aligned_dates
                date_to_tidx = {d: i for i, d in enumerate(aligned_dates)}
                s2s_fwi_patched = {}  # tidx -> (n_patches, P²)
                _n_loaded = 0
                for d, path in s2s_fwi_dates.items():
                    if d not in date_to_tidx:
                        continue
                    arr = _read_tif(path)
                    Hc = arr.shape[0] - arr.shape[0] % P
                    Wc = arr.shape[1] - arr.shape[1] % P
                    frame = arr[:Hc, :Wc, np.newaxis]
                    patched = _patchify_frame(frame, P).astype(np.float16)
                    s2s_fwi_patched[date_to_tidx[d]] = patched
                    _n_loaded += 1
                print(f"  Loaded {_n_loaded} S2S-FWI maps into patch format")

                def s2s_fwi_score(win):
                    hs, he, ts, te = win
                    # Use the S2S-FWI for the base date (he = enc_end = base date idx)
                    base_tidx = he
                    if base_tidx in s2s_fwi_patched:
                        return s2s_fwi_patched[base_tidx].astype(np.float32)
                    # Fallback: try nearby dates
                    for offset in range(1, 4):
                        if (base_tidx - offset) in s2s_fwi_patched:
                            return s2s_fwi_patched[base_tidx - offset].astype(np.float32)
                    # No S2S-FWI available — return zeros
                    return np.zeros_like(next(iter(s2s_fwi_patched.values())),
                                        dtype=np.float32)

                all_results["s2s_fwi"] = compute_baseline_lift_k(
                    s2s_fwi_score, fire_patched, val_wins, n_patches,
                    args.k_values, args.n_sample_wins, "s2s_fwi")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Baseline':<20}"
    for k in args.k_values:
        header += f"  {'Lift@'+str(k):>12}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, k_results in all_results.items():
        row = f"{name:<20}"
        for k in args.k_values:
            if k in k_results:
                row += f"  {k_results[k]['lift_k']:12.2f}x"
            else:
                row += f"  {'N/A':>12}"
        print(row)
        rows.append((name, k_results))

    # Detailed K=5000 breakdown
    display_k = 5000 if 5000 in args.k_values else args.k_values[0]
    print(f"\n--- Detailed metrics at K={display_k} ---")
    print(f"{'Baseline':<20} {'Lift@K':>10} {'Prec@K':>10} {'Rec@K':>10} "
          f"{'CSI@K':>10} {'ETS@K':>10} {'PR-AUC':>10}")
    for name, k_results in all_results.items():
        r = k_results.get(display_k, {})
        print(f"{name:<20} {r.get('lift_k',0):10.2f}x "
              f"{r.get('precision_k',0):10.4f} {r.get('recall_k',0):10.4f} "
              f"{r.get('csi_k',0):10.4f} {r.get('ets_k',0):10.4f} "
              f"{r.get('pr_auc',0):10.4f}")

    # Save CSV
    if args.output_csv:
        import csv
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["baseline", "k", "lift_k", "precision_k",
                           "recall_k", "csi_k", "ets_k", "pr_auc",
                           "tp", "n_fire", "baseline_rate"])
            for name, k_results in all_results.items():
                for k, r in sorted(k_results.items()):
                    writer.writerow([
                        name, k, f"{r['lift_k']:.4f}",
                        f"{r['precision_k']:.6f}", f"{r['recall_k']:.6f}",
                        f"{r['csi_k']:.6f}", f"{r['ets_k']:.6f}",
                        f"{r['pr_auc']:.6f}",
                        r.get('tp', ''), r['n_fire'],
                        f"{r['baseline']:.8f}",
                    ])
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
