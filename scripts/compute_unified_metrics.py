#!/usr/bin/env python3
"""
ONE script to compute EVERY metric we use for paper Table 1 + figures.

Methods covered (each computed on the SAME val windows for fair comparison):
  - model          (from a save_window_scores dir)
  - climatology   (from --climatology_tif)
  - persistence   (past 7-day fire density, from --fire_label_npy)
  - logreg        (optional, fits 6-feature logreg at start; --include_logreg)

Metrics per (method × window × label_mode) where
   label_mode ∈ {total, novel_7d, novel_30d, novel_90d}:

  Pixel-level:
    Lift@K for K ∈ {1000, 2500, 5000, 10000, 25000}
    Precision@K, Recall@K, CSI@K, ETS@K
    F1/F2/MCC at top-5000

  Spatial / event-level:
    Lift@30km (15-pixel block coarsen)
    Cluster Lift (8-connectivity, sqrt-median tile sizing) — both
                 unweighted and n_clusters-weighted means

  Probabilistic / calibration (model+logreg only; baselines for ref):
    ROC-AUC, PR-AUC, Brier (min-max normalized)

  Score separability (model+logreg only):
    mean_pos / mean_neg / ratio
    Cohen's d, KS distance
    q10 / q90 percentiles

Outputs (per method):
  outputs/unified_<method>_<run>.csv          per-window (~700 rows)
  outputs/unified_<method>_<run>_summary.csv  mean + bootstrap 95% CI

Usage:
  python -m scripts.compute_unified_metrics \\
    --scores_dir outputs/window_scores_full/v3_9ch_enc28_4y_2018/ \\
    --fire_label_npy data/fire_labels/...npy \\
    --climatology_tif data/fire_clim_annual_nbac/...tif \\
    --output_prefix outputs/unified_enc28_4y \\
    [--include_logreg --config configs/paths_narval.yaml]
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import cluster lift from train_v3 (single source of truth)
from src.training.train_v3 import _compute_cluster_lift_k  # noqa: E402


# ── basic helpers ─────────────────────────────────────────────────────────

def patchify(frame_2d, P):
    H, W = frame_2d.shape
    Hc, Wc = H - H % P, W - W % P
    nph, npw = Hc // P, Wc // P
    out = frame_2d[:Hc, :Wc].reshape(nph, P, npw, P)
    return out.transpose(0, 2, 1, 3).reshape(nph * npw, P * P)


def patches_to_image(per_patch_flat, nph, npw, P):
    arr = per_patch_flat.reshape(nph, npw, P, P)
    arr = arr.transpose(0, 2, 1, 3)
    return arr.reshape(nph * P, npw * P)


def normalize_minmax(x):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x.astype(np.float32) - lo) / (hi - lo)


# ── core metrics ──────────────────────────────────────────────────────────

def lift_at_k(score_flat, label_flat, k):
    n_fire = int(label_flat.sum())
    if n_fire == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    n = label_flat.size
    base_rate = n_fire / n
    order = np.argsort(score_flat)[::-1]
    topk = order[:k]
    tp = int(label_flat[topk].sum())
    fp = k - tp
    fn = n_fire - tp
    prec = tp / k
    rec = tp / n_fire
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    # ETS = (tp - random_hits) / (tp + fp + fn - random_hits)
    rand_hits = (k * n_fire) / n
    ets_num = tp - rand_hits
    ets_den = (tp + fp + fn) - rand_hits
    ets = ets_num / ets_den if ets_den > 0 else 0.0
    lift = prec / base_rate
    return lift, prec, rec, csi, ets, n_fire


def lift_30km(score_2d, label_2d, k_coarse, factor=15):
    H, W = score_2d.shape
    Hc, Wc = H - H % factor, W - W % factor
    nh, nw = Hc // factor, Wc // factor
    s = score_2d[:Hc, :Wc].reshape(nh, factor, nw, factor).max(axis=(1, 3))
    l = label_2d[:Hc, :Wc].reshape(nh, factor, nw, factor).max(axis=(1, 3))
    n_fire = int(l.sum())
    if n_fire == 0:
        return float('nan')
    base = n_fire / l.size
    order = np.argsort(s.ravel())[::-1]
    tp = int(l.ravel()[order[:k_coarse]].sum())
    return (tp / k_coarse) / base


def f_beta_at_k(score_flat, label_flat, k, beta):
    n_fire = int(label_flat.sum())
    if n_fire == 0:
        return float('nan')
    order = np.argsort(score_flat)[::-1]
    tp = int(label_flat[order[:k]].sum())
    fp = k - tp
    fn = n_fire - tp
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def mcc_at_k(score_flat, label_flat, k):
    n_fire = int(label_flat.sum())
    n = label_flat.size
    if n_fire == 0 or k == n:
        return float('nan')
    order = np.argsort(score_flat)[::-1]
    pred = np.zeros(n, dtype=np.uint8)
    pred[order[:k]] = 1
    tp = int(((pred == 1) & (label_flat == 1)).sum())
    tn = int(((pred == 0) & (label_flat == 0)).sum())
    fp = int(((pred == 1) & (label_flat == 0)).sum())
    fn = int(((pred == 0) & (label_flat == 1)).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return 0.0
    return (tp * tn - fp * fn) / den


def roc_auc(score, label):
    from sklearn.metrics import roc_auc_score
    if label.sum() == 0 or label.sum() == len(label):
        return float('nan')
    try:
        return float(roc_auc_score(label, score))
    except Exception:
        return float('nan')


def pr_auc(score, label):
    from sklearn.metrics import average_precision_score
    if label.sum() == 0:
        return float('nan')
    try:
        return float(average_precision_score(label, score))
    except Exception:
        return float('nan')


def brier(score_normalized, label):
    return float(np.mean((score_normalized.astype(np.float32) - label.astype(np.float32)) ** 2))


def cohen_d(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float('nan')
    sp, sn = pos.std(ddof=1), neg.std(ddof=1)
    pooled = np.sqrt(((len(pos) - 1) * sp ** 2 + (len(neg) - 1) * sn ** 2)
                     / (len(pos) + len(neg) - 2))
    if pooled < 1e-12:
        return float('nan')
    return float((pos.mean() - neg.mean()) / pooled)


def ks_dist(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a, b = np.sort(pos), np.sort(neg)
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def separability(score_flat, label_flat, prefix):
    pos = score_flat[label_flat == 1].astype(np.float32)
    neg = score_flat[label_flat == 0].astype(np.float32)
    if len(pos) == 0:
        return {f"{prefix}_n_pos": 0}
    return {
        f"{prefix}_n_pos": len(pos),
        f"{prefix}_mean_pos": float(pos.mean()),
        f"{prefix}_mean_neg": float(neg.mean()),
        f"{prefix}_ratio": float(pos.mean() / max(neg.mean(), 1e-9)),
        f"{prefix}_q10_pos": float(np.quantile(pos, 0.1)),
        f"{prefix}_q90_pos": float(np.quantile(pos, 0.9)),
        f"{prefix}_q10_neg": float(np.quantile(neg, 0.1)),
        f"{prefix}_q90_neg": float(np.quantile(neg, 0.9)),
        f"{prefix}_cohen_d": cohen_d(pos, neg),
        f"{prefix}_ks": ks_dist(pos, neg),
    }


# ── label builders ────────────────────────────────────────────────────────

def build_novel(win_date, label_total_p, fire_full, label_start, lookback_days, P):
    past_start = win_date - timedelta(days=lookback_days + 7)
    past_start_idx = max(0, (past_start - label_start).days)
    past_end_idx = min(fire_full.shape[0], (win_date - label_start).days + 1)
    burn_recent = fire_full[past_start_idx:past_end_idx].max(axis=0).astype(np.uint8)
    burn_recent_p = patchify(burn_recent, P)
    if burn_recent_p.shape != label_total_p.shape:
        return None
    return ((label_total_p > 0) & (burn_recent_p == 0)).astype(np.uint8)


# ── all-metric block for one (method, window, label_mode) ─────────────────

def all_metrics(score_p, label_p, score_2d, label_2d, k_values, prefix,
                include_separability=True, include_calibration=True):
    """Compute the full metric panel for ONE (method, window, label_mode)."""
    rec = {}
    score_flat = score_p.reshape(-1)
    label_flat = label_p.reshape(-1)

    # Pixel Lift@K family
    for k in k_values:
        lift, prec, rec_k, csi, ets, n_fire = lift_at_k(score_flat, label_flat, k)
        rec[f"{prefix}_lift_{k}"] = lift
        rec[f"{prefix}_prec_{k}"] = prec
        rec[f"{prefix}_recall_{k}"] = rec_k
        rec[f"{prefix}_csi_{k}"] = csi
        rec[f"{prefix}_ets_{k}"] = ets
    rec[f"{prefix}_n_fire"] = int(label_flat.sum())

    # F1/F2/MCC at top 5000 (if 5000 in k_values)
    if 5000 in k_values:
        rec[f"{prefix}_f1_5000"] = f_beta_at_k(score_flat, label_flat, 5000, 1.0)
        rec[f"{prefix}_f2_5000"] = f_beta_at_k(score_flat, label_flat, 5000, 2.0)
        rec[f"{prefix}_mcc_5000"] = mcc_at_k(score_flat, label_flat, 5000)

    # Lift@30km
    k30 = max(1, 5000 // (15 * 15))
    rec[f"{prefix}_lift_30km"] = lift_30km(score_2d, label_2d, k30, factor=15)

    # Cluster lift (handled by train_v3 function, pass float 2D)
    cm = _compute_cluster_lift_k(score_2d.astype(np.float32),
                                  label_2d.astype(np.uint8), k=5000)
    rec[f"{prefix}_cluster_lift"] = cm["lift_k"]
    rec[f"{prefix}_cluster_recall"] = cm["recall_k"]
    rec[f"{prefix}_cluster_n"] = cm["n_clusters"]
    rec[f"{prefix}_cluster_pr_auc"] = cm.get("cl_pr_auc", float('nan'))

    # Calibration / probabilistic
    if include_calibration:
        rec[f"{prefix}_roc_auc"] = roc_auc(score_flat, label_flat)
        rec[f"{prefix}_pr_auc"] = pr_auc(score_flat, label_flat)
        sc_n = normalize_minmax(score_flat)
        rec[f"{prefix}_brier"] = brier(sc_n, label_flat)

    # Separability
    if include_separability:
        rec.update(separability(score_flat, label_flat, prefix))

    return rec


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--climatology_tif", required=True)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--grid", nargs=2, type=int, default=[142, 169])
    ap.add_argument("--k_values", nargs="+", type=int,
                    default=[1000, 2500, 5000, 10000, 25000])
    ap.add_argument("--lookback_days_list", nargs="+", type=int,
                    default=[7, 30, 90])
    ap.add_argument("--output_prefix", required=True,
                    help="e.g. outputs/unified_enc28_4y -> writes "
                         "<prefix>_<method>.csv and <prefix>_<method>_summary.csv")
    ap.add_argument("--include_logreg", action="store_true",
                    help="Also fit + score logreg (~3-4 hours extra: "
                         "loads 2t/sm20 daily channels)")
    ap.add_argument("--config", default="configs/paths_narval.yaml",
                    help="Required if --include_logreg")
    ap.add_argument("--n_train_wins", type=int, default=80,
                    help="Train-window cap for logreg fit")
    args = ap.parse_args()

    P = args.patch_size
    nph, npw = args.grid

    print("=" * 70)
    print(f"UNIFIED METRICS — scores_dir={args.scores_dir}")
    print("=" * 70)

    # ── Load shared inputs ────────────────────────────────────────────
    fire_full = np.load(args.fire_label_npy, mmap_mode="r")
    sidecar = str(args.fire_label_npy).rsplit(".", 1)[0] + ".json"
    label_start = date.fromisoformat("2000-05-01")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            label_start = date.fromisoformat(json.load(f)["date_range"][0])
    print(f"  fire_full {fire_full.shape}, label_start {label_start}")

    with rasterio.open(args.climatology_tif) as src:
        clim_arr = np.nan_to_num(src.read(1), nan=0.0).astype(np.float32)
    H, W = clim_arr.shape
    Hc, Wc = H - H % P, W - W % P
    clim_p = patchify(clim_arr[:Hc, :Wc], P)
    clim_2d = clim_arr[:Hc, :Wc]
    print(f"  climatology {clim_p.shape}")

    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"  {len(files)} window files")
    if not files:
        sys.exit("no window files in scores_dir")

    # ── Logreg fit (optional) ─────────────────────────────────────────
    logreg = None
    scaler = None
    t2_p = None
    sm_p = None
    slope_per_patch = None
    clim_per_patch = None
    fwi_p = None  # for logreg features

    if args.include_logreg:
        print("\n  [logreg] fitting (will take a while)...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import yaml
        from src.evaluation.benchmark_baselines import (
            load_data, _build_file_index,
            _build_s2s_windows_calendar,
        )
        from scripts.benchmark_ml import (
            _load_static_patched, _load_daily_channel,
            _features_for_window, _label_for_window,
        )

        with open(args.config) as f:
            cfg = yaml.safe_load(f)["paths"]
        # Load FWI / fire / climatology for shared use
        (fwi_p, _fire_p_loader, _clim_p_loader, all_dates_lr, date_to_idx_lr,
         val_wins_lr, _val_dates_lr, _n_patches_lr, _grid_lr) = load_data(
            args.config, "2022-05-01", "2025-10-31", 7, 14, 45,
            args.patch_size, 14, True, args.climatology_tif, args.fire_label_npy,
        )
        # Static
        clim_per_patch = clim_p.mean(axis=1).astype(np.float32)
        terrain_dir = cfg.get("terrain_dir", "data/terrain")
        slope_per_patch = _load_static_patched(
            os.path.join(terrain_dir, "slope.tif"), Hc, Wc, P).mean(axis=1)
        # Daily
        obs_root = cfg.get("observation_dir")
        t2_idx = _build_file_index(os.path.join(obs_root, "2t"))
        sm_idx = _build_file_index(os.path.join(obs_root, "sm20"))
        t2_p = _load_daily_channel(t2_idx, all_dates_lr, Hc, Wc, P, "2t")
        sm_p = _load_daily_channel(sm_idx, all_dates_lr, Hc, Wc, P, "sm20")
        # Train wins
        all_w = _build_s2s_windows_calendar(
            all_dates_lr, date_to_idx_lr, 7, 14, 45)
        train_wins = [w for w in all_w if all_dates_lr[w[1]] < date(2022, 5, 1)]
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(train_wins),
                           min(args.n_train_wins, len(train_wins)), replace=False)
        train_wins_s = [train_wins[i] for i in sorted(idxs)]
        # Build train (X, y)
        X_list, y_list = [], []
        for w in train_wins_s:
            X = _features_for_window(w, fwi_p, t2_p, sm_p,
                                      clim_per_patch, slope_per_patch)
            y = _label_for_window(w, _fire_p_loader)
            X_list.append(X); y_list.append(y)
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        scaler = StandardScaler().fit(X_train)
        logreg = LogisticRegression(class_weight="balanced", max_iter=200,
                                     solver="lbfgs", n_jobs=-1)
        t0 = time.time()
        logreg.fit(scaler.transform(X_train), y_train)
        print(f"  [logreg] fit in {time.time()-t0:.0f}s")

    # ── Iterate windows, compute all metrics for all methods ──────────
    methods = {"model", "clim", "persist"}
    if logreg is not None:
        methods.add("logreg")

    # CSV output dict per method
    rows_per_method = {m: [] for m in methods}

    for fi, fpath in enumerate(files):
        z = np.load(fpath)
        prob = z["prob_agg"].astype(np.float32)        # (n_patches, P²)
        label_total = z["label_agg"].astype(np.uint8)
        win_date_str = str(z["win_date"])
        if not win_date_str:
            continue
        win_date = date.fromisoformat(win_date_str)
        hs, he = int(z["hs"]), int(z["he"])

        # Persistence score: past 7d fire density
        past_start_idx = max(0, (win_date - timedelta(days=7) - label_start).days)
        past_end_idx = min(fire_full.shape[0],
                           (win_date - label_start).days + 1)
        persist_arr = fire_full[past_start_idx:past_end_idx].mean(axis=0)
        persist_p = patchify(persist_arr.astype(np.float32), P)

        # Logreg score (if enabled)
        if logreg is not None:
            try:
                X_win = _features_for_window(
                    (hs, he, int(z["ts"]), int(z["te"])),
                    fwi_p, t2_p, sm_p, clim_per_patch, slope_per_patch)
                proba = logreg.predict_proba(scaler.transform(X_win))[:, 1]
                logreg_p = np.broadcast_to(proba.astype(np.float32)[:, None],
                                            (proba.shape[0], P * P)).copy()
            except Exception as e:
                print(f"  [logreg] window {fi} skipped: {e}")
                logreg_p = None
        else:
            logreg_p = None

        # Reconstruct 2D for spatial / cluster / 30km metrics
        prob_2d = patches_to_image(prob, nph, npw, P)
        label_2d = patches_to_image(label_total, nph, npw, P)
        clim_2d_full = clim_2d
        persist_2d = patches_to_image(persist_p, nph, npw, P)
        logreg_2d = (patches_to_image(logreg_p, nph, npw, P)
                     if logreg_p is not None else None)

        # Build novel labels
        novel_labels = {}
        for lb in args.lookback_days_list:
            n = build_novel(win_date, label_total, fire_full,
                             label_start, lb, P)
            if n is not None:
                novel_labels[f"novel_{lb}d"] = n

        # Per-method record
        for method, score_p, score_2d in [
            ("model",   prob,      prob_2d),
            ("clim",    clim_p,    clim_2d_full),
            ("persist", persist_p, persist_2d),
            ("logreg",  logreg_p,  logreg_2d),
        ]:
            if method == "logreg" and logreg_p is None:
                continue
            if method not in methods:
                continue
            rec = {"win_date": win_date_str,
                   "n_total_label": int(label_total.sum())}
            # Total label
            rec.update(all_metrics(score_p, label_total, score_2d, label_2d,
                                    args.k_values, prefix="total"))
            # Novel labels — reconstruct 2D each time
            for lb_key, novel in novel_labels.items():
                novel_2d = patches_to_image(novel, nph, npw, P)
                rec.update(all_metrics(score_p, novel, score_2d, novel_2d,
                                        args.k_values, prefix=lb_key,
                                        include_separability=True,
                                        include_calibration=False))
            rows_per_method[method].append(rec)

        if (fi + 1) % 25 == 0 or fi == 0:
            print(f"  done {fi+1}/{len(files)} windows  "
                  f"({len(methods)} methods × ~{len(args.k_values)*5+30} cols each)")

    # ── Write CSVs + summaries ────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    rng = np.random.default_rng(0)
    for method, rows in rows_per_method.items():
        if not rows:
            continue
        per_path = f"{args.output_prefix}_{method}.csv"
        sum_path = f"{args.output_prefix}_{method}_summary.csv"
        fieldnames = sorted({k for r in rows for k in r.keys()})
        fieldnames = (["win_date", "n_total_label"]
                      + [k for k in fieldnames if k not in ("win_date", "n_total_label")])
        with open(per_path, "w") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
        # Summary with bootstrap CI
        summary_rows = []
        for col in fieldnames:
            if col in ("win_date", "n_total_label"):
                continue
            vals = np.array([r.get(col, float('nan')) for r in rows])
            vals = vals[~np.isnan(vals.astype(float))] if vals.dtype == float else vals
            if len(vals) == 0:
                continue
            try:
                vals_f = vals.astype(np.float64)
                boot = np.empty(2000)
                for i in range(2000):
                    boot[i] = vals_f[rng.integers(0, len(vals_f), len(vals_f))].mean()
                lo, hi = np.percentile(boot, [2.5, 97.5])
                summary_rows.append({"metric": col,
                                      "mean": float(vals_f.mean()),
                                      "ci_lo": float(lo), "ci_hi": float(hi),
                                      "n": int(len(vals_f))})
            except Exception:
                continue
        with open(sum_path, "w") as f:
            wr = csv.DictWriter(f, fieldnames=["metric", "mean", "ci_lo", "ci_hi", "n"])
            wr.writeheader()
            for r in summary_rows:
                wr.writerow(r)
        print(f"  → {per_path}  ({len(rows)} rows)")
        print(f"  → {sum_path}  ({len(summary_rows)} metrics)")

    # ── Headline print ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HEADLINE  Lift@5000")
    print("=" * 70)
    for method in sorted(methods):
        sum_path = f"{args.output_prefix}_{method}_summary.csv"
        if not os.path.exists(sum_path):
            continue
        with open(sum_path) as f:
            data = {r["metric"]: r for r in csv.DictReader(f)}
        print(f"\n  {method}:")
        for col in ["total_lift_5000", "novel_30d_lift_5000",
                    "total_cluster_lift", "novel_30d_cluster_lift",
                    "total_lift_30km", "total_roc_auc", "total_brier",
                    "total_cohen_d"]:
            if col in data:
                r = data[col]
                print(f"    {col:30s} = {float(r['mean']):7.3f}  "
                      f"[{float(r['ci_lo']):6.3f}, {float(r['ci_hi']):6.3f}]")


if __name__ == "__main__":
    main()
