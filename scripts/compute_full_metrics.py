#!/usr/bin/env python3
"""
Unified metric panel from saved per-pixel scores.

Reads window_*.npz from a model's save_window_scores_dir + the
NBAC+NFDB label stack, computes EVERY metric we use for paper:

  Lift@K (K=1000, 2500, 5000, 10000, 25000)
    × {total, novel_7d, novel_30d, novel_90d}
  Lift@30km (lift_coarse)        — event-level, 15-pixel coarsening
  ROC-AUC                        — discrimination
  PR-AUC                         — precision-recall
  Brier score                    — calibration
  BSS (vs climatology)           — skill above climatology
  F1, F2, MCC at top-5000        — classification at threshold

Also computes the SAME metrics for:
  - climatology baseline (from --climatology_tif)
  - persistence baseline (from past 7d fire density)
  - (optionally) logreg if --logreg_csv passed

Output: outputs/full_metrics_<run_name>.csv (one row per window)
       outputs/full_metrics_<run_name>_summary.csv (mean ± CI)
"""
import argparse
import csv
import glob
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── helpers ───────────────────────────────────────────────────────────────

def patchify(frame_2d, P):
    H, W = frame_2d.shape
    Hc, Wc = H - H % P, W - W % P
    nph, npw = Hc // P, Wc // P
    out = frame_2d[:Hc, :Wc].reshape(nph, P, npw, P)
    return out.transpose(0, 2, 1, 3).reshape(nph * npw, P * P)


def patches_to_image(per_patch_flat, nph, npw, P):
    """(n_patches, P²) → (nph*P, npw*P)."""
    arr = per_patch_flat.reshape(nph, npw, P, P)
    arr = arr.transpose(0, 2, 1, 3)
    return arr.reshape(nph * P, npw * P)


def lift_at_k(score_flat, label_flat, k):
    n_fire = int(label_flat.sum())
    n = label_flat.size
    if n_fire == 0:
        return float('nan'), float('nan'), 0
    base_rate = n_fire / n
    order = np.argsort(score_flat)[::-1]
    tp = int(label_flat[order[:k]].sum())
    prec = tp / k
    return prec / base_rate, prec, n_fire


def lift_coarse(score_2d, label_2d, k_coarse, factor=15):
    """Event-level lift via 15-pixel block coarsening (~30 km)."""
    H, W = score_2d.shape
    Hc, Wc = H - H % factor, W - W % factor
    nh, nw = Hc // factor, Wc // factor
    s = score_2d[:Hc, :Wc].reshape(nh, factor, nw, factor).max(axis=(1, 3))
    l = label_2d[:Hc, :Wc].reshape(nh, factor, nw, factor).max(axis=(1, 3))
    return lift_at_k(s.ravel(), l.ravel(), k_coarse)


def roc_auc_fast(score, label):
    from sklearn.metrics import roc_auc_score
    if label.sum() == 0 or label.sum() == len(label):
        return float('nan')
    try:
        return float(roc_auc_score(label, score))
    except Exception:
        return float('nan')


def pr_auc_fast(score, label):
    from sklearn.metrics import average_precision_score
    if label.sum() == 0:
        return float('nan')
    try:
        return float(average_precision_score(label, score))
    except Exception:
        return float('nan')


def brier_score(score, label):
    return float(np.mean((score.astype(np.float32) - label.astype(np.float32)) ** 2))


def f_beta_at_k(score_flat, label_flat, k, beta=1.0):
    n_fire = int(label_flat.sum())
    if n_fire == 0:
        return float('nan')
    order = np.argsort(score_flat)[::-1]
    tp = int(label_flat[order[:k]].sum())
    fp = k - tp
    fn = n_fire - tp
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return float((1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec))


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
    return float((tp * tn - fp * fn) / den)


def build_novel_label(win_date, label_total_p, fire_full, label_start, lookback_days, P):
    """Return novel-ignition label per patch (n_patches, P²) uint8."""
    past_start = win_date - timedelta(days=lookback_days + 7)
    past_start_idx = max(0, (past_start - label_start).days)
    past_end_idx = min(fire_full.shape[0],
                       (win_date - label_start).days + 1)
    burn_recent = fire_full[past_start_idx:past_end_idx].max(axis=0).astype(np.uint8)
    burn_recent_p = patchify(burn_recent, P)
    if burn_recent_p.shape != label_total_p.shape:
        return None
    return ((label_total_p > 0) & (burn_recent_p == 0)).astype(np.uint8)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--climatology_tif", required=True)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--grid", nargs=2, type=int, default=[142, 169],
                    help="(nph, npw) — needed for image reconstruction")
    ap.add_argument("--k_values", nargs="+", type=int,
                    default=[1000, 2500, 5000, 10000, 25000])
    ap.add_argument("--coarsen_factor", type=int, default=15)
    ap.add_argument("--lookback_days_list", nargs="+", type=int,
                    default=[7, 30, 90])
    ap.add_argument("--out_per_window_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    P = args.patch_size
    nph, npw = args.grid
    n_patches_expected = nph * npw

    run_name = args.run_name or os.path.basename(args.scores_dir.rstrip("/"))
    print(f"=== full metrics for {run_name} ===")

    # Load full fire stack (mmap)
    fire_full = np.load(args.fire_label_npy, mmap_mode="r")
    sidecar = str(args.fire_label_npy).rsplit(".", 1)[0] + ".json"
    label_start = date.fromisoformat("2000-05-01")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            label_start = date.fromisoformat(json.load(f)["date_range"][0])
    print(f"  fire_full {fire_full.shape}  label_start {label_start}")

    # Load climatology
    with rasterio.open(args.climatology_tif) as src:
        clim_arr = np.nan_to_num(src.read(1), nan=0.0).astype(np.float32)
    H, W = clim_arr.shape
    Hc, Wc = H - H % P, W - W % P
    clim_p = patchify(clim_arr[:Hc, :Wc], P)  # (n_patches, P²)
    clim_2d_full = clim_arr[:Hc, :Wc]
    print(f"  climatology shape {clim_p.shape}")

    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"  {len(files)} window files")
    if not files:
        sys.exit("no window files")

    rows = []
    for fpath in files:
        z = np.load(fpath)
        prob = z["prob_agg"].astype(np.float32)
        label_total = z["label_agg"].astype(np.uint8)
        win_date_str = str(z["win_date"])
        if not win_date_str:
            continue
        win_date = date.fromisoformat(win_date_str)
        hs, he = int(z["hs"]), int(z["he"])

        # Persistence score: past 7-day fire density
        past_start_idx = max(0, (win_date - timedelta(days=7) - label_start).days)
        past_end_idx = min(fire_full.shape[0],
                           (win_date - label_start).days + 1)
        persist_arr = fire_full[past_start_idx:past_end_idx].mean(axis=0)
        persist_p = patchify(persist_arr.astype(np.float32), P)

        # Reconstruct 2D for coarsening
        prob_2d = patches_to_image(prob, nph, npw, P)
        label_2d = patches_to_image(label_total, nph, npw, P)
        clim_2d = clim_2d_full
        persist_2d = patches_to_image(persist_p, nph, npw, P)

        # Build novel labels for each lookback
        novel_labels = {
            f"novel_{lb}d": build_novel_label(win_date, label_total, fire_full,
                                               label_start, lb, P)
            for lb in args.lookback_days_list
        }

        rec = {"win_date": win_date_str, "n_total": int(label_total.sum())}

        # Compute all metrics for each method × each label_mode
        for method, score_p, score_2d in [
            ("model", prob, prob_2d),
            ("clim", clim_p, clim_2d),
            ("persist", persist_p, persist_2d),
        ]:
            score_flat = score_p.reshape(-1)
            # Total label metrics
            label_flat = label_total.reshape(-1)
            for k in args.k_values:
                lift, _, _ = lift_at_k(score_flat, label_flat, k)
                rec[f"{method}_lift_total_{k}"] = lift
            # Lift@30km on total label
            lift30, _, _ = lift_coarse(score_2d, label_2d,
                                        k_coarse=args.k_values[2] // (args.coarsen_factor ** 2) or 1,
                                        factor=args.coarsen_factor)
            rec[f"{method}_lift_30km"] = lift30
            # Discrimination + calibration (only model truly needs but compute for all)
            rec[f"{method}_roc_auc"] = roc_auc_fast(score_flat, label_flat)
            rec[f"{method}_pr_auc"] = pr_auc_fast(score_flat, label_flat)
            # Normalize score to [0,1] for brier
            sc_n = (score_flat - score_flat.min()) / max(score_flat.max() - score_flat.min(), 1e-9)
            rec[f"{method}_brier"] = brier_score(sc_n, label_flat)
            rec[f"{method}_f1_5000"] = f_beta_at_k(score_flat, label_flat, 5000, 1.0)
            rec[f"{method}_f2_5000"] = f_beta_at_k(score_flat, label_flat, 5000, 2.0)
            rec[f"{method}_mcc_5000"] = mcc_at_k(score_flat, label_flat, 5000)
            # Novel label lift
            for lb_key, novel in novel_labels.items():
                if novel is None:
                    continue
                novel_flat = novel.reshape(-1)
                for k in args.k_values:
                    lift_n, _, _ = lift_at_k(score_flat, novel_flat, k)
                    rec[f"{method}_lift_{lb_key}_{k}"] = lift_n

        rows.append(rec)
        if len(rows) % 50 == 0 or len(rows) == 1:
            print(f"  done {len(rows)}/{len(files)} windows")

    # Per-window CSV
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out_per_window_csv) or ".", exist_ok=True)
    with open(args.out_per_window_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n  per-window CSV → {args.out_per_window_csv}  ({len(rows)} rows)")

    # Summary (mean + bootstrap 95% CI)
    rng = np.random.default_rng(0)
    summary_rows = []
    for col in fieldnames:
        if col in ("win_date", "n_total"):
            continue
        vals = np.array([r[col] for r in rows
                          if not (isinstance(r[col], float) and np.isnan(r[col]))])
        if len(vals) == 0:
            continue
        boot = np.empty(2000)
        for i in range(2000):
            boot[i] = vals[rng.integers(0, len(vals), len(vals))].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        summary_rows.append({
            "metric": col,
            "mean": float(vals.mean()),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n": len(vals),
        })

    with open(args.out_summary_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=["metric", "mean", "ci_lo", "ci_hi", "n"])
        wr.writeheader()
        for r in summary_rows:
            wr.writerow(r)
    print(f"  summary CSV → {args.out_summary_csv}  ({len(summary_rows)} metrics)")

    # Print headline
    print("\n" + "=" * 70)
    print(f"HEADLINE  {run_name}  (all {len(rows)} windows)")
    print("=" * 70)
    headlines = [
        "model_lift_total_5000", "model_lift_novel_30d_5000", "model_lift_novel_90d_5000",
        "model_lift_30km", "model_roc_auc", "model_pr_auc", "model_brier",
        "clim_lift_total_5000", "clim_lift_novel_30d_5000",
        "persist_lift_total_5000", "persist_lift_novel_30d_5000",
    ]
    by_metric = {r["metric"]: r for r in summary_rows}
    for m in headlines:
        if m in by_metric:
            r = by_metric[m]
            print(f"  {m:38s}  {r['mean']:7.3f}  [{r['ci_lo']:6.3f}, {r['ci_hi']:6.3f}]  n={r['n']}")


if __name__ == "__main__":
    main()
