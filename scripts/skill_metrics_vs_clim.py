"""
Compute "dynamic skill" metrics that isolate the value of a forecast
above static climatology. Three metrics, all per-window, all four
methods (model + climatology + persistence + ecmwf_s2s) on the SAME
val windows + SAME labels.

  1. AUC and ROC Skill Score (RSS = 2·(AUC − 0.5))
       — calibration-free; how well the score ranks fire vs no-fire.
  2. Anomaly Spearman ρ
       ρ = spearmanr(score − clim, label)
       — does the score's *deviation from climatology* track real
         deviations from climatological fire density?
  3. MSE / Brier (on min-max-normalised score in [0,1])
       Brier Skill Score = 1 − Brier_method / Brier_clim
       — calibration-aware; positive = model adds skill.

Inputs come from:
  - outputs/window_scores_full/<run>/window_*.npz   (model: prob_agg, label_agg)
  - data/fire_clim_annual_nbac/fire_clim_upto_<year>.tif
  - data/fire_labels/...r14.npy   (for persistence)
  - data/ecmwf_s2s_fire_epsg3978/fwinx/<...>          (for ecmwf_s2s)

Usage:
  python -m scripts.skill_metrics_vs_clim \\
      --scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
      --label_npy data/fire_labels/.../r14.npy \\
      --fire_clim_dir data/fire_clim_annual_nbac \\
      --ecmwf_dir data/ecmwf_s2s_fire_epsg3978/fwinx \\
      --output outputs/skill_vs_clim_summary.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from scripts.recall_at_budget import (
    parse_date, list_score_files, patches_to_2d,
)
from scripts.recall_at_budget_baselines import (
    img_to_patches, _list_ecmwf_issues, _load_ecmwf_window,
)


def _flatten_valid(score, label, valid_mask):
    """Flatten + mask. Returns (s, y) 1D arrays."""
    m = valid_mask.ravel()
    return score.ravel()[m], label.ravel()[m]


def auc_score(s, y):
    """ROC-AUC. Sample if too large."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    if len(s) > 200_000:
        rng = np.random.default_rng(0)
        # Stratified subsample: keep all positives, sample equal negatives
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_keep_neg = min(len(neg_idx), max(len(pos_idx), 50_000))
        keep_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
        keep = np.concatenate([pos_idx, keep_neg])
        s, y = s[keep], y[keep]
    # Mann-Whitney U
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]
    # rank tie-correction
    ranks = np.empty_like(s_sorted, dtype=np.float64)
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[i:j + 1] = avg_rank
        i = j + 1
    n_pos = int(y_sorted.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos_ranks = ranks[y_sorted == 1].sum()
    U = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


def spearman(x, y):
    """Spearman ρ via rank-Pearson. Sample if huge."""
    if len(x) > 200_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), size=200_000, replace=False)
        x, y = x[idx], y[idx]
    rx = _rankdata(x); ry = _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _rankdata(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0; n = len(a)
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def minmax01(x, valid_mask):
    """Min-max normalise to [0, 1] over valid pixels."""
    valid = x[valid_mask]
    if len(valid) == 0:
        return x
    lo, hi = float(valid.min()), float(valid.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    out = np.where(valid_mask, (x - lo) / (hi - lo), 0.0)
    return out


def brier(p, y):
    """MSE on (predicted_prob in [0,1], binary label)."""
    return float(((p - y) ** 2).mean())


def bootstrap_ci(values, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray([v for v in values if np.isfinite(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(arr[idx].mean())
    return float(arr.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--label_npy", default=None,
                    help="(only needed for persistence)")
    ap.add_argument("--label_data_start", default="2000-05-01")
    ap.add_argument("--fire_clim_dir", default="data/fire_clim_annual_nbac")
    ap.add_argument("--ecmwf_dir", default="data/ecmwf_s2s_fire_epsg3978/fwinx")
    ap.add_argument("--pred_start", default="2022-05-01")
    ap.add_argument("--pred_end", default="2025-10-31")
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=46)
    ap.add_argument("--persistence_lookback", type=int, default=7)
    ap.add_argument("--ecmwf_max_lag_days", type=int, default=35)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--n_rows", type=int, default=142)
    ap.add_argument("--n_cols", type=int, default=169)
    ap.add_argument("--methods", nargs="+",
                    default=["model", "climatology", "persistence", "ecmwf_s2s"])
    ap.add_argument("--limit_windows", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_start = parse_date(args.pred_start)
    pred_end = parse_date(args.pred_end)
    P, NR, NC = args.patch_size, args.n_rows, args.n_cols
    H, W = NR * P, NC * P

    # Load supporting data
    label_stack = None
    if "persistence" in args.methods:
        if not args.label_npy:
            print("[ERROR] persistence needs --label_npy"); sys.exit(1)
        label_stack = np.load(args.label_npy, mmap_mode="r")
        label_start = parse_date(args.label_data_start)
    clim_cache = {}

    def load_clim_for_year(y):
        if y in clim_cache:
            return clim_cache[y]
        import rasterio
        f = Path(args.fire_clim_dir) / f"fire_clim_upto_{y - 1}.tif"
        if not f.exists():
            f = Path(args.fire_clim_dir) / "fire_clim_upto_2022.tif"
        if not f.exists():
            return None
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
        clim_cache[y] = arr
        return arr

    ecmwf_issues = _list_ecmwf_issues(args.ecmwf_dir) if "ecmwf_s2s" in args.methods else None

    # Iterate windows
    score_files = list_score_files(args.scores_dir, pred_start, pred_end, True)
    if args.limit_windows > 0:
        score_files = score_files[:args.limit_windows]
    print(f"Windows: {len(score_files)}")
    print(f"Methods: {args.methods}")

    rows = []
    t0 = datetime.now()
    for i, (t, npz_path) in enumerate(score_files):
        npz = np.load(npz_path)
        if "prob_agg" not in npz.files or "label_agg" not in npz.files:
            continue
        label_2d = (patches_to_2d(npz["label_agg"], NR, NC, P) > 0).astype(np.uint8)
        if label_2d.sum() == 0:
            continue
        # 1. Get climatology score (always needed as reference)
        clim_full = load_clim_for_year(t.year)
        if clim_full is None:
            continue
        clim_2d = clim_full[:H, :W].astype(np.float32)
        # All methods produce score on the (H, W) grid
        method_scores_2d = {}
        if "model" in args.methods:
            method_scores_2d["model"] = patches_to_2d(
                npz["prob_agg"].astype(np.float32), NR, NC, P)
        if "climatology" in args.methods:
            method_scores_2d["climatology"] = clim_2d.copy()
        if "persistence" in args.methods:
            t_lo = (t - timedelta(days=args.persistence_lookback) - label_start).days
            t_hi = (t - timedelta(days=1) - label_start).days
            if t_lo < 0 or t_hi >= label_stack.shape[0]:
                continue
            past = np.array(label_stack[t_lo:t_hi + 1])
            method_scores_2d["persistence"] = past.max(axis=0).astype(np.float32)[:H, :W]
        if "ecmwf_s2s" in args.methods:
            cands = [d for d in ecmwf_issues
                     if d <= t and (t - d).days <= args.ecmwf_max_lag_days]
            if not cands:
                continue
            issue = max(cands)
            delta = (t - issue).days
            lead_lo, lead_hi = delta + args.lead_start, delta + args.lead_end
            if lead_hi > 215:
                continue
            fc = _load_ecmwf_window(args.ecmwf_dir, issue,
                                    list(range(lead_lo, lead_hi + 1)))
            if fc is None:
                continue
            method_scores_2d["ecmwf_s2s"] = np.nanmax(fc, axis=0).astype(np.float32)[:H, :W]

        # Compute metrics (climatology is always reference)
        valid = np.ones((H, W), dtype=bool)
        # Trim invalid pixels (where clim is nan/zero or any score nan)
        valid &= np.isfinite(clim_2d)
        for m, s in method_scores_2d.items():
            valid &= np.isfinite(s)
        if valid.sum() == 0:
            continue

        clim_norm = minmax01(clim_2d, valid)
        s_clim, y_clim = _flatten_valid(clim_norm, label_2d, valid)
        brier_clim = brier(s_clim, y_clim)

        row = {"win_date": t.isoformat(), "n_fire": int(label_2d.sum()),
               "n_valid": int(valid.sum()), "by_method": {}}
        for m, s in method_scores_2d.items():
            s_norm = minmax01(s, valid)
            s_flat, y_flat = _flatten_valid(s_norm, label_2d, valid)
            auc = auc_score(s_flat, y_flat)
            br = brier(s_flat, y_flat)
            bss = 1 - br / brier_clim if brier_clim > 0 else float("nan")
            # Anomaly correlation: spearman(score - clim, label)
            anom = s.ravel()[valid.ravel()] - clim_2d.ravel()[valid.ravel()]
            rho = spearman(anom, y_flat)
            row["by_method"][m] = {
                "auc": auc,
                "rss": 2.0 * (auc - 0.5) if np.isfinite(auc) else float("nan"),
                "brier": br,
                "bss": bss,
                "anomaly_spearman": rho,
            }
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(score_files)}] valid={len(rows)} "
                  f"({(datetime.now()-t0).total_seconds():.0f}s)")

    print(f"\nValid windows: {len(rows)}")
    if not rows:
        sys.exit(1)

    # Aggregate
    summary = {"n_windows": len(rows), "by_method": {}}
    print(f"\n{'Method':<14} | {'AUC':>14} | {'RSS':>14} | {'BSS':>14} | {'Anomaly ρ':>14}")
    print("-" * 84)
    for m in args.methods:
        if not all(m in r["by_method"] for r in rows):
            continue
        auc_v = [r["by_method"][m]["auc"] for r in rows]
        rss_v = [r["by_method"][m]["rss"] for r in rows]
        bss_v = [r["by_method"][m]["bss"] for r in rows]
        rho_v = [r["by_method"][m]["anomaly_spearman"] for r in rows]
        auc_m, auc_lo, auc_hi = bootstrap_ci(auc_v)
        rss_m, rss_lo, rss_hi = bootstrap_ci(rss_v)
        bss_m, bss_lo, bss_hi = bootstrap_ci(bss_v)
        rho_m, rho_lo, rho_hi = bootstrap_ci(rho_v)
        print(f"{m:<14} | {auc_m:.3f} [{auc_lo:.3f},{auc_hi:.3f}] | "
              f"{rss_m:+.3f} [{rss_lo:+.3f},{rss_hi:+.3f}] | "
              f"{bss_m:+.3f} [{bss_lo:+.3f},{bss_hi:+.3f}] | "
              f"{rho_m:+.3f} [{rho_lo:+.3f},{rho_hi:+.3f}]")
        summary["by_method"][m] = {
            "auc": {"mean": auc_m, "lo": auc_lo, "hi": auc_hi},
            "rss": {"mean": rss_m, "lo": rss_lo, "hi": rss_hi},
            "bss": {"mean": bss_m, "lo": bss_lo, "hi": bss_hi},
            "anomaly_spearman": {"mean": rho_m, "lo": rho_lo, "hi": rho_hi},
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_window": rows}, f, indent=1)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
