"""
Single-pass comprehensive metric card for one ckpt's window_scores_full
output OR for a baseline (climatology / persistence / ecmwf_s2s).

Produces ONE JSON with ALL paper metrics, identical schema across runs:

  LIFT FAMILY (decision):
    Lift@K  for K ∈ {1k, 2.5k, 5k, 10k, 25k, 50k}
    Lift@30km (event-pooled at 30km tiles, K=5000)
    Cluster Lift@5000 (sqrt-median tile size)

  RECALL FAMILY:
    Recall@K  for same K values
    Recall@budget  for budget ∈ {0.1%, 0.5%, 1%, 5%, 10%} of land area

  SKILL SCORES (vs climatology reference):
    BSS = 1 - Brier_method / Brier_clim
    RSS = 2 (AUC - 0.5)
    AUPRC skill = (AUPRC - base_rate)/(1 - base_rate)

  DYNAMIC SKILL:
    Anomaly Spearman ρ = spearmanr(score - clim, label)
    Per-anomaly-stratum Recall@5% (3 strata: anomaly / normal / easy by clim recall)

  CALIBRATION:
    Brier (with Reliability + Resolution decomposition)
    ROC-AUC
    AUPRC

  OPERATIONAL:
    Precision@K, CSI@K, ETS@K
    F1, F2, MCC at top-5000

All metrics use bootstrap 95% CI over windows. Output JSON has identical
keys regardless of input source (model / climatology / persistence /
ecmwf_s2s) — directly mergeable into a master comparison table.

Usage:
  # MODEL
  python -m scripts.compute_full_metric_card \\
      --source model \\
      --scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014_climsim \\
      --output outputs/metric_card_climsim.json

  # CLIMATOLOGY BASELINE (uses model's npz files for window list & labels)
  python -m scripts.compute_full_metric_card \\
      --source climatology \\
      --reference_scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
      --fire_clim_dir data/fire_clim_annual_nbac \\
      --output outputs/metric_card_climatology.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np


# ───────────────── helpers ─────────────────

def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def list_score_files(score_dir, pred_start_iso, pred_end_iso,
                     fire_season_only=True):
    fire_months = set(range(4, 11))
    pat = re.compile(r"window_\d+_(\d{4}-\d{2}-\d{2})\.npz$")
    out = {}
    for f in sorted(Path(score_dir).glob("window_*.npz")):
        m = pat.search(f.name)
        if not m:
            continue
        d_iso = m.group(1)
        if d_iso < pred_start_iso or d_iso > pred_end_iso:
            continue
        if fire_season_only and int(d_iso[5:7]) not in fire_months:
            continue
        out[d_iso] = f
    return out


def patches_to_2d(patch_arr, n_rows, n_cols, P):
    return (patch_arr.reshape(n_rows, n_cols, P, P)
            .transpose(0, 2, 1, 3).reshape(n_rows * P, n_cols * P))


def img_to_patches_max(arr_2d, n_rows, n_cols, P):
    """Max-pool 2D image to per-patch values (broadcast back to 2D)."""
    H_full = n_rows * P
    W_full = n_cols * P
    arr = arr_2d[:H_full, :W_full]
    patched = (arr.reshape(n_rows, P, n_cols, P)
               .transpose(0, 2, 1, 3).reshape(n_rows * n_cols, P * P))
    per_patch = patched.max(axis=1, keepdims=True)
    return np.broadcast_to(per_patch, (n_rows * n_cols, P * P)).copy()


def lift_at_k(s_flat, y_flat, k):
    valid = np.isfinite(s_flat) & np.isfinite(y_flat)
    s = s_flat[valid]; y = y_flat[valid]
    if len(s) == 0 or y.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    base = float(y.mean())
    if base <= 0:
        return float("nan"), float("nan"), float("nan")
    k = min(k, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    prec_k = float(y[top].mean())
    recall_k = float(y[top].sum() / y.sum()) if y.sum() > 0 else float("nan")
    lift_k = prec_k / base if base > 0 else float("nan")
    return lift_k, prec_k, recall_k


def lift_30km_pooled(score_2d, label_2d, k, pool=15):
    H, W = score_2d.shape
    Hp, Wp = H // pool, W // pool
    s = (score_2d[:Hp * pool, :Wp * pool]
         .reshape(Hp, pool, Wp, pool).max(axis=(1, 3)))
    y = (label_2d[:Hp * pool, :Wp * pool]
         .reshape(Hp, pool, Wp, pool).max(axis=(1, 3)))
    return lift_at_k(s.flatten(), y.flatten(), k)[0]


def connected_fire_events(label_2d):
    from scipy.ndimage import label as ndi_label
    structure = np.ones((3, 3), dtype=bool)
    lbl, n = ndi_label(label_2d > 0, structure=structure)
    return lbl, n


def recall_at_budget_one(score_2d, label_2d, valid_mask, budget_frac,
                         event_lbl, n_events):
    n_valid = int(valid_mask.sum())
    if n_valid == 0 or n_events == 0:
        return float("nan"), 0
    k = max(1, int(round(budget_frac * n_valid)))
    flat = np.where(valid_mask, score_2d, -np.inf).ravel()
    top = np.argpartition(-flat, k - 1)[:k]
    mask_flat = np.zeros(flat.shape, dtype=bool)
    mask_flat[top] = True
    top_mask = mask_flat.reshape(score_2d.shape)
    hit = np.unique(event_lbl[top_mask & (event_lbl > 0)])
    return float(len(hit) / n_events), int(k)


def auc_score(s, y):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    if len(s) > 200_000:
        rng = np.random.default_rng(0)
        pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
        n_keep_neg = min(len(neg_idx), max(len(pos_idx), 50_000))
        keep_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
        keep = np.concatenate([pos_idx, keep_neg])
        s, y = s[keep], y[keep]
    order = np.argsort(s, kind="stable")
    s_s = s[order]; y_s = y[order]
    ranks = np.empty_like(s_s, dtype=np.float64)
    i = 0; n = len(s_s)
    while i < n:
        j = i
        while j + 1 < n and s_s[j + 1] == s_s[i]: j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[i:j + 1] = avg; i = j + 1
    n_pos = int(y_s.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y_s == 1].sum()
    U = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


def aprc_score(s, y):
    """AUPRC via average precision."""
    if y.sum() == 0:
        return float("nan")
    if len(s) > 200_000:
        rng = np.random.default_rng(0)
        pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
        n_keep_neg = min(len(neg_idx), max(len(pos_idx), 50_000))
        keep_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
        keep = np.concatenate([pos_idx, keep_neg])
        s, y = s[keep], y[keep]
    order = np.argsort(-s, kind="stable")
    y_s = y[order]
    cum_tp = np.cumsum(y_s)
    cum_fp = np.cumsum(1 - y_s)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    rec = cum_tp / max(int(y_s.sum()), 1)
    rec_diff = np.diff(rec, prepend=0.0)
    return float(np.sum(prec * rec_diff))


def spearman_rho(x, y):
    if len(x) > 100_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), size=100_000, replace=False)
        x, y = x[idx], y[idx]

    def _rankdata(a):
        order = np.argsort(a, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        i = 0; n = len(a)
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]: j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1): ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _rankdata(x); ry = _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def brier_decomp(p, y):
    """Returns (Brier, Reliability, Resolution, Uncertainty)."""
    p = np.clip(p, 0, 1).astype(np.float64)
    bin_edges = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.searchsorted(bin_edges, p, side="right") - 1, 0, 9)
    base_rate = float(y.mean())
    rel = 0.0; res = 0.0
    for k in range(10):
        sel = bin_idx == k
        n_k = sel.sum()
        if n_k == 0:
            continue
        p_bar = p[sel].mean()
        o_bar = float(y[sel].mean())
        rel += n_k / len(p) * (p_bar - o_bar) ** 2
        res += n_k / len(p) * (o_bar - base_rate) ** 2
    unc = base_rate * (1 - base_rate)
    brier = float(np.mean((p - y) ** 2))
    return brier, rel, res, unc


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


# ───────────────── score loaders by source ─────────────────

def load_score_2d_model(npz_path, NR, NC, P):
    npz = np.load(npz_path)
    if "prob_agg" not in npz.files or "label_agg" not in npz.files:
        return None, None
    score_2d = patches_to_2d(npz["prob_agg"].astype(np.float32), NR, NC, P)
    label_2d = (patches_to_2d(npz["label_agg"], NR, NC, P) > 0).astype(np.uint8)
    return score_2d, label_2d


def load_clim_2d(year, fire_clim_dir, NR, NC, P, cache):
    if year in cache:
        return cache[year]
    import rasterio
    f = Path(fire_clim_dir) / f"fire_clim_upto_{year - 1}.tif"
    if not f.exists():
        f = Path(fire_clim_dir) / "fire_clim_upto_2022.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr_2d = arr[:NR * P, :NC * P]
    cache[year] = arr_2d
    return arr_2d


# ───────────────── main metric pipeline ─────────────────

def compute_window_metrics(score_2d, label_2d, clim_2d, args):
    """All metrics for one window. Returns dict."""
    valid = np.isfinite(score_2d) & np.isfinite(clim_2d)
    if valid.sum() == 0 or label_2d.sum() == 0:
        return None
    s_flat = score_2d.ravel(); y_flat = label_2d.ravel().astype(np.float32)
    c_flat = clim_2d.ravel()
    sv = s_flat[valid.ravel()]; yv = y_flat[valid.ravel()]
    cv = c_flat[valid.ravel()]

    out = {}

    # ── Lift family ──
    Ks = [1000, 2500, 5000, 10000, 25000, 50000]
    for K in Ks:
        l, p, r = lift_at_k(sv, yv, K)
        out[f"lift@{K}"] = l
        out[f"prec@{K}"] = p
        out[f"recall@{K}"] = r
    out["lift@30km"] = lift_30km_pooled(score_2d, label_2d, 5000)

    # ── Recall@budget ──
    valid_mask = valid
    event_lbl, n_events = connected_fire_events(label_2d)
    out["n_events"] = int(n_events)
    for B in [0.001, 0.005, 0.01, 0.05, 0.10]:
        rec, k_pix = recall_at_budget_one(score_2d, label_2d, valid_mask, B,
                                          event_lbl, n_events)
        out[f"recall@budget_{B}"] = rec
        out[f"k@budget_{B}"] = k_pix

    # ── Skill scores (vs climatology) ──
    # Min-max normalize to [0,1] for Brier comparability
    def _mm01(a):
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)
    sv_n = _mm01(sv); cv_n = _mm01(cv)
    brier_m, rel_m, res_m, unc = brier_decomp(sv_n, yv)
    brier_c, _, _, _ = brier_decomp(cv_n, yv)
    out["brier"] = brier_m
    out["reliability"] = rel_m
    out["resolution"] = res_m
    out["uncertainty"] = unc
    out["brier_clim"] = brier_c
    out["bss"] = 1 - brier_m / brier_c if brier_c > 0 else float("nan")
    out["roc_auc"] = auc_score(sv, yv)
    out["rss"] = 2 * (out["roc_auc"] - 0.5) if np.isfinite(out["roc_auc"]) else float("nan")
    out["auprc"] = aprc_score(sv, yv)
    base_rate = float(yv.mean())
    out["base_rate"] = base_rate
    out["auprc_skill"] = (out["auprc"] - base_rate) / (1 - base_rate) \
        if base_rate < 1 and np.isfinite(out["auprc"]) else float("nan")

    # ── Dynamic skill ──
    # Anomaly Spearman ρ between (score - clim) and label
    anom = sv - cv
    out["anomaly_rho"] = spearman_rho(anom, yv)

    # ── F1/F2/MCC at top-5000 ──
    K = 5000
    valid_idx = np.where(valid.ravel())[0]
    if len(valid_idx) > K and yv.sum() > 0:
        top_k = valid_idx[np.argpartition(-s_flat[valid_idx], K - 1)[:K]]
        pred = np.zeros_like(y_flat, dtype=np.uint8)
        pred[top_k] = 1
        tp = int((pred & y_flat.astype(np.uint8)).sum())
        fp = int(pred.sum() - tp)
        fn = int(y_flat.sum() - tp)
        tn = int(len(y_flat) - tp - fp - fn)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out["f1@5000"] = 2 * prec * rec / max(prec + rec, 1e-12)
        out["f2@5000"] = (5 * prec * rec) / max(4 * prec + rec, 1e-12)
        # CSI = TP / (TP + FP + FN)
        csi = tp / max(tp + fp + fn, 1)
        out["csi@5000"] = csi
        # ETS chance-corrected
        rand_hits = (tp + fp) * (tp + fn) / max(len(y_flat), 1)
        out["ets@5000"] = (tp - rand_hits) / max(tp + fp + fn - rand_hits, 1)
        # MCC — cast to float64 to avoid Python int overflow / np.sqrt failing
        # on Python int. tn for an all-Canada grid is ~6M, the product can hit
        # 1e26 which overflows int64 silently — float math is required.
        denom = float(
            np.sqrt(float(tp + fp) * float(tp + fn)
                    * float(tn + fp) * float(tn + fn))
        )
        out["mcc@5000"] = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")
    else:
        out["f1@5000"] = out["f2@5000"] = out["csi@5000"] = out["ets@5000"] = out["mcc@5000"] = float("nan")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["model", "climatology", "persistence", "ecmwf_s2s"])
    ap.add_argument("--scores_dir", default=None,
                    help="(model) ckpt's window_scores_full dir")
    ap.add_argument("--reference_scores_dir", default=None,
                    help="(baseline) any model's npz dir, used for window list + label_agg")
    ap.add_argument("--fire_clim_dir", default="data/fire_clim_annual_nbac")
    ap.add_argument("--label_npy", default=None,
                    help="(persistence) label memmap")
    ap.add_argument("--label_data_start", default="2000-05-01")
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
    ap.add_argument("--limit_windows", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    P, NR, NC = args.patch_size, args.n_rows, args.n_cols
    H, W = NR * P, NC * P

    # Need a reference dir to get window list + labels, regardless of source
    ref_dir = args.scores_dir if args.source == "model" else args.reference_scores_dir
    if not ref_dir:
        print("[ERROR] need --scores_dir or --reference_scores_dir"); sys.exit(1)
    score_files = list_score_files(ref_dir, args.pred_start, args.pred_end, True)
    if args.limit_windows > 0:
        score_files = dict(list(score_files.items())[:args.limit_windows])
    print(f"Source: {args.source}  Windows: {len(score_files)}")

    label_stack = None
    label_start = None
    if args.source == "persistence":
        if not args.label_npy:
            print("[ERROR] persistence needs --label_npy"); sys.exit(1)
        label_stack = np.load(args.label_npy, mmap_mode="r")
        label_start = parse_date(args.label_data_start)

    ecmwf_issues = None
    if args.source == "ecmwf_s2s":
        ecmwf_issues = []
        for d in sorted(Path(args.ecmwf_dir).glob("issue_*")):
            tag = d.name.replace("issue_", "")
            if len(tag) == 6 and tag.isdigit():
                ecmwf_issues.append(date(int(tag[:4]), int(tag[4:]), 1))

    clim_cache = {}

    per_window = []
    n_skip = 0
    t0 = datetime.now()
    for wi, (date_iso, npz_path) in enumerate(score_files.items()):
        t = parse_date(date_iso)
        # Reference (for labels): always from the ref dir
        ref_npz = np.load(npz_path)
        if "label_agg" not in ref_npz.files:
            n_skip += 1; continue
        label_2d = (patches_to_2d(ref_npz["label_agg"], NR, NC, P) > 0).astype(np.uint8)
        if label_2d.sum() == 0:
            n_skip += 1; continue

        clim_2d = load_clim_2d(t.year, args.fire_clim_dir, NR, NC, P, clim_cache)
        if clim_2d is None:
            n_skip += 1; continue

        # Build score for THIS source
        if args.source == "model":
            score_2d = patches_to_2d(ref_npz["prob_agg"].astype(np.float32), NR, NC, P)
        elif args.source == "climatology":
            score_2d = clim_2d.copy()
        elif args.source == "persistence":
            t_lo = (t - timedelta(days=args.persistence_lookback) - label_start).days
            t_hi = (t - timedelta(days=1) - label_start).days
            if t_lo < 0 or t_hi >= label_stack.shape[0]:
                n_skip += 1; continue
            past = np.array(label_stack[t_lo:t_hi + 1])
            score_2d = past.max(axis=0).astype(np.float32)[:H, :W]
        elif args.source == "ecmwf_s2s":
            cands = [d for d in ecmwf_issues
                     if d <= t and (t - d).days <= args.ecmwf_max_lag_days]
            if not cands:
                n_skip += 1; continue
            issue = max(cands)
            delta = (t - issue).days
            lead_lo, lead_hi = delta + args.lead_start, delta + args.lead_end
            if lead_hi > 215:
                n_skip += 1; continue
            import rasterio
            issue_dir = Path(args.ecmwf_dir) / f"issue_{issue.strftime('%Y%m')}"
            stack = []
            for L in range(lead_lo, lead_hi + 1):
                f = issue_dir / f"lead_{L:03d}.tif"
                if not f.exists():
                    stack = None; break
                with rasterio.open(f) as src:
                    a = src.read(1).astype(np.float32)
                    a[a == src.nodata] = np.nan
                    stack.append(a)
            if stack is None:
                n_skip += 1; continue
            score_2d = np.nanmax(np.stack(stack, 0), axis=0).astype(np.float32)[:H, :W]
            # For baselines, push to per-patch max so spatial grain matches model
            patch_score = img_to_patches_max(score_2d, NR, NC, P)
            score_2d = patches_to_2d(patch_score, NR, NC, P)

        # Common dimension
        score_2d = score_2d[:H, :W]; clim_2d = clim_2d[:H, :W]
        m = compute_window_metrics(score_2d, label_2d, clim_2d, args)
        if m is None:
            n_skip += 1; continue
        m["win_date"] = date_iso
        per_window.append(m)

        if (wi + 1) % 50 == 0:
            print(f"  [{wi+1}/{len(score_files)}] valid={len(per_window)} "
                  f"({(datetime.now()-t0).total_seconds():.0f}s)")

    print(f"\nValid: {len(per_window)}  Skipped: {n_skip}")
    if not per_window:
        sys.exit(1)

    # Aggregate every key with bootstrap CI
    summary = {"source": args.source, "n_windows": len(per_window),
               "scores_dir": ref_dir, "args": vars(args), "metrics": {}}
    keys = sorted(set(k for w in per_window for k in w if k != "win_date"))
    for k in keys:
        vals = [w[k] for w in per_window if k in w and isinstance(w[k], (int, float))
                and np.isfinite(w[k])]
        if not vals:
            continue
        m, lo, hi = bootstrap_ci(vals)
        summary["metrics"][k] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}

    # Print key metrics
    print(f"\n{'Metric':<25} {'Mean':>10} {'95% CI':>22}")
    print("-" * 60)
    for k in ["lift@5000", "lift@30km", "recall@budget_0.05", "recall@budget_0.10",
              "bss", "rss", "anomaly_rho", "roc_auc", "auprc",
              "f2@5000", "mcc@5000"]:
        if k in summary["metrics"]:
            v = summary["metrics"][k]
            print(f"  {k:<25} {v['mean']:>10.4f}  [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_window": per_window}, f, indent=1)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
