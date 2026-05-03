"""
Recall@budget for non-model baselines on the SAME val windows + SAME
labels as the model evaluation, so paper Table N has apples-to-apples
comparison.

Reuses the model's save_window_scores npz files for two things only:
  - the canonical list of val window dates
  - the per-patch label_agg (so labels are identical across all methods)

Then computes a per-patch SCORE map for the chosen baseline:
  --method climatology  → per-patch mean of fire_clim_upto_{year-1}.tif
  --method persistence  → per-patch max over past 7 days of dilated label
                          stack (no leakage — uses dates strictly < t)
  --method ecmwf_s2s    → per-patch max FWI over [t+lead_start, t+lead_end]
                          from reprojected SEAS5 GeoTIFFs

Usage:
  python -m scripts.recall_at_budget_baselines \\
      --method climatology \\
      --scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
      --fire_clim_dir data/fire_clim_annual_nbac \\
      --output_prefix outputs/recall_at_budget_climatology

  python -m scripts.recall_at_budget_baselines \\
      --method persistence \\
      --scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
      --label_npy data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy \\
      --output_prefix outputs/recall_at_budget_persistence

  python -m scripts.recall_at_budget_baselines \\
      --method ecmwf_s2s \\
      --scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
      --ecmwf_dir data/ecmwf_s2s_fire_epsg3978/fwinx \\
      --output_prefix outputs/recall_at_budget_ecmwf_s2s
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

# Reuse helpers from the original
from scripts.recall_at_budget import (
    parse_date, list_score_files, connected_fire_events,
    recall_at_budget, bootstrap_ci, patches_to_2d,
)


def img_to_patches(arr_2d, n_rows, n_cols, P, agg="max"):
    """Aggregate (n_rows*P, n_cols*P) → (n_patches, P*P) preserving sub-pixel layout.
    For baseline scoring we want per-patch values that, after reshape back to
    2D via patches_to_2d, produce a sensible spatial map. Easiest: replicate
    the per-patch aggregate across all 256 sub-pixels of that patch.
    """
    H_full = n_rows * P
    W_full = n_cols * P
    arr = arr_2d[:H_full, :W_full]
    patched = arr.reshape(n_rows, P, n_cols, P).transpose(0, 2, 1, 3) \
        .reshape(n_rows * n_cols, P * P)
    if agg == "max":
        per_patch = patched.max(axis=1, keepdims=True)
    elif agg == "mean":
        per_patch = patched.mean(axis=1, keepdims=True)
    else:
        raise ValueError(agg)
    # broadcast back so every sub-pixel of patch i has same score
    return np.broadcast_to(per_patch, (n_rows * n_cols, P * P)).copy()


def _load_label_npy(path):
    return np.load(path, mmap_mode="r")


def _load_ecmwf_window(ecmwf_dir, issue_date, lead_indices):
    """Load reprojected SEAS5 GeoTIFFs for one issue, given lead indices.
    Returns (n_leads, H, W) float32 with NaN for nodata. None on missing."""
    import rasterio
    issue_dir = Path(ecmwf_dir) / f"issue_{issue_date.strftime('%Y%m')}"
    stack = []
    for L in lead_indices:
        f = issue_dir / f"lead_{L:03d}.tif"
        if not f.exists():
            return None
        with rasterio.open(f) as src:
            a = src.read(1).astype(np.float32)
            a[a == src.nodata] = np.nan
            stack.append(a)
    return np.stack(stack, axis=0)


def _list_ecmwf_issues(ecmwf_dir):
    out = []
    for d in sorted(Path(ecmwf_dir).glob("issue_*")):
        tag = d.name.replace("issue_", "")
        if len(tag) == 6 and tag.isdigit():
            out.append(date(int(tag[:4]), int(tag[4:]), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["climatology", "persistence", "ecmwf_s2s"])
    ap.add_argument("--scores_dir", required=True,
                    help="Model's save_window_scores dir (used for window list + label_agg)")
    ap.add_argument("--fire_clim_dir", default="data/fire_clim_annual_nbac")
    ap.add_argument("--label_npy", default=None,
                    help="Required for persistence (and as fallback)")
    ap.add_argument("--label_data_start", default="2000-05-01")
    ap.add_argument("--ecmwf_dir", default="data/ecmwf_s2s_fire_epsg3978/fwinx")
    ap.add_argument("--pred_start", default="2022-05-01")
    ap.add_argument("--pred_end", default="2025-10-31")
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=46)
    ap.add_argument("--persistence_lookback", type=int, default=7,
                    help="Days back for persistence score (no leakage: < t)")
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.05, 0.10])
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--n_rows", type=int, default=142)
    ap.add_argument("--n_cols", type=int, default=169)
    ap.add_argument("--ecmwf_max_lag_days", type=int, default=35)
    ap.add_argument("--fire_season_only", action="store_true", default=True)
    ap.add_argument("--output_prefix", required=True)
    ap.add_argument("--limit_windows", type=int, default=0)
    args = ap.parse_args()

    pred_start = parse_date(args.pred_start)
    pred_end = parse_date(args.pred_end)

    print("=" * 60)
    print(f"Recall@budget BASELINE — method={args.method}")
    print("=" * 60)

    score_files = list_score_files(args.scores_dir, pred_start, pred_end,
                                   args.fire_season_only)
    if not score_files:
        print(f"[ERROR] no score files in {args.scores_dir}")
        sys.exit(1)
    print(f"  windows: {len(score_files)}")
    if args.limit_windows > 0:
        score_files = score_files[:args.limit_windows]

    P, NR, NC = args.patch_size, args.n_rows, args.n_cols
    H = NR * P
    W = NC * P

    # Label memmap (for persistence) and label_data_start
    if args.method == "persistence" or args.label_npy:
        if not args.label_npy:
            print("[ERROR] --label_npy required for persistence")
            sys.exit(1)
        label_stack = _load_label_npy(args.label_npy)
        label_start = parse_date(args.label_data_start)
    else:
        label_stack = None
        label_start = None

    # Climatology cache: year -> (NR*P, NC*P) array
    clim_cache = {}

    def load_clim_for_year(y):
        if y in clim_cache:
            return clim_cache[y]
        import rasterio
        # Use upto_{y-1} (leak-free convention)
        f = Path(args.fire_clim_dir) / f"fire_clim_upto_{y - 1}.tif"
        if not f.exists():
            f2 = Path(args.fire_clim_dir) / f"fire_clim_upto_2022.tif"
            if not f2.exists():
                return None
            f = f2
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
        if arr.shape != (2281, 2709):
            return None
        clim_cache[y] = arr
        return arr

    # ECMWF issue list
    ecmwf_issues = _list_ecmwf_issues(args.ecmwf_dir) if args.method == "ecmwf_s2s" else None

    per_window = []
    n_skip_no_score = 0
    n_skip_no_fire = 0
    t0 = datetime.now()

    for i, (t, score_path) in enumerate(score_files):
        # Read label_agg from npz so labels are IDENTICAL across all methods
        npz = np.load(score_path)
        if "label_agg" not in npz.files:
            print(f"  [WARN] {score_path.name} missing label_agg — skip")
            continue
        label_patch = npz["label_agg"]   # (n_patches, P*P)
        if label_patch.shape != (NR * NC, P * P):
            continue
        label_2d = (patches_to_2d(label_patch, NR, NC, P) > 0).astype(np.uint8)
        if label_2d.sum() == 0:
            n_skip_no_fire += 1
            continue

        # ── BASELINE-SPECIFIC SCORING ──────────────────────────────────
        score_2d = None

        if args.method == "climatology":
            clim = load_clim_for_year(t.year)
            if clim is None:
                n_skip_no_score += 1
                continue
            score_2d = clim[:H, :W].astype(np.float32)

        elif args.method == "persistence":
            t_lo = (t - timedelta(days=args.persistence_lookback) - label_start).days
            t_hi = (t - timedelta(days=1) - label_start).days
            if t_lo < 0 or t_hi >= label_stack.shape[0]:
                n_skip_no_score += 1
                continue
            past = np.array(label_stack[t_lo:t_hi + 1])  # (lookback, H, W)
            score_2d = past.max(axis=0).astype(np.float32)[:H, :W]

        elif args.method == "ecmwf_s2s":
            # Most recent issue ≤ t within max_lag
            cands = [d for d in ecmwf_issues
                     if d <= t and (t - d).days <= args.ecmwf_max_lag_days]
            if not cands:
                n_skip_no_score += 1
                continue
            issue = max(cands)
            delta = (t - issue).days
            lead_lo = delta + args.lead_start
            lead_hi = delta + args.lead_end
            if lead_hi > 215:
                n_skip_no_score += 1
                continue
            fc = _load_ecmwf_window(args.ecmwf_dir, issue,
                                    list(range(lead_lo, lead_hi + 1)))
            if fc is None:
                n_skip_no_score += 1
                continue
            score_2d = np.nanmax(fc, axis=0).astype(np.float32)[:H, :W]

        # All baselines: convert 2D score to per-patch (max-pool to 16×16)
        # so the score and label are on the SAME spatial grid as the model.
        score_patch = img_to_patches(score_2d, NR, NC, P, agg="max")
        score_2d_final = patches_to_2d(score_patch, NR, NC, P)

        # If score has nan, use a finite mask
        valid_mask = np.isfinite(score_2d_final)
        if valid_mask.sum() == 0:
            n_skip_no_score += 1
            continue
        score_2d_final = np.where(valid_mask, score_2d_final, -np.inf)

        event_lbl, n_events = connected_fire_events(label_2d)

        rec_per_budget = []
        for B in args.budgets:
            r = recall_at_budget(score_2d_final, label_2d, valid_mask, B,
                                 event_lbl, n_events)
            rec_per_budget.append(r)

        per_window.append({
            "win_date": t.isoformat(),
            "n_fire_pixels": int(label_2d.sum()),
            "n_events": int(n_events),
            "n_valid_pixels": int(valid_mask.sum()),
            "by_budget": rec_per_budget,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(score_files)}] valid={len(per_window)} "
                  f"({(datetime.now()-t0).total_seconds():.0f}s)")

    print(f"\nSummary  total={len(score_files)}  valid={len(per_window)}  "
          f"no_score={n_skip_no_score}  no_fire={n_skip_no_fire}")
    if not per_window:
        print("[ERROR] no valid windows")
        sys.exit(1)

    summary = {"method": args.method, "n_windows": len(per_window),
               "by_budget": [], "args": vars(args)}
    print(f"\n  Recall@budget (mean ± 95% CI):")
    for bi, B in enumerate(args.budgets):
        recalls = [w["by_budget"][bi]["recall"] for w in per_window]
        m, lo, hi = bootstrap_ci(recalls)
        k_mean = float(np.mean([w["by_budget"][bi]["k_pixels"] for w in per_window]))
        cov = float(np.mean([w["by_budget"][bi]["covered_events"] for w in per_window]))
        ne = float(np.mean([w["n_events"] for w in per_window]))
        print(f"    {B*100:5.2f} %  recall = {m:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"({cov:.1f}/{ne:.1f} events)")
        summary["by_budget"].append({
            "budget_frac": float(B), "mean_k_pixels": k_mean,
            "recall_mean": m, "recall_ci_lo": lo, "recall_ci_hi": hi,
            "covered_events_mean": cov, "total_events_mean": ne,
        })

    out_pw = f"{args.output_prefix}_per_window.json"
    out_sm = f"{args.output_prefix}_summary.json"
    Path(out_pw).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pw, "w") as f: json.dump(per_window, f, indent=1)
    with open(out_sm, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n  per-window JSON: {out_pw}")
    print(f"  summary JSON   : {out_sm}")


if __name__ == "__main__":
    main()
