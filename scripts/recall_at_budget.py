"""
Recall@budget — operational metric proposed by advisor.

For each model val window t in [pred_start, pred_end]:
  1. Load predicted-risk score map for this window (from
     save_window_scores_full output: outputs/window_scores_full/<run>/window_NNNN_YYYY-MM-DD.npz).
  2. Rank all valid Canada land pixels by predicted prob.
  3. For each budget B ∈ {0.1%, 0.5%, 1%, 5%, 10%}:
       - Select the top (B × n_valid_pixels) pixels = "patrol mask"
       - Count CONNECTED FIRE EVENTS in the label that intersect
         the mask (8-connectivity on the binary fire stack max-pooled
         over the lead window)
       - Recall = covered_events / total_events_in_window

Output:
  outputs/recall_at_budget_<run_tag>_per_window.json   per-window
  outputs/recall_at_budget_<run_tag>_summary.json      mean + bootstrap CI

Usage:
  python -m scripts.recall_at_budget \\
      --scores_dir outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
      --label_npy data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy \\
      --label_data_start 2000-05-01 \\
      --pred_start 2022-05-01 --pred_end 2025-10-31 \\
      --lead_start 14 --lead_end 46 \\
      --budgets 0.001 0.005 0.01 0.05 0.10 \\
      --output_prefix outputs/recall_at_budget_enc21_12y
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def list_score_files(scores_dir, pred_start, pred_end, fire_season_only=True):
    """Return list of (date, path) for score files in date range."""
    fire_months = set(range(4, 11))
    out = []
    pat = re.compile(r"window_\d+_(\d{4}-\d{2}-\d{2})\.npz$")
    for f in sorted(Path(scores_dir).glob("window_*.npz")):
        m = pat.search(f.name)
        if not m:
            continue
        d = parse_date(m.group(1))
        if d < pred_start or d > pred_end:
            continue
        if fire_season_only and d.month not in fire_months:
            continue
        out.append((d, f))
    return out


def load_score_map(npz_path):
    """Load score map from save_window_scores npz; expected key: 'scores'."""
    arr = np.load(npz_path)
    if "scores" in arr:
        return arr["scores"]
    # fallback: first array
    return arr[arr.files[0]]


def connected_fire_events(label_2d, structure=None):
    """Return labelled connected components of binary fire raster."""
    from scipy.ndimage import label as ndi_label
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)   # 8-connectivity
    lbl, n = ndi_label(label_2d > 0, structure=structure)
    return lbl, n


def recall_at_budget(score_map, label_2d, valid_mask, budget_frac,
                     event_lbl, n_events):
    """
    Args:
      score_map  : (H, W) predicted prob
      label_2d   : (H, W) binary fire (1 = positive)
      valid_mask : (H, W) bool, True = land pixel eligible for ranking
      budget_frac: float, e.g. 0.01 = top 1 %
      event_lbl  : (H, W) int connected-component labels (output of ndi_label)
      n_events   : int, max label id (= number of distinct fire events)
    Returns:
      dict with keys: budget, k_pixels, recall, covered_events, total_events
    """
    n_valid = int(valid_mask.sum())
    if n_valid == 0 or n_events == 0:
        return {"budget": budget_frac, "k_pixels": 0, "recall": float("nan"),
                "covered_events": 0, "total_events": int(n_events)}

    k = max(1, int(round(budget_frac * n_valid)))
    # Take top-k score within valid_mask
    flat_scores = np.where(valid_mask, score_map, -np.inf).ravel()
    top_idx = np.argpartition(-flat_scores, k - 1)[:k]
    top_mask_flat = np.zeros(flat_scores.shape, dtype=bool)
    top_mask_flat[top_idx] = True
    top_mask = top_mask_flat.reshape(score_map.shape)

    # Which event labels intersect the top-mask?
    hit_labels = np.unique(event_lbl[top_mask & (event_lbl > 0)])
    covered = int(len(hit_labels))
    recall = covered / float(n_events)
    return {"budget": float(budget_frac), "k_pixels": int(k),
            "recall": float(recall), "covered_events": covered,
            "total_events": int(n_events)}


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
    ap.add_argument("--scores_dir", type=str, required=True,
                    help="outputs/window_scores_full/<run> directory")
    ap.add_argument("--label_npy", type=str, required=True)
    ap.add_argument("--label_data_start", type=str, default="2000-05-01")
    ap.add_argument("--pred_start", type=str, default="2022-05-01")
    ap.add_argument("--pred_end", type=str, default="2025-10-31")
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=46)
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.05, 0.10])
    ap.add_argument("--fire_season_only", action="store_true", default=True)
    ap.add_argument("--output_prefix", type=str, required=True)
    ap.add_argument("--limit_windows", type=int, default=0,
                    help="If > 0, only N windows for debug")
    args = ap.parse_args()

    pred_start = parse_date(args.pred_start)
    pred_end = parse_date(args.pred_end)
    label_start = parse_date(args.label_data_start)

    print("=" * 60)
    print("Recall@budget evaluation")
    print("=" * 60)
    print(f"  scores_dir : {args.scores_dir}")
    print(f"  label_npy  : {args.label_npy}")
    print(f"  val window : {pred_start} → {pred_end}")
    print(f"  lead window: t+{args.lead_start} .. t+{args.lead_end}")
    print(f"  budgets    : {args.budgets}")

    # List score files
    score_files = list_score_files(args.scores_dir, pred_start, pred_end,
                                   args.fire_season_only)
    if not score_files:
        print(f"[ERROR] no score files in {args.scores_dir}")
        sys.exit(1)
    print(f"  score files: {len(score_files)}")
    if args.limit_windows > 0:
        score_files = score_files[:args.limit_windows]
        print(f"  limited to {len(score_files)}")

    # Load label memmap
    label_stack = np.load(args.label_npy, mmap_mode="r")
    print(f"  label shape: {label_stack.shape}")
    H, W = label_stack.shape[1], label_stack.shape[2]

    per_window = []
    n_skip_no_fire = 0
    n_skip_shape_mismatch = 0
    t0 = datetime.now()

    for i, (t, score_path) in enumerate(score_files):
        # Build label window: max over [t+lead_start, t+lead_end]
        t_lo = (t + timedelta(days=args.lead_start) - label_start).days
        t_hi = (t + timedelta(days=args.lead_end) - label_start).days
        if t_lo < 0 or t_hi >= label_stack.shape[0]:
            continue
        label_win = np.array(label_stack[t_lo:t_hi + 1])  # (33, H, W)
        label_2d = (label_win.max(axis=0) > 0).astype(np.uint8)
        if label_2d.sum() == 0:
            n_skip_no_fire += 1
            continue

        # Load score map
        try:
            score_map = load_score_map(score_path)
        except Exception as e:
            print(f"  [WARN] failed to load {score_path.name}: {e}")
            continue
        if score_map.shape != (H, W):
            n_skip_shape_mismatch += 1
            continue

        # Valid land mask = pixels with finite scores AND positive base value.
        # save_window_scores files use 0 for non-land pixels typically; here we
        # accept all finite pixels as "valid".
        valid_mask = np.isfinite(score_map)

        # Connected fire events in this lead window
        event_lbl, n_events = connected_fire_events(label_2d)

        rec_per_budget = []
        for B in args.budgets:
            r = recall_at_budget(score_map, label_2d, valid_mask, B,
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
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  [{i+1}/{len(score_files)}] valid={len(per_window)} "
                  f"({elapsed:.0f}s)")

    print(f"\nSummary")
    print(f"  Total: {len(score_files)}  Valid: {len(per_window)}  "
          f"no_fire: {n_skip_no_fire}  shape_mismatch: {n_skip_shape_mismatch}")
    if not per_window:
        print("[ERROR] no valid windows")
        sys.exit(1)

    # Aggregate per budget
    summary = {"n_windows": len(per_window), "by_budget": [], "args": vars(args)}
    print(f"\n  Recall@budget (mean ± 95% bootstrap CI):")
    for bi, B in enumerate(args.budgets):
        recalls = [w["by_budget"][bi]["recall"] for w in per_window]
        m, lo, hi = bootstrap_ci(recalls)
        # Also report mean k_pixels (changes per window because n_valid varies)
        k_mean = float(np.mean([w["by_budget"][bi]["k_pixels"]
                                for w in per_window]))
        cov_mean = float(np.mean([w["by_budget"][bi]["covered_events"]
                                  for w in per_window]))
        n_evt_mean = float(np.mean([w["n_events"] for w in per_window]))
        print(f"    {B*100:5.2f} % budget (~{k_mean:.0f} pixels): "
              f"recall = {m:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"({cov_mean:.1f}/{n_evt_mean:.1f} events)")
        summary["by_budget"].append({
            "budget_frac": float(B),
            "mean_k_pixels": k_mean,
            "recall_mean": m, "recall_ci_lo": lo, "recall_ci_hi": hi,
            "covered_events_mean": cov_mean,
            "total_events_mean": n_evt_mean,
        })

    out_pw = f"{args.output_prefix}_per_window.json"
    out_sm = f"{args.output_prefix}_summary.json"
    Path(out_pw).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pw, "w") as f:
        json.dump(per_window, f, indent=1)
    with open(out_sm, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  per-window JSON: {out_pw}")
    print(f"  summary JSON   : {out_sm}")


if __name__ == "__main__":
    main()
