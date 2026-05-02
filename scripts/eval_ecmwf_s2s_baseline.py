"""
Evaluate ECMWF S2S Fire Danger Forecast as a baseline against NBAC+NFDB
binary fire labels.

For each model issue date t in the validation period (2022-2025):
  1. Find the SEAS5 forecast issue date <= t (within max_lag_days)
  2. Extract leadtimes [t+14, t+46] (33 days) from that issue
  3. Per pixel: take MAX FWI over those 33 days = predicted-risk score
  4. Compute Lift@K and Lift@30km against the binary fire label

Output:
  outputs/baseline_ecmwf_s2s_per_window.json    per-window CIs
  outputs/baseline_ecmwf_s2s_summary.json       mean + bootstrap 95% CI

Usage:
  python -m scripts.eval_ecmwf_s2s_baseline \\
      --s2s_dir data/ecmwf_s2s_fire_epsg3978/fwinx \\
      --label_npy data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy \\
      --pred_start 2022-05-01 --pred_end 2025-10-31 \\
      --lead_start 14 --lead_end 46 \\
      --val_lift_k 5000 \\
      --output_prefix outputs/baseline_ecmwf_s2s
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def list_s2s_issues(s2s_dir):
    """Return sorted list of issue dates (date objects) found in s2s_dir."""
    out = []
    for d in sorted(Path(s2s_dir).glob("issue_*")):
        tag = d.name.replace("issue_", "")
        if len(tag) != 6 or not tag.isdigit():
            continue
        yr, mo = int(tag[:4]), int(tag[4:])
        out.append(date(yr, mo, 1))
    return out


def find_best_issue(t, available_issues, max_lag_days=35):
    """Most recent issue date <= t, within max_lag_days."""
    candidates = [d for d in available_issues if d <= t and (t - d).days <= max_lag_days]
    if not candidates:
        return None
    return max(candidates)


def load_lead_window(s2s_dir, issue_date, lead_indices):
    """
    Load FWI for issue_date at the requested lead day indices (1-indexed).
    Returns: (n_leads, H, W) float32 stack, with NaN/-9999 → np.nan.
    """
    import rasterio
    issue_dir = Path(s2s_dir) / f"issue_{issue_date.strftime('%Y%m')}"
    stack = []
    for lead in lead_indices:
        f = issue_dir / f"lead_{lead:03d}.tif"
        if not f.exists():
            return None
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr == src.nodata] = np.nan
            stack.append(arr)
    return np.stack(stack, axis=0)


def lift_at_k(scores_flat, labels_flat, k):
    """Standard Lift@K: precision at top-k / base rate."""
    valid = np.isfinite(scores_flat) & np.isfinite(labels_flat)
    s = scores_flat[valid]
    y = labels_flat[valid]
    if len(s) == 0 or y.sum() == 0:
        return float("nan")
    base = y.mean()
    if base <= 0:
        return float("nan")
    k = min(k, len(s))
    top_k = np.argpartition(-s, k - 1)[:k]
    prec = y[top_k].mean()
    return prec / base


def lift_at_30km_pooled(scores_2d, labels_2d, k, pool_px=15):
    """
    Pool predictions + labels into 30 km cells (15 px @ 2 km/px) via max,
    then compute Lift@K on the pooled grid.
    """
    H, W = scores_2d.shape
    Hp, Wp = H // pool_px, W // pool_px
    s_crop = scores_2d[:Hp * pool_px, :Wp * pool_px]
    y_crop = labels_2d[:Hp * pool_px, :Wp * pool_px]

    s_pooled = s_crop.reshape(Hp, pool_px, Wp, pool_px) \
        .max(axis=(1, 3))
    y_pooled = y_crop.reshape(Hp, pool_px, Wp, pool_px) \
        .max(axis=(1, 3))

    return lift_at_k(s_pooled.flatten(), y_pooled.flatten(),
                     k * pool_px ** 2 // (pool_px ** 2))


def bootstrap_ci(values, n_boot=1000, seed=0):
    """Mean + 95% CI via bootstrap."""
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
    ap.add_argument("--s2s_dir", type=str,
                    default="data/ecmwf_s2s_fire_epsg3978/fwinx")
    ap.add_argument("--label_npy", type=str, required=True,
                    help="NBAC + NFDB binary label stack (.npy memmap), "
                         "shape (T, H, W) starting from data_start")
    ap.add_argument("--label_data_start", type=str, default="2000-05-01",
                    help="First date of label_npy (default 2000-05-01)")
    ap.add_argument("--pred_start", type=str, default="2022-05-01")
    ap.add_argument("--pred_end", type=str, default="2025-10-31")
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=46)
    ap.add_argument("--val_lift_k", type=int, default=5000)
    ap.add_argument("--max_lag_days", type=int, default=35,
                    help="Max days between SEAS5 issue and model issue date")
    ap.add_argument("--fire_season_only", action="store_true", default=True)
    ap.add_argument("--output_prefix", type=str,
                    default="outputs/baseline_ecmwf_s2s")
    ap.add_argument("--limit_windows", type=int, default=0,
                    help="If > 0, only evaluate first N windows (debug)")
    args = ap.parse_args()

    try:
        import rasterio  # noqa: F401
    except ImportError:
        print("[ERROR] rasterio not installed.")
        sys.exit(1)

    pred_start = parse_date(args.pred_start)
    pred_end = parse_date(args.pred_end)
    label_start = parse_date(args.label_data_start)
    fire_months = set(range(4, 11))   # Apr-Oct

    print("="*60)
    print("ECMWF S2S Fire Danger BASELINE evaluation")
    print("="*60)
    print(f"  Val window: {pred_start} - {pred_end}")
    print(f"  Lead window: t+{args.lead_start} .. t+{args.lead_end}")
    print(f"  K: {args.val_lift_k}")
    print(f"  S2S dir: {args.s2s_dir}")
    print(f"  Label: {args.label_npy}")

    # List available SEAS5 issue dates
    issues = list_s2s_issues(args.s2s_dir)
    if not issues:
        print(f"[ERROR] no SEAS5 issues found in {args.s2s_dir}")
        sys.exit(1)
    print(f"  SEAS5 issues available: {len(issues)}  "
          f"({issues[0]} .. {issues[-1]})")

    # Load label stack as memmap
    print(f"[STEP 1] Loading label memmap")
    # Try to infer shape from filename
    fname = Path(args.label_npy).stem
    # Expected pattern: ..._YYYY-MM-DD_YYYY-MM-DD_HxW
    parts = fname.split("_")
    H, W = None, None
    for p in parts[::-1]:
        if "x" in p and p.replace("x", "").isdigit():
            try:
                H, W = map(int, p.split("x"))
                break
            except ValueError:
                continue
    if H is None or W is None:
        print(f"[ERROR] could not parse H,W from filename {fname}")
        sys.exit(1)
    file_size = os.path.getsize(args.label_npy)
    # .npy has a header — easier: load with mmap_mode
    label_stack = np.load(args.label_npy, mmap_mode="r")
    if label_stack.ndim != 3:
        print(f"[ERROR] expected (T,H,W); got {label_stack.shape}")
        sys.exit(1)
    print(f"  label shape: {label_stack.shape}")

    # Build val date list
    val_dates = []
    cur = pred_start
    while cur <= pred_end:
        if not args.fire_season_only or cur.month in fire_months:
            val_dates.append(cur)
        cur += timedelta(days=1)
    print(f"[STEP 2] Val windows: {len(val_dates)}")
    if args.limit_windows > 0:
        val_dates = val_dates[:args.limit_windows]
        print(f"  limited to first {len(val_dates)}")

    # Iterate
    per_window = []
    n_skip_no_issue = 0
    n_skip_no_leads = 0
    n_skip_no_label = 0
    n_skip_no_fire = 0
    t0 = datetime.now()

    for i, t in enumerate(val_dates):
        # Find SEAS5 issue
        issue = find_best_issue(t, issues, args.max_lag_days)
        if issue is None:
            n_skip_no_issue += 1
            continue

        # Compute lead indices for this t
        # Lead day k (1-indexed) corresponds to issue + k days.
        # We want forecast for dates [t + lead_start, t + lead_end].
        # → lead indices = [(t - issue).days + lead_start,
        #                   (t - issue).days + lead_end]
        delta = (t - issue).days
        lead_lo = delta + args.lead_start
        lead_hi = delta + args.lead_end
        if lead_hi > 215:
            n_skip_no_leads += 1
            continue
        lead_indices = list(range(lead_lo, lead_hi + 1))

        # Load forecast stack (n_leads, H, W)
        try:
            fc = load_lead_window(args.s2s_dir, issue, lead_indices)
        except Exception as e:
            print(f"  [WARN] {t} load failed: {e}")
            continue
        if fc is None:
            n_skip_no_leads += 1
            continue

        # Per-pixel max FWI over lead window → score map
        score_map = np.nanmax(fc, axis=0).astype(np.float32)

        # Build label: max over [t+lead_start, t+lead_end] in label_stack
        t_lo = (t + timedelta(days=args.lead_start) - label_start).days
        t_hi = (t + timedelta(days=args.lead_end) - label_start).days
        if t_lo < 0 or t_hi >= label_stack.shape[0]:
            n_skip_no_label += 1
            continue
        label_window = np.array(label_stack[t_lo:t_hi + 1])  # (33, H, W)
        label_map = label_window.max(axis=0).astype(np.uint8)

        if label_map.sum() == 0:
            n_skip_no_fire += 1
            continue

        # Compute lift
        l5k = lift_at_k(score_map.flatten(), label_map.flatten(),
                        args.val_lift_k)
        l30 = lift_at_30km_pooled(score_map, label_map, args.val_lift_k)

        per_window.append({
            "win_date": t.isoformat(),
            "issue_date": issue.isoformat(),
            "delta_days": delta,
            "n_fire": int(label_map.sum()),
            "lift_5000": float(l5k),
            "lift_30km": float(l30),
        })

        if (i + 1) % 50 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  [{i+1}/{len(val_dates)}] processed {len(per_window)} "
                  f"valid windows  ({elapsed:.0f}s)")

    print(f"\n[STEP 3] Summary")
    print(f"  Total val dates: {len(val_dates)}")
    print(f"  Valid windows:   {len(per_window)}")
    print(f"  Skipped: no_issue={n_skip_no_issue}  no_leads={n_skip_no_leads}  "
          f"no_label={n_skip_no_label}  no_fire={n_skip_no_fire}")

    if not per_window:
        print("[ERROR] no valid windows")
        sys.exit(1)

    lifts_5k = [w["lift_5000"] for w in per_window]
    lifts_30 = [w["lift_30km"] for w in per_window]
    m5, lo5, hi5 = bootstrap_ci(lifts_5k)
    m30, lo30, hi30 = bootstrap_ci(lifts_30)

    print(f"\n  Lift@5000  : {m5:.3f}× [95% CI: {lo5:.3f} – {hi5:.3f}]  (n={len(lifts_5k)})")
    print(f"  Lift@30 km : {m30:.3f}× [95% CI: {lo30:.3f} – {hi30:.3f}]  (n={len(lifts_30)})")

    # Write outputs
    out_pw = f"{args.output_prefix}_per_window.json"
    out_sm = f"{args.output_prefix}_summary.json"
    Path(out_pw).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pw, "w") as f:
        json.dump(per_window, f, indent=1)
    summary = {
        "n_windows": len(per_window),
        "lift_5000": {"mean": m5, "ci_lo": lo5, "ci_hi": hi5},
        "lift_30km": {"mean": m30, "ci_lo": lo30, "ci_hi": hi30},
        "skip_counts": {
            "no_issue": n_skip_no_issue,
            "no_leads": n_skip_no_leads,
            "no_label": n_skip_no_label,
            "no_fire": n_skip_no_fire,
        },
        "args": vars(args),
    }
    with open(out_sm, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Per-window JSON: {out_pw}")
    print(f"  Summary JSON:    {out_sm}")


if __name__ == "__main__":
    main()
