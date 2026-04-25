#!/usr/bin/env python3
"""
Re-evaluate baselines under the *novel-ignition* metric.

Standard "lift_total" gives credit for predicting where fire CONTINUES.
With NBAC polygon labels + sustained mega-fires (2022-2025), this lets a
trivial persistence baseline hit Lift@5000=17x — which has zero
operational value because fire managers already know where active
fires are. The interesting prediction task is: where will NEW fires
start that aren't already burning?

This script computes BOTH metrics for every baseline:

  lift_total      : positive label = fire in [ts, te)  (standard)
  lift_novel_30d  : positive label = fire in [ts, te) AND no fire in
                    [hs - 30, he)  (excludes patches that were burning
                    in the last 30 days before the lead window)
  lift_novel_90d  : same, but 90-day lookback (whole-season novel)

Expected pattern:
  - persistence: lift_total HIGH  → lift_novel CRASHES  (only knows
    "where it's burning")
  - climatology: lift_total mid   → lift_novel mid       (spatial prior
    helps both)
  - logreg:      lift_total mid   → lift_novel mid
  - fwi_oracle:  lift_total ~0    → lift_novel ~0        (geographic
    mismatch with NBAC)

Usage:
    python -m scripts.evaluate_novel_ignition \\
        --config configs/paths_narval.yaml \\
        --pred_start 2022-05-01 --pred_end 2024-10-31 \\
        --fire_label_npy data/fire_labels/fire_labels_nbac_nfdb_*.npy \\
        --climatology_tif data/fire_clim_annual_nbac/fire_clim_upto_2022.tif \\
        --output_csv outputs/benchmark_novel.csv
"""
import argparse
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rasterio  # noqa: E402

from src.evaluation.benchmark_baselines import (  # noqa: E402
    load_data, _build_file_index, _read_tif, _patchify_frame,
    _build_s2s_windows_calendar,
)


def _per_window_lift(score, label, k_values, grid, patch_size):
    """Compute Lift@K per window. Returns dict {k: {lift, prec, n_fire, base_rate}}."""
    score_flat = score.reshape(-1)
    label_flat = label.reshape(-1)
    n_fire = int(label_flat.sum())
    n_total = label_flat.size
    base_rate = n_fire / max(n_total, 1)
    out = {}
    if n_fire == 0:
        for k in k_values:
            out[k] = {"lift": float('nan'), "prec": float('nan'),
                      "n_fire": 0, "base_rate": float('nan')}
        return out
    order = np.argsort(score_flat)[::-1]
    for k in k_values:
        topk = order[:k]
        tp = int(label_flat[topk].sum())
        prec = tp / k
        lift = prec / base_rate if base_rate > 0 else float('nan')
        out[k] = {"lift": lift, "prec": prec, "n_fire": n_fire,
                  "base_rate": base_rate}
    return out


def _build_novel_label(win, fire_p, lookback_days):
    """Return (label per pixel) where 1 = novel ignition.
    fire_p shape: (T, n_patches, P²) uint8."""
    hs, he, ts, te = win
    look_start = max(0, hs - lookback_days)
    burning_recently = fire_p[look_start:he].sum(axis=0) > 0
    will_burn = fire_p[ts:te].sum(axis=0) > 0
    novel = will_burn & ~burning_recently
    return novel.astype(np.uint8)


def _build_total_label(win, fire_p):
    hs, he, ts, te = win
    will_burn = fire_p[ts:te].sum(axis=0) > 0
    return will_burn.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred_start", default="2022-05-01")
    ap.add_argument("--pred_end", default="2024-10-31")
    ap.add_argument("--in_days", type=int, default=7)
    ap.add_argument("--lead_start", type=int, default=14)
    ap.add_argument("--lead_end", type=int, default=45)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--dilate_radius", type=int, default=14)
    ap.add_argument("--k_values", nargs="+", type=int, default=[5000])
    ap.add_argument("--n_sample_wins", type=int, default=20)
    ap.add_argument("--fire_season_only", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--climatology_tif", required=True)
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--output_csv", default="outputs/benchmark_novel.csv")
    args = ap.parse_args()

    print("=" * 70)
    print("NOVEL-IGNITION BASELINE EVALUATION")
    print("=" * 70)

    # Reuse benchmark_baselines loader
    (fwi_p, fire_p, clim_p, all_dates, date_to_idx,
     val_wins, val_win_dates, n_patches, grid) = load_data(
        args.config, args.pred_start, args.pred_end,
        args.in_days, args.lead_start, args.lead_end,
        args.patch_size, args.dilate_radius,
        args.fire_season_only, args.climatology_tif,
        args.fire_label_npy,
    )

    # Sample a fixed subset of val windows for fair comparison
    rng = np.random.default_rng(0)
    if len(val_wins) > args.n_sample_wins:
        idxs = rng.choice(len(val_wins), args.n_sample_wins, replace=False)
        val_wins_s = [val_wins[i] for i in sorted(idxs)]
    else:
        val_wins_s = val_wins
    print(f"  evaluating on {len(val_wins_s)} sampled val windows")

    # ── Score functions (same logic as benchmark_baselines.py) ──────────
    def score_fwi(win):
        hs, he, ts, te = win
        return np.nan_to_num(fwi_p[:, ts:te, :].astype(np.float32),
                             nan=0.0).max(axis=1)

    def score_clim(win):
        return clim_p.astype(np.float32)

    def score_persistence(win):
        hs, he, ts, te = win
        return fire_p[hs:he].astype(np.float32).mean(axis=0)

    score_fns = {
        "fwi_oracle": score_fwi,
        "climatology": score_clim,
        "persistence": score_persistence,
    }

    # ── Run eval per baseline × per (total | novel_30d | novel_90d) ─────
    label_modes = {
        "total":     lambda w: _build_total_label(w, fire_p),
        "novel_30d": lambda w: _build_novel_label(w, fire_p, lookback_days=30),
        "novel_90d": lambda w: _build_novel_label(w, fire_p, lookback_days=90),
    }

    rows = []
    for bname, score_fn in score_fns.items():
        for lname, label_fn in label_modes.items():
            print(f"\n  {bname:12s} × {lname:10s}")
            lifts_per_win = {k: [] for k in args.k_values}
            n_novel_per_win = []
            n_total_per_win = []
            t0 = time.time()
            for i, win in enumerate(val_wins_s):
                score = score_fn(win)
                label = label_fn(win)
                n_novel_per_win.append(int(label.sum()))
                n_total_per_win.append(int(_build_total_label(win, fire_p).sum()))
                m = _per_window_lift(score, label, args.k_values,
                                     grid, args.patch_size)
                for k in args.k_values:
                    if not np.isnan(m[k]["lift"]):
                        lifts_per_win[k].append(m[k]["lift"])
            for k in args.k_values:
                if lifts_per_win[k]:
                    mean_lift = float(np.mean(lifts_per_win[k]))
                    std_lift  = float(np.std(lifts_per_win[k]))
                    n_eval = len(lifts_per_win[k])
                else:
                    mean_lift = float('nan')
                    std_lift = float('nan')
                    n_eval = 0
                novel_frac = (np.mean(n_novel_per_win) /
                              max(np.mean(n_total_per_win), 1))
                print(f"    L@{k} = {mean_lift:.2f} ± {std_lift:.2f}x  "
                      f"(n_eval={n_eval}, novel/total={novel_frac:.1%})")
                rows.append({
                    "baseline": bname, "label_mode": lname, "k": k,
                    "lift_mean": mean_lift, "lift_std": std_lift,
                    "n_eval": n_eval,
                    "novel_pixels_mean": float(np.mean(n_novel_per_win)),
                    "total_pixels_mean": float(np.mean(n_total_per_win)),
                    "novel_frac": novel_frac,
                })

    # ── Write CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n  wrote {args.output_csv}")

    # ── Print summary table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY  (Lift@5000)")
    print("=" * 70)
    print(f"  {'Baseline':14s}  {'total':>14s}  {'novel_30d':>14s}  {'novel_90d':>14s}")
    by_b = {}
    for r in rows:
        if r["k"] != 5000:
            continue
        by_b.setdefault(r["baseline"], {})[r["label_mode"]] = r["lift_mean"]
    for b in ["fwi_oracle", "climatology", "persistence"]:
        d = by_b.get(b, {})
        t = d.get("total", float('nan'))
        n30 = d.get("novel_30d", float('nan'))
        n90 = d.get("novel_90d", float('nan'))
        print(f"  {b:14s}  {t:14.2f}  {n30:14.2f}  {n90:14.2f}")
    print("\n  (model lift for novel labels requires checkpoint inference; "
          "see follow-up script.)")


if __name__ == "__main__":
    main()
