"""Aggregate forward-chaining LOYO fold results into a single LOYO score.

For each held-out year Y, we trained on 2014-05-01 .. (Y-1)-12-31 and
evaluated on Y's fire season, producing
    outputs/loyo/v3_9ch_enc21_loyo_val{Y}_per_window.json

This script reads the 5 JSONs and reports:

  * macro-LOYO: mean +/- std of fold-level Lift@5000 / Lift@30km / BSS / F2
    (each year weighted equally — the headline LOYO number)
  * micro-LOYO: pool every val window across folds and compute mean + 95% CI
    (sanity-check via bootstrap, accounts for per-fold window count)

Run after all 5 LOYO sbatch jobs finish:

    python3 scripts/aggregate_loyo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Reuse the canonical bootstrap so numbers match metrics.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import bootstrap_ci  # noqa: E402


HEADLINE_METRICS = ["lift_k", "lift_coarse", "bss", "recall_k"]
HEADLINE_LABEL = {
    "lift_k": "Lift@5000",
    "lift_coarse": "Lift@30km",
    "bss": "BSS",
    "recall_k": "Recall@5000",
}


def load_fold(json_path: Path) -> tuple[dict, list[dict]]:
    """Return (summary_dict, per_window_list) for one fold JSON."""
    with open(json_path) as f:
        d = json.load(f)
    return d.get("summary", {}), d.get("per_window", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str,
                    default="outputs/loyo",
                    help="Directory containing v3_9ch_enc21_loyo_val{Y}_per_window.json")
    ap.add_argument("--run_prefix", type=str,
                    default="v3_9ch_enc21_loyo_val",
                    help="Filename prefix before {YEAR}_per_window.json")
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--out_csv", type=str,
                    default="outputs/loyo/loyo_summary.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    found = {}
    missing = []
    for y in args.years:
        p = out_dir / f"{args.run_prefix}{y}_per_window.json"
        if p.exists():
            found[y] = load_fold(p)
        else:
            missing.append(y)

    if missing:
        print(f"WARNING: missing folds: {missing}")
    if not found:
        print("ERROR: no fold JSONs found — nothing to aggregate.")
        return 1

    print(f"\nLoaded {len(found)} folds: {sorted(found.keys())}\n")

    # ---------- Per-fold table ----------
    print("=" * 76)
    print("Per-fold summary (each row = one held-out year)")
    print("=" * 76)
    header = f"{'year':>6} {'n_win':>6} {'n_fire':>9} " + " ".join(
        f"{HEADLINE_LABEL[m]:>12}" for m in HEADLINE_METRICS
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for y in sorted(found.keys()):
        summ, _per_win = found[y]
        row = {
            "year": y,
            "n_win": int(summ.get("n_windows", 0)),
            "n_fire": int(summ.get("n_fire", 0)),
        }
        for m in HEADLINE_METRICS:
            row[m] = float(summ.get(m, float("nan")))
        rows.append(row)
        print(f"{row['year']:>6} {row['n_win']:>6} {row['n_fire']:>9} " +
              " ".join(f"{row[m]:>12.4f}" for m in HEADLINE_METRICS))

    # ---------- Macro LOYO (mean +/- std across folds) ----------
    print()
    print("=" * 76)
    print("Macro LOYO (per-fold mean -> mean +/- std across years)")
    print("=" * 76)
    macro = {}
    for m in HEADLINE_METRICS:
        vals = np.array([r[m] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        macro[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_folds": int(len(vals)),
        }
        print(f"  {HEADLINE_LABEL[m]:>12}: "
              f"{macro[m]['mean']:.4f} +/- {macro[m]['std']:.4f}  "
              f"(min={macro[m]['min']:.4f}, max={macro[m]['max']:.4f}, "
              f"n={macro[m]['n_folds']})")

    # ---------- Micro LOYO (pool every val window, bootstrap CI) ----------
    print()
    print("=" * 76)
    print("Micro LOYO (pool every val window across folds, bootstrap 95% CI)")
    print("=" * 76)
    pooled: dict[str, list[float]] = {m: [] for m in HEADLINE_METRICS}
    for _y, (_summ, per_win) in found.items():
        for w in per_win:
            for m in HEADLINE_METRICS:
                if m in w:
                    pooled[m].append(float(w[m]))

    micro = {}
    for m in HEADLINE_METRICS:
        vals = np.array(pooled[m], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        ci = bootstrap_ci(vals.tolist(), n_boot=1000, alpha=0.05)
        micro[m] = {
            "mean": float(np.mean(vals)),
            "ci_low": float(ci["ci_low"]),
            "ci_high": float(ci["ci_high"]),
            "n_windows": int(len(vals)),
        }
        print(f"  {HEADLINE_LABEL[m]:>12}: "
              f"{micro[m]['mean']:.4f} "
              f"[{micro[m]['ci_low']:.4f}, {micro[m]['ci_high']:.4f}]  "
              f"(n_win={micro[m]['n_windows']})")

    # ---------- CSV dump ----------
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("year,n_win,n_fire," +
                ",".join(HEADLINE_METRICS) + "\n")
        for r in rows:
            f.write(f"{r['year']},{r['n_win']},{r['n_fire']}," +
                    ",".join(f"{r[m]:.6f}" for m in HEADLINE_METRICS) + "\n")
        f.write("\nmacro_loyo_summary\n")
        f.write("metric,mean,std,min,max,n_folds\n")
        for m in HEADLINE_METRICS:
            if m in macro:
                v = macro[m]
                f.write(f"{HEADLINE_LABEL[m]},{v['mean']:.6f},{v['std']:.6f},"
                        f"{v['min']:.6f},{v['max']:.6f},{v['n_folds']}\n")
        f.write("\nmicro_loyo_summary\n")
        f.write("metric,mean,ci_low,ci_high,n_windows\n")
        for m in HEADLINE_METRICS:
            if m in micro:
                v = micro[m]
                f.write(f"{HEADLINE_LABEL[m]},{v['mean']:.6f},"
                        f"{v['ci_low']:.6f},{v['ci_high']:.6f},"
                        f"{v['n_windows']}\n")
    print(f"\nCSV written: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
