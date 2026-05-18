"""Plot per-lead-day Lift@5000 and Lift@30km decay curves.

Combines:
  - Model JSON from `slurm/eval_per_lead_narval.sh`
    (outputs/per_lead/${RUN_NAME}.json)
  - Baseline CSV from `slurm/baselines_all4_full_narval.sh`
    (outputs/baselines_per_leadday.csv)

Output: one PNG per metric (Lift@5000, Lift@30km), showing the model
curve plus the 4 baseline curves. Median across windows + 95 % CI band.

Usage:
  python -m scripts.plot_per_lead_lift \
      --model_json outputs/per_lead/v3_9ch_enc21_12y_2014.json \
      --baselines_csv outputs/baselines_per_leadday.csv \
      --out_dir figures/per_lead

  # No baselines (model-only quick look):
  python -m scripts.plot_per_lead_lift \
      --model_json outputs/per_lead/v3_9ch_enc21_12y_2014.json \
      --out_dir figures/per_lead
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

try:
    import pandas as pd  # only needed for baseline CSV
except ImportError:
    pd = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Lead-day offset 0 means lead_start (14 by default). We label the
# x-axis with the actual lead day t+L.
LEAD_START_DEFAULT = 14


def load_model_per_lead(json_path: str) -> dict:
    """Return {lead_day: [lift_k_per_window], lead_day: [lift_coarse...]} dicts."""
    with open(json_path) as f:
        data = json.load(f)
    by_lead_k = defaultdict(list)
    by_lead_c = defaultdict(list)
    for w in data["per_window"]:
        for entry in w["per_lead"]:
            if entry["lift_k"] is None:
                continue
            by_lead_k[entry["lead"]].append(entry["lift_k"])
            by_lead_c[entry["lead"]].append(entry["lift_coarse"])
    return {"lift_k": by_lead_k, "lift_coarse": by_lead_c, "k": data["k"]}


def summarize(by_lead: dict, lead_offset: int = LEAD_START_DEFAULT):
    """Return (lead_days_absolute, median, ci_low, ci_high)."""
    leads = sorted(by_lead.keys())
    med, lo, hi = [], [], []
    for L in leads:
        vals = np.asarray(by_lead[L], dtype=np.float32)
        if len(vals) == 0:
            med.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
            continue
        med.append(np.median(vals))
        # Percentile CI is more robust than bootstrap mean here because
        # per-lead Lift distributions are heavy-tailed.
        lo.append(np.percentile(vals, 2.5))
        hi.append(np.percentile(vals, 97.5))
    leads_abs = [L + lead_offset for L in leads]
    return np.asarray(leads_abs), np.asarray(med), np.asarray(lo), np.asarray(hi)


def load_baselines_per_lead(csv_path: str):
    """Return {baseline_name: {lead_day: [lift_k, lift_coarse]}}.

    The eval_per_leadday CSV layout is one row per (baseline, lead)
    with summary columns. We re-shape to dict-of-dict for plotting.
    """
    if pd is None:
        raise RuntimeError("pandas required for baseline CSV.")
    df = pd.read_csv(csv_path)
    # Expect columns: baseline, lead_day, lift_k_mean, lift_coarse_mean,
    # lift_k_std, lift_coarse_std (best-effort — script tolerates aliasing).
    cols = {c.lower(): c for c in df.columns}
    name_col   = cols.get("baseline", "baseline")
    lead_col   = cols.get("lead_day", "lead_day")
    lk_mean    = cols.get("lift_k_mean", cols.get("lift_k", "lift_k"))
    lc_mean    = cols.get("lift_coarse_mean", cols.get("lift_coarse", "lift_coarse"))
    out = {}
    for b, g in df.groupby(name_col):
        g = g.sort_values(lead_col)
        out[b] = {
            "lead":        g[lead_col].astype(int).values,
            "lift_k":      g[lk_mean].astype(float).values,
            "lift_coarse": g[lc_mean].astype(float).values,
        }
    return out


def plot_metric(model_summary, baselines, metric_key: str,
                out_path: str, title: str):
    leads_abs, med, lo, hi = model_summary
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Model curve (median + CI band)
    ax.plot(leads_abs, med, "-", color="#dc2626",
            linewidth=2.2, label="Patch Transformer (median)")
    ax.fill_between(leads_abs, lo, hi, color="#dc2626", alpha=0.15,
                    label="95 % percentile band")
    # Baseline flat curves
    colors = {"climatology": "#2563eb", "persistence": "#10b981",
              "fwi_threshold": "#f59e0b", "fwi_oracle": "#7c3aed"}
    if baselines is not None:
        for name, d in baselines.items():
            ax.plot(d["lead"], d[metric_key], "--",
                    color=colors.get(name, "#666666"),
                    linewidth=1.5, label=f"{name}")
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Forecast lead day (t + L)")
    ax.set_ylabel(title)
    ax.set_title(f"{title} vs lead day")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_json", required=True)
    ap.add_argument("--baselines_csv", default=None,
                    help="outputs/baselines_per_leadday.csv (optional)")
    ap.add_argument("--out_dir", default="figures/per_lead")
    ap.add_argument("--lead_offset", type=int, default=LEAD_START_DEFAULT,
                    help="Absolute lead day for lead index 0 (default 14).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model_per_lead(args.model_json)
    baselines = (load_baselines_per_lead(args.baselines_csv)
                 if args.baselines_csv else None)

    # Lift@K (pixel scale)
    summ_k = summarize(model["lift_k"], lead_offset=args.lead_offset)
    plot_metric(
        summ_k, baselines, metric_key="lift_k",
        out_path=os.path.join(args.out_dir, "lift_k_vs_lead.png"),
        title=f"Lift@{model['k']}")

    # Lift@30km (event scale)
    summ_c = summarize(model["lift_coarse"], lead_offset=args.lead_offset)
    plot_metric(
        summ_c, baselines, metric_key="lift_coarse",
        out_path=os.path.join(args.out_dir, "lift_coarse_vs_lead.png"),
        title="Lift@30 km")

    # Print a small table to stdout for the paper text.
    print()
    print(f"{'lead':>6} {'lift_k(med)':>12} {'lift_30km(med)':>14}")
    for L, mk, mc in zip(summ_k[0], summ_k[1], summ_c[1]):
        print(f"{L:>6d} {mk:>12.3f} {mc:>14.3f}")


if __name__ == "__main__":
    main()
