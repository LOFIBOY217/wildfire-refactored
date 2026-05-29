"""Figure 1: Lift@5000 vs forecast lead day.

SOTA model: median + 95% percentile band across val windows, per lead day.
Baselines (stateless): per-lead curve from benchmark_baselines_per_leadday.csv.
Single-number deep-learning baselines: drawn as a horizontal line at their
aggregate Lift across the full 14-46d window.

Usage:
    python3 scripts/plot_fig1_per_lead.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

LEAD_OFFSET = 14   # per_lead index 0 → absolute lead day 14
K = 5000


# ---------- loaders ---------------------------------------------------------

def sota_per_lead_curve(json_path: str, n_boot: int = 2000, seed: int = 0):
    """Return (lead_days, mean, ci_lo, ci_hi) — bootstrap CI of the mean
    per lead day. Percentile CI is too heavy-tailed for clean plotting
    because many windows have zero fire (Lift=0)."""
    d = json.load(open(json_path))
    by_lead: dict[int, list[float]] = defaultdict(list)
    for w in d["per_window"]:
        for e in w["per_lead"]:
            if e["lift_k"] is None:
                continue
            by_lead[e["lead"]].append(e["lift_k"])
    leads = sorted(by_lead.keys())
    rng = np.random.default_rng(seed)
    mean, lo, hi = [], [], []
    for L in leads:
        arr = np.asarray(by_lead[L], dtype=np.float64)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        boot_means = arr[idx].mean(axis=1)
        mean.append(arr.mean())
        lo.append(np.percentile(boot_means, 2.5))
        hi.append(np.percentile(boot_means, 97.5))
    leads_abs = np.array([L + LEAD_OFFSET for L in leads])
    return leads_abs, np.array(mean), np.array(lo), np.array(hi)


def baseline_per_lead_curve(csv_path: str, baseline_name: str, k: int = K):
    """Return (lead_days, lift_k) from per-leadday CSV for one baseline."""
    df = pd.read_csv(csv_path)
    sub = df[(df["baseline"] == baseline_name) & (df["k"] == k)].sort_values("lead_day")
    if sub.empty:
        return None, None
    return sub["lead_day"].to_numpy(), sub["lift_k"].to_numpy()


def aggregate_lift(json_path: str, key: str = "lift_5000"):
    """Return (mean, ci_lo, ci_hi) from ensemble/ECMWF-style summary JSON."""
    d = json.load(open(json_path))
    v = d[key]
    return v["mean"], v["ci_lo"], v["ci_hi"]


def baseline_summary(json_path: str):
    """ConvLSTM / MLP: return aggregate lift_k mean (+ std as fallback CI)."""
    d = json.load(open(json_path))
    s = d["summary"]
    n = d.get("n_windows_with_fire", d.get("n_sample_wins", 1))
    mean = s["lift_k"]
    se = s["lift_k_std"] / max(np.sqrt(n), 1.0)
    return mean, mean - 1.96 * se, mean + 1.96 * se


# ---------- plot ------------------------------------------------------------

def main():
    apply_style()

    sota_l, sota_mean, sota_lo, sota_hi = sota_per_lead_curve(
        os.path.join(EVAL_DIR, "sota_per_lead.json"))

    csv_path = os.path.join(EVAL_DIR, "benchmark_baselines_per_leadday.csv")
    clim_l, clim_y = baseline_per_lead_curve(csv_path, "climatology")
    fwi_l,  fwi_y  = baseline_per_lead_curve(csv_path, "fwi_oracle")

    ens_prob_mean,  _, _ = aggregate_lift(os.path.join(EVAL_DIR, "ensemble_prob.json"))
    ens_logit_mean, _, _ = aggregate_lift(os.path.join(EVAL_DIR, "ensemble_logit.json"))
    ecmwf_mean,     _, _ = aggregate_lift(os.path.join(EVAL_DIR, "baseline_ecmwf_s2s.json"))
    cl_mean, *_ = baseline_summary(os.path.join(EVAL_DIR, "baseline_convlstm.json"))
    ml_mean, *_ = baseline_summary(os.path.join(EVAL_DIR, "baseline_mlp.json"))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # --- model curves with per-lead resolution ---
    ax.fill_between(sota_l, sota_lo, sota_hi,
                    color=COLORS["sota_single"], alpha=0.22, linewidth=0,
                    label="Patch Transformer (95% boot CI)")
    ax.plot(sota_l, sota_mean, "-", color=COLORS["sota_single"], linewidth=2.4,
            label=LABELS["sota_single"] + " (mean)")

    # --- baselines with per-lead curve from CSV ---
    if clim_y is not None:
        ax.plot(clim_l, clim_y, "--", color=COLORS["climatology"], linewidth=1.4,
                label=LABELS["climatology"])
    if fwi_y is not None:
        ax.plot(fwi_l, fwi_y, ":", color=COLORS["fwi_oracle"], linewidth=1.4,
                label=LABELS["fwi_oracle"])

    # --- single-number baselines as horizontal reference lines ---
    span_xs = (LEAD_OFFSET, LEAD_OFFSET + 32)
    def hline(y, color, label, ls="-"):
        ax.hlines(y, *span_xs, colors=color, linestyles=ls, linewidth=1.4,
                  label=f"{label} (avg)")
    hline(ens_prob_mean,  COLORS["ensemble_prob"],  LABELS["ensemble_prob"])
    hline(ens_logit_mean, COLORS["ensemble_logit"], LABELS["ensemble_logit"], ls=(0, (4, 2)))
    hline(cl_mean,        COLORS["convlstm"],       LABELS["convlstm"],       ls=(0, (1, 1)))
    hline(ml_mean,        COLORS["mlp"],            LABELS["mlp"],            ls=(0, (1, 1)))
    hline(ecmwf_mean,     COLORS["ecmwf_s2s"],      LABELS["ecmwf_s2s"],      ls=(0, (3, 1, 1, 1)))

    ax.axhline(1.0, color="#999999", linewidth=0.7, linestyle=":", zorder=0)

    ax.set_xlabel("Forecast lead day (t + L)")
    ax.set_ylabel(f"Lift@{K}")
    ax.set_title(f"Per-lead-day Lift@{K} — 2022-2025 validation, NBAC+NFDB labels")
    ax.set_xlim(LEAD_OFFSET - 0.5, LEAD_OFFSET + 32 + 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", ncols=1, frameon=True, fontsize=7.3)

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "fig1_per_lead_lift.pdf")
    png = os.path.join(OUT_DIR, "fig1_per_lead_lift.png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")

    # --- summary table for paper text ---
    print()
    print(f"{'lead':>6} {'SOTA(mean)':>12} {'boot p2.5':>11} {'boot p97.5':>12}")
    for L, mk, lo, hi in zip(sota_l, sota_mean, sota_lo, sota_hi):
        print(f"{L:>6d} {mk:>12.3f} {lo:>11.3f} {hi:>12.3f}")


if __name__ == "__main__":
    main()
