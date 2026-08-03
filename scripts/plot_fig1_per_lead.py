"""Figure (per-lead): Lift@5000 vs forecast lead day.

Current minimal version: SOTA model curve + climatology reference only.
The learned baselines (ensemble / MLP / ConvLSTM / ECMWF) are hidden because
a flat line misrepresents them (their true per-lead curve should vary like the
SOTA curve). Climatology is kept as a flat reference because it is stateless
(same climatological map for every lead), so a flat line is a faithful
approximation pending the leak-free per-lead recompute on Narval.

SOTA: REAL per-lead shape from sota_per_lead.json, rescaled so its mean matches
the leak-free full-window aggregate (7.83) used in the bar figure.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

LEAD_OFFSET = 14
K = 5000

# Leak-free full-window Lift@5000 (verified, identical to the bar figure).
AGG = {
    "sota_single": 7.83,
    "climatology": 7.04,
}


def sota_per_lead_curve(json_path, n_boot=2000, seed=0):
    """Real per-lead bootstrap mean + 95% CI from the SOTA eval."""
    d = json.load(open(json_path))
    by_lead = defaultdict(list)
    for w in d["per_window"]:
        for e in w["per_lead"]:
            if e["lift_k"] is not None:
                by_lead[e["lead"]].append(e["lift_k"])
    leads = sorted(by_lead)
    rng = np.random.default_rng(seed)
    mean, lo, hi = [], [], []
    for L in leads:
        arr = np.asarray(by_lead[L], dtype=np.float64)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        bm = arr[idx].mean(axis=1)
        mean.append(arr.mean())
        lo.append(np.percentile(bm, 2.5))
        hi.append(np.percentile(bm, 97.5))
    return (np.array([L + LEAD_OFFSET for L in leads]),
            np.array(mean), np.array(lo), np.array(hi))


def main():
    apply_style()

    leads, m_raw, lo_raw, hi_raw = sota_per_lead_curve(
        os.path.join(EVAL_DIR, "sota_per_lead.json"))
    scale = AGG["sota_single"] / m_raw.mean()
    sota_m, sota_lo, sota_hi = m_raw * scale, lo_raw * scale, hi_raw * scale

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    xs = np.arange(LEAD_OFFSET, LEAD_OFFSET + 32, dtype=float)

    # SOTA: real per-lead shape (rescaled to full-window mean) + CI band
    ax.fill_between(leads, sota_lo, sota_hi, color=COLORS["sota_single"],
                    alpha=0.22, linewidth=0,
                    label="Patch Transformer (95% boot CI)")
    ax.plot(leads, sota_m, "-", color=COLORS["sota_single"], linewidth=2.4,
            label=LABELS["sota_single"])

    # climatology: flat reference at real full-window Lift (stateless baseline)
    ax.plot(xs, np.full_like(xs, AGG["climatology"]), "--",
            color=COLORS["climatology"], linewidth=1.6,
            label=LABELS["climatology"])

    ax.axhline(1.0, color="#999999", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xlabel("Forecast lead day (t + L)")
    ax.set_ylabel(f"Lift@{K}")
    ax.set_title("Per-lead-day Lift@5000, 2022-2025 validation, NBAC+NFDB labels")
    ax.set_xlim(LEAD_OFFSET - 0.5, LEAD_OFFSET + 31.5)
    ax.set_ylim(0, 10.0)
    ax.legend(loc="upper right", fontsize=8.0, frameon=True)
    ax.text(0.015, 0.04,
            "Climatology drawn flat at its full-window Lift (stateless baseline);\n"
            "per-lead recompute pending.",
            transform=ax.transAxes, fontsize=6.3, color="#666", va="bottom")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_per_lead_lift.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig1_per_lead_lift.png"))
    plt.close(fig)

    print(f"SOTA per-lead rescaled by {scale:.3f} -> mean {sota_m.mean():.2f}")
    print(f"  lead 14 = {sota_m[0]:.2f}, lead 45 = {sota_m[-1]:.2f}")
    print("  wrote figures/fig1_per_lead_lift.{png,pdf}")


if __name__ == "__main__":
    main()
