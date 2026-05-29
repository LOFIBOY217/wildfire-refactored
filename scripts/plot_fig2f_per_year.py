"""Figure 2-F: per-year Lift@5000 bars (2022 / 2023 / 2024).

Tests whether SOTA is uniformly good across val years or just lucky
in one season. Each bar = mean Lift@5000 across all per-window samples
in that year, with bootstrap 95% CI.

Usage:
    python3 scripts/plot_fig2f_per_year.py
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

YEARS = [2022, 2023, 2024]


def _boot_ci(arr, n_boot=2000, seed=0):
    """Return (mean, ci_lo, ci_hi) bootstrap CI of the mean."""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sota_per_year(path: str):
    """SOTA per_lead JSON: aggregate per-lead lift into per-window first,
    then group by year."""
    d = json.load(open(path))
    by_y = defaultdict(list)
    for w in d["per_window"]:
        lifts = [e["lift_k"] for e in w["per_lead"] if e["lift_k"] is not None]
        if not lifts:
            continue
        y = int(w["date"][:4])
        by_y[y].append(float(np.mean(lifts)))
    return {y: _boot_ci(by_y[y]) for y in sorted(by_y)}


def ensemble_per_year(path: str):
    """Ensemble JSON: per_window each has win_date + lift_5000 (dict or scalar)."""
    d = json.load(open(path))
    by_y = defaultdict(list)
    for w in d["per_window"]:
        date = w["win_date"]
        y = int(date[:4])
        v = w["lift_5000"]
        if isinstance(v, dict):
            v = v.get("mean")
        if v is None:
            continue
        by_y[y].append(float(v))
    return {y: _boot_ci(by_y[y]) for y in sorted(by_y)}


def ecmwf_per_year(path: str):
    """ECMWF per_window file is a top-level list of dicts."""
    d = json.load(open(path))
    by_y = defaultdict(list)
    for w in d:
        y = int(w["win_date"][:4])
        v = w.get("lift_5000")
        if v is None:
            continue
        by_y[y].append(float(v))
    return {y: _boot_ci(by_y[y]) for y in sorted(by_y)}


def main():
    apply_style()

    rows: dict[str, dict] = {}
    rows["sota_single"]    = sota_per_year(os.path.join(EVAL_DIR, "sota_per_lead.json"))
    rows["ensemble_prob"]  = ensemble_per_year(os.path.join(EVAL_DIR, "ensemble_prob.json"))
    rows["ensemble_logit"] = ensemble_per_year(os.path.join(EVAL_DIR, "ensemble_logit.json"))
    rows["ecmwf_s2s"]      = ecmwf_per_year(os.path.join(EVAL_DIR, "baseline_ecmwf_s2s_per_window.json"))

    # Order in legend
    model_order = ["ensemble_prob", "ensemble_logit", "sota_single", "ecmwf_s2s"]

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n_models = len(model_order)
    n_years = len(YEARS)
    bar_w = 0.78 / n_models
    x = np.arange(n_years)

    for i, m in enumerate(model_order):
        means = np.array([rows[m].get(y, (np.nan, np.nan, np.nan))[0] for y in YEARS])
        errlo = np.array([rows[m].get(y, (0, 0, 0))[0] - rows[m].get(y, (0, 0, 0))[1] for y in YEARS])
        errhi = np.array([rows[m].get(y, (0, 0, 0))[2] - rows[m].get(y, (0, 0, 0))[0] for y in YEARS])
        offset = (i - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means, bar_w, color=COLORS[m],
                      edgecolor="black", linewidth=0.4, label=LABELS[m], alpha=0.92)
        ax.errorbar(x + offset, means, yerr=[errlo, errhi],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2.5, capthick=0.7)
        for xi, v in zip(x + offset, means):
            if np.isfinite(v) and v > 0.3:
                ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\nfire season" for y in YEARS])
    ax.set_ylabel(f"Lift@5000")
    ax.set_title("Per-year Lift@5000  —  NBAC+NFDB labels, 2022-2024 validation")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "fig2f_per_year_lift.pdf")
    png = os.path.join(OUT_DIR, "fig2f_per_year_lift.png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")

    # Print summary table
    print()
    print(f"{'model':<35} " + "  ".join(f"{y:>16}" for y in YEARS))
    for m in model_order:
        cells = []
        for y in YEARS:
            v = rows[m].get(y, (np.nan, np.nan, np.nan))
            cells.append(f"{v[0]:5.2f} [{v[1]:5.2f},{v[2]:5.2f}]")
        print(f"{LABELS[m]:<35} " + "  ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
