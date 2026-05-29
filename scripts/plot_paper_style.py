"""Shared style + color palette for paper figures.

Import:
    from scripts.plot_paper_style import COLORS, apply_style
"""

from __future__ import annotations

import matplotlib as mpl


# Canonical color palette (used consistently across Fig 1, Fig 2, Fig 3).
COLORS: dict[str, str] = {
    # Ours
    "ensemble_prob":  "#6A1B9A",   # deep purple, prob-mean
    "ensemble_logit": "#9C27B0",   # lighter purple, logit-mean
    "sota_single":    "#C0392B",   # deep red, single best ckpt
    # Deep learning baselines
    "convlstm":       "#8B5A2B",   # brown
    "mlp":            "#7B6F9E",   # muted indigo
    # Classical ML
    "logreg":         "#D17EAA",   # rose
    # Physical / statistical baselines
    "climatology":    "#7F8C8D",   # gray, dashed
    "persistence":    "#3498DB",   # blue, dashed
    "fwi_threshold":  "#F4B400",   # gold, dashed
    "fwi_oracle":     "#E67E22",   # orange, dashed
    # Operational system
    "ecmwf_s2s":      "#27AE60",   # green
}


# Human-readable display labels.
LABELS: dict[str, str] = {
    "ensemble_prob":  "Ensemble (prob-mean, 10 ckpts)",
    "ensemble_logit": "Ensemble (logit-mean, 10 ckpts)",
    "sota_single":    "Patch Transformer (single ckpt)",
    "convlstm":       "ConvLSTM",
    "mlp":            "MLP",
    "logreg":         "Logistic regression",
    "climatology":    "Climatology (1981-2022)",
    "persistence":    "Persistence",
    "fwi_threshold":  "FWI > 30 threshold",
    "fwi_oracle":     "FWI oracle",
    "ecmwf_s2s":      "ECMWF S2S (operational)",
}


def apply_style() -> None:
    """AAAI-compatible serif font + tight grid + vector PDF."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,    # editable text in vector PDF
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })
