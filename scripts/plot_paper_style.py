"""Shared style + color palette for paper figures.

Import:
    from scripts.plot_paper_style import COLORS, apply_style
"""

from __future__ import annotations

import matplotlib as mpl


# ============================================================================
# CANONICAL model -> color mapping. Keep IDENTICAL across every figure in the
# paper (Fig 1/2/3/4, novel bars, scaling, etc.). Never recolor a model in one
# figure only. New models: add a key here, do not hardcode a color in a script.
# Recorded also in memory: model_color_mapping.md
# ============================================================================
COLORS: dict[str, str] = {
    # Ours — single models (each a distinct fixed hue)
    "convstem":       "#C0392B",   # deep red   — conv-stem (single-model SOTA)
    "convstem_novel": "#E74C3C",   # bright red — conv-stem + per-pixel novel loss
    "fcnhead":        "#16A085",   # teal       — conv-stem + FCN output head
    "flatten":        "#F1C40F",   # yellow     — flatten patch-embed Transformer
    "sota_single":    "#C0392B",   # alias of conv-stem red (generic single ckpt)
    # Ours — ensembles (NOT drawn in single-model figures)
    "ensemble_prob":  "#6A1B9A",   # deep purple, prob-mean
    "ensemble_logit": "#9C27B0",   # lighter purple, logit-mean
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
    "convstem":       "Patch Transformer (conv-stem)",
    "convstem_novel": "Patch Transformer (conv-stem + novel loss)",
    "fcnhead":        "Patch Transformer (conv-stem + FCN head)",
    "flatten":        "Patch Transformer (flatten)",
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


# Compact, single-line-where-possible labels for crowded multi-bar / multi-panel
# figures. Keep the SAME wording across every figure so a reader who has seen one
# panel recognizes the models everywhere. Two-line only where unavoidable.
SHORT: dict[str, str] = {
    "convstem":       "conv-stem",
    "convstem_novel": "conv-stem\n+novel loss",
    "fcnhead":        "conv-stem\n+FCN head",
    "flatten":        "flatten",
    "sota_single":    "Patch Transformer",
    "ensemble_prob":  "ensemble",
    "ensemble_logit": "ensemble (logit)",
    "convlstm":       "ConvLSTM",
    "mlp":            "MLP",
    "logreg":         "logreg",
    "climatology":    "Climatology",
    "persistence":    "Persistence",
    "fwi_threshold":  "FWI > 30",
    "fwi_oracle":     "FWI oracle",
    "ecmwf_s2s":      "ECMWF S2S",
}

# Canonical left-to-right order for EVERY bar figure (Fig 4/5/6/8). Same 9-model
# set, same order everywhere — a model missing/reordered in one panel misleads.
# Grouped: ours (4) -> learned baselines (2) -> physical baselines (3).
# ecmwf_s2s / logreg / unet are intentionally NOT here (dropped, 2026-08-08).
BAR_ORDER: list[str] = [
    "fcnhead", "convstem_novel", "convstem", "flatten",   # ours
    "convlstm", "mlp",                                      # learned baselines
    "climatology", "persistence", "fwi_oracle",            # physical baselines
]


# "Ours" = the patch-transformer family (every conv-stem / flatten variant + the
# generic single ckpt). Everything else is a baseline. Used to draw the shaded
# "Ours" band that separates our models from baselines in bar figures.
OURS: frozenset[str] = frozenset({
    "convstem", "convstem_novel", "fcnhead", "flatten", "sota_single",
    "ensemble_prob", "ensemble_logit",
})

OURS_BAND = "#F4C7C3"   # very light red wash behind our-model bars


def shade_ours(ax, keys, *, y0=0.0, y1=1.0, label=True) -> None:
    """Shade the contiguous run of our-model bars (assumes they are grouped at
    the start of `keys`) with a faint band + an "Ours" bracket label. `keys` is
    the left-to-right list of model keys plotted on `ax`. y0/y1 are axis-fraction
    span of the band. No-op if our models are not a leading contiguous block."""
    ours_idx = [i for i, k in enumerate(keys) if k in OURS]
    if not ours_idx:
        return
    lo, hi = min(ours_idx), max(ours_idx)
    # only shade if it is a clean leading block (lo..hi all ours)
    if any(k not in OURS for k in keys[lo:hi + 1]):
        return
    ax.axvspan(lo - 0.5, hi + 0.5, ymin=y0, ymax=y1, color=OURS_BAND,
               alpha=0.35, zorder=0, linewidth=0)
    if label:
        ax.text((lo + hi) / 2.0, 0.985, "Ours", transform=_blend(ax),
                ha="center", va="top", fontsize=8.5, style="italic",
                color="#8B2E28", zorder=1)


def _blend(ax):
    import matplotlib.transforms as mtransforms
    return mtransforms.blended_transform_factory(ax.transData, ax.transAxes)


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
