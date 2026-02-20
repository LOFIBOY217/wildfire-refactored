"""
Multi-Model Comparison
======================
Reads all_results.csv from multiple model evaluation directories and
produces a unified comparison table, AUC-by-lead plot, and summary text.

Usage:
    python -m src.evaluation.compare_models \\
        --models logistic:outputs/evaluation_confusion_matrix/logreg_fire_prob_7day_forecast \\
                 posaware:outputs/evaluation_confusion_matrix/transformer7d_fire_prob_posaware \\
        --output_dir outputs/model_comparison

Each --models entry is <display_name>:<path_to_evaluation_dir>.
The evaluation dir must contain all_results.csv (produced by evaluate_forecast.py).

Outputs:
    comparison_table.csv  — mean AUC / CSI / POD / Brier per model × lead_time
    auc_by_lead.png       — line chart: AUC vs lead time, one line per model
    summary.txt           — plain-text summary suitable for copy-paste into a report
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ------------------------------------------------------------------ #
# Colours for up to 8 models
# ------------------------------------------------------------------ #
_PALETTE = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]


def _load_model(result_dir: str) -> pd.DataFrame:
    """Load and deduplicate all_results.csv from a model's evaluation dir.

    result_dir can be:
    1. A direct path to an all_results.csv file
    2. A directory that directly contains all_results.csv
    3. A parent directory — will auto-search one level of subdirectories
    """
    # Case 1: user passed the csv file directly
    if result_dir.endswith(".csv") and os.path.isfile(result_dir):
        csv_path = result_dir
    else:
        csv_path = os.path.join(result_dir, "all_results.csv")
        # Case 3: not found directly → search one level of subdirectories
        if not os.path.exists(csv_path):
            found = []
            if os.path.isdir(result_dir):
                for sub in sorted(os.listdir(result_dir)):
                    candidate = os.path.join(result_dir, sub, "all_results.csv")
                    if os.path.isfile(candidate):
                        found.append(candidate)
            if len(found) == 1:
                print(f"  [auto-found] {found[0]}")
                csv_path = found[0]
            elif len(found) > 1:
                raise FileNotFoundError(
                    f"Multiple all_results.csv found under {result_dir}:\n"
                    + "\n".join(f"  {p}" for p in found)
                    + "\nPlease specify the exact subdirectory."
                )
            else:
                contents = ""
                if os.path.isdir(result_dir):
                    entries = os.listdir(result_dir)
                    contents = "\nDirectory contents:\n" + "\n".join(
                        f"  {e}" for e in sorted(entries)
                    )
                raise FileNotFoundError(
                    f"all_results.csv not found in: {result_dir}{contents}\n"
                    f"Run evaluate_forecast.py first, or pass the correct path."
                )

    df = pd.read_csv(csv_path)
    # Each (base_date, lead_time) row is duplicated across thresholds for AUC/Brier
    # (those metrics don't depend on threshold). Deduplicate to get one row per date×lead.
    unique = df.drop_duplicates(subset=["base_date", "lead_time"]).copy()
    return df, unique


def _best_threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (base_date, lead_time), pick the threshold that maximises CSI.
    Returns one row per date×lead with best-threshold POD, FAR, CSI, F1.
    """
    idx = df.groupby(["base_date", "lead_time"])["csi"].idxmax()
    return df.loc[idx].copy()


def _by_lead(unique: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by lead_time."""
    auc_lead   = unique.groupby("lead_time")["auc"].mean()
    brier_lead = unique.groupby("lead_time")["brier"].mean()
    pod_lead   = best.groupby("lead_time")["pod"].mean()
    far_lead   = best.groupby("lead_time")["far"].mean()
    csi_lead   = best.groupby("lead_time")["csi"].mean()

    result = pd.DataFrame({
        "auc":   auc_lead,
        "brier": brier_lead,
        "pod":   pod_lead,
        "far":   far_lead,
        "csi":   csi_lead,
    })
    result.index.name = "lead_time"
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Compare multiple wildfire forecast models"
    )
    ap.add_argument(
        "--models", nargs="+", required=True,
        metavar="NAME:DIR",
        help="One or more <display_name>:<evaluation_dir> pairs. "
             "The dir must contain all_results.csv."
    )
    ap.add_argument(
        "--output_dir", type=str, default="outputs/model_comparison",
        help="Directory to write comparison outputs."
    )
    args = ap.parse_args()

    # Parse model entries
    models = {}
    for entry in args.models:
        if ":" not in entry:
            ap.error(f"Invalid --models entry '{entry}'. Expected <name>:<dir>.")
        name, path = entry.split(":", 1)
        models[name] = path

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("WILDFIRE FORECAST MODEL COMPARISON")
    print("=" * 70)
    for name, path in models.items():
        print(f"  {name:20s}: {path}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Load all models
    # ----------------------------------------------------------------
    model_data  = {}   # name -> (df_all, df_unique, df_best_thresh)
    model_leads = {}   # name -> DataFrame indexed by lead_time

    for name, path in models.items():
        print(f"\nLoading {name} from {path} ...")
        df_all, df_unique = _load_model(path)
        df_best = _best_threshold_metrics(df_all)
        by_lead = _by_lead(df_unique, df_best)
        model_data[name]  = (df_all, df_unique, df_best)
        model_leads[name] = by_lead

        n_dates = df_unique["base_date"].nunique()
        mean_auc = df_unique["auc"].mean()
        best_thresh = df_best.groupby("lead_time").apply(
            lambda g: g.name  # placeholder
        )
        print(f"  Dates: {n_dates}   Mean AUC (all leads): {mean_auc:.4f}")

    # ----------------------------------------------------------------
    # Build comparison table
    # ----------------------------------------------------------------
    rows = []
    for name, by_lead in model_leads.items():
        for lead, row in by_lead.iterrows():
            rows.append({
                "model":     name,
                "lead_time": lead,
                "auc":       round(row["auc"], 4),
                "brier":     round(row["brier"], 4),
                "pod":       round(row["pod"], 4),
                "far":       round(row["far"], 4),
                "csi":       round(row["csi"], 4),
            })
    comp_df = pd.DataFrame(rows)
    table_path = os.path.join(args.output_dir, "comparison_table.csv")
    comp_df.to_csv(table_path, index=False)
    print(f"\nSaved: {table_path}")

    # ----------------------------------------------------------------
    # AUC by lead plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = _PALETTE[:len(models)]

    for (name, by_lead), colour in zip(model_leads.items(), colours):
        leads = by_lead.index.tolist()
        aucs  = by_lead["auc"].tolist()
        ax.plot(leads, aucs, marker="o", label=name, color=colour, linewidth=2)

    ax.set_xlabel("Lead Time (days)", fontsize=12)
    ax.set_ylabel("Mean AUC-ROC", fontsize=12)
    ax.set_title("Model AUC by Forecast Lead Time", fontsize=13)
    ax.set_xticks(sorted({lt for by_lead in model_leads.values()
                           for lt in by_lead.index}))
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="random (0.5)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    auc_plot_path = os.path.join(args.output_dir, "auc_by_lead.png")
    plt.savefig(auc_plot_path, dpi=150)
    plt.close()
    print(f"Saved: {auc_plot_path}")

    # ----------------------------------------------------------------
    # Summary text
    # ----------------------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("WILDFIRE FORECAST MODEL COMPARISON — SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Overall AUC table
    lines.append("Mean AUC by Lead Time")
    lines.append("-" * 50)
    header = f"{'Lead':>5} | " + " | ".join(f"{n:>12}" for n in models)
    lines.append(header)
    lines.append("-" * len(header))

    all_leads = sorted({lt for by_lead in model_leads.values() for lt in by_lead.index})
    for lead in all_leads:
        row_vals = []
        for name in models:
            by_lead = model_leads[name]
            val = by_lead.loc[lead, "auc"] if lead in by_lead.index else float("nan")
            row_vals.append(f"{val:>12.4f}")
        lines.append(f"{lead:>5} | " + " | ".join(row_vals))

    lines.append("-" * len(header))
    # Overall mean
    row_vals = []
    for name in models:
        by_lead = model_leads[name]
        row_vals.append(f"{by_lead['auc'].mean():>12.4f}")
    lines.append(f"{'Mean':>5} | " + " | ".join(row_vals))
    lines.append("")

    # CSI table (at best threshold)
    lines.append("Mean CSI by Lead Time (at best-CSI threshold per date)")
    lines.append("-" * 50)
    lines.append(header)
    lines.append("-" * len(header))
    for lead in all_leads:
        row_vals = []
        for name in models:
            by_lead = model_leads[name]
            val = by_lead.loc[lead, "csi"] if lead in by_lead.index else float("nan")
            row_vals.append(f"{val:>12.4f}")
        lines.append(f"{lead:>5} | " + " | ".join(row_vals))
    lines.append("")

    # Brier score
    lines.append("Mean Brier Score by Lead Time (lower = better)")
    lines.append("-" * 50)
    lines.append(header)
    lines.append("-" * len(header))
    for lead in all_leads:
        row_vals = []
        for name in models:
            by_lead = model_leads[name]
            val = by_lead.loc[lead, "brier"] if lead in by_lead.index else float("nan")
            row_vals.append(f"{val:>12.4f}")
        lines.append(f"{lead:>5} | " + " | ".join(row_vals))
    lines.append("")

    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved: {summary_path}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print(f"  {args.output_dir}/")
    print(f"    comparison_table.csv")
    print(f"    auc_by_lead.png")
    print(f"    summary.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
