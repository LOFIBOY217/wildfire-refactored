"""
Visualization Helpers
=====================
Plotting functions for wildfire forecast evaluation.

Based on evaluate_with_confusion_matrix.py plotting sections.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics_vs_threshold(threshold_summary, output_path):
    """
    Plot 4-panel metrics vs threshold chart.

    Args:
        threshold_summary: DataFrame indexed by threshold with columns: pod, far, csi, bias
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    threshold_summary['pod'].plot(ax=axes[0, 0], marker='o', color='blue', label='POD')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('POD (Hit Rate)')
    axes[0, 0].set_title('Probability of Detection')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    threshold_summary['far'].plot(ax=axes[0, 1], marker='o', color='red', label='FAR')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('FAR')
    axes[0, 1].set_title('False Alarm Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    threshold_summary['csi'].plot(ax=axes[1, 0], marker='o', color='green', label='CSI')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('CSI')
    axes[1, 0].set_title('Critical Success Index')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    threshold_summary['bias'].plot(ax=axes[1, 1], marker='o', color='purple', label='Bias')
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Bias Score')
    axes[1, 1].set_title('Bias Score (1.0 = perfect)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_performance_by_lead(lead_summary, output_path):
    """
    Plot performance degradation by lead time.

    Args:
        lead_summary: DataFrame indexed by lead_time with columns: pod, csi, brier
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    lead_summary['pod'].plot(ax=axes[0], marker='o', color='blue')
    axes[0].set_xlabel('Lead Time (days)')
    axes[0].set_ylabel('POD')
    axes[0].set_title('Hit Rate by Lead Time')
    axes[0].grid(True, alpha=0.3)

    lead_summary['csi'].plot(ax=axes[1], marker='o', color='green')
    axes[1].set_xlabel('Lead Time (days)')
    axes[1].set_ylabel('CSI')
    axes[1].set_title('Critical Success Index by Lead Time')
    axes[1].grid(True, alpha=0.3)

    lead_summary['brier'].plot(ax=axes[2], marker='o', color='red')
    axes[2].set_xlabel('Lead Time (days)')
    axes[2].set_ylabel('Brier Score')
    axes[2].set_title('Brier Score by Lead Time (lower=better)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_confusion_matrix_heatmap(tp, fp, tn, fn, threshold, output_path):
    """
    Plot confusion matrix as heatmap.

    Args:
        tp, fp, tn, fn: Confusion matrix values
        threshold: Threshold used
        output_path: Path to save the figure
    """
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Pred No Fire', 'Pred Fire'],
        yticklabels=['Actual No Fire', 'Actual Fire']
    )
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")
