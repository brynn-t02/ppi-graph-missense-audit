"""Historical panels moved from former main Figures 1E and 2D.

Inputs are retained unchanged; these panels are not matched-protocol evidence.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "figure_inputs/data/processed"
OUTPUT = PROJECT / "figure_drafts"
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "legend.fontsize": 7.2,
    "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
curve = pd.read_csv(INPUT / "esm_gcn_partial_edge_ablation_exp44/edge_ablation_curve_summary.csv")
homology = json.loads((INPUT / "homology_rerun_core/homology_rerun_summary.json").read_text())


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85))
    a, b = axes
    a.set_title("A  Historical partial ablation (unmatched)", loc="left", fontweight="bold")
    for strategy, label, color, marker in [
        ("random_pair_removal", "Random removal", "#636363", "o"),
        ("high_confidence_first", "High first", "#DE2D26", "s"),
        ("low_confidence_first", "Low first", "#3182BD", "^"),
    ]:
        rows = curve[curve.strategy == strategy].sort_values("fraction_removed")
        a.errorbar(rows.fraction_removed * 100, rows.delta_test_auc_vs_raw_esm,
                   yerr=rows.test_auc_sd, fmt=marker+"-", lw=1.4,
                   markersize=3.7, capsize=2.2, color=color, label=label)
    a.axhline(0, color="#636363", ls="--", lw=.8)
    a.set(xlim=(-3, 103), ylim=(-.02, .045),
          xlabel="Direct train-test pairs removed (%)", ylabel="Delta AUC vs raw ESM")
    a.set_xticks([0, 25, 50, 75, 100])
    a.legend(frameon=False, loc="upper right")
    b.set_title("B  Historical homology representation", loc="left", fontweight="bold")
    for i, (arm, label, color) in enumerate([
        ("esm_raw_homology", "Raw ESM", "#9ECAE1"),
        ("gcn_enhanced_homology", "FPP", "#3182BD"),
    ]):
        row = homology[arm]
        vals = [float(row["test_auc"]), float(row["test_auprc"])]
        b.bar(np.arange(2) + (i-.5)*.34, vals, width=.34,
              color=color, edgecolor="black", linewidth=.45, label=label)
    b.set_xticks([0, 1], ["AUC", "AUPRC"])
    b.set(ylabel="Test performance", ylim=(.18, .88))
    b.legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=.08, right=.985, top=.87, bottom=.26, wspace=.40)
    fig.text(.285, .045, "A: unmatched seeds/budgets and historical control operator;\n0% and 100% endpoints reused.",
             ha="center", fontsize=6.8, linespacing=1.25)
    fig.text(.77, .045, "B: one retained fit per arm;\nno repeat uncertainty.",
             ha="center", fontsize=6.8, linespacing=1.25)
    for suffix in ["png", "pdf"]:
        path = OUTPUT / f"FigureS3_historical_comparisons.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)


if __name__ == "__main__":
    main()
