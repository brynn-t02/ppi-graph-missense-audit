from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch



ROOT = Path(__file__).resolve().parents[1] / "figure_inputs"
OUT_DIR = Path(__file__).resolve().parents[1] / "figure_drafts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.0,
    "figure.dpi": 160, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

COL_ESM = "#9ECAE1"
COL_GCN = "#3182BD"
COL_EDGE = "#FDD0A2"
COL_PERM = "#BDBDBD"
COL_RANDOM = "#636363"
COL_HIGH = "#DE2D26"
COL_LOW = "#31A354"
COL_PURPLE = "#756BB1"
COL_PURPLE_LIGHT = "#BCBDDC"
COL_ORANGE = "#FDAE6B"


PROJECT = Path(__file__).resolve().parents[1]

MATCHED = Path(os.environ.get("CANONICAL_ANALYSIS_ROOT", PROJECT / "canonical_analysis_20261025"))
if not MATCHED.is_dir():
    raise FileNotFoundError(f"Canonical analysis directory required: {MATCHED}")
import pandas as pd
newruns = pd.read_csv(MATCHED / "per_run_metrics.csv")
newboot = json.loads((MATCHED / "cluster_bootstrap.json").read_text())
def newrow(split,arm):
    if arm.endswith("permuted_FPP"):
        ss = newruns[(newruns["split"]==split) & newruns["arm"].str.startswith(arm+"_g")]
    else:
        ss = newruns[(newruns["split"]==split) & (newruns["arm"]==arm)]
    assert len(ss) in (3,9), (split,arm,len(ss))
    return {"test_auc_mean":ss.test_auc.mean(), "test_auc_sd":ss.test_auc.std(ddof=1),
            "test_auprc_mean":ss.test_auprc.mean(), "test_auprc_sd":ss.test_auprc.std(ddof=1)}

rraw=newrow("original","raw_ESM");rfull=newrow("original","full_FPP");redge=newrow("original","edge_ablated_FPP");rperm=newrow("original","permuted_FPP")
ESM_AUC=rraw["test_auc_mean"];ESM_AUPRC=rraw["test_auprc_mean"]
GCN_AUC=rfull["test_auc_mean"];GCN_AUPRC=rfull["test_auprc_mean"]
EDGE_AUC_MEAN=redge["test_auc_mean"];EDGE_AUC_SD=redge["test_auc_sd"];EDGE_AUPRC_MEAN=redge["test_auprc_mean"];EDGE_AUPRC_SD=redge["test_auprc_sd"]
PERM_AUC_MEAN=rperm["test_auc_mean"];PERM_AUC_SD=rperm["test_auc_sd"];PERM_AUPRC_MEAN=rperm["test_auprc_mean"];PERM_AUPRC_SD=rperm["test_auprc_sd"]
bb=newboot["comparisons"]["original"]["full_FPP_vs_raw_ESM"];vb=newboot["variant_bootstrap_original_full_vs_raw"]
PROT_AUC=(bb["delta_auc"],*bb["auc_ci95"]);PROT_AUPRC=(bb["delta_auprc"],*bb["auprc_ci95"])
VAR_AUC=(bb["delta_auc"],*vb["auc_ci95"]);VAR_AUPRC=(bb["delta_auprc"],*vb["auprc_ci95"])

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_a(ax):
    ax.set_title("A  Fixed-propagation audit", loc="left", fontweight="bold", pad=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    nodes = [(0.115, "ESM protein\nfeatures", COL_ESM),
             (0.375, "Fixed STRING\npropagation", "#d9e8f3"),
             (0.635, "Trained MLP\nclassifier", "#e8edf2"),
             (0.895, "Controls, splits\nand uncertainty", "#f1f4f7")]
    for x, label, color in nodes:
        ax.text(x, .42, label, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.40", fc=color, ec="#65768a", lw=.7))
    for x in [.23, .49, .75]:
        ax.add_patch(FancyArrowPatch((x, .42), (x+.03, .42), arrowstyle="->", mutation_scale=10, color="#52657a"))


def panel_b(ax):
    ax.set_title("B  Matched means (3 seeds)", loc="left", fontweight="bold")
    metrics = ["AUC", "AUPRC"]
    esm = [ESM_AUC, ESM_AUPRC]
    gcn = [GCN_AUC, GCN_AUPRC]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, esm, w, yerr=[rraw["test_auc_sd"],rraw["test_auprc_sd"]], capsize=2, color=COL_ESM, edgecolor="black", linewidth=0.45, label="Raw ESM")
    ax.bar(x + w / 2, gcn, w, yerr=[rfull["test_auc_sd"],rfull["test_auprc_sd"]], capsize=2, color=COL_GCN, edgecolor="black", linewidth=0.45, label="Full FPP")
    ax.text(0, 0.810, f"Delta={GCN_AUC-ESM_AUC:+.4f}", ha="center", va="bottom", fontsize=7.4)
    ax.text(1, 0.664, f"Delta={GCN_AUPRC-ESM_AUPRC:+.4f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.55, 0.90)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)


def panel_c(ax):
    ax.set_title("C  Bootstrap uncertainty", loc="left", fontweight="bold")
    rows = [
        ("AUC\nvariant", VAR_AUC, COL_PURPLE),
        ("AUC\nprotein", PROT_AUC, COL_PURPLE_LIGHT),
        ("AUPRC\nvariant", VAR_AUPRC, COL_PURPLE),
        ("AUPRC\nprotein", PROT_AUPRC, COL_PURPLE_LIGHT),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (label, (mean, lo, hi), color) in zip(y, rows):
        ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color,
                    ecolor=color, elinewidth=1.7, capsize=3, markersize=4.5)
    ax.axvline(0, color="#4D4D4D", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Delta performance")
    ax.set_xlim(-0.035, 0.095)
    ax.set_ylim(-0.6, 3.6)
    clean_axis(ax)


def grouped_metric_panel(ax, title, groups, legend_loc="upper center"):
    ax.set_title(title, loc="left", fontweight="bold")
    metrics = ["AUC", "AUPRC"]
    centers = np.arange(2)
    w = 0.24
    offsets = np.linspace(-w, w, len(groups))
    for off, (name, vals, errs, color) in zip(offsets, groups):
        ax.bar(centers + off, vals, width=w, yerr=errs, color=color, edgecolor="black",
               linewidth=0.45, capsize=2.2, label=name)
    ax.set_xticks(centers)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.56, 0.91)
    if legend_loc in ("inside", "upper right"):
        ax.legend(frameon=True, facecolor="white", edgecolor="white", framealpha=0.95, loc="upper right", fontsize=7.0)
    elif legend_loc == "above":
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7.4)
    else:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    clean_axis(ax)


def panel_d(ax):
    grouped_metric_panel(
        ax,
        "D  Direct-edge ablation",
        [
            ("Raw ESM", [ESM_AUC, ESM_AUPRC], [rraw["test_auc_sd"],rraw["test_auprc_sd"]], COL_ESM),
            ("Full FPP", [GCN_AUC, GCN_AUPRC], [rfull["test_auc_sd"],rfull["test_auprc_sd"]], COL_GCN),
            ("Ablated FPP", [EDGE_AUC_MEAN, EDGE_AUPRC_MEAN], [EDGE_AUC_SD, EDGE_AUPRC_SD], COL_EDGE),
        ],
        legend_loc="upper right",
    )


def panel_e(ax):
    grouped_metric_panel(
        ax,
        "E  Topology permutation",
        [
            ("Raw ESM", [ESM_AUC, ESM_AUPRC], [rraw["test_auc_sd"],rraw["test_auprc_sd"]], COL_ESM),
            ("Real FPP", [GCN_AUC, GCN_AUPRC], [rfull["test_auc_sd"],rfull["test_auprc_sd"]], COL_GCN),
            ("Permuted FPP", [PERM_AUC_MEAN, PERM_AUPRC_MEAN], [PERM_AUC_SD, PERM_AUPRC_SD], COL_PERM),
        ],
        legend_loc="upper right",
    )


def main():
    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[.52, 1.1, 1.1], wspace=.40, hspace=.67)
    panel_a(fig.add_subplot(gs[0, :]))
    panel_b(fig.add_subplot(gs[1, 0]))
    panel_c(fig.add_subplot(gs[1, 1]))
    panel_d(fig.add_subplot(gs[2, 0]))
    panel_e(fig.add_subplot(gs[2, 1]))
    fig.subplots_adjust(left=.08, right=.985, top=.96, bottom=.10)
    for suffix in ["png", "pdf"]:
        path = OUT_DIR / f"Figure1_fixed_propagation_audit.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(path)


if __name__ == "__main__":
    main()
