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

P_44 = ROOT / "data/processed/esm_gcn_partial_edge_ablation_exp44/edge_ablation_curve_summary.csv"

plt.rcParams.update({
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.6,
    "xtick.labelsize": 8.1,
    "ytick.labelsize": 8.1,
    "legend.fontsize": 7.4,
    "figure.dpi": 160,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.0, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.8})

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


# Only panel E uses a historical curve. All other numerical panels below
# read the matched-run records and paired bootstrap estimates.
df44 = pd.read_csv(P_44)

PROJECT = Path(__file__).resolve().parents[1]

MATCHED = PROJECT / "matched_validation_20261024"
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
    ax.set_title("A  Study overview", loc="left", fontweight="bold")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    for y, text, color in [(0.80,"ESM protein features",COL_ESM),
                            (0.57,"Fixed STRING propagation", "#d9e8f3"),
                            (0.34,"Trained MLP classifier", "#e8edf2")]:
        ax.text(0.5,y,text,ha="center",va="center",fontsize=6.6,
                bbox=dict(boxstyle="round,pad=0.42",fc=color,ec="#65768a",lw=0.7))
    for y1,y2 in [(0.72,0.65),(0.49,0.42)]:
        ax.add_patch(FancyArrowPatch((0.5,y1),(0.5,y2),arrowstyle="->",mutation_scale=9,color="#52657a"))
    ax.text(0.5,0.10,"Interpret gains using\ncontrols, splits and uncertainty",ha="center",va="center",fontsize=6.6,color="#21425e",fontweight="bold")


def panel_b(ax):
    ax.set_title("B  Matched means (3 seeds)", loc="left", fontweight="bold")
    metrics = ["AUC", "AUPRC"]
    esm = [ESM_AUC, ESM_AUPRC]
    gcn = [GCN_AUC, GCN_AUPRC]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, esm, w, yerr=[rraw["test_auc_sd"],rraw["test_auprc_sd"]], capsize=2, color=COL_ESM, edgecolor="black", linewidth=0.45, label="Raw ESM")
    ax.bar(x + w / 2, gcn, w, yerr=[rfull["test_auc_sd"],rfull["test_auprc_sd"]], capsize=2, color=COL_GCN, edgecolor="black", linewidth=0.45, label="Full FPP")
    ax.text(0, 0.810, f"Delta={GCN_AUC-ESM_AUC:+.4f}", ha="center", va="bottom", fontsize=6.6)
    ax.text(1, 0.664, f"Delta={GCN_AUPRC-ESM_AUPRC:+.4f}", ha="center", va="bottom", fontsize=6.6)
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
        ax.legend(frameon=True, facecolor="white", edgecolor="white", framealpha=0.95, loc="upper right", fontsize=5.8)
    elif legend_loc == "above":
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=6.6)
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
    ax.set_title("E  Historical partial ablation", loc="left", fontweight="bold")
    styles = {
        "random_pair_removal": ("Random removal", COL_RANDOM, "o"),
        "high_confidence_first": ("High first", COL_HIGH, "s"),
        "low_confidence_first": ("Low first", COL_GCN, "^"),
    }
    for strategy, (label, color, marker) in styles.items():
        sub = df44[df44["strategy"] == strategy].sort_values("fraction_removed")
        ax.errorbar(sub["fraction_removed"] * 100, sub["delta_test_auc_vs_raw_esm"],
                    yerr=sub["test_auc_sd"], color=color, ecolor=color, marker=marker,
                    markersize=4.1, lw=1.6, capsize=2.4, label=label)
    ax.axhline(0, color="#4D4D4D", lw=0.85, ls="--")
    ax.set_xlabel("Direct train-test pairs removed\n(%)")
    ax.set_ylabel("Delta AUC vs ESM raw")
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.020, 0.045)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)


def panel_f(ax):
    grouped_metric_panel(
        ax,
        "F  Topology permutation",
        [
            ("Raw ESM", [ESM_AUC, ESM_AUPRC], [rraw["test_auc_sd"],rraw["test_auprc_sd"]], COL_ESM),
            ("Real FPP", [GCN_AUC, GCN_AUPRC], [rfull["test_auc_sd"],rfull["test_auprc_sd"]], COL_GCN),
            ("Permuted FPP", [PERM_AUC_MEAN, PERM_AUPRC_MEAN], [PERM_AUC_SD, PERM_AUPRC_SD], COL_PERM),
        ],
        legend_loc="upper right",
    )


fig = plt.figure(figsize=(7.4, 4.7))
gs = fig.add_gridspec(2, 3, width_ratios=[1.08, 1.02, 1.08], wspace=0.60, hspace=0.65)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

panel_a(axes[0])
panel_b(axes[1])
panel_c(axes[2])
panel_d(axes[3])
panel_e(axes[4])
panel_f(axes[5])

fig.suptitle(
    "Matched fixed STRING propagation: connectivity and topology audit",
    fontsize=9.0,
    fontweight="bold",
    y=0.985,
)

fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.13)
fig.text(0.5, 0.015, "B/D/F: matched classifier protocols and propagation; E: historical unmatched curve.", ha="center", fontsize=6.6)
png = OUT_DIR / "Figure1_fixed_propagation_audit.png"
pdf = OUT_DIR / "Figure1_fixed_propagation_audit.pdf"
fig.savefig(png, bbox_inches="tight", dpi=300)
fig.savefig(pdf, bbox_inches="tight")
print(png)
print(pdf)
