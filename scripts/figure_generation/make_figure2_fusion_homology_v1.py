from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


LOCAL_ROOT = Path("/Users/teyian/Documents/paper1/server_snapshot_full/root")
LOCAL_OUT = Path("/Users/teyian/Documents/paper1/figure_drafts")
SERVER_ROOT = Path("/root")
SERVER_OUT = Path("/root/paper_assets_final")

ROOT = Path(os.environ.get("PAPER1_ROOT", SERVER_ROOT if (SERVER_ROOT / "data/processed").exists() else LOCAL_ROOT))
OUT_DIR = Path(os.environ.get("PAPER1_FIG_OUT", SERVER_OUT if ROOT == SERVER_ROOT else LOCAL_OUT))
OUT_DIR.mkdir(parents=True, exist_ok=True)

P = ROOT / "data/processed"

plt.rcParams.update({
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.2,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

COL_BASE = "#9ECAE1"
COL_GRAPH = "#3182BD"
COL_AM = "#74C476"
COL_CADDAM = "#FDAE6B"
COL_HOMO = "#BCBDDC"
COL_GRAY = "#7A7A7A"
COL_RED = "#DE2D26"
COL_TEXT = "#252525"


def load_json(rel):
    with open(P / rel) as f:
        return json.load(f)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


cadd_report = load_json("cadd_match_report.json")
am_report = load_json("alphamissense_match_report.json")
triple = load_json("triple_fusion/summary.json")
homo = load_json("homology_rerun_core/homology_rerun_summary.json")

boot = pd.read_csv(P / "bootstrap_significance/bootstrap_significance_summary.csv")
hboot = pd.read_csv(P / "homology_bootstrap_significance/homology_bootstrap_significance_summary.csv")

orig_test = pd.read_csv(P / "clinvar_test_main.csv", low_memory=False)
homo_test = pd.read_csv(P / "homology_aware_split/clinvar_test_homology.csv", low_memory=False)
orig_prev = float(orig_test["Label"].astype(int).mean())
homo_prev = float(homo_test["Label"].astype(int).mean())


def get_boot(df, comparison, metric):
    row = df[(df["comparison"] == comparison) & (df["metric"] == metric)].iloc[0]
    return float(row["mean_delta"]), float(row["ci95_low"]), float(row["ci95_high"])


orig_auc_ci = get_boot(boot, "full_vs_cadd_am", "delta_auc")
orig_ap_ci = get_boot(boot, "full_vs_cadd_am", "delta_auprc")
homo_auc_ci = get_boot(hboot, "full_vs_cadd_am_homology", "delta_auc")
homo_ap_ci = get_boot(hboot, "full_vs_cadd_am_homology", "delta_auprc")


def panel_a(ax):
    ax.set_title("A  External predictor matching and fusion setup", loc="left", fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def draw_box(cx, cy, w, h, label, color):
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.008",
            fc=color, ec="#4D4D4D", lw=0.8, alpha=0.50, zorder=3
        )
        ax.add_patch(patch)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.3,
                linespacing=1.05, color=COL_TEXT, zorder=4)

    source = dict(cx=0.15, cy=0.60, w=0.26, h=0.16)
    cadd = dict(cx=0.47, cy=0.74, w=0.20, h=0.15)
    am = dict(cx=0.47, cy=0.46, w=0.24, h=0.15)
    fusion = dict(cx=0.82, cy=0.60, w=0.23, h=0.16)

    draw_box(**source, label="ClinVar-STRING\nvariants\nn=177,213", color=COL_BASE)
    draw_box(**cadd, label=f"CADD\n{cadd_report['n_matched_keys']:,}\n{cadd_report['key_match_rate']*100:.3f}%", color=COL_CADDAM)
    draw_box(**am, label=f"AlphaMissense\n{am_report['n_matched_keys']:,}\n{am_report['key_match_rate']*100:.2f}%", color=COL_AM)
    draw_box(**fusion, label="Fusion model\nCADD + AM\n+ PPI-GCN", color=COL_GRAPH)

    def edge(box, side, y_offset=0.0):
        if side == "right":
            return (box["cx"] + box["w"] / 2 + 0.018, box["cy"] + y_offset)
        if side == "left":
            return (box["cx"] - box["w"] / 2 - 0.018, box["cy"] + y_offset)
        raise ValueError(side)

    arrows = [
        (edge(source, "right", 0.035), edge(cadd, "left", 0.000)),
        (edge(source, "right", -0.035), edge(am, "left", 0.000)),
        (edge(cadd, "right", 0.000), edge(fusion, "left", 0.035)),
        (edge(am, "right", 0.000), edge(fusion, "left", -0.035)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9.5,
            color="#4D4D4D", lw=1.0, connectionstyle="arc3,rad=0",
            zorder=2
        ))

    ax.text(fusion["cx"], 0.31, "same downstream classifier", ha="center", va="center",
            fontsize=7.0, color="#4D4D4D")


def grouped_bars(ax, title, labels, vals_auc, vals_ap, ylim=None):
    ax.set_title(title, loc="left", fontweight="bold")
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, vals_auc, width, color=COL_GRAPH, edgecolor="black", linewidth=0.4, label="AUC")
    ax.bar(x + width / 2, vals_ap, width, color=COL_BASE, edgecolor="black", linewidth=0.4, label="AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test performance")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        vals = vals_auc + vals_ap
        ymin = min(vals) - 0.035
        ymax = min(1.0, max(vals) + 0.025)
        ax.set_ylim(max(0, ymin), ymax)


def panel_b(ax):
    labels = ["CADD", "AM", "CADD+AM", "GCN+\nCADD+AM"]
    keys = ["cadd_only", "am_only", "cadd_plus_am", "gcn_plus_cadd_plus_am"]
    vals_auc = [float(triple[k]["test_auc"]) for k in keys]
    vals_ap = [float(triple[k]["test_auprc"]) for k in keys]
    grouped_bars(ax, "B  Original split performance", labels, vals_auc, vals_ap, ylim=(0.78, 0.99))


def interval_panel(ax, title, rows, xlim):
    ax.set_title(title, loc="left", fontweight="bold")
    y = np.arange(len(rows))[::-1]
    for yi, (label, (mean, lo, hi), color) in zip(y, rows):
        ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, elinewidth=1.8, capsize=3, markersize=4.5)
    ax.axvline(0, color="#4D4D4D", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Delta performance")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    clean_axis(ax)


def panel_c(ax):
    rows = [
        ("AUC\noriginal", orig_auc_ci, COL_GRAPH),
        ("AUPRC\noriginal", orig_ap_ci, COL_GRAPH),
    ]
    interval_panel(ax, "C  Original split bootstrap gain", rows, (-0.002, 0.040))


def panel_d(ax):
    ax.set_title("D  Homology-aware split: ESM vs PPI-GCN", loc="left", fontweight="bold")
    labels = ["AUC", "AUPRC"]
    esm = [float(homo["esm_raw_homology"]["test_auc"]), float(homo["esm_raw_homology"]["test_auprc"])]
    gcn = [float(homo["gcn_enhanced_homology"]["test_auc"]), float(homo["gcn_enhanced_homology"]["test_auprc"])]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, esm, w, color=COL_BASE, edgecolor="black", linewidth=0.4, label="ESM")
    ax.bar(x + w / 2, gcn, w, color=COL_GRAPH, edgecolor="black", linewidth=0.4, label="PPI-GCN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.18, 0.80)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)


def panel_e(ax):
    ax.set_title("E  Homology split: CADD+AM vs graph fusion", loc="left", fontweight="bold")
    labels = ["AUC", "AUPRC"]
    base = [float(homo["cadd_plus_am_homology"]["test_auc"]), float(homo["cadd_plus_am_homology"]["test_auprc"])]
    full = [float(homo["gcn_plus_cadd_plus_am_homology"]["test_auc"]), float(homo["gcn_plus_cadd_plus_am_homology"]["test_auprc"])]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, base, w, color=COL_CADDAM, edgecolor="black", linewidth=0.4, label="CADD+AM")
    ax.bar(x + w / 2, full, w, color=COL_GRAPH, edgecolor="black", linewidth=0.4, label="GCN+CADD+AM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.80, 0.99)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)


def panel_f(ax):
    ax.set_title("F  Test-set prevalence differs by split", loc="left", fontweight="bold")
    vals = [orig_prev, homo_prev]
    labels = ["Original\nsplit", "Homology-aware\nsplit"]
    ax.bar(np.arange(2), vals, color=[COL_BASE, COL_HOMO], edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(0, 0.38)
    clean_axis(ax)


fig = plt.figure(figsize=(15.8, 8.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.06, 1.08, 1.06], wspace=0.54, hspace=0.64)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

panel_a(axes[0])
panel_b(axes[1])
panel_c(axes[2])
panel_d(axes[3])
panel_e(axes[4])
panel_f(axes[5])

fig.suptitle("PPI graph context is split-dependent and bounded by strong variant-effect predictors",
             fontsize=13.2, fontweight="bold", y=0.985)

png = OUT_DIR / "Figure2_fusion_homology_v1.png"
pdf = OUT_DIR / "Figure2_fusion_homology_v1.pdf"
fig.savefig(png, bbox_inches="tight", dpi=300)
fig.savefig(pdf, bbox_inches="tight")
print(png)
print(pdf)
