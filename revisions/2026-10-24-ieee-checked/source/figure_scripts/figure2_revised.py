from pathlib import Path
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch



ROOT = Path(__file__).resolve().parents[1] / "figure_inputs"
OUT_DIR = Path(__file__).resolve().parents[1] / "figure_drafts"
PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR.mkdir(parents=True, exist_ok=True)

P = ROOT / "data/processed"

plt.rcParams.update({
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.2,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.0, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.8})

COL_BASE = "#9ECAE1"
COL_GRAPH = "#3182BD"
COL_EDGE = "#6BAED6"
COL_PERM = "#BDBDBD"
COL_RAW = "#74C476"
COL_CADDAM = "#FDAE6B"
COL_HOMO = "#BCBDDC"
COL_TEXT = "#252525"
COL_GRAY = "#636363"
COL_ORANGE = "#E6550D"


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def row_by(rows, **kwargs):
    for row in rows:
        if all(row.get(k) == v for k, v in kwargs.items()):
            return row
    raise KeyError(kwargs)


cadd_report = load_json(P / "cadd_match_report.json")
am_report = load_json(P / "alphamissense_match_report.json")
homo = load_json(P / "homology_rerun_core/homology_rerun_summary.json")

# Test prevalence is calculated from the same frozen matched-test manifests.
orig_test = read_csv_rows(PROJECT / "matched_validation_20261024/original_test_rows.csv")
homo_test = read_csv_rows(PROJECT / "matched_validation_20261024/homology_test_rows.csv")
orig_prev = sum(int(r["Label"]) for r in orig_test) / len(orig_test)
homo_prev = sum(int(r["Label"]) for r in homo_test) / len(homo_test)


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

orig_caddam=newrow("original_fusion","scores_capacity_matched")
orig_raw=newrow("original_fusion","scores_plus_raw_ESM")
orig_edge=newrow("original_fusion","scores_plus_edge_ablated_FPP")
orig_perm=newrow("original_fusion","scores_plus_permuted_FPP")
orig_real=newrow("original_fusion","scores_plus_full_FPP")
homo_caddam=newrow("homology","scores_capacity_matched")
homo_raw=newrow("homology","scores_plus_raw_ESM")
homo_graph=newrow("homology","scores_plus_full_FPP")
margin_rows={}
for old,new in [("real_vs_archmatched_cadd_am","scores_capacity_matched"),("real_vs_raw_esm_fusion","scores_plus_raw_ESM"),("real_vs_edge_ablated_fusion","scores_plus_edge_ablated_FPP"),("real_vs_topology_permuted_fusion","scores_plus_permuted_FPP_graphmean")]:
    b=newboot["comparisons"]["original_fusion"]["scores_plus_full_FPP_vs_"+new]
    margin_rows[old]={"delta_auc":b["delta_auc"],"delta_auprc":b["delta_auprc"],"delta_auc_ci95_low":b["auc_ci95"][0],"delta_auc_ci95_high":b["auc_ci95"][1],"delta_auprc_ci95_low":b["auprc_ci95"][0],"delta_auprc_ci95_high":b["auprc_ci95"][1]}

def panel_a(ax):
    ax.set_title("A  Fusion-control setup", loc="left", fontweight="bold")
    ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    def box(cx,cy,w,h,text,color):
        ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.01",fc=color,ec="#555555",lw=.6))
        ax.text(cx,cy,text,ha="center",va="center",fontsize=5.8,linespacing=1.0)
    box(.5,.90,.7,.17,"ClinVar-STRING\n177,213 variants",COL_BASE)
    box(.23,.66,.40,.22,"CADD\n177,205\n99.995%",COL_CADDAM)
    box(.77,.66,.40,.22,"AlphaMissense\n171,753\n96.92%","#A1D99B")
    box(.5,.39,.67,.15,"CADD+AM baseline", "#FDD0A2")
    box(.5,.13,.95,.18,"+ controls: raw ESM\nablated / permuted / real FPP", "#9ECAE1")
    for a,b in [((.4,.83),(.23,.77)),((.6,.83),(.77,.77)),((.23,.55),(.42,.47)),((.77,.55),(.58,.47)),((.5,.315),(.5,.22))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=6,color="#555555",lw=.7))


def panel_b(ax):
    ax.set_title("B  Matched original fusion", loc="left", fontweight="bold")
    labels = ["Scores", "+Raw", "+Abl.", "+Perm.", "+Real"]
    rows = [orig_caddam, orig_raw, orig_edge, orig_perm, orig_real]
    auc = [f(r, "test_auc_mean") for r in rows]
    ap = [f(r, "test_auprc_mean") for r in rows]
    auc_sd = [f(r, "test_auc_sd") for r in rows]
    ap_sd = [f(r, "test_auprc_sd") for r in rows]

    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, auc, w, yerr=auc_sd, color=COL_GRAPH, edgecolor="black",
           linewidth=0.4, capsize=2, label="AUC", zorder=3)
    ax.bar(x + w / 2, ap, w, yerr=ap_sd, color=COL_BASE, edgecolor="black",
           linewidth=0.4, capsize=2, label="AUPRC", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.89, 1.005)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    clean_axis(ax)


def panel_c(ax):
    ax.set_title("C  Real FPP margins", loc="left", fontweight="bold")
    controls = [
        ("vs scores", margin_rows["real_vs_archmatched_cadd_am"]),
        ("vs raw", margin_rows["real_vs_raw_esm_fusion"]),
        ("vs ablated", margin_rows["real_vs_edge_ablated_fusion"]),
        ("vs permuted", margin_rows["real_vs_topology_permuted_fusion"]),
    ]
    y = np.arange(len(controls))[::-1]
    auc_delta = [f(r, "delta_auc") for _, r in controls]
    ap_delta = [f(r, "delta_auprc") for _, r in controls]
    auc_low = [f(r, "delta_auc_ci95_low") for _, r in controls]
    auc_high = [f(r, "delta_auc_ci95_high") for _, r in controls]
    ap_low = [f(r, "delta_auprc_ci95_low") for _, r in controls]
    ap_high = [f(r, "delta_auprc_ci95_high") for _, r in controls]
    auc_xerr = np.array([[d - lo for d, lo in zip(auc_delta, auc_low)],
                         [hi - d for d, hi in zip(auc_delta, auc_high)]])
    ap_xerr = np.array([[d - lo for d, lo in zip(ap_delta, ap_low)],
                        [hi - d for d, hi in zip(ap_delta, ap_high)]])
    h = 0.32
    ax.barh(y + h / 2, auc_delta, height=h, color=COL_GRAPH, edgecolor="black",
            linewidth=0.4, label="AUC", zorder=3)
    ax.barh(y - h / 2, ap_delta, height=h, color=COL_BASE, edgecolor="black",
            linewidth=0.4, label="AUPRC", zorder=3)
    ax.errorbar(auc_delta, y + h / 2, xerr=auc_xerr, fmt="none", ecolor=COL_TEXT,
                elinewidth=0.8, capsize=2.2, zorder=4)
    ax.errorbar(ap_delta, y - h / 2, xerr=ap_xerr, fmt="none", ecolor=COL_TEXT,
                elinewidth=0.8, capsize=2.2, zorder=4)
    ax.axvline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in controls])
    ax.set_xlabel("Delta performance")
    ax.set_xlim(-0.004, 0.060)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    clean_axis(ax)


def panel_d(ax):
    ax.set_title("D  Historical homology test", loc="left", fontweight="bold")
    labels = ["AUC", "AUPRC"]
    esm = [float(homo["esm_raw_homology"]["test_auc"]), float(homo["esm_raw_homology"]["test_auprc"])]
    gcn = [float(homo["gcn_enhanced_homology"]["test_auc"]), float(homo["gcn_enhanced_homology"]["test_auprc"])]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, esm, w, color=COL_BASE, edgecolor="black", linewidth=0.4, label="ESM")
    ax.bar(x + w / 2, gcn, w, color=COL_GRAPH, edgecolor="black", linewidth=0.4, label="FPP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.18, 0.80)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)


def panel_e(ax):
    ax.set_title("E  Homology fusion", loc="left", fontweight="bold")
    labels = ["Scores\n(n=3)", "+Raw\n(n=3)", "+FPP\n(n=3)"]
    base = [f(homo_caddam, "test_auc_mean"), f(homo_caddam, "test_auprc_mean")]
    raw = [f(homo_raw, "test_auc_mean"), f(homo_raw, "test_auprc_mean")]
    graph = [f(homo_graph,"test_auc_mean"), f(homo_graph,"test_auprc_mean")]
    rows = [base, raw, graph]
    auc = [r[0] for r in rows]
    ap = [r[1] for r in rows]
    y = np.arange(len(labels))[::-1]
    h = 0.34
    ax.barh(y + h / 2, auc, height=h, xerr=[f(r,"test_auc_sd") for r in [homo_caddam,homo_raw,homo_graph]], capsize=2, color=COL_GRAPH, edgecolor="black",
            linewidth=0.4, label="AUC", zorder=3)
    ax.barh(y - h / 2, ap, height=h, xerr=[f(r,"test_auprc_sd") for r in [homo_caddam,homo_raw,homo_graph]], capsize=2, color=COL_BASE, edgecolor="black",
            linewidth=0.4, label="AUPRC", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test performance")
    ax.set_xlim(0.80, 0.985)
    ax.set_ylim(-0.55, 3.02)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.9)
    clean_axis(ax)


def panel_f(ax):
    ax.set_title("F  Test-set prevalence", loc="left", fontweight="bold")
    vals = [orig_prev, homo_prev]
    labels = ["Original\nsplit", "Homology\nsplit"]
    ax.bar(np.arange(2), vals, color=[COL_BASE, COL_HOMO], edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v * 100:.2f}%", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(0, 0.38)
    clean_axis(ax)


fig = plt.figure(figsize=(7.4, 4.7))
gs = fig.add_gridspec(2, 3, width_ratios=[1.13, 1.05, 1.04], wspace=0.68, hspace=0.68)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

panel_a(axes[0])
panel_b(axes[1])
panel_c(axes[2])
panel_d(axes[3])
panel_e(axes[4])
panel_f(axes[5])

fig.suptitle("Fixed-propagation fusion controls and homology stress test",
             fontsize=9.0, fontweight="bold", y=0.985)
fig.subplots_adjust(left=0.065, right=0.985, top=0.89, bottom=0.10, hspace=0.64)

for name in ["Figure2_fusion_homology"]:
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    print(OUT_DIR / f"{name}.png")
    print(OUT_DIR / f"{name}.pdf")
