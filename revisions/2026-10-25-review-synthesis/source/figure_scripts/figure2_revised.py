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
    "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.0,
    "figure.dpi": 160, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

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

# Test prevalence is calculated from the same frozen matched-test manifests.
orig_test = read_csv_rows(PROJECT / "matched_validation_20261024/original_test_rows.csv")
homo_test = read_csv_rows(PROJECT / "matched_validation_20261024/homology_test_rows.csv")
orig_prev = sum(int(r["Label"]) for r in orig_test) / len(orig_test)
homo_prev = sum(int(r["Label"]) for r in homo_test) / len(homo_test)


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
    ax.set_title("A  Score coverage and matched fusion controls", loc="left", fontweight="bold", pad=4)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nodes = [(0.10, "ClinVar-STRING\n177,213 variants", COL_BASE),
             (0.36, "CADD: 177,205 (99.995%)\nAM: 171,753 (96.92%)", "#FDD0A2"),
             (0.65, "Scores only or + raw ESM /\nablated / permuted / real FPP", "#e1ebf3"),
             (0.91, "Same MLP hidden layers\nand training protocol", "#eef2f5")]
    for x, label, color in nodes:
        ax.text(x, .38, label, ha="center", va="center", fontsize=6.9,
                bbox=dict(boxstyle="round,pad=.35", fc=color, ec="#65768a", lw=.7))
    for x in [.21, .50, .80]:
        ax.add_patch(FancyArrowPatch((x, .38), (x+.02, .38), arrowstyle="->", mutation_scale=9, color="#52657a"))


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
    ax.set_title("D  Matched homology fusion", loc="left", fontweight="bold")
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


def panel_e(ax):
    ax.set_title("E  Test-set prevalence", loc="left", fontweight="bold")
    vals = [orig_prev, homo_prev]
    labels = ["Original\nsplit", "Homology\nsplit"]
    ax.bar(np.arange(2), vals, color=[COL_BASE, COL_HOMO], edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v * 100:.2f}%", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(0, 0.38)
    clean_axis(ax)


def main():
    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[.52, 1.1, 1.1], wspace=.46, hspace=.73)
    panel_a(fig.add_subplot(gs[0, :]))
    panel_b(fig.add_subplot(gs[1, 0]))
    panel_c(fig.add_subplot(gs[1, 1]))
    panel_d(fig.add_subplot(gs[2, 0]))
    panel_e(fig.add_subplot(gs[2, 1]))
    fig.subplots_adjust(left=.08, right=.985, top=.96, bottom=.11)
    for suffix in ["png", "pdf"]:
        path = OUT_DIR / f"Figure2_fusion_homology.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(path)


if __name__ == "__main__":
    main()
