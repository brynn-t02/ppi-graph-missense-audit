from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



ROOT = Path(__file__).resolve().parents[1] / "figure_inputs"
OUT_DIR = Path(__file__).resolve().parents[1] / "figure_drafts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
P = ROOT / "data/processed"

plt.rcParams.update({
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.2,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.0, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.8})

COL_AUC = "#3182BD"
COL_AP = "#9ECAE1"
COL_POS = "#3182BD"
COL_NEG = "#D95F0E"
COL_GRAY = "#7A7A7A"
COL_TEXT = "#252525"


def load_json(rel):
    with open(P / rel) as f:
        return json.load(f)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_csv(rel):
    return pd.read_csv(P / rel)


align = load_json("esm_ppi_alignment_analysis/summary.json")
comm = load_json("esm_ppi_community_alignment_analysis/summary.json")
topo = read_csv("topology_gain_analysis_exp26B/topology_stratified_metrics.csv")
between = read_csv("betweenness_gain_analysis_exp30/betweenness_stratified_metrics.csv")
clust = read_csv("clustering_gain_analysis_exp31/clustering_stratified_metrics.csv")
risk = read_csv("neighborhood_risk_enrichment_exp27/neighbor_risk_stratified_metrics.csv")
joint = read_csv("topology_profile_joint_map_exp33/joint_gain_maps.csv")
joint_boot = read_csv("topology_profile_joint_map_bootstrap_exp33B/key_joint_cells_bootstrap_summary.csv")


def panel_a(ax):
    ax.set_title("A  ESM and STRING similarity", loc="left", fontweight="bold")
    labels = ["Network\npair", "Random\npair", "Within\ncomm.", "Between\ncomm."]
    means = [align["neighbor_mean"], align["random_mean"], comm["within_mean"], comm["between_mean"]]
    colors = [COL_AUC, "#BDBDBD", COL_AUC, "#BDBDBD"]
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.4, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean ESM cosine similarity")
    ax.set_ylim(0.855, 0.887)
    ax.text(1.5, .886, "Cliff's delta=0.124;\nmean difference=0.010", ha="center",va="top",fontsize=6.0,color=COL_TEXT)
    clean_axis(ax)


def paired_delta_bars(ax, title, df, labels):
    ax.set_title(title, loc="left", fontweight="bold")
    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, df["delta_auc"], w, color=COL_AUC, edgecolor="black", linewidth=0.4, label="Delta AUC")
    ax.bar(x + w / 2, df["delta_auprc"], w, color=COL_AP, edgecolor="black", linewidth=0.4, label="Delta AUPRC")
    ax.axhline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("FPP minus raw ESM")
    ax.set_ylim(-0.04, 0.115)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)


def panel_b(ax):
    degree = topo[topo["group_col"] == "degree_bucket"].copy()
    kcore = topo[topo["group_col"] == "kcore_bucket"].copy()
    degree["label"] = ["deg L", "deg M", "deg H"]
    kcore["label"] = ["core L", "core M", "core H"]
    df = pd.concat([degree, kcore], ignore_index=True)
    paired_delta_bars(ax, "B  Degree and k-core strata", df, df["label"].tolist())


def panel_c(ax):
    b = between.copy()
    c = clust.copy()
    b["label"] = ["betw L", "betw M", "betw H"]
    c["label"] = ["clust L", "clust M", "clust H"]
    df = pd.concat([b, c], ignore_index=True)
    paired_delta_bars(ax, "C  Betweenness / clustering", df, df["label"].tolist())


def panel_d(ax):
    ax.set_title("D  Neighbour-risk strata", loc="left", fontweight="bold")
    df = risk.copy()
    labels = ["Low\n(10.6%)", "Mid\n(28.8%)", "High\n(57.7%)"]
    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, df["delta_auc"], w, color=COL_AUC, edgecolor="black", linewidth=0.4, label="Delta AUC")
    ax.bar(x + w / 2, df["delta_auprc"], w, color=COL_AP, edgecolor="black", linewidth=0.4, label="Delta AUPRC")
    ax.axhline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FPP minus raw ESM")
    ax.set_ylim(-0.015, 0.075)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)


def pretty_map_name(name):
    return (name
            .replace("kcore_bucket__x__betweenness_bucket", "k-core x betweenness")
            .replace("kcore_bucket__x__clustering_bucket", "k-core x clustering")
            .replace("neighbor_risk_bucket__x__kcore_bucket", "risk x k-core")
            .replace("neighbor_risk_bucket__x__clustering_bucket", "risk x clustering")
            .replace("betweenness_bucket__x__boundary_role_bucket", "betweenness x boundary"))


def panel_e(ax):
    ax.set_title("E  Exploratory joint cells", loc="left", fontweight="bold")
    df = joint[joint["n"] >= 50].copy()
    vals = df["delta_auc"].to_numpy()
    bins = np.linspace(-0.075, 0.155, 13)
    counts, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges) * 0.92
    colors = [COL_POS if c >= 0 else COL_NEG for c in centers]
    ax.bar(centers, counts, width=widths, color=colors, edgecolor="black", linewidth=0.35, alpha=0.78)
    ax.axvline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_ylabel("Number of joint cells")
    ax.set_xlabel("Delta AUC")
    ax.set_xlim(-0.075, 0.155)
    ax.set_ylim(0, max(counts) + 2)
    ax.text(0.151, max(counts) + 1.35, f"{len(df)} tested cells", ha="right", va="top",
            fontsize=6.6, color=COL_TEXT)
    clean_axis(ax)


def panel_f(ax):
    ax.set_title("F  Selected profiles", loc="left", fontweight="bold")
    order = [
        "mid_kcore__high_betweenness",
        "mid_kcore__mid_clustering",
        "high_neighbor_risk__mid_kcore",
        "high_kcore__low_clustering",
    ]
    labels = [
        "Mid core /\nhigh betw.",
        "Mid core /\nmid clust.",
        "High risk /\nmid core",
        "High core /\nlow clust.",
    ]
    df = joint_boot.set_index("cell_name").loc[order].reset_index()
    y = np.arange(len(df))[::-1]
    colors = [COL_POS if v >= 0 else COL_NEG for v in df["point_delta_auc"]]
    for yi, (_, row), color in zip(y, df.iterrows(), colors):
        mean = row["point_delta_auc"]
        lo = row["boot_delta_auc_ci_lower"]
        hi = row["boot_delta_auc_ci_upper"]
        ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, elinewidth=1.8, capsize=3, markersize=4.5)
    ax.axvline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta AUC\n(CI unadjusted for selection)")
    ax.set_xlim(-0.075, 0.175)
    ax.set_ylim(-0.6, len(df) - 0.4)
    clean_axis(ax)


def main():
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.5))
    fig.suptitle("Fixed propagation: exploratory topology-context profiles",
                 fontsize=9.0, fontweight="bold", y=0.985)
    panel_a(axs[0, 0])
    panel_b(axs[0, 1])
    panel_c(axs[0, 2])
    panel_d(axs[1, 0])
    panel_e(axs[1, 1])
    panel_f(axs[1, 2])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.105, wspace=0.66, hspace=0.62)
    png = OUT_DIR / "Figure3_exploratory_topology.png"
    pdf = OUT_DIR / "Figure3_exploratory_topology.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
