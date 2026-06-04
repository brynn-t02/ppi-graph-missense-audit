from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path("/root")
OUT_DIR = ROOT / "data/processed/paper_figures_reframed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Real input files only
# -----------------------------
P_03F = ROOT / "data/processed/variant_mlp_comparison/summary.json"
P_12 = ROOT / "data/processed/bootstrap_significance/bootstrap_significance_summary.csv"
P_42A = ROOT / "data/processed/clustered_bootstrap_exp42/42A_gcn_vs_esm_clustered_bootstrap.json"
P_41 = ROOT / "data/processed/esm_gcn_edge_ablation_exp41/per_repeat_esm_gcn_edge_ablation_metrics.csv"
P_44 = ROOT / "data/processed/esm_gcn_partial_edge_ablation_exp44/edge_ablation_curve_summary.csv"
P_45 = ROOT / "data/processed/esm_gcn_topology_permutation_exp45/summary.json"

for p in [P_03F, P_12, P_42A, P_41, P_44, P_45]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

# -----------------------------
# Style
# -----------------------------
plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 160,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
COL_ORANGE = "#Fdae6b"

# -----------------------------
# Load 03F
# -----------------------------
with open(P_03F) as f:
    d03 = json.load(f)

for key in ["esm_raw", "gcn_enhanced"]:
    if key not in d03:
        raise KeyError(f"{P_03F} missing key: {key}")

ESM_AUC = float(d03["esm_raw"]["test_auc"])
ESM_AUPRC = float(d03["esm_raw"]["test_auprc"])
GCN_AUC = float(d03["gcn_enhanced"]["test_auc"])
GCN_AUPRC = float(d03["gcn_enhanced"]["test_auprc"])

# -----------------------------
# Load experiment 12
# -----------------------------
df12 = pd.read_csv(P_12)
need12 = ["comparison", "model_a", "model_b", "metric", "mean_delta", "ci95_low", "ci95_high"]
for c in need12:
    if c not in df12.columns:
        raise KeyError(f"{P_12} missing column: {c}")

def exp12(metric):
    sub = df12[
        (df12["comparison"] == "gcn_vs_esm") &
        (df12["metric"] == metric)
    ]
    if len(sub) != 1:
        raise ValueError(f"Expected one row for gcn_vs_esm / {metric}; found {len(sub)}")
    r = sub.iloc[0]
    return float(r["mean_delta"]), float(r["ci95_low"]), float(r["ci95_high"])

VAR_AUC = exp12("delta_auc")
VAR_AUPRC = exp12("delta_auprc")

# -----------------------------
# Load experiment 42A
# -----------------------------
with open(P_42A) as f:
    d42 = json.load(f)

for k in ["point", "clustered_bootstrap"]:
    if k not in d42:
        raise KeyError(f"{P_42A} missing key: {k}")

PROT_AUC = (
    float(d42["point"]["delta_auc"]),
    float(d42["clustered_bootstrap"]["delta_auc_ci95"][0]),
    float(d42["clustered_bootstrap"]["delta_auc_ci95"][1]),
)
PROT_AUPRC = (
    float(d42["point"]["delta_auprc"]),
    float(d42["clustered_bootstrap"]["delta_auprc_ci95"][0]),
    float(d42["clustered_bootstrap"]["delta_auprc_ci95"][1]),
)

# -----------------------------
# Load experiment 41
# -----------------------------
df41 = pd.read_csv(P_41)
for c in ["test_auc", "test_auprc"]:
    if c not in df41.columns:
        raise KeyError(f"{P_41} missing column: {c}")

EDGE_AUC_MEAN = float(df41["test_auc"].mean())
EDGE_AUC_SD = float(df41["test_auc"].std(ddof=1))
EDGE_AUPRC_MEAN = float(df41["test_auprc"].mean())
EDGE_AUPRC_SD = float(df41["test_auprc"].std(ddof=1))

# -----------------------------
# Load experiment 44
# -----------------------------
df44 = pd.read_csv(P_44)
need44 = [
    "strategy", "fraction_removed",
    "delta_test_auc_vs_raw_esm",
    "test_auc_sd",
]
for c in need44:
    if c not in df44.columns:
        raise KeyError(f"{P_44} missing column: {c}")

# -----------------------------
# Load experiment 45
# -----------------------------
with open(P_45) as f:
    d45 = json.load(f)

m45 = d45["summary_metrics"]
PERM_AUC_MEAN = float(m45["test_auc_mean"])
PERM_AUC_SD = float(m45["test_auc_sd"])
PERM_AUPRC_MEAN = float(m45["test_auprc_mean"])
PERM_AUPRC_SD = float(m45["test_auprc_sd"])

# -----------------------------
# Panel functions
# -----------------------------
def panel_a(ax):
    ax.set_title("A  Protein-level splits share one PPI graph", loc="left", fontweight="bold")
    ax.axis("off")

    train = np.array([[0.10, 0.70], [0.23, 0.83], [0.30, 0.58], [0.17, 0.43]])
    test = np.array([[0.70, 0.69], [0.84, 0.82], [0.82, 0.52], [0.62, 0.42]])
    val = np.array([[0.44, 0.24], [0.53, 0.32]])

    def edge(a, b, color="#BDBDBD", lw=1.0):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=1)

    # within-split edges
    for a, b in [(train[0], train[1]), (train[1], train[2]), (train[2], train[3]), (train[0], train[3])]:
        edge(a, b)
    for a, b in [(test[0], test[1]), (test[1], test[2]), (test[2], test[3]), (test[0], test[3])]:
        edge(a, b)
    edge(val[0], train[3])
    edge(val[1], test[3])

    # direct train-test edges
    edge(train[1], test[0], color=COL_HIGH, lw=2.0)
    edge(train[2], test[3], color=COL_HIGH, lw=2.0)

    ax.scatter(train[:,0], train[:,1], s=85, color=COL_GCN, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(test[:,0], test[:,1], s=85, color=COL_LOW, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(val[:,0], val[:,1], s=75, color=COL_ORANGE, edgecolor="white", linewidth=0.8, zorder=3)

    ax.text(0.06, 0.82, "Train", color=COL_GCN, fontweight="bold")
    ax.text(0.68, 0.82, "Test", color=COL_LOW, fontweight="bold")
    ax.text(0.37, 0.13, "Validation", color="#B15928", fontweight="bold")
    ax.text(0.36, 0.67, "train–test\nPPI edges", color=COL_HIGH, ha="center", va="center", fontsize=7.5)

    # compact pipeline
    y = 0.02
    ax.text(0.12, y, "ESM", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#4D4D4D", lw=0.7))
    ax.text(0.41, y, "PPI propagation", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#4D4D4D", lw=0.7))
    ax.text(0.72, y, "MLP", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#4D4D4D", lw=0.7))
    ax.add_patch(FancyArrowPatch((0.20, y), (0.30, y), arrowstyle="->", mutation_scale=8, color="#4D4D4D"))
    ax.add_patch(FancyArrowPatch((0.54, y), (0.63, y), arrowstyle="->", mutation_scale=8, color="#4D4D4D"))

def panel_b(ax):
    ax.set_title("B  Full-graph point estimates", loc="left", fontweight="bold")

    metrics = ["AUC", "AUPRC"]
    esm = [ESM_AUC, ESM_AUPRC]
    gcn = [GCN_AUC, GCN_AUPRC]
    x = np.arange(2)
    w = 0.34

    ax.bar(x - w/2, esm, w, color=COL_ESM, edgecolor="black", linewidth=0.5, label="ESM")
    ax.bar(x + w/2, gcn, w, color=COL_GCN, edgecolor="black", linewidth=0.5, label="PPI-GCN")

    ax.text(0, 0.825, f"Δ={GCN_AUC-ESM_AUC:+.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(1, 0.675, f"Δ={GCN_AUPRC-ESM_AUPRC:+.4f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.55, 0.85)
    ax.legend(frameon=False, loc="upper right")

def panel_c(ax):
    ax.set_title("C  Bootstrap uncertainty", loc="left", fontweight="bold")

    rows = [
        ("AUC / variant", VAR_AUC, COL_PURPLE),
        ("AUC / protein", PROT_AUC, COL_PURPLE_LIGHT),
        ("AUPRC / variant", VAR_AUPRC, COL_PURPLE),
        ("AUPRC / protein", PROT_AUPRC, COL_PURPLE_LIGHT),
    ]
    y = np.arange(len(rows))[::-1]

    for yi, (lab, (mean, lo, hi), col) in zip(y, rows):
        ax.errorbar(
            mean, yi,
            xerr=[[mean-lo], [hi-mean]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.6,
            capsize=3,
            markersize=4,
        )

    ax.axvline(0, color="#4D4D4D", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Δ performance (PPI-GCN − ESM)")
    ax.set_xlim(-0.02, 0.10)

def panel_d(ax):
    ax.set_title("D  Removing direct train–test edges", loc="left", fontweight="bold")

    x = np.array([0, 1])
    labels = ["AUC", "AUPRC"]

    data = {
        "ESM": ([ESM_AUC, ESM_AUPRC], [0, 0], COL_ESM, -0.18),
        "Full PPI-GCN": ([GCN_AUC, GCN_AUPRC], [0, 0], COL_GCN, 0.00),
        "Edge-ablated": ([EDGE_AUC_MEAN, EDGE_AUPRC_MEAN], [EDGE_AUC_SD, EDGE_AUPRC_SD], COL_EDGE, 0.18),
    }

    for name, (vals, errs, col, off) in data.items():
        ax.errorbar(x + off, vals, yerr=errs, fmt="o", markersize=6,
                    color=col, ecolor=col, capsize=3, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.56, 0.84)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)

def panel_e(ax):
    ax.set_title("E  Partial train–test edge ablation", loc="left", fontweight="bold")

    styles = {
        "random_pair_removal": ("Random", COL_RANDOM, "o"),
        "high_confidence_first": ("High-confidence first", COL_HIGH, "s"),
        "low_confidence_first": ("Low-confidence first", COL_LOW, "^"),
    }

    for strat, (lab, col, marker) in styles.items():
        sub = df44[df44["strategy"] == strat].sort_values("fraction_removed")
        if len(sub) == 0:
            raise ValueError(f"Missing strategy in exp44 CSV: {strat}")

        ax.errorbar(
            sub["fraction_removed"] * 100,
            sub["delta_test_auc_vs_raw_esm"],
            yerr=sub["test_auc_sd"],
            color=col,
            ecolor=col,
            marker=marker,
            markersize=4,
            lw=1.6,
            capsize=2.5,
            label=lab,
        )

    ax.axhline(0, color="#4D4D4D", lw=0.8, ls="--")
    ax.set_xlabel("Removed direct train–test pairs (%)")
    ax.set_ylabel("ΔAUC vs ESM")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.014, 0.028)
    ax.legend(frameon=False, loc="upper right")

def panel_f(ax):
    ax.set_title("F  Topology permutation control", loc="left", fontweight="bold")

    x = np.array([0, 1])
    labels = ["AUC", "AUPRC"]

    data = {
        "ESM": ([ESM_AUC, ESM_AUPRC], [0, 0], COL_ESM, -0.18),
        "Real PPI-GCN": ([GCN_AUC, GCN_AUPRC], [0, 0], COL_GCN, 0.00),
        "Permuted topology": ([PERM_AUC_MEAN, PERM_AUPRC_MEAN], [PERM_AUC_SD, PERM_AUPRC_SD], COL_PERM, 0.18),
    }

    for name, (vals, errs, col, off) in data.items():
        ax.errorbar(x + off, vals, yerr=errs, fmt="o", markersize=6,
                    color=col, ecolor=col, capsize=3, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.56, 0.84)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)

# -----------------------------
# Draw figure
# -----------------------------
fig = plt.figure(figsize=(14.2, 8.6))
gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.58)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2]),
]

panel_a(axes[0])
panel_b(axes[1])
panel_c(axes[2])
panel_d(axes[3])
panel_e(axes[4])
panel_f(axes[5])

fig.suptitle(
    "Topology-aligned but connectivity-sensitive PPI graph augmentation",
    fontsize=13,
    fontweight="bold",
    y=0.985,
)

png = OUT_DIR / "Figure1_connectivity_v2.png"
pdf = OUT_DIR / "Figure1_connectivity_v2.pdf"
fig.savefig(png, bbox_inches="tight", dpi=300)
fig.savefig(pdf, bbox_inches="tight")

print("Saved:", png)
print("Saved:", pdf)
