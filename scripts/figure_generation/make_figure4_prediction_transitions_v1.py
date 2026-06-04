from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOCAL_ROOT = Path("/Users/teyian/Documents/paper1/server_snapshot_full/root")
LOCAL_OUT = Path("/Users/teyian/Documents/paper1/figure_drafts")
SERVER_ROOT = Path("/root")
SERVER_OUT = Path("/root/paper_assets_final")

ROOT = Path(os.environ.get("PAPER1_ROOT", SERVER_ROOT if (SERVER_ROOT / "data/processed").exists() else LOCAL_ROOT))
OUT_DIR = Path(os.environ.get("PAPER1_FIG_OUT", SERVER_OUT if ROOT == SERVER_ROOT else LOCAL_OUT))
OUT_DIR.mkdir(parents=True, exist_ok=True)
P = ROOT / "data/processed"

plt.rcParams.update({
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

COL_CORRECT = "#9ECAE1"
COL_WRONG = "#BDBDBD"
COL_IMPROVED = "#3182BD"
COL_DEGRADED = "#D95F0E"
COL_FN = "#2171B5"
COL_FP = "#F16913"
COL_GRAY = "#7A7A7A"
COL_TEXT = "#252525"

FEATURES = [
    "degree_bucket",
    "kcore_bucket",
    "neighbor_risk_bucket",
    "neighbor_pathogenic_rate",
    "neighbor_train_variant_count",
    "betweenness_bucket",
    "clustering_coefficient",
    "boundary_role_bucket",
    "within_comm_neighbor_fraction",
]

FEATURE_LABELS = {
    "degree_bucket": "degree",
    "kcore_bucket": "k-core",
    "neighbor_risk_bucket": "neighbour risk",
    "neighbor_pathogenic_rate": "neighbour pathogenic rate",
    "neighbor_train_variant_count": "neighbour train variants",
    "betweenness_bucket": "betweenness",
    "clustering_coefficient": "clustering coefficient",
    "boundary_role_bucket": "boundary role",
    "within_comm_neighbor_fraction": "within-community fraction",
}


def load_json(rel):
    with open(P / rel) as f:
        return json.load(f)


def read_csv(rel):
    return pd.read_csv(P / rel)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


summary = load_json("corrected_sample_identification_exp34/summary.json")
esm_profile = read_csv("corrected_sample_profile_exp35/esm_to_gcn_corrected_profile_summary.csv")
full_profile = read_csv("corrected_sample_profile_exp35/cadd_am_to_full_corrected_profile_summary.csv")
esm_stats = read_csv("fp_vs_fn_stats_exp36B/esm_to_gcn_fp_vs_fn_stats.csv")
full_stats = read_csv("fp_vs_fn_stats_exp36B/cadd_am_to_full_fp_vs_fn_stats.csv")
difficulty = read_csv("difficulty_matched_correction_contrast_exp40/difficulty_matched_correction_contrast_results.csv")


def transition_panel(ax, title, key):
    ax.set_title(title, loc="left", fontweight="bold")
    data = summary[key]
    total = data["n"]
    segments = [
        ("both correct", data["count_both_correct"], COL_CORRECT),
        ("both wrong", data["count_both_wrong"], COL_WRONG),
        ("corrected", data["count_improved"], COL_IMPROVED),
        ("degraded", data["count_degraded"], COL_DEGRADED),
    ]
    left = 0
    for label, count, color in segments:
        ax.barh([0], [count / total], left=left, height=0.34, color=color,
                edgecolor="black", linewidth=0.35, label=label)
        if count / total > 0.045:
            ax.text(left + count / total / 2, 0, f"{count:,}", ha="center", va="center",
                    fontsize=7.0, color=COL_TEXT)
        left += count / total
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.42, 0.62)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of test variants")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04))

    detail = [
        ("FN rescued", data["count_FN_rescued"], COL_FN),
        ("FP suppressed", data["count_FP_suppressed"], COL_FP),
        ("TN to FP", data["count_TN_to_FP"], "#FDAE6B"),
        ("TP to FN", data["count_TP_to_FN"], "#FDD0A2"),
    ]
    x0 = 0.02
    for i, (label, count, color) in enumerate(detail):
        ax.scatter(x0 + i * 0.24, -0.25, s=36, color=color, edgecolor="black", linewidth=0.3)
        ax.text(x0 + i * 0.24 + 0.025, -0.25, f"{label}\n{count:,}",
                ha="left", va="center", fontsize=6.7)
    clean_axis(ax)


def row_scaled_matrix(profile_df):
    rows = profile_df[(profile_df["group_type"] == "error_transition") &
                      (profile_df["group_name"].isin(["FP_suppressed", "FN_rescued"]))].copy()
    rows = rows.set_index("group_name")
    mat = []
    for feat in FEATURES:
        vals = np.array([rows.loc["FP_suppressed", f"mean_{feat}"],
                         rows.loc["FN_rescued", f"mean_{feat}"]], dtype=float)
        if np.isclose(vals.max(), vals.min()):
            scaled = np.zeros_like(vals)
        else:
            scaled = (vals - vals.mean()) / (vals.max() - vals.min())
        mat.append(scaled)
    return np.array(mat)


def heatmap_panel(ax, title, profile_df):
    ax.set_title(title, loc="left", fontweight="bold")
    mat = row_scaled_matrix(profile_df)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FP suppressed", "FN rescued"], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(FEATURES)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.tick_params(length=0)
    for i in range(len(FEATURES)):
        for j in range(2):
            ax.text(j, i, "higher" if mat[i, j] > 0 else "lower",
                    ha="center", va="center", fontsize=5.9, color=COL_TEXT)
    ax.text(1.02, 1.04, "row-normalized means", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color=COL_GRAY)
    return im


def stats_dot_panel(ax):
    ax.set_title("E  Corrected groups differ across topology features", loc="left", fontweight="bold")
    stats = []
    for name, df in [("ESM to PPI-GCN", esm_stats), ("CADD+AM to graph fusion", full_stats)]:
        sub = df.set_index("feature").loc[FEATURES].reset_index()
        for _, row in sub.iterrows():
            stats.append({
                "comparison": name,
                "feature": row["feature"],
                "minus_log10_fdr": min(-np.log10(max(row["fdr_bh"], 1e-300)), 25),
                "significant": bool(row["significant_fdr_0_05"]),
                "fn_higher": row["diff_fp_minus_fn"] < 0,
            })
    s = pd.DataFrame(stats)
    x_map = {"ESM to PPI-GCN": 0, "CADD+AM to graph fusion": 1}
    y_map = {feat: i for i, feat in enumerate(FEATURES[::-1])}
    for _, row in s.iterrows():
        x = x_map[row["comparison"]]
        y = y_map[row["feature"]]
        color = COL_FN if row["fn_higher"] else COL_FP
        size = 22 + row["minus_log10_fdr"] * 4.2
        alpha = 0.85 if row["significant"] else 0.28
        ax.scatter(x, y, s=size, color=color, alpha=alpha, edgecolor="black", linewidth=0.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ESM to\nPPI-GCN", "CADD+AM to\ngraph fusion"])
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURES[::-1]])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.7, len(FEATURES) - 0.3)
    ax.text(1.50, len(FEATURES) - 0.45, "size: -log10 FDR", ha="right", va="top", fontsize=6.8, color=COL_GRAY)
    ax.scatter([], [], s=42, color=COL_FN, label="FN rescued higher")
    ax.scatter([], [], s=42, color=COL_FP, label="FP suppressed higher")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)


def difficulty_panel(ax):
    ax.set_title("F  Difficulty-matched contrasts retain topology differences", loc="left", fontweight="bold")
    keep = [
        "degree_bucket",
        "kcore_bucket",
        "neighbor_risk_bucket",
        "neighbor_pathogenic_rate",
        "betweenness_bucket",
        "boundary_role_bucket",
        "within_comm_neighbor_fraction",
    ]
    df = difficulty[(difficulty["condition"] == "FN_rescued_vs_persistent_FN") &
                    (difficulty["feature"].isin(keep))].copy()
    df["feature"] = pd.Categorical(df["feature"], keep, ordered=True)
    df = df.sort_values("feature")
    y = np.arange(len(df))[::-1]
    colors = [COL_FN if v > 0 else COL_DEGRADED for v in df["hedges_g"]]
    ax.barh(y, df["hedges_g"], color=colors, edgecolor="black", linewidth=0.35)
    ax.axvline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in df["feature"]])
    ax.set_xlabel("Hedges' g\nrescued FN vs persistent FN")
    ax.set_xlim(-1.05, 1.05)
    for yi, sig in zip(y, df["significant_fdr_0_05"]):
        if sig:
            ax.text(1.0, yi, "*", ha="center", va="center", fontsize=9.0, color=COL_TEXT)
    clean_axis(ax)


def main():
    fig, axs = plt.subplots(2, 3, figsize=(13.6, 7.8))
    fig.suptitle("Graph augmentation changes false-negative and false-positive predictions in different network contexts",
                 fontsize=13.5, fontweight="bold", y=0.985)
    transition_panel(axs[0, 0], "A  Raw ESM to PPI-GCN prediction transitions", "esm_vs_gcn")
    transition_panel(axs[0, 1], "B  CADD+AM to graph-fusion transitions", "cadd_am_vs_full")
    heatmap_panel(axs[0, 2], "C  Raw ESM corrections have distinct topology profiles", esm_profile)
    heatmap_panel(axs[1, 0], "D  Graph-fusion corrections show related profile shifts", full_profile)
    stats_dot_panel(axs[1, 1])
    difficulty_panel(axs[1, 2])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.11, wspace=0.55, hspace=0.58)
    png = OUT_DIR / "Figure4_prediction_transitions_v1.png"
    pdf = OUT_DIR / "Figure4_prediction_transitions_v1.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
