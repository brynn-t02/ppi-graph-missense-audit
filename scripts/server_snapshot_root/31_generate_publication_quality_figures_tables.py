import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path("/root")

SRC_DIR = ROOT / "data/processed/paper_figures_tables"
OUT_DIR = ROOT / "data/processed/paper_figures_tables_publication"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

ORIG_TEST = ROOT / "data/processed/clinvar_test_main.csv"
HOMO_TEST = ROOT / "data/processed/homology_aware_split/clinvar_test_homology.csv"

# probability files
PROBS = {
    "esm_raw": ROOT / "data/processed/variant_mlp_comparison/esm_raw/test_probs.npy",
    "gcn_enhanced": ROOT / "data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy",

    "esm_raw_homology": ROOT / "data/processed/homology_rerun_core/esm_raw_homology/test_probs.npy",
    "gcn_enhanced_homology": ROOT / "data/processed/homology_rerun_core/gcn_enhanced_homology/test_probs.npy",

    "cadd_plus_am": ROOT / "data/processed/triple_fusion/cadd_plus_am/test_probs.npy",
    "gcn_plus_cadd_plus_am": ROOT / "data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy",

    "cadd_only_homology": ROOT / "data/processed/homology_single_tool_fusions/cadd_only_homology/test_probs.npy",
    "gcn_plus_cadd_homology": ROOT / "data/processed/homology_single_tool_fusions/gcn_plus_cadd_homology/test_probs.npy",

    "cadd_plus_am_homology": ROOT / "data/processed/homology_rerun_core/cadd_plus_am_homology/test_probs.npy",
    "gcn_plus_cadd_plus_am_homology": ROOT / "data/processed/homology_rerun_core/gcn_plus_cadd_plus_am_homology/test_probs.npy",
}

# source tables
TABLE1_SRC = SRC_DIR / "tables/Table1_split_statistics.csv"
TABLE2_SRC = SRC_DIR / "tables/Table2_overall_performance_summary.csv"
TABLE3_SRC = SRC_DIR / "tables/Table3_bootstrap_significance_summary.csv"
TABLE4_SRC = SRC_DIR / "tables/Table4_clinical_metrics_summary.csv"
TABLE5_SRC = SRC_DIR / "tables/Table5_mechanism_controls_summary.csv"

# clinical raw sources
ORIG_DCA_FOCUS = ROOT / "data/processed/clinical_plots/dca_focus_delta_table.csv"
ORIG_CLINICAL_SUMMARY = ROOT / "data/processed/clinical_metrics/clinical_metrics_summary.csv"
HOMO_CLINICAL_SUMMARY = ROOT / "data/processed/homology_clinical_metrics/homology_clinical_metrics_summary.csv"

# plot style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#4C78A8"
RED = "#E45756"
GREEN = "#54A24B"
ORANGE = "#F58518"
PURPLE = "#B279A2"
GRAY = "#9D9D9D"
BLACK = "#222222"

MODEL_LABELS = {
    "esm_raw": "ESM",
    "gcn_enhanced": "GCN",
    "esm_raw_homology": "ESM (homology)",
    "gcn_enhanced_homology": "GCN (homology)",

    "cadd_only": "CADD",
    "gcn_plus_cadd": "GCN+CADD",
    "am_only": "AM",
    "gcn_plus_am": "GCN+AM",
    "cadd_plus_am": "CADD+AM",
    "gcn_plus_cadd_plus_am": "GCN+CADD+AM",

    "cadd_only_homology": "CADD",
    "gcn_plus_cadd_homology": "GCN+CADD",
    "am_only_homology": "AM",
    "gcn_plus_am_homology": "GCN+AM",
    "cadd_plus_am_homology": "CADD+AM",
    "gcn_plus_cadd_plus_am_homology": "GCN+CADD+AM",

    "degree_only": "Degree-only",
    "feature_shuffled_gcn": "Feature-shuffled GCN",
    "label_shuffled_gcn_mean": "Label-shuffled GCN",
    "random_raw_feature_baseline": "Random raw",
    "random_feature_gcn": "Random+GCN",
    "true_gcn": "True GCN",
    "random_raw_feature_homology": "Random raw (homology)",
    "random_feature_gcn_homology": "Random+GCN (homology)",
    "true_gcn_homology": "True GCN (homology)",
}

print("=" * 100)
print("🖼️ Step 31: Generate publication-quality figures and tables")
print("=" * 100)
print("Source dir:", SRC_DIR)
print("Output dir:", OUT_DIR)

def load_probs(path, expected_n):
    arr = np.load(path)
    if len(arr) != expected_n:
        raise ValueError(f"Length mismatch for {path}: got {len(arr)} expected {expected_n}")
    return arr

def save_fig(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def prettify_table(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    return out

def human_comparison(name):
    mapping = {
        "gcn_vs_esm_homology": "GCN vs ESM (homology)",
        "gcn_cadd_vs_cadd_homology": "GCN+CADD vs CADD (homology)",
        "gcn_am_vs_am_homology": "GCN+AM vs AM (homology)",
        "full_vs_cadd_am_homology": "GCN+CADD+AM vs CADD+AM (homology)",
    }
    return mapping.get(name, name)

# -------------------------------------------------------------------
# 1. Load labels
# -------------------------------------------------------------------
orig_test_df = pd.read_csv(ORIG_TEST, low_memory=False)
homo_test_df = pd.read_csv(HOMO_TEST, low_memory=False)
y_orig = orig_test_df["Label"].astype(int).values
y_homo = homo_test_df["Label"].astype(int).values

print("\n[1] Loaded labels")
print("Original n :", len(y_orig))
print("Homology n :", len(y_homo))

# -------------------------------------------------------------------
# 2. Publication Figure 2: Representation ROC/PR
# -------------------------------------------------------------------
orig_esm = load_probs(PROBS["esm_raw"], len(y_orig))
orig_gcn = load_probs(PROBS["gcn_enhanced"], len(y_orig))
homo_esm = load_probs(PROBS["esm_raw_homology"], len(y_homo))
homo_gcn = load_probs(PROBS["gcn_enhanced_homology"], len(y_homo))

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))

# ROC original
for label, probs, color in [
    ("ESM", orig_esm, BLUE),
    ("GCN", orig_gcn, RED),
]:
    fpr, tpr, _ = roc_curve(y_orig, probs)
    auc = roc_auc_score(y_orig, probs)
    axes[0, 0].plot(fpr, tpr, lw=2.3, color=color, label=f"{label} (AUC={auc:.3f})")
axes[0, 0].plot([0, 1], [0, 1], "--", lw=1.2, color=GRAY)
axes[0, 0].set_title("Original split ROC")
axes[0, 0].set_xlabel("False positive rate")
axes[0, 0].set_ylabel("True positive rate")
axes[0, 0].legend(frameon=False)

# PR original
for label, probs, color in [
    ("ESM", orig_esm, BLUE),
    ("GCN", orig_gcn, RED),
]:
    precision, recall, _ = precision_recall_curve(y_orig, probs)
    ap = average_precision_score(y_orig, probs)
    axes[0, 1].plot(recall, precision, lw=2.3, color=color, label=f"{label} (AP={ap:.3f})")
axes[0, 1].set_title("Original split PR")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].legend(frameon=False)

# ROC homology
for label, probs, color in [
    ("ESM (homology)", homo_esm, BLUE),
    ("GCN (homology)", homo_gcn, RED),
]:
    fpr, tpr, _ = roc_curve(y_homo, probs)
    auc = roc_auc_score(y_homo, probs)
    axes[1, 0].plot(fpr, tpr, lw=2.3, color=color, label=f"{label} (AUC={auc:.3f})")
axes[1, 0].plot([0, 1], [0, 1], "--", lw=1.2, color=GRAY)
axes[1, 0].set_title("Homology-aware split ROC")
axes[1, 0].set_xlabel("False positive rate")
axes[1, 0].set_ylabel("True positive rate")
axes[1, 0].legend(frameon=False)

# PR homology
for label, probs, color in [
    ("ESM (homology)", homo_esm, BLUE),
    ("GCN (homology)", homo_gcn, RED),
]:
    precision, recall, _ = precision_recall_curve(y_homo, probs)
    ap = average_precision_score(y_homo, probs)
    axes[1, 1].plot(recall, precision, lw=2.3, color=color, label=f"{label} (AP={ap:.3f})")
axes[1, 1].set_title("Homology-aware split PR")
axes[1, 1].set_xlabel("Recall")
axes[1, 1].set_ylabel("Precision")
axes[1, 1].legend(frameon=False)

fig.suptitle("Graph-enhanced representations improve over raw ESM under both split settings", y=1.02, fontsize=14)
fig.tight_layout()
save_fig(fig, "Figure2_representation_roc_pr_publication")

# -------------------------------------------------------------------
# 3. Publication Figure 3: Fusion performance grouped panels
# -------------------------------------------------------------------
table2 = pd.read_csv(TABLE2_SRC)

orig_models = ["cadd_only", "gcn_plus_cadd", "am_only", "gcn_plus_am", "cadd_plus_am", "gcn_plus_cadd_plus_am"]
homo_models = ["cadd_only_homology", "gcn_plus_cadd_homology", "am_only_homology", "gcn_plus_am_homology", "cadd_plus_am_homology", "gcn_plus_cadd_plus_am_homology"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="col")

for ax_row, models, split_title in zip(
    axes,
    [orig_models, homo_models],
    ["Original protein-level split", "Homology-aware split"]
):
    sub = table2[table2["model"].isin(models)].copy()
    sub["short"] = sub["model"].map(MODEL_LABELS)

    # AUC
    ax = ax_row[0]
    ax.bar(np.arange(len(sub)), sub["test_auc"], color=[BLUE if "gcn" not in m else RED for m in sub["model"]], edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["short"], rotation=25, ha="right")
    ax.set_ylim(0.88 if "Original" in split_title else 0.93, 0.99)
    ax.set_ylabel("Test AUC")
    ax.set_title(f"{split_title}: AUC")

    # AUPRC
    ax = ax_row[1]
    ax.bar(np.arange(len(sub)), sub["test_auprc"], color=[BLUE if "gcn" not in m else RED for m in sub["model"]], edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["short"], rotation=25, ha="right")
    ax.set_ylabel("Test AUPRC")
    ax.set_title(f"{split_title}: AUPRC")

fig.tight_layout()
save_fig(fig, "Figure3_fusion_performance_publication")

# -------------------------------------------------------------------
# 4. Publication Figure 4: Mechanism controls
# -------------------------------------------------------------------
table5 = pd.read_csv(TABLE5_SRC)

order = [
    "degree_only",
    "feature_shuffled_gcn",
    "label_shuffled_gcn_mean",
    "random_raw_feature_baseline",
    "random_feature_gcn",
    "true_gcn",
    "random_raw_feature_homology",
    "random_feature_gcn_homology",
    "true_gcn_homology",
]
table5["order"] = table5["model"].map({m: i for i, m in enumerate(order)})
table5 = table5.sort_values("order").copy()
table5["short"] = table5["model"].map(MODEL_LABELS)

colors = []
for m in table5["model"]:
    if "homology" in m:
        colors.append(ORANGE)
    elif "random" in m:
        colors.append(PURPLE)
    elif "degree" in m or "shuffle" in m or "label" in m:
        colors.append(GREEN)
    else:
        colors.append(RED)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))

axes[0].bar(np.arange(len(table5)), table5["test_auc"], color=colors, edgecolor="black", linewidth=0.3)
axes[0].set_xticks(np.arange(len(table5)))
axes[0].set_xticklabels(table5["short"], rotation=30, ha="right")
axes[0].set_ylabel("Test AUC")
axes[0].set_title("Mechanistic controls")

axes[1].bar(np.arange(len(table5)), table5["test_auprc"], color=colors, edgecolor="black", linewidth=0.3)
axes[1].set_xticks(np.arange(len(table5)))
axes[1].set_xticklabels(table5["short"], rotation=30, ha="right")
axes[1].set_ylabel("Test AUPRC")
axes[1].set_title("Mechanistic controls")

fig.tight_layout()
save_fig(fig, "Figure4_mechanism_controls_publication")

# -------------------------------------------------------------------
# 5. Publication Figure 5: DCA + calibration focused
# -------------------------------------------------------------------
# DCA focus
dca_delta = pd.read_csv(ORIG_DCA_FOCUS)
dca_full = pd.read_csv(ROOT / "data/processed/clinical_plots/dca_focus_delta_table.csv")

caddam_dca = pd.read_csv(ROOT / "data/processed/clinical_metrics/cadd_plus_am_dca.csv")
full_dca = pd.read_csv(ROOT / "data/processed/clinical_metrics/gcn_plus_cadd_plus_am_dca.csv")

caddam_ece = pd.read_csv(ROOT / "data/processed/clinical_metrics/cadd_plus_am_ece_bins.csv")
full_ece = pd.read_csv(ROOT / "data/processed/clinical_metrics/gcn_plus_cadd_plus_am_ece_bins.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# DCA
axes[0].plot(caddam_dca["threshold"], caddam_dca["net_benefit_model"], lw=2.4, color=BLUE, label="CADD+AM")
axes[0].plot(full_dca["threshold"], full_dca["net_benefit_model"], lw=2.4, color=RED, label="GCN+CADD+AM")
axes[0].plot(caddam_dca["threshold"], caddam_dca["net_benefit_treat_none"], ":", lw=1.5, color=BLACK, label="Treat none")
axes[0].plot(caddam_dca["threshold"], caddam_dca["net_benefit_treat_all"], "--", lw=1.5, color=GRAY, label="Treat all")
axes[0].set_xlim(0.05, 0.5)

# dynamic y limits based on models only
y_min = min(caddam_dca["net_benefit_model"].min(), full_dca["net_benefit_model"].min())
y_max = max(caddam_dca["net_benefit_model"].max(), full_dca["net_benefit_model"].max())
pad = (y_max - y_min) * 0.25 if y_max > y_min else 0.02
axes[0].set_ylim(max(-0.02, y_min - pad), y_max + pad)
axes[0].set_xlabel("Threshold probability")
axes[0].set_ylabel("Net benefit")
axes[0].set_title("Decision-curve analysis")
axes[0].legend(frameon=False, loc="lower left")

# Calibration
axes[1].plot(caddam_ece["mean_confidence"], caddam_ece["observed_frequency"], "-o", lw=2.2, color=BLUE, label="CADD+AM")
axes[1].plot(full_ece["mean_confidence"], full_ece["observed_frequency"], "-o", lw=2.2, color=RED, label="GCN+CADD+AM")
axes[1].plot([0, 1], [0, 1], "--", color=GRAY, lw=1.5, label="Perfect calibration")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Observed event frequency")
axes[1].set_title("Calibration curves")
axes[1].legend(frameon=False, loc="upper left")

fig.tight_layout()
save_fig(fig, "Figure5_dca_calibration_publication")

# -------------------------------------------------------------------
# 6. Publication Figure 6: Homology clinical summary
# -------------------------------------------------------------------
homo_clin = pd.read_csv(HOMO_CLINICAL_SUMMARY)
wanted = [
    "cadd_only_homology",
    "gcn_plus_cadd_homology",
    "am_only_homology",
    "gcn_plus_am_homology",
    "cadd_plus_am_homology",
    "gcn_plus_cadd_plus_am_homology",
]
homo_clin = homo_clin[homo_clin["model"].isin(wanted)].copy()
homo_clin["short"] = homo_clin["model"].map(MODEL_LABELS)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
metrics = [
    ("brier_score", "Brier score"),
    ("ece_10bin", "ECE"),
    ("sensitivity_at_90_specificity", "Sensitivity @ 90% specificity"),
]

for ax, (col, title) in zip(axes, metrics):
    ax.bar(np.arange(len(homo_clin)), homo_clin[col], color=[BLUE if "gcn" not in m else RED for m in homo_clin["model"]], edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(homo_clin)))
    ax.set_xticklabels(homo_clin["short"], rotation=25, ha="right")
    ax.set_title(title)

fig.tight_layout()
save_fig(fig, "Figure6_homology_clinical_metrics_publication")

# -------------------------------------------------------------------
# 7. Publication Table 1
# -------------------------------------------------------------------
table1 = pd.read_csv(TABLE1_SRC)
table1_pub = table1.copy()
table1_pub["split_type"] = table1_pub["split_type"].replace({
    "original_protein_level": "Original protein-level",
    "homology_aware": "Homology-aware",
})
table1_pub["subset"] = table1_pub["subset"].str.capitalize()
table1_pub = prettify_table(table1_pub)
table1_pub.to_csv(TAB_DIR / "Table1_split_statistics_publication.csv", index=False)

# -------------------------------------------------------------------
# 8. Publication Table 2
# -------------------------------------------------------------------
table2_pub = table2.copy()
table2_pub["split"] = table2_pub["split"].replace({"original": "Original", "homology": "Homology-aware"})
table2_pub["model_display"] = table2_pub["model"].map(MODEL_LABELS).fillna(table2_pub["model"])
table2_pub = table2_pub[["split", "model_display", "test_auc", "test_auprc"]].rename(columns={
    "split": "Split",
    "model_display": "Model",
    "test_auc": "Test AUC",
    "test_auprc": "Test AUPRC",
})
table2_pub = prettify_table(table2_pub)
table2_pub.to_csv(TAB_DIR / "Table2_overall_performance_publication.csv", index=False)

# -------------------------------------------------------------------
# 9. Publication Table 3
# -------------------------------------------------------------------
table3 = pd.read_csv(TABLE3_SRC)
table3_pub = table3.copy()

if "comparison" in table3_pub.columns:
    table3_pub["Comparison"] = table3_pub["comparison"].map(human_comparison).fillna(table3_pub["comparison"])
else:
    table3_pub["Comparison"] = np.nan

metric_map = {
    "delta_auc": "ΔAUC",
    "delta_auprc": "ΔAUPRC",
}
table3_pub["Metric"] = table3_pub["metric"].map(metric_map).fillna(table3_pub["metric"])
table3_pub["Mean Δ"] = table3_pub["mean_delta"].round(4)
table3_pub["95% CI"] = table3_pub.apply(lambda r: f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]", axis=1)
table3_pub["p-value"] = table3_pub["p_value_bootstrap_sign"].map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
table3_pub["Split"] = table3_pub["split"].replace({"original": "Original", "homology": "Homology-aware"})

table3_pub = table3_pub[["Split", "Comparison", "Metric", "Mean Δ", "95% CI", "p-value"]]
table3_pub.to_csv(TAB_DIR / "Table3_bootstrap_significance_publication.csv", index=False)

# -------------------------------------------------------------------
# 10. Publication Table 4
# -------------------------------------------------------------------
table4 = pd.read_csv(TABLE4_SRC)
table4_pub = table4.copy()
table4_pub["Split"] = table4_pub["split"].replace({"original": "Original", "homology": "Homology-aware"})
table4_pub["Model"] = table4_pub["model"].map(MODEL_LABELS).fillna(table4_pub["model"])
table4_pub = table4_pub[[
    "Split", "Model", "brier_score", "ece_10bin",
    "sensitivity_at_90_specificity", "selected_threshold"
]].rename(columns={
    "brier_score": "Brier",
    "ece_10bin": "ECE",
    "sensitivity_at_90_specificity": "Sensitivity @ 90% specificity",
    "selected_threshold": "Threshold",
})
table4_pub = prettify_table(table4_pub)
table4_pub.to_csv(TAB_DIR / "Table4_clinical_metrics_publication.csv", index=False)

# -------------------------------------------------------------------
# 11. Publication Table 5
# -------------------------------------------------------------------
table5_pub = table5.copy()
table5_pub["Split"] = table5_pub["split"].replace({"original": "Original", "homology": "Homology-aware"})
table5_pub["Model"] = table5_pub["model"].map(MODEL_LABELS).fillna(table5_pub["model"])
table5_pub = table5_pub[["Split", "Model", "test_auc", "test_auprc"]].rename(columns={
    "test_auc": "Test AUC",
    "test_auprc": "Test AUPRC",
})
table5_pub = prettify_table(table5_pub)
table5_pub.to_csv(TAB_DIR / "Table5_mechanism_controls_publication.csv", index=False)

# -------------------------------------------------------------------
# 12. Copy supplementary files
# -------------------------------------------------------------------
for src_name in [
    "Supplementary_pr_curve_auc_summary.csv",
    "Supplementary_split_prevalence_summary.csv",
    "Supplementary_q1_case_study_summary.json",
]:
    src = SRC_DIR / "tables" / src_name
    if src.exists():
        shutil.copy2(src, TAB_DIR / src_name)

# save a brief manifest
manifest = {
    "figures": sorted([p.name for p in FIG_DIR.iterdir()]),
    "tables": sorted([p.name for p in TAB_DIR.iterdir()]),
}
with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\n[2] Saved publication-quality figures")
for p in sorted(FIG_DIR.iterdir()):
    print("✅", p)

print("\n[3] Saved publication-quality tables")
for p in sorted(TAB_DIR.iterdir()):
    print("✅", p)

print("\n✅ Manifest:", OUT_DIR / "manifest.json")
print("=" * 100)
print("Done.")
print("=" * 100)
