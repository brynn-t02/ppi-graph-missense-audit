import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path("/root")

OUT_DIR = ROOT / "data/processed/paper_figures_tables"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ORIG_TEST = ROOT / "data/processed/clinvar_test_main.csv"
HOMO_TEST = ROOT / "data/processed/homology_aware_split/clinvar_test_homology.csv"

FULL_SUMMARY_JSON = ROOT / "data/processed/full_summary/full_results_summary_with_triple.json"
FULL_SUMMARY_CSV = ROOT / "data/processed/full_summary/full_results_summary_with_triple.csv"

ORIG_BOOTSTRAP_JSON = ROOT / "data/processed/bootstrap_significance/bootstrap_significance_summary.json"
ORIG_BOOTSTRAP_CSV  = ROOT / "data/processed/bootstrap_significance/bootstrap_significance_summary.csv"

HOMO_BOOTSTRAP_JSON = ROOT / "data/processed/homology_bootstrap_significance/homology_bootstrap_significance_summary.json"
HOMO_BOOTSTRAP_CSV  = ROOT / "data/processed/homology_bootstrap_significance/homology_bootstrap_significance_summary.csv"

HOMO_SPLIT_SUMMARY_JSON = ROOT / "data/processed/homology_aware_split/homology_split_summary.json"
ORIG_CLINICAL_CSV = ROOT / "data/processed/clinical_metrics/clinical_metrics_summary.csv"
HOMO_CLINICAL_CSV = ROOT / "data/processed/homology_clinical_metrics/homology_clinical_metrics_summary.csv"

PR_SUMMARY_CSV = ROOT / "data/processed/pr_curve_analysis/pr_curve_auc_summary.csv"
PREVALENCE_CSV = ROOT / "data/processed/pr_curve_analysis/split_prevalence_summary.csv"

# probability files
PROBS = {
    "esm_raw": ROOT / "data/processed/variant_mlp_comparison/esm_raw/test_probs.npy",
    "gcn_enhanced": ROOT / "data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy",
    "esm_raw_homology": ROOT / "data/processed/homology_rerun_core/esm_raw_homology/test_probs.npy",
    "gcn_enhanced_homology": ROOT / "data/processed/homology_rerun_core/gcn_enhanced_homology/test_probs.npy",

    "cadd_only_homology": ROOT / "data/processed/homology_single_tool_fusions/cadd_only_homology/test_probs.npy",
    "gcn_plus_cadd_homology": ROOT / "data/processed/homology_single_tool_fusions/gcn_plus_cadd_homology/test_probs.npy",

    "cadd_plus_am": ROOT / "data/processed/triple_fusion/cadd_plus_am/test_probs.npy",
    "gcn_plus_cadd_plus_am": ROOT / "data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy",
    "cadd_plus_am_homology": ROOT / "data/processed/homology_rerun_core/cadd_plus_am_homology/test_probs.npy",
    "gcn_plus_cadd_plus_am_homology": ROOT / "data/processed/homology_rerun_core/gcn_plus_cadd_plus_am_homology/test_probs.npy",
}

print("=" * 90)
print("📚 Step 30: Generate paper figures and tables")
print("=" * 90)
print("Output dir:", OUT_DIR)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def ensure_copy(src, dst):
    if src.exists():
        shutil.copy2(src, dst)

def safe_load_probs(path, expected_n):
    if not path.exists():
        return None
    arr = np.load(path)
    if len(arr) != expected_n:
        raise ValueError(f"Length mismatch for {path}: got {len(arr)} expected {expected_n}")
    return arr

def metric_from_probs(y, p):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    return auc, ap

# ---------------------------------------------------------------------
# 1. Load labels
# ---------------------------------------------------------------------
orig_test_df = pd.read_csv(ORIG_TEST, low_memory=False)
homo_test_df = pd.read_csv(HOMO_TEST, low_memory=False)

y_orig = orig_test_df["Label"].astype(int).values
y_homo = homo_test_df["Label"].astype(int).values

print("\n[1] Loaded test labels")
print("Original test n:", len(y_orig))
print("Homology test n:", len(y_homo))

# ---------------------------------------------------------------------
# 2. Table 1: split statistics
# ---------------------------------------------------------------------
orig_stats = pd.DataFrame([
    {
        "split_type": "original_protein_level",
        "subset": "train",
        "n_variants": 141864,
        "n_proteins": 11298,
        "n_clusters": np.nan,
        "cluster_size_mean": np.nan,
        "cluster_size_max": np.nan,
        "mmseqs_min_seq_id": np.nan,
        "mmseqs_coverage": np.nan,
    },
    {
        "split_type": "original_protein_level",
        "subset": "val",
        "n_variants": 17129,
        "n_proteins": 1412,
        "n_clusters": np.nan,
        "cluster_size_mean": np.nan,
        "cluster_size_max": np.nan,
        "mmseqs_min_seq_id": np.nan,
        "mmseqs_coverage": np.nan,
    },
    {
        "split_type": "original_protein_level",
        "subset": "test",
        "n_variants": 18220,
        "n_proteins": 1413,
        "n_clusters": np.nan,
        "cluster_size_mean": np.nan,
        "cluster_size_max": np.nan,
        "mmseqs_min_seq_id": np.nan,
        "mmseqs_coverage": np.nan,
    },
])

homo_summary = load_json(HOMO_SPLIT_SUMMARY_JSON)
homo_rows = []
if homo_summary is not None:
    for subset in ["train", "val", "test"]:
        homo_rows.append({
            "split_type": "homology_aware",
            "subset": subset,
            "n_variants": homo_summary["assigned_row_counts"][subset],
            "n_proteins": homo_summary["assigned_unique_proteins"][subset],
            "n_clusters": homo_summary["n_clusters"],
            "cluster_size_mean": homo_summary["cluster_size_mean"],
            "cluster_size_max": homo_summary["cluster_size_max"],
            "mmseqs_min_seq_id": homo_summary["mmseqs_cluster_min_seq_id"],
            "mmseqs_coverage": homo_summary["mmseqs_cluster_coverage"],
        })

table1 = pd.concat([orig_stats, pd.DataFrame(homo_rows)], ignore_index=True)
table1.to_csv(TAB_DIR / "Table1_split_statistics.csv", index=False)

print("\n[2] Saved Table 1")
print("✅", TAB_DIR / "Table1_split_statistics.csv")

# ---------------------------------------------------------------------
# 3. Table 2: overall performance summary
# ---------------------------------------------------------------------
rows = []

# original performance (manually consolidated from validated results)
orig_perf = [
    ("original", "esm_raw", 0.7997 if False else 0.7790, 0.6496 if False else 0.6075),
    ("original", "gcn_enhanced", 0.7997, 0.6496),
    ("original", "cadd_only", 0.9207, 0.8049),
    ("original", "gcn_plus_cadd", 0.9413, 0.8728),
    ("original", "am_only", 0.9468, 0.8896),
    ("original", "gcn_plus_am", 0.9616, 0.9295),
    ("original", "cadd_plus_am", 0.9575, 0.9087),
    ("original", "gcn_plus_cadd_plus_am", 0.9708, 0.9397),
]

for split, model, auc, ap in orig_perf:
    rows.append({"split": split, "model": model, "test_auc": auc, "test_auprc": ap})

# homology performance from summaries
homo_rerun = load_json(ROOT / "data/processed/homology_rerun_core/homology_rerun_summary.json")
homo_single = load_json(ROOT / "data/processed/homology_single_tool_fusions/summary.json")

if homo_rerun is not None:
    rows.extend([
        {"split": "homology", "model": "esm_raw_homology",
         "test_auc": homo_rerun["esm_raw_homology"]["test_auc"],
         "test_auprc": homo_rerun["esm_raw_homology"]["test_auprc"]},
        {"split": "homology", "model": "gcn_enhanced_homology",
         "test_auc": homo_rerun["gcn_enhanced_homology"]["test_auc"],
         "test_auprc": homo_rerun["gcn_enhanced_homology"]["test_auprc"]},
        {"split": "homology", "model": "cadd_plus_am_homology",
         "test_auc": homo_rerun["cadd_plus_am_homology"]["test_auc"],
         "test_auprc": homo_rerun["cadd_plus_am_homology"]["test_auprc"]},
        {"split": "homology", "model": "gcn_plus_cadd_plus_am_homology",
         "test_auc": homo_rerun["gcn_plus_cadd_plus_am_homology"]["test_auc"],
         "test_auprc": homo_rerun["gcn_plus_cadd_plus_am_homology"]["test_auprc"]},
    ])

if homo_single is not None:
    rows.extend([
        {"split": "homology", "model": "cadd_only_homology",
         "test_auc": homo_single["cadd_only_homology"]["test_auc"],
         "test_auprc": homo_single["cadd_only_homology"]["test_auprc"]},
        {"split": "homology", "model": "gcn_plus_cadd_homology",
         "test_auc": homo_single["gcn_plus_cadd_homology"]["test_auc"],
         "test_auprc": homo_single["gcn_plus_cadd_homology"]["test_auprc"]},
        {"split": "homology", "model": "am_only_homology",
         "test_auc": homo_single["am_only_homology"]["test_auc"],
         "test_auprc": homo_single["am_only_homology"]["test_auprc"]},
        {"split": "homology", "model": "gcn_plus_am_homology",
         "test_auc": homo_single["gcn_plus_am_homology"]["test_auc"],
         "test_auprc": homo_single["gcn_plus_am_homology"]["test_auprc"]},
    ])

table2 = pd.DataFrame(rows)
table2.to_csv(TAB_DIR / "Table2_overall_performance_summary.csv", index=False)

print("\n[3] Saved Table 2")
print("✅", TAB_DIR / "Table2_overall_performance_summary.csv")

# ---------------------------------------------------------------------
# 4. Table 3: bootstrap significance summary
# ---------------------------------------------------------------------
orig_boot = pd.read_csv(ORIG_BOOTSTRAP_CSV) if ORIG_BOOTSTRAP_CSV.exists() else pd.DataFrame()
homo_boot = pd.read_csv(HOMO_BOOTSTRAP_CSV) if HOMO_BOOTSTRAP_CSV.exists() else pd.DataFrame()

if len(orig_boot) > 0:
    orig_boot["split"] = "original"
if len(homo_boot) > 0:
    homo_boot["split"] = "homology"

table3 = pd.concat([orig_boot, homo_boot], ignore_index=True)
table3.to_csv(TAB_DIR / "Table3_bootstrap_significance_summary.csv", index=False)

print("\n[4] Saved Table 3")
print("✅", TAB_DIR / "Table3_bootstrap_significance_summary.csv")

# ---------------------------------------------------------------------
# 5. Table 4: clinical metrics summary
# ---------------------------------------------------------------------
orig_clin = pd.read_csv(ORIG_CLINICAL_CSV) if ORIG_CLINICAL_CSV.exists() else pd.DataFrame()
homo_clin = pd.read_csv(HOMO_CLINICAL_CSV) if HOMO_CLINICAL_CSV.exists() else pd.DataFrame()

if len(orig_clin) > 0:
    orig_clin["split"] = "original"
if len(homo_clin) > 0:
    homo_clin["split"] = "homology"

table4 = pd.concat([orig_clin, homo_clin], ignore_index=True)
table4.to_csv(TAB_DIR / "Table4_clinical_metrics_summary.csv", index=False)

print("\n[5] Saved Table 4")
print("✅", TAB_DIR / "Table4_clinical_metrics_summary.csv")

# ---------------------------------------------------------------------
# 6. Table 5: mechanism controls summary
# ---------------------------------------------------------------------
table5_rows = [
    {"split": "original", "model": "degree_only", "test_auc": 0.637698, "test_auprc": 0.390122},
    {"split": "original", "model": "feature_shuffled_gcn", "test_auc": 0.587998, "test_auprc": 0.371035},
    {"split": "original", "model": "label_shuffled_gcn_mean", "test_auc": 0.6250, "test_auprc": 0.3990},
    {"split": "original", "model": "random_raw_feature_baseline", "test_auc": 0.475027, "test_auprc": 0.333452},
    {"split": "original", "model": "random_feature_gcn", "test_auc": 0.863639, "test_auprc": 0.747921},
    {"split": "original", "model": "true_gcn", "test_auc": 0.7997, "test_auprc": 0.6496},
    {"split": "homology", "model": "random_raw_feature_homology", "test_auc": 0.491753, "test_auprc": 0.103969},
    {"split": "homology", "model": "random_feature_gcn_homology", "test_auc": 0.823051, "test_auprc": 0.382922},
    {"split": "homology", "model": "true_gcn_homology", "test_auc": 0.7606, "test_auprc": 0.2768},
]
table5 = pd.DataFrame(table5_rows)
table5.to_csv(TAB_DIR / "Table5_mechanism_controls_summary.csv", index=False)

print("\n[6] Saved Table 5")
print("✅", TAB_DIR / "Table5_mechanism_controls_summary.csv")

# ---------------------------------------------------------------------
# 7. Figure 2: representation ROC + PR
# ---------------------------------------------------------------------
orig_probs_esm = safe_load_probs(PROBS["esm_raw"], len(y_orig))
orig_probs_gcn = safe_load_probs(PROBS["gcn_enhanced"], len(y_orig))
homo_probs_esm = safe_load_probs(PROBS["esm_raw_homology"], len(y_homo))
homo_probs_gcn = safe_load_probs(PROBS["gcn_enhanced_homology"], len(y_homo))

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# original ROC
for name, probs, color in [
    ("ESM raw", orig_probs_esm, "#4C72B0"),
    ("GCN-enhanced", orig_probs_gcn, "#C44E52"),
]:
    fpr, tpr, _ = roc_curve(y_orig, probs)
    auc = roc_auc_score(y_orig, probs)
    axes[0, 0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2, color=color)
axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
axes[0, 0].set_title("Original split ROC")
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].legend()

# original PR
for name, probs, color in [
    ("ESM raw", orig_probs_esm, "#4C72B0"),
    ("GCN-enhanced", orig_probs_gcn, "#C44E52"),
]:
    precision, recall, _ = precision_recall_curve(y_orig, probs)
    ap = average_precision_score(y_orig, probs)
    axes[0, 1].plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2, color=color)
axes[0, 1].set_title("Original split PR")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].legend()

# homology ROC
for name, probs, color in [
    ("ESM raw (homology)", homo_probs_esm, "#4C72B0"),
    ("GCN-enhanced (homology)", homo_probs_gcn, "#C44E52"),
]:
    fpr, tpr, _ = roc_curve(y_homo, probs)
    auc = roc_auc_score(y_homo, probs)
    axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2, color=color)
axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
axes[1, 0].set_title("Homology-aware split ROC")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend()

# homology PR
for name, probs, color in [
    ("ESM raw (homology)", homo_probs_esm, "#4C72B0"),
    ("GCN-enhanced (homology)", homo_probs_gcn, "#C44E52"),
]:
    precision, recall, _ = precision_recall_curve(y_homo, probs)
    ap = average_precision_score(y_homo, probs)
    axes[1, 1].plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2, color=color)
axes[1, 1].set_title("Homology-aware split PR")
axes[1, 1].set_xlabel("Recall")
axes[1, 1].set_ylabel("Precision")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "Figure2_representation_roc_pr.png", dpi=300)
plt.savefig(FIG_DIR / "Figure2_representation_roc_pr.pdf")
plt.close()

# ---------------------------------------------------------------------
# 8. Figure 3: fusion performance grouped bars
# ---------------------------------------------------------------------
plot_df = table2[table2["model"].isin([
    "cadd_only", "gcn_plus_cadd", "am_only", "gcn_plus_am", "cadd_plus_am", "gcn_plus_cadd_plus_am",
    "cadd_only_homology", "gcn_plus_cadd_homology", "am_only_homology", "gcn_plus_am_homology",
    "cadd_plus_am_homology", "gcn_plus_cadd_plus_am_homology"
])].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric in zip(axes, ["test_auc", "test_auprc"]):
    ax.bar(np.arange(len(plot_df)), plot_df[metric].values, color=["#4C72B0" if s=="original" else "#C44E52" for s in plot_df["split"]])
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["model"], rotation=45, ha="right")
    ax.set_title(metric)
    ax.set_ylabel(metric)
plt.tight_layout()
plt.savefig(FIG_DIR / "Figure3_fusion_performance.png", dpi=300)
plt.savefig(FIG_DIR / "Figure3_fusion_performance.pdf")
plt.close()

# ---------------------------------------------------------------------
# 9. Figure 4: mechanism controls
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(np.arange(len(table5)), table5["test_auc"], color=["#4C72B0" if s=="original" else "#C44E52" for s in table5["split"]])
axes[0].set_xticks(np.arange(len(table5)))
axes[0].set_xticklabels(table5["model"], rotation=45, ha="right")
axes[0].set_title("Mechanism controls: AUC")
axes[0].set_ylabel("Test AUC")

axes[1].bar(np.arange(len(table5)), table5["test_auprc"], color=["#4C72B0" if s=="original" else "#C44E52" for s in table5["split"]])
axes[1].set_xticks(np.arange(len(table5)))
axes[1].set_xticklabels(table5["model"], rotation=45, ha="right")
axes[1].set_title("Mechanism controls: AUPRC")
axes[1].set_ylabel("Test AUPRC")

plt.tight_layout()
plt.savefig(FIG_DIR / "Figure4_mechanism_controls.png", dpi=300)
plt.savefig(FIG_DIR / "Figure4_mechanism_controls.pdf")
plt.close()

# ---------------------------------------------------------------------
# 10. Figure 5: copy focused DCA if already generated
# ---------------------------------------------------------------------
ensure_copy(ROOT / "data/processed/clinical_plots/dca_focus_caddam_vs_full.png",
            FIG_DIR / "Figure5_dca_focus_original.png")
ensure_copy(ROOT / "data/processed/clinical_plots/dca_focus_caddam_vs_full.pdf",
            FIG_DIR / "Figure5_dca_focus_original.pdf")
ensure_copy(ROOT / "data/processed/clinical_plots/calibration_focus_caddam_vs_full.png",
            FIG_DIR / "Figure5b_calibration_focus_original.png")
ensure_copy(ROOT / "data/processed/clinical_plots/calibration_focus_caddam_vs_full.pdf",
            FIG_DIR / "Figure5b_calibration_focus_original.pdf")

# ---------------------------------------------------------------------
# 11. Figure 6: homology clinical barplot
# ---------------------------------------------------------------------
if HOMO_CLINICAL_CSV.exists():
    homo_clin_plot = pd.read_csv(HOMO_CLINICAL_CSV)
    metrics = ["brier_score", "ece_10bin", "sensitivity_at_90_specificity"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        ax.bar(np.arange(len(homo_clin_plot)), homo_clin_plot[metric], color="#55A868")
        ax.set_xticks(np.arange(len(homo_clin_plot)))
        ax.set_xticklabels(homo_clin_plot["model"], rotation=45, ha="right")
        ax.set_title(metric)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure6_homology_clinical_metrics.png", dpi=300)
    plt.savefig(FIG_DIR / "Figure6_homology_clinical_metrics.pdf")
    plt.close()

# ---------------------------------------------------------------------
# 12. Copy supporting outputs
# ---------------------------------------------------------------------
ensure_copy(PREVALENCE_CSV, TAB_DIR / "Supplementary_split_prevalence_summary.csv")
ensure_copy(PR_SUMMARY_CSV, TAB_DIR / "Supplementary_pr_curve_auc_summary.csv")
ensure_copy(ROOT / "data/processed/q1_case_study/q1_case_study_summary.json",
            TAB_DIR / "Supplementary_q1_case_study_summary.json")

# ---------------------------------------------------------------------
# 13. Save manifest
# ---------------------------------------------------------------------
manifest = {
    "figures": sorted([p.name for p in FIG_DIR.iterdir()]),
    "tables": sorted([p.name for p in TAB_DIR.iterdir()]),
}
with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\n[7] Saved figure/table bundle")
print("Figures:")
for p in sorted(FIG_DIR.iterdir()):
    print("✅", p)
print("\nTables:")
for p in sorted(TAB_DIR.iterdir()):
    print("✅", p)

print("\n✅ Manifest:", OUT_DIR / "manifest.json")
print("=" * 90)
print("Done.")
print("=" * 90)
