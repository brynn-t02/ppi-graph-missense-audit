import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path("/root")

ORIG_TEST_PATH = ROOT / "data/processed/clinvar_test_main.csv"
HOMO_TEST_PATH = ROOT / "data/processed/homology_aware_split/clinvar_test_homology.csv"

OUT_DIR = ROOT / "data/processed/pr_curve_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBS = {
    # original split: representation
    "esm_raw": ROOT / "data/processed/variant_mlp_comparison/esm_raw/test_probs.npy",
    "gcn_enhanced": ROOT / "data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy",

    # homology split: representation
    "esm_raw_homology": ROOT / "data/processed/homology_rerun_core/esm_raw_homology/test_probs.npy",
    "gcn_enhanced_homology": ROOT / "data/processed/homology_rerun_core/gcn_enhanced_homology/test_probs.npy",

    # original split: random controls
    "random_feature_gcn": ROOT / "data/processed/random_feature_gcn_control_repeated/summary.json",
    "random_raw_feature_baseline": ROOT / "data/processed/random_raw_feature_baseline/summary.json",

    # homology split: random controls
    "random_feature_gcn_homology": ROOT / "data/processed/random_feature_gcn_homology/summary.json",
    "random_raw_feature_homology": ROOT / "data/processed/random_raw_feature_homology/summary.json",

    # homology split: cadd line
    "cadd_only_homology": ROOT / "data/processed/homology_single_tool_fusions/cadd_only_homology/test_probs.npy",
    "gcn_plus_cadd_homology": ROOT / "data/processed/homology_single_tool_fusions/gcn_plus_cadd_homology/test_probs.npy",
}

# For repeated summaries, we only have summary/per-repeat CSV, not stored test probs.
# So for random controls, we will read per-repeat CSV and just report mean AUPRC numerically,
# while PR curves themselves will use single-run probability files if available.
# To support plotting, we optionally look for single-run outputs from step 15 / 18.
OPTIONAL_SINGLE_RUN_PROBS = {
    "random_feature_gcn_single": ROOT / "data/processed/random_feature_gcn_control/test_probs.npy",
    "random_raw_feature_single": ROOT / "data/processed/random_raw_feature_baseline_single/test_probs.npy",  # likely absent
    "random_feature_gcn_homology_single": ROOT / "data/processed/random_feature_gcn_homology_single/test_probs.npy",  # likely absent
    "random_raw_feature_homology_single": ROOT / "data/processed/random_raw_feature_homology_single/test_probs.npy",  # likely absent
}

print("=" * 80)
print("📉 Step 22: Plot PR curves and split prevalence")
print("=" * 80)
print("Original test :", ORIG_TEST_PATH)
print("Homology test :", HOMO_TEST_PATH)
print("Outdir        :", OUT_DIR)

# -------------------------------------------------------------------
# 1. Load labels
# -------------------------------------------------------------------
orig_df = pd.read_csv(ORIG_TEST_PATH, low_memory=False)
homo_df = pd.read_csv(HOMO_TEST_PATH, low_memory=False)

y_orig = orig_df["Label"].astype(int).values
y_homo = homo_df["Label"].astype(int).values

print("\n[1] Loaded labels")
print("Original split test rows :", len(y_orig))
print("Homology split test rows :", len(y_homo))

# prevalence summary
prev_df = pd.DataFrame([
    {
        "split": "original_protein_level",
        "n": len(y_orig),
        "n_positive": int(np.sum(y_orig == 1)),
        "n_negative": int(np.sum(y_orig == 0)),
        "positive_rate": float(np.mean(y_orig == 1)),
    },
    {
        "split": "homology_aware",
        "n": len(y_homo),
        "n_positive": int(np.sum(y_homo == 1)),
        "n_negative": int(np.sum(y_homo == 0)),
        "positive_rate": float(np.mean(y_homo == 1)),
    },
])

prev_df.to_csv(OUT_DIR / "split_prevalence_summary.csv", index=False)

print("\n[2] Split prevalence summary")
print(prev_df.to_string(index=False))

# -------------------------------------------------------------------
# 2. Helper to load probability arrays
# -------------------------------------------------------------------
def load_probs(path):
    if path.suffix == ".npy" and path.exists():
        return np.load(path)
    return None

def pr_stats(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    return precision, recall, ap

# -------------------------------------------------------------------
# 3. Collect PR-ready models
# -------------------------------------------------------------------
pr_rows = []

# Core models with actual probability arrays
plot_groups = {
    "representation": [
        ("esm_raw", y_orig, PROBS["esm_raw"]),
        ("gcn_enhanced", y_orig, PROBS["gcn_enhanced"]),
        ("esm_raw_homology", y_homo, PROBS["esm_raw_homology"]),
        ("gcn_enhanced_homology", y_homo, PROBS["gcn_enhanced_homology"]),
    ],
    "homology_cadd": [
        ("cadd_only_homology", y_homo, PROBS["cadd_only_homology"]),
        ("gcn_plus_cadd_homology", y_homo, PROBS["gcn_plus_cadd_homology"]),
    ],
}

# random controls: use numerical summaries from json
random_summary_rows = []

def extract_summary_metric(json_path, block_name, metric_name):
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if block_name in obj:
        return obj[block_name].get(metric_name)
    return None

# original repeated random summaries
random_summary_rows.append({
    "model": "random_feature_gcn_repeated",
    "split": "original_protein_level",
    "auprc_mean": extract_summary_metric(
        ROOT / "data/processed/random_feature_gcn_control_repeated/summary.json",
        "random_feature_gcn_repeated",
        "test_auprc_mean",
    ),
    "auc_mean": extract_summary_metric(
        ROOT / "data/processed/random_feature_gcn_control_repeated/summary.json",
        "random_feature_gcn_repeated",
        "test_auc_mean",
    ),
})
random_summary_rows.append({
    "model": "random_raw_feature_baseline",
    "split": "original_protein_level",
    "auprc_mean": extract_summary_metric(
        ROOT / "data/processed/random_raw_feature_baseline/summary.json",
        "random_raw_feature_baseline",
        "test_auprc_mean",
    ),
    "auc_mean": extract_summary_metric(
        ROOT / "data/processed/random_raw_feature_baseline/summary.json",
        "random_raw_feature_baseline",
        "test_auc_mean",
    ),
})

# homology repeated random summaries
random_summary_rows.append({
    "model": "random_feature_gcn_homology",
    "split": "homology_aware",
    "auprc_mean": extract_summary_metric(
        ROOT / "data/processed/random_feature_gcn_homology/summary.json",
        "random_feature_gcn_homology",
        "test_auprc_mean",
    ),
    "auc_mean": extract_summary_metric(
        ROOT / "data/processed/random_feature_gcn_homology/summary.json",
        "random_feature_gcn_homology",
        "test_auc_mean",
    ),
})
random_summary_rows.append({
    "model": "random_raw_feature_homology",
    "split": "homology_aware",
    "auprc_mean": extract_summary_metric(
        ROOT / "data/processed/random_raw_feature_homology/summary.json",
        "random_raw_feature_homology",
        "test_auprc_mean",
    ),
    "auc_mean": extract_summary_metric(
        ROOT / "data/processed/random_raw_feature_homology/summary.json",
        "random_raw_feature_homology",
        "test_auc_mean",
    ),
})

# -------------------------------------------------------------------
# 4. Plot representation PR curves
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for name, y, path in plot_groups["representation"]:
    probs = load_probs(path)
    if probs is None:
        continue
    precision, recall, ap = pr_stats(y, probs)
    plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    pr_rows.append({
        "group": "representation",
        "model": name,
        "split": "homology_aware" if "homology" in name else "original_protein_level",
        "average_precision": float(ap),
    })

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curves: representation learning models")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curves_representation.png", dpi=300)
plt.savefig(OUT_DIR / "pr_curves_representation.pdf")
plt.close()

# -------------------------------------------------------------------
# 5. Plot homology CADD PR curves
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for name, y, path in plot_groups["homology_cadd"]:
    probs = load_probs(path)
    if probs is None:
        continue
    precision, recall, ap = pr_stats(y, probs)
    plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    pr_rows.append({
        "group": "homology_cadd",
        "model": name,
        "split": "homology_aware",
        "average_precision": float(ap),
    })

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curves: CADD line under homology-aware split")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curves_homology_cadd.png", dpi=300)
plt.savefig(OUT_DIR / "pr_curves_homology_cadd.pdf")
plt.close()

# -------------------------------------------------------------------
# 6. Random control summary plot (bar chart, since repeated probs not saved)
# -------------------------------------------------------------------
random_df = pd.DataFrame(random_summary_rows)

plt.figure(figsize=(8, 5))
x = np.arange(len(random_df))
plt.bar(x, random_df["auprc_mean"], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
plt.xticks(x, random_df["model"], rotation=25, ha="right")
plt.ylabel("Mean AUPRC")
plt.title("Random-control comparison across splits")
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curves_random_controls.png", dpi=300)
plt.savefig(OUT_DIR / "pr_curves_random_controls.pdf")
plt.close()

# add rows
for _, row in random_df.iterrows():
    pr_rows.append({
        "group": "random_controls_summary",
        "model": row["model"],
        "split": row["split"],
        "average_precision": row["auprc_mean"],
        "auc_mean": row["auc_mean"],
    })

# -------------------------------------------------------------------
# 7. Save summary tables
# -------------------------------------------------------------------
pr_summary_df = pd.DataFrame(pr_rows)
pr_summary_df.to_csv(OUT_DIR / "pr_curve_auc_summary.csv", index=False)
random_df.to_csv(OUT_DIR / "random_control_summary.csv", index=False)

report = {
    "prevalence_summary_file": str(OUT_DIR / "split_prevalence_summary.csv"),
    "pr_summary_file": str(OUT_DIR / "pr_curve_auc_summary.csv"),
    "plots": [
        "pr_curves_representation.png/pdf",
        "pr_curves_homology_cadd.png/pdf",
        "pr_curves_random_controls.png/pdf",
    ],
}

with open(OUT_DIR / "pr_curve_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n[3] Saved outputs")
print("✅", OUT_DIR / "split_prevalence_summary.csv")
print("✅", OUT_DIR / "pr_curve_auc_summary.csv")
print("✅", OUT_DIR / "random_control_summary.csv")
print("✅", OUT_DIR / "pr_curves_representation.png")
print("✅", OUT_DIR / "pr_curves_homology_cadd.png")
print("✅", OUT_DIR / "pr_curves_random_controls.png")
print("✅", OUT_DIR / "pr_curve_report.json")

print("\n[4] PR/AP summary")
print(pr_summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
