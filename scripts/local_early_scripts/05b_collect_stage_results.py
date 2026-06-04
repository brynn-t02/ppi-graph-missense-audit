import json
from pathlib import Path
import pandas as pd

ROOT = Path("/root")

OUT_DIR = ROOT / "data/processed/stage_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------
PATHS = {
    "variant_summary": ROOT / "data/processed/variant_mlp_comparison/summary.json",
    "degree_metrics": ROOT / "data/processed/degree_baseline/metrics.json",
    "feature_shuffle_metrics": ROOT / "data/processed/gcn_feature_shuffled_classifier/metrics.json",
    "label_shuffle_repeats": ROOT / "data/processed/label_shuffle_repeats/summary.json",
    "embedding_sanity": ROOT / "data/processed/embedding_sanity_check/embedding_sanity_summary.json",
}

print("=" * 80)
print("📦 Step 05b: Collect current stage results")
print("=" * 80)
for name, path in PATHS.items():
    print(f"{name:25s}: {path}")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_json(path):
    if not path.exists():
        print(f"⚠️ Missing file: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

variant_summary = load_json(PATHS["variant_summary"])
degree_metrics = load_json(PATHS["degree_metrics"])
feature_shuffle_metrics = load_json(PATHS["feature_shuffle_metrics"])
label_shuffle_repeats = load_json(PATHS["label_shuffle_repeats"])
embedding_sanity = load_json(PATHS["embedding_sanity"])

# -------------------------------------------------------------------
# 1. Build architecture table
# -------------------------------------------------------------------
arch_rows = []

if variant_summary is not None:
    for key in ["esm_raw", "gcn_enhanced", "graphsage_enhanced", "gat_enhanced"]:
        if key in variant_summary:
            row = {
                "section": "architecture",
                "method": key,
                "val_auc": variant_summary[key]["val_auc"],
                "val_auprc": variant_summary[key]["val_auprc"],
                "test_auc": variant_summary[key]["test_auc"],
                "test_auprc": variant_summary[key]["test_auprc"],
            }
            arch_rows.append(row)

arch_df = pd.DataFrame(arch_rows)

# -------------------------------------------------------------------
# 2. Build mechanism/control table
# -------------------------------------------------------------------
control_rows = []

if degree_metrics is not None:
    control_rows.append({
        "section": "control",
        "method": "degree_only",
        "val_auc": degree_metrics["val_auc"],
        "val_auprc": degree_metrics["val_auprc"],
        "test_auc": degree_metrics["test_auc"],
        "test_auprc": degree_metrics["test_auprc"],
    })

if feature_shuffle_metrics is not None:
    control_rows.append({
        "section": "control",
        "method": "gcn_feature_shuffled",
        "val_auc": feature_shuffle_metrics["val_auc"],
        "val_auprc": feature_shuffle_metrics["val_auprc"],
        "test_auc": feature_shuffle_metrics["test_auc"],
        "test_auprc": feature_shuffle_metrics["test_auprc"],
    })

if label_shuffle_repeats is not None:
    control_rows.append({
        "section": "control",
        "method": "gcn_label_shuffled_repeated",
        "val_auc": label_shuffle_repeats["val_auc_mean"],
        "val_auprc": label_shuffle_repeats["val_auprc_mean"],
        "test_auc": label_shuffle_repeats["test_auc_mean"],
        "test_auprc": label_shuffle_repeats["test_auprc_mean"],
        "val_auc_std": label_shuffle_repeats["val_auc_std"],
        "val_auprc_std": label_shuffle_repeats["val_auprc_std"],
        "test_auc_std": label_shuffle_repeats["test_auc_std"],
        "test_auprc_std": label_shuffle_repeats["test_auprc_std"],
    })

control_df = pd.DataFrame(control_rows)

# -------------------------------------------------------------------
# 3. Combined summary
# -------------------------------------------------------------------
combined_df = pd.concat([arch_df, control_df], ignore_index=True)

# -------------------------------------------------------------------
# 4. Derive interpretation helpers
# -------------------------------------------------------------------
summary = {
    "architecture_results": arch_rows,
    "control_results": control_rows,
}

# Best architecture by test AUC
if len(arch_df) > 0:
    best_auc_row = arch_df.sort_values("test_auc", ascending=False).iloc[0].to_dict()
    best_auprc_row = arch_df.sort_values("test_auprc", ascending=False).iloc[0].to_dict()
    summary["best_architecture_by_test_auc"] = best_auc_row
    summary["best_architecture_by_test_auprc"] = best_auprc_row

    # deltas vs esm_raw
    esm_row = arch_df.loc[arch_df["method"] == "esm_raw"]
    if len(esm_row) == 1:
        esm_test_auc = float(esm_row.iloc[0]["test_auc"])
        esm_test_auprc = float(esm_row.iloc[0]["test_auprc"])

        delta_rows = []
        for _, row in arch_df.iterrows():
            if row["method"] == "esm_raw":
                continue
            delta_rows.append({
                "method": row["method"],
                "delta_test_auc_vs_esm": float(row["test_auc"] - esm_test_auc),
                "delta_test_auprc_vs_esm": float(row["test_auprc"] - esm_test_auprc),
            })
        summary["architecture_deltas_vs_esm"] = delta_rows

# Add sanity summary snapshot
if embedding_sanity is not None:
    summary["embedding_sanity_snapshot"] = {
        "flags": embedding_sanity.get("flags", {}),
        "pairwise_sampled_comparison": embedding_sanity.get("pairwise_sampled_comparison", {}),
    }

# -------------------------------------------------------------------
# 5. Save outputs
# -------------------------------------------------------------------
combined_df.to_csv(OUT_DIR / "stage_results_summary.csv", index=False)

with open(OUT_DIR / "stage_results_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[1] Saved outputs")
print("✅", OUT_DIR / "stage_results_summary.csv")
print("✅", OUT_DIR / "stage_results_summary.json")

# -------------------------------------------------------------------
# 6. Print concise summary
# -------------------------------------------------------------------
print("\n[2] Architecture summary")
if len(arch_df) > 0:
    print(arch_df.to_string(index=False))
else:
    print("No architecture rows found.")

print("\n[3] Control summary")
if len(control_df) > 0:
    print(control_df.to_string(index=False))
else:
    print("No control rows found.")

print("\n[4] Best architecture")
if "best_architecture_by_test_auc" in summary:
    print("Best by test AUC   :", summary["best_architecture_by_test_auc"]["method"],
          f"({summary['best_architecture_by_test_auc']['test_auc']:.4f})")
if "best_architecture_by_test_auprc" in summary:
    print("Best by test AUPRC :", summary["best_architecture_by_test_auprc"]["method"],
          f"({summary['best_architecture_by_test_auprc']['test_auprc']:.4f})")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
