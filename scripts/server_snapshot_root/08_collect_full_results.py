import json
from pathlib import Path
import pandas as pd

ROOT = Path("/root")

OUT_DIR = ROOT / "data/processed/full_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATHS = {
    "arch_summary": ROOT / "data/processed/variant_mlp_comparison/summary.json",
    "degree_metrics": ROOT / "data/processed/degree_baseline/metrics.json",
    "feature_shuffle_metrics": ROOT / "data/processed/gcn_feature_shuffled_classifier/metrics.json",
    "label_shuffle_repeats": ROOT / "data/processed/label_shuffle_repeats/summary.json",
    "cadd_summary": ROOT / "data/processed/cadd_stacking/summary.json",
    "am_summary": ROOT / "data/processed/alphamissense_fusion/summary.json",
    "stage_summary": ROOT / "data/processed/stage_summary/stage_results_summary.json",
}

print("=" * 80)
print("📦 Step 08: Collect full results summary")
print("=" * 80)
for k, v in PATHS.items():
    print(f"{k:24s}: {v}")

def load_json(path):
    if not path.exists():
        print(f"⚠️ Missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

arch = load_json(PATHS["arch_summary"])
degree = load_json(PATHS["degree_metrics"])
feature_shuffle = load_json(PATHS["feature_shuffle_metrics"])
label_shuffle = load_json(PATHS["label_shuffle_repeats"])
cadd = load_json(PATHS["cadd_summary"])
am = load_json(PATHS["am_summary"])

rows = []

# -------------------------------------------------------------------
# 1. Architecture
# -------------------------------------------------------------------
if arch is not None:
    for method in ["esm_raw", "gcn_enhanced", "graphsage_enhanced", "gat_enhanced"]:
        if method in arch:
            rows.append({
                "section": "architecture",
                "method": method,
                "val_auc": arch[method]["val_auc"],
                "val_auprc": arch[method]["val_auprc"],
                "test_auc": arch[method]["test_auc"],
                "test_auprc": arch[method]["test_auprc"],
            })

# -------------------------------------------------------------------
# 2. Controls
# -------------------------------------------------------------------
if degree is not None:
    rows.append({
        "section": "control",
        "method": "degree_only",
        "val_auc": degree["val_auc"],
        "val_auprc": degree["val_auprc"],
        "test_auc": degree["test_auc"],
        "test_auprc": degree["test_auprc"],
    })

if feature_shuffle is not None:
    rows.append({
        "section": "control",
        "method": "gcn_feature_shuffled",
        "val_auc": feature_shuffle["val_auc"],
        "val_auprc": feature_shuffle["val_auprc"],
        "test_auc": feature_shuffle["test_auc"],
        "test_auprc": feature_shuffle["test_auprc"],
    })

if label_shuffle is not None:
    rows.append({
        "section": "control",
        "method": "gcn_label_shuffled_repeated",
        "val_auc": label_shuffle["val_auc_mean"],
        "val_auprc": label_shuffle["val_auprc_mean"],
        "test_auc": label_shuffle["test_auc_mean"],
        "test_auprc": label_shuffle["test_auprc_mean"],
        "val_auc_std": label_shuffle["val_auc_std"],
        "val_auprc_std": label_shuffle["val_auprc_std"],
        "test_auc_std": label_shuffle["test_auc_std"],
        "test_auprc_std": label_shuffle["test_auprc_std"],
    })

# -------------------------------------------------------------------
# 3. CADD
# -------------------------------------------------------------------
if cadd is not None:
    for method in ["cadd_only", "gcn_only", "gcn_plus_cadd"]:
        if method in cadd:
            rows.append({
                "section": "cadd",
                "method": method,
                "val_auc": cadd[method]["val_auc"],
                "val_auprc": cadd[method]["val_auprc"],
                "test_auc": cadd[method]["test_auc"],
                "test_auprc": cadd[method]["test_auprc"],
            })

# -------------------------------------------------------------------
# 4. AlphaMissense
# -------------------------------------------------------------------
if am is not None:
    for method in ["am_only", "gcn_only", "gcn_plus_am"]:
        if method in am:
            rows.append({
                "section": "alphamissense",
                "method": method,
                "val_auc": am[method]["val_auc"],
                "val_auprc": am[method]["val_auprc"],
                "test_auc": am[method]["test_auc"],
                "test_auprc": am[method]["test_auprc"],
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "full_results_summary.csv", index=False)

summary = {
    "results": rows
}

# helpers
if len(df) > 0:
    best_global_auc = df.sort_values("test_auc", ascending=False).iloc[0].to_dict()
    best_global_auprc = df.sort_values("test_auprc", ascending=False).iloc[0].to_dict()
    summary["best_global_by_test_auc"] = best_global_auc
    summary["best_global_by_test_auprc"] = best_global_auprc

# best by section
best_by_section = {}
for section in df["section"].unique():
    sub = df[df["section"] == section].copy()
    best_by_section[section] = {
        "best_test_auc": sub.sort_values("test_auc", ascending=False).iloc[0].to_dict(),
        "best_test_auprc": sub.sort_values("test_auprc", ascending=False).iloc[0].to_dict(),
    }
summary["best_by_section"] = best_by_section

with open(OUT_DIR / "full_results_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[1] Saved outputs")
print("✅", OUT_DIR / "full_results_summary.csv")
print("✅", OUT_DIR / "full_results_summary.json")

print("\n[2] Combined summary")
print(df.to_string(index=False))

print("\n[3] Best global")
print("Best by test AUC   :", summary["best_global_by_test_auc"]["section"],
      summary["best_global_by_test_auc"]["method"],
      f"({summary['best_global_by_test_auc']['test_auc']:.4f})")
print("Best by test AUPRC :", summary["best_global_by_test_auprc"]["section"],
      summary["best_global_by_test_auprc"]["method"],
      f"({summary['best_global_by_test_auprc']['test_auprc']:.4f})")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
