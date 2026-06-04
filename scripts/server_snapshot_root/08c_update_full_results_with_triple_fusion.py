import json
from pathlib import Path
import pandas as pd

ROOT = Path("/root")

FULL_SUMMARY_PATH = ROOT / "data/processed/full_summary/full_results_summary.json"
TRIPLE_SUMMARY_PATH = ROOT / "data/processed/triple_fusion/summary.json"

OUT_DIR = ROOT / "data/processed/full_summary"
OUT_JSON = OUT_DIR / "full_results_summary_with_triple.json"
OUT_CSV = OUT_DIR / "full_results_summary_with_triple.csv"

print("=" * 80)
print("📦 Step 08c: Update full summary with triple fusion")
print("=" * 80)
print("Full summary :", FULL_SUMMARY_PATH)
print("Triple fusion:", TRIPLE_SUMMARY_PATH)

with open(FULL_SUMMARY_PATH, "r", encoding="utf-8") as f:
    full = json.load(f)

with open(TRIPLE_SUMMARY_PATH, "r", encoding="utf-8") as f:
    triple = json.load(f)

rows = full["results"]

for method in ["cadd_only", "am_only", "cadd_plus_am", "gcn_plus_cadd_plus_am"]:
    if method in triple:
        rows.append({
            "section": "triple_fusion",
            "method": method,
            "val_auc": triple[method]["val_auc"],
            "val_auprc": triple[method]["val_auprc"],
            "test_auc": triple[method]["test_auc"],
            "test_auprc": triple[method]["test_auprc"],
        })

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["section", "method"], keep="last").reset_index(drop=True)

summary = {
    "results": df.to_dict(orient="records")
}

best_global_auc = df.sort_values("test_auc", ascending=False).iloc[0].to_dict()
best_global_auprc = df.sort_values("test_auprc", ascending=False).iloc[0].to_dict()

summary["best_global_by_test_auc"] = best_global_auc
summary["best_global_by_test_auprc"] = best_global_auprc

best_by_section = {}
for section in df["section"].unique():
    sub = df[df["section"] == section].copy()
    best_by_section[section] = {
        "best_test_auc": sub.sort_values("test_auc", ascending=False).iloc[0].to_dict(),
        "best_test_auprc": sub.sort_values("test_auprc", ascending=False).iloc[0].to_dict(),
    }
summary["best_by_section"] = best_by_section

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

df.to_csv(OUT_CSV, index=False)

print("\n[1] Saved outputs")
print("✅", OUT_JSON)
print("✅", OUT_CSV)

print("\n[2] Best global")
print("Best by test AUC   :", summary["best_global_by_test_auc"]["section"],
      summary["best_global_by_test_auc"]["method"],
      f"({summary['best_global_by_test_auc']['test_auc']:.4f})")
print("Best by test AUPRC :", summary["best_global_by_test_auprc"]["section"],
      summary["best_global_by_test_auprc"]["method"],
      f"({summary['best_global_by_test_auprc']['test_auprc']:.4f})")

print("\n[3] Triple fusion rows")
print(df[df["section"] == "triple_fusion"].to_string(index=False))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
