import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/processed/clinvar_missense_clean.csv"
OUT_PATH = ROOT / "data/processed/clinvar_missense_reviewed.csv"

print("=" * 70)
print("🧪 Step 00b: Filter ClinVar missense dataset by ReviewStatus")
print("=" * 70)
print("Input :", IN_PATH)
print("Output:", OUT_PATH)

df = pd.read_csv(IN_PATH)

print("\n[1] Input dataset summary")
print("Shape:", df.shape)

required_cols = ["ReviewStatus", "Label", "Name"]
for col in required_cols:
    print(f"{col}:", "✅" if col in df.columns else "❌")

print("\n[2] ReviewStatus distribution before filtering")
review_counts = df["ReviewStatus"].astype(str).value_counts(dropna=False)
print(review_counts.to_string())

# 保留的 review status
keep_status = {
    "criteria provided, single submitter",
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline"
}

df["ReviewStatus"] = df["ReviewStatus"].astype(str).str.strip()
keep_mask = df["ReviewStatus"].isin(keep_status)

df_filtered = df.loc[keep_mask].copy()

print("\n[3] After ReviewStatus filtering")
print("Retained rows:", len(df_filtered))
print("Dropped rows :", len(df) - len(df_filtered))

print("\n[4] ReviewStatus distribution after filtering")
print(df_filtered["ReviewStatus"].value_counts(dropna=False).to_string())

print("\n[5] Label distribution after filtering")
print(df_filtered["Label"].value_counts(dropna=False).to_string())

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_filtered.to_csv(OUT_PATH, index=False)

print("\n✅ Saved filtered dataset to:")
print(OUT_PATH)

print("\n[6] Sample rows")
print(df_filtered[["Name", "ClinicalSignificance", "ReviewStatus", "Label"]].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
