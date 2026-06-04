import pandas as pd
from pathlib import Path

ROOT = Path("/root")

IN_PATH = ROOT / "data/processed/clinvar_missense_reviewed_dedup_mapped.csv"
OUT_KEYS = ROOT / "data/processed/alphamissense_match_keys.csv"
OUT_FULL = ROOT / "data/processed/clinvar_for_alphamissense_matching.csv"

print("=" * 80)
print("🧬 Step 07b: Prepare variant keys for AlphaMissense matching")
print("=" * 80)
print("Input :", IN_PATH)
print("Keys  :", OUT_KEYS)
print("Full  :", OUT_FULL)

df = pd.read_csv(IN_PATH, low_memory=False)

print("\n[1] Input summary")
print("Rows :", len(df))
print("Cols :", len(df.columns))

required_cols = [
    "Assembly",
    "Chromosome",
    "Start",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "VariationID",
    "Label",
]
for col in required_cols:
    print(f"{col}:", "✅" if col in df.columns else "❌")

# -------------------------------------------------------------------
# 1. Keep only GRCh38 rows with usable key fields
# -------------------------------------------------------------------
df = df.copy()

df["Assembly"] = df["Assembly"].astype(str)
df["Chromosome"] = df["Chromosome"].astype(str)
df["ReferenceAlleleVCF"] = df["ReferenceAlleleVCF"].astype(str)
df["AlternateAlleleVCF"] = df["AlternateAlleleVCF"].astype(str)

mask = (
    (df["Assembly"] == "GRCh38") &
    df["Chromosome"].notna() &
    df["Start"].notna() &
    df["ReferenceAlleleVCF"].notna() &
    df["AlternateAlleleVCF"].notna()
)

df = df.loc[mask].copy()

print("\n[2] After field filtering")
print("Rows:", len(df))

# normalize chromosome strings for AlphaMissense file (which uses chr1, chr2, ..., chrX)
df["Chromosome"] = df["Chromosome"].str.replace("^chr", "", regex=True).str.strip()
df["am_key"] = (
    "chr" + df["Chromosome"].astype(str) + ":" +
    df["Start"].astype(int).astype(str) + ":" +
    df["ReferenceAlleleVCF"].astype(str) + ":" +
    df["AlternateAlleleVCF"].astype(str)
)

print("\n[3] Example keys")
print(df[["VariationID", "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF", "am_key"]]
      .head(5).to_string(index=False))

# -------------------------------------------------------------------
# 2. Save full matching table
# -------------------------------------------------------------------
df.to_csv(OUT_FULL, index=False)
print("\n✅ Saved full AlphaMissense matching table:")
print(OUT_FULL)

# -------------------------------------------------------------------
# 3. Save unique keys
# -------------------------------------------------------------------
key_df = df[[
    "am_key",
    "Chromosome",
    "Start",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF"
]].drop_duplicates().copy()

key_df = key_df.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
key_df.to_csv(OUT_KEYS, index=False)

print("\n✅ Saved unique AlphaMissense key table:")
print(OUT_KEYS)

print("\n[4] Key summary")
print("Rows in full matching table :", len(df))
print("Unique AlphaMissense keys   :", len(key_df))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
