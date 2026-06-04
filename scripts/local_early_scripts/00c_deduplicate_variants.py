import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/processed/clinvar_missense_reviewed.csv"
OUT_PATH = ROOT / "data/processed/clinvar_missense_reviewed_dedup.csv"

print("=" * 70)
print("🧹 Step 00c: Deduplicate reviewed ClinVar missense variants")
print("=" * 70)
print("Input :", IN_PATH)
print("Output:", OUT_PATH)

df = pd.read_csv(IN_PATH)

print("\n[1] Input dataset summary")
print("Rows:", len(df))
print("Columns:", len(df.columns))

required_cols = ["VariationID", "#AlleleID", "Name", "Label"]
for col in required_cols:
    print(f"{col}:", "✅" if col in df.columns else "❌")

# 统一转成字符串/整数风格，防止类型混乱
if "VariationID" in df.columns:
    df["VariationID"] = df["VariationID"].astype(str)
if "#AlleleID" in df.columns:
    df["#AlleleID"] = df["#AlleleID"].astype(str)

print("\n[2] Uniqueness check")
n_rows = len(df)
n_varid = df["VariationID"].nunique()
n_alleleid = df["#AlleleID"].nunique()

print("Total rows            :", n_rows)
print("Unique VariationID    :", n_varid)
print("Unique #AlleleID      :", n_alleleid)
print("Duplicate by VariationID:", n_rows - n_varid)
print("Duplicate by #AlleleID  :", n_rows - n_alleleid)

# 看看同一个 VariationID 是否对应多个 Label（冲突标签）
print("\n[3] Label conflict check by VariationID")
varid_label_counts = df.groupby("VariationID")["Label"].nunique()
conflict_varids = varid_label_counts[varid_label_counts > 1]

print("VariationIDs with conflicting labels:", len(conflict_varids))

if len(conflict_varids) > 0:
    print("\nExamples of conflicting VariationID groups:")
    conflict_examples = conflict_varids.index[:10]
    print(df[df["VariationID"].isin(conflict_examples)][
        ["VariationID", "#AlleleID", "Name", "ClinicalSignificance", "Label"]
    ].head(20).to_string(index=False))

# 去重策略：
# 1) 先丢弃存在标签冲突的 VariationID
# 2) 再按 VariationID 去重（保留第一条）
print("\n[4] Deduplication strategy")
print("Step A: remove conflicting VariationIDs")
print("Step B: drop duplicates by VariationID, keep first occurrence")

df_noconflict = df.loc[~df["VariationID"].isin(conflict_varids.index)].copy()
df_dedup = df_noconflict.drop_duplicates(subset=["VariationID"], keep="first").copy()

print("\n[5] After deduplication")
print("Rows before dedup:", len(df))
print("Rows after removing conflicts:", len(df_noconflict))
print("Rows after VariationID dedup:", len(df_dedup))
print("Dropped total:", len(df) - len(df_dedup))

print("\n[6] Label distribution after deduplication")
print(df_dedup["Label"].value_counts(dropna=False).to_string())

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_dedup.to_csv(OUT_PATH, index=False)

print("\n✅ Saved deduplicated dataset to:")
print(OUT_PATH)

print("\n[7] Sample rows")
print(df_dedup[["VariationID", "#AlleleID", "Name", "ClinicalSignificance", "Label"]].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
