import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 你项目里目前最可能的候选文件
CANDIDATES = [
    ROOT / "data/processed/esm_node_features.csv",
    ROOT / "data/processed/esm_features.csv",
    ROOT / "data/processed/esm_embeddings.csv",
    ROOT / "data/processed/node_features_esm.csv",
]

MAPPED_PATH = ROOT / "data/processed/clinvar_missense_reviewed_dedup_mapped.csv"

print("=" * 70)
print("🔍 Step 02b: Inspect candidate ESM feature files")
print("=" * 70)

print("\n[1] Current mapped dataset")
mapped = pd.read_csv(MAPPED_PATH)
print("Mapped rows    :", len(mapped))
print("Mapped genes   :", mapped["GeneSymbol"].nunique())
print("Mapped proteins:", mapped["STRING_protein_id"].nunique())

mapped_string_ids = set(mapped["STRING_protein_id"].astype(str).unique())
mapped_genes = set(mapped["GeneSymbol"].astype(str).str.upper().unique())

print("\n[2] Candidate feature files")
found_any = False

for path in CANDIDATES:
    if not path.exists():
        print(f"❌ Not found: {path.name}")
        continue

    found_any = True
    print("\n" + "-" * 70)
    print("File:", path.name)

    df = pd.read_csv(path, low_memory=False)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns[:20]))
    if len(df.columns) > 20:
        print("... (truncated)")

    # 找可能的ID列
    possible_id_cols = []
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ["protein", "string", "ensp", "gene", "symbol", "id"]):
            possible_id_cols.append(col)

    print("Possible ID columns:", possible_id_cols)

    # 粗略检查每个候选ID列的覆盖情况
    for col in possible_id_cols:
        vals = df[col].astype(str)

        # 与 STRING_protein_id 的直接匹配
        string_overlap = vals.isin(mapped_string_ids).sum()

        # 与 GeneSymbol 的匹配
        gene_overlap = vals.str.upper().isin(mapped_genes).sum()

        print(
            f"  Column={col} | "
            f"STRING overlap={string_overlap} | "
            f"Gene overlap={gene_overlap}"
        )

    print("\nPreview:")
    print(df.head(5).to_string(index=False))

if not found_any:
    print("⚠️ No candidate ESM feature files found in data/processed.")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
