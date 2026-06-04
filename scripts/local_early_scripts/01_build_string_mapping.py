import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLINVAR_PATH = ROOT / "data/processed/clinvar_missense_reviewed_dedup.csv"
ALIASES_PATH = ROOT / "data/raw/9606.protein.aliases.v12.0.txt.gz"
LINKS_PATH = ROOT / "data/raw/9606.protein.links.v12.0.txt.gz"

OUT_MAPPED = ROOT / "data/processed/clinvar_missense_reviewed_dedup_mapped.csv"
OUT_GENE_MAP = ROOT / "data/processed/gene_symbol_to_string_unique.csv"
OUT_GENE_AMBIG = ROOT / "data/processed/gene_symbol_to_string_ambiguous.csv"
OUT_GENE_UNMAPPED = ROOT / "data/processed/gene_symbol_unmapped.csv"

print("=" * 70)
print("🕸️ Step 01: Build STRING mapping from clean ClinVar variants")
print("=" * 70)
print("ClinVar input :", CLINVAR_PATH)
print("Aliases input :", ALIASES_PATH)
print("Links input   :", LINKS_PATH)
print("Mapped output :", OUT_MAPPED)

# ------------------------------------------------------------------
# 1. Load ClinVar deduplicated reviewed missense dataset
# ------------------------------------------------------------------
clinvar = pd.read_csv(CLINVAR_PATH)
print("\n[1] ClinVar input summary")
print("Rows :", len(clinvar))
print("Genes:", clinvar["GeneSymbol"].nunique())

required_cols = ["VariationID", "GeneSymbol", "Name", "Label"]
for col in required_cols:
    print(f"{col}:", "✅" if col in clinvar.columns else "❌")

# ------------------------------------------------------------------
# 2. Load STRING aliases
# Expected format usually: protein_id alias source
# ------------------------------------------------------------------
print("\n[2] Loading STRING aliases...")
aliases = pd.read_csv(ALIASES_PATH, sep="\t", low_memory=False)

print("Aliases shape:", aliases.shape)
print("Alias columns:", list(aliases.columns))

# 尝试兼容不同列名
possible_protein_cols = [c for c in aliases.columns if "protein" in c.lower() or c.lower() == "#string_protein_id"]
possible_alias_cols = [c for c in aliases.columns if c.lower() == "alias" or "alias" in c.lower()]
possible_source_cols = [c for c in aliases.columns if c.lower() == "source" or "source" in c.lower()]

if len(possible_protein_cols) == 0 or len(possible_alias_cols) == 0:
    raise ValueError("Could not identify protein_id / alias columns in STRING aliases file.")

protein_col = possible_protein_cols[0]
alias_col = possible_alias_cols[0]
source_col = possible_source_cols[0] if len(possible_source_cols) > 0 else None

print(f"Using protein column: {protein_col}")
print(f"Using alias column  : {alias_col}")
print(f"Using source column : {source_col if source_col else 'None'}")

# 统一大写，避免大小写影响 gene symbol 匹配
aliases["alias_upper"] = aliases[alias_col].astype(str).str.upper().str.strip()
clinvar["GeneSymbol_upper"] = clinvar["GeneSymbol"].astype(str).str.upper().str.strip()

# ------------------------------------------------------------------
# 3. Load STRING links to know actual graph-covered proteins
# ------------------------------------------------------------------
print("\n[3] Loading STRING links...")
links = pd.read_csv(LINKS_PATH, sep=r"\s+", compression="gzip")


print("Links shape:", links.shape)
print("Link columns:", list(links.columns))

# 兼容不同列名，一般是 protein1 protein2 combined_score
prot1_col = [c for c in links.columns if "protein1" in c.lower()][0]
prot2_col = [c for c in links.columns if "protein2" in c.lower()][0]

graph_nodes = set(links[prot1_col].astype(str)) | set(links[prot2_col].astype(str))
print("STRING graph nodes:", len(graph_nodes))

# ------------------------------------------------------------------
# 4. Build GeneSymbol -> candidate STRING protein mapping
# ------------------------------------------------------------------
print("\n[4] Building GeneSymbol → STRING candidate mappings...")

alias_hits = aliases.loc[aliases["alias_upper"].isin(clinvar["GeneSymbol_upper"].unique())].copy()

if source_col is not None:
    print("\nTop alias sources among matched rows:")
    print(alias_hits[source_col].value_counts().head(10).to_string())

gene_to_candidates = (
    alias_hits.groupby("alias_upper")[protein_col]
    .apply(lambda x: sorted(set(x.astype(str))))
    .reset_index()
    .rename(columns={"alias_upper": "GeneSymbol_upper", protein_col: "STRING_candidates"})
)

gene_to_candidates["n_candidates"] = gene_to_candidates["STRING_candidates"].apply(len)

# 唯一映射 / 歧义映射
gene_unique = gene_to_candidates.loc[gene_to_candidates["n_candidates"] == 1].copy()
gene_unique["STRING_protein_id"] = gene_unique["STRING_candidates"].apply(lambda x: x[0])

gene_ambig = gene_to_candidates.loc[gene_to_candidates["n_candidates"] > 1].copy()

all_genes = pd.DataFrame({"GeneSymbol_upper": sorted(clinvar["GeneSymbol_upper"].unique())})
gene_unmapped = all_genes.loc[~all_genes["GeneSymbol_upper"].isin(gene_to_candidates["GeneSymbol_upper"])].copy()

print("Genes in ClinVar:", len(all_genes))
print("Genes uniquely mapped:", len(gene_unique))
print("Genes ambiguously mapped:", len(gene_ambig))
print("Genes unmapped:", len(gene_unmapped))

# ------------------------------------------------------------------
# 5. Restrict unique mappings to proteins that actually exist in graph
# ------------------------------------------------------------------
print("\n[5] Restricting to STRING graph-covered proteins...")
gene_unique["in_graph"] = gene_unique["STRING_protein_id"].isin(graph_nodes)

gene_unique_graph = gene_unique.loc[gene_unique["in_graph"]].copy()
gene_unique_not_graph = gene_unique.loc[~gene_unique["in_graph"]].copy()

print("Unique-mapped genes in graph:", len(gene_unique_graph))
print("Unique-mapped genes NOT in graph:", len(gene_unique_not_graph))

# ------------------------------------------------------------------
# 6. Merge unique graph mappings back into ClinVar
# ------------------------------------------------------------------
print("\n[6] Merging mappings back to ClinVar variants...")

mapped = clinvar.merge(
    gene_unique_graph[["GeneSymbol_upper", "STRING_protein_id"]],
    on="GeneSymbol_upper",
    how="inner"
).copy()

print("Mapped ClinVar variants:", len(mapped))
print("Mapped ClinVar genes   :", mapped["GeneSymbol"].nunique())
print("Variant coverage       :", f"{len(mapped) / len(clinvar) * 100:.2f}%")

# ------------------------------------------------------------------
# 7. Save outputs
# ------------------------------------------------------------------
print("\n[7] Saving outputs...")

gene_unique_graph_out = gene_unique_graph.copy()
gene_unique_graph_out.to_csv(OUT_GENE_MAP, index=False)

gene_ambig.to_csv(OUT_GENE_AMBIG, index=False)
gene_unmapped.to_csv(OUT_GENE_UNMAPPED, index=False)
mapped.to_csv(OUT_MAPPED, index=False)

print("✅ Saved unique mapping table :", OUT_GENE_MAP)
print("✅ Saved ambiguous mapping    :", OUT_GENE_AMBIG)
print("✅ Saved unmapped genes       :", OUT_GENE_UNMAPPED)
print("✅ Saved mapped ClinVar table :", OUT_MAPPED)

# ------------------------------------------------------------------
# 8. Preview
# ------------------------------------------------------------------
print("\n[8] Preview of mapped variants")
preview_cols = [c for c in [
    "VariationID", "GeneSymbol", "STRING_protein_id", "Name", "Label"
] if c in mapped.columns]
print(mapped[preview_cols].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
