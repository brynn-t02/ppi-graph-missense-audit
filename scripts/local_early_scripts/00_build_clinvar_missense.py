import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/variant_summary.txt.gz"
OUT_PATH = ROOT / "data/processed/clinvar_missense_clean.csv"

print("=" * 70)
print("🧬 Step 00: Build clean ClinVar missense dataset from raw ClinVar")
print("=" * 70)
print("Input :", RAW_PATH)
print("Output:", OUT_PATH)

chunksize = 50000
filtered_chunks = []

raw_rows = 0
germline_rows = 0
grch38_rows = 0
snv_rows = 0
missense_rows = 0
label_rows = 0

# 明确保留的临床标签
patho_terms = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic"
}
AA3 = [
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His","Ile",
    "Leu","Lys","Met","Phe","Pro","Ser","Thr","Trp","Tyr","Val"
]
AA3_RE = "(?:" + "|".join(AA3) + ")"

MISSENSE_HGVSP_RE = rf"^p\.{AA3_RE}\d+{AA3_RE}$"
STOPGAIN_HGVSP_RE = rf"^p\.{AA3_RE}\d+(?:Ter|\*)$"

benign_terms = {
    "Benign",
    "Likely benign",
    "Benign/Likely benign"
}

print("\n[1] Start chunked filtering...")

for i, chunk in enumerate(pd.read_csv(RAW_PATH, sep="\t", low_memory=False, chunksize=chunksize)):
    raw_rows += len(chunk)

    # 需要的列
    needed = [
        "#AlleleID",
        "Type",
        "Name",
        "GeneID",
        "GeneSymbol",
        "ClinicalSignificance",
        "ClinSigSimple",
        "ReviewStatus",
        "OriginSimple",
        "Assembly",
        "Chromosome",
        "Start",
        "Stop",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "VariationID"
    ]
    missing = [c for c in needed if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # A. 只保留 germline
    chunk = chunk.loc[chunk["OriginSimple"].astype(str).str.lower() == "germline"].copy()
    germline_rows += len(chunk)

    # B. 只保留 GRCh38
    chunk = chunk.loc[chunk["Assembly"].astype(str) == "GRCh38"].copy()
    grch38_rows += len(chunk)

    # C. 只保留 SNV
    chunk = chunk.loc[chunk["Type"].astype(str) == "single nucleotide variant"].copy()
    snv_rows += len(chunk)

    # D. 从 Name 中提取 HGVS.p，再严格保留标准 missense
    hgvs_p = chunk["Name"].astype(str).str.extract(r"(p\.[A-Za-z0-9_*]+)")[0]
    hgvs_p = hgvs_p.str.strip()  # 重要：去掉你 debug 里看到的前导空格

    missense_mask = hgvs_p.str.match(MISSENSE_HGVSP_RE, na=False)
    stopgain_mask = hgvs_p.str.match(STOPGAIN_HGVSP_RE, na=False)

    # 只保留真正 missense
    chunk = chunk.loc[missense_mask].copy()
    chunk["HGVS_p"] = hgvs_p.loc[chunk.index]

    chunk = chunk.loc[missense_mask].copy()
    chunk["HGVS_p"] = hgvs_p.loc[chunk.index]
    missense_rows += len(chunk)

    # 只在第一个 chunk 打印调试信息
    if i == 0:
        print("\n[DEBUG] Example extracted HGVS_p:")
        print(hgvs_p.dropna().head(20).to_string(index=False))

        print("\n[DEBUG] Example missense variants kept:")
        print(chunk["Name"].head(20).to_string(index=False))
        print("Stop-gain in first chunk:", int(stopgain_mask.sum()))
        print("Missense in first chunk:", int(missense_mask.sum()))

    # E. 只保留明确标签
    cs = chunk["ClinicalSignificance"].astype(str).str.strip()
    patho_mask = cs.isin(patho_terms)
    benign_mask = cs.isin(benign_terms)

    chunk = chunk.loc[patho_mask | benign_mask].copy()
    label_rows += len(chunk)

    # F. 生成二分类标签
    chunk["Label"] = 0
    chunk.loc[chunk["ClinicalSignificance"].isin(patho_terms), "Label"] = 1

    # G. 只保留关键列
    keep_cols = [
        "#AlleleID",
        "Type",
        "Name",
        "HGVS_p",
        "GeneID",
        "GeneSymbol",
        "ClinicalSignificance",
        "ClinSigSimple",
        "ReviewStatus",
        "OriginSimple",
        "Assembly",
        "Chromosome",
        "Start",
        "Stop",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "VariationID",
        "Label"
    ]
    keep_cols = [c for c in keep_cols if c in chunk.columns]

    filtered_chunks.append(chunk[keep_cols])

    if (i + 1) % 10 == 0:
        print(
            f"  chunk {i+1:03d} | "
            f"raw={raw_rows:,} | "
            f"germline={germline_rows:,} | "
            f"GRCh38={grch38_rows:,} | "
            f"SNV={snv_rows:,} | "
            f"missense={missense_rows:,} | "
            f"labeled={label_rows:,}"
        )

print("\n[2] Concatenating filtered chunks...")
if len(filtered_chunks) == 0:
    df_clean = pd.DataFrame()
else:
    df_clean = pd.concat(filtered_chunks, ignore_index=True)

print("\n[3] Final summary")
print("Raw rows scanned      :", raw_rows)
print("After germline        :", germline_rows)
print("After GRCh38          :", grch38_rows)
print("After SNV filter      :", snv_rows)
print("After missense filter :", missense_rows)
print("After label filter    :", len(df_clean))

print("\n[4] Label distribution")
if len(df_clean) > 0:
    print(df_clean["Label"].value_counts(dropna=False).to_string())
else:
    print("Empty dataset")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUT_PATH, index=False)

print("\n✅ Saved clean ClinVar missense dataset to:")
print(OUT_PATH)

print("\n[5] Sample rows")
if len(df_clean) > 0:
    print(df_clean.head(10).to_string(index=False))
else:
    print("Empty DataFrame")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
