import json
from pathlib import Path
import io

import pandas as pd

ROOT = Path("/root")

KEYS_PATH = ROOT / "data/processed/alphamissense_match_keys.csv"
CLINVAR_MATCH_PATH = ROOT / "data/processed/clinvar_for_alphamissense_matching.csv"
AM_FILE = Path("/root/autodl-tmp/alphamissense/AlphaMissense_hg38.tsv.gz")

OUT_SCORES = ROOT / "data/processed/alphamissense_matched_scores.csv"
OUT_MERGED = ROOT / "data/processed/clinvar_with_alphamissense.csv"
OUT_REPORT = ROOT / "data/processed/alphamissense_match_report.json"

print("=" * 80)
print("🧬 Step 07c: Match AlphaMissense hg38 scores")
print("=" * 80)
print("Keys        :", KEYS_PATH)
print("ClinVar base:", CLINVAR_MATCH_PATH)
print("AM file     :", AM_FILE)
print("Out scores  :", OUT_SCORES)
print("Out merged  :", OUT_MERGED)

# -------------------------------------------------------------------
# 1. Load key table
# -------------------------------------------------------------------
keys_df = pd.read_csv(KEYS_PATH, low_memory=False)
need_keys = set(keys_df["am_key"].astype(str).tolist())

print("\n[1] Loaded keys")
print("Unique keys:", len(need_keys))

# -------------------------------------------------------------------
# 2. Read AlphaMissense file safely
# -------------------------------------------------------------------
print("\n[2] Reading AlphaMissense file...")

# 跳过前面的版权注释，只保留从 #CHROM 开始的表格
import gzip

lines = []
with gzip.open(AM_FILE, "rt") as f:
    started = False
    for line in f:
        if line.startswith("#CHROM"):
            started = True
            lines.append(line.lstrip("#"))  # 去掉表头前的 #
            continue
        if started:
            lines.append(line)

am = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)

print("Columns:", list(am.columns))

required_cols = ["CHROM", "POS", "REF", "ALT", "am_pathogenicity", "am_class"]
for col in required_cols:
    if col not in am.columns:
        raise ValueError(f"Missing required AlphaMissense column: {col}")

# -------------------------------------------------------------------
# 3. Build matching key
# -------------------------------------------------------------------
am["CHROM"] = am["CHROM"].astype(str).str.strip()
am["POS"] = am["POS"].astype(int)
am["REF"] = am["REF"].astype(str).str.strip()
am["ALT"] = am["ALT"].astype(str).str.strip()

am["am_key"] = (
    am["CHROM"].astype(str) + ":" +
    am["POS"].astype(str) + ":" +
    am["REF"].astype(str) + ":" +
    am["ALT"].astype(str)
)

# -------------------------------------------------------------------
# 4. Filter to keys we need
# -------------------------------------------------------------------
matched = am[am["am_key"].isin(need_keys)].copy()

print("\n[3] Matched summary")
print("Matched rows:", len(matched))
print("Match rate  :", f"{len(matched) / len(need_keys) * 100:.2f}%")

keep_cols = ["am_key"]
for col in ["am_pathogenicity", "am_class", "uniprot_id", "transcript_id", "protein_variant"]:
    if col in matched.columns:
        keep_cols.append(col)

matched = matched[keep_cols].drop_duplicates(subset=["am_key"])
matched.to_csv(OUT_SCORES, index=False)

print("✅ Saved matched score table:", OUT_SCORES)

# -------------------------------------------------------------------
# 5. Merge back into ClinVar table
# -------------------------------------------------------------------
print("\n[4] Merging with ClinVar table...")
clinvar = pd.read_csv(CLINVAR_MATCH_PATH, low_memory=False)

if "am_key" not in clinvar.columns:
    clinvar["Chromosome"] = clinvar["Chromosome"].astype(str).str.replace("^chr", "", regex=True).str.strip()
    clinvar["am_key"] = (
        "chr" + clinvar["Chromosome"].astype(str) + ":" +
        clinvar["Start"].astype(int).astype(str) + ":" +
        clinvar["ReferenceAlleleVCF"].astype(str) + ":" +
        clinvar["AlternateAlleleVCF"].astype(str)
    )

merged = clinvar.merge(matched, on="am_key", how="left")

missing = merged["am_pathogenicity"].isna().sum()
print("Merged rows                :", len(merged))
print("Rows missing AlphaMissense :", int(missing))
print("Row-level match rate       :", f"{(1 - missing / len(merged)) * 100:.2f}%")

merged.to_csv(OUT_MERGED, index=False)
print("✅ Saved merged ClinVar+AlphaMissense table:", OUT_MERGED)

# -------------------------------------------------------------------
# 6. Save report
# -------------------------------------------------------------------
report = {
    "n_keys": int(len(need_keys)),
    "n_matched_keys": int(len(matched)),
    "key_match_rate": float(len(matched) / len(need_keys)),
    "n_merged_rows": int(len(merged)),
    "n_missing_rows": int(missing),
    "row_match_rate": float(1 - missing / len(merged)),
    "am_file": str(AM_FILE),
    "out_scores": str(OUT_SCORES),
    "out_merged": str(OUT_MERGED),
}

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("✅ Saved report:", OUT_REPORT)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
