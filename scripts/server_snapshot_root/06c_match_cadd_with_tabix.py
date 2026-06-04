import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path("/root")

KEYS_PATH = ROOT / "data/processed/cadd_match_keys.csv"
CLINVAR_MATCH_PATH = ROOT / "data/processed/clinvar_for_cadd_matching.csv"

CADD_DIR = Path("/root/autodl-tmp/cadd")
CADD_FILE = CADD_DIR / "whole_genome_SNVs.tsv.gz"

OUT_SCORES = ROOT / "data/processed/cadd_matched_scores.csv"
OUT_MERGED = ROOT / "data/processed/clinvar_with_cadd.csv"
OUT_REPORT = ROOT / "data/processed/cadd_match_report.json"

print("=" * 80)
print("🧬 Step 06c: Match CADD GRCh38 scores with tabix")
print("=" * 80)
print("Keys        :", KEYS_PATH)
print("ClinVar base:", CLINVAR_MATCH_PATH)
print("CADD file   :", CADD_FILE)
print("Out scores  :", OUT_SCORES)
print("Out merged  :", OUT_MERGED)

# -------------------------------------------------------------------
# 1. Load keys
# -------------------------------------------------------------------
keys_df = pd.read_csv(KEYS_PATH)
print("\n[1] Loaded keys")
print("Rows:", len(keys_df))

required_cols = ["cadd_key", "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
for col in required_cols:
    if col not in keys_df.columns:
        raise ValueError(f"Missing required column in key file: {col}")

# normalize chr strings
keys_df["Chromosome"] = keys_df["Chromosome"].astype(str).str.replace("^chr", "", regex=True).str.strip()
keys_df["Start"] = keys_df["Start"].astype(int)

# -------------------------------------------------------------------
# 2. Check files
# -------------------------------------------------------------------
print("\n[2] Checking files...")
if not CADD_FILE.exists():
    raise FileNotFoundError(f"CADD file not found: {CADD_FILE}")
if not (CADD_FILE.parent / (CADD_FILE.name + ".tbi")).exists():
    raise FileNotFoundError(f"CADD index not found: {CADD_FILE}.tbi")

print("✅ CADD file exists")
print("✅ CADD index exists")

# -------------------------------------------------------------------
# 3. Query CADD with tabix
# -------------------------------------------------------------------
print("\n[3] Querying CADD with tabix by chromosome...")

matched_rows = []

# group by chromosome to reduce repeated tabix launches
for chrom, subdf in keys_df.groupby("Chromosome"):
    positions = sorted(subdf["Start"].unique().tolist())

    print(f"  Chromosome {chrom}: querying {len(positions):,} positions")

    # make region strings chr:start-start
    regions = [f"{chrom}:{pos}-{pos}" for pos in positions]

    # tabix can take many regions via stdin? simpler: chunk regions into manageable groups
    chunk_size = 5000
    chrom_hits = []

    for i in range(0, len(regions), chunk_size):
        region_chunk = regions[i:i + chunk_size]

        cmd = ["tabix", str(CADD_FILE)] + region_chunk
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode not in (0, 1):
            # 1 can happen for no records in some region sets
            raise RuntimeError(f"tabix failed on chromosome {chrom}: {result.stderr[:500]}")

        stdout = result.stdout.strip()
        if stdout:
            lines = stdout.split("\n")
            chrom_hits.extend(lines)

    if len(chrom_hits) == 0:
        continue

    rows = []
    for line in chrom_hits:
        parts = line.split("\t")
        # Expected CADD columns:
        # Chrom Pos Ref Alt RawScore PHRED
        if len(parts) < 6:
            continue

        c_chrom = parts[0].replace("chr", "").strip()
        c_pos = int(parts[1])
        c_ref = parts[2].strip()
        c_alt = parts[3].strip()
        c_raw = parts[4]
        c_phred = parts[5]

        cadd_key = f"{c_chrom}:{c_pos}:{c_ref}:{c_alt}"
        rows.append({
            "cadd_key": cadd_key,
            "CADD_RawScore": c_raw,
            "CADD_PHRED": c_phred,
        })

    if rows:
        chrom_df = pd.DataFrame(rows).drop_duplicates(subset=["cadd_key"])
        matched_rows.append(chrom_df)

if len(matched_rows) == 0:
    raise RuntimeError("No CADD matches were found by tabix.")

matched_df = pd.concat(matched_rows, ignore_index=True).drop_duplicates(subset=["cadd_key"])

# keep only keys we actually need
need_keys = set(keys_df["cadd_key"].astype(str))
matched_df = matched_df[matched_df["cadd_key"].isin(need_keys)].copy()

print("\n[4] Matched summary")
print("Matched rows:", len(matched_df))
print("Match rate  :", f"{len(matched_df) / len(need_keys) * 100:.2f}%")

# numeric conversion
matched_df["CADD_RawScore"] = pd.to_numeric(matched_df["CADD_RawScore"], errors="coerce")
matched_df["CADD_PHRED"] = pd.to_numeric(matched_df["CADD_PHRED"], errors="coerce")

matched_df.to_csv(OUT_SCORES, index=False)
print("✅ Saved matched score table:", OUT_SCORES)

# -------------------------------------------------------------------
# 4. Merge back
# -------------------------------------------------------------------
print("\n[5] Merging with ClinVar table...")
clinvar = pd.read_csv(CLINVAR_MATCH_PATH, low_memory=False)

if "cadd_key" not in clinvar.columns:
    clinvar["Chromosome"] = clinvar["Chromosome"].astype(str).str.replace("^chr", "", regex=True).str.strip()
    clinvar["cadd_key"] = (
        clinvar["Chromosome"].astype(str) + ":" +
        clinvar["Start"].astype(int).astype(str) + ":" +
        clinvar["ReferenceAlleleVCF"].astype(str) + ":" +
        clinvar["AlternateAlleleVCF"].astype(str)
    )

merged = clinvar.merge(matched_df, on="cadd_key", how="left")

missing = merged["CADD_PHRED"].isna().sum()
print("Merged rows         :", len(merged))
print("Rows missing CADD   :", int(missing))
print("Row-level match rate:", f"{(1 - missing / len(merged)) * 100:.2f}%")

merged.to_csv(OUT_MERGED, index=False)
print("✅ Saved merged ClinVar+CADD table:", OUT_MERGED)

# -------------------------------------------------------------------
# 5. Report
# -------------------------------------------------------------------
report = {
    "n_keys": int(len(need_keys)),
    "n_matched_keys": int(len(matched_df)),
    "key_match_rate": float(len(matched_df) / len(need_keys)),
    "n_merged_rows": int(len(merged)),
    "n_missing_cadd_rows": int(missing),
    "row_match_rate": float(1 - missing / len(merged)),
    "cadd_file": str(CADD_FILE),
    "out_scores": str(OUT_SCORES),
    "out_merged": str(OUT_MERGED),
}

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("✅ Saved report:", OUT_REPORT)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
