import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/processed/clinvar_missense_reviewed_dedup_mapped.csv"

TRAIN_OUT = ROOT / "data/processed/clinvar_train_main.csv"
VAL_OUT   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_OUT  = ROOT / "data/processed/clinvar_test_main.csv"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

print("=" * 70)
print("🧩 Step 02: Build strict gene-level zero-shot split")
print("=" * 70)
print("Input :", IN_PATH)
print("Train :", TRAIN_OUT)
print("Val   :", VAL_OUT)
print("Test  :", TEST_OUT)
print(f"Seed  : {SEED}")

# ------------------------------------------------------------
# 1. Load mapped ClinVar dataset
# ------------------------------------------------------------
df = pd.read_csv(IN_PATH)

print("\n[1] Input dataset summary")
print("Rows :", len(df))
print("Genes:", df["GeneSymbol"].nunique())
print("Proteins:", df["STRING_protein_id"].nunique())

required_cols = ["VariationID", "GeneSymbol", "STRING_protein_id", "Label"]
for col in required_cols:
    print(f"{col}:", "✅" if col in df.columns else "❌")

# ------------------------------------------------------------
# 2. Unique protein IDs for split
# ------------------------------------------------------------
protein_ids = sorted(df["STRING_protein_id"].astype(str).unique())
n_proteins = len(protein_ids)

print("\n[2] Unique STRING protein IDs")
print("Total unique proteins:", n_proteins)

rng = np.random.default_rng(SEED)
protein_ids = np.array(protein_ids)
rng.shuffle(protein_ids)

n_train = int(n_proteins * TRAIN_RATIO)
n_val = int(n_proteins * VAL_RATIO)
n_test = n_proteins - n_train - n_val

train_proteins = set(protein_ids[:n_train])
val_proteins   = set(protein_ids[n_train:n_train+n_val])
test_proteins  = set(protein_ids[n_train+n_val:])

print("\n[3] Protein-level split sizes")
print("Train proteins:", len(train_proteins))
print("Val proteins  :", len(val_proteins))
print("Test proteins :", len(test_proteins))

# ------------------------------------------------------------
# 3. Assign split
# ------------------------------------------------------------
def assign_split(pid):
    pid = str(pid)
    if pid in train_proteins:
        return "train"
    elif pid in val_proteins:
        return "val"
    elif pid in test_proteins:
        return "test"
    else:
        return "unknown"

df["split"] = df["STRING_protein_id"].astype(str).map(assign_split)

unknown_rows = (df["split"] == "unknown").sum()
if unknown_rows > 0:
    raise ValueError(f"Found {unknown_rows} rows with unknown split assignment.")

train_df = df.loc[df["split"] == "train"].copy()
val_df   = df.loc[df["split"] == "val"].copy()
test_df  = df.loc[df["split"] == "test"].copy()

# ------------------------------------------------------------
# 4. Leakage check
# ------------------------------------------------------------
print("\n[4] Leakage check")
train_set = set(train_df["STRING_protein_id"].astype(str).unique())
val_set   = set(val_df["STRING_protein_id"].astype(str).unique())
test_set  = set(test_df["STRING_protein_id"].astype(str).unique())

tv_overlap = train_set & val_set
tt_overlap = train_set & test_set
vt_overlap = val_set & test_set

print("Train ∩ Val overlap :", len(tv_overlap))
print("Train ∩ Test overlap:", len(tt_overlap))
print("Val ∩ Test overlap  :", len(vt_overlap))

if len(tv_overlap) or len(tt_overlap) or len(vt_overlap):
    raise ValueError("Leakage detected across protein-level splits!")

# ------------------------------------------------------------
# 5. Summary statistics
# ------------------------------------------------------------
def summarize_split(name, part_df):
    print(f"\n[{name}]")
    print("Rows      :", len(part_df))
    print("Genes     :", part_df["GeneSymbol"].nunique())
    print("Proteins  :", part_df["STRING_protein_id"].nunique())
    print("Label dist:")
    print(part_df["Label"].value_counts(dropna=False).to_string())

print("\n[5] Split summaries")
summarize_split("TRAIN", train_df)
summarize_split("VAL", val_df)
summarize_split("TEST", test_df)

# ------------------------------------------------------------
# 6. Save outputs
# ------------------------------------------------------------
TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print("\n[6] Saved split files")
print("✅", TRAIN_OUT)
print("✅", VAL_OUT)
print("✅", TEST_OUT)

# ------------------------------------------------------------
# 7. Preview
# ------------------------------------------------------------
print("\n[7] Example test rows")
preview_cols = [c for c in [
    "VariationID", "GeneSymbol", "STRING_protein_id", "Name", "Label", "split"
] if c in test_df.columns]
print(test_df[preview_cols].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
