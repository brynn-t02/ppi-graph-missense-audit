import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_PATH  = ROOT / "data/processed/clinvar_test_main.csv"
ESM_PATH   = ROOT / "data/processed/esm_node_features.csv"

OUT_MISSING = ROOT / "data/processed/missing_esm_proteins.csv"

print("=" * 70)
print("🧬 Step 02c: Check ESM feature coverage on rebuilt main split")
print("=" * 70)

# ------------------------------------------------------------------
# Load split files
# ------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)
esm_df   = pd.read_csv(ESM_PATH)

print("\n[1] Input summary")
print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))
print("Test rows :", len(test_df))
print("ESM rows  :", len(esm_df))

# Unique protein sets
train_prots = set(train_df["STRING_protein_id"].astype(str).unique())
val_prots   = set(val_df["STRING_protein_id"].astype(str).unique())
test_prots  = set(test_df["STRING_protein_id"].astype(str).unique())
all_prots   = train_prots | val_prots | test_prots

esm_prots = set(esm_df["Protein_ID"].astype(str).unique())

print("\n[2] Protein coverage summary")
print("Total split proteins :", len(all_prots))
print("ESM feature proteins :", len(esm_prots))
print("Covered proteins     :", len(all_prots & esm_prots))
print("Missing proteins     :", len(all_prots - esm_prots))
print("Coverage rate        :", f"{len(all_prots & esm_prots) / len(all_prots) * 100:.2f}%")

# Per-split coverage
def report_split(name, prots):
    covered = len(prots & esm_prots)
    missing = len(prots - esm_prots)
    print(f"\n[{name}]")
    print("Proteins:", len(prots))
    print("Covered :", covered)
    print("Missing :", missing)
    print("Coverage:", f"{covered / len(prots) * 100:.2f}%")

report_split("TRAIN", train_prots)
report_split("VAL", val_prots)
report_split("TEST", test_prots)

# Save missing protein IDs
missing_prots = sorted(all_prots - esm_prots)
pd.DataFrame({"STRING_protein_id": missing_prots}).to_csv(OUT_MISSING, index=False)

print("\n[3] Missing protein IDs saved to:")
print(OUT_MISSING)

print("\n[4] Example missing proteins")
print(pd.Series(missing_prots[:20]).to_string(index=False))

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
