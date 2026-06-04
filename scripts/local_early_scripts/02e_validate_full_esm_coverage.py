import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_PATH  = ROOT / "data/processed/clinvar_test_main.csv"
ESM_PATH   = ROOT / "data/processed/esm_node_features_full.csv"

print("=" * 70)
print("✅ Step 02e: Validate full ESM coverage on rebuilt main split")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)
esm_df   = pd.read_csv(ESM_PATH)

train_prots = set(train_df["STRING_protein_id"].astype(str).unique())
val_prots   = set(val_df["STRING_protein_id"].astype(str).unique())
test_prots  = set(test_df["STRING_protein_id"].astype(str).unique())
all_prots   = train_prots | val_prots | test_prots

esm_prots = set(esm_df["Protein_ID"].astype(str).unique())

print("\n[1] Global coverage")
print("Split proteins total:", len(all_prots))
print("ESM proteins total  :", len(esm_prots))
print("Covered proteins    :", len(all_prots & esm_prots))
print("Missing proteins    :", len(all_prots - esm_prots))
print("Coverage rate       :", f"{len(all_prots & esm_prots) / len(all_prots) * 100:.2f}%")

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

missing = sorted(all_prots - esm_prots)
if len(missing) > 0:
    print("\nExamples missing proteins:")
    print(pd.Series(missing[:20]).to_string(index=False))
else:
    print("\nNo missing proteins. Full coverage achieved.")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
