import pandas as pd
from pathlib import Path

ROOT = Path("/root")
IN_PATH = ROOT / "data/processed/clinvar_missense_reviewed_dedup_mapped.csv"

print("=" * 80)
print("🔍 Step 07a: Inspect fields needed for AlphaMissense matching")
print("=" * 80)
print("Input:", IN_PATH)

df = pd.read_csv(IN_PATH, nrows=5, low_memory=False)

print("\n[1] Columns")
print(list(df.columns))

candidate_cols = [c for c in df.columns if any(
    key in c.lower() for key in [
        "hgvs", "protein", "gene", "symbol", "name", "amino", "aa", "string", "variation"
    ]
)]

print("\n[2] Potential AlphaMissense-matching columns")
print(candidate_cols if candidate_cols else "None")

print("\n[3] Preview")
print(df.head(5).to_string(index=False))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
