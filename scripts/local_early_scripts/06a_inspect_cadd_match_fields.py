import pandas as pd
from pathlib import Path

ROOT = Path("/root")

CANDIDATES = [
    ROOT / "data/processed/clinvar_missense_reviewed_dedup_mapped.csv",
    ROOT / "data/processed/clinvar_missense_reviewed_dedup.csv",
    ROOT / "data/processed/clinvar_missense_reviewed.csv",
    ROOT / "data/processed/clinvar_missense_clean.csv",
]

print("=" * 80)
print("🔍 Step 06a: Inspect fields needed for CADD matching")
print("=" * 80)

for path in CANDIDATES:
    if not path.exists():
        print(f"❌ Missing: {path}")
        continue

    print("\n" + "-" * 80)
    print("File:", path)
    df = pd.read_csv(path, nrows=5, low_memory=False)

    print("Columns:")
    print(list(df.columns))

    candidate_cols = [c for c in df.columns if any(
        key in c.lower() for key in [
            "chrom", "start", "stop", "position", "ref", "alt", "vcf", "assembly"
        ]
    )]

    print("\nPotential CADD-matching columns:")
    print(candidate_cols if candidate_cols else "None")

    print("\nPreview:")
    print(df.head(3).to_string(index=False))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
