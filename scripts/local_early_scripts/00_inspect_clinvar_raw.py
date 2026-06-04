import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/variant_summary.txt.gz"

print("=" * 70)
print("🔍 Inspect raw ClinVar file")
print("=" * 70)
print("Input:", RAW_PATH)

# 只读前200行，防止内存炸
df = pd.read_csv(RAW_PATH, sep="\t", nrows=200, low_memory=False)

print("\n[1] Shape of preview:")
print(df.shape)

print("\n[2] Columns:")
for i, col in enumerate(df.columns):
    print(f"{i:02d}: {col}")

print("\n[3] Preview rows:")
print(df.head(5).to_string())

# 搜索可能和蛋白变化、变异类型、标签有关的列
keywords = ["name", "type", "clinical", "review", "protein", "amino", "hgvs", "gene", "consequence"]

print("\n[4] Candidate columns by keyword:")
for col in df.columns:
    low = col.lower()
    if any(k in low for k in keywords):
        print(col)

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
