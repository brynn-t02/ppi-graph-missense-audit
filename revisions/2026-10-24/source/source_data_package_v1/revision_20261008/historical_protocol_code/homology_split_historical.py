import gzip
import json
import random
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path("/root")

SEQ_FASTA = ROOT / "data/raw/9606.protein.sequences.v12.0.fa.gz"

TRAIN_SPLIT_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_SPLIT_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/clinvar_test_main.csv"

OUT_DIR = ROOT / "data/processed/homology_aware_split"
TMP_DIR = OUT_DIR / "tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_MIN_SEQ_ID = 0.40
CLUSTER_COV = 0.50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("🧬 Step 14c: Build homology-aware split")
print("=" * 80)
print("FASTA      :", SEQ_FASTA)
print("Train split:", TRAIN_SPLIT_PATH)
print("Val split  :", VAL_SPLIT_PATH)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)
print("MMseqs min seq id:", CLUSTER_MIN_SEQ_ID)
print("Coverage threshold:", CLUSTER_COV)

# -------------------------------------------------------------------
# 1. Check mmseqs
# -------------------------------------------------------------------
res = subprocess.run(["which", "mmseqs"], capture_output=True, text=True)
if res.returncode != 0 or not res.stdout.strip():
    raise RuntimeError("mmseqs not found in PATH.")
print("mmseqs path:", res.stdout.strip())

# -------------------------------------------------------------------
# 2. Load split tables
# -------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_SPLIT_PATH, low_memory=False)
val_df   = pd.read_csv(VAL_SPLIT_PATH, low_memory=False)
test_df  = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

all_df = pd.concat([
    train_df.assign(_orig_split="train"),
    val_df.assign(_orig_split="val"),
    test_df.assign(_orig_split="test"),
], axis=0, ignore_index=True)

all_df["STRING_protein_id"] = all_df["STRING_protein_id"].astype(str)
all_prots = sorted(all_df["STRING_protein_id"].unique().tolist())

print("\n[1] Loaded split tables")
print("All rows       :", len(all_df))
print("Unique proteins:", len(all_prots))

# target ratios from original row counts
n_total = len(all_df)
target_train = len(train_df) / n_total
target_val   = len(val_df) / n_total
target_test  = len(test_df) / n_total

print("\n[2] Original row ratios")
print("train:", target_train)
print("val  :", target_val)
print("test :", target_test)

# -------------------------------------------------------------------
# 3. Parse FASTA and write subset FASTA
# -------------------------------------------------------------------
def parse_fasta_gz(path):
    seqs = {}
    cur_id = None
    cur_seq = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(cur_seq)
    return seqs

print("\n[3] Parsing FASTA...")
all_seqs = parse_fasta_gz(SEQ_FASTA)
seqs = {p: all_seqs[p] for p in all_prots if p in all_seqs}
missing = sorted(list(set(all_prots) - set(seqs.keys())))

print("Loaded sequences:", len(seqs))
print("Missing proteins :", len(missing))
if len(missing) > 0:
    print("Example missing:", missing[:10])

subset_fa = OUT_DIR / "all_split_proteins.fa"
with open(subset_fa, "w", encoding="utf-8") as f:
    for pid in all_prots:
        if pid in seqs:
            f.write(f">{pid}\n{seqs[pid]}\n")
print("✅ Wrote subset FASTA:", subset_fa)

# -------------------------------------------------------------------
# 4. Run MMseqs clustering
# -------------------------------------------------------------------
db = OUT_DIR / "split_proteins_db"
cluster_db = OUT_DIR / "split_proteins_clu"
rep_db = OUT_DIR / "split_proteins_rep"
tsv = OUT_DIR / "split_proteins_cluster.tsv"

print("\n[4] Running MMseqs clustering...")
subprocess.run(["mmseqs", "createdb", str(subset_fa), str(db)], check=True)

subprocess.run([
    "mmseqs", "cluster",
    str(db), str(cluster_db), str(TMP_DIR),
    "--min-seq-id", str(CLUSTER_MIN_SEQ_ID),
    "-c", str(CLUSTER_COV),
    "--cov-mode", "0",
    "--threads", "8",
], check=True)

subprocess.run([
    "mmseqs", "createtsv",
    str(db), str(db), str(cluster_db), str(tsv)
], check=True)

print("✅ Wrote cluster TSV:", tsv)

# -------------------------------------------------------------------
# 5. Parse cluster TSV
# createtsv format here is typically: representative \t member
# -------------------------------------------------------------------
cluster_pairs = pd.read_csv(tsv, sep="\t", header=None, names=["cluster_rep", "member"])
print("\n[5] Cluster TSV rows:", len(cluster_pairs))

# Build member -> cluster_rep map
member_to_cluster = {}
cluster_to_members = defaultdict(list)

for _, row in cluster_pairs.iterrows():
    rep = str(row["cluster_rep"])
    mem = str(row["member"])
    member_to_cluster[mem] = rep
    cluster_to_members[rep].append(mem)

# proteins that somehow don't appear: singleton cluster
for pid in all_prots:
    if pid not in member_to_cluster:
        member_to_cluster[pid] = pid
        cluster_to_members[pid].append(pid)

print("Unique clusters:", len(cluster_to_members))

cluster_sizes = [len(v) for v in cluster_to_members.values()]
print("Cluster size summary")
print("mean :", float(np.mean(cluster_sizes)))
print("median:", float(np.median(cluster_sizes)))
print("max  :", int(np.max(cluster_sizes)))

# -------------------------------------------------------------------
# 6. Assign clusters to homology-aware splits by greedy row-balance
# -------------------------------------------------------------------
all_df["_cluster_id"] = all_df["STRING_protein_id"].map(member_to_cluster)

cluster_row_counts = (
    all_df.groupby("_cluster_id").size().rename("n_rows").reset_index()
    .sort_values("n_rows", ascending=False)
    .reset_index(drop=True)
)

targets = {
    "train": int(round(target_train * len(all_df))),
    "val": int(round(target_val * len(all_df))),
    "test": int(round(target_test * len(all_df))),
}

assigned_counts = {"train": 0, "val": 0, "test": 0}
cluster_to_split = {}

def remaining_capacity(split):
    return targets[split] - assigned_counts[split]

for _, row in cluster_row_counts.iterrows():
    cid = row["_cluster_id"]
    n_rows_cluster = int(row["n_rows"])

    # choose split with largest remaining capacity
    candidate = max(["train", "val", "test"], key=lambda s: remaining_capacity(s))
    cluster_to_split[cid] = candidate
    assigned_counts[candidate] += n_rows_cluster

all_df["_homology_split"] = all_df["_cluster_id"].map(cluster_to_split)

# -------------------------------------------------------------------
# 7. Save homology-aware split tables
# -------------------------------------------------------------------
train_h = all_df[all_df["_homology_split"] == "train"].copy()
val_h   = all_df[all_df["_homology_split"] == "val"].copy()
test_h  = all_df[all_df["_homology_split"] == "test"].copy()

train_h.to_csv(OUT_DIR / "clinvar_train_homology.csv", index=False)
val_h.to_csv(OUT_DIR / "clinvar_val_homology.csv", index=False)
test_h.to_csv(OUT_DIR / "clinvar_test_homology.csv", index=False)

cluster_assign_df = pd.DataFrame({
    "cluster_id": list(cluster_to_split.keys()),
    "assigned_split": list(cluster_to_split.values()),
    "cluster_size_proteins": [len(cluster_to_members[c]) for c in cluster_to_split.keys()],
})
cluster_assign_df.to_csv(OUT_DIR / "homology_cluster_assignments.csv", index=False)

# -------------------------------------------------------------------
# 8. Summary
# -------------------------------------------------------------------
summary = {
    "n_total_rows": int(len(all_df)),
    "n_unique_proteins": int(len(all_prots)),
    "n_clusters": int(len(cluster_to_members)),
    "cluster_size_mean": float(np.mean(cluster_sizes)),
    "cluster_size_median": float(np.median(cluster_sizes)),
    "cluster_size_max": int(np.max(cluster_sizes)),
    "target_row_ratios": {
        "train": float(target_train),
        "val": float(target_val),
        "test": float(target_test),
    },
    "assigned_row_counts": {
        "train": int(len(train_h)),
        "val": int(len(val_h)),
        "test": int(len(test_h)),
    },
    "assigned_unique_proteins": {
        "train": int(train_h["STRING_protein_id"].nunique()),
        "val": int(val_h["STRING_protein_id"].nunique()),
        "test": int(test_h["STRING_protein_id"].nunique()),
    },
    "missing_sequences": int(len(missing)),
    "mmseqs_cluster_min_seq_id": CLUSTER_MIN_SEQ_ID,
    "mmseqs_cluster_coverage": CLUSTER_COV,
}

with open(OUT_DIR / "homology_split_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[6] Saved outputs")
print("✅", OUT_DIR / "clinvar_train_homology.csv")
print("✅", OUT_DIR / "clinvar_val_homology.csv")
print("✅", OUT_DIR / "clinvar_test_homology.csv")
print("✅", OUT_DIR / "homology_cluster_assignments.csv")
print("✅", OUT_DIR / "homology_split_summary.json")

print("\n[7] Homology-aware split summary")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
