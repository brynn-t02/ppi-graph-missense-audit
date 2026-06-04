import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root")

SEQ_FASTA = ROOT / "data/raw/9606.protein.sequences.v12.0.fa.gz"
TRAIN_SPLIT_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_SPLIT_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/clinvar_test_main.csv"

OUT_DIR = ROOT / "data/processed/mmseqs_homology_audit"
TMP_DIR = OUT_DIR / "tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🧬 Step 14bB: MMseqs2 sequence homology audit")
print("=" * 80)
print("FASTA      :", SEQ_FASTA)
print("Train split:", TRAIN_SPLIT_PATH)
print("Val split  :", VAL_SPLIT_PATH)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)

# -------------------------------------------------------------------
# 1. Check mmseqs
# -------------------------------------------------------------------
res = subprocess.run(["which", "mmseqs"], capture_output=True, text=True)
if res.returncode != 0 or not res.stdout.strip():
    raise RuntimeError("mmseqs not found in PATH. Please install mmseqs2 first.")
print("mmseqs path:", res.stdout.strip())

# -------------------------------------------------------------------
# 2. Load protein IDs
# -------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_SPLIT_PATH, low_memory=False)
val_df   = pd.read_csv(VAL_SPLIT_PATH, low_memory=False)
test_df  = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

train_prots = sorted(train_df["STRING_protein_id"].astype(str).unique().tolist())
val_prots   = sorted(val_df["STRING_protein_id"].astype(str).unique().tolist())
test_prots  = sorted(test_df["STRING_protein_id"].astype(str).unique().tolist())
need = set(train_prots) | set(val_prots) | set(test_prots)

print("\n[1] Unique proteins")
print("Train:", len(train_prots))
print("Val  :", len(val_prots))
print("Test :", len(test_prots))
print("Union:", len(need))

# -------------------------------------------------------------------
# 3. Parse FASTA and write split FASTA files
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

print("\n[2] Parsing FASTA...")
all_seqs = parse_fasta_gz(SEQ_FASTA)
seqs = {p: all_seqs[p] for p in need if p in all_seqs}
missing = sorted(list(need - set(seqs.keys())))

print("Loaded needed sequences:", len(seqs))
print("Missing protein sequences:", len(missing))
if len(missing) > 0:
    print("Example missing:", missing[:10])

def write_fasta(ids, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for pid in ids:
            if pid in seqs:
                f.write(f">{pid}\n{seqs[pid]}\n")

train_fa = OUT_DIR / "train.fa"
val_fa = OUT_DIR / "val.fa"
test_fa = OUT_DIR / "test.fa"
trainval_fa = OUT_DIR / "trainval.fa"

write_fasta(train_prots, train_fa)
write_fasta(val_prots, val_fa)
write_fasta(test_prots, test_fa)
write_fasta(train_prots + val_prots, trainval_fa)

print("✅ Wrote subset FASTA files")

# -------------------------------------------------------------------
# 4. MMseqs helper
# -------------------------------------------------------------------
def run_mmseqs_top1(query_fa, target_fa, prefix):
    qdb = OUT_DIR / f"{prefix}_qdb"
    tdb = OUT_DIR / f"{prefix}_tdb"
    rdb = OUT_DIR / f"{prefix}_res"
    out_tsv = OUT_DIR / f"{prefix}.tsv"

    print(f"\n[MMseqs] {prefix}")
    subprocess.run(["mmseqs", "createdb", str(query_fa), str(qdb)], check=True)
    subprocess.run(["mmseqs", "createdb", str(target_fa), str(tdb)], check=True)

    subprocess.run([
        "mmseqs", "search",
        str(qdb), str(tdb), str(rdb), str(TMP_DIR),
        "--max-seqs", "1",
        "-e", "10",
        "--min-seq-id", "0.0",
        "-s", "7.5",
        "--threads", "8",
    ], check=True)

    subprocess.run([
        "mmseqs", "convertalis",
        str(qdb), str(tdb), str(rdb), str(out_tsv),
        "--format-output",
        "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    ], check=True)

    return out_tsv

# -------------------------------------------------------------------
# 5. Run searches
# -------------------------------------------------------------------
val_train_tsv = run_mmseqs_top1(val_fa, train_fa, "val_to_train")
test_train_tsv = run_mmseqs_top1(test_fa, train_fa, "test_to_train")
test_trainval_tsv = run_mmseqs_top1(test_fa, trainval_fa, "test_to_trainval")

# -------------------------------------------------------------------
# 6. Summaries
# -------------------------------------------------------------------
cols = ["query", "target", "pident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits"]

def load_summary(path):
    df = pd.read_csv(path, sep="\t", names=cols)
    arr = pd.to_numeric(df["pident"], errors="coerce").dropna().values

    out = {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }
    for t in [30, 40, 50, 60, 80, 90]:
        out[f"frac_ge_{t}"] = float(np.mean(arr >= t))
        out[f"count_ge_{t}"] = int(np.sum(arr >= t))
    return df, out

val_hits, val_summary = load_summary(val_train_tsv)
test_hits, test_summary = load_summary(test_train_tsv)
test_trainval_hits, test_trainval_summary = load_summary(test_trainval_tsv)

print("\n[3] MMseqs homology summary")
for name, block in [
    ("val_to_train", val_summary),
    ("test_to_train", test_summary),
    ("test_to_trainval", test_trainval_summary),
]:
    print("-" * 60)
    print(name)
    for k, v in block.items():
        print(f"{k}: {v}")

# -------------------------------------------------------------------
# 7. Save report
# -------------------------------------------------------------------
report = {
    "n_missing_sequences": int(len(missing)),
    "val_to_train_summary": val_summary,
    "test_to_train_summary": test_summary,
    "test_to_trainval_summary": test_trainval_summary,
    "notes": [
        "Top-1 MMseqs2 hit statistics by percent identity (pident).",
        "This is a standard sequence-level homology audit and is more interpretable than embedding cosine similarity proxies."
    ]
}

with open(OUT_DIR / "mmseqs_homology_audit_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n[4] Saved outputs")
print("✅", val_train_tsv)
print("✅", test_train_tsv)
print("✅", test_trainval_tsv)
print("✅", OUT_DIR / "mmseqs_homology_audit_report.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
