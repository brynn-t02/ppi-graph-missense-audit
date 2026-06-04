import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root")

SEQ_FASTA = ROOT / "data/raw/9606.protein.sequences.v12.0.fa.gz"
TRAIN_SPLIT_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_SPLIT_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/clinvar_test_main.csv"

OUT_DIR = ROOT / "data/processed/sequence_homology_audit_fast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 3
N_SAMPLE = 300
LEN_RATIO_LOW = 0.80
LEN_RATIO_HIGH = 1.25
SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 80)
print("🧬 Step 14bA-fast: Sequence homology audit (fast k-mer proxy)")
print("=" * 80)
print("FASTA      :", SEQ_FASTA)
print("Train split:", TRAIN_SPLIT_PATH)
print("Val split  :", VAL_SPLIT_PATH)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)
print("k-mer size :", K)
print("Sample size:", N_SAMPLE)

# -------------------------------------------------------------------
# 1. Load unique proteins
# -------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_SPLIT_PATH, low_memory=False)
val_df   = pd.read_csv(VAL_SPLIT_PATH, low_memory=False)
test_df  = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

train_prots = sorted(train_df["STRING_protein_id"].astype(str).unique().tolist())
val_prots   = sorted(val_df["STRING_protein_id"].astype(str).unique().tolist())
test_prots  = sorted(test_df["STRING_protein_id"].astype(str).unique().tolist())

need_prots = set(train_prots) | set(val_prots) | set(test_prots)

print("\n[1] Unique proteins")
print("Train:", len(train_prots))
print("Val  :", len(val_prots))
print("Test :", len(test_prots))
print("Union:", len(need_prots))

# -------------------------------------------------------------------
# 2. Parse FASTA
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

all_seqs = parse_fasta_gz(SEQ_FASTA)
seqs = {p: all_seqs[p] for p in need_prots if p in all_seqs}
missing = sorted(list(need_prots - set(seqs.keys())))

print("\n[2] FASTA coverage")
print("Loaded needed sequences:", len(seqs))
print("Missing protein sequences:", len(missing))

train_prots = [p for p in train_prots if p in seqs]
val_prots   = [p for p in val_prots if p in seqs]
test_prots  = [p for p in test_prots if p in seqs]

# -------------------------------------------------------------------
# 3. Sample val/test for fast audit
# -------------------------------------------------------------------
def sample_ids(ids, n):
    if len(ids) <= n:
        return ids
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(idx)]

val_sample = sample_ids(val_prots, N_SAMPLE)
test_sample = sample_ids(test_prots, N_SAMPLE)

print("\n[3] Sampled proteins")
print("Val sampled :", len(val_sample))
print("Test sampled:", len(test_sample))

# -------------------------------------------------------------------
# 4. k-mer helpers
# -------------------------------------------------------------------
def kmers(seq, k=3):
    if len(seq) < k:
        return {seq}
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}

def jaccard(a, b):
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u

kmer_sets = {p: kmers(seqs[p], K) for p in set(train_prots) | set(val_sample) | set(test_sample)}
seq_lens = {p: len(seqs[p]) for p in set(train_prots) | set(val_sample) | set(test_sample)}

# bucket train proteins by coarse length
train_by_len = {}
for p in train_prots:
    train_by_len[p] = seq_lens[p]

# -------------------------------------------------------------------
# 5. Fast nearest-neighbor search with length filter
# -------------------------------------------------------------------
def get_length_filtered_refs(query_len, ref_ids):
    low = int(query_len * LEN_RATIO_LOW)
    high = int(query_len * LEN_RATIO_HIGH)
    out = [r for r in ref_ids if low <= seq_lens[r] <= high]
    # fallback if empty
    if len(out) == 0:
        out = ref_ids
    return out

def nearest_similarity_fast(query_ids, ref_ids):
    rows = []
    for i, q in enumerate(query_ids, 1):
        qset = kmer_sets[q]
        qlen = seq_lens[q]

        candidate_refs = get_length_filtered_refs(qlen, ref_ids)

        best_ref = None
        best_sim = -1.0
        for r in candidate_refs:
            sim = jaccard(qset, kmer_sets[r])
            if sim > best_sim:
                best_sim = sim
                best_ref = r

        rows.append({
            "query_protein": q,
            "query_len": qlen,
            "n_candidates_after_len_filter": len(candidate_refs),
            "nearest_ref_protein": best_ref,
            "nearest_ref_len": seq_lens[best_ref] if best_ref else np.nan,
            "max_kmer_jaccard": best_sim,
        })

        if i % 50 == 0 or i == len(query_ids):
            print(f"  processed {i}/{len(query_ids)} queries")

    return pd.DataFrame(rows)

print("\n[4] Running fast nearest-neighbor sequence proxy audit...")
val_to_train = nearest_similarity_fast(val_sample, train_prots)
test_to_train = nearest_similarity_fast(test_sample, train_prots)

# -------------------------------------------------------------------
# 6. Summaries
# -------------------------------------------------------------------
def summarize(df, col):
    arr = df[col].values
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
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        out[f"frac_ge_{t:.1f}"] = float(np.mean(arr >= t))
        out[f"count_ge_{t:.1f}"] = int(np.sum(arr >= t))
    return out

val_summary = summarize(val_to_train, "max_kmer_jaccard")
test_summary = summarize(test_to_train, "max_kmer_jaccard")

print("\n[5] Fast sequence similarity proxy summary")
for name, block in [
    ("val_to_train_sample", val_summary),
    ("test_to_train_sample", test_summary),
]:
    print("-" * 60)
    print(name)
    for k, v in block.items():
        print(f"{k}: {v}")

# -------------------------------------------------------------------
# 7. Save outputs
# -------------------------------------------------------------------
val_to_train.to_csv(OUT_DIR / "val_to_train_sequence_proxy_fast.csv", index=False)
test_to_train.to_csv(OUT_DIR / "test_to_train_sequence_proxy_fast.csv", index=False)

report = {
    "mode": "fast_sampled_kmer_proxy",
    "kmer_size": K,
    "sample_size": N_SAMPLE,
    "len_ratio_low": LEN_RATIO_LOW,
    "len_ratio_high": LEN_RATIO_HIGH,
    "n_missing_sequences": int(len(missing)),
    "val_to_train_sample_summary": val_summary,
    "test_to_train_sample_summary": test_summary,
    "notes": [
        "This fast audit uses sampled queries and length-filtered nearest-neighbor k-mer Jaccard similarity.",
        "It is intended as a practical sequence-level homology proxy, not a replacement for alignment-based percent identity."
    ]
}

with open(OUT_DIR / "sequence_homology_audit_fast_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n[6] Saved outputs")
print("✅", OUT_DIR / "val_to_train_sequence_proxy_fast.csv")
print("✅", OUT_DIR / "test_to_train_sequence_proxy_fast.csv")
print("✅", OUT_DIR / "sequence_homology_audit_fast_report.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
