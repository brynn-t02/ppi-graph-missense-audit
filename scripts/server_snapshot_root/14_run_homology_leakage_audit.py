import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
TRAIN_SPLIT_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_SPLIT_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/clinvar_test_main.csv"

OUT_DIR = ROOT / "data/processed/homology_leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🧬 Step 14: Homology leakage audit")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("Train split:", TRAIN_SPLIT_PATH)
print("Val split  :", VAL_SPLIT_PATH)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)

# -------------------------------------------------------------------
# 1. Load main input and split tables
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)
x = payload["x"].float().cpu().numpy()

train_idx = payload["train_protein_idx"].cpu().numpy()
val_idx   = payload["val_protein_idx"].cpu().numpy()
test_idx  = payload["test_protein_idx"].cpu().numpy()

train_df = pd.read_csv(TRAIN_SPLIT_PATH, low_memory=False)
val_df   = pd.read_csv(VAL_SPLIT_PATH, low_memory=False)
test_df  = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

print("\n[1] Loaded inputs")
print("Node embedding matrix:", x.shape)
print("Train protein refs   :", len(train_idx))
print("Val protein refs     :", len(val_idx))
print("Test protein refs    :", len(test_idx))
print("Train rows           :", len(train_df))
print("Val rows             :", len(val_df))
print("Test rows            :", len(test_df))

# -------------------------------------------------------------------
# 2. Audit exact protein-ID overlap
# -------------------------------------------------------------------
train_prots = set(train_df["STRING_protein_id"].astype(str).unique())
val_prots   = set(val_df["STRING_protein_id"].astype(str).unique())
test_prots  = set(test_df["STRING_protein_id"].astype(str).unique())

exact_overlap = {
    "train_val_overlap_n": len(train_prots & val_prots),
    "train_test_overlap_n": len(train_prots & test_prots),
    "val_test_overlap_n": len(val_prots & test_prots),
    "n_train_proteins": len(train_prots),
    "n_val_proteins": len(val_prots),
    "n_test_proteins": len(test_prots),
}

print("\n[2] Exact protein overlap audit")
for k, v in exact_overlap.items():
    print(f"{k}: {v}")

# -------------------------------------------------------------------
# 3. Normalize embeddings for cosine similarity audit
# -------------------------------------------------------------------
def l2_normalize(arr, eps=1e-12):
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norm, eps, None)

x_norm = l2_normalize(x)

train_emb = x_norm[train_idx]
val_emb   = x_norm[val_idx]
test_emb  = x_norm[test_idx]

# -------------------------------------------------------------------
# 4. For each split, compute nearest-neighbor cosine similarity to TRAIN
# -------------------------------------------------------------------
def nearest_train_cos(query_emb, ref_emb, batch_size=512):
    max_sims = []
    argmax_idx = []

    for start in range(0, len(query_emb), batch_size):
        q = query_emb[start:start+batch_size]  # [B, D]
        sims = q @ ref_emb.T                  # cosine because normalized
        local_max = np.max(sims, axis=1)
        local_argmax = np.argmax(sims, axis=1)

        max_sims.append(local_max)
        argmax_idx.append(local_argmax)

    return np.concatenate(max_sims), np.concatenate(argmax_idx)

val_to_train_sim, val_to_train_nn = nearest_train_cos(val_emb, train_emb)
test_to_train_sim, test_to_train_nn = nearest_train_cos(test_emb, train_emb)

# also test to train+val reference
trainval_idx = np.concatenate([train_idx, val_idx])
trainval_emb = x_norm[trainval_idx]
test_to_trainval_sim, test_to_trainval_nn = nearest_train_cos(test_emb, trainval_emb)

# -------------------------------------------------------------------
# 5. Summaries
# -------------------------------------------------------------------
def summarize_sims(name, sims):
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]

    summary = {
        "name": name,
        "n": int(len(sims)),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "q25": float(np.quantile(sims, 0.25)),
        "median": float(np.quantile(sims, 0.50)),
        "q75": float(np.quantile(sims, 0.75)),
        "max": float(np.max(sims)),
    }

    for t in thresholds:
        summary[f"frac_ge_{t:.2f}"] = float(np.mean(sims >= t))
        summary[f"count_ge_{t:.2f}"] = int(np.sum(sims >= t))

    return summary

val_summary = summarize_sims("val_to_train", val_to_train_sim)
test_summary = summarize_sims("test_to_train", test_to_train_sim)
test_trainval_summary = summarize_sims("test_to_trainval", test_to_trainval_sim)

print("\n[3] Cosine similarity leakage proxy summary")
for block in [val_summary, test_summary, test_trainval_summary]:
    print("-" * 60)
    print(block["name"])
    for k, v in block.items():
        if k == "name":
            continue
        print(f"{k}: {v}")

# -------------------------------------------------------------------
# 6. Save detailed CSVs
# -------------------------------------------------------------------
val_detail = pd.DataFrame({
    "split": "val",
    "query_idx": val_idx,
    "max_cos_to_train": val_to_train_sim,
    "nn_train_idx_position": val_to_train_nn,
})
test_detail = pd.DataFrame({
    "split": "test",
    "query_idx": test_idx,
    "max_cos_to_train": test_to_train_sim,
    "nn_train_idx_position": test_to_train_nn,
    "max_cos_to_trainval": test_to_trainval_sim,
    "nn_trainval_idx_position": test_to_trainval_nn,
})

val_detail.to_csv(OUT_DIR / "val_to_train_cosine_audit.csv", index=False)
test_detail.to_csv(OUT_DIR / "test_to_train_cosine_audit.csv", index=False)

report = {
    "exact_protein_overlap": exact_overlap,
    "val_to_train_summary": val_summary,
    "test_to_train_summary": test_summary,
    "test_to_trainval_summary": test_trainval_summary,
    "notes": [
        "This audit uses normalized ESM embedding cosine similarity as a practical proxy for sequence-level homology risk.",
        "It does not replace a true sequence-identity-based clustering audit (e.g. CD-HIT/MMseqs2), but provides a useful first-pass diagnostic."
    ]
}

with open(OUT_DIR / "homology_leakage_audit_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n[4] Saved outputs")
print("✅", OUT_DIR / "val_to_train_cosine_audit.csv")
print("✅", OUT_DIR / "test_to_train_cosine_audit.csv")
print("✅", OUT_DIR / "homology_leakage_audit_report.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
