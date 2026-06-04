#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import gzip
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu

# =========================
# Paths
# =========================
ESM_PATH = Path("/root/data/processed/esm_node_features_full.csv")
LINKS_PATH = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")
OUT_DIR = Path("/root/data/processed/esm_ppi_alignment_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
RANDOM_SEED = 42
MAX_NEIGHBOR_EDGES = 200000   # 可先控制规模，避免太慢
MAX_RANDOM_PAIRS = 200000
HIGH_CONF_THRESHOLD = 700     # STRING高置信边阈值（可调整）

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# Helper
# =========================
def cosine_similarity_batch(a, b):
    """
    a, b: np.ndarray shape [N, D]
    return: np.ndarray shape [N]
    """
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = (a_norm * b_norm) + 1e-12
    sim = np.sum(a * b, axis=1) / denom
    return sim

def cliffs_delta(x, y):
    """
    Cliff's delta effect size
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_x = len(x)
    n_y = len(y)
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (n_x * n_y)

# =========================
# Load ESM features
# =========================
print("[1/6] Loading ESM features...")
df_esm = pd.read_csv(ESM_PATH)

protein_col = "Protein_ID"
feat_cols = [c for c in df_esm.columns if c.startswith("Node_ESM_")]

print(f"ESM shape: {df_esm.shape}")
print(f"Feature dim: {len(feat_cols)}")

protein_ids = df_esm[protein_col].tolist()
emb_matrix = df_esm[feat_cols].values.astype(np.float32)

protein_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
protein_set = set(protein_ids)

# =========================
# Load STRING links
# =========================
print("[2/6] Loading STRING links...")
with gzip.open(LINKS_PATH, "rt") as f:
    # STRING links file is usually space-separated with header
    df_links = pd.read_csv(f, sep=r"\s+")

print(df_links.head())
print(f"Raw links shape: {df_links.shape}")

# Try common column names
col_candidates = df_links.columns.tolist()

if "protein1" in col_candidates and "protein2" in col_candidates:
    p1_col, p2_col = "protein1", "protein2"
else:
    raise ValueError(f"Cannot find protein1/protein2 in columns: {col_candidates}")

if "combined_score" in col_candidates:
    score_col = "combined_score"
else:
    score_col = None

# Keep only edges where both proteins exist in ESM table
df_links = df_links[
    df_links[p1_col].isin(protein_set) & df_links[p2_col].isin(protein_set)
].copy()

# Remove self-loops
df_links = df_links[df_links[p1_col] != df_links[p2_col]].copy()

print(f"Filtered links shape (proteins in ESM table): {df_links.shape}")

# Sample neighbor edges if too many
if len(df_links) > MAX_NEIGHBOR_EDGES:
    df_links = df_links.sample(MAX_NEIGHBOR_EDGES, random_state=RANDOM_SEED)

print(f"Neighbor edges used: {len(df_links)}")

# =========================
# Compute neighbor similarities
# =========================
print("[3/6] Computing neighbor-pair cosine similarities...")
idx1 = df_links[p1_col].map(protein_to_idx).values
idx2 = df_links[p2_col].map(protein_to_idx).values

neighbor_sims = cosine_similarity_batch(emb_matrix[idx1], emb_matrix[idx2])

neighbor_df = pd.DataFrame({
    "pair_type": "neighbor",
    "protein1": df_links[p1_col].values,
    "protein2": df_links[p2_col].values,
    "cosine_similarity": neighbor_sims
})

if score_col is not None:
    neighbor_df["combined_score"] = df_links[score_col].values

# =========================
# Generate random pairs
# =========================
print("[4/6] Sampling random protein pairs...")
all_indices = np.arange(len(protein_ids))
random_pairs = set()
target_n = min(MAX_RANDOM_PAIRS, len(neighbor_df))

while len(random_pairs) < target_n:
    a = np.random.randint(0, len(protein_ids))
    b = np.random.randint(0, len(protein_ids))
    if a == b:
        continue
    if a > b:
        a, b = b, a
    random_pairs.add((a, b))

random_pairs = list(random_pairs)
rand_idx1 = np.array([x[0] for x in random_pairs], dtype=int)
rand_idx2 = np.array([x[1] for x in random_pairs], dtype=int)

random_sims = cosine_similarity_batch(emb_matrix[rand_idx1], emb_matrix[rand_idx2])

random_df = pd.DataFrame({
    "pair_type": "random",
    "protein1": [protein_ids[i] for i in rand_idx1],
    "protein2": [protein_ids[i] for i in rand_idx2],
    "cosine_similarity": random_sims
})

# =========================
# Save merged pair table
# =========================
print("[5/6] Saving pairwise results...")
pair_df = pd.concat([neighbor_df, random_df], ignore_index=True)
pair_df.to_csv(OUT_DIR / "neighbor_vs_random_similarity.csv", index=False)

# =========================
# Statistical summary
# =========================
print("[6/6] Computing statistics...")
x = neighbor_df["cosine_similarity"].values
y = random_df["cosine_similarity"].values

u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
delta = cliffs_delta(x, y)

summary = {
    "n_neighbor_pairs": int(len(x)),
    "n_random_pairs": int(len(y)),
    "neighbor_mean": float(np.mean(x)),
    "neighbor_median": float(np.median(x)),
    "neighbor_std": float(np.std(x)),
    "random_mean": float(np.mean(y)),
    "random_median": float(np.median(y)),
    "random_std": float(np.std(y)),
    "mannwhitney_u": float(u_stat),
    "p_value": float(p_value),
    "cliffs_delta": float(delta),
}

# Optional: high-confidence edges only
if score_col is not None:
    hc = neighbor_df[neighbor_df["combined_score"] >= HIGH_CONF_THRESHOLD]
    if len(hc) > 0:
        x_hc = hc["cosine_similarity"].values
        u_hc, p_hc = mannwhitneyu(x_hc, y, alternative="two-sided")
        delta_hc = cliffs_delta(x_hc, y)
        summary["high_conf_threshold"] = HIGH_CONF_THRESHOLD
        summary["n_high_conf_neighbor_pairs"] = int(len(x_hc))
        summary["high_conf_neighbor_mean"] = float(np.mean(x_hc))
        summary["high_conf_neighbor_median"] = float(np.median(x_hc))
        summary["high_conf_p_value_vs_random"] = float(p_hc)
        summary["high_conf_cliffs_delta_vs_random"] = float(delta_hc)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# =========================
# Plot violin
# =========================
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.violinplot(data=pair_df, x="pair_type", y="cosine_similarity", inner="box")
plt.title("ESM cosine similarity: PPI neighbors vs random protein pairs")
plt.xlabel("")
plt.ylabel("Cosine similarity")
plt.tight_layout()
plt.savefig(OUT_DIR / "neighbor_vs_random_violin.png", dpi=300)
plt.close()

# =========================
# Plot histogram
# =========================
plt.figure(figsize=(8, 6))
sns.histplot(neighbor_df["cosine_similarity"], color="steelblue", label="Neighbor pairs",
             stat="density", bins=100, alpha=0.5)
sns.histplot(random_df["cosine_similarity"], color="orange", label="Random pairs",
             stat="density", bins=100, alpha=0.5)
plt.legend()
plt.title("ESM cosine similarity distribution")
plt.xlabel("Cosine similarity")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(OUT_DIR / "neighbor_vs_random_hist.png", dpi=300)
plt.close()

print("Done.")
print(json.dumps(summary, indent=2))
print(f"Results saved to: {OUT_DIR}")
