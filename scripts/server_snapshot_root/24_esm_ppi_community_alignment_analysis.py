#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import gzip
import random
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.stats import mannwhitneyu
from networkx.algorithms.community import greedy_modularity_communities

# =========================
# Paths
# =========================
ESM_PATH = Path("/root/data/processed/esm_node_features_full.csv")
LINKS_PATH = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")
OUT_DIR = Path("/root/data/processed/esm_ppi_community_alignment_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
RANDOM_SEED = 42
MAX_EDGES_FOR_GRAPH = 300000
MAX_WITHIN_PAIRS = 100000
MAX_BETWEEN_PAIRS = 100000
MIN_COMMUNITY_SIZE = 5
STRING_SCORE_THRESHOLD = 400

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# Helper
# =========================
def cosine_similarity_batch(a, b):
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = (a_norm * b_norm) + 1e-12
    sim = np.sum(a * b, axis=1) / denom
    return sim

# =========================
# Load ESM
# =========================
print("[1/7] Loading ESM features...")
df_esm = pd.read_csv(ESM_PATH)
protein_col = "Protein_ID"
feat_cols = [c for c in df_esm.columns if c.startswith("Node_ESM_")]

protein_ids = df_esm[protein_col].tolist()
emb_matrix = df_esm[feat_cols].values.astype(np.float32)
protein_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
protein_set = set(protein_ids)

print(f"ESM shape: {df_esm.shape}")

# =========================
# Load STRING links
# =========================
print("[2/7] Loading STRING links...")
with gzip.open(LINKS_PATH, "rt") as f:
    df_links = pd.read_csv(f, sep=r"\s+")

df_links = df_links[
    (df_links["protein1"].isin(protein_set)) &
    (df_links["protein2"].isin(protein_set)) &
    (df_links["protein1"] != df_links["protein2"]) &
    (df_links["combined_score"] >= STRING_SCORE_THRESHOLD)
].copy()

if len(df_links) > MAX_EDGES_FOR_GRAPH:
    df_links = df_links.sample(MAX_EDGES_FOR_GRAPH, random_state=RANDOM_SEED)

print(f"Filtered links used for graph: {df_links.shape}")

# =========================
# Build graph
# =========================
print("[3/7] Building graph...")
G = nx.Graph()
G.add_edges_from(df_links[["protein1", "protein2"]].itertuples(index=False, name=None))
print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

# =========================
# Community detection
# =========================
print("[4/7] Detecting communities...")
communities = list(greedy_modularity_communities(G))
communities = [list(c) for c in communities if len(c) >= MIN_COMMUNITY_SIZE]

print(f"Detected communities (size >= {MIN_COMMUNITY_SIZE}): {len(communities)}")

# map node -> community id
node_to_comm = {}
for cid, comm in enumerate(communities):
    for node in comm:
        node_to_comm[node] = cid

community_sizes = [len(c) for c in communities]

# =========================
# Sample within-community pairs
# =========================
print("[5/7] Sampling within-community pairs...")
within_pairs = []

for comm in communities:
    if len(comm) < 2:
        continue
    comm_indices = [protein_to_idx[p] for p in comm if p in protein_to_idx]
    if len(comm_indices) < 2:
        continue

    # sample combinations if too many
    all_pairs = list(combinations(comm_indices, 2))
    if len(all_pairs) > 2000:
        sampled = random.sample(all_pairs, 2000)
    else:
        sampled = all_pairs
    within_pairs.extend(sampled)

if len(within_pairs) > MAX_WITHIN_PAIRS:
    within_pairs = random.sample(within_pairs, MAX_WITHIN_PAIRS)

print(f"Within-community pairs used: {len(within_pairs)}")

# =========================
# Sample between-community pairs
# =========================
print("[6/7] Sampling between-community pairs...")
all_nodes = list(node_to_comm.keys())
between_pairs = set()

while len(between_pairs) < min(MAX_BETWEEN_PAIRS, len(within_pairs)):
    a, b = random.sample(all_nodes, 2)
    if node_to_comm[a] == node_to_comm[b]:
        continue
    ia = protein_to_idx[a]
    ib = protein_to_idx[b]
    if ia > ib:
        ia, ib = ib, ia
    between_pairs.add((ia, ib))

between_pairs = list(between_pairs)
print(f"Between-community pairs used: {len(between_pairs)}")

# =========================
# Compute cosine similarities
# =========================
print("[7/7] Computing similarities...")
within_idx1 = np.array([x[0] for x in within_pairs], dtype=int)
within_idx2 = np.array([x[1] for x in within_pairs], dtype=int)
between_idx1 = np.array([x[0] for x in between_pairs], dtype=int)
between_idx2 = np.array([x[1] for x in between_pairs], dtype=int)

within_sims = cosine_similarity_batch(emb_matrix[within_idx1], emb_matrix[within_idx2])
between_sims = cosine_similarity_batch(emb_matrix[between_idx1], emb_matrix[between_idx2])

within_df = pd.DataFrame({
    "pair_type": "within_community",
    "cosine_similarity": within_sims
})
between_df = pd.DataFrame({
    "pair_type": "between_community",
    "cosine_similarity": between_sims
})

pair_df = pd.concat([within_df, between_df], ignore_index=True)
pair_df.to_csv(OUT_DIR / "within_vs_between_community_similarity.csv", index=False)

# stats
x = within_sims
y = between_sims
u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")

summary = {
    "n_graph_nodes": int(G.number_of_nodes()),
    "n_graph_edges": int(G.number_of_edges()),
    "n_communities": int(len(communities)),
    "community_size_mean": float(np.mean(community_sizes)) if community_sizes else 0.0,
    "community_size_median": float(np.median(community_sizes)) if community_sizes else 0.0,
    "n_within_pairs": int(len(x)),
    "n_between_pairs": int(len(y)),
    "within_mean": float(np.mean(x)),
    "within_median": float(np.median(x)),
    "within_std": float(np.std(x)),
    "between_mean": float(np.mean(y)),
    "between_median": float(np.median(y)),
    "between_std": float(np.std(y)),
    "mean_difference": float(np.mean(x) - np.mean(y)),
    "median_difference": float(np.median(x) - np.median(y)),
    "mannwhitney_u": float(u_stat),
    "p_value": float(p_value),
    "string_score_threshold": STRING_SCORE_THRESHOLD,
    "min_community_size": MIN_COMMUNITY_SIZE
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# plots
plt.figure(figsize=(8, 6))
data_to_plot = [within_sims, between_sims]
plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
plt.xticks([1, 2], ["within_community", "between_community"], rotation=10)
plt.ylabel("Cosine similarity")
plt.title("ESM similarity: within vs between PPI communities")
plt.tight_layout()
plt.savefig(OUT_DIR / "within_vs_between_community_violin.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(within_sims, bins=100, alpha=0.5, density=True, label="Within community", color="steelblue")
plt.hist(between_sims, bins=100, alpha=0.5, density=True, label="Between community", color="orange")
plt.legend()
plt.xlabel("Cosine similarity")
plt.ylabel("Density")
plt.title("ESM similarity distribution by PPI community relation")
plt.tight_layout()
plt.savefig(OUT_DIR / "within_vs_between_community_hist.png", dpi=300)
plt.close()

print(json.dumps(summary, indent=2))
print(f"Results saved to: {OUT_DIR}")