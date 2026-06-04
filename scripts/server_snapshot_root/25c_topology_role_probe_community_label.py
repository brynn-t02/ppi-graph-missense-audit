#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from networkx.algorithms.community import greedy_modularity_communities

# =========================
# Paths
# =========================
ESM_PATH = Path("/root/data/processed/esm_node_features_full.csv")
GCN_PATH = Path("/root/data/processed/gcn_embeddings/gcn_node_embeddings.npy")
LINKS_PATH = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")
OUT_DIR = Path("/root/data/processed/topology_role_probe_community_label")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
RANDOM_SEED = 42
N_SPLITS = 5
STRING_SCORE_THRESHOLD = 400
MIN_COMMUNITY_SIZE = 50

# =========================
# Load ESM
# =========================
print("[1/6] Loading raw ESM features...")
df_esm = pd.read_csv(ESM_PATH)
protein_ids = df_esm["Protein_ID"].tolist()
esm_feat_cols = [c for c in df_esm.columns if c.startswith("Node_ESM_")]
X_esm = df_esm[esm_feat_cols].values.astype(np.float32)

protein_set = set(protein_ids)
pid_to_row = {pid: i for i, pid in enumerate(protein_ids)}

print(f"Raw ESM shape: {X_esm.shape}")

# =========================
# Load GCN embeddings
# =========================
print("[2/6] Loading GCN embeddings...")
X_gcn = np.load(GCN_PATH).astype(np.float32)
print(f"GCN embedding shape: {X_gcn.shape}")

assert X_gcn.shape[0] == X_esm.shape[0], \
    "GCN embedding rows do not match ESM rows."

# =========================
# Build graph
# =========================
print("[3/6] Loading STRING graph...")
with gzip.open(LINKS_PATH, "rt") as f:
    df_links = pd.read_csv(f, sep=r"\s+")

df_links = df_links[
    (df_links["protein1"].isin(protein_set)) &
    (df_links["protein2"].isin(protein_set)) &
    (df_links["protein1"] != df_links["protein2"]) &
    (df_links["combined_score"] >= STRING_SCORE_THRESHOLD)
].copy()

G = nx.Graph()
G.add_edges_from(df_links[["protein1", "protein2"]].itertuples(index=False, name=None))

print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

# =========================
# Community detection
# =========================
print("[4/6] Detecting communities...")
communities = list(greedy_modularity_communities(G))
communities = [list(c) for c in communities if len(c) >= MIN_COMMUNITY_SIZE]

print(f"Communities kept (size >= {MIN_COMMUNITY_SIZE}): {len(communities)}")

node_to_comm = {}
for cid, comm in enumerate(communities):
    for node in comm:
        node_to_comm[node] = cid

nodes_used = [pid for pid in protein_ids if pid in node_to_comm]
y = np.array([node_to_comm[pid] for pid in nodes_used], dtype=np.int64)

rows = np.array([pid_to_row[pid] for pid in nodes_used], dtype=np.int64)
X_esm_used = X_esm[rows]
X_gcn_used = X_gcn[rows]

class_counts = pd.Series(y).value_counts().sort_index().to_dict()
print(f"Nodes used for community probe: {len(nodes_used)}")
print(f"Community class counts: {class_counts}")

# =========================
# Probe classifier
# =========================
print("[5/6] Running community label probe...")
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=3000,
        multi_class="auto",
        random_state=RANDOM_SEED
    ))
])

scoring = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(f1_score, average="macro")
}

res_esm = cross_validate(clf, X_esm_used, y, cv=cv, scoring=scoring, n_jobs=-1)
res_gcn = cross_validate(clf, X_gcn_used, y, cv=cv, scoring=scoring, n_jobs=-1)

summary = {
    "task": "community_label_probe",
    "n_nodes_used": int(len(nodes_used)),
    "n_communities_used": int(len(class_counts)),
    "community_counts": {str(k): int(v) for k, v in class_counts.items()},
    "string_score_threshold": STRING_SCORE_THRESHOLD,
    "min_community_size": MIN_COMMUNITY_SIZE,
    "raw_esm": {
        "accuracy_mean": float(np.mean(res_esm["test_accuracy"])),
        "accuracy_std": float(np.std(res_esm["test_accuracy"])),
        "macro_f1_mean": float(np.mean(res_esm["test_macro_f1"])),
        "macro_f1_std": float(np.std(res_esm["test_macro_f1"]))
    },
    "gcn_enhanced": {
        "accuracy_mean": float(np.mean(res_gcn["test_accuracy"])),
        "accuracy_std": float(np.std(res_gcn["test_accuracy"])),
        "macro_f1_mean": float(np.mean(res_gcn["test_macro_f1"])),
        "macro_f1_std": float(np.std(res_gcn["test_macro_f1"]))
    },
    "delta_gcn_minus_esm": {
        "accuracy_mean_diff": float(np.mean(res_gcn["test_accuracy"]) - np.mean(res_esm["test_accuracy"])),
        "macro_f1_mean_diff": float(np.mean(res_gcn["test_macro_f1"]) - np.mean(res_esm["test_macro_f1"]))
    },
    "assumption": "GCN embedding row order follows esm_node_features_full.csv Protein_ID order."
}

fold_df = pd.DataFrame({
    "fold": list(range(1, N_SPLITS + 1)),
    "esm_accuracy": res_esm["test_accuracy"],
    "esm_macro_f1": res_esm["test_macro_f1"],
    "gcn_accuracy": res_gcn["test_accuracy"],
    "gcn_macro_f1": res_gcn["test_macro_f1"],
})

fold_df.to_csv(OUT_DIR / "community_label_probe_fold_results.csv", index=False)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[6/6] Done.")
print(json.dumps(summary, indent=2))
print(f"Results saved to: {OUT_DIR}")
