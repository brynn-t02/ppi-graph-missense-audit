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

# =========================
# Paths
# =========================
ESM_PATH = Path("/root/data/processed/esm_node_features_full.csv")
GCN_PATH = Path("/root/data/processed/gcn_embeddings/gcn_node_embeddings.npy")
LINKS_PATH = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")
OUT_DIR = Path("/root/data/processed/topology_role_probe_kcore_bucket")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
RANDOM_SEED = 42
N_SPLITS = 5
STRING_SCORE_THRESHOLD = 400

# =========================
# Load ESM
# =========================
print("[1/5] Loading raw ESM features...")
df_esm = pd.read_csv(ESM_PATH)
protein_ids = df_esm["Protein_ID"].tolist()
esm_feat_cols = [c for c in df_esm.columns if c.startswith("Node_ESM_")]
X_esm = df_esm[esm_feat_cols].values.astype(np.float32)

protein_set = set(protein_ids)

print(f"Raw ESM shape: {X_esm.shape}")

# =========================
# Load GCN embeddings
# =========================
print("[2/5] Loading GCN embeddings...")
X_gcn = np.load(GCN_PATH).astype(np.float32)
print(f"GCN embedding shape: {X_gcn.shape}")

assert X_gcn.shape[0] == X_esm.shape[0], \
    "GCN embedding rows do not match ESM rows."

# =========================
# Build graph and compute k-core
# =========================
print("[3/5] Loading STRING graph and computing k-core...")
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

# Keep only proteins present in graph
graph_nodes = set(G.nodes())
keep_mask = np.array([pid in graph_nodes for pid in protein_ids])

protein_ids_keep = [pid for pid, keep in zip(protein_ids, keep_mask) if keep]
X_esm_keep = X_esm[keep_mask]
X_gcn_keep = X_gcn[keep_mask]

core_dict = nx.core_number(G)
core_array = np.array([core_dict[pid] for pid in protein_ids_keep], dtype=np.int64)

print(f"Graph nodes kept: {len(protein_ids_keep)}")
print(f"k-core min={core_array.min()}, median={np.median(core_array)}, max={core_array.max()}")

# Bucket by tertiles
q1, q2 = np.quantile(core_array, [1/3, 2/3])
kcore_bucket = np.zeros_like(core_array)
kcore_bucket[(core_array > q1) & (core_array <= q2)] = 1
kcore_bucket[core_array > q2] = 2

bucket_names = {0: "low", 1: "medium", 2: "high"}
bucket_counts = {
    bucket_names[i]: int(np.sum(kcore_bucket == i)) for i in [0, 1, 2]
}
print(f"k-core quantiles: q1={q1}, q2={q2}")
print(f"Bucket counts: {bucket_counts}")

# =========================
# Probe classifier
# =========================
print("[4/5] Running k-core bucket probe...")
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        random_state=RANDOM_SEED
    ))
])

scoring = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(f1_score, average="macro")
}

res_esm = cross_validate(clf, X_esm_keep, kcore_bucket, cv=cv, scoring=scoring, n_jobs=-1)
res_gcn = cross_validate(clf, X_gcn_keep, kcore_bucket, cv=cv, scoring=scoring, n_jobs=-1)

summary = {
    "task": "kcore_bucket_probe",
    "n_nodes_used": int(len(protein_ids_keep)),
    "bucket_counts": bucket_counts,
    "kcore_quantiles": {
        "q1": float(q1),
        "q2": float(q2)
    },
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
    "string_score_threshold": STRING_SCORE_THRESHOLD,
    "assumption": "GCN embedding row order follows esm_node_features_full.csv Protein_ID order."
}

fold_df = pd.DataFrame({
    "fold": list(range(1, N_SPLITS + 1)),
    "esm_accuracy": res_esm["test_accuracy"],
    "esm_macro_f1": res_esm["test_macro_f1"],
    "gcn_accuracy": res_gcn["test_accuracy"],
    "gcn_macro_f1": res_gcn["test_macro_f1"],
})

fold_df.to_csv(OUT_DIR / "kcore_bucket_probe_fold_results.csv", index=False)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[5/5] Done.")
print(json.dumps(summary, indent=2))
print(f"Results saved to: {OUT_DIR}")
