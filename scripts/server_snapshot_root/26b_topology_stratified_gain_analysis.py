#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 26B: Topology-stratified gain analysis (REAL DATA ONLY)

Inputs (real files):
- clinvar_test_main.csv (split=test)
- raw ESM test_probs.npy
- GCN enhanced test_probs.npy
- STRING v12 links file
- ESM node universe (esm_node_features_full.csv) for full-graph topology

This script does NOT create fake labels/predictions/samples.
Hard-fails if missing files or length mismatch.
"""

import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from networkx.algorithms.community import greedy_modularity_communities

# =========================
# Paths
# =========================
TEST_TABLE = Path("/root/data/processed/clinvar_test_main.csv")
ESM_NODE_TABLE = Path("/root/data/processed/esm_node_features_full.csv")
STRING_LINKS = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")

PROB_ESM = Path("/root/data/processed/variant_mlp_comparison/esm_raw/test_probs.npy")
PROB_GCN = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy")

# Optional strongest fusion
PROB_CADD_AM = Path("/root/data/processed/triple_fusion/cadd_plus_am/test_probs.npy")
PROB_FULL = Path("/root/data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy")

OUT_DIR = Path("/root/data/processed/topology_gain_analysis_exp26B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
STRING_SCORE_THRESHOLD = 400
MIN_COMMUNITY_SIZE = 50
TOP_N_COMMUNITIES = 10
MIN_GROUP_N = 50

# =========================
# Helpers
# =========================
def assert_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))

def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))

def safe_brier(y, p):
    return float(brier_score_loss(y, p))

def metrics(y, p):
    return {
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "auc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": safe_brier(y, p),
        "prob_mean": float(np.mean(p)),
        "prob_std": float(np.std(p))
    }

def metric_delta(base, new):
    out = {}
    for k in ["auc", "auprc", "brier"]:
        if base[k] is None or new[k] is None:
            out[f"delta_{k}"] = None
        else:
            out[f"delta_{k}"] = float(new[k] - base[k])
    return out

def make_quantile_bucket(values, n_bucket=3):
    cuts = np.quantile(values, [1/3, 2/3])
    bucket = np.zeros(len(values), dtype=int)
    bucket[(values > cuts[0]) & (values <= cuts[1])] = 1
    bucket[values > cuts[1]] = 2
    return bucket, [float(cuts[0]), float(cuts[1])]

# =========================
# 1) Load real test table
# =========================
print("[1/7] Loading real test table...")
assert_exists(TEST_TABLE)
df = pd.read_csv(TEST_TABLE)

required_cols = ["Label", "STRING_protein_id", "split"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column missing from test table: {col}")

df = df[df["split"] == "test"].copy().reset_index(drop=True)

y = df["Label"].astype(int).values
test_proteins = df["STRING_protein_id"].astype(str).values

print(f"Test rows: {len(df)}")
print(f"Positive rate: {np.mean(y):.4f}")

# =========================
# 2) Load real prediction probabilities
# =========================
print("[2/7] Loading real prediction probabilities...")
assert_exists(PROB_ESM)
assert_exists(PROB_GCN)

p_esm = np.load(str(PROB_ESM)).astype(float)
p_gcn = np.load(str(PROB_GCN)).astype(float)

if len(p_esm) != len(df) or len(p_gcn) != len(df):
    raise ValueError(
        f"Length mismatch: test table={len(df)}, "
        f"ESM probs={len(p_esm)}, GCN probs={len(p_gcn)}"
    )

has_fusion = PROB_CADD_AM.exists() and PROB_FULL.exists()
p_cadd_am = p_full = None
if has_fusion:
    p_cadd_am = np.load(str(PROB_CADD_AM)).astype(float)
    p_full = np.load(str(PROB_FULL)).astype(float)
    if len(p_cadd_am) != len(df) or len(p_full) != len(df):
        print("[WARN] Fusion probability lengths do not match test table. Skipping fusion analysis.")
        has_fusion = False
        p_cadd_am = p_full = None

print(f"Loaded ESM probs: {p_esm.shape}")
print(f"Loaded GCN probs: {p_gcn.shape}")
print(f"Fusion analysis enabled: {has_fusion}")

# =========================
# 3) Load node universe from real ESM node table
# =========================
print("[3/7] Loading protein node universe...")
assert_exists(ESM_NODE_TABLE)
df_nodes = pd.read_csv(ESM_NODE_TABLE, usecols=["Protein_ID"])
node_ids = df_nodes["Protein_ID"].astype(str).tolist()
node_set = set(node_ids)
print(f"Protein node universe: {len(node_ids)}")

# =========================
# 4) Build full STRING graph on node universe
# =========================
print("[4/7] Building full STRING graph...")
assert_exists(STRING_LINKS)

with gzip.open(STRING_LINKS, "rt") as f:
    links = pd.read_csv(f, sep=r"\s+")

links = links[
    (links["protein1"].isin(node_set)) &
    (links["protein2"].isin(node_set)) &
    (links["protein1"] != links["protein2"]) &
    (links["combined_score"] >= STRING_SCORE_THRESHOLD)
].copy()

G = nx.Graph()
G.add_nodes_from(node_ids)
G.add_edges_from(links[["protein1", "protein2"]].itertuples(index=False, name=None))

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")

# =========================
# 5) Compute topology features on full graph
# =========================
print("[5/7] Computing topology features...")

degree_dict = dict(G.degree())
degree_full = np.array([degree_dict.get(pid, 0) for pid in node_ids], dtype=int)

core_dict = nx.core_number(G)
kcore_full = np.array([core_dict.get(pid, 0) for pid in node_ids], dtype=int)

communities = list(greedy_modularity_communities(G))
communities = [list(c) for c in communities if len(c) >= MIN_COMMUNITY_SIZE]

node_to_comm = {}
for cid, comm in enumerate(communities):
    for node in comm:
        node_to_comm[node] = cid
community_full = np.array([node_to_comm.get(pid, -1) for pid in node_ids], dtype=int)

node_topo = pd.DataFrame({
    "STRING_protein_id": node_ids,
    "degree": degree_full,
    "kcore": kcore_full,
    "community_id": community_full
})

degree_bucket_full, degree_cuts = make_quantile_bucket(degree_full)
kcore_bucket_full, kcore_cuts = make_quantile_bucket(kcore_full)

node_topo["degree_bucket"] = degree_bucket_full
node_topo["kcore_bucket"] = kcore_bucket_full

node_topo.to_csv(OUT_DIR / "protein_topology_features_full_graph.csv", index=False)

print(f"Degree cuts: {degree_cuts}")
print(f"k-core cuts: {kcore_cuts}")
print(f"Communities kept: {len(communities)}")

# =========================
# 6) Merge topology to test variants
# =========================
print("[6/7] Merging topology to test variants...")

df_eval = df.copy()
df_eval["esm_prob"] = p_esm
df_eval["gcn_prob"] = p_gcn
df_eval["gcn_minus_esm_prob"] = p_gcn - p_esm

if has_fusion:
    df_eval["cadd_am_prob"] = p_cadd_am
    df_eval["gcn_cadd_am_prob"] = p_full
    df_eval["full_minus_cadd_am_prob"] = p_full - p_cadd_am

df_eval = df_eval.merge(node_topo, on="STRING_protein_id", how="left")
missing_topo = int(df_eval["degree"].isna().sum())
print(f"Rows with missing topology: {missing_topo}")

df_eval.to_csv(OUT_DIR / "test_variants_with_predictions_and_topology.csv", index=False)

# =========================
# 7) Stratified performance analysis
# =========================
print("[7/7] Stratified performance analysis...")

def eval_group(group_col, group_labels=None):
    rows = []
    for gid, sub in df_eval.groupby(group_col):
        if len(sub) < MIN_GROUP_N:
            continue

        yy = sub["Label"].astype(int).values
        pp_esm = sub["esm_prob"].values
        pp_gcn = sub["gcn_prob"].values

        m_esm = metrics(yy, pp_esm)
        m_gcn = metrics(yy, pp_gcn)
        d = metric_delta(m_esm, m_gcn)

        label = str(gid)
        if group_labels is not None and gid in group_labels:
            label = group_labels[gid]

        row = {
            "group_col": group_col,
            "group_id": int(gid) if pd.notna(gid) and str(gid).lstrip("-").isdigit() else str(gid),
            "group_label": label,
            "n": int(len(sub)),
            "pos_rate": float(np.mean(yy)),

            "esm_auc": m_esm["auc"],
            "gcn_auc": m_gcn["auc"],
            "delta_auc": d["delta_auc"],

            "esm_auprc": m_esm["auprc"],
            "gcn_auprc": m_gcn["auprc"],
            "delta_auprc": d["delta_auprc"],

            "esm_brier": m_esm["brier"],
            "gcn_brier": m_gcn["brier"],
            "delta_brier": d["delta_brier"],

            "mean_gcn_minus_esm_prob": float(sub["gcn_minus_esm_prob"].mean())
        }

        if has_fusion:
            pp_base = sub["cadd_am_prob"].values
            pp_full = sub["gcn_cadd_am_prob"].values
            m_base = metrics(yy, pp_base)
            m_full = metrics(yy, pp_full)
            dd = metric_delta(m_base, m_full)

            row.update({
                "cadd_am_auc": m_base["auc"],
                "gcn_cadd_am_auc": m_full["auc"],
                "delta_full_vs_cadd_am_auc": dd["delta_auc"],

                "cadd_am_auprc": m_base["auprc"],
                "gcn_cadd_am_auprc": m_full["auprc"],
                "delta_full_vs_cadd_am_auprc": dd["delta_auprc"],

                "cadd_am_brier": m_base["brier"],
                "gcn_cadd_am_brier": m_full["brier"],
                "delta_full_vs_cadd_am_brier": dd["delta_brier"],
            })

        rows.append(row)
    return rows

rows = []
rows += eval_group("degree_bucket", group_labels={0: "degree_low", 1: "degree_mid", 2: "degree_high"})
rows += eval_group("kcore_bucket", group_labels={0: "kcore_low", 1: "kcore_mid", 2: "kcore_high"})

comm_counts = df_eval["community_id"].value_counts(dropna=False)
top_comm_ids = list(comm_counts.head(TOP_N_COMMUNITIES).index)
df_comm = df_eval[df_eval["community_id"].isin(top_comm_ids)].copy()

tmp = df_eval
df_eval = df_comm
rows += eval_group("community_id", group_labels=None)
df_eval = tmp

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "topology_stratified_metrics.csv", index=False)

global_esm = metrics(y, p_esm)
global_gcn = metrics(y, p_gcn)
global_delta = metric_delta(global_esm, global_gcn)

summary = {
    "experiment": "26B_topology_stratified_gain_analysis",
    "real_data_policy": {
        "uses_real_test_table": str(TEST_TABLE),
        "uses_real_esm_predictions": str(PROB_ESM),
        "uses_real_gcn_predictions": str(PROB_GCN),
        "uses_real_string_graph": str(STRING_LINKS),
        "uses_real_node_universe": str(ESM_NODE_TABLE),
        "no_fake_data": True,
        "hard_fail_on_length_mismatch": True
    },
    "n_test": int(len(df)),
    "positive_rate": float(np.mean(y)),
    "string_score_threshold": STRING_SCORE_THRESHOLD,
    "graph_nodes": int(G.number_of_nodes()),
    "graph_edges": int(G.number_of_edges()),
    "degree_cuts": degree_cuts,
    "kcore_cuts": kcore_cuts,
    "n_communities_kept": int(len(communities)),
    "missing_topology_rows": missing_topo,
    "global_esm": global_esm,
    "global_gcn": global_gcn,
    "delta_global_gcn_minus_esm": global_delta,
    "fusion_analysis_enabled": bool(has_fusion)
}

if has_fusion:
    global_base = metrics(y, p_cadd_am)
    global_full = metrics(y, p_full)
    summary["global_cadd_plus_am"] = global_base
    summary["global_gcn_plus_cadd_plus_am"] = global_full
    summary["delta_global_full_vs_cadd_am"] = metric_delta(global_base, global_full)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print("Delta global (GCN - ESM):")
print(json.dumps(summary["delta_global_gcn_minus_esm"], indent=2))
if has_fusion:
    print("Delta global (Full - CADD+AM):")
    print(json.dumps(summary["delta_global_full_vs_cadd_am"], indent=2))

print(f"Results saved to: {OUT_DIR}")
