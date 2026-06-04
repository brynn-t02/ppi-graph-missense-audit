#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 32:
Community boundary role stratified gain analysis (REAL DATA ONLY)

Goal:
- Test whether GCN-enhanced prediction gains over raw ESM concentrate more in
  community-internal proteins or community-boundary proteins in the real STRING graph.

IMPORTANT:
- Reuse community_id produced by Exp 26B instead of recomputing communities.
- This avoids the expensive community detection step and keeps consistency with 26B.

Definitions:
- For each node, compute the fraction of neighbors that belong to the same community:
    within_comm_neighbor_fraction
- Lower fraction -> more boundary-like
- Higher fraction -> more community-internal

No fake data, no simulated labels, no placeholder predictions.
Hard-fails on missing files or length mismatch.
"""

import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# =========================
# Paths
# =========================
TEST_TABLE = Path("/root/data/processed/clinvar_test_main.csv")
ESM_NODE_TABLE = Path("/root/data/processed/esm_node_features_full.csv")
STRING_LINKS = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")

# Reuse 26B output
TOPOLOGY_26B = Path("/root/data/processed/topology_gain_analysis_exp26B/protein_topology_features_full_graph.csv")

PROB_ESM = Path("/root/data/processed/variant_mlp_comparison/esm_raw/test_probs.npy")
PROB_GCN = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy")

# Optional strongest fusion
PROB_CADD_AM = Path("/root/data/processed/triple_fusion/cadd_plus_am/test_probs.npy")
PROB_FULL = Path("/root/data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy")

OUT_DIR = Path("/root/data/processed/community_boundary_gain_analysis_exp32")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
STRING_SCORE_THRESHOLD = 400
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

def make_quantile_bucket(values):
    q1, q2 = np.quantile(values, [1/3, 2/3])
    bucket = np.zeros(len(values), dtype=int)
    bucket[(values > q1) & (values <= q2)] = 1
    bucket[values > q2] = 2
    return bucket, [float(q1), float(q2)]

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
        f"Length mismatch: test table={len(df)}, ESM probs={len(p_esm)}, GCN probs={len(p_gcn)}"
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

print(f"Fusion analysis enabled: {has_fusion}")

# =========================
# 3) Load node universe and 26B community assignments
# =========================
print("[3/7] Loading protein node universe and 26B communities...")
assert_exists(ESM_NODE_TABLE)
assert_exists(TOPOLOGY_26B)

df_nodes = pd.read_csv(ESM_NODE_TABLE, usecols=["Protein_ID"])
node_ids = df_nodes["Protein_ID"].astype(str).tolist()
node_set = set(node_ids)

topo26 = pd.read_csv(TOPOLOGY_26B, usecols=["STRING_protein_id", "community_id"])
topo26["STRING_protein_id"] = topo26["STRING_protein_id"].astype(str)

if topo26["STRING_protein_id"].nunique() != len(topo26):
    raise ValueError("Duplicate STRING_protein_id found in 26B topology file.")

node_to_comm = dict(zip(topo26["STRING_protein_id"], topo26["community_id"]))

print(f"Protein node universe: {len(node_ids)}")
print(f"26B topology file rows: {len(topo26)}")

# =========================
# 4) Build full STRING graph
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
# 5) Compute boundary-role proxy from reused community_id
# =========================
print("[5/7] Computing boundary-role proxy from 26B community assignments...")

rows = []
for pid in node_ids:
    cid = int(node_to_comm.get(pid, -1))
    neigh = list(G.neighbors(pid))
    deg = len(neigh)

    if deg == 0 or cid == -1:
        within_frac = np.nan
    else:
        same = 0
        valid = 0
        for nb in neigh:
            nb_cid = int(node_to_comm.get(nb, -1))
            if nb_cid == -1:
                continue
            valid += 1
            if nb_cid == cid:
                same += 1
        if valid == 0:
            within_frac = np.nan
        else:
            within_frac = same / valid

    rows.append({
        "STRING_protein_id": pid,
        "community_id": cid,
        "degree": int(deg),
        "within_comm_neighbor_fraction": float(within_frac) if not np.isnan(within_frac) else np.nan
    })

node_role = pd.DataFrame(rows)

valid_mask = node_role["within_comm_neighbor_fraction"].notna()
boundary_bucket = np.full(len(node_role), -1, dtype=int)

if valid_mask.sum() > 0:
    buckets, cuts = make_quantile_bucket(node_role.loc[valid_mask, "within_comm_neighbor_fraction"].values)
    boundary_bucket[np.where(valid_mask)[0]] = buckets
else:
    cuts = [None, None]

node_role["boundary_role_bucket"] = boundary_bucket
node_role.to_csv(OUT_DIR / "protein_community_boundary_features_full_graph.csv", index=False)

print(f"Within-community fraction cuts: {cuts}")
print(f"Valid proteins with boundary proxy: {int(valid_mask.sum())}")

# =========================
# 6) Merge and stratified analysis
# =========================
print("[6/7] Merging to test set and computing stratified metrics...")

df_eval = df.copy()
df_eval["esm_prob"] = p_esm
df_eval["gcn_prob"] = p_gcn
df_eval["gcn_minus_esm_prob"] = p_gcn - p_esm

if has_fusion:
    df_eval["cadd_am_prob"] = p_cadd_am
    df_eval["gcn_cadd_am_prob"] = p_full
    df_eval["full_minus_cadd_am_prob"] = p_full - p_cadd_am

df_eval = df_eval.merge(node_role, on="STRING_protein_id", how="left")
missing_role = int(df_eval["within_comm_neighbor_fraction"].isna().sum())
print(f"Rows with missing boundary-role proxy: {missing_role}")

df_eval.to_csv(OUT_DIR / "test_variants_with_boundary_role_and_predictions.csv", index=False)

rows_out = []
label_map = {
    -1: "missing_or_unassigned",
    0: "community_boundary_like",
    1: "community_mixed",
    2: "community_internal_like"
}

for gid, sub in df_eval.groupby("boundary_role_bucket"):
    if len(sub) < MIN_GROUP_N:
        continue

    yy = sub["Label"].astype(int).values
    m_esm = metrics(yy, sub["esm_prob"].values)
    m_gcn = metrics(yy, sub["gcn_prob"].values)
    d = metric_delta(m_esm, m_gcn)

    row = {
        "group_col": "boundary_role_bucket",
        "group_id": int(gid),
        "group_label": label_map.get(int(gid), str(gid)),
        "n": int(len(sub)),
        "pos_rate": float(np.mean(yy)),
        "mean_within_comm_neighbor_fraction": float(np.nanmean(sub["within_comm_neighbor_fraction"])),

        "esm_auc": m_esm["auc"],
        "gcn_auc": m_gcn["auc"],
        "delta_auc": d["delta_auc"],

        "esm_auprc": m_esm["auprc"],
        "gcn_auprc": m_gcn["auprc"],
        "delta_auprc": d["delta_auprc"],

        "esm_brier": m_esm["brier"],
        "gcn_brier": m_gcn["brier"],
        "delta_brier": d["delta_brier"],

        "mean_gcn_minus_esm_prob": float(np.mean(sub["gcn_minus_esm_prob"]))
    }

    if has_fusion:
        m_base = metrics(yy, sub["cadd_am_prob"].values)
        m_full = metrics(yy, sub["gcn_cadd_am_prob"].values)
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

    rows_out.append(row)

res = pd.DataFrame(rows_out)
res.to_csv(OUT_DIR / "community_boundary_stratified_metrics.csv", index=False)

# =========================
# 7) Summary
# =========================
print("[7/7] Writing summary...")
global_esm = metrics(y, p_esm)
global_gcn = metrics(y, p_gcn)

summary = {
    "experiment": "32_community_boundary_role_gain_analysis",
    "real_data_policy": {
        "uses_real_test_table": str(TEST_TABLE),
        "uses_real_esm_predictions": str(PROB_ESM),
        "uses_real_gcn_predictions": str(PROB_GCN),
        "uses_real_string_graph": str(STRING_LINKS),
        "uses_real_node_universe": str(ESM_NODE_TABLE),
        "reuses_real_26b_community_assignments": str(TOPOLOGY_26B),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "hard_fail_on_length_mismatch": True
    },
    "design_note": {
        "boundary_role_proxy": "within_comm_neighbor_fraction",
        "lower_values_more_boundary_like": True,
        "higher_values_more_internal_like": True,
        "communities_reused_from_exp26b": True
    },
    "n_test": int(len(df)),
    "positive_rate": float(np.mean(y)),
    "string_score_threshold": STRING_SCORE_THRESHOLD,
    "graph_nodes": int(G.number_of_nodes()),
    "graph_edges": int(G.number_of_edges()),
    "within_comm_fraction_cuts": cuts,
    "missing_boundary_role_rows": missing_role,
    "global_esm": global_esm,
    "global_gcn": global_gcn,
    "delta_global_gcn_minus_esm": metric_delta(global_esm, global_gcn),
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
print(json.dumps(summary["delta_global_gcn_minus_esm"], indent=2))
print(f"Results saved to: {OUT_DIR}")
