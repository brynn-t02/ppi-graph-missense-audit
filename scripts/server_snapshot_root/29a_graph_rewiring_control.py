#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 29A:
Degree-preserving graph rewiring control (REAL DATA ONLY)

Goal:
- Test whether the topology-/neighbor-risk-based gain patterns observed in Exp 26B/27
  are specific to the real STRING graph structure, rather than arbitrary graphs with
  similar degree sequence.

This experiment does NOT retrain GCN.
It keeps the real prediction probabilities fixed, and compares whether the explanatory
stratification patterns (k-core / neighbor-risk) remain under degree-preserving rewired graphs.

No fake data, no simulated labels, no placeholder predictions.
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
TRAIN_TABLE = Path("/root/data/processed/clinvar_train_main.csv")
TEST_TABLE = Path("/root/data/processed/clinvar_test_main.csv")
ESM_NODE_TABLE = Path("/root/data/processed/esm_node_features_full.csv")
STRING_LINKS = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")

PROB_ESM = Path("/root/data/processed/variant_mlp_comparison/esm_raw/test_probs.npy")
PROB_GCN = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy")

OUT_DIR = Path("/root/data/processed/graph_rewiring_control_exp29A")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
STRING_SCORE_THRESHOLD = 400
MIN_GROUP_N = 50
MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE = 5

N_REWIRED_REPLICATES = 3
NSWAP_PER_REPLICA = 200000
RANDOM_SEED = 42

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
        "brier": safe_brier(y, p)
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

def tertile_bucket(values):
    q1, q2 = np.quantile(values, [1/3, 2/3])
    bucket = np.zeros(len(values), dtype=int)
    bucket[(values > q1) & (values <= q2)] = 1
    bucket[values > q2] = 2
    return bucket, [float(q1), float(q2)]

# =========================
# Shared evaluation functions
# =========================
def evaluate_kcore_stratification(df_eval, y, p_esm, p_gcn, graph_label, replicate_id):
    rows = []
    for gid, sub in df_eval.groupby("kcore_bucket"):
        if len(sub) < MIN_GROUP_N:
            continue

        yy = sub["Label"].astype(int).values
        m_esm = metrics(yy, sub["esm_prob"].values)
        m_gcn = metrics(yy, sub["gcn_prob"].values)
        d = metric_delta(m_esm, m_gcn)

        rows.append({
            "graph_type": graph_label,
            "replicate_id": replicate_id,
            "group_col": "kcore_bucket",
            "group_id": int(gid),
            "group_label": {0: "kcore_low", 1: "kcore_mid", 2: "kcore_high"}.get(int(gid), str(gid)),
            "n": int(len(sub)),
            "pos_rate": float(np.mean(yy)),
            "mean_kcore": float(np.mean(sub["kcore"])),
            "delta_auc": d["delta_auc"],
            "delta_auprc": d["delta_auprc"],
            "delta_brier": d["delta_brier"],
            "mean_gcn_minus_esm_prob": float(np.mean(sub["gcn_minus_esm_prob"]))
        })
    return rows

def evaluate_neighbor_risk_stratification(df_eval, y, p_esm, p_gcn, graph_label, replicate_id):
    rows = []
    for gid, sub in df_eval.groupby("neighbor_risk_bucket"):
        if len(sub) < MIN_GROUP_N:
            continue

        yy = sub["Label"].astype(int).values
        m_esm = metrics(yy, sub["esm_prob"].values)
        m_gcn = metrics(yy, sub["gcn_prob"].values)
        d = metric_delta(m_esm, m_gcn)

        rows.append({
            "graph_type": graph_label,
            "replicate_id": replicate_id,
            "group_col": "neighbor_risk_bucket",
            "group_id": int(gid),
            "group_label": {
                -1: "insufficient_signal",
                0: "risk_low",
                1: "risk_mid",
                2: "risk_high"
            }.get(int(gid), str(gid)),
            "n": int(len(sub)),
            "pos_rate": float(np.mean(yy)),
            "mean_neighbor_pathogenic_rate": float(np.nanmean(sub["neighbor_pathogenic_rate"])),
            "mean_neighbor_train_variant_count": float(np.mean(sub["neighbor_train_variant_count"].fillna(0))),
            "delta_auc": d["delta_auc"],
            "delta_auprc": d["delta_auprc"],
            "delta_brier": d["delta_brier"],
            "mean_gcn_minus_esm_prob": float(np.mean(sub["gcn_minus_esm_prob"]))
        })
    return rows

def build_neighbor_df_from_graph(G, node_ids, protein_to_train_count, protein_to_train_path):
    rows = []
    for pid in node_ids:
        neigh = list(G.neighbors(pid))

        neigh_train_variant_count = 0
        neigh_train_pathogenic_count = 0
        neigh_with_train_labels = 0

        for nb in neigh:
            c = protein_to_train_count.get(nb, 0)
            p = protein_to_train_path.get(nb, 0)
            neigh_train_variant_count += c
            neigh_train_pathogenic_count += p
            if c > 0:
                neigh_with_train_labels += 1

        if neigh_train_variant_count > 0:
            neigh_rate = neigh_train_pathogenic_count / neigh_train_variant_count
        else:
            neigh_rate = np.nan

        rows.append({
            "STRING_protein_id": pid,
            "neighbor_train_variant_count": int(neigh_train_variant_count),
            "neighbor_train_pathogenic_count": int(neigh_train_pathogenic_count),
            "neighbor_with_train_labels_count": int(neigh_with_train_labels),
            "neighbor_pathogenic_rate": float(neigh_rate) if not np.isnan(neigh_rate) else np.nan
        })
    return pd.DataFrame(rows)

# =========================
# 1) Load real inputs
# =========================
print("[1/6] Loading real inputs...")
for p in [TRAIN_TABLE, TEST_TABLE, ESM_NODE_TABLE, STRING_LINKS, PROB_ESM, PROB_GCN]:
    assert_exists(p)

df_train = pd.read_csv(TRAIN_TABLE)
df_test = pd.read_csv(TEST_TABLE)
df_nodes = pd.read_csv(ESM_NODE_TABLE, usecols=["Protein_ID"])

for col in ["Label", "STRING_protein_id", "split"]:
    if col not in df_train.columns:
        raise ValueError(f"Missing column {col} in train table")
    if col not in df_test.columns:
        raise ValueError(f"Missing column {col} in test table")

df_train = df_train[df_train["split"] == "train"].copy().reset_index(drop=True)
df_test = df_test[df_test["split"] == "test"].copy().reset_index(drop=True)

y_test = df_test["Label"].astype(int).values
p_esm = np.load(str(PROB_ESM)).astype(float)
p_gcn = np.load(str(PROB_GCN)).astype(float)

if len(p_esm) != len(df_test) or len(p_gcn) != len(df_test):
    raise ValueError(
        f"Length mismatch: test table={len(df_test)}, esm={len(p_esm)}, gcn={len(p_gcn)}"
    )

node_ids = df_nodes["Protein_ID"].astype(str).tolist()
node_set = set(node_ids)

print(f"Train rows: {len(df_train)}")
print(f"Test rows: {len(df_test)}")
print(f"Node universe: {len(node_ids)}")

# =========================
# 2) Build real STRING graph
# =========================
print("[2/6] Building real STRING graph...")
with gzip.open(STRING_LINKS, "rt") as f:
    links = pd.read_csv(f, sep=r"\s+")

links = links[
    (links["protein1"].isin(node_set)) &
    (links["protein2"].isin(node_set)) &
    (links["protein1"] != links["protein2"]) &
    (links["combined_score"] >= STRING_SCORE_THRESHOLD)
].copy()

G_real = nx.Graph()
G_real.add_nodes_from(node_ids)
G_real.add_edges_from(links[["protein1", "protein2"]].itertuples(index=False, name=None))

print(f"Real graph nodes: {G_real.number_of_nodes()}")
print(f"Real graph edges: {G_real.number_of_edges()}")

# =========================
# 3) Precompute train protein stats
# =========================
print("[3/6] Precomputing train protein stats...")
train_protein_stats = (
    df_train.groupby("STRING_protein_id")["Label"]
    .agg(train_variant_count="count", train_pathogenic_count="sum")
    .reset_index()
)
train_protein_stats["STRING_protein_id"] = train_protein_stats["STRING_protein_id"].astype(str)

protein_to_train_count = dict(zip(
    train_protein_stats["STRING_protein_id"], train_protein_stats["train_variant_count"]
))
protein_to_train_path = dict(zip(
    train_protein_stats["STRING_protein_id"], train_protein_stats["train_pathogenic_count"]
))

# =========================
# 4) Evaluate real graph
# =========================
print("[4/6] Evaluating real graph stratifications...")

# k-core on real graph
real_kcore_dict = nx.core_number(G_real)
real_kcore = np.array([real_kcore_dict.get(pid, 0) for pid in node_ids], dtype=int)
real_kcore_bucket, real_kcore_cuts = make_quantile_bucket(real_kcore)

node_kcore_df = pd.DataFrame({
    "STRING_protein_id": node_ids,
    "kcore": real_kcore,
    "kcore_bucket": real_kcore_bucket
})

df_real_topo = df_test.copy()
df_real_topo["esm_prob"] = p_esm
df_real_topo["gcn_prob"] = p_gcn
df_real_topo["gcn_minus_esm_prob"] = p_gcn - p_esm
df_real_topo = df_real_topo.merge(node_kcore_df, on="STRING_protein_id", how="left")

real_kcore_rows = evaluate_kcore_stratification(
    df_real_topo, y_test, p_esm, p_gcn, graph_label="real", replicate_id=0
)

# neighbor risk on real graph
real_neighbor_df = build_neighbor_df_from_graph(
    G_real, node_ids, protein_to_train_count, protein_to_train_path
)

df_real_nb = df_test.copy()
df_real_nb["esm_prob"] = p_esm
df_real_nb["gcn_prob"] = p_gcn
df_real_nb["gcn_minus_esm_prob"] = p_gcn - p_esm
df_real_nb = df_real_nb.merge(real_neighbor_df, on="STRING_protein_id", how="left")

valid_mask_real = (
    df_real_nb["neighbor_train_variant_count"].fillna(0) >= MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE
) & (
    df_real_nb["neighbor_pathogenic_rate"].notna()
)

df_real_nb["neighbor_risk_bucket"] = -1
if valid_mask_real.sum() > 0:
    valid_rates = df_real_nb.loc[valid_mask_real, "neighbor_pathogenic_rate"].values
    risk_bucket_real, real_risk_cuts = tertile_bucket(valid_rates)
    df_real_nb.loc[valid_mask_real, "neighbor_risk_bucket"] = risk_bucket_real
else:
    real_risk_cuts = [None, None]

real_neighbor_rows = evaluate_neighbor_risk_stratification(
    df_real_nb, y_test, p_esm, p_gcn, graph_label="real", replicate_id=0
)

# =========================
# 5) Evaluate rewired graphs
# =========================
print("[5/6] Evaluating rewired graph controls...")

rewired_kcore_rows = []
rewired_neighbor_rows = []
rewire_summary_rows = []

for rep in range(1, N_REWIRED_REPLICATES + 1):
    print(f"  -> Rewired replicate {rep}")

    H = G_real.copy()

    # degree-preserving edge swaps
    nx.double_edge_swap(
        H,
        nswap=NSWAP_PER_REPLICA,
        max_tries=NSWAP_PER_REPLICA * 20,
        seed=RANDOM_SEED + rep
    )

    # k-core on rewired graph
    rew_kcore_dict = nx.core_number(H)
    rew_kcore = np.array([rew_kcore_dict.get(pid, 0) for pid in node_ids], dtype=int)
    rew_kcore_bucket, rew_kcore_cuts = make_quantile_bucket(rew_kcore)

    node_rew_kcore_df = pd.DataFrame({
        "STRING_protein_id": node_ids,
        "kcore": rew_kcore,
        "kcore_bucket": rew_kcore_bucket
    })

    df_rew_topo = df_test.copy()
    df_rew_topo["esm_prob"] = p_esm
    df_rew_topo["gcn_prob"] = p_gcn
    df_rew_topo["gcn_minus_esm_prob"] = p_gcn - p_esm
    df_rew_topo = df_rew_topo.merge(node_rew_kcore_df, on="STRING_protein_id", how="left")

    rewired_kcore_rows.extend(
        evaluate_kcore_stratification(
            df_rew_topo, y_test, p_esm, p_gcn, graph_label="rewired", replicate_id=rep
        )
    )

    # neighbor risk on rewired graph
    rew_neighbor_df = build_neighbor_df_from_graph(
        H, node_ids, protein_to_train_count, protein_to_train_path
    )

    df_rew_nb = df_test.copy()
    df_rew_nb["esm_prob"] = p_esm
    df_rew_nb["gcn_prob"] = p_gcn
    df_rew_nb["gcn_minus_esm_prob"] = p_gcn - p_esm
    df_rew_nb = df_rew_nb.merge(rew_neighbor_df, on="STRING_protein_id", how="left")

    valid_mask_rew = (
        df_rew_nb["neighbor_train_variant_count"].fillna(0) >= MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE
    ) & (
        df_rew_nb["neighbor_pathogenic_rate"].notna()
    )

    df_rew_nb["neighbor_risk_bucket"] = -1
    if valid_mask_rew.sum() > 0:
        valid_rates_rew = df_rew_nb.loc[valid_mask_rew, "neighbor_pathogenic_rate"].values
        rew_risk_bucket, rew_risk_cuts = tertile_bucket(valid_rates_rew)
        df_rew_nb.loc[valid_mask_rew, "neighbor_risk_bucket"] = rew_risk_bucket
    else:
        rew_risk_cuts = [None, None]

    rewired_neighbor_rows.extend(
        evaluate_neighbor_risk_stratification(
            df_rew_nb, y_test, p_esm, p_gcn, graph_label="rewired", replicate_id=rep
        )
    )

    rewire_summary_rows.append({
        "replicate_id": rep,
        "nodes": int(H.number_of_nodes()),
        "edges": int(H.number_of_edges()),
        "nswap": int(NSWAP_PER_REPLICA),
        "kcore_cut_1": rew_kcore_cuts[0],
        "kcore_cut_2": rew_kcore_cuts[1],
        "risk_cut_1": rew_risk_cuts[0],
        "risk_cut_2": rew_risk_cuts[1],
        "valid_neighbor_rows": int(valid_mask_rew.sum())
    })

# =========================
# 6) Save outputs
# =========================
print("[6/6] Saving outputs...")

real_kcore_df = pd.DataFrame(real_kcore_rows)
real_neighbor_metrics_df = pd.DataFrame(real_neighbor_rows)
rewired_kcore_df = pd.DataFrame(rewired_kcore_rows)
rewired_neighbor_metrics_df = pd.DataFrame(rewired_neighbor_rows)
rewire_summary_df = pd.DataFrame(rewire_summary_rows)

real_kcore_df.to_csv(OUT_DIR / "real_graph_kcore_stratified_metrics.csv", index=False)
real_neighbor_metrics_df.to_csv(OUT_DIR / "real_graph_neighbor_risk_stratified_metrics.csv", index=False)
rewired_kcore_df.to_csv(OUT_DIR / "rewired_graph_kcore_stratified_metrics.csv", index=False)
rewired_neighbor_metrics_df.to_csv(OUT_DIR / "rewired_graph_neighbor_risk_stratified_metrics.csv", index=False)
rewire_summary_df.to_csv(OUT_DIR / "rewired_graph_summary.csv", index=False)

summary = {
    "experiment": "29A_degree_preserving_graph_rewiring_control",
    "real_data_policy": {
        "uses_real_train_table": str(TRAIN_TABLE),
        "uses_real_test_table": str(TEST_TABLE),
        "uses_real_esm_predictions": str(PROB_ESM),
        "uses_real_gcn_predictions": str(PROB_GCN),
        "uses_real_string_graph": str(STRING_LINKS),
        "uses_real_node_universe": str(ESM_NODE_TABLE),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "hard_fail_on_length_mismatch": True
    },
    "design_note": {
        "this_experiment_does_not_retrain_gcn": True,
        "it_tests_whether_explanatory_stratification_patterns_are_specific_to_real_graph_structure": True,
        "rewiring_preserves_degree_sequence_but_disrupts_higher_order_structure": True
    },
    "string_score_threshold": STRING_SCORE_THRESHOLD,
    "n_rewired_replicates": N_REWIRED_REPLICATES,
    "nswap_per_replica": NSWAP_PER_REPLICA,
    "min_group_n": MIN_GROUP_N,
    "min_neighbor_variant_count_for_rate": MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE,
    "real_graph_nodes": int(G_real.number_of_nodes()),
    "real_graph_edges": int(G_real.number_of_edges()),
    "real_kcore_cuts": real_kcore_cuts,
    "real_risk_cuts": real_risk_cuts,
    "outputs": {
        "real_graph_kcore_stratified_metrics_csv": str(OUT_DIR / "real_graph_kcore_stratified_metrics.csv"),
        "real_graph_neighbor_risk_stratified_metrics_csv": str(OUT_DIR / "real_graph_neighbor_risk_stratified_metrics.csv"),
        "rewired_graph_kcore_stratified_metrics_csv": str(OUT_DIR / "rewired_graph_kcore_stratified_metrics.csv"),
        "rewired_graph_neighbor_risk_stratified_metrics_csv": str(OUT_DIR / "rewired_graph_neighbor_risk_stratified_metrics.csv"),
        "rewired_graph_summary_csv": str(OUT_DIR / "rewired_graph_summary.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
