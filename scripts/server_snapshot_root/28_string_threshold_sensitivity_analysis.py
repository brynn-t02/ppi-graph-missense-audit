#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 28:
STRING threshold sensitivity analysis (REAL DATA ONLY)

Goal:
- Test whether topology-stratified gain patterns and neighborhood-risk gain patterns
  are stable across different STRING combined_score thresholds.

Uses only real files. No fake data, no simulated labels/predictions.
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

OUT_DIR = Path("/root/data/processed/string_threshold_sensitivity_exp28")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
THRESHOLDS = [300, 400, 700]
MIN_GROUP_N = 50
MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE = 5

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
# Load fixed real inputs
# =========================
print("[1/5] Loading fixed real inputs...")

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
test_proteins = df_test["STRING_protein_id"].astype(str).values

p_esm = np.load(str(PROB_ESM)).astype(float)
p_gcn = np.load(str(PROB_GCN)).astype(float)

if len(p_esm) != len(df_test) or len(p_gcn) != len(df_test):
    raise ValueError(
        f"Length mismatch: test table={len(df_test)}, esm={len(p_esm)}, gcn={len(p_gcn)}"
    )

node_ids = df_nodes["Protein_ID"].astype(str).tolist()
node_set = set(node_ids)

with gzip.open(STRING_LINKS, "rt") as f:
    all_links = pd.read_csv(f, sep=r"\s+")

all_links = all_links[
    (all_links["protein1"].isin(node_set)) &
    (all_links["protein2"].isin(node_set)) &
    (all_links["protein1"] != all_links["protein2"])
].copy()

print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")
print(f"Node universe: {len(node_ids)}")
print(f"All candidate edges (before threshold): {len(all_links)}")

# =========================
# Precompute train protein stats
# =========================
print("[2/5] Precomputing train protein stats...")
train_protein_stats = (
    df_train.groupby("STRING_protein_id")["Label"]
    .agg(train_variant_count="count", train_pathogenic_count="sum")
    .reset_index()
)
train_protein_stats["STRING_protein_id"] = train_protein_stats["STRING_protein_id"].astype(str)
train_protein_stats["train_pathogenic_rate"] = (
    train_protein_stats["train_pathogenic_count"] / train_protein_stats["train_variant_count"]
)

protein_to_train_count = dict(zip(
    train_protein_stats["STRING_protein_id"], train_protein_stats["train_variant_count"]
))
protein_to_train_path = dict(zip(
    train_protein_stats["STRING_protein_id"], train_protein_stats["train_pathogenic_count"]
))

# =========================
# Main loop over thresholds
# =========================
print("[3/5] Running threshold sensitivity analysis...")

summary_rows = []
topology_rows = []
neighbor_rows = []

for thr in THRESHOLDS:
    print(f"  -> Threshold {thr}")

    links = all_links[all_links["combined_score"] >= thr].copy()

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(links[["protein1", "protein2"]].itertuples(index=False, name=None))

    # ---------- Topology features ----------
    degree_dict = dict(G.degree())
    degree_full = np.array([degree_dict.get(pid, 0) for pid in node_ids], dtype=int)

    core_dict = nx.core_number(G)
    kcore_full = np.array([core_dict.get(pid, 0) for pid in node_ids], dtype=int)

    degree_bucket_full, degree_cuts = make_quantile_bucket(degree_full)
    kcore_bucket_full, kcore_cuts = make_quantile_bucket(kcore_full)

    node_topo = pd.DataFrame({
        "STRING_protein_id": node_ids,
        "degree": degree_full,
        "kcore": kcore_full,
        "degree_bucket": degree_bucket_full,
        "kcore_bucket": kcore_bucket_full
    })

    df_eval_topo = df_test.copy()
    df_eval_topo["esm_prob"] = p_esm
    df_eval_topo["gcn_prob"] = p_gcn
    df_eval_topo["gcn_minus_esm_prob"] = p_gcn - p_esm
    df_eval_topo = df_eval_topo.merge(node_topo, on="STRING_protein_id", how="left")

    # global
    g_esm = metrics(y_test, p_esm)
    g_gcn = metrics(y_test, p_gcn)
    g_delta = metric_delta(g_esm, g_gcn)

    summary_rows.append({
        "threshold": thr,
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "global_delta_auc": g_delta["delta_auc"],
        "global_delta_auprc": g_delta["delta_auprc"],
        "global_delta_brier": g_delta["delta_brier"],
        "degree_cut_1": degree_cuts[0],
        "degree_cut_2": degree_cuts[1],
        "kcore_cut_1": kcore_cuts[0],
        "kcore_cut_2": kcore_cuts[1],
    })

    # topology group eval
    for group_col, label_map in [
        ("degree_bucket", {0: "degree_low", 1: "degree_mid", 2: "degree_high"}),
        ("kcore_bucket", {0: "kcore_low", 1: "kcore_mid", 2: "kcore_high"})
    ]:
        for gid, sub in df_eval_topo.groupby(group_col):
            if len(sub) < MIN_GROUP_N:
                continue
            yy = sub["Label"].astype(int).values
            m_esm = metrics(yy, sub["esm_prob"].values)
            m_gcn = metrics(yy, sub["gcn_prob"].values)
            d = metric_delta(m_esm, m_gcn)

            topology_rows.append({
                "threshold": thr,
                "group_col": group_col,
                "group_id": int(gid),
                "group_label": label_map[int(gid)],
                "n": int(len(sub)),
                "pos_rate": float(np.mean(yy)),
                "delta_auc": d["delta_auc"],
                "delta_auprc": d["delta_auprc"],
                "delta_brier": d["delta_brier"],
                "mean_gcn_minus_esm_prob": float(np.mean(sub["gcn_minus_esm_prob"]))
            })

    # ---------- Neighbor risk ----------
    neighbor_info = []
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

        neighbor_info.append({
            "STRING_protein_id": pid,
            "neighbor_train_variant_count": int(neigh_train_variant_count),
            "neighbor_train_pathogenic_count": int(neigh_train_pathogenic_count),
            "neighbor_with_train_labels_count": int(neigh_with_train_labels),
            "neighbor_pathogenic_rate": float(neigh_rate) if not np.isnan(neigh_rate) else np.nan
        })

    neighbor_df = pd.DataFrame(neighbor_info)

    df_eval_nb = df_test.copy()
    df_eval_nb["esm_prob"] = p_esm
    df_eval_nb["gcn_prob"] = p_gcn
    df_eval_nb["gcn_minus_esm_prob"] = p_gcn - p_esm
    df_eval_nb = df_eval_nb.merge(neighbor_df, on="STRING_protein_id", how="left")

    valid_mask = (
        df_eval_nb["neighbor_train_variant_count"].fillna(0) >= MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE
    ) & (
        df_eval_nb["neighbor_pathogenic_rate"].notna()
    )

    df_eval_nb["neighbor_risk_bucket"] = -1
    if valid_mask.sum() > 0:
        valid_rates = df_eval_nb.loc[valid_mask, "neighbor_pathogenic_rate"].values
        buckets, risk_cuts = tertile_bucket(valid_rates)
        df_eval_nb.loc[valid_mask, "neighbor_risk_bucket"] = buckets
    else:
        risk_cuts = [None, None]

    for gid, label in [(-1, "insufficient_signal"), (0, "risk_low"), (1, "risk_mid"), (2, "risk_high")]:
        sub = df_eval_nb[df_eval_nb["neighbor_risk_bucket"] == gid].copy()
        if len(sub) < MIN_GROUP_N:
            continue

        yy = sub["Label"].astype(int).values
        m_esm = metrics(yy, sub["esm_prob"].values)
        m_gcn = metrics(yy, sub["gcn_prob"].values)
        d = metric_delta(m_esm, m_gcn)

        neighbor_rows.append({
            "threshold": thr,
            "group_id": int(gid),
            "group_label": label,
            "n": int(len(sub)),
            "pos_rate": float(np.mean(yy)),
            "mean_neighbor_pathogenic_rate": float(np.nanmean(sub["neighbor_pathogenic_rate"])),
            "mean_neighbor_train_variant_count": float(np.mean(sub["neighbor_train_variant_count"].fillna(0))),
            "delta_auc": d["delta_auc"],
            "delta_auprc": d["delta_auprc"],
            "delta_brier": d["delta_brier"],
            "mean_gcn_minus_esm_prob": float(np.mean(sub["gcn_minus_esm_prob"])),
            "risk_cut_1": risk_cuts[0],
            "risk_cut_2": risk_cuts[1]
        })

# =========================
# Save outputs
# =========================
print("[4/5] Saving outputs...")

summary_df = pd.DataFrame(summary_rows)
topology_df = pd.DataFrame(topology_rows)
neighbor_df = pd.DataFrame(neighbor_rows)

summary_df.to_csv(OUT_DIR / "threshold_global_summary.csv", index=False)
topology_df.to_csv(OUT_DIR / "threshold_topology_stratified_metrics.csv", index=False)
neighbor_df.to_csv(OUT_DIR / "threshold_neighbor_risk_stratified_metrics.csv", index=False)

summary = {
    "experiment": "28_string_threshold_sensitivity_analysis",
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
    "thresholds_tested": THRESHOLDS,
    "n_train": int(len(df_train)),
    "n_test": int(len(df_test)),
    "test_positive_rate": float(np.mean(y_test)),
    "min_group_n": MIN_GROUP_N,
    "min_neighbor_variant_count_for_rate": MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE,
    "outputs": {
        "threshold_global_summary_csv": str(OUT_DIR / "threshold_global_summary.csv"),
        "threshold_topology_stratified_metrics_csv": str(OUT_DIR / "threshold_topology_stratified_metrics.csv"),
        "threshold_neighbor_risk_stratified_metrics_csv": str(OUT_DIR / "threshold_neighbor_risk_stratified_metrics.csv"),
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[5/5] Done.")
print(f"Results saved to: {OUT_DIR}")
