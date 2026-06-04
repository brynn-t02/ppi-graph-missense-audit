#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 27:
Neighborhood risk enrichment analysis (REAL DATA ONLY)

Goal:
- Test whether proteins with higher training-label risk in their STRING neighborhood
  receive larger gains from GCN over raw ESM.

No fake data, no simulated labels, no placeholder predictions.
Hard-fail on missing files or length mismatch.
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

# Optional strongest fusion
PROB_CADD_AM = Path("/root/data/processed/triple_fusion/cadd_plus_am/test_probs.npy")
PROB_FULL = Path("/root/data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy")

OUT_DIR = Path("/root/data/processed/neighborhood_risk_enrichment_exp27")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
STRING_SCORE_THRESHOLD = 400
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

def tertile_bucket(values):
    q1, q2 = np.quantile(values, [1/3, 2/3])
    bucket = np.zeros(len(values), dtype=int)
    bucket[(values > q1) & (values <= q2)] = 1
    bucket[values > q2] = 2
    return bucket, [float(q1), float(q2)]

# =========================
# 1) Load real train/test tables
# =========================
print("[1/7] Loading real train/test tables...")
assert_exists(TRAIN_TABLE)
assert_exists(TEST_TABLE)

df_train = pd.read_csv(TRAIN_TABLE)
df_test = pd.read_csv(TEST_TABLE)

for col in ["Label", "STRING_protein_id", "split"]:
    if col not in df_train.columns:
        raise ValueError(f"Missing column {col} in train table")
    if col not in df_test.columns:
        raise ValueError(f"Missing column {col} in test table")

df_train = df_train[df_train["split"] == "train"].copy().reset_index(drop=True)
df_test = df_test[df_test["split"] == "test"].copy().reset_index(drop=True)

y_test = df_test["Label"].astype(int).values
test_proteins = df_test["STRING_protein_id"].astype(str).values

print(f"Train rows: {len(df_train)}")
print(f"Test rows: {len(df_test)}")

# =========================
# 2) Load real prediction probabilities
# =========================
print("[2/7] Loading real prediction probabilities...")
assert_exists(PROB_ESM)
assert_exists(PROB_GCN)

p_esm = np.load(str(PROB_ESM)).astype(float)
p_gcn = np.load(str(PROB_GCN)).astype(float)

if len(p_esm) != len(df_test) or len(p_gcn) != len(df_test):
    raise ValueError(
        f"Length mismatch: test table={len(df_test)}, "
        f"ESM probs={len(p_esm)}, GCN probs={len(p_gcn)}"
    )

has_fusion = PROB_CADD_AM.exists() and PROB_FULL.exists()
p_cadd_am = p_full = None
if has_fusion:
    p_cadd_am = np.load(str(PROB_CADD_AM)).astype(float)
    p_full = np.load(str(PROB_FULL)).astype(float)
    if len(p_cadd_am) != len(df_test) or len(p_full) != len(df_test):
        print("[WARN] Fusion arrays length mismatch. Disable fusion analysis.")
        has_fusion = False
        p_cadd_am = p_full = None

print(f"Fusion analysis enabled: {has_fusion}")

# =========================
# 3) Build full STRING graph on real node universe
# =========================
print("[3/7] Building full STRING graph...")
assert_exists(ESM_NODE_TABLE)
assert_exists(STRING_LINKS)

df_nodes = pd.read_csv(ESM_NODE_TABLE, usecols=["Protein_ID"])
node_ids = df_nodes["Protein_ID"].astype(str).tolist()
node_set = set(node_ids)

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
# 4) Compute train-side per-protein risk
# =========================
print("[4/7] Computing train-side per-protein risk...")

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
    train_protein_stats["STRING_protein_id"],
    train_protein_stats["train_variant_count"]
))
protein_to_train_path = dict(zip(
    train_protein_stats["STRING_protein_id"],
    train_protein_stats["train_pathogenic_count"]
))

# For every protein in node universe, aggregate neighbor train stats
neighbor_rows = []

for pid in node_ids:
    neigh = list(G.neighbors(pid))
    n_neighbor_proteins = len(neigh)

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
        neigh_pathogenic_rate = neigh_train_pathogenic_count / neigh_train_variant_count
    else:
        neigh_pathogenic_rate = np.nan

    neighbor_rows.append({
        "STRING_protein_id": pid,
        "neighbor_protein_count": int(n_neighbor_proteins),
        "neighbor_with_train_labels_count": int(neigh_with_train_labels),
        "neighbor_train_variant_count": int(neigh_train_variant_count),
        "neighbor_train_pathogenic_count": int(neigh_train_pathogenic_count),
        "neighbor_pathogenic_rate": float(neigh_pathogenic_rate) if not np.isnan(neigh_pathogenic_rate) else np.nan
    })

neighbor_df = pd.DataFrame(neighbor_rows)
neighbor_df.to_csv(OUT_DIR / "protein_neighbor_train_risk_full_graph.csv", index=False)

print("Saved protein-level neighbor risk table.")

# =========================
# 5) Merge to test set
# =========================
print("[5/7] Merging neighbor risk to test set...")

df_eval = df_test.copy()
df_eval["esm_prob"] = p_esm
df_eval["gcn_prob"] = p_gcn
df_eval["gcn_minus_esm_prob"] = p_gcn - p_esm

if has_fusion:
    df_eval["cadd_am_prob"] = p_cadd_am
    df_eval["gcn_cadd_am_prob"] = p_full
    df_eval["full_minus_cadd_am_prob"] = p_full - p_cadd_am

df_eval = df_eval.merge(neighbor_df, on="STRING_protein_id", how="left")

missing_neighbor_risk = int(df_eval["neighbor_protein_count"].isna().sum())
print(f"Rows with missing neighbor risk fields: {missing_neighbor_risk}")

# Define risk buckets using only rows with sufficient neighbor-labeled variants
valid_mask = (
    df_eval["neighbor_train_variant_count"].fillna(0) >= MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE
) & (
    df_eval["neighbor_pathogenic_rate"].notna()
)

df_eval["neighbor_risk_bucket"] = -1  # -1 = insufficient signal

if valid_mask.sum() > 0:
    valid_rates = df_eval.loc[valid_mask, "neighbor_pathogenic_rate"].values
    buckets, cuts = tertile_bucket(valid_rates)
    df_eval.loc[valid_mask, "neighbor_risk_bucket"] = buckets
else:
    cuts = [None, None]

df_eval.to_csv(OUT_DIR / "test_variants_with_neighbor_risk_and_predictions.csv", index=False)

print(f"Valid rows for risk bucketing: {int(valid_mask.sum())}")
print(f"Neighbor risk cuts: {cuts}")

# =========================
# 6) Stratified gain analysis by neighbor risk
# =========================
print("[6/7] Running stratified gain analysis by neighbor risk...")

def eval_group(group_col, group_labels):
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

        row = {
            "group_col": group_col,
            "group_id": int(gid),
            "group_label": group_labels.get(int(gid), str(gid)),
            "n": int(len(sub)),
            "pos_rate": float(np.mean(yy)),

            "mean_neighbor_pathogenic_rate": float(np.nanmean(sub["neighbor_pathogenic_rate"])),
            "mean_neighbor_train_variant_count": float(np.mean(sub["neighbor_train_variant_count"].fillna(0))),
            "mean_neighbor_with_train_labels_count": float(np.mean(sub["neighbor_with_train_labels_count"].fillna(0))),

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

risk_labels = {
    -1: "insufficient_neighbor_signal",
    0: "neighbor_risk_low",
    1: "neighbor_risk_mid",
    2: "neighbor_risk_high"
}

res = pd.DataFrame(eval_group("neighbor_risk_bucket", risk_labels))
res.to_csv(OUT_DIR / "neighbor_risk_stratified_metrics.csv", index=False)

# =========================
# 7) Global summary
# =========================
print("[7/7] Writing summary...")

global_esm = metrics(y_test, p_esm)
global_gcn = metrics(y_test, p_gcn)

summary = {
    "experiment": "27_neighborhood_risk_enrichment_analysis",
    "real_data_policy": {
        "uses_real_train_table": str(TRAIN_TABLE),
        "uses_real_test_table": str(TEST_TABLE),
        "uses_real_esm_predictions": str(PROB_ESM),
        "uses_real_gcn_predictions": str(PROB_GCN),
        "uses_real_string_graph": str(STRING_LINKS),
        "uses_real_node_universe": str(ESM_NODE_TABLE),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "neighbor_risk_defined_only_from_train_labels": True,
        "hard_fail_on_length_mismatch": True
    },
    "string_score_threshold": STRING_SCORE_THRESHOLD,
    "n_train": int(len(df_train)),
    "n_test": int(len(df_test)),
    "test_positive_rate": float(np.mean(y_test)),
    "graph_nodes": int(G.number_of_nodes()),
    "graph_edges": int(G.number_of_edges()),
    "min_neighbor_variant_count_for_rate": int(MIN_NEIGHBOR_VARIANT_COUNT_FOR_RATE),
    "valid_rows_for_risk_bucketing": int(valid_mask.sum()),
    "risk_rate_cuts": cuts,
    "missing_neighbor_risk_rows": missing_neighbor_risk,
    "global_esm": global_esm,
    "global_gcn": global_gcn,
    "delta_global_gcn_minus_esm": metric_delta(global_esm, global_gcn),
    "fusion_analysis_enabled": bool(has_fusion)
}

if has_fusion:
    global_base = metrics(y_test, p_cadd_am)
    global_full = metrics(y_test, p_full)
    summary["global_cadd_plus_am"] = global_base
    summary["global_gcn_plus_cadd_plus_am"] = global_full
    summary["delta_global_full_vs_cadd_am"] = metric_delta(global_base, global_full)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(json.dumps(summary["delta_global_gcn_minus_esm"], indent=2))
if has_fusion:
    print("Fusion delta:")
    print(json.dumps(summary["delta_global_full_vs_cadd_am"], indent=2))
print(f"Results saved to: {OUT_DIR}")
