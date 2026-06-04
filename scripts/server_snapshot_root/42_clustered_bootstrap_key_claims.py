#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 42:
Protein-clustered bootstrap for key claims.

Goal:
Address non-independence of multiple variants from the same protein/gene by
performing clustered bootstrap at STRING_protein_id level.

Main analyses:
42A. GCN-enhanced vs raw ESM on full test set.
42B. Key joint topology cells from Exp33/33C.
42C. Optional CADD+AM vs Full if probability files are discoverable.

REAL DATA ONLY:
- Uses real Exp35 / Exp33 tables.
- Uses real saved probability files.
- No fake labels.
- No simulated predictions.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# =========================
# Paths
# =========================
P35 = Path(
    "/root/data/processed/corrected_sample_profile_exp35/"
    "test_variants_with_corrected_labels_and_topology_profile.csv"
)

P33 = Path(
    "/root/data/processed/topology_profile_joint_map_exp33/"
    "merged_topology_profile_features.csv"
)

ESM_PROBS = Path("/root/data/processed/variant_mlp_comparison/esm_raw/test_probs.npy")
GCN_PROBS = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy")

OUT_DIR = Path("/root/data/processed/clustered_bootstrap_exp42")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
N_BOOT = 1000
SEED = 42
CLUSTER_COL_CANDIDATES = [
    "STRING_protein_id",
    "string_protein_id",
    "protein_id",
    "Protein",
    "gene",
    "Gene",
    "GeneSymbol",
    "gene_symbol",
]

KEY_CELLS = [
    {
        "name": "mid_kcore_x_high_betweenness",
        "row_col": "kcore_bucket",
        "row_val": 1,
        "col_col": "betweenness_bucket",
        "col_val": 2,
    },
    {
        "name": "mid_kcore_x_mid_clustering",
        "row_col": "kcore_bucket",
        "row_val": 1,
        "col_col": "clustering_bucket",
        "col_val": 1,
    },
    {
        "name": "high_neighbor_risk_x_mid_kcore",
        "row_col": "neighbor_risk_bucket",
        "row_val": 2,
        "col_col": "kcore_bucket",
        "col_val": 1,
    },
    {
        "name": "high_kcore_x_low_clustering",
        "row_col": "kcore_bucket",
        "row_val": 2,
        "col_col": "clustering_bucket",
        "col_val": 0,
    },
]

# =========================
# Helpers
# =========================
def find_cluster_col(df):
    for c in CLUSTER_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"No cluster column found. Tried: {CLUSTER_COL_CANDIDATES}. "
        f"Available columns include: {list(df.columns)[:50]}"
    )

def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def safe_auprc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)

def compute_metrics(y, p):
    return {
        "auc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": brier_score_loss(y, p),
    }

def compute_delta(y, p_base, p_graph):
    m_base = compute_metrics(y, p_base)
    m_graph = compute_metrics(y, p_graph)
    return {
        "delta_auc": m_graph["auc"] - m_base["auc"],
        "delta_auprc": m_graph["auprc"] - m_base["auprc"],
        "delta_brier": m_graph["brier"] - m_base["brier"],
        "base_auc": m_base["auc"],
        "graph_auc": m_graph["auc"],
        "base_auprc": m_base["auprc"],
        "graph_auprc": m_graph["auprc"],
        "base_brier": m_base["brier"],
        "graph_brier": m_graph["brier"],
    }

def ci(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return [np.nan, np.nan]
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]

def clustered_bootstrap_delta(df, y_col, p_base_col, p_graph_col, cluster_col, n_boot=1000, seed=42):
    """
    Clustered bootstrap:
    sample clusters with replacement, include all rows from selected clusters.
    If a cluster is sampled multiple times, its rows are repeated.
    """
    rng = np.random.default_rng(seed)

    work = df[[cluster_col, y_col, p_base_col, p_graph_col]].dropna().copy()
    clusters = work[cluster_col].dropna().unique()

    point = compute_delta(
        work[y_col].values.astype(int),
        work[p_base_col].values.astype(float),
        work[p_graph_col].values.astype(float),
    )

    by_cluster = {c: g.index.values for c, g in work.groupby(cluster_col)}

    boot = {
        "delta_auc": [],
        "delta_auprc": [],
        "delta_brier": [],
    }

    cluster_arr = np.array(clusters)

    for _ in range(n_boot):
        sampled_clusters = rng.choice(cluster_arr, size=len(cluster_arr), replace=True)

        idx_list = []
        for c in sampled_clusters:
            idx_list.append(by_cluster[c])

        idx = np.concatenate(idx_list)

        sub = work.loc[idx]

        d = compute_delta(
            sub[y_col].values.astype(int),
            sub[p_base_col].values.astype(float),
            sub[p_graph_col].values.astype(float),
        )

        boot["delta_auc"].append(d["delta_auc"])
        boot["delta_auprc"].append(d["delta_auprc"])
        boot["delta_brier"].append(d["delta_brier"])

    out = {
        "n_variants": int(len(work)),
        "n_clusters": int(len(clusters)),
        "point": {k: float(v) if v == v else np.nan for k, v in point.items()},
        "clustered_bootstrap": {
            "n_boot": n_boot,
            "delta_auc_ci95": ci(boot["delta_auc"]),
            "delta_auprc_ci95": ci(boot["delta_auprc"]),
            "delta_brier_ci95": ci(boot["delta_brier"]),
            "delta_auc_mean": float(np.nanmean(boot["delta_auc"])),
            "delta_auprc_mean": float(np.nanmean(boot["delta_auprc"])),
            "delta_brier_mean": float(np.nanmean(boot["delta_brier"])),
        }
    }
    return out

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def discover_prob_files():
    """
    Best-effort discovery for CADD+AM and Full probability files.
    This is optional; if not found, experiment proceeds without C.
    """
    root = Path("/root/data/processed")
    candidates = list(root.rglob("test_probs.npy"))

    found = {}
    for p in candidates:
        s = str(p).lower()
        if ("cadd" in s and "am" in s and "gcn" not in s and "full" not in s):
            found.setdefault("cadd_am", []).append(p)
        if ("full" in s or ("gcn" in s and "cadd" in s and "am" in s)):
            found.setdefault("full", []).append(p)

    return {k: [str(x) for x in v] for k, v in found.items()}

# =========================
# Main
# =========================
def main():
    print("[1/5] Loading Exp35 table for full-test clustered bootstrap...")

    if not P35.exists():
        raise FileNotFoundError(P35)
    if not ESM_PROBS.exists():
        raise FileNotFoundError(ESM_PROBS)
    if not GCN_PROBS.exists():
        raise FileNotFoundError(GCN_PROBS)

    df35 = pd.read_csv(P35)
    cluster_col35 = find_cluster_col(df35)
    print(f"Exp35 cluster column: {cluster_col35}")

    df35["Label"] = pd.to_numeric(df35["Label"], errors="coerce").astype(int)

    esm = np.load(ESM_PROBS)
    gcn = np.load(GCN_PROBS)

    if len(esm) != len(df35) or len(gcn) != len(df35):
        raise ValueError("Probability length mismatch with Exp35 table.")

    df35["esm_prob"] = esm
    df35["gcn_prob"] = gcn

    print("[2/5] Running 42A clustered bootstrap: GCN vs ESM full test...")

    res_full = clustered_bootstrap_delta(
        df=df35,
        y_col="Label",
        p_base_col="esm_prob",
        p_graph_col="gcn_prob",
        cluster_col=cluster_col35,
        n_boot=N_BOOT,
        seed=SEED,
    )

    with open(OUT_DIR / "42A_gcn_vs_esm_clustered_bootstrap.json", "w") as f:
        json.dump(res_full, f, indent=2)

    print("[3/5] Loading Exp33 table for key-cell clustered bootstrap...")

    if not P33.exists():
        raise FileNotFoundError(P33)

    df33 = pd.read_csv(P33)
    cluster_col33 = find_cluster_col(df33)
    print(f"Exp33 cluster column: {cluster_col33}")

    require_cols(df33, ["Label", "esm_prob", "gcn_prob"], "Exp33 table")
    df33["Label"] = pd.to_numeric(df33["Label"], errors="coerce").astype(int)
    df33["esm_prob"] = pd.to_numeric(df33["esm_prob"], errors="coerce")
    df33["gcn_prob"] = pd.to_numeric(df33["gcn_prob"], errors="coerce")

    key_results = {}

    print("[4/5] Running 42B clustered bootstrap for key joint topology cells...")

    for i, spec in enumerate(KEY_CELLS):
        require_cols(df33, [spec["row_col"], spec["col_col"]], "Exp33 table")

        sub = df33[
            (df33[spec["row_col"]] == spec["row_val"]) &
            (df33[spec["col_col"]] == spec["col_val"])
        ].copy()

        print(f"  {spec['name']}: n={len(sub)}")

        if len(sub) < 50:
            key_results[spec["name"]] = {
                "error": "too_few_rows",
                "n": int(len(sub))
            }
            continue

        res = clustered_bootstrap_delta(
            df=sub,
            y_col="Label",
            p_base_col="esm_prob",
            p_graph_col="gcn_prob",
            cluster_col=cluster_col33,
            n_boot=N_BOOT,
            seed=SEED + 100 + i,
        )

        res["cell_spec"] = spec
        key_results[spec["name"]] = res

    with open(OUT_DIR / "42B_key_cells_clustered_bootstrap.json", "w") as f:
        json.dump(key_results, f, indent=2)

    # Optional probability discovery for CADD/AM, not executed automatically.
    print("[5/5] Discovering optional CADD/AM probability files...")
    discovered = discover_prob_files()

    summary = {
        "experiment": "42_clustered_bootstrap_key_claims",
        "real_data_policy": {
            "uses_real_exp35_table": str(P35),
            "uses_real_exp33_table": str(P33),
            "uses_real_esm_probs": str(ESM_PROBS),
            "uses_real_gcn_probs": str(GCN_PROBS),
            "no_fake_labels": True,
            "no_simulated_predictions": True,
            "clustered_bootstrap_unit": "STRING_protein_id_or_available_protein_cluster_column"
        },
        "config": {
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "cluster_col_exp35": cluster_col35,
            "cluster_col_exp33": cluster_col33,
        },
        "outputs": {
            "42A_gcn_vs_esm": str(OUT_DIR / "42A_gcn_vs_esm_clustered_bootstrap.json"),
            "42B_key_cells": str(OUT_DIR / "42B_key_cells_clustered_bootstrap.json"),
            "summary": str(OUT_DIR / "summary.json")
        },
        "optional_cadd_am_probability_file_discovery": discovered,
        "note": (
            "CADD/AM clustered bootstrap is not executed automatically because "
            "probability file paths may vary. Use discovered paths if needed."
        )
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
