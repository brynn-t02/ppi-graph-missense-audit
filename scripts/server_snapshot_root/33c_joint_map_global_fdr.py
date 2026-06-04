#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 33C:
Global multiple-testing control for joint topology gain maps.

Goal:
- Reuse Exp33 merged topology profile table.
- Compute GCN-vs-ESM delta metrics for all valid joint topology cells.
- Use bootstrap sign test to estimate empirical p-values.
- Apply both BH-FDR and BY-FDR across all tested cells.
- Identify globally significant positive and negative joint topology cells.

REAL DATA ONLY:
- No fake labels.
- No simulated predictions.
- No graph recomputation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# =========================
# Paths
# =========================
P33 = Path("/root/data/processed/topology_profile_joint_map_exp33/merged_topology_profile_features.csv")

OUT_DIR = Path("/root/data/processed/joint_map_global_fdr_exp33C")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
N_BOOT = 1000
SEED = 42
MIN_CELL_N = 200
MIN_POS = 20
MIN_NEG = 20

JOINT_SPECS = [
    ("kcore_bucket", "betweenness_bucket"),
    ("kcore_bucket", "clustering_bucket"),
    ("neighbor_risk_bucket", "kcore_bucket"),
    ("betweenness_bucket", "boundary_role_bucket"),
    ("clustering_bucket", "boundary_role_bucket"),
    ("neighbor_risk_bucket", "betweenness_bucket"),
    ("neighbor_risk_bucket", "clustering_bucket"),
]

# =========================
# Helpers
# =========================
def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

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

def compute_delta_metrics(y, p_base, p_graph):
    auc_base = safe_auc(y, p_base)
    auc_graph = safe_auc(y, p_graph)

    auprc_base = safe_auprc(y, p_base)
    auprc_graph = safe_auprc(y, p_graph)

    brier_base = brier_score_loss(y, p_base)
    brier_graph = brier_score_loss(y, p_graph)

    return {
        "delta_auc": auc_graph - auc_base if not np.isnan(auc_base) and not np.isnan(auc_graph) else np.nan,
        "delta_auprc": auprc_graph - auprc_base if not np.isnan(auprc_base) and not np.isnan(auprc_graph) else np.nan,
        "delta_brier": brier_graph - brier_base,
    }

def bootstrap_delta(y, p_base, p_graph, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)

    d_auc = []
    d_auprc = []
    d_brier = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        bb = p_base[idx]
        gg = p_graph[idx]

        d = compute_delta_metrics(yy, bb, gg)
        d_auc.append(d["delta_auc"])
        d_auprc.append(d["delta_auprc"])
        d_brier.append(d["delta_brier"])

    return {
        "delta_auc": np.asarray(d_auc, dtype=float),
        "delta_auprc": np.asarray(d_auprc, dtype=float),
        "delta_brier": np.asarray(d_brier, dtype=float),
    }

def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))

def empirical_sign_p(arr, observed):
    """
    Two-sided bootstrap sign-style empirical p-value:
    if observed > 0, p = P(boot <= 0)
    if observed < 0, p = P(boot >= 0)
    doubled and capped at 1.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0 or np.isnan(observed):
        return np.nan

    if observed >= 0:
        one_side = (np.sum(arr <= 0) + 1) / (len(arr) + 1)
    else:
        one_side = (np.sum(arr >= 0) + 1) / (len(arr) + 1)

    return float(min(1.0, 2 * one_side))

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out

def by_fdr(pvals):
    """
    Benjamini-Yekutieli FDR, more conservative under arbitrary dependency.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    c_m = np.sum(1 / np.arange(1, n + 1))
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n * c_m / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out

# =========================
# Main
# =========================
print("[1/5] Loading Exp33 merged topology profile table...")

if not P33.exists():
    raise FileNotFoundError(f"Missing required file: {P33}")

df = pd.read_csv(P33)

required = [
    "Label",
    "esm_prob",
    "gcn_prob",
    "kcore_bucket",
    "betweenness_bucket",
    "clustering_bucket",
    "neighbor_risk_bucket",
    "boundary_role_bucket",
]

require_cols(df, required, "Exp33 merged table")

df["Label"] = pd.to_numeric(df["Label"], errors="coerce").astype(int)
df["esm_prob"] = pd.to_numeric(df["esm_prob"], errors="coerce")
df["gcn_prob"] = pd.to_numeric(df["gcn_prob"], errors="coerce")

df = df.dropna(subset=["Label", "esm_prob", "gcn_prob"]).copy()

print(f"Rows loaded after required probability cleanup: {len(df)}")

print("[2/5] Enumerating valid joint topology cells...")

rows = []
boot_rows = []

cell_id = 0

for row_col, col_col in JOINT_SPECS:
    require_cols(df, [row_col, col_col], "Exp33 merged table")

    valid_df = df.dropna(subset=[row_col, col_col]).copy()

    for row_val in sorted(valid_df[row_col].dropna().unique()):
        for col_val in sorted(valid_df[col_col].dropna().unique()):

            sub = valid_df[
                (valid_df[row_col] == row_val) &
                (valid_df[col_col] == col_val)
            ].copy()

            n = len(sub)
            if n < MIN_CELL_N:
                continue

            n_pos = int((sub["Label"] == 1).sum())
            n_neg = int((sub["Label"] == 0).sum())

            if n_pos < MIN_POS or n_neg < MIN_NEG:
                continue

            y = sub["Label"].values.astype(int)
            p_esm = sub["esm_prob"].values.astype(float)
            p_gcn = sub["gcn_prob"].values.astype(float)

            point = compute_delta_metrics(y, p_esm, p_gcn)

            boot = bootstrap_delta(
                y=y,
                p_base=p_esm,
                p_graph=p_gcn,
                n_boot=N_BOOT,
                seed=SEED + cell_id
            )

            auc_lo, auc_hi = ci95(boot["delta_auc"])
            auprc_lo, auprc_hi = ci95(boot["delta_auprc"])
            brier_lo, brier_hi = ci95(boot["delta_brier"])

            p_auc = empirical_sign_p(boot["delta_auc"], point["delta_auc"])
            p_auprc = empirical_sign_p(boot["delta_auprc"], point["delta_auprc"])
            p_brier = empirical_sign_p(boot["delta_brier"], point["delta_brier"])

            rows.append({
                "cell_id": cell_id,
                "joint_spec": f"{row_col}__x__{col_col}",
                "row_col": row_col,
                "row_val": row_val,
                "col_col": col_col,
                "col_val": col_val,
                "n": int(n),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "pos_rate": float(n_pos / n),

                "point_delta_auc": point["delta_auc"],
                "delta_auc_ci95_low": auc_lo,
                "delta_auc_ci95_high": auc_hi,
                "p_auc": p_auc,

                "point_delta_auprc": point["delta_auprc"],
                "delta_auprc_ci95_low": auprc_lo,
                "delta_auprc_ci95_high": auprc_hi,
                "p_auprc": p_auprc,

                "point_delta_brier": point["delta_brier"],
                "delta_brier_ci95_low": brier_lo,
                "delta_brier_ci95_high": brier_hi,
                "p_brier": p_brier,
            })

            cell_id += 1

print(f"Valid tested cells: {len(rows)}")

if len(rows) == 0:
    raise RuntimeError("No valid joint cells found. Check MIN_CELL_N / feature columns.")

res = pd.DataFrame(rows)

print("[3/5] Applying BH-FDR and BY-FDR globally across tested cells...")

for metric in ["auc", "auprc", "brier"]:
    p_col = f"p_{metric}"
    valid = res[p_col].notna().values

    bh = np.full(len(res), np.nan)
    by = np.full(len(res), np.nan)

    if valid.sum() > 0:
        bh[valid] = bh_fdr(res.loc[valid, p_col].values)
        by[valid] = by_fdr(res.loc[valid, p_col].values)

    res[f"q_bh_{metric}"] = bh
    res[f"q_by_{metric}"] = by

    if metric in ["auc", "auprc"]:
        res[f"significant_bh_{metric}_0_05"] = res[f"q_bh_{metric}"] < 0.05
        res[f"significant_by_{metric}_0_05"] = res[f"q_by_{metric}"] < 0.05
    else:
        # for brier, negative delta is beneficial, but significance just means nonzero
        res[f"significant_bh_{metric}_0_05"] = res[f"q_bh_{metric}"] < 0.05
        res[f"significant_by_{metric}_0_05"] = res[f"q_by_{metric}"] < 0.05

print("[4/5] Saving result tables...")

out_all = OUT_DIR / "joint_cells_global_fdr_results.csv"
res.to_csv(out_all, index=False)

# Key cells from Exp33B for easy lookup
key_mask = (
    ((res["row_col"] == "kcore_bucket") & (res["row_val"] == 1) &
     (res["col_col"] == "betweenness_bucket") & (res["col_val"] == 2))
    |
    ((res["row_col"] == "kcore_bucket") & (res["row_val"] == 1) &
     (res["col_col"] == "clustering_bucket") & (res["col_val"] == 1))
    |
    ((res["row_col"] == "neighbor_risk_bucket") & (res["row_val"] == 2) &
     (res["col_col"] == "kcore_bucket") & (res["col_val"] == 1))
    |
    ((res["row_col"] == "kcore_bucket") & (res["row_val"] == 2) &
     (res["col_col"] == "clustering_bucket") & (res["col_val"] == 0))
)

key_res = res[key_mask].copy()
key_res.to_csv(OUT_DIR / "key_cells_global_fdr_lookup.csv", index=False)

# Top positive and negative cells by AUC and AUPRC
top_auc_pos = res.sort_values("point_delta_auc", ascending=False).head(20)
top_auc_neg = res.sort_values("point_delta_auc", ascending=True).head(20)
top_auprc_pos = res.sort_values("point_delta_auprc", ascending=False).head(20)
top_auprc_neg = res.sort_values("point_delta_auprc", ascending=True).head(20)

top_auc_pos.to_csv(OUT_DIR / "top20_positive_delta_auc_cells.csv", index=False)
top_auc_neg.to_csv(OUT_DIR / "top20_negative_delta_auc_cells.csv", index=False)
top_auprc_pos.to_csv(OUT_DIR / "top20_positive_delta_auprc_cells.csv", index=False)
top_auprc_neg.to_csv(OUT_DIR / "top20_negative_delta_auprc_cells.csv", index=False)

print("[5/5] Writing summary.json...")

summary = {
    "experiment": "33C_joint_map_global_fdr",
    "real_data_policy": {
        "reuses_real_output_from_33": str(P33),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_features": True
    },
    "bootstrap": {
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "empirical_sign_p": True
    },
    "cell_filtering": {
        "min_cell_n": MIN_CELL_N,
        "min_pos": MIN_POS,
        "min_neg": MIN_NEG
    },
    "joint_specs_tested": JOINT_SPECS,
    "n_valid_tested_cells": int(len(res)),
    "global_multiple_testing": {
        "BH_FDR": True,
        "BY_FDR": True,
        "scope": "all valid joint cells across all configured joint specs"
    },
    "counts_significant": {
        "bh_auc_0_05": int(res["significant_bh_auc_0_05"].sum()),
        "by_auc_0_05": int(res["significant_by_auc_0_05"].sum()),
        "bh_auprc_0_05": int(res["significant_bh_auprc_0_05"].sum()),
        "by_auprc_0_05": int(res["significant_by_auprc_0_05"].sum()),
    },
    "outputs": {
        "all_cells_global_fdr_csv": str(out_all),
        "key_cells_lookup_csv": str(OUT_DIR / "key_cells_global_fdr_lookup.csv"),
        "top20_positive_delta_auc_cells_csv": str(OUT_DIR / "top20_positive_delta_auc_cells.csv"),
        "top20_negative_delta_auc_cells_csv": str(OUT_DIR / "top20_negative_delta_auc_cells.csv"),
        "top20_positive_delta_auprc_cells_csv": str(OUT_DIR / "top20_positive_delta_auprc_cells.csv"),
        "top20_negative_delta_auprc_cells_csv": str(OUT_DIR / "top20_negative_delta_auprc_cells.csv"),
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
print(json.dumps(summary["counts_significant"], indent=2))
