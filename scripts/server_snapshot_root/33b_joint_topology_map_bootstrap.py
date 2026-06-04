#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 33B:
Bootstrap stability analysis for key joint topology gain-map cells (REAL DATA ONLY)

Goal:
- Add bootstrap confidence intervals to key cells identified in Exp 33.
- Test whether strongest positive / representative negative joint cells are stable.

Reuses:
- Exp 33 merged topology profile features table

No fake data, no simulated labels, no placeholder predictions.
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

OUT_DIR = Path("/root/data/processed/topology_profile_joint_map_bootstrap_exp33B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
N_BOOT = 1000
BOOT_SEED = 42

KEY_CELLS = [
    {
        "cell_name": "mid_kcore__high_betweenness",
        "row_col": "kcore_bucket",
        "row_val": 1,
        "col_col": "betweenness_bucket",
        "col_val": 2,
    },
    {
        "cell_name": "mid_kcore__mid_clustering",
        "row_col": "kcore_bucket",
        "row_val": 1,
        "col_col": "clustering_bucket",
        "col_val": 1,
    },
    {
        "cell_name": "high_neighbor_risk__mid_kcore",
        "row_col": "neighbor_risk_bucket",
        "row_val": 2,
        "col_col": "kcore_bucket",
        "col_val": 1,
    },
    {
        "cell_name": "high_kcore__low_clustering",
        "row_col": "kcore_bucket",
        "row_val": 2,
        "col_col": "clustering_bucket",
        "col_val": 0,
    },
]

# =========================
# Helpers
# =========================
def assert_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))

def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))

def safe_brier(y, p):
    return float(brier_score_loss(y, p))

def compute_delta_metrics(df_sub):
    y = df_sub["Label"].astype(int).values
    p_esm = df_sub["esm_prob"].astype(float).values
    p_gcn = df_sub["gcn_prob"].astype(float).values

    auc_esm = safe_auc(y, p_esm)
    auc_gcn = safe_auc(y, p_gcn)

    auprc_esm = safe_auprc(y, p_esm)
    auprc_gcn = safe_auprc(y, p_gcn)

    brier_esm = safe_brier(y, p_esm)
    brier_gcn = safe_brier(y, p_gcn)

    return {
        "delta_auc": auc_gcn - auc_esm if not np.isnan(auc_esm) and not np.isnan(auc_gcn) else np.nan,
        "delta_auprc": auprc_gcn - auprc_esm if not np.isnan(auprc_esm) and not np.isnan(auprc_gcn) else np.nan,
        "delta_brier": brier_gcn - brier_esm,
    }

def bootstrap_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    lower = float(np.quantile(values, alpha / 2))
    median = float(np.quantile(values, 0.5))
    upper = float(np.quantile(values, 1 - alpha / 2))
    return lower, median, upper

# =========================
# Main
# =========================
print("[1/4] Loading Exp 33 merged profile table...")
assert_exists(P33)
df = pd.read_csv(P33)

required_cols = [
    "Label", "esm_prob", "gcn_prob",
    "kcore_bucket", "betweenness_bucket",
    "clustering_bucket", "neighbor_risk_bucket"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Exp33 merged table missing required columns: {missing}")

print(f"Rows loaded: {len(df)}")

rng = np.random.default_rng(BOOT_SEED)

print("[2/4] Running bootstrap for key cells...")
summary_rows = []
boot_rows = []

for spec in KEY_CELLS:
    sub = df[
        (df[spec["row_col"]] == spec["row_val"]) &
        (df[spec["col_col"]] == spec["col_val"])
    ].copy().reset_index(drop=True)

    if len(sub) == 0:
        raise ValueError(f"Empty cell: {spec}")

    point = compute_delta_metrics(sub)

    auc_list = []
    auprc_list = []
    brier_list = []

    n = len(sub)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_sub = sub.iloc[idx]
        d = compute_delta_metrics(boot_sub)
        auc_list.append(d["delta_auc"])
        auprc_list.append(d["delta_auprc"])
        brier_list.append(d["delta_brier"])

    auc_l, auc_m, auc_u = bootstrap_ci(auc_list)
    auprc_l, auprc_m, auprc_u = bootstrap_ci(auprc_list)
    brier_l, brier_m, brier_u = bootstrap_ci(brier_list)

    summary_rows.append({
        "cell_name": spec["cell_name"],
        "row_col": spec["row_col"],
        "row_val": spec["row_val"],
        "col_col": spec["col_col"],
        "col_val": spec["col_val"],
        "n": int(len(sub)),
        "pos_rate": float(sub["Label"].mean()),

        "point_delta_auc": point["delta_auc"],
        "boot_delta_auc_ci_lower": auc_l,
        "boot_delta_auc_ci_median": auc_m,
        "boot_delta_auc_ci_upper": auc_u,

        "point_delta_auprc": point["delta_auprc"],
        "boot_delta_auprc_ci_lower": auprc_l,
        "boot_delta_auprc_ci_median": auprc_m,
        "boot_delta_auprc_ci_upper": auprc_u,

        "point_delta_brier": point["delta_brier"],
        "boot_delta_brier_ci_lower": brier_l,
        "boot_delta_brier_ci_median": brier_m,
        "boot_delta_brier_ci_upper": brier_u,
    })

    for i in range(len(auc_list)):
        boot_rows.append({
            "cell_name": spec["cell_name"],
            "bootstrap_id": i,
            "delta_auc": auc_list[i],
            "delta_auprc": auprc_list[i],
            "delta_brier": brier_list[i],
        })

summary_df = pd.DataFrame(summary_rows)
boot_df = pd.DataFrame(boot_rows)

summary_df.to_csv(OUT_DIR / "key_joint_cells_bootstrap_summary.csv", index=False)
boot_df.to_csv(OUT_DIR / "key_joint_cells_bootstrap_samples.csv", index=False)

print("[3/4] Saving outputs...")

summary = {
    "experiment": "33B_joint_topology_map_bootstrap_stability_analysis",
    "real_data_policy": {
        "reuses_real_output_from_33": str(P33),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_features": True
    },
    "n_bootstrap": int(N_BOOT),
    "bootstrap_seed": int(BOOT_SEED),
    "key_cells": KEY_CELLS,
    "outputs": {
        "bootstrap_summary_csv": str(OUT_DIR / "key_joint_cells_bootstrap_summary.csv"),
        "bootstrap_samples_csv": str(OUT_DIR / "key_joint_cells_bootstrap_samples.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[4/4] Done.")
print(f"Results saved to: {OUT_DIR}")
