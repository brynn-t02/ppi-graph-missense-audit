#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 36B:
Statistical significance testing for FP_suppressed vs FN_rescued (REAL DATA ONLY)

Goal:
- Add formal statistical testing to Exp 36
- Compare FP_suppressed vs FN_rescued using Mann-Whitney U tests
- Apply BH-FDR correction

Comparisons:
1) ESM -> GCN
2) CADD+AM -> GCN+CADD+AM
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# =========================
# Paths
# =========================
P35 = Path("/root/data/processed/corrected_sample_profile_exp35/test_variants_with_corrected_labels_and_topology_profile.csv")

OUT_DIR = Path("/root/data/processed/fp_vs_fn_stats_exp36B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
FEATURES = [
    "degree_bucket",
    "kcore_bucket",
    "neighbor_risk_bucket",
    "neighbor_pathogenic_rate",
    "neighbor_train_variant_count",
    "betweenness_bucket",
    "betweenness",
    "clustering_bucket",
    "clustering_coefficient",
    "boundary_role_bucket",
    "within_comm_neighbor_fraction",
]

# =========================
# Helpers
# =========================
def assert_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction
    """
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

def run_test(df, transition_col, pair_name):
    fp = df[df[transition_col] == "FP_suppressed"].copy()
    fn = df[df[transition_col] == "FN_rescued"].copy()

    rows = []
    pvals = []

    for feat in FEATURES:
        x = pd.to_numeric(fp[feat], errors="coerce").dropna().values
        y = pd.to_numeric(fn[feat], errors="coerce").dropna().values

        if len(x) == 0 or len(y) == 0:
            stat = np.nan
            p = np.nan
        else:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")

        rows.append({
            "model_pair": pair_name,
            "feature": feat,
            "n_fp_suppressed": int(len(x)),
            "n_fn_rescued": int(len(y)),
            "mean_fp_suppressed": float(np.mean(x)) if len(x) > 0 else np.nan,
            "mean_fn_rescued": float(np.mean(y)) if len(y) > 0 else np.nan,
            "diff_fp_minus_fn": float(np.mean(x) - np.mean(y)) if len(x) > 0 and len(y) > 0 else np.nan,
            "u_statistic": float(stat) if not np.isnan(stat) else np.nan,
            "p_value": float(p) if not np.isnan(p) else np.nan
        })
        pvals.append(p)

    res = pd.DataFrame(rows)

    valid_mask = res["p_value"].notna().values
    adj = np.full(len(res), np.nan, dtype=float)
    if valid_mask.sum() > 0:
        adj_vals = bh_fdr(res.loc[valid_mask, "p_value"].values)
        adj[valid_mask] = adj_vals

    res["fdr_bh"] = adj
    res["significant_fdr_0_05"] = res["fdr_bh"] < 0.05
    return res

# =========================
# Main
# =========================
print("[1/3] Loading real corrected-profile table...")
assert_exists(P35)
df = pd.read_csv(P35)

required = ["esm_vs_gcn_transition", "cadd_am_vs_full_transition"] + FEATURES
require_cols(df, required, "P35")

print(f"Rows loaded: {len(df)}")

print("[2/3] Running Mann-Whitney U tests...")
res_esm = run_test(df, "esm_vs_gcn_transition", "esm_vs_gcn")
res_full = run_test(df, "cadd_am_vs_full_transition", "cadd_am_vs_full")

res_esm.to_csv(OUT_DIR / "esm_to_gcn_fp_vs_fn_stats.csv", index=False)
res_full.to_csv(OUT_DIR / "cadd_am_to_full_fp_vs_fn_stats.csv", index=False)

print("[3/3] Writing summary...")
summary = {
    "experiment": "36B_fp_vs_fn_statistical_testing",
    "real_data_policy": {
        "reuses_real_output_from_35": str(P35),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_features": True
    },
    "statistical_test": "Mann-Whitney U test",
    "multiple_testing_correction": "Benjamini-Hochberg FDR",
    "features_tested": FEATURES,
    "outputs": {
        "esm_to_gcn_stats_csv": str(OUT_DIR / "esm_to_gcn_fp_vs_fn_stats.csv"),
        "cadd_am_to_full_stats_csv": str(OUT_DIR / "cadd_am_to_full_fp_vs_fn_stats.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
