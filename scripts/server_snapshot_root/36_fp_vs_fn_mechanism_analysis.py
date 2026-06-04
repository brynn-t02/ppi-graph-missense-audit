#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 36:
FP correction vs FN rescue mechanism analysis (REAL DATA ONLY)

Goal:
- Explicitly compare FP_suppressed vs FN_rescued samples for:
    1) ESM -> GCN
    2) CADD+AM -> GCN+CADD+AM
- Quantify whether these two correction types correspond to different topology /
  neighborhood-risk profiles.

Reuses Exp 35 merged corrected-profile table.
No fake data, no simulated labels, no placeholder predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
P35 = Path("/root/data/processed/corrected_sample_profile_exp35/test_variants_with_corrected_labels_and_topology_profile.csv")

OUT_DIR = Path("/root/data/processed/fp_vs_fn_mechanism_exp36")
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

def require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def smd(x1, x2):
    """
    Standardized mean difference:
    (mean1 - mean2) / pooled_std
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    m1 = np.mean(x1)
    m2 = np.mean(x2)
    s1 = np.std(x1, ddof=1) if len(x1) > 1 else 0.0
    s2 = np.std(x2, ddof=1) if len(x2) > 1 else 0.0

    pooled_var = ((s1 ** 2) + (s2 ** 2)) / 2.0
    if pooled_var <= 0:
        return 0.0
    return float((m1 - m2) / np.sqrt(pooled_var))

def summarize_one_group(df, group_name, feature_cols):
    out = {
        "group_name": group_name,
        "n": int(len(df)),
        "pos_rate": float(df["Label"].mean())
    }
    for c in feature_cols:
        out[f"mean_{c}"] = float(df[c].mean())
        out[f"std_{c}"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return out

def compare_two_groups(df, transition_col, group_a, group_b, pair_name, feature_cols):
    sub_a = df[df[transition_col] == group_a].copy()
    sub_b = df[df[transition_col] == group_b].copy()

    if len(sub_a) == 0 or len(sub_b) == 0:
        raise ValueError(f"{pair_name}: one of the groups is empty ({group_a}, {group_b})")

    rows = []
    for c in feature_cols:
        rows.append({
            "model_pair": pair_name,
            "group_a": group_a,
            "group_b": group_b,
            "feature": c,
            "mean_a": float(sub_a[c].mean()),
            "mean_b": float(sub_b[c].mean()),
            "diff_a_minus_b": float(sub_a[c].mean() - sub_b[c].mean()),
            "smd_a_minus_b": smd(sub_a[c].values, sub_b[c].values)
        })

    return pd.DataFrame(rows), sub_a, sub_b

# =========================
# 1) Load reused corrected-profile table
# =========================
print("[1/4] Loading reused corrected-profile table...")
assert_exists(P35)
df = pd.read_csv(P35)

required = [
    "Label",
    "esm_vs_gcn_transition",
    "cadd_am_vs_full_transition",
] + FEATURES

require_cols(df, required, "35_merged_profile")

print(f"Rows loaded: {len(df)}")

# =========================
# 2) Compare FP_suppressed vs FN_rescued
# =========================
print("[2/4] Comparing FP_suppressed vs FN_rescued...")

esm_cmp, esm_fp, esm_fn = compare_two_groups(
    df,
    transition_col="esm_vs_gcn_transition",
    group_a="FP_suppressed",
    group_b="FN_rescued",
    pair_name="esm_vs_gcn",
    feature_cols=FEATURES
)

full_cmp, full_fp, full_fn = compare_two_groups(
    df,
    transition_col="cadd_am_vs_full_transition",
    group_a="FP_suppressed",
    group_b="FN_rescued",
    pair_name="cadd_am_vs_full",
    feature_cols=FEATURES
)

esm_cmp.to_csv(OUT_DIR / "esm_to_gcn_fp_vs_fn_feature_comparison.csv", index=False)
full_cmp.to_csv(OUT_DIR / "cadd_am_to_full_fp_vs_fn_feature_comparison.csv", index=False)

# =========================
# 3) Save group summaries
# =========================
print("[3/4] Saving group summaries...")

esm_group_summary = pd.DataFrame([
    summarize_one_group(esm_fp, "FP_suppressed", FEATURES),
    summarize_one_group(esm_fn, "FN_rescued", FEATURES),
])
esm_group_summary.insert(0, "model_pair", "esm_vs_gcn")
esm_group_summary.to_csv(OUT_DIR / "esm_to_gcn_fp_vs_fn_group_summary.csv", index=False)

full_group_summary = pd.DataFrame([
    summarize_one_group(full_fp, "FP_suppressed", FEATURES),
    summarize_one_group(full_fn, "FN_rescued", FEATURES),
])
full_group_summary.insert(0, "model_pair", "cadd_am_vs_full")
full_group_summary.to_csv(OUT_DIR / "cadd_am_to_full_fp_vs_fn_group_summary.csv", index=False)

# Also export actual subsets
esm_fp.to_csv(OUT_DIR / "esm_to_gcn_fp_suppressed_profiled.csv", index=False)
esm_fn.to_csv(OUT_DIR / "esm_to_gcn_fn_rescued_profiled.csv", index=False)
full_fp.to_csv(OUT_DIR / "cadd_am_to_full_fp_suppressed_profiled.csv", index=False)
full_fn.to_csv(OUT_DIR / "cadd_am_to_full_fn_rescued_profiled.csv", index=False)

# =========================
# 4) Write summary JSON
# =========================
print("[4/4] Writing summary...")

summary = {
    "experiment": "36_fp_vs_fn_mechanism_analysis",
    "real_data_policy": {
        "reuses_real_output_from_35": str(P35),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_features": True
    },
    "features_compared": FEATURES,
    "esm_vs_gcn": {
        "n_fp_suppressed": int(len(esm_fp)),
        "n_fn_rescued": int(len(esm_fn)),
        "comparison_csv": str(OUT_DIR / "esm_to_gcn_fp_vs_fn_feature_comparison.csv"),
        "group_summary_csv": str(OUT_DIR / "esm_to_gcn_fp_vs_fn_group_summary.csv")
    },
    "cadd_am_vs_full": {
        "n_fp_suppressed": int(len(full_fp)),
        "n_fn_rescued": int(len(full_fn)),
        "comparison_csv": str(OUT_DIR / "cadd_am_to_full_fp_vs_fn_feature_comparison.csv"),
        "group_summary_csv": str(OUT_DIR / "cadd_am_to_full_fp_vs_fn_group_summary.csv")
    },
    "outputs": {
        "esm_to_gcn_fp_vs_fn_feature_comparison_csv": str(OUT_DIR / "esm_to_gcn_fp_vs_fn_feature_comparison.csv"),
        "esm_to_gcn_fp_vs_fn_group_summary_csv": str(OUT_DIR / "esm_to_gcn_fp_vs_fn_group_summary.csv"),
        "cadd_am_to_full_fp_vs_fn_feature_comparison_csv": str(OUT_DIR / "cadd_am_to_full_fp_vs_fn_feature_comparison.csv"),
        "cadd_am_to_full_fp_vs_fn_group_summary_csv": str(OUT_DIR / "cadd_am_to_full_fp_vs_fn_group_summary.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
