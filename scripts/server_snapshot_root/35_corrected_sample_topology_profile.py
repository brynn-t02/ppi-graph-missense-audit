#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 35:
Corrected sample topology / neighborhood-risk profiling (REAL DATA ONLY)

Goal:
- Profile improved / degraded / FP_suppressed / FN_rescued samples identified in Exp 34
  using topology and neighborhood-risk features computed in Exp 26B / 27 / 30 / 31 / 32.

This experiment REUSES previous real outputs and does NOT recompute graph features.

No fake data, no simulated labels, no placeholder predictions.
Hard-fails on missing files or alignment mismatch.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
P34 = Path("/root/data/processed/corrected_sample_identification_exp34/test_variants_with_corrected_sample_labels.csv")
P26B = Path("/root/data/processed/topology_gain_analysis_exp26B/test_variants_with_predictions_and_topology.csv")
P27  = Path("/root/data/processed/neighborhood_risk_enrichment_exp27/test_variants_with_neighbor_risk_and_predictions.csv")
P30  = Path("/root/data/processed/betweenness_gain_analysis_exp30/test_variants_with_betweenness_and_predictions.csv")
P31  = Path("/root/data/processed/clustering_gain_analysis_exp31/test_variants_with_clustering_and_predictions.csv")
P32  = Path("/root/data/processed/community_boundary_gain_analysis_exp32/test_variants_with_boundary_role_and_predictions.csv")

OUT_DIR = Path("/root/data/processed/corrected_sample_profile_exp35")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
ALIGN_COLS = ["Label", "STRING_protein_id"]

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

def check_alignment(named_dfs):
    base_name, base_df = named_dfs[0]
    require_cols(base_df, ALIGN_COLS, base_name)
    base = base_df[ALIGN_COLS].copy().reset_index(drop=True)
    base["STRING_protein_id"] = base["STRING_protein_id"].astype(str)

    for name, df in named_dfs[1:]:
        require_cols(df, ALIGN_COLS, name)
        cur = df[ALIGN_COLS].copy().reset_index(drop=True)
        cur["STRING_protein_id"] = cur["STRING_protein_id"].astype(str)

        if len(cur) != len(base):
            raise ValueError(
                f"Length mismatch: base={base_name} ({len(base)} rows), {name} ({len(cur)} rows)"
            )

        same = (
            (base["Label"].values == cur["Label"].values).all() and
            (base["STRING_protein_id"].values == cur["STRING_protein_id"].values).all()
        )
        if not same:
            raise ValueError(f"Row alignment mismatch detected between {base_name} and {name}")

def summarize_status_group(df, status_col, transition_col, model_pair_name):
    rows = []

    status_groups = sorted(df[status_col].dropna().unique().tolist())
    transition_groups = sorted([x for x in df[transition_col].dropna().unique().tolist() if x != "none"])

    def one_row(sub, group_type, group_name):
        return {
            "model_pair": model_pair_name,
            "group_type": group_type,
            "group_name": group_name,
            "n": int(len(sub)),
            "pos_rate": float(sub["Label"].mean()),

            "mean_degree_bucket": float(sub["degree_bucket"].mean()),
            "mean_kcore_bucket": float(sub["kcore_bucket"].mean()),
            "mean_community_id": float(sub["community_id"].mean()),

            "mean_neighbor_risk_bucket": float(sub["neighbor_risk_bucket"].mean()),
            "mean_neighbor_pathogenic_rate": float(sub["neighbor_pathogenic_rate"].mean()),
            "mean_neighbor_train_variant_count": float(sub["neighbor_train_variant_count"].mean()),

            "mean_betweenness_bucket": float(sub["betweenness_bucket"].mean()),
            "mean_betweenness": float(sub["betweenness"].mean()),

            "mean_clustering_bucket": float(sub["clustering_bucket"].mean()),
            "mean_clustering_coefficient": float(sub["clustering_coefficient"].mean()),

            "mean_boundary_role_bucket": float(sub["boundary_role_bucket"].mean()),
            "mean_within_comm_neighbor_fraction": float(sub["within_comm_neighbor_fraction"].mean()),
        }

    for g in status_groups:
        sub = df[df[status_col] == g].copy()
        if len(sub) > 0:
            rows.append(one_row(sub, "pair_status", g))

    for g in transition_groups:
        sub = df[df[transition_col] == g].copy()
        if len(sub) > 0:
            rows.append(one_row(sub, "error_transition", g))

    return pd.DataFrame(rows)

# =========================
# 1) Load reused outputs
# =========================
print("[1/5] Loading reused outputs...")
for p in [P34, P26B, P27, P30, P31, P32]:
    assert_exists(p)

df34 = pd.read_csv(P34)
df26 = pd.read_csv(P26B)
df27 = pd.read_csv(P27)
df30 = pd.read_csv(P30)
df31 = pd.read_csv(P31)
df32 = pd.read_csv(P32)

check_alignment([
    ("34", df34),
    ("26B", df26),
    ("27", df27),
    ("30", df30),
    ("31", df31),
    ("32", df32),
])

print(f"Aligned rows: {len(df34)}")

# =========================
# 2) Build merged corrected-profile table
# =========================
print("[2/5] Building merged corrected-profile table...")

require_cols(df34, [
    "Label", "STRING_protein_id",
    "esm_vs_gcn_status", "esm_vs_gcn_transition",
    "cadd_am_vs_full_status", "cadd_am_vs_full_transition"
], "34")

require_cols(df26, ["degree_bucket", "kcore_bucket", "community_id"], "26B")
require_cols(df27, ["neighbor_risk_bucket", "neighbor_pathogenic_rate", "neighbor_train_variant_count"], "27")
require_cols(df30, ["betweenness_bucket", "betweenness"], "30")
require_cols(df31, ["clustering_bucket", "clustering_coefficient"], "31")
require_cols(df32, ["boundary_role_bucket", "within_comm_neighbor_fraction"], "32")

base_cols = ["Label", "STRING_protein_id",
             "esm_vs_gcn_status", "esm_vs_gcn_transition",
             "cadd_am_vs_full_status", "cadd_am_vs_full_transition"]

if "Name" in df34.columns:
    base_cols = ["Name"] + base_cols

df = df34[base_cols].copy().reset_index(drop=True)

df["degree_bucket"] = df26["degree_bucket"].values
df["kcore_bucket"] = df26["kcore_bucket"].values
df["community_id"] = df26["community_id"].values

df["neighbor_risk_bucket"] = df27["neighbor_risk_bucket"].values
df["neighbor_pathogenic_rate"] = df27["neighbor_pathogenic_rate"].values
df["neighbor_train_variant_count"] = df27["neighbor_train_variant_count"].values

df["betweenness_bucket"] = df30["betweenness_bucket"].values
df["betweenness"] = df30["betweenness"].values

df["clustering_bucket"] = df31["clustering_bucket"].values
df["clustering_coefficient"] = df31["clustering_coefficient"].values

df["boundary_role_bucket"] = df32["boundary_role_bucket"].values
df["within_comm_neighbor_fraction"] = df32["within_comm_neighbor_fraction"].values

df.to_csv(OUT_DIR / "test_variants_with_corrected_labels_and_topology_profile.csv", index=False)
print("Saved merged corrected-profile table.")

# =========================
# 3) Build summaries
# =========================
print("[3/5] Summarizing corrected sample profiles...")

esm_gcn_summary = summarize_status_group(
    df,
    status_col="esm_vs_gcn_status",
    transition_col="esm_vs_gcn_transition",
    model_pair_name="esm_vs_gcn"
)
esm_gcn_summary.to_csv(OUT_DIR / "esm_to_gcn_corrected_profile_summary.csv", index=False)

base_full_summary = summarize_status_group(
    df,
    status_col="cadd_am_vs_full_status",
    transition_col="cadd_am_vs_full_transition",
    model_pair_name="cadd_am_vs_full"
)
base_full_summary.to_csv(OUT_DIR / "cadd_am_to_full_corrected_profile_summary.csv", index=False)

# =========================
# 4) Save key subsets
# =========================
print("[4/5] Saving key subsets...")

subset_specs = [
    ("esm_vs_gcn_transition", "FP_suppressed", "esm_to_gcn_fp_suppressed.csv"),
    ("esm_vs_gcn_transition", "FN_rescued", "esm_to_gcn_fn_rescued.csv"),
    ("cadd_am_vs_full_transition", "FP_suppressed", "cadd_am_to_full_fp_suppressed.csv"),
    ("cadd_am_vs_full_transition", "FN_rescued", "cadd_am_to_full_fn_rescued.csv"),
]

for col, val, fname in subset_specs:
    df[df[col] == val].to_csv(OUT_DIR / fname, index=False)

# =========================
# 5) Summary JSON
# =========================
print("[5/5] Writing summary...")

summary = {
    "experiment": "35_corrected_sample_topology_profile",
    "real_data_policy": {
        "reuses_real_outputs_from_34": str(P34),
        "reuses_real_outputs_from_26b": str(P26B),
        "reuses_real_outputs_from_27": str(P27),
        "reuses_real_outputs_from_30": str(P30),
        "reuses_real_outputs_from_31": str(P31),
        "reuses_real_outputs_from_32": str(P32),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_features": True,
        "hard_fail_on_alignment_mismatch": True
    },
    "n_rows": int(len(df)),
    "esm_vs_gcn_group_counts": df["esm_vs_gcn_status"].value_counts(dropna=False).to_dict(),
    "cadd_am_vs_full_group_counts": df["cadd_am_vs_full_status"].value_counts(dropna=False).to_dict(),
    "outputs": {
        "merged_profile_csv": str(OUT_DIR / "test_variants_with_corrected_labels_and_topology_profile.csv"),
        "esm_to_gcn_summary_csv": str(OUT_DIR / "esm_to_gcn_corrected_profile_summary.csv"),
        "cadd_am_to_full_summary_csv": str(OUT_DIR / "cadd_am_to_full_corrected_profile_summary.csv"),
        "esm_to_gcn_fp_suppressed_csv": str(OUT_DIR / "esm_to_gcn_fp_suppressed.csv"),
        "esm_to_gcn_fn_rescued_csv": str(OUT_DIR / "esm_to_gcn_fn_rescued.csv"),
        "cadd_am_to_full_fp_suppressed_csv": str(OUT_DIR / "cadd_am_to_full_fp_suppressed.csv"),
        "cadd_am_to_full_fn_rescued_csv": str(OUT_DIR / "cadd_am_to_full_fn_rescued.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
