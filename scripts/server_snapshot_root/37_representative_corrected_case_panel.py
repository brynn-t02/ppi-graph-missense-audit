#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 37:
Representative corrected-case panel (REAL DATA ONLY)

Goal:
- Build representative case panels for corrected samples identified in Exp 34/35.
- Focus on:
    1) ESM -> GCN : FN_rescued
    2) ESM -> GCN : FP_suppressed
    3) CADD+AM -> Full : FN_rescued
    4) CADD+AM -> Full : FP_suppressed

No fake data, no simulated labels, no placeholder predictions.
Reuses real outputs only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
P35 = Path("/root/data/processed/corrected_sample_profile_exp35/test_variants_with_corrected_labels_and_topology_profile.csv")

OUT_DIR = Path("/root/data/processed/representative_corrected_case_panel_exp37")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
TOP_K_PER_GROUP = 30

DISPLAY_COLS_CORE = [
    "Name",
    "GeneSymbol",
    "Label",
    "STRING_protein_id",
]

DISPLAY_COLS_PRED = [
    "esm_prob",
    "gcn_prob",
    "cadd_am_prob",
    "gcn_cadd_am_prob",
]

DISPLAY_COLS_STATUS = [
    "esm_vs_gcn_status",
    "esm_vs_gcn_transition",
    "cadd_am_vs_full_status",
    "cadd_am_vs_full_transition",
]

DISPLAY_COLS_TOPO = [
    "degree_bucket",
    "kcore_bucket",
    "community_id",
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

def ensure_optional_cols(df: pd.DataFrame):
    """
    Some columns may or may not exist depending on previous export choices.
    We'll create them as NaN if absent so output schema stays stable.
    """
    wanted = DISPLAY_COLS_CORE + DISPLAY_COLS_PRED + DISPLAY_COLS_STATUS + DISPLAY_COLS_TOPO
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan
    return df

def build_score_columns(df: pd.DataFrame):
    # ESM -> GCN
    if "esm_prob" in df.columns and "gcn_prob" in df.columns:
        df["delta_prob_esm_to_gcn"] = df["gcn_prob"] - df["esm_prob"]
        df["abs_delta_prob_esm_to_gcn"] = np.abs(df["delta_prob_esm_to_gcn"])
    else:
        df["delta_prob_esm_to_gcn"] = np.nan
        df["abs_delta_prob_esm_to_gcn"] = np.nan

    # CADD+AM -> Full
    if "cadd_am_prob" in df.columns and "gcn_cadd_am_prob" in df.columns:
        df["delta_prob_cadd_am_to_full"] = df["gcn_cadd_am_prob"] - df["cadd_am_prob"]
        df["abs_delta_prob_cadd_am_to_full"] = np.abs(df["delta_prob_cadd_am_to_full"])
    else:
        df["delta_prob_cadd_am_to_full"] = np.nan
        df["abs_delta_prob_cadd_am_to_full"] = np.nan

    return df

def select_panel(df: pd.DataFrame, transition_col: str, transition_value: str, rank_col: str, top_k: int):
    sub = df[df[transition_col] == transition_value].copy()
    sub = sub.sort_values(rank_col, ascending=False).head(top_k).copy()
    return sub

def summarize_panel(df: pd.DataFrame, panel_name: str):
    out = {
        "panel_name": panel_name,
        "n": int(len(df)),
        "pos_rate": float(df["Label"].mean()) if len(df) > 0 else None,
    }

    numeric_cols = [
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
        "delta_prob_esm_to_gcn",
        "delta_prob_cadd_am_to_full",
    ]

    for c in numeric_cols:
        if c in df.columns and len(df) > 0:
            out[f"mean_{c}"] = float(pd.to_numeric(df[c], errors="coerce").mean())
        else:
            out[f"mean_{c}"] = None

    return out

# =========================
# 1) Load merged corrected-profile table
# =========================
print("[1/4] Loading merged corrected-profile table...")
assert_exists(P35)
df = pd.read_csv(P35)
df = ensure_optional_cols(df)

required_min = [
    "Label",
    "esm_vs_gcn_transition",
    "cadd_am_vs_full_transition",
]
require_cols(df, required_min, "35_merged_profile")

df = build_score_columns(df)

print(f"Rows loaded: {len(df)}")

# =========================
# 2) Build four representative panels
# =========================
print("[2/4] Building representative panels...")

panel_specs = [
    {
        "panel_name": "esm_to_gcn_fn_rescued",
        "transition_col": "esm_vs_gcn_transition",
        "transition_value": "FN_rescued",
        "rank_col": "abs_delta_prob_esm_to_gcn",
        "out_csv": OUT_DIR / "esm_to_gcn_fn_rescued_top_cases.csv",
    },
    {
        "panel_name": "esm_to_gcn_fp_suppressed",
        "transition_col": "esm_vs_gcn_transition",
        "transition_value": "FP_suppressed",
        "rank_col": "abs_delta_prob_esm_to_gcn",
        "out_csv": OUT_DIR / "esm_to_gcn_fp_suppressed_top_cases.csv",
    },
    {
        "panel_name": "cadd_am_to_full_fn_rescued",
        "transition_col": "cadd_am_vs_full_transition",
        "transition_value": "FN_rescued",
        "rank_col": "abs_delta_prob_cadd_am_to_full",
        "out_csv": OUT_DIR / "cadd_am_to_full_fn_rescued_top_cases.csv",
    },
    {
        "panel_name": "cadd_am_to_full_fp_suppressed",
        "transition_col": "cadd_am_vs_full_transition",
        "transition_value": "FP_suppressed",
        "rank_col": "abs_delta_prob_cadd_am_to_full",
        "out_csv": OUT_DIR / "cadd_am_to_full_fp_suppressed_top_cases.csv",
    },
]

panel_summaries = []
panel_frames = {}

keep_cols = (
    DISPLAY_COLS_CORE +
    DISPLAY_COLS_PRED +
    DISPLAY_COLS_STATUS +
    DISPLAY_COLS_TOPO +
    [
        "delta_prob_esm_to_gcn",
        "abs_delta_prob_esm_to_gcn",
        "delta_prob_cadd_am_to_full",
        "abs_delta_prob_cadd_am_to_full",
    ]
)

for spec in panel_specs:
    panel_df = select_panel(
        df=df,
        transition_col=spec["transition_col"],
        transition_value=spec["transition_value"],
        rank_col=spec["rank_col"],
        top_k=TOP_K_PER_GROUP
    )
    panel_df = panel_df[keep_cols].copy()
    panel_df.to_csv(spec["out_csv"], index=False)

    panel_frames[spec["panel_name"]] = panel_df
    panel_summaries.append(summarize_panel(panel_df, spec["panel_name"]))

panel_summary_df = pd.DataFrame(panel_summaries)
panel_summary_df.to_csv(OUT_DIR / "representative_case_panel_summary.csv", index=False)

# =========================
# 3) Build one combined panel file
# =========================
print("[3/4] Saving combined panel file...")

combined_rows = []
for panel_name, panel_df in panel_frames.items():
    tmp = panel_df.copy()
    tmp.insert(0, "panel_name", panel_name)
    combined_rows.append(tmp)

combined_df = pd.concat(combined_rows, axis=0, ignore_index=True)
combined_df.to_csv(OUT_DIR / "all_representative_case_panels_combined.csv", index=False)

# =========================
# 4) Write summary JSON
# =========================
print("[4/4] Writing summary...")

summary = {
    "experiment": "37_representative_corrected_case_panel",
    "real_data_policy": {
        "reuses_real_output_from_35": str(P35),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_features": True
    },
    "top_k_per_group": int(TOP_K_PER_GROUP),
    "panels_built": [x["panel_name"] for x in panel_specs],
    "outputs": {
        "esm_to_gcn_fn_rescued_top_cases_csv": str(OUT_DIR / "esm_to_gcn_fn_rescued_top_cases.csv"),
        "esm_to_gcn_fp_suppressed_top_cases_csv": str(OUT_DIR / "esm_to_gcn_fp_suppressed_top_cases.csv"),
        "cadd_am_to_full_fn_rescued_top_cases_csv": str(OUT_DIR / "cadd_am_to_full_fn_rescued_top_cases.csv"),
        "cadd_am_to_full_fp_suppressed_top_cases_csv": str(OUT_DIR / "cadd_am_to_full_fp_suppressed_top_cases.csv"),
        "representative_case_panel_summary_csv": str(OUT_DIR / "representative_case_panel_summary.csv"),
        "all_representative_case_panels_combined_csv": str(OUT_DIR / "all_representative_case_panels_combined.csv"),
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
