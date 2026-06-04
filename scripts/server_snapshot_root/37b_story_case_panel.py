#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 37B:
Story-oriented representative corrected-case panel (REAL DATA ONLY)

Goal:
- Build a compact, story-friendly case panel from existing corrected-sample outputs.
- No model rerun, no graph recomputation.
- Reuses real corrected sample profile table from Exp 35.

Panels:
1) ESM -> GCN : FN_rescued
2) ESM -> GCN : FP_suppressed
3) CADD+AM -> Full : FN_rescued
4) CADD+AM -> Full : FP_suppressed
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
P35 = Path("/root/data/processed/corrected_sample_profile_exp35/test_variants_with_corrected_labels_and_topology_profile.csv")

OUT_DIR = Path("/root/data/processed/story_case_panel_exp37B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
N_PER_GROUP = 8

CORE_COLS = [
    "Name",
    "GeneSymbol",
    "HGVS_p",
    "ClinicalSignificance",
    "Label",
    "STRING_protein_id",
]

PRED_COLS = [
    "esm_prob",
    "gcn_prob",
    "cadd_am_prob",
    "gcn_cadd_am_prob",
]

STATUS_COLS = [
    "esm_vs_gcn_status",
    "esm_vs_gcn_transition",
    "cadd_am_vs_full_status",
    "cadd_am_vs_full_transition",
]

FEATURE_COLS = [
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

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def zscore_series(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

def prepare_scores(df):
    df = df.copy()
    df["delta_prob_esm_to_gcn"] = pd.to_numeric(df["gcn_prob"], errors="coerce") - pd.to_numeric(df["esm_prob"], errors="coerce")
    df["abs_delta_prob_esm_to_gcn"] = df["delta_prob_esm_to_gcn"].abs()

    df["delta_prob_cadd_am_to_full"] = pd.to_numeric(df["gcn_cadd_am_prob"], errors="coerce") - pd.to_numeric(df["cadd_am_prob"], errors="coerce")
    df["abs_delta_prob_cadd_am_to_full"] = df["delta_prob_cadd_am_to_full"].abs()
    return df

def rank_story_cases(df, case_type):
    x = df.copy()

    if case_type == "esm_to_gcn_fn_rescued":
        # favor large delta, high risk, higher bridge/degree, not too internal
        score = (
            1.5 * zscore_series(x["abs_delta_prob_esm_to_gcn"]) +
            1.0 * zscore_series(x["neighbor_risk_bucket"]) +
            0.7 * zscore_series(x["betweenness_bucket"]) +
            0.5 * zscore_series(x["degree_bucket"]) -
            0.4 * zscore_series(x["within_comm_neighbor_fraction"])
        )

    elif case_type == "esm_to_gcn_fp_suppressed":
        # favor large delta, low risk, more internal, tighter clustering
        score = (
            1.5 * zscore_series(x["abs_delta_prob_esm_to_gcn"]) -
            1.0 * zscore_series(x["neighbor_risk_bucket"]) +
            0.8 * zscore_series(x["clustering_bucket"]) +
            0.8 * zscore_series(x["within_comm_neighbor_fraction"]) -
            0.5 * zscore_series(x["betweenness_bucket"])
        )

    elif case_type == "cadd_am_to_full_fn_rescued":
        # favor large delta, high risk, higher core/bridge
        score = (
            1.5 * zscore_series(x["abs_delta_prob_cadd_am_to_full"]) +
            1.1 * zscore_series(x["neighbor_risk_bucket"]) +
            0.7 * zscore_series(x["betweenness_bucket"]) +
            0.5 * zscore_series(x["kcore_bucket"]) -
            0.3 * zscore_series(x["within_comm_neighbor_fraction"])
        )

    elif case_type == "cadd_am_to_full_fp_suppressed":
        # favor large delta, lower risk, high core but low clustering
        score = (
            1.5 * zscore_series(x["abs_delta_prob_cadd_am_to_full"]) -
            0.8 * zscore_series(x["neighbor_risk_bucket"]) +
            0.7 * zscore_series(x["kcore_bucket"]) -
            0.9 * zscore_series(x["clustering_bucket"]) +
            0.4 * zscore_series(x["within_comm_neighbor_fraction"])
        )

    else:
        raise ValueError(f"Unknown case_type: {case_type}")

    x["story_score"] = score
    return x.sort_values("story_score", ascending=False)

def summarize_panel(df, panel_name):
    return {
        "panel_name": panel_name,
        "n_selected": int(len(df)),
        "pos_rate": float(pd.to_numeric(df["Label"], errors="coerce").mean()),
        "mean_story_score": float(pd.to_numeric(df["story_score"], errors="coerce").mean()),
        "mean_degree_bucket": float(pd.to_numeric(df["degree_bucket"], errors="coerce").mean()),
        "mean_kcore_bucket": float(pd.to_numeric(df["kcore_bucket"], errors="coerce").mean()),
        "mean_neighbor_risk_bucket": float(pd.to_numeric(df["neighbor_risk_bucket"], errors="coerce").mean()),
        "mean_betweenness_bucket": float(pd.to_numeric(df["betweenness_bucket"], errors="coerce").mean()),
        "mean_clustering_bucket": float(pd.to_numeric(df["clustering_bucket"], errors="coerce").mean()),
        "mean_boundary_role_bucket": float(pd.to_numeric(df["boundary_role_bucket"], errors="coerce").mean()),
        "mean_within_comm_neighbor_fraction": float(pd.to_numeric(df["within_comm_neighbor_fraction"], errors="coerce").mean()),
    }

# =========================
# Main
# =========================
print("[1/4] Loading corrected sample profile table...")
assert_exists(P35)
df = pd.read_csv(P35)

df = ensure_cols(df, CORE_COLS + PRED_COLS + STATUS_COLS + FEATURE_COLS)
df = prepare_scores(df)

print(f"Rows loaded: {len(df)}")

print("[2/4] Selecting story-oriented representative cases...")

specs = [
    ("esm_to_gcn_fn_rescued", "esm_vs_gcn_transition", "FN_rescued"),
    ("esm_to_gcn_fp_suppressed", "esm_vs_gcn_transition", "FP_suppressed"),
    ("cadd_am_to_full_fn_rescued", "cadd_am_vs_full_transition", "FN_rescued"),
    ("cadd_am_to_full_fp_suppressed", "cadd_am_vs_full_transition", "FP_suppressed"),
]

panel_frames = []
panel_summaries = []

keep_cols = (
    CORE_COLS +
    PRED_COLS +
    STATUS_COLS +
    FEATURE_COLS +
    ["delta_prob_esm_to_gcn", "abs_delta_prob_esm_to_gcn",
     "delta_prob_cadd_am_to_full", "abs_delta_prob_cadd_am_to_full",
     "story_score"]
)

for panel_name, col, val in specs:
    sub = df[df[col] == val].copy()
    sub = rank_story_cases(sub, panel_name).head(N_PER_GROUP).copy()
    sub.insert(0, "panel_name", panel_name)

    sub[keep_cols if "panel_name" not in keep_cols else keep_cols].to_csv(
        OUT_DIR / f"{panel_name}.csv", index=False
    )

    tmp = sub[["panel_name"] + keep_cols].copy()
    panel_frames.append(tmp)
    panel_summaries.append(summarize_panel(sub, panel_name))

print("[3/4] Saving combined outputs...")
combined = pd.concat(panel_frames, axis=0, ignore_index=True)
combined.to_csv(OUT_DIR / "story_case_panel_compact.csv", index=False)

summary_df = pd.DataFrame(panel_summaries)
summary_df.to_csv(OUT_DIR / "story_case_panel_summary.csv", index=False)

print("[4/4] Writing summary.json...")
summary = {
    "experiment": "37B_story_case_panel",
    "real_data_policy": {
        "reuses_real_output_from_35": str(P35),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_features": True
    },
    "n_per_group": int(N_PER_GROUP),
    "panels": [x[0] for x in specs],
    "outputs": {
        "story_case_panel_compact_csv": str(OUT_DIR / "story_case_panel_compact.csv"),
        "story_case_panel_summary_csv": str(OUT_DIR / "story_case_panel_summary.csv"),
        "esm_to_gcn_fn_rescued_csv": str(OUT_DIR / "esm_to_gcn_fn_rescued.csv"),
        "esm_to_gcn_fp_suppressed_csv": str(OUT_DIR / "esm_to_gcn_fp_suppressed.csv"),
        "cadd_am_to_full_fn_rescued_csv": str(OUT_DIR / "cadd_am_to_full_fn_rescued.csv"),
        "cadd_am_to_full_fp_suppressed_csv": str(OUT_DIR / "cadd_am_to_full_fp_suppressed.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
