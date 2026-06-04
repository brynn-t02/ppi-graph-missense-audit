#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 33:
Topology profile joint map / gain landscape (REAL DATA ONLY)

Goal:
- Integrate outputs from Exp 26B / 27 / 30 / 31 / 32
- Build a joint gain map showing which combinations of topology / neighborhood roles
  receive the largest GCN-over-ESM gains.

IMPORTANT:
- This script REUSES previous real outputs.
- It does NOT recompute graph features.
- It uses row-order alignment checks first, then concatenates columns directly
  to avoid merge duplication issues.

No fake data, no simulated labels, no placeholder predictions.
Hard-fails on missing files or alignment mismatch.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# =========================
# Paths (reuse previous outputs)
# =========================
P26B = Path("/root/data/processed/topology_gain_analysis_exp26B/test_variants_with_predictions_and_topology.csv")
P27  = Path("/root/data/processed/neighborhood_risk_enrichment_exp27/test_variants_with_neighbor_risk_and_predictions.csv")
P30  = Path("/root/data/processed/betweenness_gain_analysis_exp30/test_variants_with_betweenness_and_predictions.csv")
P31  = Path("/root/data/processed/clustering_gain_analysis_exp31/test_variants_with_clustering_and_predictions.csv")
P32  = Path("/root/data/processed/community_boundary_gain_analysis_exp32/test_variants_with_boundary_role_and_predictions.csv")

OUT_DIR = Path("/root/data/processed/topology_profile_joint_map_exp33")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
ALIGN_COLS = ["Label", "STRING_protein_id"]
MIN_GROUP_N = 50

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
        "brier": safe_brier(y, p)
    }

def metric_delta(base, new):
    out = {}
    for k in ["auc", "auprc", "brier"]:
        if base[k] is None or new[k] is None:
            out[f"delta_{k}"] = None
        else:
            out[f"delta_{k}"] = float(new[k] - base[k])
    return out

def check_alignment(named_dfs):
    """
    named_dfs: list of tuples -> [(name, df), ...]
    Checks row-order alignment using Label + STRING_protein_id.
    """
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
                f"Length mismatch: base={base_name} ({len(base)} rows), "
                f"{name} ({len(cur)} rows)"
            )

        same = (
            (base["Label"].values == cur["Label"].values).all() and
            (base["STRING_protein_id"].values == cur["STRING_protein_id"].values).all()
        )

        if not same:
            raise ValueError(f"Row alignment mismatch detected between {base_name} and {name}")

def eval_joint_group(df, row_col, col_col):
    rows = []
    subdf = df[[row_col, col_col, "Label", "esm_prob", "gcn_prob", "gcn_minus_esm_prob"]].copy()

    # drop rows with missing buckets
    subdf = subdf[subdf[row_col].notna() & subdf[col_col].notna()].copy()

    for (r, c), sub in subdf.groupby([row_col, col_col]):
        if len(sub) < MIN_GROUP_N:
            continue

        yy = sub["Label"].astype(int).values
        m_esm = metrics(yy, sub["esm_prob"].values)
        m_gcn = metrics(yy, sub["gcn_prob"].values)
        d = metric_delta(m_esm, m_gcn)

        rows.append({
            "row_col": row_col,
            "row_group": int(r),
            "col_col": col_col,
            "col_group": int(c),
            "n": int(len(sub)),
            "pos_rate": float(np.mean(yy)),
            "delta_auc": d["delta_auc"],
            "delta_auprc": d["delta_auprc"],
            "delta_brier": d["delta_brier"],
            "mean_gcn_minus_esm_prob": float(np.mean(sub["gcn_minus_esm_prob"]))
        })

    return pd.DataFrame(rows)

# =========================
# 1) Load reused outputs
# =========================
print("[1/6] Loading reused outputs...")
for p in [P26B, P27, P30, P31, P32]:
    assert_exists(p)

df26 = pd.read_csv(P26B)
df27 = pd.read_csv(P27)
df30 = pd.read_csv(P30)
df31 = pd.read_csv(P31)
df32 = pd.read_csv(P32)

check_alignment([
    ("26B", df26),
    ("27", df27),
    ("30", df30),
    ("31", df31),
    ("32", df32),
])

print(f"Aligned rows: {len(df26)}")

# =========================
# 2) Build merged profile table by row order
# =========================
print("[2/6] Building merged topology profile table...")

require_cols(
    df26,
    ["Label", "STRING_protein_id", "esm_prob", "gcn_prob", "gcn_minus_esm_prob",
     "degree_bucket", "kcore_bucket", "community_id"],
    "26B"
)
require_cols(
    df27,
    ["neighbor_risk_bucket", "neighbor_pathogenic_rate", "neighbor_train_variant_count"],
    "27"
)
require_cols(
    df30,
    ["betweenness_bucket", "betweenness"],
    "30"
)
require_cols(
    df31,
    ["clustering_bucket", "clustering_coefficient"],
    "31"
)
require_cols(
    df32,
    ["boundary_role_bucket", "within_comm_neighbor_fraction"],
    "32"
)

base_cols = ["Label", "STRING_protein_id", "esm_prob", "gcn_prob", "gcn_minus_esm_prob",
             "degree_bucket", "kcore_bucket", "community_id"]

if "Name" in df26.columns:
    base_cols = ["Name"] + base_cols

df = df26[base_cols].copy().reset_index(drop=True)
df["STRING_protein_id"] = df["STRING_protein_id"].astype(str)

# Direct row-wise reuse after alignment checks
df["neighbor_risk_bucket"] = df27["neighbor_risk_bucket"].values
df["neighbor_pathogenic_rate"] = df27["neighbor_pathogenic_rate"].values
df["neighbor_train_variant_count"] = df27["neighbor_train_variant_count"].values

df["betweenness_bucket"] = df30["betweenness_bucket"].values
df["betweenness"] = df30["betweenness"].values

df["clustering_bucket"] = df31["clustering_bucket"].values
df["clustering_coefficient"] = df31["clustering_coefficient"].values

df["boundary_role_bucket"] = df32["boundary_role_bucket"].values
df["within_comm_neighbor_fraction"] = df32["within_comm_neighbor_fraction"].values

df.to_csv(OUT_DIR / "merged_topology_profile_features.csv", index=False)
print("Saved merged topology profile table.")

# =========================
# 3) Define joint maps to evaluate
# =========================
print("[3/6] Evaluating joint gain maps...")

joint_specs = [
    ("kcore_bucket", "betweenness_bucket"),
    ("kcore_bucket", "clustering_bucket"),
    ("neighbor_risk_bucket", "kcore_bucket"),
    ("neighbor_risk_bucket", "clustering_bucket"),
    ("betweenness_bucket", "boundary_role_bucket"),
    ("clustering_bucket", "boundary_role_bucket"),
]

all_joint = []
for row_col, col_col in joint_specs:
    tmp = eval_joint_group(df, row_col, col_col)
    if len(tmp) > 0:
        tmp["map_name"] = f"{row_col}__x__{col_col}"
        all_joint.append(tmp)

if len(all_joint) == 0:
    raise ValueError("No valid joint groups found. Check bucket columns and MIN_GROUP_N.")

joint_df = pd.concat(all_joint, axis=0, ignore_index=True)
joint_df.to_csv(OUT_DIR / "joint_gain_maps.csv", index=False)
print("Saved joint gain maps.")

# =========================
# 4) Rank strongest positive cells
# =========================
print("[4/6] Ranking strongest gain cells...")

rank_auc = joint_df.sort_values(["delta_auc", "delta_auprc"], ascending=[False, False]).copy()
rank_auprc = joint_df.sort_values(["delta_auprc", "delta_auc"], ascending=[False, False]).copy()

rank_auc.head(30).to_csv(OUT_DIR / "top_joint_cells_by_delta_auc.csv", index=False)
rank_auprc.head(30).to_csv(OUT_DIR / "top_joint_cells_by_delta_auprc.csv", index=False)
print("Saved ranked top cells.")

# =========================
# 5) Write summary
# =========================
print("[5/6] Writing summary...")

summary = {
    "experiment": "33_topology_profile_joint_map",
    "real_data_policy": {
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
    "joint_maps_evaluated": [f"{a}__x__{b}" for a, b in joint_specs],
    "min_group_n": MIN_GROUP_N,
    "outputs": {
        "merged_topology_profile_features_csv": str(OUT_DIR / "merged_topology_profile_features.csv"),
        "joint_gain_maps_csv": str(OUT_DIR / "joint_gain_maps.csv"),
        "top_joint_cells_by_delta_auc_csv": str(OUT_DIR / "top_joint_cells_by_delta_auc.csv"),
        "top_joint_cells_by_delta_auprc_csv": str(OUT_DIR / "top_joint_cells_by_delta_auprc.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# =========================
# 6) Done
# =========================
print("[6/6] Done.")
print(f"Results saved to: {OUT_DIR}")

