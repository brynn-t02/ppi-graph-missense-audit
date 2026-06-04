#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 29B:
Real-vs-rewired stratification effect quantification (REAL DATA ONLY)

Goal:
- Quantify how key stratification effects differ between the real STRING graph
  and degree-preserving rewired graphs using outputs from Exp 29A.

Focus:
1) mid k-core gain: real vs rewired
2) neighbor-risk stratification key effects: real vs rewired

This experiment reuses Exp 29A output tables only.
No fake data, no simulated labels, no placeholder predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
REAL_KCORE = Path("/root/data/processed/graph_rewiring_control_exp29A/real_graph_kcore_stratified_metrics.csv")
REWIRED_KCORE = Path("/root/data/processed/graph_rewiring_control_exp29A/rewired_graph_kcore_stratified_metrics.csv")

REAL_RISK = Path("/root/data/processed/graph_rewiring_control_exp29A/real_graph_neighbor_risk_stratified_metrics.csv")
REWIRED_RISK = Path("/root/data/processed/graph_rewiring_control_exp29A/rewired_graph_neighbor_risk_stratified_metrics.csv")

OUT_DIR = Path("/root/data/processed/graph_rewiring_control_quant_exp29B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def empirical_p(real_val, rewired_vals, direction="greater"):
    """
    Empirical p-value with small-sample correction.
    direction:
      - "greater": p = P(rewired >= real)
      - "less":    p = P(rewired <= real)
    """
    rewired_vals = np.asarray(rewired_vals, dtype=float)
    n = len(rewired_vals)

    if direction == "greater":
        count = np.sum(rewired_vals >= real_val)
    elif direction == "less":
        count = np.sum(rewired_vals <= real_val)
    else:
        raise ValueError("direction must be 'greater' or 'less'")

    return float((count + 1) / (n + 1))

def summarize_real_vs_rewired(real_val, rewired_vals, direction):
    rewired_vals = np.asarray(rewired_vals, dtype=float)
    mean = float(np.mean(rewired_vals))
    sd = float(np.std(rewired_vals, ddof=1)) if len(rewired_vals) > 1 else 0.0

    if sd > 0:
        z_like = float((real_val - mean) / sd)
    else:
        z_like = np.nan

    p_emp = empirical_p(real_val, rewired_vals, direction=direction)

    return {
        "real_value": float(real_val),
        "rewired_mean": mean,
        "rewired_sd": sd,
        "real_minus_rewired_mean": float(real_val - mean),
        "z_like_diff": z_like,
        "empirical_p": p_emp,
        "n_rewired_replicates": int(len(rewired_vals))
    }

# =========================
# 1) Load tables
# =========================
print("[1/4] Loading Exp 29A outputs...")
for p in [REAL_KCORE, REWIRED_KCORE, REAL_RISK, REWIRED_RISK]:
    assert_exists(p)

real_k = pd.read_csv(REAL_KCORE)
rew_k = pd.read_csv(REWIRED_KCORE)
real_r = pd.read_csv(REAL_RISK)
rew_r = pd.read_csv(REWIRED_RISK)

require_cols(real_k, ["group_label", "delta_auc", "delta_auprc"], "real_kcore")
require_cols(rew_k, ["group_label", "delta_auc", "delta_auprc", "replicate_id"], "rewired_kcore")
require_cols(real_r, ["group_label", "delta_auc", "delta_auprc"], "real_risk")
require_cols(rew_r, ["group_label", "delta_auc", "delta_auprc", "replicate_id"], "rewired_risk")

# =========================
# 2) mid k-core quantification
# =========================
print("[2/4] Quantifying mid k-core effect...")

real_mid_k = real_k[real_k["group_label"] == "kcore_mid"].copy()
rew_mid_k = rew_k[rew_k["group_label"] == "kcore_mid"].copy()

if len(real_mid_k) != 1:
    raise ValueError("Expected exactly one real kcore_mid row.")
if len(rew_mid_k) == 0:
    raise ValueError("No rewired kcore_mid rows found.")

mid_k_auc = summarize_real_vs_rewired(
    real_val=real_mid_k["delta_auc"].iloc[0],
    rewired_vals=rew_mid_k["delta_auc"].values,
    direction="greater"
)
mid_k_auprc = summarize_real_vs_rewired(
    real_val=real_mid_k["delta_auprc"].iloc[0],
    rewired_vals=rew_mid_k["delta_auprc"].values,
    direction="greater"
)

mid_k_df = pd.DataFrame([
    {"effect": "mid_kcore_delta_auc", **mid_k_auc},
    {"effect": "mid_kcore_delta_auprc", **mid_k_auprc},
])
mid_k_df.to_csv(OUT_DIR / "mid_kcore_real_vs_rewired_quantification.csv", index=False)

# =========================
# 3) neighbor-risk quantification
# =========================
print("[3/4] Quantifying neighbor-risk effects...")

# Focus on risk_low because real graph showed strongest unique lift there
real_risk_low = real_r[real_r["group_label"] == "risk_low"].copy()
rew_risk_low = rew_r[rew_r["group_label"] == "risk_low"].copy()

if len(real_risk_low) != 1:
    raise ValueError("Expected exactly one real risk_low row.")
if len(rew_risk_low) == 0:
    raise ValueError("No rewired risk_low rows found.")

risk_low_auc = summarize_real_vs_rewired(
    real_val=real_risk_low["delta_auc"].iloc[0],
    rewired_vals=rew_risk_low["delta_auc"].values,
    direction="greater"
)
risk_low_auprc = summarize_real_vs_rewired(
    real_val=real_risk_low["delta_auprc"].iloc[0],
    rewired_vals=rew_risk_low["delta_auprc"].values,
    direction="greater"
)

risk_low_df = pd.DataFrame([
    {"effect": "risk_low_delta_auc", **risk_low_auc},
    {"effect": "risk_low_delta_auprc", **risk_low_auprc},
])
risk_low_df.to_csv(OUT_DIR / "risk_low_real_vs_rewired_quantification.csv", index=False)

# Also quantify ranking instability across rewired replicates
ranking_rows = []
for rid, sub in rew_r.groupby("replicate_id"):
    tmp = sub[["group_label", "delta_auc", "delta_auprc"]].copy()

    auc_rank = tmp.sort_values("delta_auc", ascending=False)["group_label"].tolist()
    auprc_rank = tmp.sort_values("delta_auprc", ascending=False)["group_label"].tolist()

    ranking_rows.append({
        "replicate_id": int(rid),
        "auc_rank_order": " > ".join(auc_rank),
        "auprc_rank_order": " > ".join(auprc_rank),
    })

ranking_df = pd.DataFrame(ranking_rows)
ranking_df.to_csv(OUT_DIR / "rewired_neighbor_risk_rank_orders.csv", index=False)

# =========================
# 4) Write summary
# =========================
print("[4/4] Writing summary...")

summary = {
    "experiment": "29B_real_vs_rewired_stratification_effect_quantification",
    "real_data_policy": {
        "reuses_real_output_from_29a_real_kcore": str(REAL_KCORE),
        "reuses_real_output_from_29a_rewired_kcore": str(REWIRED_KCORE),
        "reuses_real_output_from_29a_real_risk": str(REAL_RISK),
        "reuses_real_output_from_29a_rewired_risk": str(REWIRED_RISK),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "does_not_recompute_graph_or_predictions": True
    },
    "focus_effects": [
        "mid_kcore_delta_auc",
        "mid_kcore_delta_auprc",
        "risk_low_delta_auc",
        "risk_low_delta_auprc"
    ],
    "outputs": {
        "mid_kcore_quant_csv": str(OUT_DIR / "mid_kcore_real_vs_rewired_quantification.csv"),
        "risk_low_quant_csv": str(OUT_DIR / "risk_low_real_vs_rewired_quantification.csv"),
        "rewired_neighbor_risk_rank_orders_csv": str(OUT_DIR / "rewired_neighbor_risk_rank_orders.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
