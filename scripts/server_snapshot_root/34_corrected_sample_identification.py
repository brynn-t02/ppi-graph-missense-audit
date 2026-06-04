#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 34:
Corrected sample identification (REAL DATA ONLY)

Goal:
- Identify which test variants are corrected or degraded by graph-enhanced models.
- Two comparisons:
    1) raw ESM vs GCN enhanced
    2) CADD+AM vs GCN+CADD+AM

Definitions use fixed probability threshold = 0.5.

No fake data, no simulated labels, no placeholder predictions.
Hard-fails on missing files or length mismatch.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
TEST_TABLE = Path("/root/data/processed/clinvar_test_main.csv")

PROB_ESM = Path("/root/data/processed/variant_mlp_comparison/esm_raw/test_probs.npy")
PROB_GCN = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy")

PROB_CADD_AM = Path("/root/data/processed/triple_fusion/cadd_plus_am/test_probs.npy")
PROB_FULL = Path("/root/data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy")

OUT_DIR = Path("/root/data/processed/corrected_sample_identification_exp34")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
PRED_THRESHOLD = 0.5

# =========================
# Helpers
# =========================
def assert_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

def classify_pair(y_true, pred_a, pred_b):
    """
    Compare model A vs model B under binary decisions.
    Returns DataFrame columns:
      a_correct, b_correct, pair_status, error_transition
    """
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    pair_status = np.full(len(y_true), "", dtype=object)
    pair_status[(~a_correct) & ( b_correct)] = "improved"
    pair_status[( a_correct) & (~b_correct)] = "degraded"
    pair_status[( a_correct) & ( b_correct)] = "both_correct"
    pair_status[(~a_correct) & (~b_correct)] = "both_wrong"

    error_transition = np.full(len(y_true), "none", dtype=object)

    # improved cases
    error_transition[(pred_a == 1) & (y_true == 0) & (pred_b == 0)] = "FP_suppressed"
    error_transition[(pred_a == 0) & (y_true == 1) & (pred_b == 1)] = "FN_rescued"

    # degraded cases
    error_transition[(pred_a == 0) & (y_true == 0) & (pred_b == 1)] = "TN_to_FP"
    error_transition[(pred_a == 1) & (y_true == 1) & (pred_b == 0)] = "TP_to_FN"

    return pd.DataFrame({
        "a_correct": a_correct.astype(int),
        "b_correct": b_correct.astype(int),
        "pair_status": pair_status,
        "error_transition": error_transition
    })

def summarize_pair(df, pair_name):
    out = {
        "pair_name": pair_name,
        "n": int(len(df))
    }
    for key in ["improved", "degraded", "both_correct", "both_wrong"]:
        out[f"count_{key}"] = int((df["pair_status"] == key).sum())
        out[f"rate_{key}"] = float((df["pair_status"] == key).mean())

    for key in ["FP_suppressed", "FN_rescued", "TN_to_FP", "TP_to_FN"]:
        out[f"count_{key}"] = int((df["error_transition"] == key).sum())
        out[f"rate_{key}"] = float((df["error_transition"] == key).mean())

    return out

# =========================
# 1) Load real test table
# =========================
print("[1/5] Loading real test table...")
assert_exists(TEST_TABLE)
df = pd.read_csv(TEST_TABLE)

required_cols = ["Label", "STRING_protein_id", "split"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column missing from test table: {col}")

df = df[df["split"] == "test"].copy().reset_index(drop=True)
y = df["Label"].astype(int).values

print(f"Test rows: {len(df)}")
print(f"Positive rate: {np.mean(y):.4f}")

# =========================
# 2) Load real prediction probabilities
# =========================
print("[2/5] Loading real prediction probabilities...")
for p in [PROB_ESM, PROB_GCN, PROB_CADD_AM, PROB_FULL]:
    assert_exists(p)

p_esm = np.load(str(PROB_ESM)).astype(float)
p_gcn = np.load(str(PROB_GCN)).astype(float)
p_cadd_am = np.load(str(PROB_CADD_AM)).astype(float)
p_full = np.load(str(PROB_FULL)).astype(float)

for name, arr in {
    "ESM": p_esm,
    "GCN": p_gcn,
    "CADD+AM": p_cadd_am,
    "GCN+CADD+AM": p_full
}.items():
    if len(arr) != len(df):
        raise ValueError(f"Length mismatch: test table={len(df)}, {name} probs={len(arr)}")

print("All probability arrays aligned.")

# =========================
# 3) Convert to binary predictions
# =========================
print("[3/5] Converting probabilities to binary predictions...")
pred_esm = (p_esm >= PRED_THRESHOLD).astype(int)
pred_gcn = (p_gcn >= PRED_THRESHOLD).astype(int)
pred_cadd_am = (p_cadd_am >= PRED_THRESHOLD).astype(int)
pred_full = (p_full >= PRED_THRESHOLD).astype(int)

# =========================
# 4) Build corrected-sample tables
# =========================
print("[4/5] Building corrected/degraded sample tables...")

df_out = df.copy()
df_out["esm_prob"] = p_esm
df_out["gcn_prob"] = p_gcn
df_out["cadd_am_prob"] = p_cadd_am
df_out["gcn_cadd_am_prob"] = p_full

df_out["esm_pred"] = pred_esm
df_out["gcn_pred"] = pred_gcn
df_out["cadd_am_pred"] = pred_cadd_am
df_out["gcn_cadd_am_pred"] = pred_full

# Pair 1: ESM vs GCN
pair_esm_gcn = classify_pair(y, pred_esm, pred_gcn)
pair_esm_gcn = pair_esm_gcn.rename(columns={
    "a_correct": "esm_correct",
    "b_correct": "gcn_correct",
    "pair_status": "esm_vs_gcn_status",
    "error_transition": "esm_vs_gcn_transition"
})
df_out = pd.concat([df_out, pair_esm_gcn], axis=1)

# Pair 2: CADD+AM vs FULL
pair_base_full = classify_pair(y, pred_cadd_am, pred_full)
pair_base_full = pair_base_full.rename(columns={
    "a_correct": "cadd_am_correct",
    "b_correct": "gcn_cadd_am_correct",
    "pair_status": "cadd_am_vs_full_status",
    "error_transition": "cadd_am_vs_full_transition"
})
df_out = pd.concat([df_out, pair_base_full], axis=1)

df_out.to_csv(OUT_DIR / "test_variants_with_corrected_sample_labels.csv", index=False)

# Subsets
df_out[df_out["esm_vs_gcn_status"] == "improved"].to_csv(
    OUT_DIR / "esm_to_gcn_improved_samples.csv", index=False
)
df_out[df_out["esm_vs_gcn_status"] == "degraded"].to_csv(
    OUT_DIR / "esm_to_gcn_degraded_samples.csv", index=False
)
df_out[df_out["cadd_am_vs_full_status"] == "improved"].to_csv(
    OUT_DIR / "cadd_am_to_full_improved_samples.csv", index=False
)
df_out[df_out["cadd_am_vs_full_status"] == "degraded"].to_csv(
    OUT_DIR / "cadd_am_to_full_degraded_samples.csv", index=False
)

# =========================
# 5) Write summary
# =========================
print("[5/5] Writing summary...")

summary = {
    "experiment": "34_corrected_sample_identification",
    "real_data_policy": {
        "uses_real_test_table": str(TEST_TABLE),
        "uses_real_esm_predictions": str(PROB_ESM),
        "uses_real_gcn_predictions": str(PROB_GCN),
        "uses_real_cadd_am_predictions": str(PROB_CADD_AM),
        "uses_real_full_predictions": str(PROB_FULL),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True,
        "hard_fail_on_length_mismatch": True
    },
    "prediction_threshold": float(PRED_THRESHOLD),
    "n_test": int(len(df)),
    "positive_rate": float(np.mean(y)),
    "esm_vs_gcn": summarize_pair(
        df_out[["esm_vs_gcn_status", "esm_vs_gcn_transition"]].rename(columns={
            "esm_vs_gcn_status": "pair_status",
            "esm_vs_gcn_transition": "error_transition"
        }),
        "esm_vs_gcn"
    ),
    "cadd_am_vs_full": summarize_pair(
        df_out[["cadd_am_vs_full_status", "cadd_am_vs_full_transition"]].rename(columns={
            "cadd_am_vs_full_status": "pair_status",
            "cadd_am_vs_full_transition": "error_transition"
        }),
        "cadd_am_vs_full"
    ),
    "outputs": {
        "all_test_variants_csv": str(OUT_DIR / "test_variants_with_corrected_sample_labels.csv"),
        "esm_to_gcn_improved_csv": str(OUT_DIR / "esm_to_gcn_improved_samples.csv"),
        "esm_to_gcn_degraded_csv": str(OUT_DIR / "esm_to_gcn_degraded_samples.csv"),
        "cadd_am_to_full_improved_csv": str(OUT_DIR / "cadd_am_to_full_improved_samples.csv"),
        "cadd_am_to_full_degraded_csv": str(OUT_DIR / "cadd_am_to_full_degraded_samples.csv")
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(json.dumps(summary["esm_vs_gcn"], indent=2))
print(json.dumps(summary["cadd_am_vs_full"], indent=2))
print(f"Results saved to: {OUT_DIR}")
