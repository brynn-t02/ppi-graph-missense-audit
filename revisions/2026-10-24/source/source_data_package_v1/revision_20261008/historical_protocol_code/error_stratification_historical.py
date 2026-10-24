#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 40:
Difficulty-matched correction contrast.

Goal:
Test whether graph-corrected samples differ from persistently incorrect samples
within the same baseline-error class.

This addresses the concern that Exp38 controls label but not baseline difficulty.

Comparisons:
1. FN_rescued vs persistent_FN
   - Both are pathogenic variants misclassified by ESM.
   - FN_rescued: ESM wrong, GCN correct.
   - persistent_FN: ESM wrong, GCN wrong.

2. FP_suppressed vs persistent_FP
   - Both are benign variants misclassified by ESM.
   - FP_suppressed: ESM wrong, GCN correct.
   - persistent_FP: ESM wrong, GCN wrong.

REAL DATA ONLY:
- Reuses Exp35 corrected-label + topology profile table.
- Loads real saved ESM/GCN test probabilities from Exp03F.
- Does not recompute graph features.
- Does not use fake labels.
- Does not use simulated predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
P35 = Path(
    "/root/data/processed/corrected_sample_profile_exp35/"
    "test_variants_with_corrected_labels_and_topology_profile.csv"
)

ESM_PROBS = Path("/root/data/processed/variant_mlp_comparison/esm_raw/test_probs.npy")
GCN_PROBS = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy")

OUT_DIR = Path("/root/data/processed/difficulty_matched_correction_contrast_exp40")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
N_PERM = 1000
SEED = 42
THRESHOLD = 0.5

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
# Helper functions
# =========================
def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction without statsmodels.
    Returns adjusted q-values in original order.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(pvals)
    ranked = pvals[order]

    adj_ranked = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj_ranked[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj_ranked, 0, 1)
    return out

def hedges_g(x, y):
    """
    Hedges' g effect size for x - y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n1 = len(x)
    n2 = len(y)

    if n1 < 2 or n2 < 2:
        return np.nan

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    pooled_num = (n1 - 1) * var_x + (n2 - 1) * var_y
    pooled_den = n1 + n2 - 2

    if pooled_den <= 0:
        return np.nan

    pooled = np.sqrt(pooled_num / pooled_den)

    if pooled == 0 or np.isnan(pooled):
        return np.nan

    d = (mean_x - mean_y) / pooled
    J = 1 - 3 / (4 * (n1 + n2) - 9)

    return float(J * d)

def permutation_pvalue(values_all, subset_indicator, observed_g, n_perm=1000, seed=42):
    """
    Within-difficulty-class permutation test.

    H0:
    The corrected subset is a random subset of the same size among
    ESM baseline-error samples.
    """
    values_all = np.asarray(values_all, dtype=float)
    subset_indicator = np.asarray(subset_indicator, dtype=bool)

    if len(values_all) != len(subset_indicator):
        raise ValueError("values_all and subset_indicator length mismatch.")

    n_total = len(values_all)
    n_subset = int(subset_indicator.sum())

    if n_subset < 2 or n_subset > n_total - 2:
        return np.nan

    if np.isnan(observed_g):
        return np.nan

    rng = np.random.default_rng(seed)
    idx_all = np.arange(n_total)

    null_g = []

    for _ in range(n_perm):
        sampled_idx = rng.choice(idx_all, size=n_subset, replace=False)

        perm_mask = np.zeros(n_total, dtype=bool)
        perm_mask[sampled_idx] = True

        x = values_all[perm_mask]
        y = values_all[~perm_mask]

        g = hedges_g(x, y)
        if not np.isnan(g):
            null_g.append(g)

    null_g = np.asarray(null_g, dtype=float)

    if len(null_g) == 0:
        return np.nan

    p = (np.sum(np.abs(null_g) >= abs(observed_g)) + 1) / (len(null_g) + 1)
    return float(p)

def analyze_condition(df, condition_name, subset_mask, persistent_mask, seed_offset=0):
    """
    Compare corrected subset vs persistent-error group.
    """
    rows = []

    subset_mask = pd.Series(subset_mask, index=df.index).astype(bool)
    persistent_mask = pd.Series(persistent_mask, index=df.index).astype(bool)

    parent_mask = subset_mask | persistent_mask

    for i, feat in enumerate(FEATURES):
        parent_df = pd.DataFrame({
            "value": pd.to_numeric(df.loc[parent_mask, feat], errors="coerce"),
            "is_subset": subset_mask.loc[parent_mask].values
        })

        parent_df = parent_df.dropna(subset=["value"]).reset_index(drop=True)

        values = parent_df["value"].values.astype(float)
        is_subset = parent_df["is_subset"].values.astype(bool)

        subset_values = values[is_subset]
        persistent_values = values[~is_subset]

        g = hedges_g(subset_values, persistent_values)

        p_perm = permutation_pvalue(
            values_all=values,
            subset_indicator=is_subset,
            observed_g=g,
            n_perm=N_PERM,
            seed=SEED + seed_offset + i
        )

        rows.append({
            "condition": condition_name,
            "feature": feat,
            "n_corrected_subset": int(len(subset_values)),
            "n_persistent_error": int(len(persistent_values)),
            "mean_corrected_subset": float(np.mean(subset_values)) if len(subset_values) > 0 else np.nan,
            "mean_persistent_error": float(np.mean(persistent_values)) if len(persistent_values) > 0 else np.nan,
            "diff_corrected_minus_persistent": (
                float(np.mean(subset_values) - np.mean(persistent_values))
                if len(subset_values) > 0 and len(persistent_values) > 0
                else np.nan
            ),
            "hedges_g": g,
            "perm_p": p_perm
        })

    res = pd.DataFrame(rows)

    valid = res["perm_p"].notna().values
    qvals = np.full(len(res), np.nan, dtype=float)

    if valid.sum() > 0:
        qvals[valid] = bh_fdr(res.loc[valid, "perm_p"].values)

    res["fdr_bh"] = qvals
    res["significant_fdr_0_05"] = res["fdr_bh"] < 0.05

    return res

# =========================
# Main
# =========================
def main():
    print("[1/6] Loading Exp35 corrected-profile table...")

    if not P35.exists():
        raise FileNotFoundError(f"Missing required file: {P35}")
    if not ESM_PROBS.exists():
        raise FileNotFoundError(f"Missing required file: {ESM_PROBS}")
    if not GCN_PROBS.exists():
        raise FileNotFoundError(f"Missing required file: {GCN_PROBS}")

    df = pd.read_csv(P35)

    required_cols = ["Label"] + FEATURES
    require_cols(df, required_cols, "Exp35 table")

    print(f"Rows loaded: {len(df)}")

    print("[2/6] Loading saved ESM/GCN test probabilities...")

    esm_prob = np.load(ESM_PROBS)
    gcn_prob = np.load(GCN_PROBS)

    print(f"ESM probs shape: {esm_prob.shape}")
    print(f"GCN probs shape: {gcn_prob.shape}")

    if len(esm_prob) != len(df):
        raise ValueError(f"ESM probs length {len(esm_prob)} != Exp35 rows {len(df)}")
    if len(gcn_prob) != len(df):
        raise ValueError(f"GCN probs length {len(gcn_prob)} != Exp35 rows {len(df)}")

    df["esm_prob_exp40"] = esm_prob
    df["gcn_prob_exp40"] = gcn_prob
    df["esm_pred_exp40"] = (df["esm_prob_exp40"] >= THRESHOLD).astype(int)
    df["gcn_pred_exp40"] = (df["gcn_prob_exp40"] >= THRESHOLD).astype(int)

    print("[3/6] Defining difficulty-matched groups...")

    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").astype(int)

    label_path = df["Label"] == 1
    label_benign = df["Label"] == 0

    # Pathogenic baseline-error samples:
    # ESM predicts benign (0)
    fn_rescued = label_path & (df["esm_pred_exp40"] == 0) & (df["gcn_pred_exp40"] == 1)
    persistent_fn = label_path & (df["esm_pred_exp40"] == 0) & (df["gcn_pred_exp40"] == 0)

    # Benign baseline-error samples:
    # ESM predicts pathogenic (1)
    fp_suppressed = label_benign & (df["esm_pred_exp40"] == 1) & (df["gcn_pred_exp40"] == 0)
    persistent_fp = label_benign & (df["esm_pred_exp40"] == 1) & (df["gcn_pred_exp40"] == 1)

    transition_check = {}
    if "esm_vs_gcn_transition" in df.columns:
        trans_fn = df["esm_vs_gcn_transition"] == "FN_rescued"
        trans_fp = df["esm_vs_gcn_transition"] == "FP_suppressed"

        transition_check = {
            "transition_col_present": True,
            "FN_rescued_by_transition": int(trans_fn.sum()),
            "FN_rescued_by_prediction_logic": int(fn_rescued.sum()),
            "FN_rescued_mismatch": int((trans_fn != fn_rescued).sum()),
            "FP_suppressed_by_transition": int(trans_fp.sum()),
            "FP_suppressed_by_prediction_logic": int(fp_suppressed.sum()),
            "FP_suppressed_mismatch": int((trans_fp != fp_suppressed).sum()),
        }
    else:
        transition_check = {
            "transition_col_present": False
        }

    group_counts = {
        "threshold": THRESHOLD,
        "all_pathogenic": int(label_path.sum()),
        "all_benign": int(label_benign.sum()),
        "FN_rescued": int(fn_rescued.sum()),
        "persistent_FN": int(persistent_fn.sum()),
        "FP_suppressed": int(fp_suppressed.sum()),
        "persistent_FP": int(persistent_fp.sum()),
        "ESM_baseline_error_pathogenic_total": int((fn_rescued | persistent_fn).sum()),
        "ESM_baseline_error_benign_total": int((fp_suppressed | persistent_fp).sum()),
    }

    print("Group counts:")
    for k, v in group_counts.items():
        print(f"  {k}: {v}")

    print("Transition consistency:")
    for k, v in transition_check.items():
        print(f"  {k}: {v}")

    if group_counts["persistent_FN"] < 5:
        print("[WARNING] persistent_FN very small; FN comparison may be unstable.")
    if group_counts["persistent_FP"] < 5:
        print("[WARNING] persistent_FP very small; FP comparison may be unstable.")

    print("[4/6] Running FN_rescued vs persistent_FN contrast...")

    res_fn = analyze_condition(
        df=df,
        condition_name="FN_rescued_vs_persistent_FN",
        subset_mask=fn_rescued,
        persistent_mask=persistent_fn,
        seed_offset=0
    )

    print("[5/6] Running FP_suppressed vs persistent_FP contrast...")

    res_fp = analyze_condition(
        df=df,
        condition_name="FP_suppressed_vs_persistent_FP",
        subset_mask=fp_suppressed,
        persistent_mask=persistent_fp,
        seed_offset=10000
    )

    res = pd.concat([res_fn, res_fp], axis=0, ignore_index=True)

    out_csv = OUT_DIR / "difficulty_matched_correction_contrast_results.csv"
    res.to_csv(out_csv, index=False)

    sig = (
        res[res["significant_fdr_0_05"]]
        .groupby("feature")["condition"]
        .nunique()
    )
    sig_both = sorted(sig[sig == 2].index.tolist())

    sig_json = OUT_DIR / "features_significant_in_both_difficulty_matched_conditions.json"
    with open(sig_json, "w") as f:
        json.dump(
            {
                "features_significant_in_both_difficulty_matched_conditions": sig_both
            },
            f,
            indent=2
        )

    print("[6/6] Writing summary.json...")

    summary = {
        "experiment": "40_difficulty_matched_correction_contrast",
        "real_data_policy": {
            "reuses_real_output_from_35": str(P35),
            "uses_real_saved_esm_test_probabilities": str(ESM_PROBS),
            "uses_real_saved_gcn_test_probabilities": str(GCN_PROBS),
            "no_fake_data": True,
            "no_simulated_labels": True,
            "no_placeholder_predictions": True,
            "does_not_recompute_graph_features": True,
            "uses_existing_topology_profile_from_exp35": True
        },
        "purpose": (
            "Test whether graph-corrected samples differ from persistently "
            "incorrect samples after matching on baseline ESM difficulty."
        ),
        "group_definitions": {
            "FN_rescued": "Label=1, ESM_pred=0, GCN_pred=1",
            "persistent_FN": "Label=1, ESM_pred=0, GCN_pred=0",
            "FP_suppressed": "Label=0, ESM_pred=1, GCN_pred=0",
            "persistent_FP": "Label=0, ESM_pred=1, GCN_pred=1"
        },
        "group_counts": group_counts,
        "transition_consistency_check": transition_check,
        "features_tested": FEATURES,
        "statistical_test": {
            "effect_size": "Hedges_g_corrected_subset_vs_persistent_error",
            "permutation_test": f"{N_PERM} within-difficulty-class permutations",
            "multiple_testing": "Benjamini-Hochberg FDR within each condition"
        },
        "features_significant_in_both_difficulty_matched_conditions": sig_both,
        "outputs": {
            "contrast_results_csv": str(out_csv),
            "features_significant_in_both_json": str(sig_json)
        }
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Features significant in both difficulty-matched conditions: {sig_both}")

if __name__ == "__main__":
    main()
