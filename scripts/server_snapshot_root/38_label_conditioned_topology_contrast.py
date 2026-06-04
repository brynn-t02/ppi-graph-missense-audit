#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 38:
Label-conditioned topology contrast for corrected samples.

Goal:
Test whether FN_rescued / FP_suppressed topology profiles are specific to
graph correction mechanisms, rather than simply reflecting pathogenic-vs-benign
label distribution differences.

Core comparisons:
1. FN_rescued vs other pathogenic variants
2. FP_suppressed vs other benign variants

REAL DATA ONLY:
- Reuses Exp35 merged corrected-label + topology profile table.
- Does not recompute graph features.
- Does not use fake labels.
- Does not use simulated predictions.
- Does not use placeholder data.

Input:
/root/data/processed/corrected_sample_profile_exp35/test_variants_with_corrected_labels_and_topology_profile.csv

Output:
/root/data/processed/label_conditioned_topology_contrast_exp38/
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

OUT_DIR = Path("/root/data/processed/label_conditioned_topology_contrast_exp38")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
N_PERM = 1000
SEED = 42

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

    Input:
        pvals: array-like p-values

    Output:
        adjusted q-values in original order
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

    x: subset values
    y: parent-complement values
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

    # Small-sample correction
    J = 1 - 3 / (4 * (n1 + n2) - 9)

    return float(J * d)

def permutation_pvalue(parent_values, subset_indicator, observed_g, n_perm=1000, seed=42):
    """
    Within-label permutation test.

    H0:
    The corrected subset is a random subset of the same size within its parent label group.

    parent_values:
        numeric feature values for all samples in the parent label group after NA removal

    subset_indicator:
        boolean array of same length as parent_values
        True = observed corrected subset
        False = parent complement

    observed_g:
        observed Hedges' g between subset and parent complement

    Returns:
        two-sided empirical permutation p-value
    """
    parent_values = np.asarray(parent_values, dtype=float)
    subset_indicator = np.asarray(subset_indicator, dtype=bool)

    if len(parent_values) != len(subset_indicator):
        raise ValueError("parent_values and subset_indicator length mismatch.")

    n_total = len(parent_values)
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

        x = parent_values[perm_mask]
        y = parent_values[~perm_mask]

        g = hedges_g(x, y)

        if not np.isnan(g):
            null_g.append(g)

    null_g = np.asarray(null_g, dtype=float)

    if len(null_g) == 0:
        return np.nan

    p = (np.sum(np.abs(null_g) >= abs(observed_g)) + 1) / (len(null_g) + 1)

    return float(p)

def analyze_condition(df, condition_name, subset_mask, parent_mask, seed_offset=0):
    """
    Compare:
        subset vs parent label complement

    Example:
        FN_rescued vs other pathogenic
        FP_suppressed vs other benign
    """
    rows = []

    subset_mask = pd.Series(subset_mask, index=df.index).astype(bool)
    parent_mask = pd.Series(parent_mask, index=df.index).astype(bool)

    # hard containment check
    if (subset_mask & (~parent_mask)).sum() != 0:
        raise ValueError(f"{condition_name}: subset is not contained in parent group.")

    for i, feat in enumerate(FEATURES):
        # Extract parent group feature and subset indicator together
        parent_df = pd.DataFrame({
            "value": pd.to_numeric(df.loc[parent_mask, feat], errors="coerce"),
            "is_subset": subset_mask.loc[parent_mask].values
        })

        # Drop NA values for this feature
        parent_df = parent_df.dropna(subset=["value"]).reset_index(drop=True)

        values = parent_df["value"].values.astype(float)
        is_subset = parent_df["is_subset"].values.astype(bool)

        subset_values = values[is_subset]
        comp_values = values[~is_subset]

        g = hedges_g(subset_values, comp_values)

        p_perm = permutation_pvalue(
            parent_values=values,
            subset_indicator=is_subset,
            observed_g=g,
            n_perm=N_PERM,
            seed=SEED + seed_offset + i
        )

        rows.append({
            "condition": condition_name,
            "feature": feat,
            "n_subset": int(len(subset_values)),
            "n_parent_complement": int(len(comp_values)),
            "mean_subset": float(np.mean(subset_values)) if len(subset_values) > 0 else np.nan,
            "mean_parent_complement": float(np.mean(comp_values)) if len(comp_values) > 0 else np.nan,
            "diff_subset_minus_complement": (
                float(np.mean(subset_values) - np.mean(comp_values))
                if len(subset_values) > 0 and len(comp_values) > 0
                else np.nan
            ),
            "hedges_g": g,
            "perm_p": p_perm
        })

    res = pd.DataFrame(rows)

    # FDR within this condition
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
    print("[1/5] Loading Exp35 merged corrected-profile table...")

    if not P35.exists():
        raise FileNotFoundError(f"Missing required file: {P35}")

    df = pd.read_csv(P35)

    required_cols = ["Label", "esm_vs_gcn_transition"] + FEATURES
    require_cols(df, required_cols, "Exp35 table")

    print(f"Rows loaded: {len(df)}")

    print("[2/5] Defining label-conditioned groups...")

    # Ensure numeric label
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

    label_pathogenic = df["Label"] == 1
    label_benign = df["Label"] == 0

    fn_rescued = df["esm_vs_gcn_transition"] == "FN_rescued"
    fp_suppressed = df["esm_vs_gcn_transition"] == "FP_suppressed"

    # Hard checks
    n_bad_fn = int((fn_rescued & (~label_pathogenic)).sum())
    n_bad_fp = int((fp_suppressed & (~label_benign)).sum())

    if n_bad_fn != 0:
        raise ValueError(f"FN_rescued contains non-pathogenic samples: {n_bad_fn}")

    if n_bad_fp != 0:
        raise ValueError(f"FP_suppressed contains non-benign samples: {n_bad_fp}")

    group_counts = {
        "all_pathogenic": int(label_pathogenic.sum()),
        "all_benign": int(label_benign.sum()),
        "FN_rescued": int(fn_rescued.sum()),
        "FP_suppressed": int(fp_suppressed.sum()),
        "other_pathogenic": int((label_pathogenic & (~fn_rescued)).sum()),
        "other_benign": int((label_benign & (~fp_suppressed)).sum()),
    }

    print("Group counts:")
    for k, v in group_counts.items():
        print(f"  {k}: {v}")

    print("[3/5] Running FN_rescued vs other pathogenic contrast...")

    res_fn = analyze_condition(
        df=df,
        condition_name="FN_rescued_vs_other_pathogenic",
        subset_mask=fn_rescued,
        parent_mask=label_pathogenic,
        seed_offset=0
    )

    print("[4/5] Running FP_suppressed vs other benign contrast...")

    res_fp = analyze_condition(
        df=df,
        condition_name="FP_suppressed_vs_other_benign",
        subset_mask=fp_suppressed,
        parent_mask=label_benign,
        seed_offset=10000
    )

    res = pd.concat([res_fn, res_fp], axis=0, ignore_index=True)

    out_csv = OUT_DIR / "label_conditioned_topology_contrast_results.csv"
    res.to_csv(out_csv, index=False)

    # Features significant in both conditions
    sig = (
        res[res["significant_fdr_0_05"]]
        .groupby("feature")["condition"]
        .nunique()
    )

    sig_both = sorted(sig[sig == 2].index.tolist())

    sig_json = OUT_DIR / "features_significant_in_both_conditions.json"
    with open(sig_json, "w") as f:
        json.dump(
            {
                "features_significant_in_both_conditions": sig_both
            },
            f,
            indent=2
        )

    print("[5/5] Writing summary.json...")

    summary = {
        "experiment": "38_label_conditioned_topology_contrast",
        "real_data_policy": {
            "reuses_real_output_from_35": str(P35),
            "no_fake_data": True,
            "no_simulated_labels": True,
            "no_placeholder_predictions": True,
            "does_not_recompute_graph_features": True,
            "uses_existing_topology_profile_from_exp35": True
        },
        "purpose": (
            "Test whether corrected-sample topology profiles remain specific "
            "after conditioning on pathogenic/benign label."
        ),
        "group_counts": group_counts,
        "features_tested": FEATURES,
        "statistical_test": {
            "effect_size": "Hedges_g_subset_vs_parent_label_complement",
            "permutation_test": f"{N_PERM} within-label permutations",
            "multiple_testing": "Benjamini-Hochberg FDR within each condition",
            "fdr_implementation": "custom_bh_fdr_no_statsmodels_dependency"
        },
        "outputs": {
            "contrast_results_csv": str(out_csv),
            "features_significant_in_both_conditions_json": str(sig_json)
        }
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Features significant in both conditions: {sig_both}")

if __name__ == "__main__":
    main()
