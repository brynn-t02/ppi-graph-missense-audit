#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 49:
Paired bootstrap for real PPI-GCN fusion margins over fusion controls.

Purpose:
Quantify whether the small topology-specific margin in Fig. 2C is supported
under variant-level and protein-clustered uncertainty.

Comparisons:
- real PPI-GCN fusion vs raw ESM fusion
- real PPI-GCN fusion vs edge-ablated PPI-GCN fusion
- real PPI-GCN fusion vs topology-permuted PPI-GCN fusion
- real PPI-GCN fusion vs architecture-matched CADD+AlphaMissense, if Exp48 exists

REAL DATA POLICY:
- Uses retained test probabilities from Exp46/47/48 and original ClinVar test labels.
- No fake labels and no simulated predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/root")
TEST_PATH = ROOT / "data/processed/clinvar_test_main.csv"

EXP46 = ROOT / "data/processed/cadd_am_raw_esm_baseline_exp46/original"
EXP47 = ROOT / "data/processed/cadd_am_graph_control_fusions_exp47"
EXP48 = ROOT / "data/processed/cadd_am_architecture_matched_exp48/original"

OUT_DIR = ROOT / "data/processed/fusion_control_margin_bootstrap_exp49"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
BOOT_SEED = 42


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def load_probs_from_dirs(dirs):
    probs = []
    used = []
    for d in dirs:
        p = Path(d) / "test_probs.npy"
        if p.exists():
            probs.append(np.load(p))
            used.append(str(p))
    if not probs:
        raise FileNotFoundError("No test_probs.npy found for requested model.")
    arr = np.vstack([np.asarray(x).reshape(1, -1) for x in probs])
    return arr.mean(axis=0), used


def seed_dirs(root, model):
    return [Path(root) / model / f"seed_{s}" for s in [42, 43, 44]]


def topology_perm_dirs():
    root = EXP47 / "cadd_am_plus_topology_permuted_ppi_gcn"
    return sorted(root.glob("graphperm_*_seed_*"))


def load_model_predictions():
    models = {}
    sources = {}

    specs = {
        "real_ppi_gcn_fusion": seed_dirs(EXP47, "cadd_am_plus_real_ppi_gcn"),
        "raw_esm_fusion": seed_dirs(EXP46, "cadd_am_plus_raw_esm"),
        "edge_ablated_ppi_gcn_fusion": seed_dirs(EXP47, "cadd_am_plus_edge_ablated_ppi_gcn"),
        "topology_permuted_ppi_gcn_fusion": topology_perm_dirs(),
    }

    archmatched_dirs = seed_dirs(EXP48, "cadd_plus_am_archmatched_512_128")
    if any((Path(d) / "test_probs.npy").exists() for d in archmatched_dirs):
        specs["cadd_am_archmatched_512_128"] = archmatched_dirs

    score_only_dirs = seed_dirs(EXP47, "cadd_plus_am")
    if any((Path(d) / "test_probs.npy").exists() for d in score_only_dirs):
        specs["cadd_am_score_only_reference"] = score_only_dirs

    for name, dirs in specs.items():
        models[name], sources[name] = load_probs_from_dirs(dirs)

    return models, sources


def bootstrap_delta(y, a, b, clusters, clustered=False):
    rng = np.random.default_rng(BOOT_SEED)
    y = np.asarray(y)
    a = np.asarray(a)
    b = np.asarray(b)
    clusters = np.asarray(clusters)
    n = len(y)
    out_auc = []
    out_auprc = []

    if clustered:
        unique_clusters = np.unique(clusters)
        cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
        for _ in range(N_BOOT):
            sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
            if len(np.unique(y[idx])) < 2:
                continue
            out_auc.append(safe_auc(y[idx], a[idx]) - safe_auc(y[idx], b[idx]))
            out_auprc.append(safe_auprc(y[idx], a[idx]) - safe_auprc(y[idx], b[idx]))
    else:
        for _ in range(N_BOOT):
            idx = rng.choice(n, size=n, replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            out_auc.append(safe_auc(y[idx], a[idx]) - safe_auc(y[idx], b[idx]))
            out_auprc.append(safe_auprc(y[idx], a[idx]) - safe_auprc(y[idx], b[idx]))

    return {
        "n_boot_used": int(len(out_auc)),
        "delta_auc_mean": float(np.mean(out_auc)),
        "delta_auc_ci95_low": float(np.percentile(out_auc, 2.5)),
        "delta_auc_ci95_high": float(np.percentile(out_auc, 97.5)),
        "delta_auprc_mean": float(np.mean(out_auprc)),
        "delta_auprc_ci95_low": float(np.percentile(out_auprc, 2.5)),
        "delta_auprc_ci95_high": float(np.percentile(out_auprc, 97.5)),
    }


def point_metrics(y, a, b):
    return {
        "model_a_auc": safe_auc(y, a),
        "model_b_auc": safe_auc(y, b),
        "delta_auc": safe_auc(y, a) - safe_auc(y, b),
        "model_a_auprc": safe_auprc(y, a),
        "model_b_auprc": safe_auprc(y, b),
        "delta_auprc": safe_auprc(y, a) - safe_auprc(y, b),
    }


def main():
    test = pd.read_csv(TEST_PATH, low_memory=False)
    y = test["Label"].astype(int).to_numpy()
    clusters = test["STRING_protein_id"].astype(str).to_numpy()
    models, sources = load_model_predictions()

    real = models["real_ppi_gcn_fusion"]
    comparisons = [
        ("real_vs_raw_esm_fusion", "raw_esm_fusion"),
        ("real_vs_edge_ablated_fusion", "edge_ablated_ppi_gcn_fusion"),
        ("real_vs_topology_permuted_fusion", "topology_permuted_ppi_gcn_fusion"),
    ]
    if "cadd_am_archmatched_512_128" in models:
        comparisons.append(("real_vs_archmatched_cadd_am", "cadd_am_archmatched_512_128"))
    if "cadd_am_score_only_reference" in models:
        comparisons.append(("real_vs_score_only_cadd_am_reference", "cadd_am_score_only_reference"))

    rows = []
    nested = {}
    for comparison, model_b in comparisons:
        b = models[model_b]
        point = point_metrics(y, real, b)
        variant = bootstrap_delta(y, real, b, clusters, clustered=False)
        clustered = bootstrap_delta(y, real, b, clusters, clustered=True)

        for mode, stats in [("variant_level", variant), ("protein_clustered", clustered)]:
            row = {
                "comparison": comparison,
                "model_a": "real_ppi_gcn_fusion",
                "model_b": model_b,
                "bootstrap_mode": mode,
                **point,
                **stats,
            }
            rows.append(row)

        nested[comparison] = {
            "point": point,
            "variant_level_bootstrap": variant,
            "protein_clustered_bootstrap": clustered,
        }

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "fusion_control_margin_bootstrap_summary.csv", index=False)
    with open(OUT_DIR / "fusion_control_margin_bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "49_fusion_control_margin_bootstrap",
                "n_test": int(len(y)),
                "n_protein_clusters": int(len(np.unique(clusters))),
                "n_boot": N_BOOT,
                "bootstrap_seed": BOOT_SEED,
                "sources": sources,
                "comparisons": nested,
            },
            f,
            indent=2,
        )

    print(summary.to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
