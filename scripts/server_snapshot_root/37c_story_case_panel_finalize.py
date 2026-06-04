#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 37C:
Finalize story case panel with complete display fields (REAL DATA ONLY)

Goal:
- Merge full descriptive fields from Exp 34 back into the compact story-oriented
  case panel selected in Exp 37B.
- Produce final display-ready case tables for manuscript / supplementary / slides.

Reuses:
- Exp 34 full corrected sample table
- Exp 37B compact case panel

No fake data, no simulated labels, no placeholder predictions.
"""

import json
from pathlib import Path

import pandas as pd

# =========================
# Paths
# =========================
P34 = Path("/root/data/processed/corrected_sample_identification_exp34/test_variants_with_corrected_sample_labels.csv")
P37B = Path("/root/data/processed/story_case_panel_exp37B/story_case_panel_compact.csv")

OUT_DIR = Path("/root/data/processed/story_case_panel_final_exp37C")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
KEY_COLS = ["Name", "STRING_protein_id", "Label"]

DISPLAY_COLS_FROM_34 = [
    "AlleleID",
    "Type",
    "Name",
    "HGVS_p",
    "GeneID",
    "GeneSymbol",
    "ClinicalSignificance",
    "ClinSigSimple",
    "ReviewStatus",
    "OriginSimple",
    "Assembly",
    "Chromosome",
    "Start",
    "Stop",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "VariationID",
    "Label",
    "STRING_protein_id",
    "split",
    "esm_prob",
    "gcn_prob",
    "cadd_am_prob",
    "gcn_cadd_am_prob",
    "esm_pred",
    "gcn_pred",
    "cadd_am_pred",
    "gcn_cadd_am_pred",
    "esm_vs_gcn_status",
    "esm_vs_gcn_transition",
    "cadd_am_vs_full_status",
    "cadd_am_vs_full_transition",
]

DISPLAY_COLS_FROM_37B = [
    "panel_name",
    "story_score",
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
    "abs_delta_prob_esm_to_gcn",
    "delta_prob_cadd_am_to_full",
    "abs_delta_prob_cadd_am_to_full",
]

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

# =========================
# Main
# =========================
print("[1/4] Loading Exp 34 and Exp 37B outputs...")
assert_exists(P34)
assert_exists(P37B)

df34 = pd.read_csv(P34)
df37b = pd.read_csv(P37B)

require_cols(df34, KEY_COLS, "Exp34")
require_cols(df37b, KEY_COLS + ["panel_name"], "Exp37B")

for c in ["STRING_protein_id", "Name"]:
    df34[c] = df34[c].astype(str)
    df37b[c] = df37b[c].astype(str)

print(f"Exp34 rows: {len(df34)}")
print(f"Exp37B rows: {len(df37b)}")

print("[2/4] Selecting display columns and merging...")
keep34 = [c for c in DISPLAY_COLS_FROM_34 if c in df34.columns]
keep37 = [c for c in DISPLAY_COLS_FROM_37B if c in df37b.columns]

left = df37b[KEY_COLS + keep37].copy()
right = df34[keep34].copy()

# remove duplicated non-key columns from right if they already exist in left
right_nonkey = [c for c in right.columns if c not in KEY_COLS]
dup_nonkey = [c for c in right_nonkey if c in left.columns]
right = right.drop(columns=dup_nonkey)

merged = left.merge(right, on=KEY_COLS, how="left", validate="one_to_one")

missing_rows = int(merged["GeneSymbol"].isna().sum()) if "GeneSymbol" in merged.columns else 0

merged.to_csv(OUT_DIR / "story_case_panel_final_display.csv", index=False)
print("Saved combined final display panel.")

print("[3/4] Writing per-panel final tables...")
panel_names = sorted(merged["panel_name"].dropna().unique().tolist())

for panel in panel_names:
    sub = merged[merged["panel_name"] == panel].copy()
    sub.to_csv(OUT_DIR / f"{panel}_final.csv", index=False)

print("[4/4] Writing summary.json...")
summary = {
    "experiment": "37C_story_case_panel_finalize",
    "real_data_policy": {
        "reuses_real_output_from_34": str(P34),
        "reuses_real_output_from_37b": str(P37B),
        "no_fake_data": True,
        "no_simulated_labels": True,
        "no_placeholder_predictions": True
    },
    "n_rows_input_37b": int(len(df37b)),
    "n_rows_final_panel": int(len(merged)),
    "missing_gene_symbol_rows_after_merge": missing_rows,
    "panels_finalized": panel_names,
    "outputs": {
        "story_case_panel_final_display_csv": str(OUT_DIR / "story_case_panel_final_display.csv"),
        "per_panel_final_tables": [str(OUT_DIR / f"{p}_final.csv") for p in panel_names]
    }
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
print(f"Results saved to: {OUT_DIR}")
