import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root")

TEST_SPLIT_PATH = ROOT / "data/processed/clinvar_test_main.csv"
CADD_TABLE_PATH = ROOT / "data/processed/clinvar_with_cadd.csv"
AM_TABLE_PATH = ROOT / "data/processed/clinvar_with_alphamissense.csv"

QUARTILE_SUMMARY_PATH = ROOT / "data/processed/quartile_stratified_analysis/quartile_analysis_summary.json"

CADD_ONLY_PROBS = ROOT / "data/processed/cadd_stacking/cadd_only/test_probs.npy"
GCN_CADD_PROBS  = ROOT / "data/processed/cadd_stacking/gcn_plus_cadd/test_probs.npy"

AM_ONLY_PROBS = ROOT / "data/processed/alphamissense_fusion/am_only/test_probs.npy"
GCN_AM_PROBS  = ROOT / "data/processed/alphamissense_fusion/gcn_plus_am/test_probs.npy"

OUT_DIR = ROOT / "data/processed/q1_case_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20
THRESHOLD = 0.5

print("=" * 80)
print("🔍 Step 17: Q1 case study (success + failure cases)")
print("=" * 80)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)
print("Top N      :", TOP_N)
print("Threshold  :", THRESHOLD)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
test_df = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)
cadd_df = pd.read_csv(CADD_TABLE_PATH, low_memory=False)
am_df   = pd.read_csv(AM_TABLE_PATH, low_memory=False)

with open(QUARTILE_SUMMARY_PATH, "r", encoding="utf-8") as f:
    quart_summary = json.load(f)

cadd_only_probs = np.load(CADD_ONLY_PROBS)
gcn_cadd_probs  = np.load(GCN_CADD_PROBS)
am_only_probs   = np.load(AM_ONLY_PROBS)
gcn_am_probs    = np.load(GCN_AM_PROBS)

if not (len(test_df) == len(cadd_only_probs) == len(gcn_cadd_probs) == len(am_only_probs) == len(gcn_am_probs)):
    raise ValueError("Length mismatch between test_df and probability arrays.")

print("\n[1] Loaded inputs")
print("Test rows:", len(test_df))

# -------------------------------------------------------------------
# 2. Merge external scores
# -------------------------------------------------------------------
cadd_keep = [c for c in ["VariationID", "CADD_PHRED", "CADD_RawScore"] if c in cadd_df.columns]
am_keep   = [c for c in ["VariationID", "am_pathogenicity", "am_class", "protein_variant", "uniprot_id", "transcript_id"] if c in am_df.columns]

cadd_small = cadd_df[cadd_keep].drop_duplicates(subset=["VariationID"]).copy()
am_small   = am_df[am_keep].drop_duplicates(subset=["VariationID"]).copy()

df = test_df.merge(cadd_small, on="VariationID", how="left")
df = df.merge(am_small, on="VariationID", how="left")

df["CADD_PHRED"] = pd.to_numeric(df["CADD_PHRED"], errors="coerce")
df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")

# attach probabilities
df["cadd_only_prob"] = cadd_only_probs
df["gcn_cadd_prob"]  = gcn_cadd_probs
df["am_only_prob"]   = am_only_probs
df["gcn_am_prob"]    = gcn_am_probs

# -------------------------------------------------------------------
# 3. Rebuild quartiles using saved training cutpoints
# -------------------------------------------------------------------
def assign_bins(values, edges):
    return pd.cut(values, bins=edges, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)

cadd_q = quart_summary["train_defined_cutpoints"]["cadd"]
am_q = quart_summary["train_defined_cutpoints"]["alphamissense"]

cadd_edges = [-np.inf, cadd_q["q1"], cadd_q["q2"], cadd_q["q3"], np.inf]
am_edges   = [-np.inf, am_q["q1"], am_q["q2"], am_q["q3"], np.inf]

# same fill policy as before
train_cadd_fill = cadd_q["q2"]   # median cutpoint came from training median
train_am_fill   = am_q["q2"]

df["CADD_PHRED_filled"] = df["CADD_PHRED"].fillna(train_cadd_fill)
df["am_pathogenicity_filled"] = df["am_pathogenicity"].fillna(train_am_fill)

df["cadd_quartile"] = assign_bins(df["CADD_PHRED_filled"], cadd_edges)
df["am_quartile"]   = assign_bins(df["am_pathogenicity_filled"], am_edges)

print("\n[2] Rebuilt quartile labels")
print("CADD Q1 count:", int((df["cadd_quartile"] == "Q1").sum()))
print("AM   Q1 count:", int((df["am_quartile"] == "Q1").sum()))

# -------------------------------------------------------------------
# 4. Helper functions
# -------------------------------------------------------------------
def binary_pred(probs, thr=0.5):
    return (probs >= thr).astype(int)

def add_case_flags(df, baseline_col, graph_col, label_col="Label"):
    out = df.copy()

    out["baseline_pred"] = binary_pred(out[baseline_col].values, THRESHOLD)
    out["graph_pred"] = binary_pred(out[graph_col].values, THRESHOLD)

    y = out[label_col].astype(int).values
    bpred = out["baseline_pred"].values
    gpred = out["graph_pred"].values

    out["baseline_correct"] = (bpred == y).astype(int)
    out["graph_correct"] = (gpred == y).astype(int)

    out["baseline_abs_error"] = np.abs(out[baseline_col] - y)
    out["graph_abs_error"] = np.abs(out[graph_col] - y)
    out["improvement"] = out["baseline_abs_error"] - out["graph_abs_error"]

    out["case_type"] = "other"
    out.loc[(out["baseline_correct"] == 0) & (out["graph_correct"] == 1), "case_type"] = "success"
    out.loc[(out["baseline_correct"] == 1) & (out["graph_correct"] == 0), "case_type"] = "failure"

    return out

def choose_columns(df):
    preferred = [
        "VariationID",
        "GeneSymbol",
        "gene_symbol",
        "Name",
        "STRING_protein_id",
        "Label",
        "CADD_PHRED",
        "am_pathogenicity",
        "cadd_quartile",
        "am_quartile",
        "cadd_only_prob",
        "gcn_cadd_prob",
        "am_only_prob",
        "gcn_am_prob",
        "baseline_pred",
        "graph_pred",
        "baseline_correct",
        "graph_correct",
        "baseline_abs_error",
        "graph_abs_error",
        "improvement",
        "case_type",
        "ClinicalSignificance",
        "Protein_change",
        "HGVSp",
        "HGVSc",
    ]
    cols = [c for c in preferred if c in df.columns]
    # add any remaining useful columns
    extra = [c for c in df.columns if c not in cols]
    return cols + extra[:10]

def export_case_table(df, out_path):
    cols = choose_columns(df)
    df[cols].to_csv(out_path, index=False)

# -------------------------------------------------------------------
# 5. CADD Q1 cases
# -------------------------------------------------------------------
cadd_q1 = df[df["cadd_quartile"] == "Q1"].copy()
cadd_cases = add_case_flags(cadd_q1, "cadd_only_prob", "gcn_cadd_prob")

cadd_success = cadd_cases[cadd_cases["case_type"] == "success"].copy()
cadd_failure = cadd_cases[cadd_cases["case_type"] == "failure"].copy()

cadd_success = cadd_success.sort_values("improvement", ascending=False).head(TOP_N)
cadd_failure = cadd_failure.sort_values("improvement", ascending=True).head(TOP_N)

# -------------------------------------------------------------------
# 6. AM Q1 cases
# -------------------------------------------------------------------
am_q1 = df[df["am_quartile"] == "Q1"].copy()
am_cases = add_case_flags(am_q1, "am_only_prob", "gcn_am_prob")

am_success = am_cases[am_cases["case_type"] == "success"].copy()
am_failure = am_cases[am_cases["case_type"] == "failure"].copy()

am_success = am_success.sort_values("improvement", ascending=False).head(TOP_N)
am_failure = am_failure.sort_values("improvement", ascending=True).head(TOP_N)

# -------------------------------------------------------------------
# 7. Save outputs
# -------------------------------------------------------------------
export_case_table(cadd_success, OUT_DIR / "cadd_q1_success_cases.csv")
export_case_table(cadd_failure, OUT_DIR / "cadd_q1_failure_cases.csv")
export_case_table(am_success, OUT_DIR / "am_q1_success_cases.csv")
export_case_table(am_failure, OUT_DIR / "am_q1_failure_cases.csv")

summary = {
    "threshold": THRESHOLD,
    "top_n": TOP_N,
    "counts": {
        "cadd_q1_total": int(len(cadd_q1)),
        "cadd_q1_success": int((cadd_cases["case_type"] == "success").sum()),
        "cadd_q1_failure": int((cadd_cases["case_type"] == "failure").sum()),
        "am_q1_total": int(len(am_q1)),
        "am_q1_success": int((am_cases["case_type"] == "success").sum()),
        "am_q1_failure": int((am_cases["case_type"] == "failure").sum()),
    }
}

with open(OUT_DIR / "q1_case_study_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[3] Case counts")
for k, v in summary["counts"].items():
    print(f"{k}: {v}")

print("\n[4] Saved outputs")
print("✅", OUT_DIR / "cadd_q1_success_cases.csv")
print("✅", OUT_DIR / "cadd_q1_failure_cases.csv")
print("✅", OUT_DIR / "am_q1_success_cases.csv")
print("✅", OUT_DIR / "am_q1_failure_cases.csv")
print("✅", OUT_DIR / "q1_case_study_summary.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
