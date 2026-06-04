import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"

TRAIN_SPLIT_PATH = ROOT / "data/processed/clinvar_train_main.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/clinvar_test_main.csv"

CADD_TABLE_PATH = ROOT / "data/processed/clinvar_with_cadd.csv"
AM_TABLE_PATH   = ROOT / "data/processed/clinvar_with_alphamissense.csv"

CADD_ONLY_PROBS_PATH = ROOT / "data/processed/cadd_stacking/cadd_only/test_probs.npy"
GCN_CADD_PROBS_PATH  = ROOT / "data/processed/cadd_stacking/gcn_plus_cadd/test_probs.npy"

AM_ONLY_PROBS_PATH = ROOT / "data/processed/alphamissense_fusion/am_only/test_probs.npy"
GCN_AM_PROBS_PATH  = ROOT / "data/processed/alphamissense_fusion/gcn_plus_am/test_probs.npy"

OUT_DIR = ROOT / "data/processed/quartile_stratified_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📊 Step 09a: Q1–Q4 stratified analysis")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("Train split :", TRAIN_SPLIT_PATH)
print("Test split  :", TEST_SPLIT_PATH)
print("CADD table  :", CADD_TABLE_PATH)
print("AM table    :", AM_TABLE_PATH)
print("Outdir      :", OUT_DIR)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))

def safe_auprc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))

def assign_bins(values, edges):
    """
    edges: [-inf, q1, q2, q3, inf]
    returns labels Q1..Q4
    """
    labels = ["Q1", "Q2", "Q3", "Q4"]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True)

def summarize_by_quartile(df, quartile_col, label_col, prob_a_col, prob_b_col, method_a_name, method_b_name):
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[df[quartile_col] == q].copy()
        y = sub[label_col].values

        row = {
            "quartile": q,
            "n": int(len(sub)),
            "positive_rate": float(sub[label_col].mean()) if len(sub) > 0 else np.nan,
            f"{method_a_name}_auc": safe_auc(y, sub[prob_a_col].values) if len(sub) > 0 else np.nan,
            f"{method_a_name}_auprc": safe_auprc(y, sub[prob_a_col].values) if len(sub) > 0 else np.nan,
            f"{method_b_name}_auc": safe_auc(y, sub[prob_b_col].values) if len(sub) > 0 else np.nan,
            f"{method_b_name}_auprc": safe_auprc(y, sub[prob_b_col].values) if len(sub) > 0 else np.nan,
        }

        if not np.isnan(row[f"{method_a_name}_auc"]) and not np.isnan(row[f"{method_b_name}_auc"]):
            row["delta_auc"] = row[f"{method_b_name}_auc"] - row[f"{method_a_name}_auc"]
        else:
            row["delta_auc"] = np.nan

        if not np.isnan(row[f"{method_a_name}_auprc"]) and not np.isnan(row[f"{method_b_name}_auprc"]):
            row["delta_auprc"] = row[f"{method_b_name}_auprc"] - row[f"{method_a_name}_auprc"]
        else:
            row["delta_auprc"] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)

# -------------------------------------------------------------------
# 1. Load split tables
# -------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_SPLIT_PATH, low_memory=False)
test_df  = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

print("\n[1] Loaded split tables")
print("Train rows:", len(train_df))
print("Test rows :", len(test_df))

# -------------------------------------------------------------------
# 2. Load external score tables
# -------------------------------------------------------------------
cadd_df = pd.read_csv(CADD_TABLE_PATH, low_memory=False)
am_df   = pd.read_csv(AM_TABLE_PATH, low_memory=False)

print("\n[2] Loaded external score tables")
print("CADD rows:", len(cadd_df))
print("AM rows  :", len(am_df))

# keep only needed columns
cadd_keep = [c for c in ["VariationID", "CADD_PHRED"] if c in cadd_df.columns]
am_keep   = [c for c in ["VariationID", "am_pathogenicity"] if c in am_df.columns]

cadd_small = cadd_df[cadd_keep].drop_duplicates(subset=["VariationID"]).copy()
am_small   = am_df[am_keep].drop_duplicates(subset=["VariationID"]).copy()

train_df = train_df.merge(cadd_small, on="VariationID", how="left")
test_df  = test_df.merge(cadd_small, on="VariationID", how="left")

train_df = train_df.merge(am_small, on="VariationID", how="left")
test_df  = test_df.merge(am_small, on="VariationID", how="left")

# numeric conversion
for df in [train_df, test_df]:
    df["CADD_PHRED"] = pd.to_numeric(df["CADD_PHRED"], errors="coerce")
    df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")

# -------------------------------------------------------------------
# 3. Fill missing using training statistics only
# -------------------------------------------------------------------
train_cadd_fill = train_df["CADD_PHRED"].median()
train_am_fill   = train_df["am_pathogenicity"].median()

train_df["CADD_PHRED_filled"] = train_df["CADD_PHRED"].fillna(train_cadd_fill)
test_df["CADD_PHRED_filled"]  = test_df["CADD_PHRED"].fillna(train_cadd_fill)

train_df["am_pathogenicity_filled"] = train_df["am_pathogenicity"].fillna(train_am_fill)
test_df["am_pathogenicity_filled"]  = test_df["am_pathogenicity"].fillna(train_am_fill)

print("\n[3] Fill values")
print("Train CADD fill value:", float(train_cadd_fill))
print("Train AM fill value  :", float(train_am_fill))
print("Train AM missing     :", int(train_df["am_pathogenicity"].isna().sum()))
print("Test AM missing      :", int(test_df["am_pathogenicity"].isna().sum()))

# -------------------------------------------------------------------
# 4. Define quartile cutpoints on TRAIN only
# -------------------------------------------------------------------
cadd_q1, cadd_q2, cadd_q3 = train_df["CADD_PHRED_filled"].quantile([0.25, 0.50, 0.75]).tolist()
am_q1, am_q2, am_q3 = train_df["am_pathogenicity_filled"].quantile([0.25, 0.50, 0.75]).tolist()

cadd_edges = [-np.inf, cadd_q1, cadd_q2, cadd_q3, np.inf]
am_edges   = [-np.inf, am_q1, am_q2, am_q3, np.inf]

test_df["cadd_quartile"] = assign_bins(test_df["CADD_PHRED_filled"], cadd_edges)
test_df["am_quartile"]   = assign_bins(test_df["am_pathogenicity_filled"], am_edges)

print("\n[4] Train-defined quartile cutpoints")
print("CADD:", cadd_edges)
print("AM  :", am_edges)

# -------------------------------------------------------------------
# 5. Load saved test probabilities
# -------------------------------------------------------------------
cadd_only_probs = np.load(CADD_ONLY_PROBS_PATH)
gcn_cadd_probs  = np.load(GCN_CADD_PROBS_PATH)

am_only_probs = np.load(AM_ONLY_PROBS_PATH)
gcn_am_probs  = np.load(GCN_AM_PROBS_PATH)

if not (
    len(cadd_only_probs) == len(gcn_cadd_probs) == len(am_only_probs) == len(gcn_am_probs) == len(test_df)
):
    raise ValueError(
        f"Length mismatch: "
        f"test_df={len(test_df)}, "
        f"cadd_only={len(cadd_only_probs)}, gcn_cadd={len(gcn_cadd_probs)}, "
        f"am_only={len(am_only_probs)}, gcn_am={len(gcn_am_probs)}"
    )

test_df["cadd_only_prob"] = cadd_only_probs
test_df["gcn_cadd_prob"]  = gcn_cadd_probs
test_df["am_only_prob"]   = am_only_probs
test_df["gcn_am_prob"]    = gcn_am_probs

print("\n[5] Loaded saved test probabilities")
print("Length check OK:", len(test_df))

# -------------------------------------------------------------------
# 6. CADD quartile analysis
# -------------------------------------------------------------------
cadd_quart_df = summarize_by_quartile(
    df=test_df,
    quartile_col="cadd_quartile",
    label_col="Label",
    prob_a_col="cadd_only_prob",
    prob_b_col="gcn_cadd_prob",
    method_a_name="cadd_only",
    method_b_name="gcn_plus_cadd",
)

# -------------------------------------------------------------------
# 7. AM quartile analysis
# -------------------------------------------------------------------
am_quart_df = summarize_by_quartile(
    df=test_df,
    quartile_col="am_quartile",
    label_col="Label",
    prob_a_col="am_only_prob",
    prob_b_col="gcn_am_prob",
    method_a_name="am_only",
    method_b_name="gcn_plus_am",
)

# -------------------------------------------------------------------
# 8. Save outputs
# -------------------------------------------------------------------
cadd_quart_df.to_csv(OUT_DIR / "cadd_quartile_analysis.csv", index=False)
am_quart_df.to_csv(OUT_DIR / "am_quartile_analysis.csv", index=False)

summary = {
    "train_defined_cutpoints": {
        "cadd": {
            "q1": float(cadd_q1),
            "q2": float(cadd_q2),
            "q3": float(cadd_q3),
        },
        "alphamissense": {
            "q1": float(am_q1),
            "q2": float(am_q2),
            "q3": float(am_q3),
        },
    },
    "cadd_quartile_analysis": cadd_quart_df.to_dict(orient="records"),
    "am_quartile_analysis": am_quart_df.to_dict(orient="records"),
}

with open(OUT_DIR / "quartile_analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[6] Saved outputs")
print("✅", OUT_DIR / "cadd_quartile_analysis.csv")
print("✅", OUT_DIR / "am_quartile_analysis.csv")
print("✅", OUT_DIR / "quartile_analysis_summary.json")

print("\n[7] CADD quartile analysis")
print(cadd_quart_df.to_string(index=False))

print("\n[8] AM quartile analysis")
print(am_quart_df.to_string(index=False))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
