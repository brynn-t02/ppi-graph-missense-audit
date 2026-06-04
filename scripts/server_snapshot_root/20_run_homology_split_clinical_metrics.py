import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, brier_score_loss

ROOT = Path("/root")

TEST_SPLIT_PATH = ROOT / "data/processed/homology_aware_split/clinvar_test_homology.csv"

PROB_PATHS = {
    "cadd_only_homology": ROOT / "data/processed/homology_single_tool_fusions/cadd_only_homology/test_probs.npy",
    "gcn_plus_cadd_homology": ROOT / "data/processed/homology_single_tool_fusions/gcn_plus_cadd_homology/test_probs.npy",
    "am_only_homology": ROOT / "data/processed/homology_single_tool_fusions/am_only_homology/test_probs.npy",
    "gcn_plus_am_homology": ROOT / "data/processed/homology_single_tool_fusions/gcn_plus_am_homology/test_probs.npy",
    "cadd_plus_am_homology": ROOT / "data/processed/homology_rerun_core/cadd_plus_am_homology/test_probs.npy",
    "gcn_plus_cadd_plus_am_homology": ROOT / "data/processed/homology_rerun_core/gcn_plus_cadd_plus_am_homology/test_probs.npy",
}

OUT_DIR = ROOT / "data/processed/homology_clinical_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🏥 Step 20: Homology-aware clinical metrics")
print("=" * 80)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)

# -------------------------------------------------------------------
# 1. Load labels and probabilities
# -------------------------------------------------------------------
test_df = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)
y_true = test_df["Label"].astype(int).values

model_probs = {}
for name, path in PROB_PATHS.items():
    if not path.exists():
        print(f"⚠️ Missing probability file for {name}: {path}")
        continue
    arr = np.load(path)
    if len(arr) != len(y_true):
        raise ValueError(f"Length mismatch for {name}: probs={len(arr)} vs labels={len(y_true)}")
    model_probs[name] = arr

print("\n[1] Loaded inputs")
print("Test rows:", len(y_true))
for name, probs in model_probs.items():
    print(f"{name:32s}: {len(probs)} probs")

# -------------------------------------------------------------------
# 2. Helpers
# -------------------------------------------------------------------
def compute_ece(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue

        bin_conf = float(np.mean(p[mask]))
        bin_acc = float(np.mean(y[mask]))
        weight = float(np.mean(mask))

        ece += weight * abs(bin_acc - bin_conf)

        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "mean_confidence": bin_conf,
            "observed_frequency": bin_acc,
            "abs_gap": abs(bin_acc - bin_conf),
        })

    return float(ece), pd.DataFrame(rows)

def sensitivity_at_specificity(y, p, target_specificity=0.90):
    fpr, tpr, thresholds = roc_curve(y, p)
    specificity = 1.0 - fpr

    valid = specificity >= target_specificity
    if not np.any(valid):
        return {
            "threshold": np.nan,
            "specificity": np.nan,
            "sensitivity": np.nan,
        }

    idx = np.argmax(tpr[valid])
    valid_idx = np.where(valid)[0][idx]

    return {
        "threshold": float(thresholds[valid_idx]),
        "specificity": float(specificity[valid_idx]),
        "sensitivity": float(tpr[valid_idx]),
    }

def decision_curve_analysis(y, p, thresholds):
    y = np.asarray(y)
    p = np.asarray(p)
    N = len(y)

    prevalence = float(np.mean(y))
    rows = []

    for thr in thresholds:
        if thr >= 1.0:
            continue

        pred_pos = p >= thr
        tp = np.sum((pred_pos == 1) & (y == 1))
        fp = np.sum((pred_pos == 1) & (y == 0))

        nb_model = (tp / N) - (fp / N) * (thr / (1 - thr))
        nb_all = prevalence - (1 - prevalence) * (thr / (1 - thr))
        nb_none = 0.0

        rows.append({
            "threshold": float(thr),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": float(nb_none),
        })

    return pd.DataFrame(rows)

# -------------------------------------------------------------------
# 3. Compute metrics
# -------------------------------------------------------------------
summary_rows = []
ece_tables = {}
dca_tables = {}

thresholds = np.arange(0.05, 0.96, 0.05)

for name, probs in model_probs.items():
    brier = brier_score_loss(y_true, probs)
    ece, ece_df = compute_ece(y_true, probs, n_bins=10)
    sens90 = sensitivity_at_specificity(y_true, probs, target_specificity=0.90)
    dca_df = decision_curve_analysis(y_true, probs, thresholds)

    summary_rows.append({
        "model": name,
        "brier_score": float(brier),
        "ece_10bin": float(ece),
        "sensitivity_at_90_specificity": float(sens90["sensitivity"]) if not np.isnan(sens90["sensitivity"]) else np.nan,
        "specificity_at_selected_threshold": float(sens90["specificity"]) if not np.isnan(sens90["specificity"]) else np.nan,
        "selected_threshold": float(sens90["threshold"]) if not np.isnan(sens90["threshold"]) else np.nan,
    })

    ece_tables[name] = ece_df
    dca_tables[name] = dca_df

summary_df = pd.DataFrame(summary_rows)

# -------------------------------------------------------------------
# 4. Save outputs
# -------------------------------------------------------------------
summary_df.to_csv(OUT_DIR / "homology_clinical_metrics_summary.csv", index=False)

for name, ece_df in ece_tables.items():
    ece_df.to_csv(OUT_DIR / f"{name}_ece_bins.csv", index=False)

for name, dca_df in dca_tables.items():
    dca_df.to_csv(OUT_DIR / f"{name}_dca.csv", index=False)

with open(OUT_DIR / "homology_clinical_metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump({"models": summary_rows}, f, indent=2)

print("\n[2] Saved outputs")
print("✅", OUT_DIR / "homology_clinical_metrics_summary.csv")
print("✅", OUT_DIR / "homology_clinical_metrics_summary.json")

print("\n[3] Homology clinical metric summary")
print(summary_df.to_string(index=False))

print("\n[4] DCA files")
for name in model_probs.keys():
    print("✅", OUT_DIR / f"{name}_dca.csv")

print("\n[5] ECE bin files")
for name in model_probs.keys():
    print("✅", OUT_DIR / f"{name}_ece_bins.csv")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
