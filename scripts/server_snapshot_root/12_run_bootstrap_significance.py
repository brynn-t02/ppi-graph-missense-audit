import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

TEST_SPLIT_PATH = ROOT / "data/processed/clinvar_test_main.csv"

PROB_PATHS = {
    "esm_raw": ROOT / "data/processed/variant_mlp_comparison/esm_raw/test_probs.npy",
    "gcn_enhanced": ROOT / "data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy",

    "cadd_only": ROOT / "data/processed/cadd_stacking/cadd_only/test_probs.npy",
    "gcn_plus_cadd": ROOT / "data/processed/cadd_stacking/gcn_plus_cadd/test_probs.npy",

    "am_only": ROOT / "data/processed/alphamissense_fusion/am_only/test_probs.npy",
    "gcn_plus_am": ROOT / "data/processed/alphamissense_fusion/gcn_plus_am/test_probs.npy",

    "cadd_plus_am": ROOT / "data/processed/triple_fusion/cadd_plus_am/test_probs.npy",
    "gcn_plus_cadd_plus_am": ROOT / "data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy",
}

COMPARISONS = [
    ("gcn_vs_esm", "gcn_enhanced", "esm_raw"),
    ("gcn_cadd_vs_cadd", "gcn_plus_cadd", "cadd_only"),
    ("gcn_am_vs_am", "gcn_plus_am", "am_only"),
    ("full_vs_cadd_am", "gcn_plus_cadd_plus_am", "cadd_plus_am"),
]

OUT_DIR = ROOT / "data/processed/bootstrap_significance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 80)
print("📈 Step 12: Bootstrap significance analysis")
print("=" * 80)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)
print("Bootstrap replicates:", N_BOOT)

# -------------------------------------------------------------------
# 1. Load labels and probs
# -------------------------------------------------------------------
test_df = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)
y_true = test_df["Label"].astype(int).values
n = len(y_true)

probs = {}
for name, path in PROB_PATHS.items():
    arr = np.load(path)
    if len(arr) != n:
        raise ValueError(f"Length mismatch for {name}: {len(arr)} vs {n}")
    probs[name] = arr

print("\n[1] Loaded labels and probabilities")
print("Test rows:", n)
for name in probs:
    print(f"{name:24s}: {len(probs[name])}")

# -------------------------------------------------------------------
# 2. Helpers
# -------------------------------------------------------------------
def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)

def bootstrap_compare(y, p_a, p_b, n_boot=1000):
    delta_auc = []
    delta_auprc = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        y_b = y[idx]
        p_a_b = p_a[idx]
        p_b_b = p_b[idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc_a = safe_auc(y_b, p_a_b)
        auc_b = safe_auc(y_b, p_b_b)
        auprc_a = safe_auprc(y_b, p_a_b)
        auprc_b = safe_auprc(y_b, p_b_b)

        delta_auc.append(auc_a - auc_b)
        delta_auprc.append(auprc_a - auprc_b)

    delta_auc = np.array(delta_auc)
    delta_auprc = np.array(delta_auprc)

    def summarize(arr):
        mean = float(np.mean(arr))
        lo = float(np.quantile(arr, 0.025))
        hi = float(np.quantile(arr, 0.975))

        # simple two-sided sign-based p estimate
        p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        p = float(min(p, 1.0))

        return {
            "mean_delta": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "p_value_bootstrap_sign": p,
            "n_boot_used": int(len(arr)),
        }

    return summarize(delta_auc), summarize(delta_auprc), delta_auc, delta_auprc

# -------------------------------------------------------------------
# 3. Run comparisons
# -------------------------------------------------------------------
summary = {}
rows = []

for comp_name, model_a, model_b in COMPARISONS:
    print("\n" + "-" * 60)
    print(f"[{comp_name}] {model_a} vs {model_b}")
    print("-" * 60)

    auc_res, auprc_res, delta_auc_arr, delta_auprc_arr = bootstrap_compare(
        y_true,
        probs[model_a],
        probs[model_b],
        n_boot=N_BOOT,
    )

    summary[comp_name] = {
        "model_a": model_a,
        "model_b": model_b,
        "delta_auc": auc_res,
        "delta_auprc": auprc_res,
    }

    rows.append({
        "comparison": comp_name,
        "model_a": model_a,
        "model_b": model_b,
        "metric": "delta_auc",
        **auc_res,
    })
    rows.append({
        "comparison": comp_name,
        "model_a": model_a,
        "model_b": model_b,
        "metric": "delta_auprc",
        **auprc_res,
    })

    np.save(OUT_DIR / f"{comp_name}_delta_auc_bootstrap.npy", delta_auc_arr)
    np.save(OUT_DIR / f"{comp_name}_delta_auprc_bootstrap.npy", delta_auprc_arr)

    print("ΔAUC   :", f"{auc_res['mean_delta']:.4f}",
          f"[{auc_res['ci95_low']:.4f}, {auc_res['ci95_high']:.4f}]",
          f"p={auc_res['p_value_bootstrap_sign']:.4g}")
    print("ΔAUPRC :", f"{auprc_res['mean_delta']:.4f}",
          f"[{auprc_res['ci95_low']:.4f}, {auprc_res['ci95_high']:.4f}]",
          f"p={auprc_res['p_value_bootstrap_sign']:.4g}")

# -------------------------------------------------------------------
# 4. Save outputs
# -------------------------------------------------------------------
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "bootstrap_significance_summary.csv", index=False)

with open(OUT_DIR / "bootstrap_significance_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[2] Saved outputs")
print("✅", OUT_DIR / "bootstrap_significance_summary.csv")
print("✅", OUT_DIR / "bootstrap_significance_summary.json")

print("\n[3] Summary table")
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
