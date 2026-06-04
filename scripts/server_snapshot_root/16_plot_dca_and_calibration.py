import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/root")

CLINICAL_DIR = ROOT / "data/processed/clinical_metrics"
OUT_DIR = ROOT / "data/processed/clinical_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DCA_MODELS = [
    "cadd_only",
    "am_only",
    "cadd_plus_am",
    "gcn_plus_cadd_plus_am",
]

ECE_MODELS = DCA_MODELS

print("=" * 80)
print("📊 Step 16: Plot DCA and calibration")
print("=" * 80)
print("Clinical dir:", CLINICAL_DIR)
print("Outdir      :", OUT_DIR)

# -------------------------------------------------------------------
# 1. Load DCA and ECE files
# -------------------------------------------------------------------
dca_tables = {}
ece_tables = {}

for model in DCA_MODELS:
    dca_path = CLINICAL_DIR / f"{model}_dca.csv"
    if dca_path.exists():
        dca_tables[model] = pd.read_csv(dca_path)

for model in ECE_MODELS:
    ece_path = CLINICAL_DIR / f"{model}_ece_bins.csv"
    if ece_path.exists():
        ece_tables[model] = pd.read_csv(ece_path)

print("\n[1] Loaded files")
print("DCA models:", list(dca_tables.keys()))
print("ECE models:", list(ece_tables.keys()))

# -------------------------------------------------------------------
# 2. Plot DCA
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for model, df in dca_tables.items():
    plt.plot(df["threshold"], df["net_benefit_model"], label=model, linewidth=2)

# add treat-all / treat-none from one table if available
if len(dca_tables) > 0:
    ref_df = next(iter(dca_tables.values()))
    plt.plot(ref_df["threshold"], ref_df["net_benefit_treat_all"],
             linestyle="--", color="gray", label="treat_all")
    plt.plot(ref_df["threshold"], ref_df["net_benefit_treat_none"],
             linestyle=":", color="black", label="treat_none")

plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision curve analysis")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "dca_all_models.png", dpi=300)
plt.savefig(OUT_DIR / "dca_all_models.pdf")
plt.close()

# focused DCA plot
focus_models = ["cadd_plus_am", "gcn_plus_cadd_plus_am"]
plt.figure(figsize=(7, 5))
for model in focus_models:
    if model in dca_tables:
        df = dca_tables[model]
        plt.plot(df["threshold"], df["net_benefit_model"], label=model, linewidth=2)

if len(dca_tables) > 0:
    ref_df = next(iter(dca_tables.values()))
    plt.plot(ref_df["threshold"], ref_df["net_benefit_treat_all"],
             linestyle="--", color="gray", label="treat_all")
    plt.plot(ref_df["threshold"], ref_df["net_benefit_treat_none"],
             linestyle=":", color="black", label="treat_none")

plt.xlim(0.05, 0.5)
plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision curve analysis (focused)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "dca_focus_caddam_vs_full.png", dpi=300)
plt.savefig(OUT_DIR / "dca_focus_caddam_vs_full.pdf")
plt.close()

print("\n[2] Saved DCA plots")
print("✅", OUT_DIR / "dca_all_models.png")
print("✅", OUT_DIR / "dca_all_models.pdf")
print("✅", OUT_DIR / "dca_focus_caddam_vs_full.png")
print("✅", OUT_DIR / "dca_focus_caddam_vs_full.pdf")

# -------------------------------------------------------------------
# 3. Plot calibration / reliability curves
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for model, df in ece_tables.items():
    plt.plot(df["mean_confidence"], df["observed_frequency"], marker="o", label=model)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect_calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed event frequency")
plt.title("Calibration / reliability plot")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "calibration_all_models.png", dpi=300)
plt.savefig(OUT_DIR / "calibration_all_models.pdf")
plt.close()

plt.figure(figsize=(7, 5))
for model in focus_models:
    if model in ece_tables:
        df = ece_tables[model]
        plt.plot(df["mean_confidence"], df["observed_frequency"], marker="o", label=model)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect_calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed event frequency")
plt.title("Calibration (focused)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "calibration_focus_caddam_vs_full.png", dpi=300)
plt.savefig(OUT_DIR / "calibration_focus_caddam_vs_full.pdf")
plt.close()

print("\n[3] Saved calibration plots")
print("✅", OUT_DIR / "calibration_all_models.png")
print("✅", OUT_DIR / "calibration_all_models.pdf")
print("✅", OUT_DIR / "calibration_focus_caddam_vs_full.png")
print("✅", OUT_DIR / "calibration_focus_caddam_vs_full.pdf")

# -------------------------------------------------------------------
# 4. DCA numeric summary
# -------------------------------------------------------------------
summary_rows = []

if "cadd_plus_am" in dca_tables and "gcn_plus_cadd_plus_am" in dca_tables:
    base_df = dca_tables["cadd_plus_am"].copy()
    full_df = dca_tables["gcn_plus_cadd_plus_am"].copy()

    merged = base_df.merge(
        full_df[["threshold", "net_benefit_model"]],
        on="threshold",
        suffixes=("_cadd_am", "_full")
    )
    merged["delta_net_benefit_full_minus_cadd_am"] = (
        merged["net_benefit_model_full"] - merged["net_benefit_model_cadd_am"]
    )

    for thr in [0.10, 0.20, 0.30]:
        sub = merged[np.isclose(merged["threshold"], thr)]
        if len(sub) == 1:
            row = sub.iloc[0]
            summary_rows.append({
                "threshold": float(thr),
                "net_benefit_cadd_plus_am": float(row["net_benefit_model_cadd_am"]),
                "net_benefit_gcn_plus_cadd_plus_am": float(row["net_benefit_model_full"]),
                "delta_full_minus_cadd_am": float(row["delta_net_benefit_full_minus_cadd_am"]),
            })

    # average delta over threshold range 0.05-0.30
    sub_range = merged[(merged["threshold"] >= 0.05) & (merged["threshold"] <= 0.30)]
    if len(sub_range) > 0:
        summary_rows.append({
            "threshold": "mean_0.05_0.30",
            "net_benefit_cadd_plus_am": float(sub_range["net_benefit_model_cadd_am"].mean()),
            "net_benefit_gcn_plus_cadd_plus_am": float(sub_range["net_benefit_model_full"].mean()),
            "delta_full_minus_cadd_am": float(sub_range["delta_net_benefit_full_minus_cadd_am"].mean()),
        })

    merged.to_csv(OUT_DIR / "dca_focus_delta_table.csv", index=False)
    print("\n[4] Saved detailed DCA delta table")
    print("✅", OUT_DIR / "dca_focus_delta_table.csv")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "dca_numeric_summary.csv", index=False)

print("\n[5] DCA numeric summary")
if len(summary_df) > 0:
    print(summary_df.to_string(index=False))
else:
    print("No focused DCA summary available.")

print("\n✅", OUT_DIR / "dca_numeric_summary.csv")

# -------------------------------------------------------------------
# 5. Save mini report
# -------------------------------------------------------------------
report = {
    "dca_models": list(dca_tables.keys()),
    "ece_models": list(ece_tables.keys()),
    "outputs": [
        "dca_all_models.png/pdf",
        "dca_focus_caddam_vs_full.png/pdf",
        "calibration_all_models.png/pdf",
        "calibration_focus_caddam_vs_full.png/pdf",
        "dca_numeric_summary.csv",
    ]
}

with open(OUT_DIR / "plot_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n[6] Saved report")
print("✅", OUT_DIR / "plot_report.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
