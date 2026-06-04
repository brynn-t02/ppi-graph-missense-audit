import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/root")

SRC_DIR = ROOT / "data/processed/paper_figures_tables_publication"
OUT_DIR = ROOT / "data/processed/paper_figures_tables_final"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# source tables
TABLE1 = SRC_DIR / "tables/Table1_split_statistics_publication.csv"
TABLE2 = SRC_DIR / "tables/Table2_overall_performance_publication.csv"
TABLE3 = SRC_DIR / "tables/Table3_bootstrap_significance_publication.csv"
TABLE4 = SRC_DIR / "tables/Table4_clinical_metrics_publication.csv"
TABLE5 = SRC_DIR / "tables/Table5_mechanism_controls_publication.csv"

# source raw clinical files
CADDAM_DCA = ROOT / "data/processed/clinical_metrics/cadd_plus_am_dca.csv"
FULL_DCA = ROOT / "data/processed/clinical_metrics/gcn_plus_cadd_plus_am_dca.csv"
CADDAM_ECE = ROOT / "data/processed/clinical_metrics/cadd_plus_am_ece_bins.csv"
FULL_ECE = ROOT / "data/processed/clinical_metrics/gcn_plus_cadd_plus_am_ece_bins.csv"

# source homology clinical
HOMO_CLIN = ROOT / "data/processed/homology_clinical_metrics/homology_clinical_metrics_summary.csv"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#4C78A8"
RED = "#E45756"
GREEN = "#54A24B"
PURPLE = "#B279A2"
ORANGE = "#F58518"
GRAY = "#B0B0B0"
BLACK = "#222222"

print("=" * 100)
print("🎨 Step 32: Final polish of publication figures and tables")
print("=" * 100)
print("Source dir:", SRC_DIR)
print("Output dir:", OUT_DIR)

def save_fig(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------
# 1. Figure 3 refined: paired comparison layout
# ---------------------------------------------------------------------
table2 = pd.read_csv(TABLE2)

# normalize names
orig = table2[table2["Split"] == "Original"].copy()
homo = table2[table2["Split"] == "Homology-aware"].copy()

def get_metric(df, model, metric):
    row = df[df["Model"] == model]
    return float(row.iloc[0][metric])

orig_pairs = [
    ("CADD", get_metric(orig, "CADD", "Test AUC"), get_metric(orig, "GCN+CADD", "Test AUC"),
             get_metric(orig, "CADD", "Test AUPRC"), get_metric(orig, "GCN+CADD", "Test AUPRC")),
    ("AM", get_metric(orig, "AM", "Test AUC"), get_metric(orig, "GCN+AM", "Test AUC"),
           get_metric(orig, "AM", "Test AUPRC"), get_metric(orig, "GCN+AM", "Test AUPRC")),
    ("CADD+AM", get_metric(orig, "CADD+AM", "Test AUC"), get_metric(orig, "GCN+CADD+AM", "Test AUC"),
                get_metric(orig, "CADD+AM", "Test AUPRC"), get_metric(orig, "GCN+CADD+AM", "Test AUPRC")),
]

homo_pairs = [
    ("CADD", get_metric(homo, "CADD", "Test AUC"), get_metric(homo, "GCN+CADD", "Test AUC"),
             get_metric(homo, "CADD", "Test AUPRC"), get_metric(homo, "GCN+CADD", "Test AUPRC")),
    ("AM", get_metric(homo, "AM", "Test AUC"), get_metric(homo, "GCN+AM", "Test AUC"),
           get_metric(homo, "AM", "Test AUPRC"), get_metric(homo, "GCN+AM", "Test AUPRC")),
    ("CADD+AM", get_metric(homo, "CADD+AM", "Test AUC"), get_metric(homo, "GCN+CADD+AM", "Test AUC"),
                get_metric(homo, "CADD+AM", "Test AUPRC"), get_metric(homo, "GCN+CADD+AM", "Test AUPRC")),
]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
width = 0.34

for row_idx, pairs, title in [
    (0, orig_pairs, "Original protein-level split"),
    (1, homo_pairs, "Homology-aware split"),
]:
    x = np.arange(len(pairs))
    labels = [p[0] for p in pairs]

    # AUC
    base_auc = [p[1] for p in pairs]
    gcn_auc = [p[2] for p in pairs]
    ax = axes[row_idx, 0]
    ax.bar(x - width/2, base_auc, width, color=BLUE, label="Baseline", edgecolor="black", linewidth=0.3)
    ax.bar(x + width/2, gcn_auc, width, color=RED, label="Graph-enhanced", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test AUC")
    ax.set_title(f"{title}: AUC")
    ymin = min(base_auc + gcn_auc)
    ymax = max(base_auc + gcn_auc)
    pad = max((ymax - ymin) * 0.35, 0.005)
    ax.set_ylim(ymin - pad, min(1.0, ymax + pad))
    if row_idx == 0:
        ax.legend(frameon=False, loc="upper left")

    # AUPRC
    base_ap = [p[3] for p in pairs]
    gcn_ap = [p[4] for p in pairs]
    ax = axes[row_idx, 1]
    ax.bar(x - width/2, base_ap, width, color=BLUE, label="Baseline", edgecolor="black", linewidth=0.3)
    ax.bar(x + width/2, gcn_ap, width, color=RED, label="Graph-enhanced", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test AUPRC")
    ax.set_title(f"{title}: AUPRC")
    ymin = min(base_ap + gcn_ap)
    ymax = max(base_ap + gcn_ap)
    pad = max((ymax - ymin) * 0.35, 0.01)
    ax.set_ylim(max(0, ymin - pad), min(1.0, ymax + pad))

fig.tight_layout()
save_fig(fig, "Figure3_fusion_performance_final")

# ---------------------------------------------------------------------
# 2. Figure 5 refined: DCA + calibration cleaner
# ---------------------------------------------------------------------
caddam_dca = pd.read_csv(CADDAM_DCA)
full_dca = pd.read_csv(FULL_DCA)
caddam_ece = pd.read_csv(CADDAM_ECE)
full_ece = pd.read_csv(FULL_ECE)

fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

# DCA
ax = axes[0]
ax.plot(caddam_dca["threshold"], caddam_dca["net_benefit_model"], lw=2.6, color=BLUE, label="CADD+AM")
ax.plot(full_dca["threshold"], full_dca["net_benefit_model"], lw=2.6, color=RED, label="GCN+CADD+AM")
ax.plot(caddam_dca["threshold"], caddam_dca["net_benefit_treat_none"], ":", lw=1.4, color=BLACK, alpha=0.85, label="Treat none")
ax.plot(caddam_dca["threshold"], caddam_dca["net_benefit_treat_all"], "--", lw=1.4, color=GRAY, alpha=0.95, label="Treat all")
ax.set_xlim(0.05, 0.50)
y_min = min(caddam_dca["net_benefit_model"].min(), full_dca["net_benefit_model"].min())
y_max = max(caddam_dca["net_benefit_model"].max(), full_dca["net_benefit_model"].max())
pad = max((y_max - y_min) * 0.25, 0.015)
ax.set_ylim(max(-0.02, y_min - pad), y_max + pad)
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.set_title("Decision-curve analysis")
ax.legend(frameon=False, loc="lower left")

# Calibration
ax = axes[1]
ax.plot(caddam_ece["mean_confidence"], caddam_ece["observed_frequency"], "-o", ms=5, lw=2.2, color=BLUE, label="CADD+AM")
ax.plot(full_ece["mean_confidence"], full_ece["observed_frequency"], "-o", ms=5, lw=2.2, color=RED, label="GCN+CADD+AM")
ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.5, label="Perfect calibration")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed event frequency")
ax.set_title("Calibration curves")
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
save_fig(fig, "Figure5_dca_calibration_final")

# ---------------------------------------------------------------------
# 3. Figure 6 refined: homology clinical metrics
# ---------------------------------------------------------------------
homo_clin = pd.read_csv(HOMO_CLIN)
wanted = [
    "cadd_only_homology",
    "gcn_plus_cadd_homology",
    "am_only_homology",
    "gcn_plus_am_homology",
    "cadd_plus_am_homology",
    "gcn_plus_cadd_plus_am_homology",
]
label_map = {
    "cadd_only_homology": "CADD",
    "gcn_plus_cadd_homology": "GCN+CADD",
    "am_only_homology": "AM",
    "gcn_plus_am_homology": "GCN+AM",
    "cadd_plus_am_homology": "CADD+AM",
    "gcn_plus_cadd_plus_am_homology": "GCN+CADD+AM",
}
homo_clin = homo_clin[homo_clin["model"].isin(wanted)].copy()
homo_clin["short"] = homo_clin["model"].map(label_map)

colors = [BLUE if "gcn" not in m else RED for m in homo_clin["model"]]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
metric_info = [
    ("brier_score", "Brier score"),
    ("ece_10bin", "ECE"),
    ("sensitivity_at_90_specificity", "Sensitivity @ 90% specificity"),
]

for ax, (col, title) in zip(axes, metric_info):
    ax.bar(np.arange(len(homo_clin)), homo_clin[col], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(homo_clin)))
    ax.set_xticklabels(homo_clin["short"], rotation=20, ha="right")
    ax.set_title(title)

fig.tight_layout()
save_fig(fig, "Figure6_homology_clinical_metrics_final")

# ---------------------------------------------------------------------
# 4. Final tables: save XLSX + cleaned CSV
# ---------------------------------------------------------------------
table1 = pd.read_csv(TABLE1)
table2 = pd.read_csv(TABLE2)
table3 = pd.read_csv(TABLE3)
table4 = pd.read_csv(TABLE4)
table5 = pd.read_csv(TABLE5)

# light cleanup for table 1
table1_final = table1.copy()
table1_final = table1_final.fillna("—")

# sort table2
split_order = {"Original": 0, "Homology-aware": 1}
model_order = {
    "ESM": 0, "GCN": 1,
    "CADD": 2, "GCN+CADD": 3,
    "AM": 4, "GCN+AM": 5,
    "CADD+AM": 6, "GCN+CADD+AM": 7,
    "ESM (homology)": 0, "GCN (homology)": 1,
}
table2_final = table2.copy()
table2_final["split_order"] = table2_final["Split"].map(split_order)
table2_final["model_order"] = table2_final["Model"].map(model_order).fillna(999)
table2_final = table2_final.sort_values(["split_order", "model_order"]).drop(columns=["split_order", "model_order"])

# shorten comparison labels in table3 if needed
table3_final = table3.copy()
table3_final["Comparison"] = table3_final["Comparison"].replace({
    "GCN vs ESM (homology)": "GCN vs ESM (H)",
    "GCN+CADD vs CADD (homology)": "GCN+CADD vs CADD (H)",
    "GCN+AM vs AM (homology)": "GCN+AM vs AM (H)",
    "GCN+CADD+AM vs CADD+AM (homology)": "Full vs CADD+AM (H)",
})

table4_final = table4.copy()
table5_final = table5.copy()

table1_final.to_csv(TAB_DIR / "Table1_split_statistics_final.csv", index=False)
table2_final.to_csv(TAB_DIR / "Table2_overall_performance_final.csv", index=False)
table3_final.to_csv(TAB_DIR / "Table3_bootstrap_significance_final.csv", index=False)
table4_final.to_csv(TAB_DIR / "Table4_clinical_metrics_final.csv", index=False)
table5_final.to_csv(TAB_DIR / "Table5_mechanism_controls_final.csv", index=False)

xlsx_path = TAB_DIR / "Final_publication_tables.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    table1_final.to_excel(writer, sheet_name="Table1_split_stats", index=False)
    table2_final.to_excel(writer, sheet_name="Table2_performance", index=False)
    table3_final.to_excel(writer, sheet_name="Table3_bootstrap", index=False)
    table4_final.to_excel(writer, sheet_name="Table4_clinical", index=False)
    table5_final.to_excel(writer, sheet_name="Table5_controls", index=False)

# ---------------------------------------------------------------------
# 5. Manifest
# ---------------------------------------------------------------------
manifest = {
    "figures": sorted([p.name for p in FIG_DIR.iterdir()]),
    "tables": sorted([p.name for p in TAB_DIR.iterdir()]),
}
with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\n[1] Saved final-polish figures")
for p in sorted(FIG_DIR.iterdir()):
    print("✅", p)

print("\n[2] Saved final-polish tables")
for p in sorted(TAB_DIR.iterdir()):
    print("✅", p)

print("\n✅ Manifest:", OUT_DIR / "manifest.json")
print("=" * 100)
print("Done.")
print("=" * 100)
