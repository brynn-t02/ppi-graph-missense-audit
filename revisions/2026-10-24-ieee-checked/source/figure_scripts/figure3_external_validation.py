"""Figure 3: verified paired ensemble AUC margins, without fitted-data changes."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SOURCE = Path(__file__).resolve().parents[1]
OUT = SOURCE / "figure_drafts"
RESULTS = SOURCE / "external_validation_20261024" / "cluster_bootstrap.json"
data = json.loads(RESULTS.read_text())
assert data["bootstrap_repeats"] == 2000
comparisons = data["comparisons"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 240,
})

COL_STRING = "#2166AC"
COL_BIOGRID = "#C46C24"
COL_ORIGINAL = "#397C68"
COL_HOMOLOGY = "#8862A8"
fig = plt.figure(figsize=(7.0, 3.40), facecolor="white")
ax_a = fig.add_axes([0.180, 0.280, 0.310, 0.570])
ax_b = fig.add_axes([0.714, 0.280, 0.270, 0.570])

def style_axis(ax, limits, ticks):
    ax.axvline(0, color="#505050", lw=0.85, ls=(0, (3, 2)), zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e7e7e7", lw=0.5)
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:+.3f}"))
    ax.set_xlabel("Paired ensemble ΔAUC", labelpad=5)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=3, pad=4)

def point(ax, population, comparison, y, color, marker):
    row = comparisons[population][comparison]
    x = row["delta_auc"]
    lo, hi = row["auc_ci95"]
    assert lo <= x <= hi and row["finite_replicates"] == 2000
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=marker,
                color=color, ecolor=color, markersize=5, capsize=2.5,
                elinewidth=1.3, capthick=1.1, zorder=3)

a_rows = [
    ("external_primary", "full_FPP_vs_raw_ESM", 3.8, "Primary: STRING", COL_STRING, "o"),
    ("external_primary", "BioGRID_FPP_vs_raw_ESM", 2.8, "Primary: BioGRID", COL_BIOGRID, "s"),
    ("external_strict", "full_FPP_vs_raw_ESM", 1.2, "Strict: STRING", COL_STRING, "o"),
    ("external_strict", "BioGRID_FPP_vs_raw_ESM", 0.2, "Strict: BioGRID", COL_BIOGRID, "s"),
]
for pop, comp, y, label, color, marker in a_rows:
    point(ax_a, pop, comp, y, color, marker)
ax_a.set_yticks([r[2] for r in a_rows], [r[3] for r in a_rows])
ax_a.set_ylim(-0.55, 4.55)
ax_a.axhline(2.0, color="#d7d7d7", lw=0.6)
style_axis(ax_a, (-0.031, 0.073), [-0.025, 0, 0.025, 0.050])

b_rows = []
for pop, group, positions, color, marker in [
    ("original_fusion", "Original", [5.7, 4.7, 3.7], COL_ORIGINAL, "o"),
    ("homology", "Homology", [2.0, 1.0, 0.0], COL_HOMOLOGY, "s"),
]:
    for (comparator, short), y in zip([
        ("scores_capacity_matched", "scores"),
        ("scores_plus_raw_ESM", "raw"),
        ("scores_plus_full_FPP", "STRING"),
    ], positions):
        key = "scores_plus_BioGRID_FPP_vs_" + comparator
        point(ax_b, pop, key, y, color, marker)
        b_rows.append((y, f"{group} / {short}"))
ax_b.set_yticks([r[0] for r in b_rows], [r[1] for r in b_rows])
ax_b.set_ylim(-0.7, 6.4)
ax_b.axhline(2.85, color="#d7d7d7", lw=0.6)
style_axis(ax_b, (-0.0065, 0.023), [-0.005, 0, 0.010, 0.020])

fig.text(0.013, 0.958, "A  External source: graph − raw ESM", ha="left", va="top", fontsize=9.0, weight="bold")
fig.text(0.528, 0.958, "B  Fusion: BioGRID − comparator", ha="left", va="top", fontsize=9.0, weight="bold")
fig.text(0.5, 0.025,
         "Points: 3-seed ensembles; bars: paired protein-bootstrap 95% intervals (2,000 replicates).\n"
         "Strict is nested within primary. Panels use different x-axis scales.",
         ha="center", va="bottom", fontsize=8.0, color="#333333", linespacing=1.35)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"Figure3_external_validation.{ext}", facecolor="white")
plt.close(fig)
