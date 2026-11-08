#!/usr/bin/env python3
"""Vector redraw of Figure 4 from this revision's recomputed matched ensembles.

This script reads only the current analysis directory for numerical inputs.
The accepted Figure 4 supplies its physical page dimensions and six axes
positions, but none of its old numerical values are copied into this figure.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ANALYSIS = HERE.parent / "analysis"
BASE = ROOT / "outputs/BIBM_workshop_targeted_candidate_v3_spacing_20261108/source"
DEST = ROOT / "outputs/BIBM_workshop_targeted_candidate_v4_figure4_20261108/source/figure_drafts"
BASE_PDF = BASE / "figure_drafts/Figure4_prediction_transitions_v2_accepted_minimal_clarified.pdf"
STEM = "Figure4_prediction_transitions_matched_ensembles"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplconfig"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pypdf import PdfReader
import pdfplumber

FONT_DIR = BASE / "resources/fonts"
for name in ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]:
    font_manager.fontManager.addfont(str(FONT_DIR / name))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0,
    "axes.titlesize": 9.0, "axes.labelsize": 8.3,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0, "figure.dpi": 160, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8,
})

COL_CORRECT = "#9ECAE1"
COL_WRONG = "#BDBDBD"
COL_IMPROVED = "#3182BD"
COL_DEGRADED = "#D95F0E"
COL_FN = "#2171B5"
COL_FP = "#F16913"
COL_GRAY = "#7A7A7A"
COL_TEXT = "#252525"

FEATURES = [
    "degree_bucket", "kcore_bucket", "neighbor_risk_bucket",
    "neighbor_pathogenic_rate", "neighbor_train_variant_count",
    "betweenness_bucket", "clustering_coefficient", "boundary_role_bucket",
    "within_comm_neighbor_fraction",
]
FEATURE_LABELS = {
    "degree_bucket": "degree", "kcore_bucket": "k-core",
    "neighbor_risk_bucket": "neighbour risk",
    "neighbor_pathogenic_rate": "neighbour pathogenic rate",
    "neighbor_train_variant_count": "neighbour train variants",
    "betweenness_bucket": "betweenness",
    "clustering_coefficient": "clustering coefficient",
    "boundary_role_bucket": "boundary role",
    "within_comm_neighbor_fraction": "within-community fraction",
}
KEEP_F = [
    "degree_bucket", "kcore_bucket", "neighbor_risk_bucket",
    "neighbor_pathogenic_rate", "betweenness_bucket", "boundary_role_bucket",
    "within_comm_neighbor_fraction",
]

# Exact accepted-PDF axes boxes, in PDF points. No text or graphic is scaled.
AXES_BOXES_PT = [
    [110.809375, 307.488323, 329.338155, 475.097625],
    [449.528985, 307.488323, 668.057765, 475.097625],
    [788.248595, 307.488323, 1006.777380, 475.097625],
    [110.809375, 42.665625, 329.338155, 210.274927],
    [449.528985, 42.665625, 668.057765, 210.274927],
    [788.248595, 42.665625, 1006.777380, 210.274927],
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs():
    data = ANALYSIS / "data/processed"
    files = {
        "summary": data / "corrected_sample_identification_exp34/summary.json",
        "esm_profile": data / "corrected_sample_profile_exp35/esm_to_gcn_corrected_profile_summary.csv",
        "full_profile": data / "corrected_sample_profile_exp35/cadd_am_to_full_corrected_profile_summary.csv",
        "esm_stats": data / "fp_vs_fn_stats_exp36B/esm_to_gcn_fp_vs_fn_stats.csv",
        "full_stats": data / "fp_vs_fn_stats_exp36B/cadd_am_to_full_fp_vs_fn_stats.csv",
        "contrast": data / "error_stratified_correction_contrast_exp40/error_stratified_correction_contrast_results.csv",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Current recomputation input is not ready: {path}")
    values = {k: (json.loads(v.read_text()) if v.suffix == ".json" else pd.read_csv(v))
              for k, v in files.items()}
    return files, values


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def transition_panel(ax, title, data, panel):
    ax.set_title(title, loc="left", fontweight="bold")
    n = int(data["n"])
    fields = ["count_both_correct", "count_both_wrong", "count_improved", "count_degraded"]
    colors = [COL_CORRECT, COL_WRONG, COL_IMPROVED, COL_DEGRADED]
    assert sum(int(data[f]) for f in fields) == n
    assert int(data["count_FN_rescued"]) + int(data["count_FP_suppressed"]) == int(data["count_improved"])
    assert int(data["count_TN_to_FP"]) + int(data["count_TP_to_FN"]) == int(data["count_degraded"])
    left = 0.0
    records = []
    for field, color in zip(fields, colors):
        count = int(data[field])
        fraction = count / n
        ax.barh([0.18], [fraction], left=left, height=0.28, color=color,
                edgecolor="black", linewidth=0.35)
        center = left + fraction / 2
        if fraction > 0.10:
            ax.text(center, 0.18, f"{count:,}", ha="center", va="center",
                    fontsize=7.0, color=COL_TEXT)
            placement = "inside segment"
        else:
            placement = "not labeled in narrow segment; corrected/degraded totals in existing subdivision lines"
        records.append({"field": field, "count": count, "fraction": fraction,
                        "color": color, "left": left, "label_placement": placement})
        left += fraction
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.58, 0.72)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of test variants")
    ax.text(0.02, -0.18,
            f"Corrected {int(data['count_improved']):,}: FN rescued {int(data['count_FN_rescued']):,}; "
            f"FP suppressed {int(data['count_FP_suppressed']):,}",
            ha="left", va="center", fontsize=7.0, color=COL_FN)
    ax.text(0.02, -0.39,
            f"Degraded {int(data['count_degraded']):,}: TN to FP {int(data['count_TN_to_FP']):,}; "
            f"TP to FN {int(data['count_TP_to_FN']):,}",
            ha="left", va="center", fontsize=7.0, color=COL_DEGRADED)
    clean_axis(ax)
    return {"panel": panel, "n": n, "segments": records,
            "error_transition_counts": {k: int(data[k]) for k in
                ["count_FN_rescued", "count_FP_suppressed", "count_TN_to_FP", "count_TP_to_FN"]}}


def profile_data(df):
    if "group_type" in df:
        df = df[df.group_type == "error_transition"]
    rows = df.set_index("group_name")
    mat = []
    records = []
    for feat in FEATURES:
        vals = np.array([rows.loc["FP_suppressed", f"mean_{feat}"],
                         rows.loc["FN_rescued", f"mean_{feat}"]], dtype=float)
        assert np.isfinite(vals).all()
        scaled = np.zeros_like(vals) if np.isclose(vals.max(), vals.min()) else (vals - vals.mean()) / (vals.max() - vals.min())
        # With two groups the two nonconstant values are mathematically +/-0.5.
        # Remove floating-point roundoff before the discrete colormap lookup.
        scaled = np.round(scaled, 12)
        mat.append(scaled)
        records.append({"feature": feat, "FP_suppressed_mean": float(vals[0]),
                        "FN_rescued_mean": float(vals[1]), "scaled_values": scaled.tolist()})
    return np.array(mat), records


def heatmap_panel(ax, title, df, panel):
    ax.set_title(title, loc="left", fontweight="bold")
    mat, records = profile_data(df)
    # pcolormesh keeps every cell vector-based rather than embedding an image.
    ax.pcolormesh(np.arange(3) - 0.5, np.arange(10) - 0.5, mat,
                  cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="flat",
                  edgecolors="face", linewidth=0.05, antialiased=False, rasterized=False)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FP suppressed", "FN rescued"], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(FEATURES)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.tick_params(length=0)
    return {"panel": panel, "rows": records, "column_order": ["FP_suppressed", "FN_rescued"],
            "scaling": "(value - within-row mean) / within-row range; np.isclose constant rows set to zero; 12-decimal display stabilization",
            "colormap": "RdBu_r", "color_limits": [-1.0, 1.0], "vector_cells": True}


def mean_direction_panel(ax, values):
    ax.set_title("E  Mean topology differences between corrected groups", loc="left", fontweight="bold")
    output = []
    for j, key in enumerate(["esm_stats", "full_stats"]):
        stats = values[key].set_index("feature")
        _, profile = profile_data(values["esm_profile" if j == 0 else "full_profile"])
        raw = {r["feature"]: r for r in profile}
        for yi, feat in enumerate(FEATURES[::-1]):
            difference = float(stats.loc[feat, "diff_fp_minus_fn"])
            expected = raw[feat]["FP_suppressed_mean"] - raw[feat]["FN_rescued_mean"]
            assert np.isclose(difference, expected, rtol=1e-10, atol=1e-12)
            color = COL_FN if difference < 0 else COL_FP if difference > 0 else COL_GRAY
            ax.scatter(j, yi, s=64, color=color, alpha=0.85, edgecolor="black", linewidth=0.25)
            output.append({"comparison": key, "feature": feat, "diff_fp_minus_fn": difference,
                           "mean_higher": "FN_rescued" if difference < 0 else "FP_suppressed" if difference > 0 else "equal",
                           "color": color, "dot_area": 64, "opacity": 0.85})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ESM to\nPPI-GCN", "CADD+AM to\ngraph fusion"])
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURES[::-1]])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.7, len(FEATURES) - 0.3)
    clean_axis(ax)
    return {"panel": "E", "points": output, "significance_encoding": False}


def effect_panel(ax, contrast):
    ax.set_title("F  Rescued versus persistent FN profiles", loc="left", fontweight="bold")
    df = contrast[contrast.condition == "FN_rescued_vs_persistent_FN"].copy()
    if "model_pair" in df:
        pairs = df.model_pair.unique().tolist()
        # The analysis producer uses the same pair key as the transition summary.
        if "esm_vs_gcn" not in pairs:
            raise AssertionError(f"Expected raw-ESM pair esm_vs_gcn; found {pairs}")
        df = df[df.model_pair == "esm_vs_gcn"]
    df = df.set_index("feature").loc[KEEP_F].reset_index()
    assert len(df) == 7 and np.isfinite(df.hedges_g).all()
    y = np.arange(len(df))[::-1]
    ax.barh(y, df.hedges_g, color=[COL_FN if g > 0 else COL_DEGRADED for g in df.hedges_g],
            edgecolor="black", linewidth=0.35)
    ax.axvline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in df.feature])
    ax.set_xlabel("Hedges' g\nrescued FN vs persistent FN")
    ax.set_xlim(min(-1.05, float(df.hedges_g.min()) - 0.08),
                max(1.12, float(df.hedges_g.max()) + 0.08))
    # Original accepted figure used quarter-unit ticks. Keep that density.
    ax.set_xticks(np.arange(-1, 1.001, 0.25))
    clean_axis(ax)
    return {"panel": "F", "contrast": "FN_rescued_vs_persistent_FN", "baseline_stratum": "raw ESM false negatives",
            "feature_order": KEEP_F, "data": json.loads(df.to_json(orient="records", double_precision=15)),
            "positive_color": COL_FN, "negative_color": COL_DEGRADED,
            "significance_encoding": False, "difficulty_matched_claim": False}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    inputs, values = read_inputs()
    base_page = PdfReader(BASE_PDF).pages[0]
    width, height = float(base_page.mediabox.width), float(base_page.mediabox.height)
    assert [width, height] == [1094.88609, 541.265625]
    fig = plt.figure(figsize=(width / 72, height / 72))
    axes = []
    for x0, y0, x1, y1 in AXES_BOXES_PT:
        axes.append(fig.add_axes([x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]))
    fig.text(531.865375 / width, (height - 7.2) / height,
             "Matched-ensemble prediction transitions and topology profiles",
             fontsize=13.5, fontweight="bold", ha="center", va="top")
    panels = {}
    panels["A"] = transition_panel(axes[0], "A  Raw ESM to PPI-GCN transitions", values["summary"]["esm_vs_gcn"], "A")
    panels["B"] = transition_panel(axes[1], "B  CADD+AM to graph-fusion transitions", values["summary"]["cadd_am_vs_full"], "B")
    panels["C"] = heatmap_panel(axes[2], "C  Raw ESM correction profiles", values["esm_profile"], "C")
    panels["D"] = heatmap_panel(axes[3], "D  Graph-fusion correction profiles", values["full_profile"], "D")
    panels["E"] = mean_direction_panel(axes[4], values)
    panels["F"] = effect_panel(axes[5], values["contrast"])

    output = HERE / f"{STEM}.pdf"
    fig.savefig(output, metadata={
        "Title": "Figure 4: matched-ensemble prediction transitions and topology profiles",
        "Subject": "Float64 probability ensembles from matched seeds 42, 43 and 44 at a fixed 0.5 threshold; descriptive topology contrasts.",
        "Creator": "Matplotlib; source-controlled Figure 4 redraw", "CreationDate": None, "ModDate": None,
    })
    plt.close(fig)
    final = PdfReader(output).pages[0]
    assert [float(final.mediabox.width), float(final.mediabox.height)] == [width, height]
    poppler = Path("/Users/teyian/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin")
    subprocess.run([str(poppler / "pdftoppm"), "-singlefile", "-r", "300", "-png", str(output), str(output.with_suffix(""))], check=True)
    font_report = subprocess.run([str(poppler / "pdffonts"), str(output)], capture_output=True, text=True, check=True).stdout
    (HERE / "font_embedding_audit.txt").write_text(font_report)
    assert not any(" no " in line for line in font_report.splitlines()[2:]), font_report
    assert "Helvetica" not in font_report, font_report
    # There are no image XObjects: every plot mark/cell and all text are vector.
    xobjects = final.get("/Resources").get_object().get("/XObject", {})
    if hasattr(xobjects, "get_object"):
        xobjects = xobjects.get_object()
    assert all(v.get_object().get("/Subtype") != "/Image" for v in xobjects.values())

    with pdfplumber.open(output) as doc:
        page = doc.pages[0]
        text = page.extract_text()
        font_sizes = sorted(set(round(float(c["size"]), 5) for c in page.chars if c["upright"]))
        font_names = sorted(set(c["fontname"] for c in page.chars))
    assert "Difficulty-matched" not in text and "Topology features separate" not in text
    assert text.count("persistent FN") >= 2
    audit = {
        "input_files": {k: {"path": str(v.relative_to(ROOT)), "bytes": v.stat().st_size, "sha256": sha(v)} for k, v in inputs.items()},
        "layout_source": {"path": str(BASE_PDF.relative_to(ROOT)), "sha256": sha(BASE_PDF)},
        "script_sha256": sha(__file__), "canvas_pt": [width, height], "axes_boxes_pt": AXES_BOXES_PT,
        "layout": "A/B/C top row; D/E/F lower row; exact accepted Figure 4 physical coordinates",
        "font_family": "DejaVu Sans", "nominal_font_sizes_pt": {
            "suptitle": 13.5, "panel_titles": 9.0, "axis_labels": 8.3,
            "x_tick_labels": 7.6, "y_tick_labels": 7.5, "counts_and_subdivisions": 7.0},
        "font_files": {n: sha(FONT_DIR / n) for n in ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]},
        "pdf_font_names": font_names, "pdf_upright_effective_font_sizes_pt": font_sizes,
        "all_pdf_font_resources_embedded": True, "unembedded_Helvetica": False,
        "image_xobjects": 0, "all_figure_marks_vector": True, "preview_dpi": 300,
        "panels": panels, "output": {"pdf": {"path": str(output), "sha256": sha(output)},
                                      "png": {"path": str(output.with_suffix('.png')), "sha256": sha(output.with_suffix('.png'))}},
    }
    (HERE / "panel_data.json").write_text(json.dumps(panels, indent=2, ensure_ascii=False) + "\n")
    (HERE / "figure4_render_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    DEST.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        shutil.copyfile(output.with_suffix(suffix), DEST / f"{STEM}{suffix}")
    print(json.dumps({"pdf": str(DEST / f"{STEM}.pdf"), "png": str(DEST / f"{STEM}.png"),
                      "audit": str(HERE / "figure4_render_audit.json"), "canvas_pt": [width, height]}, indent=2))


if __name__ == "__main__":
    main()
