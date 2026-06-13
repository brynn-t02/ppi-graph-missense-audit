from pathlib import Path
import csv
import json
import math
import statistics
from xml.sax.saxutils import escape


ROOT = Path("/Users/teyian/Documents/paper1/server_snapshot_full/root")
OUT_DIR = Path("/Users/teyian/Documents/paper1/figure_drafts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_03F = ROOT / "data/processed/variant_mlp_comparison/summary.json"
P_12 = ROOT / "data/processed/bootstrap_significance/bootstrap_significance_summary.csv"
P_42A = ROOT / "data/processed/clustered_bootstrap_exp42/42A_gcn_vs_esm_clustered_bootstrap.json"
P_41 = ROOT / "data/processed/esm_gcn_edge_ablation_exp41/per_repeat_esm_gcn_edge_ablation_metrics.csv"
P_44 = ROOT / "data/processed/esm_gcn_partial_edge_ablation_exp44/edge_ablation_curve_summary.csv"
P_45 = ROOT / "data/processed/esm_gcn_topology_permutation_exp45/summary.json"

W, H = 1500, 940

COL = {
    "esm": "#9ECAE1",
    "gcn": "#3182BD",
    "edge": "#FDD0A2",
    "perm": "#BDBDBD",
    "random": "#636363",
    "high": "#DE2D26",
    "low": "#31A354",
    "purple": "#756BB1",
    "purple_light": "#BCBDDC",
    "orange": "#FDAE6B",
    "ink": "#252525",
    "grid": "#D9D9D9",
    "axis": "#555555",
}


def load_json(path):
    return json.loads(path.read_text())


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(vals):
    vals = [float(v) for v in vals]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


d03 = load_json(P_03F)
ESM_AUC = float(d03["esm_raw"]["test_auc"])
ESM_AUPRC = float(d03["esm_raw"]["test_auprc"])
GCN_AUC = float(d03["gcn_enhanced"]["test_auc"])
GCN_AUPRC = float(d03["gcn_enhanced"]["test_auprc"])

rows12 = load_csv(P_12)


def boot12(metric):
    for row in rows12:
        if row["comparison"] == "gcn_vs_esm" and row["metric"] == metric:
            return float(row["mean_delta"]), float(row["ci95_low"]), float(row["ci95_high"])
    raise KeyError(metric)


VAR_AUC = boot12("delta_auc")
VAR_AUPRC = boot12("delta_auprc")

d42 = load_json(P_42A)
PROT_AUC = (
    float(d42["point"]["delta_auc"]),
    float(d42["clustered_bootstrap"]["delta_auc_ci95"][0]),
    float(d42["clustered_bootstrap"]["delta_auc_ci95"][1]),
)
PROT_AUPRC = (
    float(d42["point"]["delta_auprc"]),
    float(d42["clustered_bootstrap"]["delta_auprc_ci95"][0]),
    float(d42["clustered_bootstrap"]["delta_auprc_ci95"][1]),
)

rows41 = load_csv(P_41)
EDGE_AUC_MEAN, EDGE_AUC_SD = mean_sd([r["test_auc"] for r in rows41])
EDGE_AUPRC_MEAN, EDGE_AUPRC_SD = mean_sd([r["test_auprc"] for r in rows41])

rows44 = load_csv(P_44)

d45 = load_json(P_45)
m45 = d45["summary_metrics"]
PERM_AUC_MEAN = float(m45["test_auc_mean"])
PERM_AUC_SD = float(m45["test_auc_sd"])
PERM_AUPRC_MEAN = float(m45["test_auprc_mean"])
PERM_AUPRC_SD = float(m45["test_auprc_sd"])


parts = []


def add(s):
    parts.append(s)


def text(x, y, value, size=14, weight="normal", fill=None, anchor="start", extra=""):
    fill = fill or COL["ink"]
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" {extra}>'
        f"{escape(str(value))}</text>"
    )


def line(x1, y1, x2, y2, stroke=None, width=1.2, dash=None):
    stroke = stroke or COL["axis"]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')


def rect(x, y, w, h, fill="white", stroke=None, width=1, rx=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{stroke_attr}/>')


def circle(x, y, r, fill, stroke="white", width=1.2):
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def polyline(points, stroke, width=1.5, fill="none", dash=None):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')


def path(d, stroke, width=1.5, fill="none", dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')


def panel_title(x, y, title):
    text(x, y, title, size=18, weight="bold")


def map_linear(v, vmin, vmax, omin, omax):
    return omin + (v - vmin) / (vmax - vmin) * (omax - omin)


def axes_box(x, y, w, h, y_min, y_max, y_ticks, x_labels):
    line(x, y + h, x + w, y + h, COL["axis"], 1.2)
    line(x, y, x, y + h, COL["axis"], 1.2)
    for t in y_ticks:
        yy = map_linear(t, y_min, y_max, y + h, y)
        line(x - 4, yy, x, yy, COL["axis"], 1)
        line(x, yy, x + w, yy, COL["grid"], 0.7)
        text(x - 8, yy + 4, f"{t:.2f}", size=10, anchor="end", fill="#555")
    for i, lab in enumerate(x_labels):
        xx = x + (i + 0.5) * w / len(x_labels)
        text(xx, y + h + 20, lab, size=12, anchor="middle")


def bar(x, base, bw, val, y_min, y_max, plot_y, plot_h, fill, err=0):
    yy = map_linear(val, y_min, y_max, plot_y + plot_h, plot_y)
    h = plot_y + plot_h - yy
    rect(x, yy, bw, h, fill=fill, stroke="#222", width=0.8)
    if err:
        ylo = map_linear(val - err, y_min, y_max, plot_y + plot_h, plot_y)
        yhi = map_linear(val + err, y_min, y_max, plot_y + plot_h, plot_y)
        cx = x + bw / 2
        line(cx, ylo, cx, yhi, "#222", 1)
        line(cx - 5, ylo, cx + 5, ylo, "#222", 1)
        line(cx - 5, yhi, cx + 5, yhi, "#222", 1)


def legend(x, y, items, size=11):
    for i, (label, color) in enumerate(items):
        yy = y + i * 18
        rect(x, yy - 9, 12, 12, fill=color, stroke="#222", width=0.5)
        text(x + 18, yy + 1, label, size=size)


def panel_a(x, y, w, h):
    panel_title(x, y, "A  Protein-level split on a shared PPI graph")
    ox, oy = x, y + 18
    train = [(ox + 55, oy + 88), (ox + 158, oy + 55), (ox + 204, oy + 137), (ox + 90, oy + 184)]
    test = [(ox + 318, oy + 88), (ox + 430, oy + 55), (ox + 425, oy + 176), (ox + 280, oy + 190)]
    val = [(ox + 205, oy + 250), (ox + 270, oy + 270)]
    for a, b in [(train[0], train[1]), (train[1], train[2]), (train[2], train[3]), (train[0], train[3]),
                 (test[0], test[1]), (test[1], test[2]), (test[2], test[3]), (test[0], test[3]),
                 (val[0], train[3]), (val[1], test[3])]:
        line(*a, *b, "#AFAFAF", 1.5)
    line(*train[1], *test[0], COL["high"], 3)
    line(*train[2], *test[3], COL["high"], 3)
    for p in train:
        circle(*p, 15, COL["gcn"])
    for p in test:
        circle(*p, 15, COL["low"])
    for p in val:
        circle(*p, 14, COL["orange"])
    text(ox + 10, oy + 20, "Train proteins", size=13, weight="bold", fill=COL["gcn"])
    text(ox + 315, oy + 20, "Test proteins", size=13, weight="bold", fill=COL["low"])
    text(ox + 176, oy + 292, "Validation", size=13, weight="bold", fill="#A65300")
    text(ox + 245, oy + 118, "direct train-test", size=11, fill=COL["high"], anchor="middle")
    text(ox + 245, oy + 133, "PPI links", size=11, fill=COL["high"], anchor="middle")
    by = y + h - 45
    labels = [("ESM\\nfeatures", ox + 75), ("PPI\\npropagation", ox + 235), ("MLP\\noutput", ox + 395)]
    for label, bx in labels:
        rect(bx - 50, by - 25, 100, 45, fill="white", stroke="#555", width=1.1, rx=6)
        a, b = label.split("\\n")
        text(bx, by - 5, a, size=11, anchor="middle")
        text(bx, by + 10, b, size=11, anchor="middle")
    text(ox + 151, by + 6, "->", size=18, anchor="middle")
    text(ox + 314, by + 6, "->", size=18, anchor="middle")


def panel_b(x, y, w, h):
    panel_title(x, y, "B  Full-graph PPI-GCN point gains")
    px, py, pw, ph = x + 48, y + 46, w - 75, h - 100
    y_min, y_max = 0.55, 0.84
    axes_box(px, py, pw, ph, y_min, y_max, [0.55, 0.65, 0.75], ["AUC", "AUPRC"])
    vals = [[ESM_AUC, GCN_AUC], [ESM_AUPRC, GCN_AUPRC]]
    for i, pair in enumerate(vals):
        cx = px + (i + 0.5) * pw / 2
        bar(cx - 38, py + ph, 28, pair[0], y_min, y_max, py, ph, COL["esm"])
        bar(cx + 8, py + ph, 28, pair[1], y_min, y_max, py, ph, COL["gcn"])
    text(px + pw * 0.25, py + 33, f"Delta={GCN_AUC - ESM_AUC:+.4f}", size=11, anchor="middle")
    text(px + pw * 0.75, py + 97, f"Delta={GCN_AUPRC - ESM_AUPRC:+.4f}", size=11, anchor="middle")
    text(px - 42, py + ph / 2, "Test performance", size=12, anchor="middle", extra='transform="rotate(-90 %.1f %.1f)"' % (px - 42, py + ph / 2))
    legend(px + pw - 105, py + 18, [("ESM raw", COL["esm"]), ("Full PPI-GCN", COL["gcn"])])


def panel_c(x, y, w, h):
    panel_title(x, y, "C  Variant vs protein-cluster bootstrap")
    px, py, pw, ph = x + 165, y + 48, w - 205, h - 95
    x_min, x_max = -0.02, 0.085
    line(px, py + ph, px + pw, py + ph, COL["axis"], 1.2)
    zero = map_linear(0, x_min, x_max, px, px + pw)
    line(zero, py, zero, py + ph, COL["axis"], 1, dash="4 4")
    for t in [-0.02, 0, 0.02, 0.04, 0.06, 0.08]:
        xx = map_linear(t, x_min, x_max, px, px + pw)
        line(xx, py + ph, xx, py + ph + 4, COL["axis"], 1)
        text(xx, py + ph + 20, f"{t:.2f}", size=10, anchor="middle")
    rows = [
        ("AUC, variant-level", VAR_AUC, COL["purple"]),
        ("AUC, protein-clustered", PROT_AUC, COL["purple_light"]),
        ("AUPRC, variant-level", VAR_AUPRC, COL["purple"]),
        ("AUPRC, protein-clustered", PROT_AUPRC, COL["purple_light"]),
    ]
    for i, (label, (mean, lo, hi), color) in enumerate(rows):
        yy = py + 24 + i * 43
        text(px - 12, yy + 4, label, size=11, anchor="end")
        xlo = map_linear(lo, x_min, x_max, px, px + pw)
        xhi = map_linear(hi, x_min, x_max, px, px + pw)
        xm = map_linear(mean, x_min, x_max, px, px + pw)
        line(xlo, yy, xhi, yy, color, 3)
        line(xlo, yy - 6, xlo, yy + 6, color, 2)
        line(xhi, yy - 6, xhi, yy + 6, color, 2)
        circle(xm, yy, 5, color, stroke="#222", width=0.5)
    text(px + pw / 2, py + ph + 45, "Delta performance (PPI-GCN - ESM)", size=12, anchor="middle")


def grouped_panel(x, y, w, h, title, groups, note=None):
    panel_title(x, y, title)
    px, py, pw, ph = x + 50, y + 48, w - 78, h - 125
    y_min, y_max = 0.56, 0.83
    axes_box(px, py, pw, ph, y_min, y_max, [0.60, 0.70, 0.80], ["AUC", "AUPRC"])
    bw = 23
    offsets = [-38, -8, 22]
    for gi, (_name, vals, errs, color) in enumerate(groups):
        for mi, val in enumerate(vals):
            cx = px + (mi + 0.5) * pw / 2
            bar(cx + offsets[gi], py + ph, bw, val, y_min, y_max, py, ph, color, errs[mi])
    text(px - 42, py + ph / 2, "Test performance", size=12, anchor="middle", extra='transform="rotate(-90 %.1f %.1f)"' % (px - 42, py + ph / 2))
    legend(px + 25, py + ph + 47, [(g[0], g[3]) for g in groups], size=10)
    if note:
        text(px + pw - 38, py + 26, note[0], size=11, fill="#555", anchor="middle")
        text(px + pw - 38, py + 41, note[1], size=11, fill="#555", anchor="middle")


def panel_e(x, y, w, h):
    panel_title(x, y, "E  Partial train-test edge ablation")
    px, py, pw, ph = x + 60, y + 48, w - 95, h - 100
    x_min, x_max = 0, 100
    y_min, y_max = -0.010, 0.026
    line(px, py + ph, px + pw, py + ph, COL["axis"], 1.2)
    line(px, py, px, py + ph, COL["axis"], 1.2)
    for yt in [-0.01, 0.00, 0.01, 0.02]:
        yy = map_linear(yt, y_min, y_max, py + ph, py)
        line(px - 4, yy, px, yy, COL["axis"], 1)
        line(px, yy, px + pw, yy, COL["grid"], 0.7)
        text(px - 8, yy + 4, f"{yt:.2f}", size=10, anchor="end")
    for xt in [0, 25, 50, 75, 100]:
        xx = map_linear(xt, x_min, x_max, px, px + pw)
        line(xx, py + ph, xx, py + ph + 4, COL["axis"], 1)
        text(xx, py + ph + 20, str(xt), size=10, anchor="middle")
    yzero = map_linear(0, y_min, y_max, py + ph, py)
    line(px, yzero, px + pw, yzero, COL["axis"], 1, dash="4 4")
    styles = {
        "random_pair_removal": ("Random removal", COL["random"]),
        "high_confidence_first": ("High-confidence first", COL["high"]),
        "low_confidence_first": ("Low-confidence first", COL["gcn"]),
    }
    for strategy, (label, color) in styles.items():
        sub = sorted([r for r in rows44 if r["strategy"] == strategy], key=lambda r: float(r["fraction_removed"]))
        pts = []
        for r in sub:
            xx = map_linear(float(r["fraction_removed"]) * 100, x_min, x_max, px, px + pw)
            yy = map_linear(float(r["delta_test_auc_vs_raw_esm"]), y_min, y_max, py + ph, py)
            pts.append((xx, yy))
            circle(xx, yy, 4, color, stroke="#222", width=0.5)
        polyline(pts, color, 2.2)
    text(px + pw / 2, py + ph + 45, "Removed direct train-test PPI pairs (%)", size=12, anchor="middle")
    text(px - 44, py + ph / 2, "Delta AUC vs ESM raw", size=12, anchor="middle", extra='transform="rotate(-90 %.1f %.1f)"' % (px - 44, py + ph / 2))
    legend(px + pw - 145, py + 14, [(v[0], v[1]) for v in styles.values()], size=10)


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
rect(0, 0, W, H, fill="white")
text(W / 2, 34, "PPI graph augmentation is sensitive to protein-topology correspondence and train-test connectivity", size=22, weight="bold", anchor="middle")

panel_w, panel_h = 455, 385
xs = [42, 522, 1002]
ys = [70, 505]
for yy in ys:
    for xx in xs:
        rect(xx - 18, yy - 30, panel_w + 24, panel_h + 34, fill="#FFFFFF", stroke="#E6E6E6", width=0.8, rx=4)

panel_a(xs[0], ys[0], panel_w, panel_h)
panel_b(xs[1], ys[0], panel_w, panel_h)
panel_c(xs[2], ys[0], panel_w, panel_h)
grouped_panel(
    xs[0], ys[1], panel_w, panel_h,
    "D  Removing direct train-test PPI edges",
    [
        ("ESM raw", [ESM_AUC, ESM_AUPRC], [0, 0], COL["esm"]),
        ("Full PPI-GCN", [GCN_AUC, GCN_AUPRC], [0, 0], COL["gcn"]),
        ("Edge-ablated", [EDGE_AUC_MEAN, EDGE_AUPRC_MEAN], [EDGE_AUC_SD, EDGE_AUPRC_SD], COL["edge"]),
    ],
    note=("AUC gain", "removed"),
)
panel_e(xs[1], ys[1], panel_w, panel_h)
grouped_panel(
    xs[2], ys[1], panel_w, panel_h,
    "F  Topology permutation control",
    [
        ("ESM raw", [ESM_AUC, ESM_AUPRC], [0, 0], COL["esm"]),
        ("Real PPI-GCN", [GCN_AUC, GCN_AUPRC], [0, 0], COL["gcn"]),
        ("Topology-permuted", [PERM_AUC_MEAN, PERM_AUPRC_MEAN], [PERM_AUC_SD, PERM_AUPRC_SD], COL["perm"]),
    ],
    note=("returns to", "ESM level"),
)

add("</svg>")

out = OUT_DIR / "Figure1_connectivity_topology_controls_v2.svg"
out.write_text("\n".join(parts), encoding="utf-8")
print(out)
