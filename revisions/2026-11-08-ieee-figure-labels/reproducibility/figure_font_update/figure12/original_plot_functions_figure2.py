def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def f(row, key):
    return float(row[key])

def row_by(rows, **kwargs):
    for row in rows:
        if all(row.get(k) == v for k, v in kwargs.items()):
            return row
    raise KeyError(kwargs)

def panel_a(ax):
    ax.set_title("A  Fusion-control setup", loc="left", fontweight="bold")
    ax.axis("off")
    ax.set_xlim(-0.04, 1.66)
    ax.set_ylim(0, 1)

    def draw_box(cx, cy, w, h, label, color, fontsize=7.1):
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.008",
            fc=color, ec="#4D4D4D", lw=0.8, alpha=0.98, zorder=3
        )
        ax.add_patch(patch)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
                linespacing=1.05, color=COL_TEXT, zorder=4)

    draw_box(0.20, 0.60, 0.35, 0.21, "ClinVar-STRING\nmissense variants\nn=177,213", COL_BASE, fontsize=7.1)
    draw_box(0.55, 0.80, 0.24, 0.17, f"CADD\n{cadd_report['n_matched_keys']:,}\n99.995%", COL_CADDAM, fontsize=7.4)
    draw_box(0.55, 0.24, 0.31, 0.18, f"AlphaMissense\n{am_report['n_matched_keys']:,}\n96.92%", "#A1D99B", fontsize=7.4)
    draw_box(0.90, 0.60, 0.24, 0.20, "CADD+AM\nbaseline", "#FDD0A2", fontsize=7.4)
    draw_box(
        1.34, 0.60, 0.50, 0.34,
        "Representation controls:\nraw ESM\nedge-ablated\npermuted\nreal PPI-GCN",
        COL_GRAPH,
        fontsize=7.0,
    )

    arrows = [
        ((0.375, 0.69), (0.425, 0.77)),
        ((0.375, 0.50), (0.425, 0.34)),
        ((0.675, 0.77), (0.765, 0.66)),
        ((0.715, 0.34), (0.765, 0.54)),
        ((1.025, 0.60), (1.085, 0.60)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9.5,
            color="#4D4D4D", lw=1.0, connectionstyle="arc3,rad=0", zorder=2
        ))

def panel_b(ax):
    ax.set_title("B  Original split: matched fusion controls", loc="left", fontweight="bold")
    labels = ["CADD+AM", "+raw\nESM", "+edge-\nablated", "+topology-\npermuted", "+real\nPPI-GCN"]
    rows = [orig_caddam, orig_raw, orig_edge, orig_perm, orig_real]
    auc = [f(r, "test_auc_mean") for r in rows]
    ap = [f(r, "test_auprc_mean") for r in rows]
    auc_sd = [f(r, "test_auc_sd") for r in rows]
    ap_sd = [f(r, "test_auprc_sd") for r in rows]

    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, auc, w, yerr=auc_sd, color=COL_GRAPH, edgecolor="black",
           linewidth=0.4, capsize=2, label="AUC", zorder=3)
    ax.bar(x + w / 2, ap, w, yerr=ap_sd, color=COL_BASE, edgecolor="black",
           linewidth=0.4, capsize=2, label="AUPRC", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.89, 0.982)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)

def panel_c(ax):
    ax.set_title("C  Real PPI-GCN margin over controls", loc="left", fontweight="bold")
    controls = [
        ("vs CADD+AM\narch-matched", margin_rows["real_vs_archmatched_cadd_am"]),
        ("vs raw ESM\nfusion", margin_rows["real_vs_raw_esm_fusion"]),
        ("vs edge-ablated\nfusion", margin_rows["real_vs_edge_ablated_fusion"]),
        ("vs topology-\npermuted fusion", margin_rows["real_vs_topology_permuted_fusion"]),
    ]
    y = np.arange(len(controls))[::-1]
    auc_delta = [f(r, "delta_auc") for _, r in controls]
    ap_delta = [f(r, "delta_auprc") for _, r in controls]
    auc_low = [f(r, "delta_auc_ci95_low") for _, r in controls]
    auc_high = [f(r, "delta_auc_ci95_high") for _, r in controls]
    ap_low = [f(r, "delta_auprc_ci95_low") for _, r in controls]
    ap_high = [f(r, "delta_auprc_ci95_high") for _, r in controls]
    auc_xerr = np.array([[d - lo for d, lo in zip(auc_delta, auc_low)],
                         [hi - d for d, hi in zip(auc_delta, auc_high)]])
    ap_xerr = np.array([[d - lo for d, lo in zip(ap_delta, ap_low)],
                        [hi - d for d, hi in zip(ap_delta, ap_high)]])
    h = 0.32
    ax.barh(y + h / 2, auc_delta, height=h, color=COL_GRAPH, edgecolor="black",
            linewidth=0.4, label="AUC", zorder=3)
    ax.barh(y - h / 2, ap_delta, height=h, color=COL_BASE, edgecolor="black",
            linewidth=0.4, label="AUPRC", zorder=3)
    ax.errorbar(auc_delta, y + h / 2, xerr=auc_xerr, fmt="none", ecolor=COL_TEXT,
                elinewidth=0.8, capsize=2.2, zorder=4)
    ax.errorbar(ap_delta, y - h / 2, xerr=ap_xerr, fmt="none", ecolor=COL_TEXT,
                elinewidth=0.8, capsize=2.2, zorder=4)
    ax.axvline(0, color=COL_GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in controls])
    ax.set_xlabel("Delta performance")
    ax.set_xlim(-0.004, 0.060)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    clean_axis(ax)

def panel_d(ax):
    ax.set_title("D  Homology-aware split: ESM vs PPI-GCN", loc="left", fontweight="bold")
    labels = ["AUC", "AUPRC"]
    esm = [float(homo["esm_raw_homology"]["test_auc"]), float(homo["esm_raw_homology"]["test_auprc"])]
    gcn = [float(homo["gcn_enhanced_homology"]["test_auc"]), float(homo["gcn_enhanced_homology"]["test_auprc"])]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, esm, w, color=COL_BASE, edgecolor="black", linewidth=0.4, label="ESM")
    ax.bar(x + w / 2, gcn, w, color=COL_GRAPH, edgecolor="black", linewidth=0.4, label="PPI-GCN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.18, 0.80)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)

def panel_e(ax):
    ax.set_title("E  Homology-aware: strong-predictor fusion", loc="left", fontweight="bold")
    labels = ["CADD+AM", "+raw ESM", "+PPI-GCN"]
    base = [f(homo_caddam, "test_auc_mean"), f(homo_caddam, "test_auprc_mean")]
    raw = [f(homo_raw, "test_auc_mean"), f(homo_raw, "test_auprc_mean")]
    graph = [float(homo["gcn_plus_cadd_plus_am_homology"]["test_auc"]),
             float(homo["gcn_plus_cadd_plus_am_homology"]["test_auprc"])]
    rows = [base, raw, graph]
    auc = [r[0] for r in rows]
    ap = [r[1] for r in rows]
    y = np.arange(len(labels))[::-1]
    h = 0.34
    ax.barh(y + h / 2, auc, height=h, color=COL_GRAPH, edgecolor="black",
            linewidth=0.4, label="AUC", zorder=3)
    ax.barh(y - h / 2, ap, height=h, color=COL_BASE, edgecolor="black",
            linewidth=0.4, label="AUPRC", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test performance")
    ax.set_xlim(0.80, 0.985)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

def panel_f(ax):
    ax.set_title("F  Test-set prevalence differs by split", loc="left", fontweight="bold")
    vals = [orig_prev, homo_prev]
    labels = ["Original\nsplit", "Homology-aware\nsplit"]
    ax.bar(np.arange(2), vals, color=[COL_BASE, COL_HOMO], edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v * 100:.2f}%", ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(0, 0.38)
    clean_axis(ax)
