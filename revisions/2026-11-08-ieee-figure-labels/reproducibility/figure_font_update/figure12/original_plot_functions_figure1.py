def load_json(path):
    with open(path) as f:
        return json.load(f)

def exp12(metric):
    row = df12[(df12["comparison"] == "gcn_vs_esm") & (df12["metric"] == metric)].iloc[0]
    return float(row["mean_delta"]), float(row["ci95_low"]), float(row["ci95_high"])

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def panel_a(ax):
    ax.set_title("A  Protein-level split on a shared PPI graph", loc="left", fontweight="bold")
    ax.axis("off")
    train = np.array([[0.12, 0.62], [0.25, 0.75], [0.33, 0.53], [0.18, 0.37]])
    test = np.array([[0.70, 0.62], [0.84, 0.75], [0.82, 0.46], [0.62, 0.37]])
    val = np.array([[0.42, 0.20], [0.52, 0.28]])

    def edge(a, b, color="#BDBDBD", lw=1.1):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=1)

    for a, b in [(train[0], train[1]), (train[1], train[2]), (train[2], train[3]), (train[0], train[3])]:
        edge(a, b)
    for a, b in [(test[0], test[1]), (test[1], test[2]), (test[2], test[3]), (test[0], test[3])]:
        edge(a, b)
    edge(val[0], train[3])
    edge(val[1], test[3])
    edge(train[1], test[0], COL_HIGH, 2.2)
    edge(train[2], test[3], COL_HIGH, 2.2)

    ax.scatter(train[:, 0], train[:, 1], s=92, color=COL_GCN, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(test[:, 0], test[:, 1], s=92, color=COL_LOW, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(val[:, 0], val[:, 1], s=82, color=COL_ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(0.42, 0.10, "Validation", color="#B15928", fontweight="bold", ha="center")
    ax.text(0.45, 0.61, "direct train-test\nPPI links", color=COL_HIGH, ha="center", va="center", fontsize=7.2)

    legend_y = 0.88
    legend_items = [(0.10, COL_GCN, "train"), (0.26, COL_ORANGE, "validation"), (0.48, COL_LOW, "test")]
    for lx, color, label in legend_items:
        ax.scatter([lx], [legend_y], s=42, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(lx + 0.035, legend_y, label, va="center", fontsize=7.3, color="#4D4D4D")

    y = 0.02
    boxes = [(0.12, "ESM\nfeatures"), (0.42, "PPI\npropagation"), (0.72, "MLP\noutput")]
    for x, label in boxes:
        ax.text(x, y, label, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#4D4D4D", lw=0.8))
    ax.add_patch(FancyArrowPatch((0.22, y), (0.31, y), arrowstyle="->", mutation_scale=9, color="#4D4D4D"))
    ax.add_patch(FancyArrowPatch((0.53, y), (0.62, y), arrowstyle="->", mutation_scale=9, color="#4D4D4D"))

def panel_b(ax):
    ax.set_title("B  Full-graph PPI-GCN point gains", loc="left", fontweight="bold")
    metrics = ["AUC", "AUPRC"]
    esm = [ESM_AUC, ESM_AUPRC]
    gcn = [GCN_AUC, GCN_AUPRC]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, esm, w, color=COL_ESM, edgecolor="black", linewidth=0.45, label="ESM raw")
    ax.bar(x + w / 2, gcn, w, color=COL_GCN, edgecolor="black", linewidth=0.45, label="Full PPI-GCN")
    ax.text(0, 0.812, f"Delta={GCN_AUC-ESM_AUC:+.4f}", ha="center", va="bottom", fontsize=7.1)
    ax.text(1, 0.664, f"Delta={GCN_AUPRC-ESM_AUPRC:+.4f}", ha="center", va="bottom", fontsize=7.1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.55, 0.84)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)

def panel_c(ax):
    ax.set_title("C  Variant vs protein-cluster bootstrap", loc="left", fontweight="bold")
    rows = [
        ("AUC\nvariant", VAR_AUC, COL_PURPLE),
        ("AUC\nprotein-clustered", PROT_AUC, COL_PURPLE_LIGHT),
        ("AUPRC\nvariant", VAR_AUPRC, COL_PURPLE),
        ("AUPRC\nprotein-clustered", PROT_AUPRC, COL_PURPLE_LIGHT),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (label, (mean, lo, hi), color) in zip(y, rows):
        ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color,
                    ecolor=color, elinewidth=1.7, capsize=3, markersize=4.5)
    ax.axvline(0, color="#4D4D4D", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Delta performance")
    ax.set_xlim(-0.02, 0.085)
    ax.set_ylim(-0.6, 3.6)
    clean_axis(ax)

def grouped_metric_panel(ax, title, groups, legend_loc="upper center"):
    ax.set_title(title, loc="left", fontweight="bold")
    metrics = ["AUC", "AUPRC"]
    centers = np.arange(2)
    w = 0.24
    offsets = np.linspace(-w, w, len(groups))
    for off, (name, vals, errs, color) in zip(offsets, groups):
        ax.bar(centers + off, vals, width=w, yerr=errs, color=color, edgecolor="black",
               linewidth=0.45, capsize=2.2, label=name)
    ax.set_xticks(centers)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.56, 0.83)
    if legend_loc == "inside":
        ax.legend(frameon=False, loc="upper right", fontsize=7.0)
    elif legend_loc == "above":
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7.0)
    else:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    clean_axis(ax)

def panel_d(ax):
    grouped_metric_panel(
        ax,
        "D  Removing direct train-test PPI edges",
        [
            ("ESM raw", [ESM_AUC, ESM_AUPRC], [0, 0], COL_ESM),
            ("Full PPI-GCN", [GCN_AUC, GCN_AUPRC], [0, 0], COL_GCN),
            ("Edge-ablated PPI-GCN", [EDGE_AUC_MEAN, EDGE_AUPRC_MEAN], [EDGE_AUC_SD, EDGE_AUPRC_SD], COL_EDGE),
        ],
        legend_loc="inside",
    )
    ax.text(1.04, 0.745, "AUC gain\nremoved", ha="center", va="top", fontsize=7.4, color="#4D4D4D")

def panel_e(ax):
    ax.set_title("E  Partial train-test edge ablation", loc="left", fontweight="bold")
    styles = {
        "random_pair_removal": ("Random removal", COL_RANDOM, "o"),
        "high_confidence_first": ("High-confidence first", COL_HIGH, "s"),
        "low_confidence_first": ("Low-confidence first", COL_GCN, "^"),
    }
    for strategy, (label, color, marker) in styles.items():
        sub = df44[df44["strategy"] == strategy].sort_values("fraction_removed")
        ax.errorbar(sub["fraction_removed"] * 100, sub["delta_test_auc_vs_raw_esm"],
                    yerr=sub["test_auc_sd"], color=color, ecolor=color, marker=marker,
                    markersize=4.1, lw=1.6, capsize=2.4, label=label)
    ax.axhline(0, color="#4D4D4D", lw=0.85, ls="--")
    ax.set_xlabel("Removed direct train-test PPI pairs (%)")
    ax.set_ylabel("Delta AUC vs ESM raw")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.010, 0.026)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)

def panel_f(ax):
    grouped_metric_panel(
        ax,
        "F  Topology permutation control",
        [
            ("ESM raw", [ESM_AUC, ESM_AUPRC], [0, 0], COL_ESM),
            ("Real PPI-GCN", [GCN_AUC, GCN_AUPRC], [0, 0], COL_GCN),
            ("Topology-permuted", [PERM_AUC_MEAN, PERM_AUPRC_MEAN], [PERM_AUC_SD, PERM_AUPRC_SD], COL_PERM),
        ],
        legend_loc="inside",
    )
    ax.text(1.04, 0.745, "returns to\nESM level", ha="center", va="top", fontsize=7.4, color="#4D4D4D")
