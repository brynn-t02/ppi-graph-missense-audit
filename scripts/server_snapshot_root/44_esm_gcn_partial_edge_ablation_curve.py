#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 44:
Pair-level partial train-test edge ablation curve for ESM-feature GCN.

Goal:
Quantify whether ESM-feature GCN performance decays as a function of the
fraction of direct train-test PPI connectivity removed.

Key improvement over naive partial edge-entry removal:
- Ablation is performed at unique undirected train-test protein-pair level.
- If a selected train-test pair is removed, all corresponding edge_index entries
  for that pair are removed.
- This avoids leaving the reverse direction intact during symmetric propagation.

Strategies:
1. random_pair_removal
2. high_confidence_first
3. low_confidence_first

STRING confidence scores are loaded from:
/root/data/raw/9606.protein.links.v12.0.txt.gz

REAL DATA POLICY:
- Uses real model_inputs_main.pt.
- Uses real ESM node features.
- Uses real PPI graph.
- Uses real STRING combined_score for stratified ablation.
- Uses real split indices and labels.
- Only removes direct train-test graph edges as controlled ablation.
- No fake labels.
- No simulated predictions.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# Paths
# =========================
MODEL_INPUTS = Path("/root/data/processed/model_inputs_main.pt")
STRING_LINKS = Path("/root/data/raw/9606.protein.links.v12.0.txt.gz")

OUT_DIR = Path("/root/data/processed/esm_gcn_partial_edge_ablation_exp44")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_ESM_RAW_DIR = Path("/root/data/processed/variant_mlp_comparison/esm_raw")
REF_GCN_FULL_DIR = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced")

EXP41_DIR = Path("/root/data/processed/esm_gcn_edge_ablation_exp41")
EXP41_METRICS = EXP41_DIR / "per_repeat_esm_gcn_edge_ablation_metrics.csv"
EXP41_SUMMARY = EXP41_DIR / "summary.json"

# =========================
# Config
# =========================
SEED = 42

FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
STRATEGIES = [
    "random_pair_removal",
    "high_confidence_first",
    "low_confidence_first",
]

N_REPEATS_PARTIAL = 3

MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Utilities
# =========================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))

def safe_auprc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))

class MLP(nn.Module):
    def __init__(self, input_dim=1280, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def require_keys(data, keys):
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"model_inputs_main.pt missing required keys: {missing}")

def load_json_if_exists(path):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f), str(path)
        except Exception:
            return None, str(path)
    return None, None

def compute_saved_reference_metrics(data):
    refs = {}

    y_val = data["val_labels"].cpu().numpy() if torch.is_tensor(data["val_labels"]) else np.asarray(data["val_labels"])
    y_test = data["test_labels"].cpu().numpy() if torch.is_tensor(data["test_labels"]) else np.asarray(data["test_labels"])

    for name, d in [
        ("esm_raw", REF_ESM_RAW_DIR),
        ("gcn_full_0pct_removed", REF_GCN_FULL_DIR),
    ]:
        val_p = d / "val_probs.npy"
        test_p = d / "test_probs.npy"

        refs[name] = {
            "path": str(d),
            "available": False
        }

        if test_p.exists():
            test_probs = np.load(test_p)
            refs[name]["available"] = True
            refs[name]["test_probs_path"] = str(test_p)
            refs[name]["test_auc"] = safe_auc(y_test, test_probs)
            refs[name]["test_auprc"] = safe_auprc(y_test, test_probs)

        if val_p.exists():
            val_probs = np.load(val_p)
            refs[name]["val_probs_path"] = str(val_p)
            refs[name]["val_auc"] = safe_auc(y_val, val_probs)
            refs[name]["val_auprc"] = safe_auprc(y_val, val_probs)

    return refs

def identify_train_test_pairs_and_entries(data):
    """
    Identify unique undirected train-test protein pairs present in edge_index,
    and collect all edge_index entries belonging to each pair.
    """
    edge_index = data["edge_index"]
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long().cpu()

    num_nodes = int(data["x"].shape[0])

    train_idx = data["train_protein_idx"].long().cpu()
    test_idx = data["test_protein_idx"].long().cpu()

    train_nodes = torch.unique(train_idx)
    test_nodes = torch.unique(test_idx)

    train_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_node_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_node_mask[train_nodes] = True
    test_node_mask[test_nodes] = True

    row = edge_index[0]
    col = edge_index[1]

    tt_mask = (
        (train_node_mask[row] & test_node_mask[col])
        |
        (test_node_mask[row] & train_node_mask[col])
    )

    tt_entry_indices = torch.where(tt_mask)[0].cpu().numpy()

    pair_to_entries = {}

    for idx in tt_entry_indices:
        a = int(row[idx].item())
        b = int(col[idx].item())
        if a == b:
            continue
        pair = (a, b) if a < b else (b, a)
        pair_to_entries.setdefault(pair, []).append(int(idx))

    pairs = list(pair_to_entries.keys())

    pair_entry_counts = np.array([len(pair_to_entries[p]) for p in pairs], dtype=int)

    stats = {
        "num_nodes": int(num_nodes),
        "num_original_edges": int(edge_index.shape[1]),
        "num_train_test_edge_entries": int(len(tt_entry_indices)),
        "num_unique_train_test_pairs": int(len(pairs)),
        "fraction_train_test_edge_entries": float(len(tt_entry_indices) / edge_index.shape[1]),
        "num_unique_train_nodes": int(len(train_nodes)),
        "num_unique_test_nodes": int(len(test_nodes)),
        "pair_entry_count_min": int(pair_entry_counts.min()) if len(pair_entry_counts) else 0,
        "pair_entry_count_median": float(np.median(pair_entry_counts)) if len(pair_entry_counts) else 0,
        "pair_entry_count_max": int(pair_entry_counts.max()) if len(pair_entry_counts) else 0,
    }

    return edge_index, pairs, pair_to_entries, stats

def load_string_scores_for_pairs(pairs, idx_to_protein):
    """
    Load STRING combined_score for train-test pairs.

    pairs:
        list of (node_idx_a, node_idx_b), unordered

    idx_to_protein:
        maps node idx -> STRING protein id

    Returns:
        pair_scores dict: pair -> score
    """
    if not STRING_LINKS.exists():
        raise FileNotFoundError(f"Missing STRING links file: {STRING_LINKS}")

    print(f"[STRING] Loading scores from {STRING_LINKS}")

    protein_to_idx = {v: k for k, v in idx_to_protein.items()}
    pair_set = set(pairs)
    pair_scores = {}

    with gzip.open(STRING_LINKS, "rt") as f:
        header = f.readline().strip().split()
        # Expected: protein1 protein2 combined_score
        for line_i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            p1, p2, score_s = parts[0], parts[1], parts[2]

            if p1 not in protein_to_idx or p2 not in protein_to_idx:
                continue

            a = int(protein_to_idx[p1])
            b = int(protein_to_idx[p2])
            if a == b:
                continue

            pair = (a, b) if a < b else (b, a)

            if pair in pair_set:
                try:
                    score = float(score_s)
                except Exception:
                    score = np.nan
                pair_scores[pair] = score

    scores = np.array([pair_scores.get(p, np.nan) for p in pairs], dtype=float)

    stats = {
        "num_pairs": int(len(pairs)),
        "num_pairs_with_string_score": int(np.sum(~np.isnan(scores))),
        "num_pairs_missing_string_score": int(np.sum(np.isnan(scores))),
        "score_min": float(np.nanmin(scores)) if np.sum(~np.isnan(scores)) else np.nan,
        "score_median": float(np.nanmedian(scores)) if np.sum(~np.isnan(scores)) else np.nan,
        "score_max": float(np.nanmax(scores)) if np.sum(~np.isnan(scores)) else np.nan,
    }

    print("[STRING] Score coverage:")
    print(json.dumps(stats, indent=2))

    return pair_scores, stats

def select_pairs_for_removal(pairs, pair_scores, fraction, strategy, seed=42):
    """
    Select unique undirected train-test pairs for removal.

    fraction:
        fraction of unique train-test pairs to remove.

    strategy:
        random_pair_removal
        high_confidence_first
        low_confidence_first
    """
    n_pairs = len(pairs)

    if fraction <= 0:
        return []

    n_remove = int(round(fraction * n_pairs))
    n_remove = min(max(n_remove, 0), n_pairs)

    if n_remove == 0:
        return []

    pairs_arr = np.array(pairs, dtype=object)

    if strategy == "random_pair_removal":
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(n_pairs), size=n_remove, replace=False)
        return [pairs[i] for i in idx]

    scores = np.array([pair_scores.get(p, np.nan) for p in pairs], dtype=float)

    if strategy == "high_confidence_first":
        # Missing scores go last
        sortable = np.nan_to_num(scores, nan=-1.0)
        order = np.argsort(sortable)[::-1]
        idx = order[:n_remove]
        return [pairs[i] for i in idx]

    if strategy == "low_confidence_first":
        # Missing scores go last
        sortable = np.nan_to_num(scores, nan=1e12)
        order = np.argsort(sortable)
        idx = order[:n_remove]
        return [pairs[i] for i in idx]

    raise ValueError(f"Unknown strategy: {strategy}")

def make_pair_level_ablated_edge_index(edge_index, pair_to_entries, selected_pairs):
    """
    Remove all edge_index entries associated with selected undirected pairs.
    """
    n_edges = edge_index.shape[1]
    keep = torch.ones(n_edges, dtype=torch.bool)

    removed_entries = []

    for pair in selected_pairs:
        entries = pair_to_entries.get(pair, [])
        if entries:
            removed_entries.extend(entries)

    if len(removed_entries) > 0:
        keep[torch.tensor(removed_entries, dtype=torch.long)] = False

    ablated_edge_index = edge_index[:, keep].contiguous()

    return ablated_edge_index, removed_entries

def generate_esm_feature_gcn_embedding(data, edge_index, out_path):
    """
    Apply two-hop GCN-style propagation to real ESM node features on provided edge_index.

    h = (x + A_norm x + A_norm^2 x) / 3
    """
    print(f"[Embedding] Generating ESM-GCN embedding -> {out_path}")

    x = data["x"]
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float().cpu()

    num_nodes = int(x.shape[0])
    edge_index = edge_index.long().cpu()

    row, col = edge_index[0], edge_index[1]

    rev_row = col
    rev_col = row
    self_nodes = torch.arange(num_nodes, dtype=torch.long)

    all_row = torch.cat([row, rev_row, self_nodes])
    all_col = torch.cat([col, rev_col, self_nodes])

    deg = torch.bincount(all_row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1), -0.5)
    norm = deg_inv_sqrt[all_row] * deg_inv_sqrt[all_col]

    def propagate(feat):
        out = torch.zeros_like(feat)
        chunk = 500000
        for start in range(0, len(all_row), chunk):
            end = min(start + chunk, len(all_row))
            rr = all_row[start:end]
            cc = all_col[start:end]
            ww = norm[start:end]
            msg = feat[cc] * ww.unsqueeze(1)
            out.index_add_(0, rr, msg)
        return out

    h1 = propagate(x)
    h2 = propagate(h1)
    h = (x + h1 + h2) / 3.0

    torch.save(h, out_path)
    return h

def get_split_tensors_flat(data, embeddings):
    out = {}

    mapping = {
        "train": ("train_protein_idx", "train_labels"),
        "val": ("val_protein_idx", "val_labels"),
        "test": ("test_protein_idx", "test_labels"),
    }

    for split, (idx_key, label_key) in mapping.items():
        protein_idx = data[idx_key]
        labels = data[label_key]

        if not torch.is_tensor(protein_idx):
            protein_idx = torch.tensor(protein_idx, dtype=torch.long)
        else:
            protein_idx = protein_idx.long()

        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = labels.float()

        x = embeddings[protein_idx]

        out[split] = {
            "x": x.float(),
            "y": labels.float(),
            "protein_idx": protein_idx,
        }

    return out

def train_one_repeat(split_data, repeat_id, seed, fraction, strategy, out_dir):
    set_seed(seed)

    x_train = split_data["train"]["x"]
    y_train = split_data["train"]["y"]

    x_val = split_data["val"]["x"]
    y_val = split_data["val"]["y"]

    x_test = split_data["test"]["x"]
    y_test = split_data["test"]["y"]

    n_pos = float((y_train == 1).sum().item())
    n_neg = float((y_train == 0).sum().item())
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1.0)],
        dtype=torch.float32,
        device=DEVICE
    )

    model = MLP(input_dim=x_train.shape[1], dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1.0
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE

    n = len(y_train)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        idx = torch.randperm(n)

        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]

            xb = x_train[batch_idx].to(DEVICE)
            yb = y_train[batch_idx].to(DEVICE)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for start in range(0, len(y_val), BATCH_SIZE):
                xb = x_val[start:start + BATCH_SIZE].to(DEVICE)
                prob = torch.sigmoid(model(xb)).detach().cpu()
                val_probs.append(prob)

        val_probs = torch.cat(val_probs).numpy()
        y_val_np = y_val.numpy()
        val_auc = safe_auc(y_val_np, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    def predict(x):
        probs = []
        with torch.no_grad():
            for start in range(0, len(x), BATCH_SIZE):
                xb = x[start:start + BATCH_SIZE].to(DEVICE)
                prob = torch.sigmoid(model(xb)).detach().cpu()
                probs.append(prob)
        return torch.cat(probs).numpy()

    val_probs = predict(x_val)
    test_probs = predict(x_test)

    np.save(out_dir / f"{strategy}_fraction_{fraction:.2f}_repeat_{repeat_id}_val_probs.npy", val_probs)
    np.save(out_dir / f"{strategy}_fraction_{fraction:.2f}_repeat_{repeat_id}_test_probs.npy", test_probs)

    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    return {
        "strategy": strategy,
        "fraction_removed": float(fraction),
        "repeat_id": int(repeat_id),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": safe_auc(y_val_np, val_probs),
        "val_auprc": safe_auprc(y_val_np, val_probs),
        "test_auc": safe_auc(y_test_np, test_probs),
        "test_auprc": safe_auprc(y_test_np, test_probs),
    }

def summarize_df(df):
    out = {}
    for k in ["val_auc", "val_auprc", "test_auc", "test_auprc"]:
        vals = df[k].astype(float).values
        out[k + "_mean"] = float(np.nanmean(vals))
        out[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out

def load_exp41_100pct():
    if EXP41_METRICS.exists():
        df = pd.read_csv(EXP41_METRICS)
        df["fraction_removed"] = 1.0
        df["strategy"] = "all_strategies_reuse_exp41_100pct"
        df["source"] = "exp41_full_train_test_edge_ablation"
        return df
    return None

# =========================
# Main
# =========================
def main():
    print(f"Device: {DEVICE}")

    print("[1/8] Loading model_inputs_main.pt...")
    if not MODEL_INPUTS.exists():
        raise FileNotFoundError(MODEL_INPUTS)

    data = torch.load(MODEL_INPUTS, map_location="cpu")

    required = [
        "x",
        "edge_index",
        "protein_to_idx",
        "idx_to_protein",
        "train_protein_idx",
        "val_protein_idx",
        "test_protein_idx",
        "train_labels",
        "val_labels",
        "test_labels",
    ]
    require_keys(data, required)

    print(f"Nodes/features: {tuple(data['x'].shape)}")
    print(f"Original edge_index: {tuple(data['edge_index'].shape)}")

    print("[2/8] Loading saved reference metrics...")
    reference_metrics = compute_saved_reference_metrics(data)
    print(json.dumps(reference_metrics, indent=2))

    print("[3/8] Identifying unique train-test protein pairs...")
    edge_index, pairs, pair_to_entries, pair_stats = identify_train_test_pairs_and_entries(data)
    print(json.dumps(pair_stats, indent=2))

    with open(OUT_DIR / "pair_level_train_test_edge_stats.json", "w") as f:
        json.dump(pair_stats, f, indent=2)

    print("[4/8] Loading STRING combined scores for train-test pairs...")
    pair_scores, score_stats = load_string_scores_for_pairs(
        pairs=pairs,
        idx_to_protein=data["idx_to_protein"]
    )

    with open(OUT_DIR / "string_score_coverage_for_train_test_pairs.json", "w") as f:
        json.dump(score_stats, f, indent=2)

    all_rows = []
    all_edge_stats = []

    print("[5/8] Adding 0% full-GCN reference rows...")
    if reference_metrics["gcn_full_0pct_removed"]["available"]:
        ref = reference_metrics["gcn_full_0pct_removed"]

        for strategy in STRATEGIES:
            row0 = {
                "strategy": strategy,
                "fraction_removed": 0.0,
                "repeat_id": -1,
                "seed": -1,
                "best_epoch": -1,
                "best_val_auc": ref.get("val_auc", np.nan),
                "val_auc": ref.get("val_auc", np.nan),
                "val_auprc": ref.get("val_auprc", np.nan),
                "test_auc": ref.get("test_auc", np.nan),
                "test_auprc": ref.get("test_auprc", np.nan),
                "source": "saved_full_gcn_reference"
            }
            all_rows.append(row0)

    print("[6/8] Adding 100% Exp41 rows if available...")
    df41 = load_exp41_100pct()
    if df41 is not None:
        for strategy in STRATEGIES:
            tmp = df41.copy()
            tmp["strategy"] = strategy
            tmp["source"] = "exp41_full_train_test_edge_ablation"
            for _, row in tmp.iterrows():
                all_rows.append(row.to_dict())

    print("[7/8] Running partial ablation strategies...")

    # Only train partial fractions excluding 0 and 1
    partial_fractions = [f for f in FRACTIONS if f not in [0.0, 1.0]]

    for strategy in STRATEGIES:
        for fraction in partial_fractions:
            frac_key = f"{fraction:.2f}"
            run_dir = OUT_DIR / f"{strategy}_fraction_{frac_key}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[Strategy={strategy}, fraction={fraction:.2f}] Selecting pairs...")

            selected_pairs = select_pairs_for_removal(
                pairs=pairs,
                pair_scores=pair_scores,
                fraction=fraction,
                strategy=strategy,
                seed=SEED + int(fraction * 1000)
            )

            ablated_edge_index, removed_entries = make_pair_level_ablated_edge_index(
                edge_index=edge_index,
                pair_to_entries=pair_to_entries,
                selected_pairs=selected_pairs
            )

            selected_scores = np.array([pair_scores.get(p, np.nan) for p in selected_pairs], dtype=float)

            edge_stats = {
                "strategy": strategy,
                "fraction_removed": float(fraction),
                "num_original_edges": int(edge_index.shape[1]),
                "num_unique_train_test_pairs_total": int(len(pairs)),
                "num_selected_pairs_removed": int(len(selected_pairs)),
                "num_train_test_edge_entries_total": int(pair_stats["num_train_test_edge_entries"]),
                "num_removed_edge_entries": int(len(removed_entries)),
                "num_remaining_edges": int(ablated_edge_index.shape[1]),
                "fraction_unique_pairs_removed": float(len(selected_pairs) / len(pairs)),
                "fraction_all_edge_entries_removed": float(len(removed_entries) / edge_index.shape[1]),
                "removed_pair_score_min": float(np.nanmin(selected_scores)) if np.sum(~np.isnan(selected_scores)) else np.nan,
                "removed_pair_score_median": float(np.nanmedian(selected_scores)) if np.sum(~np.isnan(selected_scores)) else np.nan,
                "removed_pair_score_max": float(np.nanmax(selected_scores)) if np.sum(~np.isnan(selected_scores)) else np.nan,
                "num_removed_pairs_with_score": int(np.sum(~np.isnan(selected_scores))),
                "num_removed_pairs_missing_score": int(np.sum(np.isnan(selected_scores))),
            }

            all_edge_stats.append(edge_stats)

            with open(run_dir / "edge_stats.json", "w") as f:
                json.dump(edge_stats, f, indent=2)

            print("[Edge stats]")
            print(json.dumps(edge_stats, indent=2))

            emb_path = run_dir / f"esm_gcn_embedding_{strategy}_fraction_{frac_key}.pt"

            if emb_path.exists():
                print(f"[Embedding] Reusing embedding: {emb_path}")
                embeddings = torch.load(emb_path, map_location="cpu").float()
            else:
                embeddings = generate_esm_feature_gcn_embedding(
                    data=data,
                    edge_index=ablated_edge_index,
                    out_path=emb_path
                )

            split_data = get_split_tensors_flat(data, embeddings)

            rows_this = []
            for r in range(N_REPEATS_PARTIAL):
                seed = SEED + 5000 + int(fraction * 1000) + r
                if strategy == "high_confidence_first":
                    seed += 100000
                elif strategy == "low_confidence_first":
                    seed += 200000

                print(f"[{strategy}, fraction={fraction:.2f}] Repeat {r+1}/{N_REPEATS_PARTIAL}, seed={seed}")

                metrics = train_one_repeat(
                    split_data=split_data,
                    repeat_id=r,
                    seed=seed,
                    fraction=fraction,
                    strategy=strategy,
                    out_dir=run_dir
                )
                metrics["source"] = "partial_pair_level_edge_ablation_retrained"
                print(metrics)

                rows_this.append(metrics)
                all_rows.append(metrics)

            pd.DataFrame(rows_this).to_csv(run_dir / "per_repeat_metrics.csv", index=False)

    print("[8/8] Writing outputs...")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(OUT_DIR / "per_repeat_all_strategies_fractions.csv", index=False)

    edge_stats_df = pd.DataFrame(all_edge_stats)
    edge_stats_df.to_csv(OUT_DIR / "edge_stats_all_partial_runs.csv", index=False)

    # Curve summary
    curve_rows = []
    raw_ref = reference_metrics.get("esm_raw", {})
    raw_auc = raw_ref.get("test_auc", np.nan)
    raw_auprc = raw_ref.get("test_auprc", np.nan)

    for (strategy, fraction), sub in all_df.groupby(["strategy", "fraction_removed"]):
        summ = summarize_df(sub)
        row = {
            "strategy": strategy,
            "fraction_removed": float(fraction),
            **summ,
            "delta_test_auc_vs_raw_esm": summ["test_auc_mean"] - raw_auc if raw_auc == raw_auc else np.nan,
            "delta_test_auprc_vs_raw_esm": summ["test_auprc_mean"] - raw_auprc if raw_auprc == raw_auprc else np.nan,
            "n_rows": int(len(sub)),
        }
        curve_rows.append(row)

    curve_df = pd.DataFrame(curve_rows).sort_values(["strategy", "fraction_removed"])
    curve_df.to_csv(OUT_DIR / "edge_ablation_curve_summary.csv", index=False)

    exp41_summary, exp41_summary_path = load_json_if_exists(EXP41_SUMMARY)

    summary = {
        "experiment": "44_pair_level_partial_train_test_edge_ablation_curve",
        "real_data_policy": {
            "uses_real_model_inputs_main": str(MODEL_INPUTS),
            "uses_real_esm_node_features": True,
            "uses_real_graph_before_ablation": True,
            "uses_real_STRING_combined_scores": str(STRING_LINKS),
            "uses_real_split_indices": True,
            "uses_real_train_validation_test_labels": True,
            "only_removes_direct_train_test_edges": True,
            "no_fake_labels": True,
            "no_simulated_predictions": True
        },
        "design": {
            "ablation_unit": "unique undirected train-test protein pair",
            "strategies": STRATEGIES,
            "fractions_removed": FRACTIONS,
            "n_repeats_partial": N_REPEATS_PARTIAL,
            "fraction_0_source": "saved full GCN reference",
            "fraction_1_source": "Exp41 reused if available",
            "propagation": "two-hop symmetric-normalized fixed propagation, h=(x+h1+h2)/3",
            "device": DEVICE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE
        },
        "pair_level_train_test_edge_stats": pair_stats,
        "string_score_coverage_for_train_test_pairs": score_stats,
        "reference_metrics": reference_metrics,
        "exp41_reference_if_found": {
            "path": exp41_summary_path,
            "summary": exp41_summary
        },
        "outputs": {
            "per_repeat_all_strategies_fractions_csv": str(OUT_DIR / "per_repeat_all_strategies_fractions.csv"),
            "edge_ablation_curve_summary_csv": str(OUT_DIR / "edge_ablation_curve_summary.csv"),
            "edge_stats_all_partial_runs_csv": str(OUT_DIR / "edge_stats_all_partial_runs.csv"),
            "pair_level_train_test_edge_stats_json": str(OUT_DIR / "pair_level_train_test_edge_stats.json"),
            "string_score_coverage_json": str(OUT_DIR / "string_score_coverage_for_train_test_pairs.json"),
            "summary_json": str(OUT_DIR / "summary.json")
        }
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Curve summary:")
    print(curve_df.to_string(index=False))
    print(f"Results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
