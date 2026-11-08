"""Exact original pair identity/order, STRING-score and pair selection functions."""
import gzip,json,os
from pathlib import Path
import numpy as np
import torch
STRING_LINKS=Path(os.environ.get("PPI_UNIFICATION_ROOT","/root/autodl-tmp/ppi_unification_20261108"))/"data/raw/9606.protein.links.v12.0.txt.gz"

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
