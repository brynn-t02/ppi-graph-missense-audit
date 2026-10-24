#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 45:
Real ESM features + topology-permuted PPI graph control.

Goal:
Test whether real ESM-GCN performance depends on the biological alignment
between protein identities and PPI topology, or whether generic dense graph
smoothing over real ESM features is sufficient.

Design:
- Use real model_inputs_main.pt.
- Use real ESM node features data["x"].
- Randomly permute graph node labels:
    edge_index_perm = perm[edge_index]
  This preserves the graph structure globally but assigns PPI neighborhoods to
  different proteins, breaking protein-specific topology-feature-label alignment.
- Apply two-hop GCN-style propagation on the permuted graph.
- Train downstream MLP using true train labels.
- Evaluate on real val/test labels.
- Repeat over multiple graph permutations and downstream seeds.

REAL DATA POLICY:
- Uses real ESM features.
- Uses real labels and real split indices.
- Uses real PPI graph structure before permutation.
- Only permutes node labels as a controlled topology-alignment negative control.
- No fake labels.
- No simulated predictions.
"""

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

OUT_DIR = Path("/root/data/processed/esm_gcn_topology_permutation_exp45")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_ESM_RAW_DIR = Path("/root/data/processed/variant_mlp_comparison/esm_raw")
REF_GCN_FULL_DIR = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced")

EXP41_SUMMARY = Path("/root/data/processed/esm_gcn_edge_ablation_exp41/summary.json")
EXP44_SUMMARY = Path("/root/data/processed/esm_gcn_partial_edge_ablation_exp44/summary.json")

# =========================
# Config
# =========================
SEED = 42

N_GRAPH_PERMUTATIONS = 3
N_DOWNSTREAM_REPEATS = 3

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
        ("real_ppi_gcn_full", REF_GCN_FULL_DIR),
    ]:
        refs[name] = {
            "path": str(d),
            "available": False
        }

        val_p = d / "val_probs.npy"
        test_p = d / "test_probs.npy"

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

def make_topology_permuted_edge_index(edge_index, num_nodes, seed):
    """
    Randomly permute graph node labels.

    If original edge is u -> v, permuted edge is perm[u] -> perm[v].

    This preserves global graph topology but breaks the biological alignment
    between protein identity/ESM features/labels and graph neighborhood.
    """
    rng = np.random.default_rng(seed)
    perm_np = rng.permutation(num_nodes)
    perm = torch.tensor(perm_np, dtype=torch.long)

    edge_index = edge_index.long().cpu()
    edge_perm = perm[edge_index]

    return edge_perm.contiguous(), perm

def generate_esm_feature_gcn_embedding(data, edge_index, out_path):
    """
    Apply two-hop GCN-style propagation to real ESM node features.

        h = (x + A_norm x + A_norm^2 x) / 3
    """
    print(f"[Embedding] Generating topology-permuted ESM-GCN embedding -> {out_path}")

    x = data["x"]
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float().cpu()

    num_nodes = int(x.shape[0])
    edge_index = edge_index.long().cpu()

    row, col = edge_index[0], edge_index[1]

    # Add reverse edges and self-loops, consistent with prior experiments.
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

def train_one_repeat(split_data, graph_perm_id, repeat_id, seed, out_dir):
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

        # Validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for start in range(0, len(y_val), BATCH_SIZE):
                xb = x_val[start:start + BATCH_SIZE].to(DEVICE)
                prob = torch.sigmoid(model(xb)).detach().cpu()
                val_probs.append(prob)

        val_probs = torch.cat(val_probs).numpy()
        val_auc = safe_auc(y_val.numpy(), val_probs)

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

    np.save(out_dir / f"graphperm_{graph_perm_id}_repeat_{repeat_id}_val_probs.npy", val_probs)
    np.save(out_dir / f"graphperm_{graph_perm_id}_repeat_{repeat_id}_test_probs.npy", test_probs)

    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    return {
        "graph_perm_id": int(graph_perm_id),
        "repeat_id": int(repeat_id),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": safe_auc(y_val_np, val_probs),
        "val_auprc": safe_auprc(y_val_np, val_probs),
        "test_auc": safe_auc(y_test_np, test_probs),
        "test_auprc": safe_auprc(y_test_np, test_probs),
    }

def summarize_rows(rows):
    keys = ["val_auc", "val_auprc", "test_auc", "test_auprc"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(np.nanmean(vals))
        out[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out

# =========================
# Main
# =========================
def main():
    print(f"Device: {DEVICE}")
    print("[1/6] Loading model_inputs_main.pt...")

    if not MODEL_INPUTS.exists():
        raise FileNotFoundError(MODEL_INPUTS)

    data = torch.load(MODEL_INPUTS, map_location="cpu")

    required = [
        "x",
        "edge_index",
        "train_protein_idx",
        "val_protein_idx",
        "test_protein_idx",
        "train_labels",
        "val_labels",
        "test_labels",
    ]
    require_keys(data, required)

    num_nodes = int(data["x"].shape[0])

    print(f"Nodes/features: {tuple(data['x'].shape)}")
    print(f"Original edge_index: {tuple(data['edge_index'].shape)}")
    print(f"Train variants: {len(data['train_labels'])}")
    print(f"Val variants: {len(data['val_labels'])}")
    print(f"Test variants: {len(data['test_labels'])}")

    print("[2/6] Computing saved reference metrics...")
    reference_metrics = compute_saved_reference_metrics(data)
    print(json.dumps(reference_metrics, indent=2))

    rows = []
    graph_summaries = []

    print("[3/6] Running topology permutations...")

    edge_index = data["edge_index"]
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long().cpu()

    for g in range(N_GRAPH_PERMUTATIONS):
        graph_seed = SEED + 6000 + g
        graph_dir = OUT_DIR / f"graph_permutation_{g}"
        graph_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Graph permutation {g+1}/{N_GRAPH_PERMUTATIONS}], seed={graph_seed}")

        edge_perm_path = graph_dir / "permuted_edge_index.pt"
        perm_path = graph_dir / "node_permutation.pt"
        emb_path = graph_dir / "esm_gcn_topology_permuted_embedding.pt"

        if edge_perm_path.exists() and perm_path.exists():
            print("[Graph] Reusing existing permuted edge_index.")
            edge_perm = torch.load(edge_perm_path, map_location="cpu").long()
            perm = torch.load(perm_path, map_location="cpu").long()
        else:
            edge_perm, perm = make_topology_permuted_edge_index(
                edge_index=edge_index,
                num_nodes=num_nodes,
                seed=graph_seed
            )
            torch.save(edge_perm, edge_perm_path)
            torch.save(perm, perm_path)

        # Sanity stats for permutation
        unchanged_nodes = int((perm == torch.arange(num_nodes)).sum().item())
        graph_summary = {
            "graph_perm_id": int(g),
            "graph_seed": int(graph_seed),
            "num_nodes": int(num_nodes),
            "num_edges": int(edge_perm.shape[1]),
            "num_unchanged_node_labels": unchanged_nodes,
            "fraction_unchanged_node_labels": float(unchanged_nodes / num_nodes),
            "permuted_edge_index_path": str(edge_perm_path),
            "node_permutation_path": str(perm_path),
            "embedding_path": str(emb_path),
        }
        graph_summaries.append(graph_summary)

        with open(graph_dir / "graph_permutation_summary.json", "w") as f:
            json.dump(graph_summary, f, indent=2)

        if emb_path.exists():
            print(f"[Embedding] Reusing embedding: {emb_path}")
            embeddings = torch.load(emb_path, map_location="cpu").float()
        else:
            embeddings = generate_esm_feature_gcn_embedding(
                data=data,
                edge_index=edge_perm,
                out_path=emb_path
            )

        print("[4/6] Mapping embeddings to splits...")
        split_data = get_split_tensors_flat(data, embeddings)

        print("[5/6] Training downstream repeats...")
        for r in range(N_DOWNSTREAM_REPEATS):
            seed = SEED + 7000 + g * 100 + r
            print(f"[Graph {g}] Downstream repeat {r+1}/{N_DOWNSTREAM_REPEATS}, seed={seed}")

            metrics = train_one_repeat(
                split_data=split_data,
                graph_perm_id=g,
                repeat_id=r,
                seed=seed,
                out_dir=graph_dir
            )
            print(metrics)
            rows.append(metrics)

    print("[6/6] Saving outputs...")

    per_repeat_csv = OUT_DIR / "per_repeat_topology_permutation_metrics.csv"
    pd.DataFrame(rows).to_csv(per_repeat_csv, index=False)

    summary_stats = summarize_rows(rows)

    # Per graph summary
    per_graph = []
    df_rows = pd.DataFrame(rows)
    for g, sub in df_rows.groupby("graph_perm_id"):
        s = summarize_rows(sub.to_dict("records"))
        s["graph_perm_id"] = int(g)
        s["n_repeats"] = int(len(sub))
        per_graph.append(s)

    per_graph_csv = OUT_DIR / "per_graph_topology_permutation_summary.csv"
    pd.DataFrame(per_graph).to_csv(per_graph_csv, index=False)

    ref41, ref41_path = load_json_if_exists(EXP41_SUMMARY)
    ref44, ref44_path = load_json_if_exists(EXP44_SUMMARY)

    comparison = {}

    if reference_metrics.get("esm_raw", {}).get("available", False):
        comparison["topology_permuted_minus_raw_esm_test_auc"] = (
            summary_stats["test_auc_mean"] - reference_metrics["esm_raw"]["test_auc"]
        )
        comparison["topology_permuted_minus_raw_esm_test_auprc"] = (
            summary_stats["test_auprc_mean"] - reference_metrics["esm_raw"]["test_auprc"]
        )

    if reference_metrics.get("real_ppi_gcn_full", {}).get("available", False):
        comparison["topology_permuted_minus_real_ppi_gcn_test_auc"] = (
            summary_stats["test_auc_mean"] - reference_metrics["real_ppi_gcn_full"]["test_auc"]
        )
        comparison["topology_permuted_minus_real_ppi_gcn_test_auprc"] = (
            summary_stats["test_auprc_mean"] - reference_metrics["real_ppi_gcn_full"]["test_auprc"]
        )

    summary = {
        "experiment": "45_real_esm_topology_permuted_ppi_gcn_control",
        "real_data_policy": {
            "uses_real_model_inputs_main": str(MODEL_INPUTS),
            "uses_real_esm_node_features": True,
            "uses_real_labels_and_splits": True,
            "uses_real_ppi_graph_before_permutation": True,
            "control_operation": "random node-label permutation of PPI graph",
            "no_fake_labels": True,
            "no_simulated_predictions": True
        },
        "purpose": (
            "Test whether real ESM-GCN performance depends on biological alignment "
            "between protein identity and PPI topology, or whether generic graph "
            "smoothing over ESM features is sufficient."
        ),
        "design": {
            "n_graph_permutations": N_GRAPH_PERMUTATIONS,
            "n_downstream_repeats_per_graph": N_DOWNSTREAM_REPEATS,
            "total_runs": int(N_GRAPH_PERMUTATIONS * N_DOWNSTREAM_REPEATS),
            "propagation": "two-hop symmetric-normalized fixed propagation, h=(x+h1+h2)/3",
            "device": DEVICE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
        },
        "graph_permutation_summaries": graph_summaries,
        "summary_metrics": summary_stats,
        "per_graph_summary": per_graph,
        "per_repeat_metrics": rows,
        "reference_metrics": reference_metrics,
        "comparison_to_references": comparison,
        "reference_exp41_edge_ablation_if_found": {
            "path": ref41_path,
            "summary": ref41
        },
        "reference_exp44_partial_ablation_if_found": {
            "path": ref44_path,
            "summary": ref44
        },
        "outputs": {
            "per_repeat_metrics_csv": str(per_repeat_csv),
            "per_graph_summary_csv": str(per_graph_csv),
            "graph_permutation_dirs": str(OUT_DIR / "graph_permutation_<id>/"),
            "summary_json": str(OUT_DIR / "summary.json")
        }
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Summary metrics:")
    print(json.dumps(summary_stats, indent=2))
    print("Comparison to references:")
    print(json.dumps(comparison, indent=2))
    print(f"Results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
