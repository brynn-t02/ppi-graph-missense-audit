#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 41:
ESM-feature GCN train-test edge ablation control.

Goal:
Quantify how much the main ESM-feature GCN gain depends on direct PPI edges
connecting train proteins to test proteins.

Design:
- Use real model_inputs_main.pt.
- Use real ESM node features data["x"].
- Remove all direct train<->test edges from edge_index.
- Apply two-hop GCN-style propagation on the edge-ablated graph.
- Train downstream variant-level MLP using true train labels.
- Evaluate on real validation/test labels.

REAL DATA POLICY:
- Uses real graph and real ESM node features.
- Uses real split indices and real labels.
- Only removes direct train-test graph edges as a controlled ablation.
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

OUT_DIR = Path("/root/data/processed/esm_gcn_edge_ablation_exp41")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = OUT_DIR / "esm_feature_gcn_train_test_edge_ablated_embeddings.pt"

# Reference probability files from main representation comparison
REF_ESM_RAW_DIR = Path("/root/data/processed/variant_mlp_comparison/esm_raw")
REF_GCN_DIR = Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced")

# Reference summaries if available
EXP39B_SUMMARY = Path("/root/data/processed/random_feature_gcn_edge_ablation_exp39B/summary.json")
EXP39A_SUMMARY = Path("/root/data/processed/random_feature_gcn_label_shuffle_exp39A/summary.json")

# =========================
# Config
# =========================
SEED = 42
N_REPEATS = 5
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Utility functions
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

def ablate_train_test_edges(data):
    """
    Remove all direct edges connecting train protein nodes and test protein nodes.
    """
    edge_index = data["edge_index"]
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()

    num_nodes = int(data["x"].shape[0])

    train_idx = data["train_protein_idx"].long()
    test_idx = data["test_protein_idx"].long()

    train_nodes = torch.unique(train_idx)
    test_nodes = torch.unique(test_idx)

    train_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_node_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_node_mask[train_nodes] = True
    test_node_mask[test_nodes] = True

    row = edge_index[0]
    col = edge_index[1]

    is_train_test = (
        (train_node_mask[row] & test_node_mask[col])
        |
        (test_node_mask[row] & train_node_mask[col])
    )

    keep = ~is_train_test
    ablated_edge_index = edge_index[:, keep].contiguous()

    stats = {
        "num_nodes": int(num_nodes),
        "num_original_edges": int(edge_index.shape[1]),
        "num_ablated_edges": int(is_train_test.sum().item()),
        "num_remaining_edges": int(ablated_edge_index.shape[1]),
        "fraction_edges_removed": float(is_train_test.float().mean().item()),
        "num_unique_train_nodes": int(len(train_nodes)),
        "num_unique_test_nodes": int(len(test_nodes)),
        "ablation_mode": "remove_direct_train_test_edges_only"
    }

    return ablated_edge_index, stats

def generate_esm_feature_gcn_on_edge_index(data, edge_index):
    """
    Apply two-hop GCN-style propagation to real ESM node features on provided edge_index.

    h1 = A_norm x
    h2 = A_norm h1
    h_final = (x + h1 + h2) / 3

    This matches the fixed propagation style used in the random-feature controls.
    """
    print("[Embedding] Generating ESM-feature GCN embeddings on train-test-edge-ablated graph...")

    x = data["x"]
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float().cpu()

    num_nodes = int(x.shape[0])

    edge_index = edge_index.long().cpu()
    row, col = edge_index[0], edge_index[1]

    # Add reverse edges and self-loops, consistent with Exp39B.
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

    torch.save(h, EMB_PATH)
    print(f"[Embedding] Saved embeddings to {EMB_PATH}")

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

def train_one_repeat(split_data, repeat_id, seed):
    """
    Train downstream MLP with true train labels.
    """
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
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
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

    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    metrics = {
        "repeat_id": int(repeat_id),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": safe_auc(y_val_np, val_probs),
        "val_auprc": safe_auprc(y_val_np, val_probs),
        "test_auc": safe_auc(y_test_np, test_probs),
        "test_auprc": safe_auprc(y_test_np, test_probs),
    }

    np.save(OUT_DIR / f"repeat_{repeat_id}_val_probs.npy", val_probs)
    np.save(OUT_DIR / f"repeat_{repeat_id}_test_probs.npy", test_probs)

    return metrics

def summarize_metrics(rows):
    keys = ["val_auc", "val_auprc", "test_auc", "test_auprc"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(np.nanmean(vals))
        out[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out

def compute_reference_metrics_from_saved_probs(data):
    """
    Load saved raw ESM and full GCN probability files if available.
    Compute metrics on real val/test labels.
    """
    refs = {}

    for name, d in [
        ("esm_raw_saved", REF_ESM_RAW_DIR),
        ("gcn_full_saved", REF_GCN_DIR),
    ]:
        refs[name] = {
            "path": str(d),
            "available": False
        }

        val_p = d / "val_probs.npy"
        test_p = d / "test_probs.npy"

        if test_p.exists():
            test_probs = np.load(test_p)
            y_test = data["test_labels"].cpu().numpy() if torch.is_tensor(data["test_labels"]) else np.asarray(data["test_labels"])

            refs[name]["available"] = True
            refs[name]["test_probs_path"] = str(test_p)
            refs[name]["test_auc"] = safe_auc(y_test, test_probs)
            refs[name]["test_auprc"] = safe_auprc(y_test, test_probs)

        if val_p.exists():
            val_probs = np.load(val_p)
            y_val = data["val_labels"].cpu().numpy() if torch.is_tensor(data["val_labels"]) else np.asarray(data["val_labels"])

            refs[name]["val_probs_path"] = str(val_p)
            refs[name]["val_auc"] = safe_auc(y_val, val_probs)
            refs[name]["val_auprc"] = safe_auprc(y_val, val_probs)

    return refs

# =========================
# Main
# =========================
def main():
    print(f"Device: {DEVICE}")
    print("[1/7] Loading model_inputs_main.pt...")

    if not MODEL_INPUTS.exists():
        raise FileNotFoundError(f"Missing required file: {MODEL_INPUTS}")

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

    print(f"Nodes/features: {tuple(data['x'].shape)}")
    print(f"Original edge_index: {tuple(data['edge_index'].shape)}")
    print(f"Train variants: {len(data['train_labels'])}")
    print(f"Val variants: {len(data['val_labels'])}")
    print(f"Test variants: {len(data['test_labels'])}")

    print("[2/7] Computing saved-reference metrics if probability files exist...")
    reference_metrics = compute_reference_metrics_from_saved_probs(data)
    print(json.dumps(reference_metrics, indent=2))

    print("[3/7] Removing direct train-test graph edges...")

    ablated_edge_index, ablation_stats = ablate_train_test_edges(data)

    print("Ablation stats:")
    for k, v in ablation_stats.items():
        print(f"  {k}: {v}")

    with open(OUT_DIR / "edge_ablation_stats.json", "w") as f:
        json.dump(ablation_stats, f, indent=2)

    print("[4/7] Preparing ESM-feature GCN embeddings on ablated graph...")

    if EMB_PATH.exists():
        print(f"[Embedding] Reusing existing ablated ESM-GCN embedding: {EMB_PATH}")
        embeddings = torch.load(EMB_PATH, map_location="cpu").float()
    else:
        embeddings = generate_esm_feature_gcn_on_edge_index(
            data=data,
            edge_index=ablated_edge_index
        )

    print(f"Embedding shape: {tuple(embeddings.shape)}")

    print("[5/7] Mapping embeddings to variant splits...")

    split_data = get_split_tensors_flat(data, embeddings)

    print("[6/7] Training downstream classifier with true labels...")

    rows = []
    for r in range(N_REPEATS):
        seed = SEED + 3000 + r
        print(f"Repeat {r + 1}/{N_REPEATS}, seed={seed}")
        metrics = train_one_repeat(split_data, repeat_id=r, seed=seed)
        print(metrics)
        rows.append(metrics)

    summary_stats = summarize_metrics(rows)

    print("[7/7] Saving outputs...")

    per_repeat_csv = OUT_DIR / "per_repeat_esm_gcn_edge_ablation_metrics.csv"
    pd.DataFrame(rows).to_csv(per_repeat_csv, index=False)

    ref_39b, ref_39b_path = load_json_if_exists(EXP39B_SUMMARY)
    ref_39a, ref_39a_path = load_json_if_exists(EXP39A_SUMMARY)

    # Calculate main deltas against saved refs if available.
    comparison = {}
    if reference_metrics.get("esm_raw_saved", {}).get("available", False):
        comparison["ablated_esm_gcn_minus_raw_esm_test_auc"] = (
            summary_stats["test_auc_mean"] - reference_metrics["esm_raw_saved"]["test_auc"]
        )
        comparison["ablated_esm_gcn_minus_raw_esm_test_auprc"] = (
            summary_stats["test_auprc_mean"] - reference_metrics["esm_raw_saved"]["test_auprc"]
        )

    if reference_metrics.get("gcn_full_saved", {}).get("available", False):
        comparison["ablated_esm_gcn_minus_full_gcn_test_auc"] = (
            summary_stats["test_auc_mean"] - reference_metrics["gcn_full_saved"]["test_auc"]
        )
        comparison["ablated_esm_gcn_minus_full_gcn_test_auprc"] = (
            summary_stats["test_auprc_mean"] - reference_metrics["gcn_full_saved"]["test_auprc"]
        )

    summary = {
        "experiment": "41_esm_feature_gcn_train_test_edge_ablation_control",
        "real_data_policy": {
            "uses_real_model_inputs_main": str(MODEL_INPUTS),
            "uses_real_esm_node_features": True,
            "uses_real_graph_before_ablation": True,
            "uses_real_split_indices": True,
            "uses_real_train_validation_test_labels": True,
            "only_removes_direct_train_test_edges": True,
            "no_fake_labels": True,
            "no_simulated_predictions": True
        },
        "design": {
            "ablation_mode": "remove_direct_train_test_edges_only",
            "fixed_esm_feature_gcn_embeddings": str(EMB_PATH),
            "feature_dim": int(data["x"].shape[1]),
            "n_repeats": N_REPEATS,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "propagation": "two-hop symmetric-normalized fixed propagation, h=(x+h1+h2)/3",
            "model_inputs_structure": "flat_keys_train_protein_idx_val_protein_idx_test_protein_idx_train_labels_val_labels_test_labels"
        },
        "edge_ablation_stats": ablation_stats,
        "summary_metrics": summary_stats,
        "per_repeat_metrics": rows,
        "reference_saved_metrics": reference_metrics,
        "comparison_to_saved_references": comparison,
        "reference_random_feature_edge_ablation_exp39B_if_found": {
            "path": ref_39b_path,
            "summary": ref_39b
        },
        "reference_random_feature_label_shuffle_exp39A_if_found": {
            "path": ref_39a_path,
            "summary": ref_39a
        },
        "outputs": {
            "edge_ablation_stats_json": str(OUT_DIR / "edge_ablation_stats.json"),
            "per_repeat_metrics_csv": str(per_repeat_csv),
            "val_probs_pattern": str(OUT_DIR / "repeat_<id>_val_probs.npy"),
            "test_probs_pattern": str(OUT_DIR / "repeat_<id>_test_probs.npy")
        }
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Summary metrics:")
    print(json.dumps(summary_stats, indent=2))
    print("Comparison to saved references:")
    print(json.dumps(comparison, indent=2))
    print(f"Results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
