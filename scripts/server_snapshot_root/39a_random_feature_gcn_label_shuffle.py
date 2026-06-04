#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 39A:
Random-feature GCN label-permutation control.

Goal:
Test whether the strong random-feature GCN performance depends on meaningful
train labels. We fix graph-propagated random node embeddings, shuffle train
labels, and retrain the downstream variant-level classifier.

Adapted to model_inputs_main.pt flat-key structure:
- x
- edge_index
- train_protein_idx / val_protein_idx / test_protein_idx
- train_labels / val_labels / test_labels

REAL DATA POLICY:
- Uses real model_inputs_main.pt.
- Uses real split protein indices.
- Uses real validation/test labels.
- Only shuffles TRAIN labels as a negative control.
- No fake validation/test labels.
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

OUT_DIR = Path("/root/data/processed/random_feature_gcn_label_shuffle_exp39A")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = OUT_DIR / "fixed_random_feature_gcn_embeddings.pt"

EXP15B_SUMMARY_CANDIDATES = [
    Path("/root/data/processed/random_feature_gcn_repeated/summary.json"),
    Path("/root/data/processed/random_feature_gcn_control/summary.json"),
]

# =========================
# Config
# =========================
SEED = 42
FEATURE_DIM = 1280
N_REPEATS = 5
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

def get_split_tensors_flat(data, embeddings):
    """
    Adapted to the actual model_inputs_main.pt flat-key structure.
    """
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

def generate_fixed_random_feature_gcn(data, seed=42):
    """
    Generate fixed random node features and apply two-step GCN-style propagation:
        h1 = A_norm x
        h2 = A_norm h1
        h_final = (x + h1 + h2) / 3

    This matches prior fixed-propagation style.
    """
    print("[Embedding] Generating fixed random-feature GCN embeddings...")

    set_seed(seed)

    edge_index = data["edge_index"]
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()

    num_nodes = int(data["x"].shape[0])

    x = torch.randn(num_nodes, FEATURE_DIM, dtype=torch.float32)

    row, col = edge_index[0], edge_index[1]

    # Add reverse edges and self-loops
    rev_row = col
    rev_col = row
    self_nodes = torch.arange(num_nodes, dtype=torch.long)

    all_row = torch.cat([row, rev_row, self_nodes])
    all_col = torch.cat([col, rev_col, self_nodes])

    # Symmetric normalization
    deg = torch.bincount(all_row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1), -0.5)
    norm = deg_inv_sqrt[all_row] * deg_inv_sqrt[all_col]

    def propagate(feat):
        out = torch.zeros_like(feat)
        # CPU memory-safe scatter add
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
    print(f"[Embedding] Saved fixed random-feature GCN embeddings to {EMB_PATH}")

    return h

def train_one_repeat(split_data, repeat_id, seed):
    """
    Train downstream MLP with SHUFFLED train labels.
    Validation/test labels remain real.
    """
    set_seed(seed)

    x_train = split_data["train"]["x"]
    y_train_true = split_data["train"]["y"]

    x_val = split_data["val"]["x"]
    y_val = split_data["val"]["y"]

    x_test = split_data["test"]["x"]
    y_test = split_data["test"]["y"]

    # Shuffle train labels only
    perm = torch.randperm(len(y_train_true))
    y_train = y_train_true[perm].clone()

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

    # Load best
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

def load_reference_exp15b():
    for p in EXP15B_SUMMARY_CANDIDATES:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f), str(p)
            except Exception:
                pass
    return None, None

# =========================
# Main
# =========================
def main():
    print(f"Device: {DEVICE}")
    print("[1/5] Loading model_inputs_main.pt...")

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

    print(f"Nodes: {tuple(data['x'].shape)}")
    print(f"Edges: {tuple(data['edge_index'].shape)}")
    print(f"Train variants: {len(data['train_labels'])}")
    print(f"Val variants: {len(data['val_labels'])}")
    print(f"Test variants: {len(data['test_labels'])}")

    print("[2/5] Preparing fixed random-feature GCN embeddings...")

    if EMB_PATH.exists():
        print(f"[Embedding] Reusing existing fixed embedding: {EMB_PATH}")
        embeddings = torch.load(EMB_PATH, map_location="cpu").float()
    else:
        embeddings = generate_fixed_random_feature_gcn(data, seed=SEED)

    print(f"Embedding shape: {tuple(embeddings.shape)}")

    print("[3/5] Mapping embeddings to variant splits...")

    split_data = get_split_tensors_flat(data, embeddings)

    print("[4/5] Running label-shuffle downstream training repeats...")

    rows = []
    for r in range(N_REPEATS):
        seed = SEED + 1000 + r
        print(f"Repeat {r + 1}/{N_REPEATS}, seed={seed}")
        metrics = train_one_repeat(split_data, repeat_id=r, seed=seed)
        print(metrics)
        rows.append(metrics)

    summary_stats = summarize_metrics(rows)

    ref, ref_path = load_reference_exp15b()

    print("[5/5] Saving outputs...")

    per_repeat_csv = OUT_DIR / "per_repeat_label_shuffle_metrics.csv"
    pd.DataFrame(rows).to_csv(per_repeat_csv, index=False)

    summary = {
        "experiment": "39A_random_feature_gcn_label_permutation_control",
        "real_data_policy": {
            "uses_real_model_inputs_main": str(MODEL_INPUTS),
            "uses_real_split_indices": True,
            "uses_real_validation_and_test_labels": True,
            "only_train_labels_shuffled": True,
            "no_fake_validation_or_test_labels": True,
            "no_simulated_predictions": True
        },
        "design": {
            "fixed_random_feature_gcn_embeddings": str(EMB_PATH),
            "feature_dim": FEATURE_DIM,
            "n_repeats": N_REPEATS,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "model_inputs_structure": "flat_keys_train_protein_idx_val_protein_idx_test_protein_idx_train_labels_val_labels_test_labels"
        },
        "summary_metrics": summary_stats,
        "per_repeat_metrics": rows,
        "reference_true_label_random_feature_gcn_if_found": {
            "path": ref_path,
            "summary": ref
        },
        "outputs": {
            "per_repeat_metrics_csv": str(per_repeat_csv),
            "val_probs_pattern": str(OUT_DIR / "repeat_<id>_val_probs.npy"),
            "test_probs_pattern": str(OUT_DIR / "repeat_<id>_test_probs.npy")
        }
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary_stats, indent=2))
    print(f"Results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
