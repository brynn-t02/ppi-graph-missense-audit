#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 47:
CADD+AlphaMissense fusion with graph-control embeddings.

Purpose:
Separate three possible sources of fusion gain under the original split:
1. the strong external variant-effect predictors CADD and AlphaMissense;
2. adding a protein-level graph-propagated ESM representation;
3. whether the graph-propagated representation still helps after removing
   direct train-test PPI edges or after permuting topology labels.

REAL DATA POLICY:
- Uses real ClinVar/STRING splits, labels, CADD matches, AlphaMissense matches,
  ESM node features and STRING graph.
- Controlled operations are only train-test edge removal and graph node-label
  permutation, matching Exp41 and Exp45.
- No fake labels and no simulated predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/root")
MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
CADD_PATH = ROOT / "data/processed/clinvar_with_cadd.csv"
AM_PATH = ROOT / "data/processed/clinvar_with_alphamissense.csv"

ORIGINAL_SPLITS = {
    "train": ROOT / "data/processed/clinvar_train_main.csv",
    "val": ROOT / "data/processed/clinvar_val_main.csv",
    "test": ROOT / "data/processed/clinvar_test_main.csv",
}

REAL_GCN_PATH = ROOT / "data/processed/gcn_embeddings/gcn_node_embeddings.pt"
EDGE_ABLATED_PATH = (
    ROOT
    / "data/processed/esm_gcn_edge_ablation_exp41/esm_feature_gcn_train_test_edge_ablated_embeddings.pt"
)
TOPOLOGY_PERM_ROOT = ROOT / "data/processed/esm_gcn_topology_permutation_exp45"

OUT_DIR = ROOT / "data/processed/cadd_am_graph_control_fusions_exp47"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 43, 44]
MAX_EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-5


class VariantDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class VariantMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        if in_dim <= 3:
            hidden1, hidden2, dropout = 32, 16, 0.2
        else:
            hidden1, hidden2, dropout = 512, 128, 0.3

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def evaluate(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
            probs_all.append(probs)
            labels_all.append(yb.numpy())

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    return {
        "auc": safe_auc(labels_all, probs_all),
        "auprc": safe_auprc(labels_all, probs_all),
        "probs": probs_all,
    }


def train_one(model_name, features, labels, seed, run_tag=None):
    set_seed(seed)
    x_train, x_val, x_test = features
    y_train, y_val, y_test = labels

    train_loader = torch.utils.data.DataLoader(
        VariantDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        VariantDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        VariantDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = VariantMLP(x_train.shape[1]).to(DEVICE)
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_seen += len(yb)

        val_metrics = evaluate(model, val_loader)
        print(
            f"{model_name} | seed {seed} | epoch {epoch:02d} | "
            f"loss={running_loss / n_seen:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | val_auprc={val_metrics['auprc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    safe_tag = run_tag or f"seed_{seed}"
    run_dir = OUT_DIR / model_name / safe_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "val_probs.npy", val_metrics["probs"])
    np.save(run_dir / "test_probs.npy", test_metrics["probs"])

    metrics = {
        "model": model_name,
        "run_tag": safe_tag,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": float(val_metrics["auc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_auprc": float(test_metrics["auprc"]),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"FINAL | {model_name} | {safe_tag} | "
        f"test_auc={metrics['test_auc']:.4f} | test_auprc={metrics['test_auprc']:.4f}"
    )
    return metrics


def load_external_tables():
    cadd = pd.read_csv(CADD_PATH, low_memory=False)
    am = pd.read_csv(AM_PATH, low_memory=False)
    cadd_cols = [c for c in ["VariationID", "CADD_PHRED", "CADD_RawScore"] if c in cadd.columns]
    am_cols = [c for c in ["VariationID", "am_pathogenicity", "am_class"] if c in am.columns]
    return (
        cadd[cadd_cols].drop_duplicates("VariationID").copy(),
        am[am_cols].drop_duplicates("VariationID").copy(),
    )


def load_embedding_tensor(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "embeddings" in obj:
            return obj["embeddings"].float()
        raise KeyError(f"Embedding dict at {path} does not contain 'embeddings'. Keys: {list(obj.keys())}")
    return obj.float()


def ablate_train_test_edges(data):
    edge_index = data["edge_index"]
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()

    num_nodes = int(data["x"].shape[0])
    train_nodes = torch.unique(data["train_protein_idx"].long())
    test_nodes = torch.unique(data["test_protein_idx"].long())

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_nodes] = True
    test_mask[test_nodes] = True

    row, col = edge_index[0], edge_index[1]
    is_train_test = (train_mask[row] & test_mask[col]) | (test_mask[row] & train_mask[col])
    keep = ~is_train_test

    stats = {
        "num_nodes": int(num_nodes),
        "num_original_edges": int(edge_index.shape[1]),
        "num_ablated_edges": int(is_train_test.sum().item()),
        "num_remaining_edges": int(keep.sum().item()),
        "fraction_edges_removed": float(is_train_test.float().mean().item()),
        "num_unique_train_nodes": int(len(train_nodes)),
        "num_unique_test_nodes": int(len(test_nodes)),
    }
    return edge_index[:, keep].contiguous(), stats


def make_topology_permuted_edge_index(edge_index, num_nodes, seed):
    rng = np.random.default_rng(seed)
    perm = torch.tensor(rng.permutation(num_nodes), dtype=torch.long)
    return perm[edge_index.long().cpu()].contiguous(), perm


def generate_fixed_gcn_embedding(data, edge_index, out_path):
    x = data["x"]
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float().cpu()

    num_nodes = int(x.shape[0])
    edge_index = edge_index.long().cpu()
    row, col = edge_index[0], edge_index[1]
    self_nodes = torch.arange(num_nodes, dtype=torch.long)

    all_row = torch.cat([row, col, self_nodes])
    all_col = torch.cat([col, row, self_nodes])

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
            out.index_add_(0, rr, feat[cc] * ww.unsqueeze(1))
        return out

    h1 = propagate(x)
    h2 = propagate(h1)
    h = (x + h1 + h2) / 3.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(h, out_path)
    return h


def prepare_original_split(payload, cadd_small, am_small):
    protein_to_idx = payload.get("protein_to_idx", payload.get("protein_id_to_node_idx"))
    if protein_to_idx is None:
        raise KeyError(f"No protein mapping found in {MAIN_INPUT_PATH}")
    protein_to_idx = {str(k): int(v) for k, v in protein_to_idx.items()}

    dfs = {}
    for part, path in ORIGINAL_SPLITS.items():
        df = pd.read_csv(path, low_memory=False)
        df = df.merge(cadd_small, on="VariationID", how="left")
        df = df.merge(am_small, on="VariationID", how="left")
        df["CADD_PHRED"] = pd.to_numeric(df["CADD_PHRED"], errors="coerce")
        df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")
        df["am_missing"] = df["am_pathogenicity"].isna().astype(float)

        df["STRING_protein_id"] = df["STRING_protein_id"].astype(str)
        df["node_idx"] = df["STRING_protein_id"].map(protein_to_idx)
        missing_nodes = int(df["node_idx"].isna().sum())
        if missing_nodes:
            print(f"WARNING: original/{part} drops {missing_nodes} rows without node mapping")
            df = df.dropna(subset=["node_idx"]).copy()
        df["node_idx"] = df["node_idx"].astype(int)
        dfs[part] = df

    train_cadd_fill = dfs["train"]["CADD_PHRED"].median()
    train_am_fill = dfs["train"]["am_pathogenicity"].median()
    for df in dfs.values():
        df["CADD_PHRED"] = df["CADD_PHRED"].fillna(train_cadd_fill)
        df["am_pathogenicity"] = df["am_pathogenicity"].fillna(train_am_fill)

    labels = tuple(
        torch.tensor(dfs[part]["Label"].astype(float).values, dtype=torch.float32)
        for part in ["train", "val", "test"]
    )
    node_idx = {
        part: torch.tensor(dfs[part]["node_idx"].values, dtype=torch.long)
        for part in ["train", "val", "test"]
    }
    x_cadd_am = tuple(
        torch.tensor(dfs[part][["CADD_PHRED", "am_pathogenicity", "am_missing"]].values, dtype=torch.float32)
        for part in ["train", "val", "test"]
    )

    audit = {
        "split": "original",
        "n_train": int(len(dfs["train"])),
        "n_val": int(len(dfs["val"])),
        "n_test": int(len(dfs["test"])),
        "train_positive_rate": float(dfs["train"]["Label"].mean()),
        "val_positive_rate": float(dfs["val"]["Label"].mean()),
        "test_positive_rate": float(dfs["test"]["Label"].mean()),
        "train_cadd_fill": float(train_cadd_fill),
        "train_am_fill": float(train_am_fill),
        "train_am_missing": int(dfs["train"]["am_missing"].sum()),
        "val_am_missing": int(dfs["val"]["am_missing"].sum()),
        "test_am_missing": int(dfs["test"]["am_missing"].sum()),
    }
    return labels, node_idx, x_cadd_am, audit


def concatenate_external_and_embedding(x_cadd_am, node_idx, embeddings):
    return tuple(
        torch.cat([x_cadd_am[i], embeddings[node_idx[part]]], dim=1)
        for i, part in enumerate(["train", "val", "test"])
    )


def get_or_make_edge_ablated_embedding(payload):
    if EDGE_ABLATED_PATH.exists():
        print(f"Reusing edge-ablated embedding: {EDGE_ABLATED_PATH}")
        return load_embedding_tensor(EDGE_ABLATED_PATH), {"reused_existing": True}

    edge_index, stats = ablate_train_test_edges(payload)
    print("Generating edge-ablated embedding because existing file was not found.")
    emb = generate_fixed_gcn_embedding(payload, edge_index, EDGE_ABLATED_PATH)
    stats["reused_existing"] = False
    stats["embedding_path"] = str(EDGE_ABLATED_PATH)
    return emb, stats


def get_or_make_topology_permuted_embeddings(payload):
    out = []
    edge_index = payload["edge_index"]
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long().cpu()
    num_nodes = int(payload["x"].shape[0])

    for graph_id in range(3):
        graph_dir = TOPOLOGY_PERM_ROOT / f"graph_permutation_{graph_id}"
        emb_path = graph_dir / "esm_gcn_topology_permuted_embedding.pt"
        if emb_path.exists():
            print(f"Reusing topology-permuted embedding: {emb_path}")
            emb = load_embedding_tensor(emb_path)
            out.append((graph_id, emb, {"reused_existing": True, "embedding_path": str(emb_path)}))
            continue

        graph_seed = 42 + 6000 + graph_id
        graph_dir.mkdir(parents=True, exist_ok=True)
        edge_perm, perm = make_topology_permuted_edge_index(edge_index, num_nodes, graph_seed)
        torch.save(edge_perm, graph_dir / "permuted_edge_index.pt")
        torch.save(perm, graph_dir / "node_permutation.pt")
        emb = generate_fixed_gcn_embedding(payload, edge_perm, emb_path)
        out.append(
            (
                graph_id,
                emb,
                {
                    "reused_existing": False,
                    "graph_seed": int(graph_seed),
                    "embedding_path": str(emb_path),
                },
            )
        )
    return out


def summarize(results):
    df = pd.DataFrame(results)
    summary_rows = []
    for model, sub in df.groupby("model"):
        summary_rows.append(
            {
                "model": model,
                "n_repeats": int(len(sub)),
                "test_auc_mean": float(sub["test_auc"].mean()),
                "test_auc_sd": float(sub["test_auc"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "test_auprc_mean": float(sub["test_auprc"].mean()),
                "test_auprc_sd": float(sub["test_auprc"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "val_auc_mean": float(sub["val_auc"].mean()),
                "val_auprc_mean": float(sub["val_auprc"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("model")
    s = summary.set_index("model")
    deltas = []
    baseline = "cadd_plus_am"
    if baseline in s.index:
        for model in s.index:
            if model == baseline:
                continue
            deltas.append(
                {
                    "comparison": f"{model}_vs_{baseline}",
                    "delta_test_auc_mean": float(s.loc[model, "test_auc_mean"] - s.loc[baseline, "test_auc_mean"]),
                    "delta_test_auprc_mean": float(
                        s.loc[model, "test_auprc_mean"] - s.loc[baseline, "test_auprc_mean"]
                    ),
                }
            )

    return df, summary, pd.DataFrame(deltas)


def main():
    print("=" * 80)
    print("Experiment 47: CADD+AlphaMissense graph-control fusion baselines")
    print("=" * 80)
    print("Device:", DEVICE)
    print("Output:", OUT_DIR)

    payload = torch.load(MAIN_INPUT_PATH, map_location="cpu", weights_only=False)
    cadd_small, am_small = load_external_tables()
    labels, node_idx, x_cadd_am, split_audit = prepare_original_split(payload, cadd_small, am_small)
    print("Split audit:", json.dumps(split_audit, indent=2))

    feature_sets = {"cadd_plus_am": x_cadd_am}
    embedding_audits = {}

    if REAL_GCN_PATH.exists():
        real_gcn = load_embedding_tensor(REAL_GCN_PATH)
        feature_sets["cadd_am_plus_real_ppi_gcn"] = concatenate_external_and_embedding(
            x_cadd_am, node_idx, real_gcn
        )
        embedding_audits["real_ppi_gcn"] = {"available": True, "path": str(REAL_GCN_PATH)}
    else:
        embedding_audits["real_ppi_gcn"] = {"available": False, "path": str(REAL_GCN_PATH)}

    edge_ablated, edge_audit = get_or_make_edge_ablated_embedding(payload)
    feature_sets["cadd_am_plus_edge_ablated_ppi_gcn"] = concatenate_external_and_embedding(
        x_cadd_am, node_idx, edge_ablated
    )
    embedding_audits["edge_ablated_ppi_gcn"] = edge_audit

    topology_embeddings = get_or_make_topology_permuted_embeddings(payload)
    topology_feature_sets = []
    for graph_id, emb, audit in topology_embeddings:
        topology_feature_sets.append(
            (
                graph_id,
                concatenate_external_and_embedding(x_cadd_am, node_idx, emb),
                audit,
            )
        )
    embedding_audits["topology_permuted_ppi_gcn"] = [x[2] for x in topology_feature_sets]

    all_results = []
    for model_name, features in feature_sets.items():
        for seed in SEEDS:
            all_results.append(train_one(model_name, features, labels, seed))

    for graph_id, features, _audit in topology_feature_sets:
        for seed in SEEDS:
            all_results.append(
                train_one(
                    "cadd_am_plus_topology_permuted_ppi_gcn",
                    features,
                    labels,
                    seed,
                    run_tag=f"graphperm_{graph_id}_seed_{seed}",
                )
            )

    per_repeat, summary, deltas = summarize(all_results)
    per_repeat.to_csv(OUT_DIR / "per_repeat_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    deltas.to_csv(OUT_DIR / "delta_summary.csv", index=False)

    out_json = {
        "experiment": "47_cadd_am_graph_control_fusion_baselines",
        "purpose": (
            "Test whether CADD+AlphaMissense fusion gains remain after replacing real PPI-GCN "
            "with train-test edge-ablated or topology-permuted graph-propagated ESM representations."
        ),
        "seeds": SEEDS,
        "split_audit": split_audit,
        "embedding_audits": embedding_audits,
        "summary": summary.to_dict(orient="records"),
        "deltas": deltas.to_dict(orient="records"),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print("\nSummary metrics")
    print(summary.to_string(index=False))
    print("\nDelta summary")
    print(deltas.to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
