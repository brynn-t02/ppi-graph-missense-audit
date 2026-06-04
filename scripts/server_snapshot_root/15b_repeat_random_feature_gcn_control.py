import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
DEGREE_METRICS_PATH = ROOT / "data/processed/degree_baseline/metrics.json"
FEATURE_SHUFFLE_METRICS_PATH = ROOT / "data/processed/gcn_feature_shuffled_classifier/metrics.json"

OUT_DIR = ROOT / "data/processed/random_feature_gcn_control_repeated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 42
N_REPEATS = 3
RANDOM_FEATURE_DIM = 1280

print("=" * 80)
print("🧪 Step 15b: Repeated random-feature GCN control")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("Degree file:", DEGREE_METRICS_PATH)
print("Shuffle file:", FEATURE_SHUFFLE_METRICS_PATH)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)
print("Repeats    :", N_REPEATS)

# -------------------------------------------------------------------
# 1. Load payload
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)

x = payload["x"].float()
edge_index = payload["edge_index"].long()

train_idx = payload["train_protein_idx"].long()
val_idx   = payload["val_protein_idx"].long()
test_idx  = payload["test_protein_idx"].long()

train_labels = payload["train_labels"].float()
val_labels   = payload["val_labels"].float()
test_labels  = payload["test_labels"].float()

num_nodes = x.shape[0]

print("\n[1] Loaded main payload")
print("Original feature shape:", tuple(x.shape))
print("Edge index shape      :", tuple(edge_index.shape))
print("Train rows            :", len(train_labels))
print("Val rows              :", len(val_labels))
print("Test rows             :", len(test_labels))

# -------------------------------------------------------------------
# 2. Existing baselines
# -------------------------------------------------------------------
def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

degree_metrics = load_json(DEGREE_METRICS_PATH)
feature_shuffle_metrics = load_json(FEATURE_SHUFFLE_METRICS_PATH)

if degree_metrics is not None:
    print("\n[2] Loaded degree-only baseline")
    print("Degree-only test AUC  :", degree_metrics.get("test_auc"))
    print("Degree-only test AUPRC:", degree_metrics.get("test_auprc"))

if feature_shuffle_metrics is not None:
    print("\n[3] Loaded feature-shuffled baseline")
    print("Feature-shuffled test AUC  :", feature_shuffle_metrics.get("test_auc"))
    print("Feature-shuffled test AUPRC:", feature_shuffle_metrics.get("test_auprc"))

# -------------------------------------------------------------------
# 3. Build normalized adjacency once
# -------------------------------------------------------------------
def build_normalized_adj(edge_index, num_nodes):
    row, col = edge_index
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_loops = torch.stack([self_loops, self_loops], dim=0)

    edge_index2 = torch.cat([edge_index, self_loops], dim=1)

    row2, col2 = edge_index2
    deg = torch.bincount(row2, minlength=num_nodes).float()

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[row2] * deg_inv_sqrt[col2]

    adj = torch.sparse_coo_tensor(
        indices=edge_index2,
        values=values,
        size=(num_nodes, num_nodes)
    ).coalesce()

    return adj

print("\n[4] Building normalized adjacency...")
adj = build_normalized_adj(edge_index, num_nodes)

# -------------------------------------------------------------------
# 4. Dataset / model
# -------------------------------------------------------------------
class VariantDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class VariantMLP(nn.Module):
    def __init__(self, in_dim, hidden1=512, hidden2=128, dropout=0.3):
        super().__init__()
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

def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "probs": all_probs,
    }

def run_one_repeat(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # random node features
    rand_x = torch.randn((num_nodes, RANDOM_FEATURE_DIM), dtype=torch.float32)

    # 2-layer GCN-style propagation
    H1 = torch.sparse.mm(adj, rand_x)
    H2 = torch.sparse.mm(adj, H1)
    emb = H2

    X_train = emb[train_idx]
    X_val   = emb[val_idx]
    X_test  = emb[test_idx]

    train_loader = torch.utils.data.DataLoader(
        VariantDataset(X_train, train_labels),
        batch_size=1024, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        VariantDataset(X_val, val_labels),
        batch_size=1024, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        VariantDataset(X_test, test_labels),
        batch_size=1024, shuffle=False, num_workers=0
    )

    model = VariantMLP(in_dim=X_train.shape[1]).to(DEVICE)

    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    patience = 5
    patience_counter = 0
    max_epochs = 50

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(yb)
            n_samples += len(yb)

        train_loss = running_loss / n_samples
        val_metrics = evaluate(model, val_loader)

        print(
            f"random_gcn | seed={seed} | Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping for seed={seed} at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "val_auc": float(val_metrics["auc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_auprc": float(test_metrics["auprc"]),
    }

# -------------------------------------------------------------------
# 5. Run repeats
# -------------------------------------------------------------------
print("\n[5] Running repeated random-feature control...")
repeat_results = []

for r in range(N_REPEATS):
    seed = BASE_SEED + r
    print("\n" + "-" * 60)
    print(f"[Repeat {r+1}/{N_REPEATS}] seed={seed}")
    print("-" * 60)
    res = run_one_repeat(seed)
    repeat_results.append(res)

# -------------------------------------------------------------------
# 6. Summaries
# -------------------------------------------------------------------
val_auc = [x["val_auc"] for x in repeat_results]
val_auprc = [x["val_auprc"] for x in repeat_results]
test_auc = [x["test_auc"] for x in repeat_results]
test_auprc = [x["test_auprc"] for x in repeat_results]

summary = {
    "n_repeats": N_REPEATS,
    "random_feature_gcn_repeated": {
        "val_auc_mean": float(np.mean(val_auc)),
        "val_auc_std": float(np.std(val_auc)),
        "val_auprc_mean": float(np.mean(val_auprc)),
        "val_auprc_std": float(np.std(val_auprc)),
        "test_auc_mean": float(np.mean(test_auc)),
        "test_auc_std": float(np.std(test_auc)),
        "test_auprc_mean": float(np.mean(test_auprc)),
        "test_auprc_std": float(np.std(test_auprc)),
    },
    "per_repeat_results": repeat_results,
    "degree_only": degree_metrics,
    "feature_shuffled_gcn": feature_shuffle_metrics,
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(repeat_results).to_csv(OUT_DIR / "per_repeat_results.csv", index=False)

summary_table = pd.DataFrame([
    {
        "model": "degree_only",
        "test_auc_mean": degree_metrics["test_auc"] if degree_metrics else np.nan,
        "test_auc_std": 0.0 if degree_metrics else np.nan,
        "test_auprc_mean": degree_metrics["test_auprc"] if degree_metrics else np.nan,
        "test_auprc_std": 0.0 if degree_metrics else np.nan,
    },
    {
        "model": "feature_shuffled_gcn",
        "test_auc_mean": feature_shuffle_metrics["test_auc"] if feature_shuffle_metrics else np.nan,
        "test_auc_std": 0.0 if feature_shuffle_metrics else np.nan,
        "test_auprc_mean": feature_shuffle_metrics["test_auprc"] if feature_shuffle_metrics else np.nan,
        "test_auprc_std": 0.0 if feature_shuffle_metrics else np.nan,
    },
    {
        "model": "random_feature_gcn_repeated",
        "test_auc_mean": float(np.mean(test_auc)),
        "test_auc_std": float(np.std(test_auc)),
        "test_auprc_mean": float(np.mean(test_auprc)),
        "test_auprc_std": float(np.std(test_auprc)),
    },
])

summary_table.to_csv(OUT_DIR / "comparison_summary.csv", index=False)

print("\n[6] Repeated random-feature summary")
print(f"Val  AUC   : {np.mean(val_auc):.4f} ± {np.std(val_auc):.4f}")
print(f"Val  AUPRC : {np.mean(val_auprc):.4f} ± {np.std(val_auprc):.4f}")
print(f"Test AUC   : {np.mean(test_auc):.4f} ± {np.std(test_auc):.4f}")
print(f"Test AUPRC : {np.mean(test_auprc):.4f} ± {np.std(test_auprc):.4f}")

print("\n[7] Comparison summary")
print(summary_table.to_string(index=False))

print("\n✅ Saved outputs:")
print("✅", OUT_DIR / "summary.json")
print("✅", OUT_DIR / "per_repeat_results.csv")
print("✅", OUT_DIR / "comparison_summary.csv")
print("=" * 80)
print("Done.")
print("=" * 80)
