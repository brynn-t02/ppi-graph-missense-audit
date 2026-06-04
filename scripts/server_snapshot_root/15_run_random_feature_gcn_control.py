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

OUT_DIR = ROOT / "data/processed/random_feature_gcn_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
RANDOM_FEATURE_DIM = 1280

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("🧪 Step 15: Random-feature GCN control")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("Degree file:", DEGREE_METRICS_PATH)
print("Shuffle file:", FEATURE_SHUFFLE_METRICS_PATH)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)

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
# 2. Load existing baseline/control metrics if available
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
# 3. Build random node features
# -------------------------------------------------------------------
rand_x = torch.randn((num_nodes, RANDOM_FEATURE_DIM), dtype=torch.float32)
print("\n[4] Random features")
print("Random feature shape:", tuple(rand_x.shape))

# -------------------------------------------------------------------
# 4. Simple GCN-style propagation
#    H1 = A_hat X
#    H2 = A_hat H1
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

print("\n[5] Building normalized adjacency...")
adj = build_normalized_adj(edge_index, num_nodes)

print("Running 2-layer propagation on random features...")
H1 = torch.sparse.mm(adj, rand_x)
H2 = torch.sparse.mm(adj, H1)

random_gcn_embeddings = H2
print("Random GCN embedding shape:", tuple(random_gcn_embeddings.shape))

# -------------------------------------------------------------------
# 5. Build split matrices
# -------------------------------------------------------------------
X_train = random_gcn_embeddings[train_idx]
X_val   = random_gcn_embeddings[val_idx]
X_test  = random_gcn_embeddings[test_idx]

print("\n[6] Split feature matrices")
print("Train:", tuple(X_train.shape))
print("Val  :", tuple(X_val.shape))
print("Test :", tuple(X_test.shape))

# -------------------------------------------------------------------
# 6. Dataset / model
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

# -------------------------------------------------------------------
# 7. Train random-feature GCN classifier
# -------------------------------------------------------------------
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

print("\n[7] Training random-feature GCN classifier...")
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
        f"random_gcn | Epoch {epoch:02d} | "
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
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
model = model.to(DEVICE)

val_metrics = evaluate(model, val_loader)
test_metrics = evaluate(model, test_loader)

metrics = {
    "best_epoch": int(best_epoch),
    "val_auc": float(val_metrics["auc"]),
    "val_auprc": float(val_metrics["auprc"]),
    "test_auc": float(test_metrics["auc"]),
    "test_auprc": float(test_metrics["auprc"]),
}

torch.save(model.state_dict(), OUT_DIR / "best.pt")
np.save(OUT_DIR / "val_probs.npy", val_metrics["probs"])
np.save(OUT_DIR / "test_probs.npy", test_metrics["probs"])

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# -------------------------------------------------------------------
# 8. Consolidated summary
# -------------------------------------------------------------------
summary = {
    "random_feature_gcn": metrics,
    "degree_only": degree_metrics,
    "feature_shuffled_gcn": feature_shuffle_metrics,
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[8] Final random-feature GCN")
print("Val  AUC   :", f"{metrics['val_auc']:.4f}")
print("Val  AUPRC :", f"{metrics['val_auprc']:.4f}")
print("Test AUC   :", f"{metrics['test_auc']:.4f}")
print("Test AUPRC :", f"{metrics['test_auprc']:.4f}")

if degree_metrics is not None or feature_shuffle_metrics is not None:
    print("\n[9] Comparison summary")
    if degree_metrics is not None:
        print("Degree-only test AUC        :", f"{degree_metrics['test_auc']:.4f}")
        print("Degree-only test AUPRC      :", f"{degree_metrics['test_auprc']:.4f}")
    if feature_shuffle_metrics is not None:
        print("Feature-shuffled test AUC   :", f"{feature_shuffle_metrics['test_auc']:.4f}")
        print("Feature-shuffled test AUPRC :", f"{feature_shuffle_metrics['test_auprc']:.4f}")
    print("Random-feature GCN test AUC :", f"{metrics['test_auc']:.4f}")
    print("Random-feature GCN test AUPRC:", f"{metrics['test_auprc']:.4f}")

print("\n✅ Saved outputs:")
print("✅", OUT_DIR / "metrics.json")
print("✅", OUT_DIR / "summary.json")
print("=" * 80)
print("Done.")
print("=" * 80)
