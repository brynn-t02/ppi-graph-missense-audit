import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
OUT_DIR = ROOT / "data/processed/degree_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("📉 Step 04a: Run degree-only baseline")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)

# -------------------------------------------------------------------
# 1. Load input package
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)

edge_index = payload["edge_index"].long()
num_nodes = payload["num_nodes"]

train_idx = payload["train_protein_idx"].long()
val_idx   = payload["val_protein_idx"].long()
test_idx  = payload["test_protein_idx"].long()

train_labels = payload["train_labels"].float()
val_labels   = payload["val_labels"].float()
test_labels  = payload["test_labels"].float()

print("\n[1] Loaded payload")
print("edge_index     :", tuple(edge_index.shape))
print("num_nodes      :", num_nodes)
print("Train variants :", len(train_labels))
print("Val variants   :", len(val_labels))
print("Test variants  :", len(test_labels))

# -------------------------------------------------------------------
# 2. Compute node degree
# -------------------------------------------------------------------
src = edge_index[0]
dst = edge_index[1]

deg = torch.bincount(src, minlength=num_nodes) + torch.bincount(dst, minlength=num_nodes)
deg = deg.float()

# log transform
deg_feat = torch.log1p(deg).unsqueeze(1)  # [N_nodes, 1]

print("\n[2] Degree feature")
print("deg_feat shape:", tuple(deg_feat.shape))
print("Degree stats:")
print("min :", float(deg.min()))
print("max :", float(deg.max()))
print("mean:", float(deg.mean()))

# -------------------------------------------------------------------
# 3. Map variants to node features
# -------------------------------------------------------------------
X_train = deg_feat[train_idx]
X_val   = deg_feat[val_idx]
X_test  = deg_feat[test_idx]

print("\n[3] Variant-level degree features")
print("X_train:", tuple(X_train.shape))
print("X_val  :", tuple(X_val.shape))
print("X_test :", tuple(X_test.shape))

class VariantDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(
    VariantDataset(X_train, train_labels),
    batch_size=1024,
    shuffle=True,
    num_workers=0,
)

val_loader = torch.utils.data.DataLoader(
    VariantDataset(X_val, val_labels),
    batch_size=1024,
    shuffle=False,
    num_workers=0,
)

test_loader = torch.utils.data.DataLoader(
    VariantDataset(X_test, test_labels),
    batch_size=1024,
    shuffle=False,
    num_workers=0,
)

# -------------------------------------------------------------------
# 4. Small MLP
# -------------------------------------------------------------------
class DegreeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = DegreeMLP().to(DEVICE)

pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_probs = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

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
        "labels": all_labels,
    }

# -------------------------------------------------------------------
# 5. Training
# -------------------------------------------------------------------
max_epochs = 50
patience = 5
best_val_auc = -np.inf
best_state = None
best_epoch = -1
patience_counter = 0

print("\n[4] Training degree-only baseline...")

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
    val_metrics = evaluate(val_loader)

    print(
        f"Epoch {epoch:02d} | "
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
            print(f"\nEarly stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
model = model.to(DEVICE)

print("\n[5] Best validation checkpoint")
print("Best epoch  :", best_epoch)
print("Best val AUC:", f"{best_val_auc:.4f}")

val_metrics = evaluate(val_loader)
test_metrics = evaluate(test_loader)

print("\n[6] Final performance")
print(f"Val  AUC   : {val_metrics['auc']:.4f}")
print(f"Val  AUPRC : {val_metrics['auprc']:.4f}")
print(f"Test AUC   : {test_metrics['auc']:.4f}")
print(f"Test AUPRC : {test_metrics['auprc']:.4f}")

torch.save(model.state_dict(), OUT_DIR / "degree_best.pt")
np.save(OUT_DIR / "val_probs.npy", val_metrics["probs"])
np.save(OUT_DIR / "test_probs.npy", test_metrics["probs"])

metrics = {
    "best_epoch": int(best_epoch),
    "best_val_auc": float(best_val_auc),
    "val_auc": float(val_metrics["auc"]),
    "val_auprc": float(val_metrics["auprc"]),
    "test_auc": float(test_metrics["auc"]),
    "test_auprc": float(test_metrics["auprc"]),
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n[7] Saved outputs")
print("✅", OUT_DIR / "degree_best.pt")
print("✅", OUT_DIR / "metrics.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
