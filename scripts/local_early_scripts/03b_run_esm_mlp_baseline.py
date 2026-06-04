import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/processed/model_inputs_main.pt"
OUT_DIR = ROOT / "data/processed/esm_mlp_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("📈 Step 03b: Run ESM-only strong baseline (3-layer MLP)")
print("=" * 80)
print("Input :", IN_PATH)
print("Outdir :", OUT_DIR)
print("Device:", DEVICE)

# -------------------------------------------------------------------
# 1. Load prepared package
# -------------------------------------------------------------------
payload = torch.load(IN_PATH, weights_only=False)

x = payload["x"]  # [N_nodes, 1280]

train_idx = payload["train_protein_idx"]
val_idx   = payload["val_protein_idx"]
test_idx  = payload["test_protein_idx"]

train_labels = payload["train_labels"].float()
val_labels   = payload["val_labels"].float()
test_labels  = payload["test_labels"].float()

print("\n[1] Loaded payload")
print("Node feature matrix:", tuple(x.shape))
print("Train variants:", len(train_labels))
print("Val variants  :", len(val_labels))
print("Test variants :", len(test_labels))

# Build variant-level inputs
X_train = x[train_idx]
X_val   = x[val_idx]
X_test  = x[test_idx]

print("\n[2] Variant-level tensors")
print("X_train:", tuple(X_train.shape))
print("X_val  :", tuple(X_val.shape))
print("X_test :", tuple(X_test.shape))

# -------------------------------------------------------------------
# 2. Dataset / DataLoader
# -------------------------------------------------------------------
class VariantDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 1024

train_loader = torch.utils.data.DataLoader(
    VariantDataset(X_train, train_labels),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = torch.utils.data.DataLoader(
    VariantDataset(X_val, val_labels),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

test_loader = torch.utils.data.DataLoader(
    VariantDataset(X_test, test_labels),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

# -------------------------------------------------------------------
# 3. Model
# -------------------------------------------------------------------
class ESMMLP(nn.Module):
    def __init__(self, in_dim=1280, hidden1=512, hidden2=128, dropout=0.3):
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

model = ESMMLP().to(DEVICE)

# Class imbalance handling
pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print("\n[3] Model setup")
print(model)
print("pos_weight:", float(pos_weight))

# -------------------------------------------------------------------
# 4. Evaluation helper
# -------------------------------------------------------------------
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
# 5. Training loop with early stopping
# -------------------------------------------------------------------
max_epochs = 50
patience = 5
best_val_auc = -np.inf
best_state = None
best_epoch = -1
patience_counter = 0

print("\n[4] Training...")

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
    val_auc = val_metrics["auc"]
    val_auprc = val_metrics["auprc"]

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_auc={val_auc:.4f} | "
        f"val_auprc={val_auprc:.4f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

# Restore best model
model.load_state_dict(best_state)

print("\n[5] Best validation checkpoint")
print("Best epoch :", best_epoch)
print("Best val AUC:", f"{best_val_auc:.4f}")

# -------------------------------------------------------------------
# 6. Final evaluation
# -------------------------------------------------------------------
val_metrics = evaluate(val_loader)
test_metrics = evaluate(test_loader)

print("\n[6] Final performance")
print(f"Val  AUC   : {val_metrics['auc']:.4f}")
print(f"Val  AUPRC : {val_metrics['auprc']:.4f}")
print(f"Test AUC   : {test_metrics['auc']:.4f}")
print(f"Test AUPRC : {test_metrics['auprc']:.4f}")

# -------------------------------------------------------------------
# 7. Save outputs
# -------------------------------------------------------------------
torch.save(model.state_dict(), OUT_DIR / "esm_mlp_best.pt")
np.save(OUT_DIR / "val_probs.npy", val_metrics["probs"])
np.save(OUT_DIR / "val_labels.npy", val_metrics["labels"])
np.save(OUT_DIR / "test_probs.npy", test_metrics["probs"])
np.save(OUT_DIR / "test_labels.npy", test_metrics["labels"])

metrics = {
    "best_epoch": int(best_epoch),
    "best_val_auc": float(best_val_auc),
    "val_auc": float(val_metrics["auc"]),
    "val_auprc": float(val_metrics["auprc"]),
    "test_auc": float(test_metrics["auc"]),
    "test_auprc": float(test_metrics["auprc"]),
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "patience": patience,
    "device": DEVICE,
    "seed": SEED,
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n[7] Saved outputs")
print("✅", OUT_DIR / "esm_mlp_best.pt")
print("✅", OUT_DIR / "val_probs.npy")
print("✅", OUT_DIR / "test_probs.npy")
print("✅", OUT_DIR / "metrics.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
