import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
GCN_EMB_PATH = ROOT / "data/processed/gcn_embeddings/gcn_node_embeddings.pt"

OUT_DIR = ROOT / "data/processed/label_shuffle_repeats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 42
N_REPEATS = 5

torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)

print("=" * 80)
print("🧪 Step 04e: Repeated label-shuffle control on GCN-enhanced embeddings")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)
print("Repeats    :", N_REPEATS)

# -------------------------------------------------------------------
# 1. Load split package and embeddings
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)

train_idx = payload["train_protein_idx"].long()
val_idx   = payload["val_protein_idx"].long()
test_idx  = payload["test_protein_idx"].long()

train_labels = payload["train_labels"].float()
val_labels   = payload["val_labels"].float()
test_labels  = payload["test_labels"].float()

gcn_payload = torch.load(GCN_EMB_PATH, weights_only=False)
gcn_embeddings = gcn_payload["embeddings"].float()

print("\n[1] Loaded GCN-enhanced embeddings")
print("Embeddings shape:", tuple(gcn_embeddings.shape))

# -------------------------------------------------------------------
# 2. Build variant-level matrices
# -------------------------------------------------------------------
X_train = gcn_embeddings[train_idx]
X_val   = gcn_embeddings[val_idx]
X_test  = gcn_embeddings[test_idx]

print("\n[2] Built variant-level matrices")
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

class VariantMLP(nn.Module):
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

def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
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
    }

# -------------------------------------------------------------------
# 3. Repeated label shuffle runs
# -------------------------------------------------------------------
all_results = []

for repeat_idx in range(N_REPEATS):
    run_seed = BASE_SEED + repeat_idx
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    print("\n" + "-" * 60)
    print(f"[Repeat {repeat_idx+1}/{N_REPEATS}] seed={run_seed}")
    print("-" * 60)

    perm = torch.randperm(len(train_labels))
    train_labels_shuffled = train_labels[perm].clone()

    train_loader = torch.utils.data.DataLoader(
        VariantDataset(X_train, train_labels_shuffled),
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

    model = VariantMLP(in_dim=gcn_embeddings.shape[1]).to(DEVICE)

    pos_weight = (len(train_labels_shuffled) - train_labels_shuffled.sum()) / train_labels_shuffled.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    max_epochs = 30
    patience = 5
    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    patience_counter = 0

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
            f"Repeat {repeat_idx+1} | Epoch {epoch:02d} | "
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

    result = {
        "repeat": repeat_idx + 1,
        "seed": run_seed,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": float(val_metrics["auc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_auprc": float(test_metrics["auprc"]),
    }
    all_results.append(result)

    print(f"[Repeat {repeat_idx+1}] Final")
    print(f"Val  AUC   : {val_metrics['auc']:.4f}")
    print(f"Val  AUPRC : {val_metrics['auprc']:.4f}")
    print(f"Test AUC   : {test_metrics['auc']:.4f}")
    print(f"Test AUPRC : {test_metrics['auprc']:.4f}")

# -------------------------------------------------------------------
# 4. Summarize
# -------------------------------------------------------------------
test_auc_list = [r["test_auc"] for r in all_results]
test_auprc_list = [r["test_auprc"] for r in all_results]
val_auc_list = [r["val_auc"] for r in all_results]
val_auprc_list = [r["val_auprc"] for r in all_results]

summary = {
    "n_repeats": N_REPEATS,
    "val_auc_mean": float(np.mean(val_auc_list)),
    "val_auc_std": float(np.std(val_auc_list)),
    "val_auprc_mean": float(np.mean(val_auprc_list)),
    "val_auprc_std": float(np.std(val_auprc_list)),
    "test_auc_mean": float(np.mean(test_auc_list)),
    "test_auc_std": float(np.std(test_auc_list)),
    "test_auprc_mean": float(np.mean(test_auprc_list)),
    "test_auprc_std": float(np.std(test_auprc_list)),
    "per_repeat": all_results,
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("Repeated label-shuffle summary")
print("=" * 80)
print(f"Val  AUC   : {summary['val_auc_mean']:.4f} ± {summary['val_auc_std']:.4f}")
print(f"Val  AUPRC : {summary['val_auprc_mean']:.4f} ± {summary['val_auprc_std']:.4f}")
print(f"Test AUC   : {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}")
print(f"Test AUPRC : {summary['test_auprc_mean']:.4f} ± {summary['test_auprc_std']:.4f}")
print("\n✅ Saved summary:", OUT_DIR / "summary.json")
print("=" * 80)
print("Done.")
print("=" * 80)
