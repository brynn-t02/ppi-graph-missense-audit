import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
GCN_EMB_PATH = ROOT / "data/processed/gcn_embeddings/gcn_node_embeddings.pt"
SAGE_EMB_PATH = ROOT / "data/processed/graphsage_embeddings/graphsage_node_embeddings.pt"
GAT_EMB_PATH = ROOT / "data/processed/gat_embeddings/gat_node_embeddings.pt"

OUT_ROOT = ROOT / "data/processed/variant_mlp_comparison"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("📈 Step 03f: Run variant-level MLP on four protein embedding sources")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("SAGE emb   :", SAGE_EMB_PATH)
print("GAT emb    :", GAT_EMB_PATH)
print("Outdir     :", OUT_ROOT)
print("Device     :", DEVICE)

# -------------------------------------------------------------------
# 1. Load shared split/index package
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)

train_idx = payload["train_protein_idx"].long()
val_idx   = payload["val_protein_idx"].long()
test_idx  = payload["test_protein_idx"].long()

train_labels = payload["train_labels"].float()
val_labels   = payload["val_labels"].float()
test_labels  = payload["test_labels"].float()

esm_embeddings = payload["x"].float()
gcn_payload = torch.load(GCN_EMB_PATH, weights_only=False)
gcn_embeddings = gcn_payload["embeddings"].float()

sage_payload = torch.load(SAGE_EMB_PATH, weights_only=False)
sage_embeddings = sage_payload["embeddings"].float()

gat_payload = torch.load(GAT_EMB_PATH, weights_only=False)
gat_embeddings = gat_payload["embeddings"].float()

print("\n[1] Loaded embeddings")
print("ESM embeddings :", tuple(esm_embeddings.shape))
print("GCN embeddings :", tuple(gcn_embeddings.shape))
print("SAGE embeddings:", tuple(sage_embeddings.shape))
print("GAT embeddings :", tuple(gat_embeddings.shape))

# -------------------------------------------------------------------
# 2. Dataset / Dataloader
# -------------------------------------------------------------------
class VariantDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------------------------------------------
# 3. Classifier
# -------------------------------------------------------------------
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

@torch.no_grad()
def evaluate(model, loader):
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

def train_one_embedding(embedding_name, node_embeddings):
    print("\n" + "=" * 60)
    print(f"[Training classifier on {embedding_name}]")
    print("=" * 60)

    X_train = node_embeddings[train_idx]
    X_val   = node_embeddings[val_idx]
    X_test  = node_embeddings[test_idx]

    print("X_train:", tuple(X_train.shape))
    print("X_val  :", tuple(X_val.shape))
    print("X_test :", tuple(X_test.shape))

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

    model = VariantMLP(in_dim=node_embeddings.shape[1]).to(DEVICE)

    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    max_epochs = 50
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
            f"{embedding_name} | Epoch {epoch:02d} | "
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
                print(f"Early stopping for {embedding_name} at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    out_dir = OUT_ROOT / embedding_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "best.pt")
    np.save(out_dir / "val_probs.npy", val_metrics["probs"])
    np.save(out_dir / "test_probs.npy", test_metrics["probs"])

    metrics = {
        "embedding_name": embedding_name,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": float(val_metrics["auc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_auprc": float(test_metrics["auprc"]),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{embedding_name}] Final performance")
    print(f"Val  AUC   : {val_metrics['auc']:.4f}")
    print(f"Val  AUPRC : {val_metrics['auprc']:.4f}")
    print(f"Test AUC   : {test_metrics['auc']:.4f}")
    print(f"Test AUPRC : {test_metrics['auprc']:.4f}")

    return metrics

# -------------------------------------------------------------------
# 4. Run all comparisons
# -------------------------------------------------------------------
esm_metrics = train_one_embedding("esm_raw", esm_embeddings)
gcn_metrics = train_one_embedding("gcn_enhanced", gcn_embeddings)
sage_metrics = train_one_embedding("graphsage_enhanced", sage_embeddings)
gat_metrics = train_one_embedding("gat_enhanced", gat_embeddings)

summary = {
    "esm_raw": esm_metrics,
    "gcn_enhanced": gcn_metrics,
    "graphsage_enhanced": sage_metrics,
    "gat_enhanced": gat_metrics,

    "delta_auc_gcn_vs_esm": gcn_metrics["test_auc"] - esm_metrics["test_auc"],
    "delta_auprc_gcn_vs_esm": gcn_metrics["test_auprc"] - esm_metrics["test_auprc"],

    "delta_auc_sage_vs_esm": sage_metrics["test_auc"] - esm_metrics["test_auc"],
    "delta_auprc_sage_vs_esm": sage_metrics["test_auprc"] - esm_metrics["test_auprc"],

    "delta_auc_gat_vs_esm": gat_metrics["test_auc"] - esm_metrics["test_auc"],
    "delta_auprc_gat_vs_esm": gat_metrics["test_auprc"] - esm_metrics["test_auprc"],
}

with open(OUT_ROOT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("Comparison summary")
print("=" * 80)
print("ESM raw test AUC            :", f"{esm_metrics['test_auc']:.4f}")
print("GCN-enhanced test AUC       :", f"{gcn_metrics['test_auc']:.4f}")
print("GraphSAGE-enhanced test AUC :", f"{sage_metrics['test_auc']:.4f}")
print("GAT-enhanced test AUC       :", f"{gat_metrics['test_auc']:.4f}")

print("ΔAUC (GCN vs ESM)           :", f"{summary['delta_auc_gcn_vs_esm']:.4f}")
print("ΔAUC (SAGE vs ESM)          :", f"{summary['delta_auc_sage_vs_esm']:.4f}")
print("ΔAUC (GAT vs ESM)           :", f"{summary['delta_auc_gat_vs_esm']:.4f}")

print("ESM raw test AUPRC            :", f"{esm_metrics['test_auprc']:.4f}")
print("GCN-enhanced test AUPRC       :", f"{gcn_metrics['test_auprc']:.4f}")
print("GraphSAGE-enhanced test AUPRC :", f"{sage_metrics['test_auprc']:.4f}")
print("GAT-enhanced test AUPRC       :", f"{gat_metrics['test_auprc']:.4f}")

print("ΔAUPRC (GCN vs ESM)           :", f"{summary['delta_auprc_gcn_vs_esm']:.4f}")
print("ΔAUPRC (SAGE vs ESM)          :", f"{summary['delta_auprc_sage_vs_esm']:.4f}")
print("ΔAUPRC (GAT vs ESM)           :", f"{summary['delta_auprc_gat_vs_esm']:.4f}")

print("\n✅ Saved summary:", OUT_ROOT / "summary.json")
print("=" * 80)
print("Done.")
print("=" * 80)
