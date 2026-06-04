import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
GCN_EMB_PATH = ROOT / "data/processed/gcn_embeddings/gcn_node_embeddings.pt"
AM_MERGED_PATH = ROOT / "data/processed/clinvar_with_alphamissense.csv"

OUT_DIR = ROOT / "data/processed/alphamissense_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("🧬 Step 07d: Run AlphaMissense baseline and GCN+AM fusion")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("AM table   :", AM_MERGED_PATH)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)

# -------------------------------------------------------------------
# 1. Load inputs
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)
gcn_payload = torch.load(GCN_EMB_PATH, weights_only=False)

train_idx = payload["train_protein_idx"].long()
val_idx   = payload["val_protein_idx"].long()
test_idx  = payload["test_protein_idx"].long()

train_labels = payload["train_labels"].float()
val_labels   = payload["val_labels"].float()
test_labels  = payload["test_labels"].float()

gcn_embeddings = gcn_payload["embeddings"].float()
clinvar_am = pd.read_csv(AM_MERGED_PATH, low_memory=False)

print("\n[1] Loaded inputs")
print("GCN embeddings   :", tuple(gcn_embeddings.shape))
print("ClinVar+AM rows  :", len(clinvar_am))

# -------------------------------------------------------------------
# 2. Load split tables and merge AM scores by VariationID
# -------------------------------------------------------------------
train_df = pd.read_csv(ROOT / "data/processed/clinvar_train_main.csv", low_memory=False)
val_df   = pd.read_csv(ROOT / "data/processed/clinvar_val_main.csv", low_memory=False)
test_df  = pd.read_csv(ROOT / "data/processed/clinvar_test_main.csv", low_memory=False)

merge_cols = ["VariationID", "am_pathogenicity", "am_class", "protein_variant", "uniprot_id", "transcript_id"]
available_merge_cols = [c for c in merge_cols if c in clinvar_am.columns]
am_small = clinvar_am[available_merge_cols].drop_duplicates(subset=["VariationID"]).copy()

train_df = train_df.merge(am_small, on="VariationID", how="left")
val_df   = val_df.merge(am_small, on="VariationID", how="left")
test_df  = test_df.merge(am_small, on="VariationID", how="left")

# numeric conversion
for df in [train_df, val_df, test_df]:
    df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")

# missing indicators BEFORE fill
for df in [train_df, val_df, test_df]:
    df["am_missing"] = df["am_pathogenicity"].isna().astype(float)

# train-only median fill
train_am_fill = train_df["am_pathogenicity"].median()
for df in [train_df, val_df, test_df]:
    df["am_pathogenicity"] = df["am_pathogenicity"].fillna(train_am_fill)

print("\n[2] AlphaMissense merge summary")
print("Train missing before fill:", int(train_df["am_missing"].sum()))
print("Val missing before fill  :", int(val_df["am_missing"].sum()))
print("Test missing before fill :", int(test_df["am_missing"].sum()))
print("Train fill value (median):", float(train_am_fill))

# -------------------------------------------------------------------
# 3. Build feature sets
# -------------------------------------------------------------------
X_train_gcn = gcn_embeddings[train_idx]
X_val_gcn   = gcn_embeddings[val_idx]
X_test_gcn  = gcn_embeddings[test_idx]

# AM-only: score + missing indicator
X_train_am = torch.tensor(train_df[["am_pathogenicity", "am_missing"]].values, dtype=torch.float32)
X_val_am   = torch.tensor(val_df[["am_pathogenicity", "am_missing"]].values, dtype=torch.float32)
X_test_am  = torch.tensor(test_df[["am_pathogenicity", "am_missing"]].values, dtype=torch.float32)

# Fusion: GCN + AM score + missing indicator
X_train_fusion = torch.cat([X_train_gcn, X_train_am], dim=1)
X_val_fusion   = torch.cat([X_val_gcn, X_val_am], dim=1)
X_test_fusion  = torch.cat([X_test_gcn, X_test_am], dim=1)

print("\n[3] Feature matrices")
print("GCN only train :", tuple(X_train_gcn.shape))
print("AM only train  :", tuple(X_train_am.shape))
print("Fusion train   :", tuple(X_train_fusion.shape))

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
        if in_dim <= 2:
            hidden1 = 32
            hidden2 = 16
            dropout = 0.2

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
        "probs": all_probs,
    }

def run_one(name, X_train, X_val, X_test):
    print("\n" + "=" * 60)
    print(f"[Training {name}]")
    print("=" * 60)

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

    model = VariantMLP(in_dim=X_train.shape[1]).to(DEVICE)

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
            f"{name} | Epoch {epoch:02d} | "
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
                print(f"Early stopping for {name} at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    subdir = OUT_DIR / name
    subdir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), subdir / "best.pt")
    np.save(subdir / "val_probs.npy", val_metrics["probs"])
    np.save(subdir / "test_probs.npy", test_metrics["probs"])

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": float(val_metrics["auc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_auprc": float(test_metrics["auprc"]),
    }

    with open(subdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{name}] Final")
    print(f"Val  AUC   : {val_metrics['auc']:.4f}")
    print(f"Val  AUPRC : {val_metrics['auprc']:.4f}")
    print(f"Test AUC   : {test_metrics['auc']:.4f}")
    print(f"Test AUPRC : {test_metrics['auprc']:.4f}")

    return metrics

# -------------------------------------------------------------------
# 5. Run
# -------------------------------------------------------------------
am_metrics = run_one("am_only", X_train_am, X_val_am, X_test_am)
gcn_metrics = run_one("gcn_only", X_train_gcn, X_val_gcn, X_test_gcn)
fusion_metrics = run_one("gcn_plus_am", X_train_fusion, X_val_fusion, X_test_fusion)

summary = {
    "am_only": am_metrics,
    "gcn_only": gcn_metrics,
    "gcn_plus_am": fusion_metrics,
    "delta_auc_fusion_vs_gcn": fusion_metrics["test_auc"] - gcn_metrics["test_auc"],
    "delta_auprc_fusion_vs_gcn": fusion_metrics["test_auprc"] - gcn_metrics["test_auprc"],
    "delta_auc_fusion_vs_am": fusion_metrics["test_auc"] - am_metrics["test_auc"],
    "delta_auprc_fusion_vs_am": fusion_metrics["test_auprc"] - am_metrics["test_auprc"],
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("AlphaMissense fusion summary")
print("=" * 80)
print("AM only test AUC        :", f"{am_metrics['test_auc']:.4f}")
print("GCN only test AUC       :", f"{gcn_metrics['test_auc']:.4f}")
print("GCN+AM test AUC         :", f"{fusion_metrics['test_auc']:.4f}")
print("ΔAUC (fusion vs GCN)    :", f"{summary['delta_auc_fusion_vs_gcn']:.4f}")
print("ΔAUC (fusion vs AM)     :", f"{summary['delta_auc_fusion_vs_am']:.4f}")

print("AM only test AUPRC      :", f"{am_metrics['test_auprc']:.4f}")
print("GCN only test AUPRC     :", f"{gcn_metrics['test_auprc']:.4f}")
print("GCN+AM test AUPRC       :", f"{fusion_metrics['test_auprc']:.4f}")
print("ΔAUPRC (fusion vs GCN)  :", f"{summary['delta_auprc_fusion_vs_gcn']:.4f}")
print("ΔAUPRC (fusion vs AM)   :", f"{summary['delta_auprc_fusion_vs_am']:.4f}")

print("\n✅ Saved summary:", OUT_DIR / "summary.json")
print("=" * 80)
print("Done.")
print("=" * 80)
