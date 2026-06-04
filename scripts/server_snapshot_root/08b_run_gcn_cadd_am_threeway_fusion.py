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
CADD_PATH = ROOT / "data/processed/clinvar_with_cadd.csv"
AM_PATH = ROOT / "data/processed/clinvar_with_alphamissense.csv"

OUT_DIR = ROOT / "data/processed/triple_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("🧬 Step 08b: Run GCN + CADD + AlphaMissense three-way fusion")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("CADD table :", CADD_PATH)
print("AM table   :", AM_PATH)
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

cadd_df = pd.read_csv(CADD_PATH, low_memory=False)
am_df   = pd.read_csv(AM_PATH, low_memory=False)

print("\n[1] Loaded inputs")
print("GCN embeddings:", tuple(gcn_embeddings.shape))
print("CADD rows     :", len(cadd_df))
print("AM rows       :", len(am_df))

# -------------------------------------------------------------------
# 2. Load split tables
# -------------------------------------------------------------------
train_df = pd.read_csv(ROOT / "data/processed/clinvar_train_main.csv", low_memory=False)
val_df   = pd.read_csv(ROOT / "data/processed/clinvar_val_main.csv", low_memory=False)
test_df  = pd.read_csv(ROOT / "data/processed/clinvar_test_main.csv", low_memory=False)

# -------------------------------------------------------------------
# 3. Merge CADD + AM into split tables using VariationID
# -------------------------------------------------------------------
cadd_keep = [c for c in ["VariationID", "CADD_PHRED", "CADD_RawScore"] if c in cadd_df.columns]
am_keep   = [c for c in ["VariationID", "am_pathogenicity", "am_class", "protein_variant", "uniprot_id", "transcript_id"] if c in am_df.columns]

cadd_small = cadd_df[cadd_keep].drop_duplicates(subset=["VariationID"]).copy()
am_small   = am_df[am_keep].drop_duplicates(subset=["VariationID"]).copy()

def merge_external(df):
    out = df.merge(cadd_small, on="VariationID", how="left")
    out = out.merge(am_small, on="VariationID", how="left")
    return out

train_df = merge_external(train_df)
val_df   = merge_external(val_df)
test_df  = merge_external(test_df)

# numeric conversion
for df in [train_df, val_df, test_df]:
    df["CADD_PHRED"] = pd.to_numeric(df["CADD_PHRED"], errors="coerce")
    df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")

# missing indicators
for df in [train_df, val_df, test_df]:
    df["am_missing"] = df["am_pathogenicity"].isna().astype(float)

# train-only fill
train_cadd_fill = train_df["CADD_PHRED"].median()
train_am_fill = train_df["am_pathogenicity"].median()

for df in [train_df, val_df, test_df]:
    df["CADD_PHRED"] = df["CADD_PHRED"].fillna(train_cadd_fill)
    df["am_pathogenicity"] = df["am_pathogenicity"].fillna(train_am_fill)

print("\n[2] External score merge summary")
print("Train CADD fill value:", float(train_cadd_fill))
print("Train AM fill value  :", float(train_am_fill))
print("Train AM missing     :", int(train_df['am_missing'].sum()))
print("Val AM missing       :", int(val_df['am_missing'].sum()))
print("Test AM missing      :", int(test_df['am_missing'].sum()))

# -------------------------------------------------------------------
# 4. Build feature sets
# -------------------------------------------------------------------
X_train_gcn = gcn_embeddings[train_idx]
X_val_gcn   = gcn_embeddings[val_idx]
X_test_gcn  = gcn_embeddings[test_idx]

X_train_cadd = torch.tensor(train_df[["CADD_PHRED"]].values, dtype=torch.float32)
X_val_cadd   = torch.tensor(val_df[["CADD_PHRED"]].values, dtype=torch.float32)
X_test_cadd  = torch.tensor(test_df[["CADD_PHRED"]].values, dtype=torch.float32)

X_train_am = torch.tensor(train_df[["am_pathogenicity", "am_missing"]].values, dtype=torch.float32)
X_val_am   = torch.tensor(val_df[["am_pathogenicity", "am_missing"]].values, dtype=torch.float32)
X_test_am  = torch.tensor(test_df[["am_pathogenicity", "am_missing"]].values, dtype=torch.float32)

X_train_cadd_am = torch.cat([X_train_cadd, X_train_am], dim=1)  # 3 dim
X_val_cadd_am   = torch.cat([X_val_cadd, X_val_am], dim=1)
X_test_cadd_am  = torch.cat([X_test_cadd, X_test_am], dim=1)

X_train_full = torch.cat([X_train_gcn, X_train_cadd, X_train_am], dim=1)  # 1283 dim
X_val_full   = torch.cat([X_val_gcn, X_val_cadd, X_val_am], dim=1)
X_test_full  = torch.cat([X_test_gcn, X_test_cadd, X_test_am], dim=1)

print("\n[3] Feature matrices")
print("CADD only       :", tuple(X_train_cadd.shape))
print("AM only         :", tuple(X_train_am.shape))
print("CADD+AM         :", tuple(X_train_cadd_am.shape))
print("GCN+CADD+AM     :", tuple(X_train_full.shape))

# -------------------------------------------------------------------
# 5. Dataset / model
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
        if in_dim <= 3:
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
# 6. Run comparisons
# -------------------------------------------------------------------
cadd_metrics = run_one("cadd_only", X_train_cadd, X_val_cadd, X_test_cadd)
am_metrics = run_one("am_only", X_train_am, X_val_am, X_test_am)
cadd_am_metrics = run_one("cadd_plus_am", X_train_cadd_am, X_val_cadd_am, X_test_cadd_am)
full_metrics = run_one("gcn_plus_cadd_plus_am", X_train_full, X_val_full, X_test_full)

summary = {
    "cadd_only": cadd_metrics,
    "am_only": am_metrics,
    "cadd_plus_am": cadd_am_metrics,
    "gcn_plus_cadd_plus_am": full_metrics,
    "delta_auc_full_vs_cadd_am": full_metrics["test_auc"] - cadd_am_metrics["test_auc"],
    "delta_auprc_full_vs_cadd_am": full_metrics["test_auprc"] - cadd_am_metrics["test_auprc"],
    "delta_auc_full_vs_am": full_metrics["test_auc"] - am_metrics["test_auc"],
    "delta_auprc_full_vs_am": full_metrics["test_auprc"] - am_metrics["test_auprc"],
    "delta_auc_full_vs_cadd": full_metrics["test_auc"] - cadd_metrics["test_auc"],
    "delta_auprc_full_vs_cadd": full_metrics["test_auprc"] - cadd_metrics["test_auprc"],
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("Three-way fusion summary")
print("=" * 80)
print("CADD only test AUC            :", f"{cadd_metrics['test_auc']:.4f}")
print("AM only test AUC              :", f"{am_metrics['test_auc']:.4f}")
print("CADD+AM test AUC              :", f"{cadd_am_metrics['test_auc']:.4f}")
print("GCN+CADD+AM test AUC          :", f"{full_metrics['test_auc']:.4f}")
print("ΔAUC (full vs CADD+AM)        :", f"{summary['delta_auc_full_vs_cadd_am']:.4f}")

print("CADD only test AUPRC          :", f"{cadd_metrics['test_auprc']:.4f}")
print("AM only test AUPRC            :", f"{am_metrics['test_auprc']:.4f}")
print("CADD+AM test AUPRC            :", f"{cadd_am_metrics['test_auprc']:.4f}")
print("GCN+CADD+AM test AUPRC        :", f"{full_metrics['test_auprc']:.4f}")
print("ΔAUPRC (full vs CADD+AM)      :", f"{summary['delta_auprc_full_vs_cadd_am']:.4f}")

print("\n✅ Saved summary:", OUT_DIR / "summary.json")
print("=" * 80)
print("Done.")
print("=" * 80)
