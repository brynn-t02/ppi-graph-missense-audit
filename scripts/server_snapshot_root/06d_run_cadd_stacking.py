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
CADD_MERGED_PATH = ROOT / "data/processed/clinvar_with_cadd.csv"

OUT_DIR = ROOT / "data/processed/cadd_stacking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("🧬 Step 06d: Run CADD baseline and GCN+CADD stacking")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("CADD table :", CADD_MERGED_PATH)
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

clinvar_cadd = pd.read_csv(CADD_MERGED_PATH, low_memory=False)

print("\n[1] Loaded inputs")
print("GCN embeddings:", tuple(gcn_embeddings.shape))
print("ClinVar+CADD rows:", len(clinvar_cadd))

# -------------------------------------------------------------------
# 2. Reconstruct split row ordering from original split tables
# -------------------------------------------------------------------
# We need exact variant-level alignment for CADD values.
train_df = pd.read_csv(ROOT / "data/processed/clinvar_train_main.csv", low_memory=False)
val_df   = pd.read_csv(ROOT / "data/processed/clinvar_val_main.csv", low_memory=False)
test_df  = pd.read_csv(ROOT / "data/processed/clinvar_test_main.csv", low_memory=False)

key_cols = ["VariationID", "STRING_protein_id", "Label"]

for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    for col in key_cols:
        if col not in df.columns:
            raise ValueError(f"{df_name} split missing required column: {col}")

# Merge CADD scores into each split table
merge_cols = [
    "VariationID",
    "Chromosome",
    "Start",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "CADD_PHRED",
    "CADD_RawScore",
]

available_merge_cols = [c for c in merge_cols if c in clinvar_cadd.columns]
cadd_small = clinvar_cadd[available_merge_cols].drop_duplicates(subset=["VariationID"]).copy()

train_df = train_df.merge(cadd_small, on="VariationID", how="left")
val_df   = val_df.merge(cadd_small, on="VariationID", how="left")
test_df  = test_df.merge(cadd_small, on="VariationID", how="left")

# fill any rare missing values conservatively
for df in [train_df, val_df, test_df]:
    df["CADD_PHRED"] = pd.to_numeric(df["CADD_PHRED"], errors="coerce")
    df["CADD_RawScore"] = pd.to_numeric(df["CADD_RawScore"], errors="coerce")

train_cadd_fill = train_df["CADD_PHRED"].median()
train_raw_fill = train_df["CADD_RawScore"].median()

for df in [train_df, val_df, test_df]:
    df["CADD_PHRED"] = df["CADD_PHRED"].fillna(train_cadd_fill)
    df["CADD_RawScore"] = df["CADD_RawScore"].fillna(train_raw_fill)

print("\n[2] CADD merge summary")
print("Train missing CADD after fill:", int(train_df["CADD_PHRED"].isna().sum()))
print("Val missing CADD after fill  :", int(val_df["CADD_PHRED"].isna().sum()))
print("Test missing CADD after fill :", int(test_df["CADD_PHRED"].isna().sum()))

# -------------------------------------------------------------------
# 3. Build three feature sets
# -------------------------------------------------------------------
X_train_gcn = gcn_embeddings[train_idx]
X_val_gcn   = gcn_embeddings[val_idx]
X_test_gcn  = gcn_embeddings[test_idx]

X_train_cadd = torch.tensor(train_df["CADD_PHRED"].values, dtype=torch.float32).unsqueeze(1)
X_val_cadd   = torch.tensor(val_df["CADD_PHRED"].values, dtype=torch.float32).unsqueeze(1)
X_test_cadd  = torch.tensor(test_df["CADD_PHRED"].values, dtype=torch.float32).unsqueeze(1)

X_train_stack = torch.cat([X_train_gcn, X_train_cadd], dim=1)
X_val_stack   = torch.cat([X_val_gcn, X_val_cadd], dim=1)
X_test_stack  = torch.cat([X_test_gcn, X_test_cadd], dim=1)

print("\n[3] Feature matrices")
print("GCN only train :", tuple(X_train_gcn.shape))
print("CADD only train:", tuple(X_train_cadd.shape))
print("Stack train    :", tuple(X_train_stack.shape))

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
        if in_dim == 1:
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
# 5. Run comparisons
# -------------------------------------------------------------------
cadd_metrics = run_one("cadd_only", X_train_cadd, X_val_cadd, X_test_cadd)
gcn_metrics = run_one("gcn_only", X_train_gcn, X_val_gcn, X_test_gcn)
stack_metrics = run_one("gcn_plus_cadd", X_train_stack, X_val_stack, X_test_stack)

summary = {
    "cadd_only": cadd_metrics,
    "gcn_only": gcn_metrics,
    "gcn_plus_cadd": stack_metrics,
    "delta_auc_stack_vs_gcn": stack_metrics["test_auc"] - gcn_metrics["test_auc"],
    "delta_auprc_stack_vs_gcn": stack_metrics["test_auprc"] - gcn_metrics["test_auprc"],
    "delta_auc_stack_vs_cadd": stack_metrics["test_auc"] - cadd_metrics["test_auc"],
    "delta_auprc_stack_vs_cadd": stack_metrics["test_auprc"] - cadd_metrics["test_auprc"],
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("CADD stacking summary")
print("=" * 80)
print("CADD only test AUC      :", f"{cadd_metrics['test_auc']:.4f}")
print("GCN only test AUC       :", f"{gcn_metrics['test_auc']:.4f}")
print("GCN+CADD test AUC       :", f"{stack_metrics['test_auc']:.4f}")
print("ΔAUC (stack vs GCN)     :", f"{summary['delta_auc_stack_vs_gcn']:.4f}")
print("ΔAUC (stack vs CADD)    :", f"{summary['delta_auc_stack_vs_cadd']:.4f}")

print("CADD only test AUPRC    :", f"{cadd_metrics['test_auprc']:.4f}")
print("GCN only test AUPRC     :", f"{gcn_metrics['test_auprc']:.4f}")
print("GCN+CADD test AUPRC     :", f"{stack_metrics['test_auprc']:.4f}")
print("ΔAUPRC (stack vs GCN)   :", f"{summary['delta_auprc_stack_vs_gcn']:.4f}")
print("ΔAUPRC (stack vs CADD)  :", f"{summary['delta_auprc_stack_vs_cadd']:.4f}")

print("\n✅ Saved summary:", OUT_DIR / "summary.json")
print("=" * 80)
print("Done.")
print("=" * 80)
