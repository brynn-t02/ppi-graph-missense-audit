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

TRAIN_SPLIT_PATH = ROOT / "data/processed/homology_aware_split/clinvar_train_homology.csv"
VAL_SPLIT_PATH   = ROOT / "data/processed/homology_aware_split/clinvar_val_homology.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/homology_aware_split/clinvar_test_homology.csv"

OUT_DIR = ROOT / "data/processed/homology_single_tool_fusions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("🧬 Step 19: Single-tool fusions on homology-aware split")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("CADD table :", CADD_PATH)
print("AM table   :", AM_PATH)
print("Train split:", TRAIN_SPLIT_PATH)
print("Val split  :", VAL_SPLIT_PATH)
print("Test split :", TEST_SPLIT_PATH)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)

# -------------------------------------------------------------------
# 1. Load inputs
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)
gcn_payload = torch.load(GCN_EMB_PATH, weights_only=False)

esm_embeddings = payload["x"].float()
gcn_embeddings = gcn_payload["embeddings"].float()

if "protein_to_idx" in payload:
    protein_to_idx = payload["protein_to_idx"]
elif "protein_id_to_node_idx" in payload:
    protein_to_idx = payload["protein_id_to_node_idx"]
else:
    raise KeyError(f"Could not find protein mapping. Available keys: {list(payload.keys())}")

protein_to_idx = {str(k): int(v) for k, v in protein_to_idx.items()}

cadd_df = pd.read_csv(CADD_PATH, low_memory=False)
am_df   = pd.read_csv(AM_PATH, low_memory=False)

train_df = pd.read_csv(TRAIN_SPLIT_PATH, low_memory=False)
val_df   = pd.read_csv(VAL_SPLIT_PATH, low_memory=False)
test_df  = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

print("\n[1] Loaded base inputs")
print("ESM embeddings:", tuple(esm_embeddings.shape))
print("GCN embeddings:", tuple(gcn_embeddings.shape))
print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))
print("Test rows :", len(test_df))
print("Protein mapping entries:", len(protein_to_idx))

# -------------------------------------------------------------------
# 2. Merge external scores
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

for df in [train_df, val_df, test_df]:
    df["CADD_PHRED"] = pd.to_numeric(df["CADD_PHRED"], errors="coerce")
    df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")
    df["am_missing"] = df["am_pathogenicity"].isna().astype(float)

train_cadd_fill = train_df["CADD_PHRED"].median()
train_am_fill   = train_df["am_pathogenicity"].median()

for df in [train_df, val_df, test_df]:
    df["CADD_PHRED"] = df["CADD_PHRED"].fillna(train_cadd_fill)
    df["am_pathogenicity"] = df["am_pathogenicity"].fillna(train_am_fill)

print("\n[2] External score fill summary")
print("Train CADD fill value:", float(train_cadd_fill))
print("Train AM fill value  :", float(train_am_fill))
print("Train AM missing     :", int(train_df["am_missing"].sum()))
print("Val AM missing       :", int(val_df["am_missing"].sum()))
print("Test AM missing      :", int(test_df["am_missing"].sum()))

# -------------------------------------------------------------------
# 3. Map proteins to node indices
# -------------------------------------------------------------------
def attach_node_idx(df, split_name):
    out = df.copy()
    out["STRING_protein_id"] = out["STRING_protein_id"].astype(str)
    out["node_idx"] = out["STRING_protein_id"].map(protein_to_idx)

    missing = out["node_idx"].isna().sum()
    print(f"\n[3] {split_name} protein mapping")
    print(f"Rows                : {len(out)}")
    print(f"Missing node mapping : {int(missing)}")

    out = out.dropna(subset=["node_idx"]).copy()
    out["node_idx"] = out["node_idx"].astype(int)

    print(f"Rows after drop      : {len(out)}")
    return out

train_df = attach_node_idx(train_df, "train")
val_df   = attach_node_idx(val_df, "val")
test_df  = attach_node_idx(test_df, "test")

train_labels = torch.tensor(train_df["Label"].astype(float).values, dtype=torch.float32)
val_labels   = torch.tensor(val_df["Label"].astype(float).values, dtype=torch.float32)
test_labels  = torch.tensor(test_df["Label"].astype(float).values, dtype=torch.float32)

train_idx = torch.tensor(train_df["node_idx"].values, dtype=torch.long)
val_idx   = torch.tensor(val_df["node_idx"].values, dtype=torch.long)
test_idx  = torch.tensor(test_df["node_idx"].values, dtype=torch.long)

# -------------------------------------------------------------------
# 4. Build feature matrices
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

X_train_gcn_cadd = torch.cat([X_train_gcn, X_train_cadd], dim=1)
X_val_gcn_cadd   = torch.cat([X_val_gcn, X_val_cadd], dim=1)
X_test_gcn_cadd  = torch.cat([X_test_gcn, X_test_cadd], dim=1)

X_train_gcn_am = torch.cat([X_train_gcn, X_train_am], dim=1)
X_val_gcn_am   = torch.cat([X_val_gcn, X_val_am], dim=1)
X_test_gcn_am  = torch.cat([X_test_gcn, X_test_am], dim=1)

feature_sets = {
    "cadd_only_homology": (X_train_cadd, X_val_cadd, X_test_cadd),
    "gcn_plus_cadd_homology": (X_train_gcn_cadd, X_val_gcn_cadd, X_test_gcn_cadd),
    "am_only_homology": (X_train_am, X_val_am, X_test_am),
    "gcn_plus_am_homology": (X_train_gcn_am, X_val_gcn_am, X_test_gcn_am),
}

print("\n[4] Feature sets")
for name, (xt, xv, xs) in feature_sets.items():
    print(f"{name:24s} train={tuple(xt.shape)} val={tuple(xv.shape)} test={tuple(xs.shape)}")

# -------------------------------------------------------------------
# 5. Dataset / model helpers
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
# 6. Run
# -------------------------------------------------------------------
results = {}
for model_name, (X_train, X_val, X_test) in feature_sets.items():
    results[model_name] = run_one(model_name, X_train, X_val, X_test)

summary = {
    **results,
    "delta_auc_gcn_cadd_vs_cadd_homology":
        results["gcn_plus_cadd_homology"]["test_auc"] - results["cadd_only_homology"]["test_auc"],
    "delta_auprc_gcn_cadd_vs_cadd_homology":
        results["gcn_plus_cadd_homology"]["test_auprc"] - results["cadd_only_homology"]["test_auprc"],
    "delta_auc_gcn_am_vs_am_homology":
        results["gcn_plus_am_homology"]["test_auc"] - results["am_only_homology"]["test_auc"],
    "delta_auprc_gcn_am_vs_am_homology":
        results["gcn_plus_am_homology"]["test_auprc"] - results["am_only_homology"]["test_auprc"],
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("Homology single-tool fusion summary")
print("=" * 80)
print("CADD-only (homology) test AUC      :", f"{results['cadd_only_homology']['test_auc']:.4f}")
print("GCN+CADD (homology) test AUC       :", f"{results['gcn_plus_cadd_homology']['test_auc']:.4f}")
print("ΔAUC (GCN+CADD vs CADD)            :", f"{summary['delta_auc_gcn_cadd_vs_cadd_homology']:.4f}")

print("AM-only (homology) test AUC        :", f"{results['am_only_homology']['test_auc']:.4f}")
print("GCN+AM (homology) test AUC         :", f"{results['gcn_plus_am_homology']['test_auc']:.4f}")
print("ΔAUC (GCN+AM vs AM)                :", f"{summary['delta_auc_gcn_am_vs_am_homology']:.4f}")

print("\nCADD-only (homology) test AUPRC    :", f"{results['cadd_only_homology']['test_auprc']:.4f}")
print("GCN+CADD (homology) test AUPRC     :", f"{results['gcn_plus_cadd_homology']['test_auprc']:.4f}")
print("ΔAUPRC (GCN+CADD vs CADD)          :", f"{summary['delta_auprc_gcn_cadd_vs_cadd_homology']:.4f}")

print("AM-only (homology) test AUPRC      :", f"{results['am_only_homology']['test_auprc']:.4f}")
print("GCN+AM (homology) test AUPRC       :", f"{results['gcn_plus_am_homology']['test_auprc']:.4f}")
print("ΔAUPRC (GCN+AM vs AM)              :", f"{summary['delta_auprc_gcn_am_vs_am_homology']:.4f}")

print("\n✅ Saved summary:", OUT_DIR / "summary.json")
print("=" * 80)
print("Done.")
print("=" * 80)
