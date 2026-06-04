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

TRAIN_SPLIT_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_SPLIT_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/clinvar_test_main.csv"

OUT_DIR = ROOT / "data/processed/core_repeated_benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 42
N_REPEATS = 3

print("=" * 80)
print("🔁 Step 13: Core repeated benchmarks")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("CADD table :", CADD_PATH)
print("AM table   :", AM_PATH)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)
print("Repeats    :", N_REPEATS)

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

esm_embeddings = payload["x"].float()
gcn_embeddings = gcn_payload["embeddings"].float()

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
# 3. Build feature matrices once
# -------------------------------------------------------------------
X_train_esm = esm_embeddings[train_idx]
X_val_esm   = esm_embeddings[val_idx]
X_test_esm  = esm_embeddings[test_idx]

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

X_train_cadd_am = torch.cat([X_train_cadd, X_train_am], dim=1)
X_val_cadd_am   = torch.cat([X_val_cadd, X_val_am], dim=1)
X_test_cadd_am  = torch.cat([X_test_cadd, X_test_am], dim=1)

X_train_full = torch.cat([X_train_gcn, X_train_cadd, X_train_am], dim=1)
X_val_full   = torch.cat([X_val_gcn, X_val_cadd, X_val_am], dim=1)
X_test_full  = torch.cat([X_test_gcn, X_test_cadd, X_test_am], dim=1)

feature_sets = {
    "esm_raw": (X_train_esm, X_val_esm, X_test_esm),
    "gcn_enhanced": (X_train_gcn, X_val_gcn, X_test_gcn),
    "cadd_only": (X_train_cadd, X_val_cadd, X_test_cadd),
    "gcn_plus_cadd": (X_train_gcn_cadd, X_val_gcn_cadd, X_test_gcn_cadd),
    "am_only": (X_train_am, X_val_am, X_test_am),
    "gcn_plus_am": (X_train_gcn_am, X_val_gcn_am, X_test_gcn_am),
    "cadd_plus_am": (X_train_cadd_am, X_val_cadd_am, X_test_cadd_am),
    "gcn_plus_cadd_plus_am": (X_train_full, X_val_full, X_test_full),
}

print("\n[3] Feature sets")
for name, (xt, xv, xs) in feature_sets.items():
    print(f"{name:24s} train={tuple(xt.shape)} val={tuple(xv.shape)} test={tuple(xs.shape)}")

# -------------------------------------------------------------------
# 4. Dataset / model helpers
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

def run_one_repeat(model_name, X_train, X_val, X_test, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            f"{model_name} | seed={seed} | Epoch {epoch:02d} | "
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
                print(f"Early stopping for {model_name} seed={seed} at epoch {epoch}")
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
all_model_results = {}
summary_rows = []

for model_name, (X_train, X_val, X_test) in feature_sets.items():
    print("\n" + "=" * 80)
    print(f"[Model: {model_name}]")
    print("=" * 80)

    repeat_results = []
    for r in range(N_REPEATS):
        seed = BASE_SEED + r
        result = run_one_repeat(model_name, X_train, X_val, X_test, seed)
        repeat_results.append(result)

    all_model_results[model_name] = repeat_results

    val_auc = [x["val_auc"] for x in repeat_results]
    val_auprc = [x["val_auprc"] for x in repeat_results]
    test_auc = [x["test_auc"] for x in repeat_results]
    test_auprc = [x["test_auprc"] for x in repeat_results]

    summary_rows.append({
        "model": model_name,
        "n_repeats": N_REPEATS,
        "val_auc_mean": float(np.mean(val_auc)),
        "val_auc_std": float(np.std(val_auc)),
        "val_auprc_mean": float(np.mean(val_auprc)),
        "val_auprc_std": float(np.std(val_auprc)),
        "test_auc_mean": float(np.mean(test_auc)),
        "test_auc_std": float(np.std(test_auc)),
        "test_auprc_mean": float(np.mean(test_auprc)),
        "test_auprc_std": float(np.std(test_auprc)),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("test_auc_mean", ascending=False).reset_index(drop=True)

# -------------------------------------------------------------------
# 6. Save
# -------------------------------------------------------------------
with open(OUT_DIR / "per_repeat_results.json", "w", encoding="utf-8") as f:
    json.dump(all_model_results, f, indent=2)

summary_df.to_csv(OUT_DIR / "core_repeated_summary.csv", index=False)

summary_json = {
    "n_repeats": N_REPEATS,
    "summary": summary_rows,
    "best_by_test_auc_mean": summary_df.iloc[0].to_dict(),
    "best_by_test_auprc_mean": summary_df.sort_values("test_auprc_mean", ascending=False).iloc[0].to_dict(),
}

with open(OUT_DIR / "core_repeated_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)

print("\n[4] Saved outputs")
print("✅", OUT_DIR / "per_repeat_results.json")
print("✅", OUT_DIR / "core_repeated_summary.csv")
print("✅", OUT_DIR / "core_repeated_summary.json")

print("\n[5] Summary table")
print(summary_df.to_string(index=False))

print("\n[6] Best repeated models")
best_auc = summary_json["best_by_test_auc_mean"]
best_auprc = summary_json["best_by_test_auprc_mean"]
print("Best by repeated test AUC mean   :", best_auc["model"], f"({best_auc['test_auc_mean']:.4f} ± {best_auc['test_auc_std']:.4f})")
print("Best by repeated test AUPRC mean :", best_auprc["model"], f"({best_auprc['test_auprc_mean']:.4f} ± {best_auprc['test_auprc_std']:.4f})")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
