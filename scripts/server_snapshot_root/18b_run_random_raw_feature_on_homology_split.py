import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"

TRAIN_SPLIT_PATH = ROOT / "data/processed/homology_aware_split/clinvar_train_homology.csv"
VAL_SPLIT_PATH   = ROOT / "data/processed/homology_aware_split/clinvar_val_homology.csv"
TEST_SPLIT_PATH  = ROOT / "data/processed/homology_aware_split/clinvar_test_homology.csv"

RANDOM_GCN_HOMOLOGY_SUMMARY = ROOT / "data/processed/random_feature_gcn_homology/summary.json"

OUT_DIR = ROOT / "data/processed/random_raw_feature_homology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 42
N_REPEATS = 3
FEATURE_DIM = 1280

print("=" * 80)
print("🎲 Step 18b: Random raw-feature baseline on homology-aware split")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("Train split:", TRAIN_SPLIT_PATH)
print("Val split  :", VAL_SPLIT_PATH)
print("Test split :", TEST_SPLIT_PATH)
print("Random GCN homology summary:", RANDOM_GCN_HOMOLOGY_SUMMARY)
print("Outdir     :", OUT_DIR)
print("Device     :", DEVICE)
print("Repeats    :", N_REPEATS)

# -------------------------------------------------------------------
# 1. Load payload and mapping
# -------------------------------------------------------------------
payload = torch.load(MAIN_INPUT_PATH, weights_only=False)

num_nodes = int(payload["num_nodes"]) if "num_nodes" in payload else int(payload["x"].shape[0])

if "protein_to_idx" in payload:
    protein_to_idx = payload["protein_to_idx"]
elif "protein_id_to_node_idx" in payload:
    protein_to_idx = payload["protein_id_to_node_idx"]
else:
    raise KeyError(f"Could not find protein mapping. Available keys: {list(payload.keys())}")

protein_to_idx = {str(k): int(v) for k, v in protein_to_idx.items()}

print("\n[1] Loaded main payload")
print("Num nodes      :", num_nodes)
print("Protein mapping:", len(protein_to_idx))

# -------------------------------------------------------------------
# 2. Load homology-aware split tables and map proteins
# -------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_SPLIT_PATH, low_memory=False)
val_df   = pd.read_csv(VAL_SPLIT_PATH, low_memory=False)
test_df  = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

def attach_node_idx(df, split_name):
    out = df.copy()
    out["STRING_protein_id"] = out["STRING_protein_id"].astype(str)
    out["node_idx"] = out["STRING_protein_id"].map(protein_to_idx)

    missing = out["node_idx"].isna().sum()
    print(f"{split_name}: rows={len(out)}, missing_node_idx={int(missing)}")

    out = out.dropna(subset=["node_idx"]).copy()
    out["node_idx"] = out["node_idx"].astype(int)
    return out

train_df = attach_node_idx(train_df, "train")
val_df   = attach_node_idx(val_df, "val")
test_df  = attach_node_idx(test_df, "test")

train_idx = torch.tensor(train_df["node_idx"].values, dtype=torch.long)
val_idx   = torch.tensor(val_df["node_idx"].values, dtype=torch.long)
test_idx  = torch.tensor(test_df["node_idx"].values, dtype=torch.long)

train_labels = torch.tensor(train_df["Label"].astype(float).values, dtype=torch.float32)
val_labels   = torch.tensor(val_df["Label"].astype(float).values, dtype=torch.float32)
test_labels  = torch.tensor(test_df["Label"].astype(float).values, dtype=torch.float32)

print("\n[2] Homology-aware mapped splits")
print("Train rows:", len(train_labels))
print("Val rows  :", len(val_labels))
print("Test rows :", len(test_labels))

# -------------------------------------------------------------------
# 3. Load random-feature GCN homology summary
# -------------------------------------------------------------------
random_gcn_summary = None
if RANDOM_GCN_HOMOLOGY_SUMMARY.exists():
    with open(RANDOM_GCN_HOMOLOGY_SUMMARY, "r", encoding="utf-8") as f:
        random_gcn_summary = json.load(f)

if random_gcn_summary is not None:
    rg = random_gcn_summary["random_feature_gcn_homology"]
    print("\n[3] Loaded random-feature GCN homology summary")
    print("Random-feature GCN homology test AUC mean  :", rg["test_auc_mean"])
    print("Random-feature GCN homology test AUPRC mean:", rg["test_auprc_mean"])

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

def run_one_repeat(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # random raw node features (NO graph propagation)
    rand_x = torch.randn((num_nodes, FEATURE_DIM), dtype=torch.float32)

    X_train = rand_x[train_idx]
    X_val   = rand_x[val_idx]
    X_test  = rand_x[test_idx]

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

    model = VariantMLP(in_dim=FEATURE_DIM).to(DEVICE)

    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    patience = 5
    patience_counter = 0
    max_epochs = 50

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
            f"random_raw_h | seed={seed} | Epoch {epoch:02d} | "
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
                print(f"Early stopping for seed={seed} at epoch {epoch}")
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
print("\n[4] Running repeated random raw-feature baseline on homology split...")
repeat_results = []

for r in range(N_REPEATS):
    seed = BASE_SEED + r
    print("\n" + "-" * 60)
    print(f"[Repeat {r+1}/{N_REPEATS}] seed={seed}")
    print("-" * 60)
    res = run_one_repeat(seed)
    repeat_results.append(res)

# -------------------------------------------------------------------
# 6. Summaries
# -------------------------------------------------------------------
val_auc = [x["val_auc"] for x in repeat_results]
val_auprc = [x["val_auprc"] for x in repeat_results]
test_auc = [x["test_auc"] for x in repeat_results]
test_auprc = [x["test_auprc"] for x in repeat_results]

summary = {
    "n_repeats": N_REPEATS,
    "random_raw_feature_homology": {
        "val_auc_mean": float(np.mean(val_auc)),
        "val_auc_std": float(np.std(val_auc)),
        "val_auprc_mean": float(np.mean(val_auprc)),
        "val_auprc_std": float(np.std(val_auprc)),
        "test_auc_mean": float(np.mean(test_auc)),
        "test_auc_std": float(np.std(test_auc)),
        "test_auprc_mean": float(np.mean(test_auprc)),
        "test_auprc_std": float(np.std(test_auprc)),
    },
    "per_repeat_results": repeat_results,
    "random_feature_gcn_homology": (
        random_gcn_summary["random_feature_gcn_homology"] if random_gcn_summary else None
    )
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(repeat_results).to_csv(OUT_DIR / "per_repeat_results.csv", index=False)

comparison_rows = [
    {
        "model": "random_raw_feature_homology",
        "test_auc_mean": float(np.mean(test_auc)),
        "test_auc_std": float(np.std(test_auc)),
        "test_auprc_mean": float(np.mean(test_auprc)),
        "test_auprc_std": float(np.std(test_auprc)),
    }
]

if random_gcn_summary is not None:
    rg = random_gcn_summary["random_feature_gcn_homology"]
    comparison_rows.append({
        "model": "random_feature_gcn_homology",
        "test_auc_mean": rg["test_auc_mean"],
        "test_auc_std": rg["test_auc_std"],
        "test_auprc_mean": rg["test_auprc_mean"],
        "test_auprc_std": rg["test_auprc_std"],
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUT_DIR / "comparison_summary.csv", index=False)

print("\n[5] Random raw-feature homology summary")
print(f"Val  AUC   : {np.mean(val_auc):.4f} ± {np.std(val_auc):.4f}")
print(f"Val  AUPRC : {np.mean(val_auprc):.4f} ± {np.std(val_auprc):.4f}")
print(f"Test AUC   : {np.mean(test_auc):.4f} ± {np.std(test_auc):.4f}")
print(f"Test AUPRC : {np.mean(test_auprc):.4f} ± {np.std(test_auprc):.4f}")

print("\n[6] Comparison summary")
print(comparison_df.to_string(index=False))

print("\n✅ Saved outputs:")
print("✅", OUT_DIR / "summary.json")
print("✅", OUT_DIR / "per_repeat_results.csv")
print("✅", OUT_DIR / "comparison_summary.csv")
print("=" * 80)
print("Done.")
print("=" * 80)
