import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
CADD_PATH = ROOT / "data/processed/clinvar_with_cadd.csv"
AM_PATH = ROOT / "data/processed/clinvar_with_alphamissense.csv"

ORIGINAL_SPLITS = {
    "train": ROOT / "data/processed/clinvar_train_main.csv",
    "val": ROOT / "data/processed/clinvar_val_main.csv",
    "test": ROOT / "data/processed/clinvar_test_main.csv",
}

HOMOLOGY_SPLITS = {
    "train": ROOT / "data/processed/homology_aware_split/clinvar_train_homology.csv",
    "val": ROOT / "data/processed/homology_aware_split/clinvar_val_homology.csv",
    "test": ROOT / "data/processed/homology_aware_split/clinvar_test_homology.csv",
}

OUT_DIR = ROOT / "data/processed/cadd_am_raw_esm_baseline_exp46"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 43, 44]


class VariantDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class VariantMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        if in_dim <= 3:
            hidden1, hidden2, dropout = 32, 16, 0.2
        else:
            hidden1, hidden2, dropout = 512, 128, 0.3

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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
            probs_all.append(probs)
            labels_all.append(yb.numpy())

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    return {
        "auc": float(roc_auc_score(labels_all, probs_all)),
        "auprc": float(average_precision_score(labels_all, probs_all)),
        "probs": probs_all,
    }


def train_one(split_name, model_name, features, labels, seed):
    set_seed(seed)

    x_train, x_val, x_test = features
    y_train, y_val, y_test = labels

    train_loader = torch.utils.data.DataLoader(
        VariantDataset(x_train, y_train), batch_size=1024, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        VariantDataset(x_val, y_val), batch_size=1024, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        VariantDataset(x_test, y_test), batch_size=1024, shuffle=False, num_workers=0
    )

    model = VariantMLP(x_train.shape[1]).to(DEVICE)
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    patience = 5
    patience_counter = 0

    for epoch in range(1, 51):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_seen += len(yb)

        val_metrics = evaluate(model, val_loader)
        print(
            f"{split_name} | {model_name} | seed {seed} | "
            f"epoch {epoch:02d} | train_loss={running_loss / n_seen:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | val_auprc={val_metrics['auprc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    run_dir = OUT_DIR / split_name / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "val_probs.npy", val_metrics["probs"])
    np.save(run_dir / "test_probs.npy", test_metrics["probs"])

    metrics = {
        "split": split_name,
        "model": model_name,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_auc": float(val_metrics["auc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_auprc": float(test_metrics["auprc"]),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"FINAL | {split_name} | {model_name} | seed {seed} | "
        f"test_auc={metrics['test_auc']:.4f} | test_auprc={metrics['test_auprc']:.4f}"
    )
    return metrics


def load_external_tables():
    cadd = pd.read_csv(CADD_PATH, low_memory=False)
    am = pd.read_csv(AM_PATH, low_memory=False)

    cadd_cols = [c for c in ["VariationID", "CADD_PHRED", "CADD_RawScore"] if c in cadd.columns]
    am_cols = [
        c
        for c in ["VariationID", "am_pathogenicity", "am_class", "protein_variant", "uniprot_id", "transcript_id"]
        if c in am.columns
    ]
    return (
        cadd[cadd_cols].drop_duplicates("VariationID").copy(),
        am[am_cols].drop_duplicates("VariationID").copy(),
    )


def prepare_split(split_name, split_paths, payload, cadd_small, am_small):
    esm = payload["x"].float()
    protein_to_idx = payload.get("protein_to_idx", payload.get("protein_id_to_node_idx"))
    if protein_to_idx is None:
        raise KeyError(f"No protein mapping found in {MAIN_INPUT_PATH}")
    protein_to_idx = {str(k): int(v) for k, v in protein_to_idx.items()}

    dfs = {}
    for part, path in split_paths.items():
        df = pd.read_csv(path, low_memory=False)
        df = df.merge(cadd_small, on="VariationID", how="left")
        df = df.merge(am_small, on="VariationID", how="left")
        df["CADD_PHRED"] = pd.to_numeric(df["CADD_PHRED"], errors="coerce")
        df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")
        df["am_missing"] = df["am_pathogenicity"].isna().astype(float)

        df["STRING_protein_id"] = df["STRING_protein_id"].astype(str)
        df["node_idx"] = df["STRING_protein_id"].map(protein_to_idx)
        missing_nodes = int(df["node_idx"].isna().sum())
        if missing_nodes:
            print(f"WARNING: {split_name}/{part} drops {missing_nodes} rows without node mapping")
            df = df.dropna(subset=["node_idx"]).copy()
        df["node_idx"] = df["node_idx"].astype(int)
        dfs[part] = df

    train_cadd_fill = dfs["train"]["CADD_PHRED"].median()
    train_am_fill = dfs["train"]["am_pathogenicity"].median()
    for df in dfs.values():
        df["CADD_PHRED"] = df["CADD_PHRED"].fillna(train_cadd_fill)
        df["am_pathogenicity"] = df["am_pathogenicity"].fillna(train_am_fill)

    labels = tuple(
        torch.tensor(dfs[part]["Label"].astype(float).values, dtype=torch.float32)
        for part in ["train", "val", "test"]
    )

    idx = {
        part: torch.tensor(dfs[part]["node_idx"].values, dtype=torch.long)
        for part in ["train", "val", "test"]
    }
    x_esm = tuple(esm[idx[part]] for part in ["train", "val", "test"])
    x_cadd_am = tuple(
        torch.tensor(dfs[part][["CADD_PHRED", "am_pathogenicity", "am_missing"]].values, dtype=torch.float32)
        for part in ["train", "val", "test"]
    )
    x_cadd_am_esm = tuple(
        torch.cat([x_cadd_am[i], x_esm[i]], dim=1)
        for i in range(3)
    )

    audit = {
        "split": split_name,
        "n_train": int(len(dfs["train"])),
        "n_val": int(len(dfs["val"])),
        "n_test": int(len(dfs["test"])),
        "train_positive_rate": float(dfs["train"]["Label"].mean()),
        "val_positive_rate": float(dfs["val"]["Label"].mean()),
        "test_positive_rate": float(dfs["test"]["Label"].mean()),
        "train_cadd_fill": float(train_cadd_fill),
        "train_am_fill": float(train_am_fill),
        "train_am_missing": int(dfs["train"]["am_missing"].sum()),
        "val_am_missing": int(dfs["val"]["am_missing"].sum()),
        "test_am_missing": int(dfs["test"]["am_missing"].sum()),
        "cadd_am_dim": int(x_cadd_am[0].shape[1]),
        "cadd_am_raw_esm_dim": int(x_cadd_am_esm[0].shape[1]),
    }
    return labels, {"cadd_plus_am": x_cadd_am, "cadd_am_plus_raw_esm": x_cadd_am_esm}, audit


def summarize(results):
    df = pd.DataFrame(results)
    summary_rows = []
    for (split, model), sub in df.groupby(["split", "model"]):
        summary_rows.append(
            {
                "split": split,
                "model": model,
                "n_repeats": int(len(sub)),
                "test_auc_mean": float(sub["test_auc"].mean()),
                "test_auc_sd": float(sub["test_auc"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "test_auprc_mean": float(sub["test_auprc"].mean()),
                "test_auprc_sd": float(sub["test_auprc"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "val_auc_mean": float(sub["val_auc"].mean()),
                "val_auprc_mean": float(sub["val_auprc"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["split", "model"])
    deltas = []
    for split in sorted(df["split"].unique()):
        s = summary[summary["split"] == split].set_index("model")
        if {"cadd_plus_am", "cadd_am_plus_raw_esm"}.issubset(set(s.index)):
            deltas.append(
                {
                    "split": split,
                    "comparison": "cadd_am_plus_raw_esm_vs_cadd_plus_am",
                    "delta_test_auc_mean": float(
                        s.loc["cadd_am_plus_raw_esm", "test_auc_mean"]
                        - s.loc["cadd_plus_am", "test_auc_mean"]
                    ),
                    "delta_test_auprc_mean": float(
                        s.loc["cadd_am_plus_raw_esm", "test_auprc_mean"]
                        - s.loc["cadd_plus_am", "test_auprc_mean"]
                    ),
                }
            )

    return df, summary, pd.DataFrame(deltas)


def main():
    print("=" * 80)
    print("Experiment 46: CADD+AlphaMissense+raw ESM baseline")
    print("=" * 80)
    print("Device:", DEVICE)
    print("Output:", OUT_DIR)

    payload = torch.load(MAIN_INPUT_PATH, weights_only=False)
    cadd_small, am_small = load_external_tables()

    all_results = []
    audits = []
    for split_name, split_paths in {
        "original": ORIGINAL_SPLITS,
        "homology_aware": HOMOLOGY_SPLITS,
    }.items():
        labels, feature_sets, audit = prepare_split(split_name, split_paths, payload, cadd_small, am_small)
        audits.append(audit)
        print("\nSplit audit:", json.dumps(audit, indent=2))

        for model_name, features in feature_sets.items():
            for seed in SEEDS:
                all_results.append(train_one(split_name, model_name, features, labels, seed))

    per_repeat, summary, deltas = summarize(all_results)
    per_repeat.to_csv(OUT_DIR / "per_repeat_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    deltas.to_csv(OUT_DIR / "delta_summary.csv", index=False)

    out_json = {
        "experiment": "46_cadd_am_plus_raw_esm_baseline",
        "purpose": "Separate the value of adding raw ESM protein representation from the value of PPI propagation in CADD+AlphaMissense fusion.",
        "seeds": SEEDS,
        "audits": audits,
        "summary": summary.to_dict(orient="records"),
        "deltas": deltas.to_dict(orient="records"),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print("\n" + "=" * 80)
    print("Summary metrics")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("\nDelta summary")
    print(deltas.to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
