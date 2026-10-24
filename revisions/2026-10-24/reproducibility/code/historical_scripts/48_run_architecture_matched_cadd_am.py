#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 48:
Architecture-matched CADD+AlphaMissense baseline.

Purpose:
Test whether the CADD+AlphaMissense-to-fusion gain in Fig. 2 can be
explained by the larger downstream classifier used by representation/fusion
models. This reruns CADD+AlphaMissense score-only inputs with the same
512/128 hidden-layer MLP and dropout 0.3 used by representation/fusion models.

REAL DATA POLICY:
- Uses the same frozen ClinVar/STRING splits and matched CADD/AlphaMissense
  features as Exp46.
- Changes only downstream MLP capacity.
- No fake labels and no simulated predictions.

Expected server location:
  /root/48_run_architecture_matched_cadd_am.py

Requires:
  /root/46_run_cadd_am_plus_raw_esm_baseline.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE_SCRIPT = Path("/root/46_run_cadd_am_plus_raw_esm_baseline.py")
OUT_DIR = Path("/root/data/processed/cadd_am_architecture_matched_exp48")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_exp46_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing {BASE_SCRIPT}. Copy server_scripts/46_run_cadd_am_plus_raw_esm_baseline.py "
            "to /root first, then rerun this script."
        )
    spec = importlib.util.spec_from_file_location("exp46", BASE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ArchitectureMatchedMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def summarize(results):
    df = pd.DataFrame(results)
    rows = []
    for (split, model), sub in df.groupby(["split", "model"]):
        rows.append(
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
    return df, pd.DataFrame(rows).sort_values(["split", "model"])


def main():
    exp46 = load_exp46_module()
    exp46.OUT_DIR = OUT_DIR
    exp46.OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp46.VariantMLP = ArchitectureMatchedMLP

    print("=" * 80)
    print("Experiment 48: architecture-matched CADD+AlphaMissense baseline")
    print("=" * 80)
    print("Device:", exp46.DEVICE)
    print("Output:", OUT_DIR)

    payload = torch.load(exp46.MAIN_INPUT_PATH, weights_only=False)
    cadd_small, am_small = exp46.load_external_tables()

    all_results = []
    audits = []
    for split_name, split_paths in {
        "original": exp46.ORIGINAL_SPLITS,
        "homology_aware": exp46.HOMOLOGY_SPLITS,
    }.items():
        labels, feature_sets, audit = exp46.prepare_split(
            split_name, split_paths, payload, cadd_small, am_small
        )
        audits.append(audit)
        print("\nSplit audit:", json.dumps(audit, indent=2))

        features = feature_sets["cadd_plus_am"]
        for seed in exp46.SEEDS:
            all_results.append(
                exp46.train_one(
                    split_name,
                    "cadd_plus_am_archmatched_512_128",
                    features,
                    labels,
                    seed,
                )
            )

    per_repeat, summary = summarize(all_results)
    per_repeat.to_csv(OUT_DIR / "per_repeat_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    out_json = {
        "experiment": "48_architecture_matched_cadd_am",
        "purpose": "Rerun CADD+AlphaMissense score-only baseline with the same 512/128 dropout-0.3 MLP used by representation/fusion models.",
        "architecture": {"hidden_layers": [512, 128], "dropout": 0.3},
        "seeds": exp46.SEEDS,
        "audits": audits,
        "summary": summary.to_dict(orient="records"),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print("\n" + "=" * 80)
    print("Summary metrics")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
