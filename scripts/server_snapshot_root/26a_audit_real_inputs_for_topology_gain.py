#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 26A:
Audit real input files for topology-stratified gain analysis.

Purpose:
- Find real variant/sample tables, labels, split files, prediction arrays, and topology inputs.
- Ensure downstream Experiment 26B uses real aligned data.
- This script does NOT generate fake data, simulated labels, or placeholder predictions.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
ROOTS_TO_SCAN = [
    Path("/root/data/raw"),
    Path("/root/data/processed"),
]

OUT_DIR = Path("/root/data/processed/topology_gain_input_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUT_DIR / "input_audit_report.json"
REPORT_CSV = OUT_DIR / "candidate_table_files.csv"

# =========================
# Known important files
# =========================
KNOWN_FILES = {
    "esm_node_features_full": Path("/root/data/processed/esm_node_features_full.csv"),
    "string_links": Path("/root/data/raw/9606.protein.links.v12.0.txt.gz"),
    "gcn_embeddings": Path("/root/data/processed/gcn_embeddings/gcn_node_embeddings.npy"),
    "gcn_metadata": Path("/root/data/processed/gcn_embeddings/metadata.json"),

    # Main representation comparison probabilities
    "esm_raw_test_probs": Path("/root/data/processed/variant_mlp_comparison/esm_raw/test_probs.npy"),
    "gcn_enhanced_test_probs": Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/test_probs.npy"),
    "esm_raw_val_probs": Path("/root/data/processed/variant_mlp_comparison/esm_raw/val_probs.npy"),
    "gcn_enhanced_val_probs": Path("/root/data/processed/variant_mlp_comparison/gcn_enhanced/val_probs.npy"),

    # Fusion predictions, useful later
    "cadd_only_test_probs": Path("/root/data/processed/triple_fusion/cadd_only/test_probs.npy"),
    "am_only_test_probs": Path("/root/data/processed/triple_fusion/am_only/test_probs.npy"),
    "cadd_am_test_probs": Path("/root/data/processed/triple_fusion/cadd_plus_am/test_probs.npy"),
    "gcn_cadd_am_test_probs": Path("/root/data/processed/triple_fusion/gcn_plus_cadd_plus_am/test_probs.npy"),
}

# =========================
# Keywords for candidate files
# =========================
TABLE_EXTS = {".csv", ".tsv", ".txt"}
NPY_EXTS = {".npy"}

KEYWORDS_FILE = [
    "variant", "clinvar", "split", "test", "train", "val", "label",
    "mapping", "protein", "sample", "dataset", "final", "processed"
]

KEYWORDS_COLUMNS = [
    "protein", "Protein_ID", "string", "ensp", "gene",
    "label", "y", "target", "pathogenic", "benign",
    "split", "set", "train", "test", "val",
    "variant", "chrom", "pos", "ref", "alt",
    "cadd", "alphamissense", "am"
]

def safe_read_table_head(path: Path, nrows=5):
    """
    Read only a few rows from a table to inspect columns.
    This avoids loading huge files fully.
    """
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, nrows=nrows)
        elif path.suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=nrows)
        elif path.suffix == ".txt":
            # Try tab first; if only one column, try whitespace
            df = pd.read_csv(path, sep="\t", nrows=nrows)
            if df.shape[1] == 1:
                df = pd.read_csv(path, sep=r"\s+", nrows=nrows)
        else:
            return None, None
        return df, None
    except Exception as e:
        return None, str(e)

def inspect_npy(path: Path):
    try:
        arr = np.load(path, mmap_mode="r")
        return {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype)
        }, None
    except Exception as e:
        return None, str(e)

def is_candidate_table(path: Path):
    name = path.name.lower()
    return any(k.lower() in name for k in KEYWORDS_FILE)

def column_score(columns):
    score = 0
    matched = []
    lower_cols = [str(c).lower() for c in columns]
    for kw in KEYWORDS_COLUMNS:
        kw_l = kw.lower()
        for c in lower_cols:
            if kw_l in c:
                score += 1
                matched.append(kw)
                break
    return score, sorted(set(matched))

# =========================
# 1. Check known files
# =========================
known_status = {}
for name, path in KNOWN_FILES.items():
    item = {
        "path": str(path),
        "exists": path.exists()
    }
    if path.exists() and path.suffix in NPY_EXTS:
        meta, err = inspect_npy(path)
        item["npy_meta"] = meta
        item["error"] = err
    elif path.exists() and path.suffix in TABLE_EXTS:
        df, err = safe_read_table_head(path)
        item["error"] = err
        if df is not None:
            item["columns"] = list(df.columns)
            item["head_shape"] = list(df.shape)
    elif path.exists() and path.suffix == ".json":
        try:
            with open(path) as f:
                item["json"] = json.load(f)
        except Exception as e:
            item["error"] = str(e)

    known_status[name] = item

# =========================
# 2. Scan table candidates
# =========================
candidate_tables = []

for root in ROOTS_TO_SCAN:
    if not root.exists():
        continue

    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            path = Path(dirpath) / fn

            # Skip Windows metadata files
            if "Zone.Identifier" in str(path):
                continue

            if path.suffix.lower() not in TABLE_EXTS:
                continue

            if not is_candidate_table(path):
                continue

            df, err = safe_read_table_head(path, nrows=5)
            if df is None:
                candidate_tables.append({
                    "path": str(path),
                    "readable": False,
                    "error": err,
                    "n_columns": None,
                    "columns": None,
                    "matched_column_keywords": None,
                    "column_score": 0
                })
                continue

            score, matched = column_score(df.columns)

            candidate_tables.append({
                "path": str(path),
                "readable": True,
                "error": None,
                "n_columns": int(df.shape[1]),
                "columns": "|".join(map(str, df.columns)),
                "matched_column_keywords": "|".join(matched),
                "column_score": int(score)
            })

# Sort likely useful files first
candidate_tables = sorted(candidate_tables, key=lambda x: x["column_score"], reverse=True)

pd.DataFrame(candidate_tables).to_csv(REPORT_CSV, index=False)

# =========================
# 3. Scan prediction arrays
# =========================
npy_files = []

for root in ROOTS_TO_SCAN:
    if not root.exists():
        continue

    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            path = Path(dirpath) / fn

            if "Zone.Identifier" in str(path):
                continue

            if path.suffix.lower() != ".npy":
                continue

            name_l = path.name.lower()
            full_l = str(path).lower()

            if any(k in full_l for k in ["prob", "label", "y_", "test", "val", "pred", "embedding"]):
                meta, err = inspect_npy(path)
                npy_files.append({
                    "path": str(path),
                    "meta": meta,
                    "error": err
                })

# =========================
# 4. Create final report
# =========================
report = {
    "experiment": "26A_real_input_audit_for_topology_gain",
    "purpose": "Audit real files needed for topology-stratified gain analysis. No fake or simulated data are generated.",
    "known_files": known_status,
    "n_candidate_tables": len(candidate_tables),
    "candidate_tables_csv": str(REPORT_CSV),
    "top_candidate_tables": candidate_tables[:30],
    "n_npy_files": len(npy_files),
    "npy_files": npy_files,
    "required_for_exp26B": {
        "must_have": [
            "A test sample table aligned to test_probs.npy order",
            "Each test row must contain true label and mapped Protein_ID",
            "ESM raw test probabilities",
            "GCN enhanced test probabilities",
            "STRING graph or precomputed topology features"
        ],
        "warning": "Experiment 26B must not be run until aligned test labels and Protein_ID order are confirmed."
    },
    "no_fake_data_policy": [
        "This audit script only reads existing files.",
        "It does not create labels, predictions, variant tables, or simulated samples.",
        "If required aligned files are missing, downstream analysis should stop rather than fabricate inputs."
    ]
}

with open(REPORT_JSON, "w") as f:
    json.dump(report, f, indent=2)

# =========================
# 5. Print summary
# =========================
print("=== Experiment 26A: Real input audit completed ===")
print(f"Report JSON: {REPORT_JSON}")
print(f"Candidate table CSV: {REPORT_CSV}")
print("")
print("Known file status:")
for k, v in known_status.items():
    print(f"- {k}: exists={v['exists']} path={v['path']}")
    if "npy_meta" in v and v["npy_meta"] is not None:
        print(f"  shape={v['npy_meta']['shape']}, dtype={v['npy_meta']['dtype']}")

print("")
print("Top candidate table files:")
for item in candidate_tables[:15]:
    print(f"- score={item['column_score']} readable={item['readable']} path={item['path']}")
    print(f"  matched={item['matched_column_keywords']}")
    print(f"  columns={item['columns']}")

print("")
print("Important note:")
print("Experiment 26B requires an aligned test sample table with label + Protein_ID in the same order as test_probs.npy.")
print("If such a file is not found, we must not fabricate one.")
