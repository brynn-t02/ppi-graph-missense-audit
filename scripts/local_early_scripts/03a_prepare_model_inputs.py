import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/clinvar_train_main.csv"
VAL_PATH   = ROOT / "data/processed/clinvar_val_main.csv"
TEST_PATH  = ROOT / "data/processed/clinvar_test_main.csv"

LINKS_PATH = ROOT / "data/raw/9606.protein.links.v12.0.txt.gz"
ESM_PATH   = ROOT / "data/processed/esm_node_features_full.csv"

OUT_PATH   = ROOT / "data/processed/model_inputs_main.pt"

print("=" * 80)
print("📦 Step 03a: Prepare unified model inputs for rebuilt main experiments")
print("=" * 80)
print("Train :", TRAIN_PATH)
print("Val   :", VAL_PATH)
print("Test  :", TEST_PATH)
print("Links :", LINKS_PATH)
print("ESM   :", ESM_PATH)
print("Out   :", OUT_PATH)

# -------------------------------------------------------------------
# 1. Load split tables
# -------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("\n[1] Loaded split tables")
print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))
print("Test rows :", len(test_df))

required_cols = ["STRING_protein_id", "Label"]
for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{name} missing required column: {col}")

# -------------------------------------------------------------------
# 2. Load full ESM node features
# -------------------------------------------------------------------
esm_df = pd.read_csv(ESM_PATH)

if "Protein_ID" not in esm_df.columns:
    raise ValueError("ESM file missing Protein_ID column.")

feature_cols = [c for c in esm_df.columns if c.startswith("Node_ESM_")]
if len(feature_cols) == 0:
    raise ValueError("No Node_ESM_* columns found in ESM feature file.")

print("\n[2] Loaded ESM node features")
print("Rows:", len(esm_df))
print("Feature dim:", len(feature_cols))

# -------------------------------------------------------------------
# 3. Load STRING links and build graph node universe
# -------------------------------------------------------------------
links = pd.read_csv(LINKS_PATH, sep=r"\s+", compression="gzip")

prot1_col = [c for c in links.columns if "protein1" in c.lower()][0]
prot2_col = [c for c in links.columns if "protein2" in c.lower()][0]

graph_nodes = sorted(set(links[prot1_col].astype(str)) | set(links[prot2_col].astype(str)))
protein_to_idx = {pid: i for i, pid in enumerate(graph_nodes)}
idx_to_protein = {i: pid for pid, i in protein_to_idx.items()}

print("\n[3] Graph node universe")
print("Total graph nodes:", len(graph_nodes))

# -------------------------------------------------------------------
# 4. Align ESM features to graph node order
# -------------------------------------------------------------------
esm_df["Protein_ID"] = esm_df["Protein_ID"].astype(str)
esm_df = esm_df.drop_duplicates(subset=["Protein_ID"]).copy()
esm_df = esm_df.set_index("Protein_ID")

missing_graph_nodes = [pid for pid in graph_nodes if pid not in esm_df.index]
if len(missing_graph_nodes) > 0:
    raise ValueError(f"Missing ESM features for {len(missing_graph_nodes)} graph nodes.")

x = torch.tensor(
    esm_df.loc[graph_nodes, feature_cols].values,
    dtype=torch.float32
)

print("\n[4] Node feature matrix")
print("x shape:", tuple(x.shape))

# -------------------------------------------------------------------
# 5. Build edge_index
# -------------------------------------------------------------------
src = links[prot1_col].astype(str).map(protein_to_idx).values
dst = links[prot2_col].astype(str).map(protein_to_idx).values

edge_index = torch.tensor([src, dst], dtype=torch.long)

print("\n[5] Edge index")
print("edge_index shape:", tuple(edge_index.shape))
print("Number of edges:", edge_index.shape[1])

# -------------------------------------------------------------------
# 6. Map each variant to protein node index
# -------------------------------------------------------------------
def build_variant_tensors(df, split_name):
    pids = df["STRING_protein_id"].astype(str)

    missing = (~pids.isin(protein_to_idx)).sum()
    if missing > 0:
        raise ValueError(f"{split_name}: {missing} variants map to unknown proteins.")

    protein_idx = torch.tensor(
        pids.map(protein_to_idx).values,
        dtype=torch.long
    )
    labels = torch.tensor(df["Label"].astype(int).values, dtype=torch.long)

    return protein_idx, labels

train_protein_idx, train_labels = build_variant_tensors(train_df, "TRAIN")
val_protein_idx, val_labels     = build_variant_tensors(val_df, "VAL")
test_protein_idx, test_labels   = build_variant_tensors(test_df, "TEST")

print("\n[6] Variant-to-node mapping")
print("Train protein_idx shape:", tuple(train_protein_idx.shape))
print("Val protein_idx shape  :", tuple(val_protein_idx.shape))
print("Test protein_idx shape :", tuple(test_protein_idx.shape))

# -------------------------------------------------------------------
# 7. Save unified package
# -------------------------------------------------------------------
payload = {
    "x": x,
    "edge_index": edge_index,
    "protein_to_idx": protein_to_idx,
    "idx_to_protein": idx_to_protein,

    "train_protein_idx": train_protein_idx,
    "val_protein_idx": val_protein_idx,
    "test_protein_idx": test_protein_idx,

    "train_labels": train_labels,
    "val_labels": val_labels,
    "test_labels": test_labels,

    "feature_dim": x.shape[1],
    "num_nodes": x.shape[0],
    "num_edges": edge_index.shape[1],
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, OUT_PATH)

print("\n[7] Saved model input package")
print("✅", OUT_PATH)

# -------------------------------------------------------------------
# 8. Final summary
# -------------------------------------------------------------------
print("\n[8] Final summary")
print("Nodes       :", payload["num_nodes"])
print("Edges       :", payload["num_edges"])
print("Feature dim :", payload["feature_dim"])
print("Train vars  :", len(train_labels))
print("Val vars    :", len(val_labels))
print("Test vars   :", len(test_labels))

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
