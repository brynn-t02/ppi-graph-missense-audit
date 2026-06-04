import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root")

IN_PATH = ROOT / "data/processed/model_inputs_main.pt"
OUT_DIR = ROOT / "data/processed/gcn_embeddings_feature_shuffled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("🔀 Step 04b: Encode GCN-style embeddings with shuffled node features")
print("=" * 80)
print("Input :", IN_PATH)
print("Outdir:", OUT_DIR)
print("Device:", DEVICE)

# -------------------------------------------------------------------
# 1. Load prepared package
# -------------------------------------------------------------------
payload = torch.load(IN_PATH, weights_only=False)

x = payload["x"].float()
edge_index = payload["edge_index"].long()
protein_to_idx = payload["protein_to_idx"]
idx_to_protein = payload["idx_to_protein"]

num_nodes = x.shape[0]
feat_dim = x.shape[1]

print("\n[1] Loaded payload")
print("x shape    :", tuple(x.shape))
print("edge_index :", tuple(edge_index.shape))
print("num_nodes  :", num_nodes)
print("feat_dim   :", feat_dim)

# -------------------------------------------------------------------
# 2. Shuffle node features across proteins
# -------------------------------------------------------------------
perm = torch.randperm(num_nodes)
x_shuffled = x[perm].clone()

print("\n[2] Shuffled node features")
print("Permutation size:", len(perm))

# -------------------------------------------------------------------
# 3. Build normalized sparse adjacency
# -------------------------------------------------------------------
print("\n[3] Building normalized sparse adjacency...")

src = edge_index[0]
dst = edge_index[1]

self_loops = torch.arange(num_nodes, dtype=torch.long)
src_all = torch.cat([src, self_loops], dim=0)
dst_all = torch.cat([dst, self_loops], dim=0)

src_sym = torch.cat([src_all, dst_all], dim=0)
dst_sym = torch.cat([dst_all, src_all], dim=0)

indices = torch.stack([src_sym, dst_sym], dim=0)
values = torch.ones(indices.shape[1], dtype=torch.float32)

A = torch.sparse_coo_tensor(
    indices,
    values,
    size=(num_nodes, num_nodes)
).coalesce()

deg = torch.sparse.sum(A, dim=1).to_dense()
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

row, col = A.indices()
norm_values = deg_inv_sqrt[row] * A.values() * deg_inv_sqrt[col]

A_norm = torch.sparse_coo_tensor(
    A.indices(),
    norm_values,
    size=A.shape
).coalesce().to(DEVICE)

print("Sparse adjacency built.")
print("Non-zero entries:", A_norm._nnz())

# -------------------------------------------------------------------
# 4. GCN-style propagation on shuffled features
# -------------------------------------------------------------------
print("\n[4] Running GCN-style propagation on shuffled node features...")

x_shuffled = x_shuffled.to(DEVICE)

with torch.no_grad():
    h1 = torch.sparse.mm(A_norm, x_shuffled)
    h2 = torch.sparse.mm(A_norm, h1)
    h_final = (x_shuffled + h1 + h2) / 3.0

h_final = h_final.cpu()

print("Encoded embedding shape:", tuple(h_final.shape))

# -------------------------------------------------------------------
# 5. Save outputs
# -------------------------------------------------------------------
torch.save(
    {
        "embeddings": h_final,
        "protein_to_idx": protein_to_idx,
        "idx_to_protein": idx_to_protein,
        "method": "gcn_style_propagation_feature_shuffled",
        "num_nodes": num_nodes,
        "feat_dim": feat_dim,
        "seed": SEED,
    },
    OUT_DIR / "gcn_shuffled_node_embeddings.pt"
)

np.save(OUT_DIR / "gcn_shuffled_node_embeddings.npy", h_final.numpy())

meta = {
    "method": "gcn_style_propagation_feature_shuffled",
    "num_nodes": int(num_nodes),
    "feat_dim": int(feat_dim),
    "seed": SEED,
    "fusion": "mean(x_shuffled, A_norm x_shuffled, A_norm^2 x_shuffled)",
}

with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n[5] Saved outputs")
print("✅", OUT_DIR / "gcn_shuffled_node_embeddings.pt")
print("✅", OUT_DIR / "gcn_shuffled_node_embeddings.npy")
print("✅", OUT_DIR / "metadata.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
