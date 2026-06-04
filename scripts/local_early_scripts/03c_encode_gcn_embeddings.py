import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root")

IN_PATH = ROOT / "data/processed/model_inputs_main.pt"
OUT_DIR = ROOT / "data/processed/gcn_embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🧠 Step 03c: Encode GCN-style graph-enhanced protein embeddings")
print("=" * 80)
print("Input :", IN_PATH)
print("Outdir:", OUT_DIR)
print("Device:", DEVICE)

# -------------------------------------------------------------------
# 1. Load prepared package
# -------------------------------------------------------------------
payload = torch.load(IN_PATH, weights_only=False)

x = payload["x"].float()                    # [N_nodes, 1280]
edge_index = payload["edge_index"].long()  # [2, E]
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
# 2. Build normalized sparse adjacency
# -------------------------------------------------------------------
print("\n[2] Building normalized sparse adjacency...")

src = edge_index[0]
dst = edge_index[1]

# self-loop
self_loops = torch.arange(num_nodes, dtype=torch.long)
src_all = torch.cat([src, self_loops], dim=0)
dst_all = torch.cat([dst, self_loops], dim=0)

# symmetrize
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
# 3. Graph propagation encoder
# -------------------------------------------------------------------
print("\n[3] Running GCN-style propagation...")

x = x.to(DEVICE)

with torch.no_grad():
    # 1-hop propagation
    h1 = torch.sparse.mm(A_norm, x)

    # 2-hop propagation
    h2 = torch.sparse.mm(A_norm, h1)

    # simple residual-style fusion
    h_final = (x + h1 + h2) / 3.0

h_final = h_final.cpu()

print("Encoded embedding shape:", tuple(h_final.shape))

# -------------------------------------------------------------------
# 4. Save outputs
# -------------------------------------------------------------------
torch.save(
    {
        "embeddings": h_final,
        "protein_to_idx": protein_to_idx,
        "idx_to_protein": idx_to_protein,
        "method": "gcn_style_propagation",
        "num_nodes": num_nodes,
        "feat_dim": feat_dim,
    },
    OUT_DIR / "gcn_node_embeddings.pt"
)

# 也保存一个 npy，方便后面直接加载
np.save(OUT_DIR / "gcn_node_embeddings.npy", h_final.numpy())

# 保存 metadata
meta = {
    "method": "gcn_style_propagation",
    "num_nodes": int(num_nodes),
    "feat_dim": int(feat_dim),
    "fusion": "mean(x, A_norm x, A_norm^2 x)",
}

with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n[4] Saved outputs")
print("✅", OUT_DIR / "gcn_node_embeddings.pt")
print("✅", OUT_DIR / "gcn_node_embeddings.npy")
print("✅", OUT_DIR / "metadata.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
