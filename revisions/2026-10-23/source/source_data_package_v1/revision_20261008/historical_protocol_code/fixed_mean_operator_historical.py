import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root")

IN_PATH = ROOT / "data/processed/model_inputs_main.pt"
OUT_DIR = ROOT / "data/processed/graphsage_embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🧠 Step 03d: Encode GraphSAGE-style graph-enhanced protein embeddings")
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
# 2. Build row-normalized mean aggregator adjacency
# -------------------------------------------------------------------
print("\n[2] Building mean-aggregation sparse adjacency...")

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
deg_inv = deg.pow(-1.0)
deg_inv[torch.isinf(deg_inv)] = 0.0

row = A.indices()[0]
mean_values = deg_inv[row] * A.values()

A_mean = torch.sparse_coo_tensor(
    A.indices(),
    mean_values,
    size=A.shape
).coalesce().to(DEVICE)

print("Mean aggregator adjacency built.")
print("Non-zero entries:", A_mean._nnz())

# -------------------------------------------------------------------
# 3. GraphSAGE-style propagation
# -------------------------------------------------------------------
print("\n[3] Running GraphSAGE-style mean aggregation...")

x = x.to(DEVICE)

with torch.no_grad():
    # first neighbor mean aggregation
    h1 = torch.sparse.mm(A_mean, x)

    # second hop
    h2 = torch.sparse.mm(A_mean, h1)

    # residual-style fusion
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
        "method": "graphsage_style_mean_propagation",
        "num_nodes": num_nodes,
        "feat_dim": feat_dim,
    },
    OUT_DIR / "graphsage_node_embeddings.pt"
)

np.save(OUT_DIR / "graphsage_node_embeddings.npy", h_final.numpy())

meta = {
    "method": "graphsage_style_mean_propagation",
    "num_nodes": int(num_nodes),
    "feat_dim": int(feat_dim),
    "fusion": "mean(x, A_mean x, A_mean^2 x)",
}

with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n[4] Saved outputs")
print("✅", OUT_DIR / "graphsage_node_embeddings.pt")
print("✅", OUT_DIR / "graphsage_node_embeddings.npy")
print("✅", OUT_DIR / "metadata.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
