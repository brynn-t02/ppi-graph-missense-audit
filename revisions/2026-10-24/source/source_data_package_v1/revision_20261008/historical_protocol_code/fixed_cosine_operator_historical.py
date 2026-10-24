import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/root")

IN_PATH = ROOT / "data/processed/model_inputs_main.pt"
OUT_DIR = ROOT / "data/processed/gat_embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🧠 Step 03e: Encode GAT-style graph-enhanced protein embeddings")
print("=" * 80)
print("Input :", IN_PATH)
print("Outdir:", OUT_DIR)
print("Device:", DEVICE)

# -------------------------------------------------------------------
# 1. Load data
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

x = x.to(DEVICE)
edge_index = edge_index.to(DEVICE)

# -------------------------------------------------------------------
# 2. Build adjacency list
# -------------------------------------------------------------------
print("\n[2] Building adjacency lists...")

src = edge_index[0].cpu().numpy()
dst = edge_index[1].cpu().numpy()

neighbors = [[] for _ in range(num_nodes)]
for s, d in zip(src, dst):
    neighbors[s].append(d)
    neighbors[d].append(s)

for i in range(num_nodes):
    neighbors[i].append(i)
    neighbors[i] = list(set(neighbors[i]))

print("Adjacency lists built.")

# -------------------------------------------------------------------
# 3. Attention-style aggregation
# -------------------------------------------------------------------
print("\n[3] Running GAT-style attention propagation...")

@torch.no_grad()
def attention_aggregate(x_in, neighbors, chunk_size=256):
    """
    For each node i:
      alpha_ij = softmax_j( cosine(x_i, x_j) )
      h_i = sum_j alpha_ij * x_j
    """
    out = torch.zeros_like(x_in)

    for start in range(0, x_in.shape[0], chunk_size):
        end = min(start + chunk_size, x_in.shape[0])

        for i in range(start, end):
            nbr_idx = torch.tensor(neighbors[i], device=x_in.device, dtype=torch.long)
            x_i = x_in[i]                  # [D]
            x_n = x_in[nbr_idx]            # [K, D]

            sims = F.cosine_similarity(x_i.unsqueeze(0), x_n, dim=1)  # [K]
            alpha = torch.softmax(sims, dim=0)

            out[i] = torch.sum(alpha.unsqueeze(1) * x_n, dim=0)

        if start % 2000 == 0:
            print(f"  processed nodes {start} - {end}")

    return out

with torch.no_grad():
    h1 = attention_aggregate(x, neighbors, chunk_size=256)
    h2 = attention_aggregate(h1, neighbors, chunk_size=256)

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
        "method": "gat_style_attention_propagation",
        "num_nodes": num_nodes,
        "feat_dim": feat_dim,
    },
    OUT_DIR / "gat_node_embeddings.pt"
)

np.save(OUT_DIR / "gat_node_embeddings.npy", h_final.numpy())

meta = {
    "method": "gat_style_attention_propagation",
    "num_nodes": int(num_nodes),
    "feat_dim": int(feat_dim),
    "fusion": "mean(x, attn(x), attn(attn(x)))",
    "attention": "cosine_similarity_softmax",
}

with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n[4] Saved outputs")
print("✅", OUT_DIR / "gat_node_embeddings.pt")
print("✅", OUT_DIR / "gat_node_embeddings.npy")
print("✅", OUT_DIR / "metadata.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
