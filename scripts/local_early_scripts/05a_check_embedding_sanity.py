import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/root")

MAIN_INPUT_PATH = ROOT / "data/processed/model_inputs_main.pt"
GCN_EMB_PATH = ROOT / "data/processed/gcn_embeddings/gcn_node_embeddings.pt"
SAGE_EMB_PATH = ROOT / "data/processed/graphsage_embeddings/graphsage_node_embeddings.pt"
GAT_EMB_PATH = ROOT / "data/processed/gat_embeddings/gat_node_embeddings.pt"

OUT_DIR = ROOT / "data/processed/embedding_sanity_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🔎 Step 05a: Embedding sanity check")
print("=" * 80)
print("Main input :", MAIN_INPUT_PATH)
print("GCN emb    :", GCN_EMB_PATH)
print("SAGE emb   :", SAGE_EMB_PATH)
print("GAT emb    :", GAT_EMB_PATH)
print("Outdir     :", OUT_DIR)

# -------------------------------------------------------------------
# 1. Load embeddings
# -------------------------------------------------------------------
main_payload = torch.load(MAIN_INPUT_PATH, weights_only=False)
esm = main_payload["x"].float()

gcn = torch.load(GCN_EMB_PATH, weights_only=False)["embeddings"].float()
sage = torch.load(SAGE_EMB_PATH, weights_only=False)["embeddings"].float()
gat = torch.load(GAT_EMB_PATH, weights_only=False)["embeddings"].float()

embeddings = {
    "esm_raw": esm,
    "gcn_enhanced": gcn,
    "graphsage_enhanced": sage,
    "gat_enhanced": gat,
}

print("\n[1] Loaded embeddings")
for name, emb in embeddings.items():
    print(f"{name:20s}: shape={tuple(emb.shape)}")

# -------------------------------------------------------------------
# 2. Basic stats
# -------------------------------------------------------------------
def tensor_stats(x):
    return {
        "shape": list(x.shape),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "nan_count": int(torch.isnan(x).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item()),
        "zero_fraction": float((x == 0).float().mean().item()),
    }

summary = {"basic_stats": {}}

print("\n[2] Basic statistics")
for name, emb in embeddings.items():
    stats = tensor_stats(emb)
    summary["basic_stats"][name] = stats

    print(f"\n{name}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

# -------------------------------------------------------------------
# 3. Sample-wise pairwise differences
# -------------------------------------------------------------------
print("\n[3] Pairwise comparison on sampled nodes")

num_nodes = esm.shape[0]
sample_n = min(1000, num_nodes)
sample_idx = torch.randperm(num_nodes)[:sample_n]

sample_embs = {name: emb[sample_idx] for name, emb in embeddings.items()}

pairwise = {}

def mean_cosine(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return float((a * b).sum(dim=1).mean().item())

def mean_l2(a, b):
    return float(torch.norm(a - b, dim=1).mean().item())

names = list(sample_embs.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        n1, n2 = names[i], names[j]
        a = sample_embs[n1]
        b = sample_embs[n2]

        comp = {
            "mean_cosine_similarity": mean_cosine(a, b),
            "mean_l2_distance": mean_l2(a, b),
        }
        pairwise[f"{n1}__vs__{n2}"] = comp

        print(f"{n1} vs {n2}")
        print(f"  mean_cosine_similarity: {comp['mean_cosine_similarity']:.6f}")
        print(f"  mean_l2_distance      : {comp['mean_l2_distance']:.6f}")

summary["pairwise_sampled_comparison"] = pairwise

# -------------------------------------------------------------------
# 4. Quick checks / flags
# -------------------------------------------------------------------
print("\n[4] Sanity flags")
flags = {}

for name, emb in embeddings.items():
    stats = summary["basic_stats"][name]

    flags[name] = {
        "shape_ok": (emb.shape[0] == num_nodes and emb.shape[1] == esm.shape[1]),
        "no_nan": (stats["nan_count"] == 0),
        "no_inf": (stats["inf_count"] == 0),
        "not_all_zero": (stats["zero_fraction"] < 0.999999),
    }

    print(name)
    for k, v in flags[name].items():
        print(f"  {k}: {v}")

summary["flags"] = flags

# -------------------------------------------------------------------
# 5. Save
# -------------------------------------------------------------------
with open(OUT_DIR / "embedding_sanity_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[5] Saved summary")
print("✅", OUT_DIR / "embedding_sanity_summary.json")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
