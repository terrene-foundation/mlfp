# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 6: Graph Neural Networks (GCN and GAT)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build GCN and GAT layers as torch.nn.Module subclasses
#   - Implement graph convolution as  H' = sigma( D^{-1/2} A D^{-1/2} H W )
#   - Implement graph attention (GAT) with torch.softmax over neighbours
#   - Train node classifiers on a synthetic stochastic-block-model graph
#   - Visualise learned node embeddings and training curves
#
# PREREQUISITES: M5/ex_4 (attention mechanisms, nn.Module training).
# ESTIMATED TIME: ~60 min
# DATASET: Synthetic 3-community stochastic block model (SBM).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kailash_ml import ModelVisualizer

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# Build a synthetic graph — Stochastic Block Model with 3 communities
# ════════════════════════════════════════════════════════════════════════
# Each node belongs to one of 3 communities. Within a community, edges are
# dense (p_in). Between communities, edges are sparse (p_out). Each node
# has a 16-dim feature vector sampled from a community-specific Gaussian.
def make_sbm_graph(n_per_community: int = 30, p_in: float = 0.4, p_out: float = 0.02, feature_dim: int = 16):
    n = n_per_community * 3
    labels = np.concatenate([np.full(n_per_community, c, dtype=np.int64) for c in range(3)])
    # Symmetric adjacency
    rand = np.random.rand(n, n)
    probs = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    A = (rand < probs).astype(np.float32)
    A = np.triu(A, k=1)
    A = A + A.T
    # Feature vectors: community means shifted apart
    centres = np.random.randn(3, feature_dim).astype(np.float32) * 2
    X = centres[labels] + 0.5 * np.random.randn(n, feature_dim).astype(np.float32)
    return X, A, labels


X_np, A_np, y_np = make_sbm_graph(n_per_community=30)
N = X_np.shape[0]
print(f"Graph: {N} nodes, {int(A_np.sum() // 2)} undirected edges, "
      f"{int((y_np == 0).sum())} / {int((y_np == 1).sum())} / {int((y_np == 2).sum())} per class")

X = torch.from_numpy(X_np).to(device)
A = torch.from_numpy(A_np).to(device)
y = torch.from_numpy(y_np).to(device)

# Add self-loops and build the symmetric Laplacian D^{-1/2} A D^{-1/2}
A_hat = A + torch.eye(N, device=device)
deg = A_hat.sum(dim=1)
d_inv_sqrt = deg.pow(-0.5)
d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
A_norm = d_inv_sqrt.unsqueeze(1) * A_hat * d_inv_sqrt.unsqueeze(0)

# Train/val mask — 30% of each class trains, rest is val
train_mask = torch.zeros(N, dtype=torch.bool, device=device)
rng = np.random.default_rng(0)
for c in range(3):
    idx = np.where(y_np == c)[0]
    chosen = rng.choice(idx, size=int(0.3 * len(idx)), replace=False)
    train_mask[chosen] = True
val_mask = ~train_mask


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Graph Convolutional Layer (Kipf & Welling 2017)
# ════════════════════════════════════════════════════════════════════════
# One layer computes H' = sigma( A_norm @ H @ W ). Notice there are NO
# Python loops over nodes — a single matmul aggregates every neighbourhood
# at once. This is the core insight of GCNs: message passing as a matrix
# multiplication.
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.W(h)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, n_classes)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l1(h, a_norm))
        h = F.dropout(h, p=0.3, training=self.training)
        return self.l2(h, a_norm)

    def embed(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return F.relu(self.l1(h, a_norm))


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Graph Attention Layer (Veličković 2018)
# ════════════════════════════════════════════════════════════════════════
# Instead of symmetric-normalised aggregation, GAT learns attention weights
# alpha_ij between each pair of connected nodes:
#
#   e_ij = LeakyReLU( a^T [W h_i || W h_j] )
#   alpha_ij = softmax_j(e_ij)  over the neighbourhood of i
#   h'_i = sigma( Sum_j alpha_ij * W h_j )
#
# We compute e_ij for ALL node pairs via broadcasting, then mask out
# non-neighbours with -inf so the softmax ignores them. No node-index loop.
class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = self.W(h)                                       # (N, out_dim)
        e_src = self.a_src(Wh)                               # (N, 1)
        e_dst = self.a_dst(Wh)                               # (N, 1)
        # e_ij = e_src_i + e_dst_j broadcast to (N, N)
        scores = F.leaky_relu(e_src + e_dst.T, negative_slope=0.2)
        mask = adj + torch.eye(adj.size(0), device=adj.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alpha = F.softmax(scores, dim=1)                     # softmax over neighbours of each i
        return alpha @ Wh


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.l1 = GATLayer(in_dim, hidden_dim)
        self.l2 = GATLayer(hidden_dim, n_classes)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.l1(h, adj))
        h = F.dropout(h, p=0.3, training=self.training)
        return self.l2(h, adj)

    def embed(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.elu(self.l1(h, adj))


# ════════════════════════════════════════════════════════════════════════
# Training harness
# ════════════════════════════════════════════════════════════════════════
def train_node_classifier(
    model: nn.Module,
    name: str,
    forward_arg: torch.Tensor,
    epochs: int = 60,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
) -> tuple[list[float], list[float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses: list[float] = []
    val_accs: list[float] = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X, forward_arg)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            preds = model(X, forward_arg).argmax(dim=-1)
            acc = (preds[val_mask] == y[val_mask]).float().mean().item()
        val_accs.append(acc)

        if (epoch + 1) % 15 == 0:
            print(f"  [{name}] epoch {epoch+1:3d}  loss={loss.item():.4f}  val_acc={acc:.3f}")
    return train_losses, val_accs


# ── Train GCN ──────────────────────────────────────────────────────────
print("\n── Training GCN ──")
gcn = GCN(in_dim=16, hidden_dim=16, n_classes=3)
gcn_losses, gcn_accs = train_node_classifier(gcn, "GCN", A_norm, epochs=60)

# ── Train GAT ──────────────────────────────────────────────────────────
print("\n── Training GAT ──")
gat = GAT(in_dim=16, hidden_dim=16, n_classes=3)
gat_losses, gat_accs = train_node_classifier(gat, "GAT", A, epochs=60)


# ════════════════════════════════════════════════════════════════════════
# PART 3 — Visualise learned node embeddings
# ════════════════════════════════════════════════════════════════════════
gcn.eval()
with torch.no_grad():
    emb = gcn.embed(X, A_norm).cpu().numpy()

# Simple 2-D projection: PCA via SVD (no sklearn dependency needed)
emb_centered = emb - emb.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
coords = (emb_centered @ Vt.T[:, :2])

print("\nGCN hidden embedding 2-D projection (first 5 nodes per class):")
for c in range(3):
    rows = coords[y_np == c][:5]
    pretty = ", ".join(f"({r[0]:+.2f}, {r[1]:+.2f})" for r in rows)
    print(f"  class {c}: {pretty}")


# ════════════════════════════════════════════════════════════════════════
# PART 4 — Visualise training histories
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "GCN train loss": gcn_losses,
        "GCN val acc": gcn_accs,
        "GAT train loss": gat_losses,
        "GAT val acc": gat_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig.write_html("ex_6_training.html")
print("\nTraining history saved to ex_6_training.html")


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Built a GCN layer as  H' = sigma( D^{-1/2} A D^{-1/2} H W )
  [x] Built a GAT layer with learned attention weights over neighbours
  [x] Trained both as node classifiers on a 3-community SBM graph
  [x] Projected hidden embeddings to 2-D with SVD (no Python loops)
  [x] Compared GCN and GAT training dynamics on the same task
"""
)
