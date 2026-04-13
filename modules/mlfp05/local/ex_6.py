# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 6: Graph Neural Networks (GCN, GAT, GraphSAGE)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build GCN, GAT, and GraphSAGE layers as torch.nn.Module subclasses
#   - Implement graph convolution as  H' = sigma( D^{-1/2} A D^{-1/2} H W )
#   - Implement graph attention (GAT) with torch.softmax over neighbours
#   - Implement GraphSAGE neighbourhood sampling and aggregation
#   - Train node classifiers on a REAL citation network (Cora, 2708 papers)
#   - Perform link prediction (predict missing edges) on the same graph
#   - Track every GNN variant with kailash-ml's ExperimentTracker
#   - Register the best model in kailash-ml's ModelRegistry
#   - Visualise training curves and learned node embeddings
#   - Compare GCN, GAT, and GraphSAGE quantitatively on accuracy,
#     convergence speed, and parameter count
#
# PREREQUISITES: M5/ex_4 (attention mechanisms, nn.Module training).
# ESTIMATED TIME: ~120-150 min
# DATASETS:
#   - PRIMARY: Cora citation network via torch_geometric.datasets.Planetoid
#       2708 papers, 1433-dim bag-of-words features, 7 research categories.
#       Edges = paper A cites paper B. Standard GNN benchmark.
#       Cached to data/mlfp05/cora/.
#   - FALLBACK: Zachary's Karate Club via networkx.karate_club_graph()
#       34 members of a real karate club studied by Zachary in 1977.
#       After a dispute the club split into two factions; we predict the
#       faction of each member from the friendship graph.
#
# TASKS:
#   1. Load Cora graph and set up kailash-ml engines
#   2. Build and train a GCN node classifier, log to ExperimentTracker
#   3. Build and train a GAT node classifier, log to ExperimentTracker
#   4. Build and train a GraphSAGE node classifier, log to ExperimentTracker
#   5. Compare all three GNN architectures quantitatively
#   6. Visualise learned node embeddings (2-D PCA projection)
#   7. Link prediction: predict missing edges with dot-product decoder
#   8. Register the best model in the ModelRegistry
#   9. Visualise all training histories
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load a REAL graph dataset and set up kailash-ml engines
# ════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cora"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_cora() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Cora — 2708 papers, 1433 bag-of-words features, 7 classes.

    Returns:
        X_np: node features (N, F)
        A_np: dense adjacency matrix (N, N)
        y_np: node labels (N,)
        edge_index_np: edge list (2, E) for link prediction
        dataset_name: "Cora"
        n_classes: 7
    """
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root=str(DATA_DIR), name="Cora")
    data = dataset[0]
    n = data.num_nodes
    X_np = data.x.numpy().astype(np.float32)
    y_np = data.y.numpy().astype(np.int64)

    # Build a dense adjacency matrix from the edge_index. Cora has ~10k
    # directed edges (5278 undirected) over 2708 nodes; the dense matrix
    # is ~7M entries which fits comfortably in CPU memory.
    A_np = np.zeros((n, n), dtype=np.float32)
    edge_index_np = data.edge_index.numpy()
    src = edge_index_np[0]
    dst = edge_index_np[1]
    A_np[src, dst] = 1.0
    A_np[dst, src] = 1.0  # symmetrise just in case
    n_classes = int(dataset.num_classes)
    return X_np, A_np, y_np, edge_index_np, "Cora", n_classes


def load_karate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Zachary's Karate Club — 34 nodes, 78 edges, 2 factions."""
    import networkx as nx

    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    A_np = nx.to_numpy_array(G, dtype=np.float32)
    labels = np.array(
        [0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)],
        dtype=np.int64,
    )
    # Karate has no node features; use one-hot identity (transductive)
    X_np = np.eye(n, dtype=np.float32)
    # Build edge_index from adjacency
    src, dst = np.where(A_np > 0)
    edge_index_np = np.stack([src, dst]).astype(np.int64)
    return X_np, A_np, labels, edge_index_np, "Karate Club", 2


try:
    X_np, A_np, y_np, edge_index_np, dataset_name, n_classes = load_cora()
except Exception as exc:
    print(
        f"Could not load Cora ({type(exc).__name__}: {exc}); "
        "falling back to Karate Club"
    )
    X_np, A_np, y_np, edge_index_np, dataset_name, n_classes = load_karate()

N = X_np.shape[0]
F_dim = X_np.shape[1]
n_edges_undirected = int(A_np.sum() // 2)
print(
    f"Graph: {dataset_name} — {N} nodes, {n_edges_undirected} undirected edges, "
    f"feature_dim={F_dim}, classes={n_classes}"
)
class_counts = ", ".join(f"c{c}={int((y_np == c).sum())}" for c in range(n_classes))
print(f"  per-class counts: {class_counts}")

X = torch.from_numpy(X_np).to(device)
A = torch.from_numpy(A_np).to(device)
y = torch.from_numpy(y_np).to(device)

# Add self-loops and build the symmetric Laplacian D^{-1/2} A D^{-1/2}
A_hat = A + torch.eye(N, device=device)
deg = A_hat.sum(dim=1)
d_inv_sqrt = deg.pow(-0.5)
d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
A_norm = d_inv_sqrt.unsqueeze(1) * A_hat * d_inv_sqrt.unsqueeze(0)

# Train/val/test split — 20% train, 20% val, 60% test (per class)
train_mask = torch.zeros(N, dtype=torch.bool, device=device)
val_mask = torch.zeros(N, dtype=torch.bool, device=device)
rng = np.random.default_rng(0)
for c in range(n_classes):
    idx = np.where(y_np == c)[0]
    if len(idx) == 0:
        continue
    rng.shuffle(idx)
    n_train = max(1, int(0.2 * len(idx)))
    n_val = max(1, int(0.2 * len(idx)))
    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train : n_train + n_val]] = True
test_mask = ~(train_mask | val_mask)
print(
    f"  train: {int(train_mask.sum().item())}, "
    f"val: {int(val_mask.sum().item())}, "
    f"test: {int(test_mask.sum().item())}"
)


# Set up kailash-ml engines: ExperimentTracker + ModelRegistry
async def setup_engines():
    # TODO: Create ConnectionManager with "sqlite:///mlfp05_gnns.db"
    # Hint: ConnectionManager("sqlite:///mlfp05_gnns.db"), then await conn.initialize()
    conn = ____  # noqa: F821
    await ____  # noqa: F821

    # TODO: Create ExperimentTracker and start an experiment named "m5_gnns"
    # Hint: ExperimentTracker(conn), then await tracker.create_experiment(name=..., description=...)
    tracker = ____  # noqa: F821
    exp_name = await tracker.create_experiment(
        name=____,  # noqa: F821
        description=____,  # noqa: F821
    )

    try:
        # TODO: Create ModelRegistry(conn) and set has_registry = True
        # Hint: registry = ModelRegistry(conn)
        registry = ____  # noqa: F821
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X.shape[0] >= 34, "Should have loaded a graph with nodes"
assert A.shape == (N, N), "Adjacency matrix should be NxN"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: We loaded a real citation graph where nodes are papers
# and edges are citations. Each paper has a 1433-dim bag-of-words feature
# vector and belongs to one of 7 research categories. The ExperimentTracker
# will record every GNN training run for systematic comparison.
print("\n--- Checkpoint 1 passed --- graph loaded and engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Graph Convolutional Network (Kipf & Welling 2017)
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
        # TODO: Return the GCN message pass: a_norm @ self.W(h)
        # Hint: one line — matrix multiply normalised adjacency by transformed features
        return ____  # noqa: F821


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, n_classes)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l1(h, a_norm))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.l2(h, a_norm)

    def embed(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """Return the hidden-layer embedding (before classification head)."""
        return F.relu(self.l1(h, a_norm))


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Graph Attention Network (Velickovic 2018)
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
        Wh = self.W(h)  # (N, out_dim)
        # TODO: Compute attention scores e_src and e_dst, then broadcast to (N, N)
        # Hint: e_src = self.a_src(Wh)  # (N, 1)
        #       e_dst = self.a_dst(Wh)  # (N, 1)
        #       scores = F.leaky_relu(e_src + e_dst.T, negative_slope=0.2)
        e_src = ____  # noqa: F821
        e_dst = ____  # noqa: F821
        scores = ____  # noqa: F821
        # TODO: Mask non-neighbours with -inf and apply softmax, then return alpha @ Wh
        # Hint: mask = adj + torch.eye(adj.size(0), device=adj.device)
        #       scores.masked_fill(mask == 0, float("-inf"))
        #       alpha = F.softmax(scores, dim=1)
        mask = adj + torch.eye(adj.size(0), device=adj.device)
        scores = ____  # noqa: F821
        alpha = ____  # noqa: F821
        return ____  # noqa: F821


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.l1 = GATLayer(in_dim, hidden_dim)
        self.l2 = GATLayer(hidden_dim, n_classes)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.l1(h, adj))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.l2(h, adj)

    def embed(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Return the hidden-layer embedding (before classification head)."""
        return F.elu(self.l1(h, adj))


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — GraphSAGE (Hamilton, Ying & Leskovec 2017)
# ════════════════════════════════════════════════════════════════════════
# GraphSAGE differs from GCN in three key ways:
#   1. SAMPLE: randomly sample a fixed number of neighbours (not all)
#   2. AGGREGATE: use a learnable aggregator (mean, LSTM, or pool)
#   3. COMBINE: concatenate self + aggregated neighbours, then project
#
# This makes GraphSAGE INDUCTIVE — it can generalise to unseen nodes
# at inference time because it learns an aggregation function rather
# than fixed node embeddings. We implement the mean aggregator variant:
#
#   h'_i = sigma( W_self @ h_i + W_neigh @ MEAN(h_j for j in N(i)) )
#
# For the dense adjacency format, "sampling" is implemented by keeping
# at most K neighbours per node via random masking.
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sample_k: int = 10):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.sample_k = sample_k

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = h.size(0)

        # Neighbour sampling: for each node, keep at most sample_k neighbours
        # by zeroing out excess connections. At eval time, use all neighbours
        # for deterministic output (like dropout).
        if self.training and self.sample_k < n:
            # Count neighbours per node and build a sampling mask
            sample_mask = torch.zeros_like(adj)
            for i in range(n):
                neigh_idx = torch.where(adj[i] > 0)[0]
                if len(neigh_idx) <= self.sample_k:
                    sample_mask[i, neigh_idx] = 1.0
                else:
                    perm = torch.randperm(len(neigh_idx), device=h.device)[
                        : self.sample_k
                    ]
                    sample_mask[i, neigh_idx[perm]] = 1.0
            adj_sampled = sample_mask
        else:
            adj_sampled = adj

        # TODO: Compute mean neighbourhood aggregation and combine self + neighbour
        # Hint: deg_sampled = adj_sampled.sum(dim=1, keepdim=True).clamp(min=1.0)
        #       h_neigh = (adj_sampled @ h) / deg_sampled
        #       return self.W_self(h) + self.W_neigh(h_neigh)
        deg_sampled = ____  # noqa: F821
        h_neigh = ____  # noqa: F821
        h_self = ____  # noqa: F821
        h_agg = ____  # noqa: F821
        return ____  # noqa: F821


class GraphSAGE(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, n_classes: int, sample_k: int = 10
    ):
        super().__init__()
        self.l1 = GraphSAGELayer(in_dim, hidden_dim, sample_k=sample_k)
        self.l2 = GraphSAGELayer(hidden_dim, n_classes, sample_k=sample_k)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l1(h, adj))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.l2(h, adj)

    def embed(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Return the hidden-layer embedding (before classification head)."""
        return F.relu(self.l1(h, adj))


# ════════════════════════════════════════════════════════════════════════
# Training harness with ExperimentTracker integration
# ════════════════════════════════════════════════════════════════════════
HIDDEN_DIM = 16 if dataset_name == "Karate Club" else 64
EPOCHS = 100


def train_node_classifier(
    model: nn.Module,
    name: str,
    forward_arg: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
) -> tuple[list[float], list[float], list[float]]:
    """Train a GNN for node classification and log metrics to ExperimentTracker.

    Returns:
        train_losses: per-epoch training loss
        val_accs: per-epoch validation accuracy
        test_accs: per-epoch test accuracy
    """
    return asyncio.run(
        _train_node_classifier_async(model, name, forward_arg, epochs, lr, weight_decay)
    )


async def _train_node_classifier_async(
    model: nn.Module,
    name: str,
    forward_arg: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
) -> tuple[list[float], list[float], list[float]]:
    """Async core — uses the modern tracker.run(...) context manager."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] parameters: {n_params:,}")

    train_losses: list[float] = []
    val_accs: list[float] = []
    test_accs: list[float] = []

    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        # TODO: Log all hyperparams as a dict with ctx.log_params({...})
        # Hint: await ctx.log_params({"model_type": name, "hidden_dim": str(HIDDEN_DIM), ...})
        await ctx.log_params(
            {
                "model_type": ____,  # Hint: name
                "hidden_dim": ____,  # Hint: str(HIDDEN_DIM)
                "epochs": ____,  # Hint: str(epochs)
                "lr": ____,  # Hint: str(lr)
                "weight_decay": ____,  # Hint: str(weight_decay)
                "n_params": ____,  # Hint: str(n_params)
                "dataset": ____,  # Hint: dataset_name
                "n_nodes": ____,  # Hint: str(N)
                "n_edges": ____,  # Hint: str(n_edges_undirected)
            }
        )

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
                v_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
                t_acc = (preds[test_mask] == y[test_mask]).float().mean().item()
            val_accs.append(v_acc)
            test_accs.append(t_acc)

            await ctx.log_metrics(
                {
                    ____: loss.item(),
                    ____: v_acc,
                    ____: t_acc,
                },  # Hint: "train_loss", "val_accuracy", "test_accuracy"
                step=epoch + 1,
            )

            if (epoch + 1) % 25 == 0:
                print(
                    f"  [{name}] epoch {epoch+1:3d}  "
                    f"loss={loss.item():.4f}  val_acc={v_acc:.3f}  test_acc={t_acc:.3f}"
                )

        await ctx.log_metrics(
            {
                ____: train_losses[-1],  # Hint: "final_loss"
                ____: val_accs[-1],  # Hint: "final_val_accuracy"
                ____: test_accs[-1],  # Hint: "final_test_accuracy"
                ____: max(val_accs),  # Hint: "best_val_accuracy"
                ____: max(test_accs),  # Hint: "best_test_accuracy"
            }
        )

    return train_losses, val_accs, test_accs


# ── Train GCN ──────────────────────────────────────────────────────────
print(f"\n== Training GCN on {dataset_name} ==")
gcn = GCN(in_dim=F_dim, hidden_dim=HIDDEN_DIM, n_classes=n_classes)
gcn_losses, gcn_val, gcn_test = train_node_classifier(gcn, "GCN", A_norm)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(gcn_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses for GCN"
assert gcn_losses[-1] < gcn_losses[0], "GCN loss should decrease"
# INTERPRETATION: GCN uses a fixed aggregation scheme based on the graph
# Laplacian. Every node's new representation is a weighted average of its
# neighbours' features, where the weights come from the degree-normalised
# adjacency. This is equivalent to a 1-hop spectral filter on the graph.
print("\n--- Checkpoint 2 passed --- GCN trained\n")


# ── Train GAT ──────────────────────────────────────────────────────────
print(f"\n== Training GAT on {dataset_name} ==")
gat = GAT(in_dim=F_dim, hidden_dim=HIDDEN_DIM, n_classes=n_classes)
gat_losses, gat_val, gat_test = train_node_classifier(gat, "GAT", A)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(gat_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses for GAT"
assert gat_losses[-1] < gat_losses[0], "GAT loss should decrease"
# INTERPRETATION: GAT replaces the fixed Laplacian weights with LEARNED
# attention scores. Each node decides how much to attend to each neighbour
# based on the content of both nodes' features. This lets the model
# assign different importance to different neighbours — a citation from
# a highly relevant paper gets more weight than a tangential one.
print("\n--- Checkpoint 3 passed --- GAT trained\n")


# ── Train GraphSAGE ──────────────────────────────────────────────────
print(f"\n== Training GraphSAGE on {dataset_name} ==")
sage = GraphSAGE(in_dim=F_dim, hidden_dim=HIDDEN_DIM, n_classes=n_classes, sample_k=10)
sage_losses, sage_val, sage_test = train_node_classifier(sage, "GraphSAGE", A)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(sage_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses for GraphSAGE"
assert sage_losses[-1] < sage_losses[0], "GraphSAGE loss should decrease"
# INTERPRETATION: GraphSAGE is INDUCTIVE — it learns a generalised
# aggregation function that works on unseen nodes. During training, it
# randomly samples K neighbours per node (like dropout for graphs),
# which provides regularisation and makes it scalable to large graphs.
# The separate W_self and W_neigh projections let the model learn
# different transformations for a node's own features versus its
# neighbours' features.
print("\n--- Checkpoint 4 passed --- GraphSAGE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Quantitative comparison of all three GNN architectures
# ════════════════════════════════════════════════════════════════════════
# Compare the three models on: best validation accuracy, best test accuracy,
# convergence speed (epoch at which 90% of best accuracy is reached),
# and parameter count.

results = {
    "GCN": {
        "val_accs": gcn_val,
        "test_accs": gcn_test,
        "losses": gcn_losses,
        "model": gcn,
    },
    "GAT": {
        "val_accs": gat_val,
        "test_accs": gat_test,
        "losses": gat_losses,
        "model": gat,
    },
    "GraphSAGE": {
        "val_accs": sage_val,
        "test_accs": sage_test,
        "losses": sage_losses,
        "model": sage,
    },
}

print("\n=== GNN Architecture Comparison ===")
print(
    f"{'Model':>12} {'Params':>8} {'Best Val':>10} {'Best Test':>10} "
    f"{'Final Loss':>12} {'Conv@90%':>10}"
)
print("-" * 66)

for name, r in results.items():
    n_params = sum(p.numel() for p in r["model"].parameters())
    best_val = max(r["val_accs"])
    best_test = max(r["test_accs"])
    final_loss = r["losses"][-1]
    threshold = 0.9 * best_val
    conv_epoch = next(
        (i + 1 for i, a in enumerate(r["val_accs"]) if a >= threshold),
        EPOCHS,
    )
    print(
        f"{name:>12} {n_params:>8,} {best_val:>10.4f} {best_test:>10.4f} "
        f"{final_loss:>12.4f} {conv_epoch:>10}"
    )

best_name = max(results, key=lambda k: max(results[k]["val_accs"]))
best_model_obj = results[best_name]["model"]
print(f"\nBest model by validation accuracy: {best_name}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(results) == 3, "Should have results for all 3 architectures"
assert all(
    max(r["val_accs"]) > 0.3 for r in results.values()
), "All models should achieve > 30% accuracy (well above random for 7 classes)"
# INTERPRETATION: The comparison reveals architectural trade-offs:
# - GCN is simplest (fewest params) but uses fixed aggregation weights
# - GAT adds learnable attention but costs more parameters for the
#   attention heads — sometimes overkill on homogeneous graphs
# - GraphSAGE separates self vs neighbour projections and uses sampling,
#   making it more scalable but slightly different in convergence
# Cora is a homogeneous citation graph where all three tend to perform
# similarly. Differences become more pronounced on heterogeneous graphs.
print("\n--- Checkpoint 5 passed --- quantitative comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise learned node embeddings (2-D PCA projection)
# ════════════════════════════════════════════════════════════════════════
# Extract the hidden-layer embedding from each model and project to 2D
# using PCA (SVD-based, no sklearn dependency needed). If the GNN has
# learned meaningful representations, nodes of the same class should
# cluster together in the 2D projection.

print("\n== Node Embedding Visualisations ==")

embeddings = {}
for name, r in results.items():
    model = r["model"]
    model.eval()
    with torch.no_grad():
        if name == "GCN":
            emb = model.embed(X, A_norm).cpu().numpy()
        else:
            emb = model.embed(X, A).cpu().numpy()
    embeddings[name] = emb

# 2-D projection via PCA (SVD)
for name, emb in embeddings.items():
    emb_centered = emb - emb.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
    coords = emb_centered @ Vt.T[:, :2]

    print(f"\n{name} hidden embedding 2-D projection (first 3 nodes per class):")
    for c in range(min(n_classes, 7)):
        rows = coords[y_np == c][:3]
        if len(rows) == 0:
            continue
        pretty = ", ".join(f"({r[0]:+.2f}, {r[1]:+.2f})" for r in rows)
        print(f"  class {c}: {pretty}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
for name, emb in embeddings.items():
    assert emb.shape[0] == N, f"{name} embedding should have {N} rows"
    assert emb.shape[1] == HIDDEN_DIM, f"{name} embedding dim should be {HIDDEN_DIM}"
# INTERPRETATION: Good GNN embeddings show clear class separation in the
# 2D projection. Nodes of the same class cluster together because the
# GNN aggregated features from their citation neighbourhoods — papers
# in the same research area cite similar papers and share vocabulary,
# so their aggregated representations converge.
print("\n--- Checkpoint 6 passed --- node embeddings visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Link prediction: predict missing edges with dot-product decoder
# ════════════════════════════════════════════════════════════════════════
# Link prediction is the second major GNN task after node classification.
# Given a graph with some edges removed, can the model predict which
# node pairs should be connected?
#
# We use the trained GCN's embeddings as node representations, then
# predict edge existence with a dot-product decoder:
#   score(i, j) = sigmoid( z_i^T z_j )
#
# Positive edges: real edges in the graph
# Negative edges: random pairs that are NOT connected
print("\n== Link Prediction ==")


class LinkPredictor(nn.Module):
    """Encoder-decoder for link prediction.

    Encoder: any GNN that produces node embeddings.
    Decoder: dot product between node pairs -> edge probability.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

    def encode(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = self.encoder(h)
        h = F.relu(self.gcn1(h, a_norm))
        h = self.gcn2(h, a_norm)
        return h

    def decode(
        self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        """Dot-product decoder: score(i,j) = z_i^T z_j."""
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        a_norm: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(h, a_norm)
        return self.decode(z, src, dst)


# Prepare positive and negative edge samples
pos_src = torch.from_numpy(edge_index_np[0]).to(device)
pos_dst = torch.from_numpy(edge_index_np[1]).to(device)
n_pos = len(pos_src)

# Negative: sample random node pairs that are NOT connected
neg_src_list = []
neg_dst_list = []
rng_link = np.random.default_rng(42)
neg_count = 0
while neg_count < n_pos:
    s = rng_link.integers(0, N)
    d = rng_link.integers(0, N)
    if s != d and A_np[s, d] == 0:
        neg_src_list.append(s)
        neg_dst_list.append(d)
        neg_count += 1
neg_src = torch.tensor(neg_src_list, dtype=torch.long, device=device)
neg_dst = torch.tensor(neg_dst_list, dtype=torch.long, device=device)

print(f"  positive edges: {n_pos}, negative edges: {len(neg_src)}")

# Train the link predictor
link_model = LinkPredictor(F_dim, HIDDEN_DIM).to(device)
link_opt = torch.optim.Adam(link_model.parameters(), lr=1e-2, weight_decay=1e-4)

LINK_EPOCHS = 80
link_losses: list[float] = []
link_aucs: list[float] = []


async def _train_link_predictor_async():
    """Train the link predictor under a tracker.run(...) context."""
    async with tracker.run(experiment_name=exp_name, run_name="link_prediction") as ctx:
        await ctx.log_params(
            {
                ____: ____,  # Hint: "task", "link_prediction"
                ____: ____,  # Hint: "hidden_dim", str(HIDDEN_DIM)
                ____: ____,  # Hint: "epochs", str(LINK_EPOCHS)
                ____: ____,  # Hint: "n_pos_edges", str(n_pos)
                ____: ____,  # Hint: "n_neg_edges", str(len(neg_src))
            }
        )

        for epoch in range(LINK_EPOCHS):
            link_model.train()
            link_opt.zero_grad()

            pos_scores = link_model(X, A_norm, pos_src, pos_dst)
            neg_scores = link_model(X, A_norm, neg_src, neg_dst)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat(
                [
                    torch.ones(n_pos, device=device),
                    torch.zeros(len(neg_src), device=device),
                ]
            )
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            link_opt.step()
            link_losses.append(loss.item())

            link_model.eval()
            with torch.no_grad():
                p_scores = link_model(X, A_norm, pos_src, pos_dst)
                n_scores = link_model(X, A_norm, neg_src, neg_dst)
                n_sample = min(1000, n_pos, len(neg_src))
                p_sample = p_scores[:n_sample]
                n_sample_scores = n_scores[:n_sample]
                auc_approx = (p_sample > n_sample_scores).float().mean().item()
            link_aucs.append(auc_approx)

            await ctx.log_metrics(
                {
                    ____: loss.item(),
                    ____: auc_approx,
                },  # Hint: "link_loss", "link_auc_approx"
                step=epoch + 1,
            )

            if (epoch + 1) % 20 == 0:
                print(
                    f"  [LinkPred] epoch {epoch+1:3d}  "
                    f"loss={loss.item():.4f}  auc_approx={auc_approx:.3f}"
                )

        await ctx.log_metrics(
            {
                ____: link_losses[-1],  # Hint: "final_link_loss"
                ____: link_aucs[-1],  # Hint: "final_link_auc"
            }
        )


asyncio.run(_train_link_predictor_async())

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert len(link_losses) == LINK_EPOCHS, "Link prediction should train for all epochs"
assert link_losses[-1] < link_losses[0], "Link prediction loss should decrease"
assert (
    link_aucs[-1] > 0.55
), f"Link prediction AUC {link_aucs[-1]:.3f} should exceed random (0.5)"
# INTERPRETATION: The link predictor learns that connected papers have
# similar GNN embeddings. The dot-product decoder measures embedding
# similarity — high similarity predicts a citation link. An AUC > 0.5
# means the model ranks real edges higher than random non-edges. This
# is the foundation of recommendation systems on graphs: "papers you
# might want to cite" is just link prediction on the citation graph.
print("\n--- Checkpoint 7 passed --- link prediction trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Register the best model in the ModelRegistry
# ════════════════════════════════════════════════════════════════════════
# The ModelRegistry provides versioned model storage with metrics. We
# register the best node classifier and the link predictor.


async def register_best_models():
    """Register the best GNN model and link predictor in the registry."""
    if not has_registry:
        print("  ModelRegistry not available — skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}

    best_model_data = results[best_name]
    # TODO: Serialize the best model state_dict to bytes and register it
    # Hint: model_bytes = pickle.dumps(best_model_obj.state_dict())
    #       then registry.register_model(name=..., artifact=model_bytes, metrics=[...])
    model_bytes = ____  # noqa: F821
    version = await registry.register_model(
        name=f"m5_gnn_{best_name.lower()}",
        artifact=____,  # noqa: F821
        metrics=[
            MetricSpec(
                name="best_val_accuracy", value=max(best_model_data["val_accs"])
            ),
            MetricSpec(
                name="best_test_accuracy", value=max(best_model_data["test_accs"])
            ),
            MetricSpec(name="final_loss", value=best_model_data["losses"][-1]),
            MetricSpec(name="hidden_dim", value=float(HIDDEN_DIM)),
            MetricSpec(name="epochs", value=float(EPOCHS)),
        ],
    )
    model_versions[best_name] = version
    print(
        f"  Registered {best_name}: version={version.version}, "
        f"val_acc={max(best_model_data['val_accs']):.4f}"
    )

    # Register link predictor
    link_bytes = pickle.dumps(link_model.state_dict())
    link_version = await registry.register_model(
        name="m5_gnn_link_predictor",
        artifact=link_bytes,
        metrics=[
            MetricSpec(name="final_link_auc", value=link_aucs[-1]),
            MetricSpec(name="final_link_loss", value=link_losses[-1]),
        ],
    )
    model_versions["link_predictor"] = link_version
    print(
        f"  Registered link_predictor: version={link_version.version}, "
        f"auc={link_aucs[-1]:.4f}"
    )

    return model_versions


model_versions = asyncio.run(register_best_models())

# ── Checkpoint 8 ─────────────────────────────────────────────────────
if has_registry:
    assert len(model_versions) >= 2, "Should register at least 2 models"
    assert best_name in model_versions, f"{best_name} should be registered"
    assert "link_predictor" in model_versions, "Link predictor should be registered"
# INTERPRETATION: The ModelRegistry gives you versioned, queryable storage
# for model artifacts. Instead of saving .pt files to random directories,
# every model is tagged with its architecture, dataset, and performance
# metrics. When you need to deploy a GNN to production, the registry
# tells you which version achieved the best accuracy and what graph
# it was trained on.
print("\n--- Checkpoint 8 passed --- models registered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Visualise all training histories
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()

# TODO: Plot training loss curves for all three GNN architectures
# Hint: viz.training_history(metrics={"GCN train loss": gcn_losses, ...},
#                             x_label="Epoch", y_label="Training Loss")
fig_loss = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig_loss.write_html("ex_6_loss_curves.html")
print("Loss curves saved to ex_6_loss_curves.html")

# TODO: Plot validation accuracy curves for all three architectures
# Hint: metrics={"GCN val acc": gcn_val, "GAT val acc": gat_val, ...}
fig_acc = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig_acc.write_html("ex_6_accuracy_curves.html")
print("Accuracy curves saved to ex_6_accuracy_curves.html")

# TODO: Plot link prediction training (loss + AUC)
# Hint: metrics={"Link pred loss": link_losses, "Link pred AUC (approx)": link_aucs}
fig_link = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig_link.write_html("ex_6_link_prediction.html")
print("Link prediction curves saved to ex_6_link_prediction.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
import os

assert os.path.exists("ex_6_loss_curves.html"), "Loss curves HTML should be saved"
assert os.path.exists(
    "ex_6_accuracy_curves.html"
), "Accuracy curves HTML should be saved"
assert os.path.exists(
    "ex_6_link_prediction.html"
), "Link prediction HTML should be saved"
print("\n--- Checkpoint 9 passed --- all visualisations saved\n")


# Print final summary
print("\n=== Final Summary ===")
print(
    f"Dataset: {dataset_name} ({N} nodes, {n_edges_undirected} edges, {n_classes} classes)"
)
print(f"\nNode Classification (best validation accuracy):")
for name, r in results.items():
    n_params = sum(p.numel() for p in r["model"].parameters())
    print(
        f"  {name:>12}: val={max(r['val_accs']):.4f}  "
        f"test={max(r['test_accs']):.4f}  params={n_params:,}"
    )
print(f"\nLink Prediction:")
print(f"  AUC (approx): {link_aucs[-1]:.4f}")
print(f"\nBest model registered: {best_name}")

# Clean up database connection
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  GNN ARCHITECTURES:
  [x] GCN: message passing as matrix multiplication  H' = A_norm @ H @ W
      Fixed aggregation via degree-normalised adjacency. Simplest and
      fastest. Works well on homogeneous graphs like citation networks.
  [x] GAT: learned attention weights over neighbours
      Each node decides how much to attend to each neighbour based on
      feature content. More expressive but more parameters.
  [x] GraphSAGE: inductive learning with neighbourhood sampling
      Learns a generalised aggregation function — works on unseen nodes.
      Separates self vs neighbour projections for richer representations.

  GRAPH TASKS:
  [x] Node classification: predict a property of each node from the graph
  [x] Link prediction: predict missing edges using dot-product decoder
  [x] Node embeddings: visualise GNN representations with 2D PCA

  PRODUCTION TOOLS:
  [x] ExperimentTracker: logged all 3 architectures + link prediction run
  [x] ModelRegistry: registered best node classifier + link predictor

  NEXT: In Exercise 7, you will apply transfer learning — the same idea
  of reusing learned representations, but for images (ResNet-18 on CIFAR-10).
  Transfer learning is to images what GNN pre-training is to graphs.
"""
)
