# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 6: Graph Neural Networks
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement GCN layer with normalised adjacency matrix message passing
#   - Implement GraphSAGE with mean aggregation and inductive reasoning
#   - Implement GAT with learned per-edge attention weights
#   - Explain the over-smoothing problem and why 2-3 layers is typical
#   - Apply GNNs to node classification and visualise community structure
#
# PREREQUISITES:
#   Exercise 4 (attention mechanisms from transformers). Linear algebra
#   (matrix multiplication, eigenvectors for PCA). Understanding of graphs
#   (nodes, edges, adjacency matrix).
#
# ESTIMATED TIME: 60-90 minutes
#
# OBJECTIVE: Build GCN, GraphSAGE, and GAT from scratch using numpy to
#   understand message passing on graphs. Train all three on a node
#   classification task and compare learned representations.
#
# TASKS:
#   1. Create a synthetic graph with community structure (34+ nodes)
#   2. Implement GCN layer: normalised adjacency + message passing
#   3. Implement simplified GraphSAGE with mean aggregation
#   4. Implement GAT with learned attention weights
#   5. Train all three on node classification
#   6. Compare accuracy and learned representations
#   7. Visualise graph with predicted node colours
#   8. Discuss message passing intuition
#
# THEORY:
#   Graph neural networks learn node representations by aggregating
#   information from local neighbourhoods. Each layer propagates
#   features one hop further. After K layers, each node's embedding
#   encodes its K-hop neighbourhood structure.
#
#   GCN:       H^(l+1) = sigma(D^{-1/2} A D^{-1/2} H^(l) W^(l))
#   GraphSAGE: h_v = sigma(W * CONCAT(h_v, AGG({h_u : u in N(v)})))
#   GAT:       e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
#              alpha_ij = softmax_j(e_ij)
#              h_i' = sigma(SUM_j alpha_ij W h_j)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Create a synthetic graph with community structure
# ══════════════════════════════════════════════════════════════════════
# We build a planted partition graph: 4 communities of ~10 nodes each.
# Intra-community edges are dense, inter-community edges are sparse.
# This gives clear community structure for node classification.

rng = np.random.default_rng(seed=42)

# Build a stochastic block model: 4 communities
community_sizes = [10, 10, 10, 10]
n_nodes = sum(community_sizes)
n_communities = len(community_sizes)

# Assign ground-truth labels
labels = np.array(
    [c for c, size in enumerate(community_sizes) for _ in range(size)]
)

# Build adjacency matrix with planted community structure
# p_intra = probability of edge within community
# p_inter = probability of edge between communities
p_intra = 0.6
p_inter = 0.05

adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        p = p_intra if labels[i] == labels[j] else p_inter
        if rng.random() < p:
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0

# Add self-loops (standard in GCN: A_hat = A + I)
adj_with_self = adj_matrix + np.eye(n_nodes)

# Create node features: one-hot identity + Gaussian noise per community
# Each community gets a distinct feature centroid so the task is learnable
feature_dim = 16
node_features = rng.normal(0, 0.3, size=(n_nodes, feature_dim))
for c in range(n_communities):
    mask = labels == c
    # Add a community-specific signal to features
    centroid = rng.normal(0, 1.0, size=feature_dim)
    node_features[mask] += centroid

# Build networkx graph for visualisation later
G = nx.from_numpy_array(adj_matrix)
n_edges = int(adj_matrix.sum() / 2)

print(f"=== Synthetic Graph ===")
print(f"Nodes: {n_nodes}, Edges: {n_edges}")
print(f"Communities: {n_communities} (sizes: {community_sizes})")
print(f"Node feature dim: {feature_dim}")
print(f"Adjacency matrix shape: {adj_matrix.shape}")
print(f"Density: {n_edges / (n_nodes * (n_nodes - 1) / 2):.3f}")
print(f"p_intra={p_intra}, p_inter={p_inter}")

# Train/test split (use 50% for training, 50% for testing)
indices = np.arange(n_nodes)
rng.shuffle(indices)
n_train = n_nodes // 2
train_mask = np.zeros(n_nodes, dtype=bool)
train_mask[indices[:n_train]] = True
test_mask = ~train_mask

print(f"Train nodes: {train_mask.sum()}, Test nodes: {test_mask.sum()}")


# ══════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for 2D arrays. Numerically stable."""
    if x.ndim == 1:
        e = np.exp(x - x.max())
        return e / e.sum()
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss over a set of nodes.

    logits: (N, C) raw scores
    targets: (N,) integer class labels
    """
    probs = softmax(logits)
    n = len(targets)
    # Clip for numerical stability
    probs_clipped = np.clip(probs[np.arange(n), targets], 1e-12, 1.0)
    return -np.mean(np.log(probs_clipped))


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Classification accuracy."""
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == targets)


def one_hot(targets: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot encode integer labels."""
    oh = np.zeros((len(targets), n_classes), dtype=np.float64)
    oh[np.arange(len(targets)), targets] = 1.0
    return oh


# ══════════════════════════════════════════════════════════════════════
# TASK 2: GCN — Graph Convolutional Network from scratch
# ══════════════════════════════════════════════════════════════════════
# THEORY: GCN performs spectral convolution approximated as:
#   H^(l+1) = sigma( D_hat^{-1/2} A_hat D_hat^{-1/2} H^(l) W^(l) )
#
# where A_hat = A + I (adjacency with self-loops),
#       D_hat = degree matrix of A_hat,
#       W^(l) = learnable weight matrix for layer l.
#
# The normalised adjacency D^{-1/2} A D^{-1/2} is a symmetric
# normalisation that:
#   1. Prevents feature scale explosion across layers
#   2. Weights neighbours inversely by their degree (high-degree
#      nodes contribute less per-edge)
#   3. Is computed ONCE before training (it depends only on topology)
#
# Message passing interpretation:
#   Each node sends its features to all neighbours.
#   Each node averages received messages (with degree normalisation).
#   The result is transformed by W and passed through activation.

def compute_normalised_adjacency(A: np.ndarray) -> np.ndarray:
    """Compute D^{-1/2} A D^{-1/2} for symmetric normalisation.

    A should already include self-loops (A_hat = A + I).
    """
    # Degree vector
    d = A.sum(axis=1)
    # TODO: Compute D^{-1/2} as a vector, handle isolated nodes (d=0)
    d_inv_sqrt = ____  # Hint: np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    # TODO: Apply symmetric normalisation: D^{-1/2} A D^{-1/2}
    #   Multiply row i by d_inv_sqrt[i], col j by d_inv_sqrt[j]
    A_norm = ____  # Hint: A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return A_norm


# Precompute normalised adjacency (topology is fixed)
A_hat_norm = compute_normalised_adjacency(adj_with_self)

print(f"\n=== GCN: Normalised Adjacency ===")
print(f"A_hat (A + I) shape: {adj_with_self.shape}")
print(f"Normalised adjacency range: [{A_hat_norm.min():.4f}, {A_hat_norm.max():.4f}]")
print(f"Row sums (should be ~1 for regular graphs): "
      f"mean={A_hat_norm.sum(axis=1).mean():.3f}")


class GCNLayer:
    """Single GCN layer: H' = sigma(A_norm @ H @ W + b)."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        # Xavier initialisation
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = rng.normal(0, scale, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        # Gradient storage
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        # Cache for backward pass
        self._input = None
        self._pre_act = None

    def forward(self, A_norm: np.ndarray, H: np.ndarray, apply_relu: bool = True) -> np.ndarray:
        """Forward pass: aggregate neighbours then transform.

        A_norm: (N, N) normalised adjacency
        H: (N, in_dim) node features
        Returns: (N, out_dim) updated features
        """
        # TODO: Implement GCN message passing
        #   1. Aggregate: aggregated = A_norm @ H  (each node gets weighted sum of neighbour features)
        #   2. Transform: pre_act = aggregated @ W + b
        #   3. Apply relu if apply_relu=True
        aggregated = ____  # Hint: A_norm @ H
        self._input = aggregated
        self._pre_act = ____  # Hint: aggregated @ self.W + self.b
        if apply_relu:
            return relu(self._pre_act)
        return self._pre_act


class GCN:
    """2-layer GCN for node classification.

    Layer 1: (feature_dim) -> (hidden_dim) with ReLU
    Layer 2: (hidden_dim) -> (n_classes) with softmax (no ReLU)
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, rng: np.random.Generator):
        self.layer1 = GCNLayer(in_dim, hidden_dim, rng)
        self.layer2 = GCNLayer(hidden_dim, n_classes, rng)
        self.name = "GCN"

    def forward(self, A_norm: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Forward pass through 2-layer GCN.

        Returns raw logits (N, n_classes).
        """
        # TODO: Forward through layer1 (with ReLU), then layer2 (no ReLU)
        h = ____  # Hint: self.layer1.forward(A_norm, X, apply_relu=True)
        self._hidden = h  # Save for visualisation
        logits = ____  # Hint: self.layer2.forward(A_norm, h, apply_relu=False)
        return logits

    def get_embeddings(self, A_norm: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Get hidden layer embeddings for visualisation."""
        self.forward(A_norm, X)
        return self._hidden


def train_gcn(
    model: GCN,
    A_norm: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.01,
) -> dict:
    """Train GCN using manual gradient descent.

    We compute gradients of cross-entropy loss w.r.t. all parameters
    via manual backpropagation through the GCN layers.
    """
    n_classes = model.layer2.W.shape[1]
    targets_oh = one_hot(labels, n_classes)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs):
        # ── Forward pass ──
        h1 = model.layer1.forward(A_norm, X, apply_relu=True)
        logits = model.layer2.forward(A_norm, h1, apply_relu=False)
        probs = softmax(logits)

        # ── Loss (only on train nodes) ──
        train_loss = cross_entropy_loss(logits[train_mask], labels[train_mask])
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # ── Backward pass ──
        # Gradient of cross-entropy w.r.t. logits: (probs - one_hot) / N_train
        grad_logits = probs.copy()
        grad_logits[train_mask] -= targets_oh[train_mask]
        # Zero out gradient for test nodes (they don't contribute to loss)
        grad_logits[~train_mask] = 0.0
        grad_logits /= train_mask.sum()

        # Layer 2 gradients (no ReLU)
        # logits = A_norm @ h1 @ W2 + b2
        # d(loss)/d(W2) = (A_norm @ h1)^T @ grad_logits
        agg2 = A_norm @ h1
        model.layer2.grad_W = agg2.T @ grad_logits
        model.layer2.grad_b = grad_logits.sum(axis=0)

        # Backprop through layer 2 to get gradient w.r.t. h1
        # grad_h1_agg = grad_logits @ W2^T  (gradient w.r.t. aggregated input of layer 2)
        grad_h1_agg = grad_logits @ model.layer2.W.T
        # Backprop through A_norm aggregation: grad_h1 = A_norm^T @ grad_h1_agg
        # (A_norm is symmetric so A_norm^T = A_norm)
        grad_h1 = A_norm @ grad_h1_agg

        # ReLU mask from layer 1
        relu_mask = (model.layer1._pre_act > 0).astype(np.float64)
        grad_pre_relu = grad_h1 * relu_mask

        # Layer 1 gradients
        agg1 = A_norm @ X
        model.layer1.grad_W = agg1.T @ grad_pre_relu
        model.layer1.grad_b = grad_pre_relu.sum(axis=0)

        # ── Parameter update (SGD) ──
        model.layer1.W -= lr * model.layer1.grad_W
        model.layer1.b -= lr * model.layer1.grad_b
        model.layer2.W -= lr * model.layer2.grad_W
        model.layer2.b -= lr * model.layer2.grad_b

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    return history


print(f"\n=== Training GCN (2 layers, hidden_dim=16) ===")
gcn_model = GCN(in_dim=feature_dim, hidden_dim=16, n_classes=n_communities, rng=rng)
gcn_history = train_gcn(
    gcn_model, A_hat_norm, node_features, labels,
    train_mask, test_mask, n_epochs=200, lr=0.01,
)

gcn_final_logits = gcn_model.forward(A_hat_norm, node_features)
gcn_test_acc = accuracy(gcn_final_logits[test_mask], labels[test_mask])
print(f"GCN final test accuracy: {gcn_test_acc:.3f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: GraphSAGE — Sampling and Aggregating from Neighbourhoods
# ══════════════════════════════════════════════════════════════════════
# THEORY: GraphSAGE differs from GCN in two key ways:
#   1. It CONCATENATES self-features with aggregated neighbour features
#      (instead of mixing them via the normalised adjacency)
#   2. It can SAMPLE a fixed number of neighbours (for scalability)
#   3. It is INDUCTIVE: can generalise to unseen nodes at test time
#      because it learns an aggregation FUNCTION, not node-specific
#      embeddings
#
# Mean aggregation (simplest variant):
#   h_N(v) = MEAN({h_u : u in N(v)})    -- aggregate neighbours
#   h_v' = sigma(W * CONCAT(h_v, h_N(v)))  -- combine with self
#
# Other aggregation options: LSTM, pooling, attention (-> GAT)

def get_neighbours(adj: np.ndarray, node: int) -> np.ndarray:
    """Get indices of neighbours for a node (excluding self)."""
    return np.where(adj[node] > 0)[0]


class GraphSAGELayer:
    """Single GraphSAGE layer with mean aggregation.

    Differs from GCN: concatenates self-features with mean of neighbour
    features, then applies linear transform.
    """

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        # W operates on CONCAT(self, agg_neighbours) so input is 2 * in_dim
        scale = np.sqrt(2.0 / (2 * in_dim + out_dim))
        self.W = rng.normal(0, scale, size=(2 * in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self._concat_input = None
        self._pre_act = None

    def forward(
        self, adj: np.ndarray, H: np.ndarray, apply_relu: bool = True
    ) -> np.ndarray:
        """Forward pass: mean-aggregate neighbours, concat with self, transform.

        adj: (N, N) raw adjacency matrix (no self-loops needed)
        H: (N, in_dim) node features
        """
        n_nodes = H.shape[0]
        in_dim = H.shape[1]

        # TODO: Implement mean aggregation of neighbours for each node
        #   For each node v: agg[v] = mean of H[nbrs] where nbrs = get_neighbours(adj, v)
        #   If isolated node (no neighbours), agg[v] stays zero
        agg = np.zeros((n_nodes, in_dim), dtype=np.float64)
        for v in range(n_nodes):
            nbrs = get_neighbours(adj, v)
            if len(nbrs) > 0:
                agg[v] = ____  # Hint: H[nbrs].mean(axis=0)

        # TODO: Concatenate self-features with aggregated neighbour features
        concat = ____  # Hint: np.concatenate([H, agg], axis=1)  -> (N, 2*in_dim)
        self._concat_input = concat

        # TODO: Linear transform + optional activation
        self._pre_act = ____  # Hint: concat @ self.W + self.b
        if apply_relu:
            return relu(self._pre_act)
        return self._pre_act


class GraphSAGE:
    """2-layer GraphSAGE for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, rng: np.random.Generator):
        self.layer1 = GraphSAGELayer(in_dim, hidden_dim, rng)
        self.layer2 = GraphSAGELayer(hidden_dim, n_classes, rng)
        self.name = "GraphSAGE"

    def forward(self, adj: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Forward pass through 2-layer GraphSAGE."""
        # TODO: Forward through layer1 (apply_relu=True), then layer2 (apply_relu=False)
        h = ____  # Hint: self.layer1.forward(adj, X, apply_relu=True)
        self._hidden = h
        logits = ____  # Hint: self.layer2.forward(adj, h, apply_relu=False)
        return logits

    def get_embeddings(self, adj: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Get hidden layer embeddings for visualisation."""
        self.forward(adj, X)
        return self._hidden


def train_sage(
    model: GraphSAGE,
    adj: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.01,
) -> dict:
    """Train GraphSAGE using manual gradient descent.

    Backprop through the concatenation and mean aggregation steps.
    """
    n_classes = model.layer2.W.shape[1]
    n_nodes = X.shape[0]
    targets_oh = one_hot(labels, n_classes)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs):
        # ── Forward pass ──
        h1 = model.layer1.forward(adj, X, apply_relu=True)
        logits = model.layer2.forward(adj, h1, apply_relu=False)
        probs = softmax(logits)

        # ── Loss ──
        train_loss = cross_entropy_loss(logits[train_mask], labels[train_mask])
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # ── Backward pass ──
        grad_logits = probs.copy()
        grad_logits[train_mask] -= targets_oh[train_mask]
        grad_logits[~train_mask] = 0.0
        grad_logits /= train_mask.sum()

        # Layer 2 gradients
        model.layer2.grad_W = model.layer2._concat_input.T @ grad_logits
        model.layer2.grad_b = grad_logits.sum(axis=0)

        # Backprop to h1 via layer 2
        grad_concat2 = grad_logits @ model.layer2.W.T  # (N, 2*hidden_dim)
        hidden_dim = h1.shape[1]

        # Split gradient for concat(h1, agg_neighbours(h1))
        grad_self2 = grad_concat2[:, :hidden_dim]
        grad_agg2 = grad_concat2[:, hidden_dim:]

        # Backprop mean aggregation: distribute gradient to neighbours
        grad_h1 = grad_self2.copy()
        for v in range(n_nodes):
            nbrs = get_neighbours(adj, v)
            if len(nbrs) > 0:
                grad_h1[nbrs] += grad_agg2[v] / len(nbrs)

        # ReLU mask from layer 1
        relu_mask = (model.layer1._pre_act > 0).astype(np.float64)
        grad_pre_relu = grad_h1 * relu_mask

        # Layer 1 gradients
        model.layer1.grad_W = model.layer1._concat_input.T @ grad_pre_relu
        model.layer1.grad_b = grad_pre_relu.sum(axis=0)

        # ── Parameter update ──
        model.layer1.W -= lr * model.layer1.grad_W
        model.layer1.b -= lr * model.layer1.grad_b
        model.layer2.W -= lr * model.layer2.grad_W
        model.layer2.b -= lr * model.layer2.grad_b

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    return history


print(f"\n=== Training GraphSAGE (2 layers, mean aggregation) ===")
sage_model = GraphSAGE(
    in_dim=feature_dim, hidden_dim=16, n_classes=n_communities, rng=rng,
)
sage_history = train_sage(
    sage_model, adj_matrix, node_features, labels,
    train_mask, test_mask, n_epochs=200, lr=0.01,
)

sage_final_logits = sage_model.forward(adj_matrix, node_features)
sage_test_acc = accuracy(sage_final_logits[test_mask], labels[test_mask])
print(f"GraphSAGE final test accuracy: {sage_test_acc:.3f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: GAT — Graph Attention Network from scratch
# ══════════════════════════════════════════════════════════════════════
# THEORY: GAT learns DIFFERENT weights for different neighbours using
#   an attention mechanism. Instead of treating all neighbours equally
#   (GCN) or using a simple mean (GraphSAGE), GAT computes:
#
#   1. Transform features: z_i = W h_i
#   2. Compute attention coefficients:
#      e_ij = LeakyReLU(a^T [z_i || z_j])
#      where || denotes concatenation, a is a learnable vector
#   3. Normalise with softmax over neighbours:
#      alpha_ij = exp(e_ij) / SUM_{k in N(i)} exp(e_ik)
#   4. Aggregate with attention weights:
#      h_i' = sigma( SUM_{j in N(i)} alpha_ij * z_j )
#
# The attention lets each node SELECTIVELY attend to its most
# informative neighbours. A node in community A connected to one
# node in community B can learn to down-weight that cross-community
# edge.

def leaky_relu(x: np.ndarray, negative_slope: float = 0.2) -> np.ndarray:
    """LeakyReLU: x if x > 0, else negative_slope * x."""
    return np.where(x > 0, x, negative_slope * x)


def leaky_relu_derivative(x: np.ndarray, negative_slope: float = 0.2) -> np.ndarray:
    """Derivative of LeakyReLU."""
    return np.where(x > 0, 1.0, negative_slope)


class GATLayer:
    """Single GAT layer with single-head attention.

    Computes attention coefficients for each edge, then aggregates
    neighbour features weighted by attention.
    """

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        # Feature transform W: (in_dim, out_dim)
        self.W = rng.normal(0, scale, size=(in_dim, out_dim))
        # Attention vector a: (2 * out_dim,)
        # a^T [Wh_i || Wh_j] = a_left^T Wh_i + a_right^T Wh_j
        self.a_left = rng.normal(0, scale, size=out_dim)
        self.a_right = rng.normal(0, scale, size=out_dim)
        # Gradient storage
        self.grad_W = np.zeros_like(self.W)
        self.grad_a_left = np.zeros_like(self.a_left)
        self.grad_a_right = np.zeros_like(self.a_right)

        # Cache
        self._Z = None
        self._attention = None
        self._e_raw = None
        self._input = None

    def forward(
        self, adj: np.ndarray, H: np.ndarray, apply_relu: bool = True,
    ) -> np.ndarray:
        """Forward pass with attention mechanism.

        adj: (N, N) adjacency with self-loops
        H: (N, in_dim) node features
        Returns: (N, out_dim) attention-weighted features
        """
        n_nodes = H.shape[0]
        self._input = H

        # TODO: Step 1 — Linear transform all node features
        Z = ____  # Hint: H @ self.W  -> (N, out_dim)
        self._Z = Z

        # TODO: Step 2 — Compute attention scores for all edges
        #   e_ij = LeakyReLU(a_left^T z_i + a_right^T z_j)
        #   score_left[i] = Z[i] @ a_left, score_right[j] = Z[j] @ a_right
        #   e_raw[i, j] = score_left[i] + score_right[j]
        score_left = ____  # Hint: Z @ self.a_left   -> (N,)
        score_right = ____  # Hint: Z @ self.a_right  -> (N,)
        e_raw = ____  # Hint: score_left[:, None] + score_right[None, :]  -> (N, N)
        self._e_raw = e_raw
        e = leaky_relu(e_raw)

        # TODO: Step 3 — Mask non-edges (set to -1e9 so softmax gives 0)
        mask = (adj > 0)
        e_masked = ____  # Hint: np.where(mask, e, -1e9)

        # TODO: Step 4 — Row-wise softmax over masked attention scores
        attention = ____  # Hint: softmax(e_masked)  -> (N, N)
        # Zero out non-edges explicitly (softmax of -1e9 is ~0, but be safe)
        attention = attention * mask
        self._attention = attention

        # TODO: Step 5 — Weighted aggregation of transformed features
        out = ____  # Hint: attention @ Z  -> (N, out_dim)

        if apply_relu:
            self._pre_act = out
            return relu(out)
        return out


class GAT:
    """2-layer GAT for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, rng: np.random.Generator):
        self.layer1 = GATLayer(in_dim, hidden_dim, rng)
        self.layer2 = GATLayer(hidden_dim, n_classes, rng)
        self.name = "GAT"

    def forward(self, adj: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Forward pass through 2-layer GAT."""
        # TODO: Forward through layer1 (apply_relu=True), then layer2 (apply_relu=False)
        h = ____  # Hint: self.layer1.forward(adj, X, apply_relu=True)
        self._hidden = h
        logits = ____  # Hint: self.layer2.forward(adj, h, apply_relu=False)
        return logits

    def get_embeddings(self, adj: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Get hidden layer embeddings for visualisation."""
        self.forward(adj, X)
        return self._hidden


def train_gat(
    model: GAT,
    adj: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.005,
) -> dict:
    """Train GAT using manual gradient descent.

    Backprop through attention mechanism is more involved than GCN:
    gradients flow through softmax attention weights and the learned
    attention vector.
    """
    n_classes = model.layer2.W.shape[1]
    n_nodes = X.shape[0]
    targets_oh = one_hot(labels, n_classes)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs):
        # ── Forward pass ──
        h1 = model.layer1.forward(adj, X, apply_relu=True)
        logits = model.layer2.forward(adj, h1, apply_relu=False)
        probs = softmax(logits)

        # ── Loss ──
        train_loss = cross_entropy_loss(logits[train_mask], labels[train_mask])
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # ── Backward pass through layer 2 (no ReLU) ──
        grad_logits = probs.copy()
        grad_logits[train_mask] -= targets_oh[train_mask]
        grad_logits[~train_mask] = 0.0
        grad_logits /= train_mask.sum()

        # Gradient w.r.t. layer 2 output = attention @ Z2
        attn2 = model.layer2._attention  # (N, N)
        Z2 = model.layer2._Z            # (N, out_dim)

        # d(loss)/d(Z2) = attn2^T @ grad_logits
        grad_Z2 = attn2.T @ grad_logits  # (N, n_classes)

        # d(loss)/d(W2) = H1^T @ grad_Z2
        model.layer2.grad_W = h1.T @ grad_Z2

        # Gradient of attention vector a (layer 2)
        grad_attn = grad_logits @ Z2.T  # (N, N)
        mask2 = (adj > 0)
        grad_e = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            a_i = attn2[i]  # attention weights for node i
            g_i = grad_attn[i]  # gradient w.r.t. attention for node i
            # Jacobian of softmax: diag(a) - a a^T
            s = (g_i * a_i).sum()
            grad_e[i] = a_i * (g_i - s)
        grad_e *= mask2
        # Through LeakyReLU
        grad_e *= leaky_relu_derivative(model.layer2._e_raw)

        # a_left gradient: sum_i sum_j grad_e_ij * Z2[i]
        model.layer2.grad_a_left = (grad_e.sum(axis=1) @ Z2)
        model.layer2.grad_a_right = (grad_e.sum(axis=0) @ Z2)

        # Backprop to h1: gradient through Z2 = h1 @ W2
        grad_h1 = grad_Z2 @ model.layer2.W.T  # (N, hidden_dim)

        # Through ReLU of layer 1
        relu_mask = (model.layer1._pre_act > 0).astype(np.float64)
        grad_h1 *= relu_mask

        # Layer 1 gradients (simplified: treat attention as fixed)
        attn1 = model.layer1._attention
        Z1 = model.layer1._Z
        grad_Z1 = attn1.T @ grad_h1
        model.layer1.grad_W = X.T @ grad_Z1

        # Attention vector gradients for layer 1
        grad_attn1 = grad_h1 @ Z1.T
        mask1 = (adj > 0)
        grad_e1 = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            a_i = attn1[i]
            g_i = grad_attn1[i]
            s = (g_i * a_i).sum()
            grad_e1[i] = a_i * (g_i - s)
        grad_e1 *= mask1
        grad_e1 *= leaky_relu_derivative(model.layer1._e_raw)
        model.layer1.grad_a_left = (grad_e1.sum(axis=1) @ Z1)
        model.layer1.grad_a_right = (grad_e1.sum(axis=0) @ Z1)

        # ── Parameter update ──
        for layer in [model.layer1, model.layer2]:
            layer.W -= lr * layer.grad_W
            layer.a_left -= lr * layer.grad_a_left
            layer.a_right -= lr * layer.grad_a_right

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    return history


print(f"\n=== Training GAT (2 layers, single-head attention) ===")
gat_model = GAT(
    in_dim=feature_dim, hidden_dim=16, n_classes=n_communities, rng=rng,
)
gat_history = train_gat(
    gat_model, adj_with_self, node_features, labels,
    train_mask, test_mask, n_epochs=200, lr=0.005,
)

gat_final_logits = gat_model.forward(adj_with_self, node_features)
gat_test_acc = accuracy(gat_final_logits[test_mask], labels[test_mask])
print(f"GAT final test accuracy: {gat_test_acc:.3f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare accuracy and training dynamics
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"=== Model Comparison ===")
print(f"{'='*60}")
print(f"{'Model':<15} {'Test Acc':>10} {'Final Loss':>12} {'Parameters':>12}")
print(f"{'-'*15} {'-'*10} {'-'*12} {'-'*12}")

# Count parameters
gcn_params = (gcn_model.layer1.W.size + gcn_model.layer1.b.size +
              gcn_model.layer2.W.size + gcn_model.layer2.b.size)
sage_params = (sage_model.layer1.W.size + sage_model.layer1.b.size +
               sage_model.layer2.W.size + sage_model.layer2.b.size)
gat_params = (gat_model.layer1.W.size + gat_model.layer1.a_left.size +
              gat_model.layer1.a_right.size +
              gat_model.layer2.W.size + gat_model.layer2.a_left.size +
              gat_model.layer2.a_right.size)

results = [
    ("GCN", gcn_test_acc, gcn_history["train_loss"][-1], gcn_params),
    ("GraphSAGE", sage_test_acc, sage_history["train_loss"][-1], sage_params),
    ("GAT", gat_test_acc, gat_history["train_loss"][-1], gat_params),
]

for name, acc, loss, params in results:
    print(f"{name:<15} {acc:>10.3f} {loss:>12.4f} {params:>12}")

print(f"\n--- Analysis ---")
print(f"GCN: Simplest. Symmetric normalisation treats all neighbours equally.")
print(f"  Strength: computationally efficient (matrix multiply with precomputed A_norm).")
print(f"  Weakness: cannot learn to weight neighbours differently.")
print(f"")
print(f"GraphSAGE: Concatenates self with mean of neighbours.")
print(f"  Strength: inductive (works on unseen nodes), separates self vs neighbour info.")
print(f"  Weakness: mean aggregation loses structural information.")
print(f"")
print(f"GAT: Learns attention weights per edge.")
print(f"  Strength: can focus on informative neighbours, ignore noisy connections.")
print(f"  Weakness: more parameters, harder to train, attention is pairwise (O(E)).")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare learned representations
# ══════════════════════════════════════════════════════════════════════
# Use t-SNE (via simple PCA fallback in numpy) to visualise the
# hidden representations learned by each model.

def pca_2d(X: np.ndarray) -> np.ndarray:
    """Project to 2D using PCA (eigenvectors of covariance matrix)."""
    X_centered = X - X.mean(axis=0)
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top 2 eigenvectors (eigh returns ascending order)
    top2 = eigenvectors[:, -2:][:, ::-1]
    return X_centered @ top2


print(f"\n=== Learned Representations (PCA to 2D) ===")

# Get embeddings from each model
gcn_emb = gcn_model.get_embeddings(A_hat_norm, node_features)
sage_emb = sage_model.get_embeddings(adj_matrix, node_features)
gat_emb = gat_model.get_embeddings(adj_with_self, node_features)

for name, emb in [("GCN", gcn_emb), ("GraphSAGE", sage_emb), ("GAT", gat_emb)]:
    proj = pca_2d(emb)
    # Measure cluster separation: ratio of inter-cluster to intra-cluster distance
    intra_dists = []
    inter_dists = []
    for c in range(n_communities):
        c_mask = labels == c
        c_points = proj[c_mask]
        centroid = c_points.mean(axis=0)
        intra_dists.append(np.mean(np.linalg.norm(c_points - centroid, axis=1)))
        for c2 in range(c + 1, n_communities):
            c2_mask = labels == c2
            c2_centroid = proj[c2_mask].mean(axis=0)
            inter_dists.append(np.linalg.norm(centroid - c2_centroid))

    avg_intra = np.mean(intra_dists)
    avg_inter = np.mean(inter_dists)
    separation = avg_inter / (avg_intra + 1e-8)
    print(f"  {name}: intra-cluster={avg_intra:.3f}, inter-cluster={avg_inter:.3f}, "
          f"separation ratio={separation:.2f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Visualise graph with predicted node colours
# ══════════════════════════════════════════════════════════════════════

# Colour maps for communities
community_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def plot_graph_predictions(
    G: nx.Graph,
    logits: np.ndarray,
    true_labels: np.ndarray,
    model_name: str,
    test_mask: np.ndarray,
    filename: str,
) -> None:
    """Visualise graph with nodes coloured by predicted class.

    Correct predictions: filled circles.
    Incorrect predictions: marked with 'X'.
    """
    preds = np.argmax(logits, axis=1)
    pred_colors = [community_colors[p] for p in preds]
    correct = preds == true_labels

    # Use spring layout for clear community visualisation
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(n_nodes))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: ground truth
    true_colors = [community_colors[l] for l in true_labels]
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=axes[0])
    nx.draw_networkx_nodes(
        G, pos, node_color=true_colors, node_size=200,
        edgecolors="black", linewidths=1.0, ax=axes[0],
    )
    nx.draw_networkx_labels(G, pos, font_size=7, ax=axes[0])
    axes[0].set_title("Ground Truth Communities", fontsize=14)
    axes[0].axis("off")

    # Right: predictions (highlight train vs test)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=axes[1])
    # Draw train nodes
    train_nodes = [i for i in range(n_nodes) if train_mask[i]]
    test_nodes = [i for i in range(n_nodes) if test_mask[i]]
    nx.draw_networkx_nodes(
        G, pos, nodelist=train_nodes,
        node_color=[pred_colors[i] for i in train_nodes],
        node_size=200, edgecolors="black", linewidths=1.0,
        node_shape="o", ax=axes[1],
    )
    # Test nodes: square if correct, diamond if wrong
    correct_test = [i for i in test_nodes if correct[i]]
    wrong_test = [i for i in test_nodes if not correct[i]]
    if correct_test:
        nx.draw_networkx_nodes(
            G, pos, nodelist=correct_test,
            node_color=[pred_colors[i] for i in correct_test],
            node_size=250, edgecolors="green", linewidths=2.0,
            node_shape="s", ax=axes[1],
        )
    if wrong_test:
        nx.draw_networkx_nodes(
            G, pos, nodelist=wrong_test,
            node_color=[pred_colors[i] for i in wrong_test],
            node_size=300, edgecolors="red", linewidths=2.5,
            node_shape="D", ax=axes[1],
        )
    nx.draw_networkx_labels(G, pos, font_size=7, ax=axes[1])

    test_acc = accuracy(logits[test_mask], true_labels[test_mask])
    axes[1].set_title(
        f"{model_name} Predictions (test acc: {test_acc:.1%})\n"
        f"circle=train, square=test correct, diamond=test wrong",
        fontsize=12,
    )
    axes[1].axis("off")

    fig.suptitle(f"Graph Node Classification: {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


print(f"\n=== Graph Visualisation ===")
plot_graph_predictions(
    G, gcn_final_logits, labels, "GCN", test_mask, "ex6_gcn_graph.png",
)
plot_graph_predictions(
    G, sage_final_logits, labels, "GraphSAGE", test_mask, "ex6_sage_graph.png",
)
plot_graph_predictions(
    G, gat_final_logits, labels, "GAT", test_mask, "ex6_gat_graph.png",
)


# Training loss comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, hist) in zip(axes, [
    ("GCN", gcn_history), ("GraphSAGE", sage_history), ("GAT", gat_history),
]):
    ax.plot(hist["train_loss"], label="Train Loss", color="#e41a1c")
    ax.plot(hist["train_acc"], label="Train Acc", color="#377eb8", linestyle="--")
    ax.plot(hist["test_acc"], label="Test Acc", color="#4daf4a", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(f"{name} Training Dynamics")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ex6_training_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: ex6_training_comparison.png")


# Attention weight visualisation for GAT
print(f"\n=== GAT Attention Weights (Layer 1) ===")
attn_weights = gat_model.layer1._attention
# Show attention distribution for a few nodes
for node_id in [0, 10, 20, 30]:
    nbrs = get_neighbours(adj_with_self, node_id)
    attn_vals = attn_weights[node_id, nbrs]
    top_k = min(5, len(nbrs))
    top_indices = np.argsort(attn_vals)[-top_k:][::-1]
    print(f"  Node {node_id} (community {labels[node_id]}): "
          f"{len(nbrs)} neighbours")
    for idx in top_indices:
        nbr = nbrs[idx]
        same_comm = "same" if labels[nbr] == labels[node_id] else "diff"
        print(f"    -> node {nbr} (comm {labels[nbr]}, {same_comm}): "
              f"attn={attn_vals[idx]:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Message passing intuition
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"=== Message Passing Intuition ===")
print(f"{'='*60}")
print(f"""
All three architectures share the MESSAGE PASSING paradigm:

  1. AGGREGATE: Each node collects features from its neighbours.
  2. UPDATE:    Each node updates its own representation using
                the aggregated information.
  3. READOUT:   Final node representations are used for the task
                (classification, regression, link prediction).

How they differ in the AGGREGATE step:

  GCN:       Weighted sum using D^{{-1/2}} A D^{{-1/2}}.
             All neighbours contribute equally (modulo degree).
             Simple, fast, effective for homophilous graphs.

  GraphSAGE: Mean of neighbour features, CONCATENATED with self.
             Explicit separation of "what I know" vs "what my
             neighbours say". Inductive: works on unseen nodes.

  GAT:       Learned ATTENTION weights per edge.
             Each node decides HOW MUCH to listen to each
             neighbour. Most expressive, but most expensive.

Depth and receptive field:
  - 1 GNN layer  = 1-hop neighbourhood
  - 2 GNN layers = 2-hop neighbourhood
  - K GNN layers = K-hop neighbourhood

  CAREFUL: Too many layers causes OVER-SMOOTHING. After many
  rounds of averaging, all node representations converge to the
  same vector. 2-3 layers is typical for most tasks.

Why this matters for real applications:
  - Social networks:  predict user interests from friend activity
  - Molecules:        predict drug properties from atomic bonds
  - Knowledge graphs: infer missing relationships between entities
  - Fraud detection:  identify suspicious transaction patterns
""")

# Demonstrate over-smoothing with increasing layers
print(f"=== Over-Smoothing Demonstration ===")
print(f"Measuring representation similarity as we add more GCN layers:")

H = node_features.copy()
for n_layers in range(1, 7):
    # TODO: Apply one more round of neighbourhood averaging (A_hat_norm @ H)
    H = ____  # Hint: A_hat_norm @ H
    # Measure how similar all node representations are (cosine similarity)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    H_normed = H / norms
    cos_sim_matrix = H_normed @ H_normed.T
    # Average pairwise cosine similarity (excluding diagonal)
    mask_diag = ~np.eye(n_nodes, dtype=bool)
    avg_sim = cos_sim_matrix[mask_diag].mean()
    print(f"  {n_layers} layer(s): avg pairwise cosine similarity = {avg_sim:.4f}")

print(f"\nAs layers increase, representations become more similar (over-smoothing).")
print(f"This is why practical GNNs use 2-3 layers, not 10+.")
