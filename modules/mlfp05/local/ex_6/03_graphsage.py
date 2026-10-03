# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 6.3: GraphSAGE (Sample and Aggregate)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why GCN doesn't scale to large graphs (full adjacency in memory)
#   - Inductive learning: why GraphSAGE CAN embed unseen nodes (and why
#     this Cora exercise does not yet prove it)
#   - Neighbour sampling strategy: fixed-size random subsets per node
#   - Separate self/neighbour projections for richer representations
#   - Train a scalable node classifier on the Cora citation network
#   - Track training with kailash-ml ExperimentTracker
#
# PREREQUISITES: M5/ex_6.1 (GCN), M5/ex_6.2 (GAT).
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_6 import (
    OUTPUT_DIR,
    device,
    load_graph_data,
    plot_graph_with_embeddings,
    plot_node_embeddings,
    plot_training_curves,
    register_model,
    setup_engines,
    train_node_classifier,
)
from kailash_ml.types import MetricSpec

import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why GCN Doesn't Scale
# ════════════════════════════════════════════════════════════════════════
#
# GCN and GAT both require the FULL adjacency matrix during training.
# For Cora (2,708 nodes), that's a 2708 x 2708 matrix — no problem.
# But real-world graphs are much bigger:
#
#   - A city-scale food delivery network: ~500K users x ~50K restaurants
#   - A global social network: billions of user nodes
#   - A web-scale knowledge graph: hundreds of billions of facts (edges)
#
# A 500K x 500K dense adjacency matrix needs ~1 TB of memory. Even
# sparse representations strain GPU memory when you need multi-hop
# neighbourhood aggregation.
#
# GraphSAGE (SAmple and aggrEGATE) solves this with three key ideas:
#
# 1. SAMPLE: Instead of aggregating ALL neighbours, randomly sample
#    a fixed number K (e.g., 10) per node. This bounds memory usage
#    regardless of graph size.
#
# 2. AGGREGATE: Use a learnable aggregator (mean, LSTM, or pooling)
#    over the sampled neighbours. The aggregator is a FUNCTION, not
#    a lookup table — it works on any set of neighbours.
#
# 3. INDUCTIVE: Because GraphSAGE learns an aggregation FUNCTION
#    rather than per-node embeddings, it can generalise to nodes it
#    has never seen during training. A new restaurant added to the
#    platform can be classified immediately using its neighbours.
#
# The formula:
#   h'_i = sigma( W_self @ h_i + W_neigh @ MEAN(h_j for j in Sample(N(i))) )
#
# Notice: SEPARATE weight matrices for self (W_self) and neighbours
# (W_neigh). This lets the model learn different transformations for
# "what I know about myself" vs "what my neighbours tell me".
print("=" * 70)
print("  PHASE 1 — THEORY: GraphSAGE — Sample and Aggregate")
print("=" * 70)
print(
    """
  WHY GCN DOESN'T SCALE:
  - GCN needs full adjacency matrix in memory: O(N^2) for dense, O(E) for sparse
  - Cora (2.7K nodes): fine. Real graphs (500K+ nodes): memory explosion
  - Multi-hop aggregation expands exponentially: 2 hops of degree-50 = 2,500 nodes

  GRAPHSAGE'S THREE KEY IDEAS:

  1. SAMPLE: randomly pick K neighbours per node (not ALL)
     -> Fixed memory budget regardless of graph size
     -> Like dropout: different samples each epoch = regularisation

  2. AGGREGATE: learnable function over sampled neighbours
     -> Mean, LSTM, or pooling aggregator
     -> A FUNCTION, not a lookup — works on any neighbour set

  3. INDUCTIVE: learns HOW to aggregate, not WHAT to embed
     -> New nodes at inference time? Sample their neighbours and run
        the learned aggregator
     -> GCN as trained in ex_6.1 is used TRANSDUCTIVELY: one fixed,
        full-graph normalised adjacency, with the test nodes already in
        the graph during training. (GAT also learns a function of node
        features and was shown to be inductive in its own paper.)

  Formula: h'_i = sigma( W_self @ h_i + W_neigh @ MEAN(sample(N(i))) )
  Separate W_self and W_neigh = "what I know" vs "what neighbours say"
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: GraphSAGE Layer Implementation
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: GraphSAGE Layer + Model")
print("=" * 70)

# Load graph data and set up engines
graph_data = load_graph_data()
conn, tracker, exp_name, registry, has_registry = setup_engines()

X = graph_data["X"]
A = graph_data["A"]
y = graph_data["y"]
y_np = graph_data["y_np"]
A_np = graph_data["A_np"]
N = graph_data["N"]
F_dim = graph_data["F_dim"]
n_classes = graph_data["n_classes"]
dataset_name = graph_data["dataset_name"]

HIDDEN_DIM = 16 if dataset_name == "Karate Club" else 64
EPOCHS = 100
SAMPLE_K = 10  # Max neighbours to sample per node


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer with mean aggregator and neighbour sampling.

    During training: randomly sample at most K neighbours per node
    (regularisation + scalability). At eval: use all neighbours for
    deterministic output (like dropout).

    Separate W_self and W_neigh allow the model to learn different
    transformations for a node's own features vs its neighbours'.
    """

    def __init__(self, in_dim: int, out_dim: int, sample_k: int = 10):
        super().__init__()
        # TODO: two SEPARATE bias-free projections in_dim -> out_dim — one
        #       for the node's own features, one for its neighbours' mean
        self.W_self = ____
        self.W_neigh = ____
        self.sample_k = sample_k

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = h.size(0)

        # Neighbour sampling: for each node, keep at most sample_k neighbours
        # by zeroing out excess connections. At eval time, use all neighbours
        # for deterministic output (like dropout).
        if self.training and self.sample_k < n:
            sample_mask = torch.zeros_like(adj)
            for i in range(n):
                neigh_idx = torch.where(adj[i] > 0)[0]
                # TODO: keep every neighbour if there are at most sample_k;
                #       otherwise keep a RANDOM subset of sample_k of them
                #       (set those entries of sample_mask[i] to 1)
                # Hint: torch.randperm gives a random ordering of indices
                ____
            adj_sampled = sample_mask
        else:
            adj_sampled = adj

        # Mean aggregation: average the features of sampled neighbours
        # TODO: MEAN of the sampled neighbours' features, shape (N, in_dim)
        # Hint: a matrix multiply sums neighbours; divide by the number kept
        #       (clamp at 1 so isolated nodes do not divide by zero)
        deg_sampled = ____
        h_neigh = ____

        # TODO: combine self and neighbour representations (Phase 1 formula,
        #       before the activation)
        h_self = ____
        h_agg = ____
        return h_self + h_agg  # additive combination


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE for node classification."""

    def __init__(
        self, in_dim: int, hidden_dim: int, n_classes: int, sample_k: int = 10
    ):
        super().__init__()
        # TODO: two GraphSAGELayers (pass sample_k through to both)
        self.l1 = ____
        self.l2 = ____

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # TODO: layer 1 + ReLU, dropout(p=0.5) while training, layer 2 raw
        h = ____
        h = ____
        return ____

    def embed(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Return the hidden-layer embedding (before classification head)."""
        # TODO: the activated output of the first layer (no dropout)
        return ____


sage = GraphSAGE(
    in_dim=F_dim, hidden_dim=HIDDEN_DIM, n_classes=n_classes, sample_k=SAMPLE_K
)
n_params = sum(p.numel() for p in sage.parameters())
print(f"\n  GraphSAGE architecture:")
print(f"    Layer 1: GraphSAGELayer({F_dim} -> {HIDDEN_DIM}, sample_k={SAMPLE_K})")
print(f"    Layer 2: GraphSAGELayer({HIDDEN_DIM} -> {n_classes}, sample_k={SAMPLE_K})")
print(f"    Total parameters: {n_params:,}")
print(f"\n  How sampling works:")
print(f"    Training: each node samples up to {SAMPLE_K} neighbours per epoch")
print(f"    Eval: use all neighbours (deterministic, like dropout)")
print(f"    Effect: regularisation + bounded memory per mini-batch")

# Show how sampling affects neighbourhood size
degrees = A_np.sum(axis=1)
nodes_needing_sample = (degrees > SAMPLE_K).sum()
print(f"\n  Sampling impact on {dataset_name}:")
print(f"    Avg degree: {degrees.mean():.1f}")
print(f"    Max degree: {int(degrees.max())}")
print(f"    Nodes with degree > {SAMPLE_K}: {nodes_needing_sample} / {N}")
print(f"    -> {nodes_needing_sample} nodes will have sampled neighbourhoods")

# ── Build Checkpoint ────────────────────────────────────────────────
assert isinstance(sage, nn.Module), "GraphSAGE should be an nn.Module"
assert n_params > 0, "GraphSAGE should have learnable parameters"
print("\n--- Build checkpoint passed --- GraphSAGE architecture created\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Node Classification on Cora
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"  PHASE 3 — TRAIN: GraphSAGE on {dataset_name}")
print("=" * 70)

sage_losses, sage_val, sage_test = train_node_classifier(
    model=sage,
    name="GraphSAGE",
    forward_arg=A,
    graph_data=graph_data,
    tracker=tracker,
    exp_name=exp_name,
    epochs=EPOCHS,
)

# ── Train Checkpoint ────────────────────────────────────────────────
assert len(sage_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses for GraphSAGE"
assert sage_losses[-1] < sage_losses[0], "GraphSAGE loss should decrease"
# Model selection by VALIDATION accuracy; report test accuracy at that
# epoch (the harness has already restored that epoch's weights).
# TODO: epoch by validation accuracy, then that epoch's test accuracy
best_epoch = ____
best_val = sage_val[best_epoch]
best_test = ____
print(f"\n  GraphSAGE Results:")
print(f"    Best validation accuracy: {best_val:.4f} (epoch {best_epoch + 1})")
print(f"    Test accuracy, that epoch: {best_test:.4f}")
print(f"    Final loss:               {sage_losses[-1]:.4f}")
# INTERPRETATION: GraphSAGE is designed to be INDUCTIVE — it learns an
# aggregation FUNCTION that can be applied to nodes it never saw. Note
# that this run is still transductive: every Cora node, including the
# test nodes, sits in the graph during training. During training, it
# randomly samples K neighbours per node (like dropout for graphs),
# which provides regularisation and makes it scalable to large graphs.
# The separate W_self and W_neigh projections let the model learn
# different transformations for a node's own features versus its
# neighbours' features.
print("\n--- Train checkpoint passed --- GraphSAGE trained successfully\n")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Prescription Pad before Visualise
# ══════════════════════════════════════════════════════════════════
# run_diagnostic_checkpoint instruments the trained model, replays a few
# forward/backward passes of the REAL training objective (cross-entropy
# on the labelled training nodes; no weights are updated) and replays
# the per-epoch training losses. The whole graph is one "batch", so the
# loader is the same full-graph tuple repeated.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _node_loss(m, batch):
    feats, graph, labels, mask = batch
    return F.cross_entropy(m(feats, graph)[mask], labels[mask])


diag, findings = run_diagnostic_checkpoint(
    sage,
    [(X, A, y, graph_data["train_mask"])] * 4,
    _node_loss,
    title="GraphSAGE — Sample and Aggregate",
    n_batches=4,
    train_losses=sage_losses,
    show=False,
)
print_prescription_pad(findings, "GraphSAGE — Sample and Aggregate")
# HOW TO READ IT (your readings depend on your run):
#  GRADIENT FLOW — a 2-layer GNN rarely vanishes. Exploding readings
#     usually mean the propagation matrix is not normalised (a raw
#     adjacency multiplies feature scale by node degree) or the learning
#     rate is too high.
#  DEAD NEURONS — this model applies its activation functionally
#     (F.relu / F.elu), so there is no activation LAYER for the
#     instrument to hook; an UNKNOWN reading here is expected, not a
#     fault. Use nn.ReLU modules if you want this reading.
#  LOSS TREND — this sees only the training loss. Over-fitting shows up
#     in the gap between the validation and training curves, not here.
# ══════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Embeddings + Sampling Effect
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: GraphSAGE Embeddings + Sampling Analysis")
print("=" * 70)

sage.eval()
with torch.no_grad():
    sage_emb = sage.embed(X, A).cpu().numpy()

# Plot 1: 2-D PCA of node embeddings
coords = plot_node_embeddings(
    embeddings=sage_emb,
    labels=y_np,
    n_classes=n_classes,
    title=f"GraphSAGE Node Embeddings — {dataset_name}",
    filename="sage_node_embeddings.png",
)

# Plot 2: Graph structure on embedding space
plot_graph_with_embeddings(
    A_np=A_np,
    embeddings_2d=coords,
    labels=y_np,
    n_classes=n_classes,
    title=f"GraphSAGE — Graph Structure in Embedding Space ({dataset_name})",
    filename="sage_graph_embeddings.png",
)

# Plot 3: Training curves
plot_training_curves(
    metrics_dict={"GraphSAGE train loss": sage_losses},
    title="GraphSAGE Training Loss",
    y_label="Cross-Entropy Loss",
    filename="sage_loss_curve.html",
)
plot_training_curves(
    metrics_dict={
        "GraphSAGE val accuracy": sage_val,
        "GraphSAGE test accuracy": sage_test,
    },
    title="GraphSAGE Accuracy",
    y_label="Accuracy",
    filename="sage_accuracy_curves.html",
)

# Plot 4: Analyse the effect of sampling K on embedding variance
# Run multiple forward passes in training mode to show stochastic embeddings
print("\n  Sampling stochasticity analysis:")
embedding_variances = []
sage.train()  # Enable sampling
with torch.no_grad():
    embeddings_list = []
    for trial in range(5):
        emb_trial = sage.embed(X, A).cpu().numpy()
        embeddings_list.append(emb_trial)
    embeddings_stack = np.stack(embeddings_list)  # (5, N, hidden)
    # TODO: one number per node — variance across the 5 trials, averaged
    #       over the hidden dimensions
    per_node_var = ____  # (N,)

sage.eval()  # Restore eval mode

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of per-node embedding variance
axes[0].hist(per_node_var, bins=50, color="coral", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Embedding Variance Across Samples", fontsize=11)
axes[0].set_ylabel("Number of Nodes", fontsize=11)
axes[0].set_title(
    "Stochastic Embedding Variance\n(5 forward passes with sampling)", fontsize=12
)

# Right: variance vs degree
axes[1].scatter(degrees, per_node_var, s=8, alpha=0.4, color="steelblue")
axes[1].set_xlabel("Node Degree", fontsize=11)
axes[1].set_ylabel("Embedding Variance", fontsize=11)
axes[1].set_title(
    f"Variance vs Degree (sample_k={SAMPLE_K})\nHigher degree -> more sampling -> more variance",
    fontsize=12,
)

plt.tight_layout()
filepath = OUTPUT_DIR / "sage_sampling_variance.png"
plt.savefig(filepath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {filepath}")

high_var = per_node_var > np.percentile(per_node_var, 90)
low_var = per_node_var < np.percentile(per_node_var, 10)
print(
    f"    High-variance nodes (top 10%): {int(high_var.sum())}, "
    f"mean degree {degrees[high_var].mean():.1f}"
)
print(
    f"    Low-variance nodes (bottom 10%): {int(low_var.sum())}, "
    f"mean degree {degrees[low_var].mean():.1f}"
)
small = degrees <= SAMPLE_K
print(
    f"    Nodes with degree <= {SAMPLE_K}: their layer-1 neighbourhood is never "
    f"subsampled (mean variance {per_node_var[small].mean():.2e} vs "
    f"{per_node_var[~small].mean():.2e} for larger-degree nodes)"
)

# ── Visualise Checkpoint ────────────────────────────────────────────
assert sage_emb.shape == (
    N,
    HIDDEN_DIM,
), f"Embedding shape should be ({N}, {HIDDEN_DIM})"
print("\n--- Visualise checkpoint passed --- GraphSAGE embeddings + variance plotted\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: Recommendation Engine for Food Delivery
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 5 — APPLY: Food Delivery Recommendations")
print("=" * 70)
print(
    """
  SCENARIO (illustrative): You're building a recommendation engine for a
  Singapore food delivery platform.

  THE GRAPH:
  - User nodes: ~500K users with features (location, order frequency, cuisine prefs)
  - Restaurant nodes: ~50K restaurants (cuisine type, price range, rating)
  - Edges: user-restaurant orders (weighted by frequency)
  - Bipartite graph: users only connect to restaurants, not to each other

  WHY GRAPHSAGE IS THE RIGHT CHOICE:
  1. SCALE: 550K nodes = GCN's adjacency matrix would need 302 billion entries
     GraphSAGE samples 10 neighbours per node = bounded memory
  2. INDUCTIVE: new restaurants join daily. A full-graph GCN like ex_6.1's
     has to be re-run (and usually retrained) on the whole updated graph.
     GraphSAGE is trained to embed a new restaurant directly from a
     sample of its first customers' features.
  3. COLD START: a new restaurant with just 3 orders can be embedded —
     GraphSAGE aggregates those 3 users. Whether that embedding is GOOD
     is something you measure on held-out new restaurants, not assume.

  RECOMMENDATION PIPELINE:
  1. Train GraphSAGE on the user-restaurant graph
  2. Each user gets an embedding (captures taste preferences via neighbours)
  3. Each restaurant gets an embedding (captures customer base profile)
  4. Score = dot_product(user_embedding, restaurant_embedding)
  5. Rank restaurants by score for each user -> personalised recommendations
"""
)

# Demonstrate a neighbour-majority (CF-style) baseline vs GraphSAGE
# Using Cora as proxy: predict class membership from neighbourhood
print("  Neighbour-Majority Baseline vs GraphSAGE:")

# Baseline: predict a node's class by majority vote over the labels of its
# TRAINING-set neighbours only. Using every neighbour's label would read
# the true labels of validation/test nodes — label leakage.
train_mask = graph_data["train_mask"]
test_mask = graph_data["test_mask"]
fallback_class = int(torch.bincount(y[train_mask], minlength=n_classes).argmax())
majority_preds = torch.full((N,), fallback_class, dtype=torch.long, device=device)

for i in range(N):
    # TODO: node i's neighbours that are in the TRAINING set (their labels
    #       are the only ones we may look at)
    neighbours = ____
    if len(neighbours) == 0:
        continue  # no labelled neighbour: keep the majority training class
    neighbour_labels = y[neighbours]
    # Majority vote
    # TODO: the most common label among them
    # Hint: torch.bincount counts each class (set minlength=n_classes)
    ____

cf_acc = (majority_preds[test_mask] == y[test_mask]).float().mean().item()

print(f"    Neighbour majority (train labels only): {cf_acc:.4f}")
print(f"    GraphSAGE (learned aggregation):        {best_test:.4f}")
improvement = best_test - cf_acc
print(
    f"    Difference:                             {improvement:+.4f} ({improvement*100:+.1f} pp)"
)

print(
    """
  HOW GRAPHSAGE DIFFERS FROM THE NEIGHBOUR-MAJORITY BASELINE:
  - CF just counts neighbours — GraphSAGE LEARNS what to aggregate
  - CF has no features — GraphSAGE combines structure with node features
  - CF is one-hop — 2-layer GraphSAGE captures 2-hop patterns
  - CF is fixed — GraphSAGE adapts its aggregation during training

  DEPLOYMENT CONSIDERATIONS:
  1. Mini-batch training: sample subgraphs, not full graph (torch_geometric)
  2. Pre-compute embeddings offline; serve recommendations from cache
  3. Retrain weekly with new orders; update embeddings for active users daily
  4. Track recommendation quality with ExperimentTracker (CTR, order rate)
  5. A/B test: GraphSAGE recs vs popularity-based recs vs CF baseline
"""
)

# Register the GraphSAGE model
if has_registry:
    version = register_model(
        registry=registry,
        name=f"m5_graphsage_{dataset_name.lower().replace(' ', '_')}",
        model=sage,
        metrics=[
            MetricSpec(name="best_val_accuracy", value=best_val),
            MetricSpec(name="test_accuracy_at_best_val", value=best_test),
            MetricSpec(name="final_loss", value=sage_losses[-1]),
            MetricSpec(name="cf_baseline_accuracy", value=cf_acc),
            MetricSpec(name="improvement_over_cf", value=improvement),
            MetricSpec(name="sample_k", value=float(SAMPLE_K)),
            MetricSpec(name="hidden_dim", value=float(HIDDEN_DIM)),
            MetricSpec(name="epochs", value=float(EPOCHS)),
        ],
    )
    print(f"  Registered GraphSAGE: version={version.version}, val_acc={best_val:.4f}")

# ── Apply Checkpoint ────────────────────────────────────────────────
assert cf_acc > 0.0, "CF baseline should produce non-zero accuracy"
print("\n--- Apply checkpoint passed --- recommendation scenario demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — GraphSAGE")
print("=" * 70)
print(
    f"""
  GRAPHSAGE (Hamilton, Ying & Leskovec, 2017):
  [x] Neighbour sampling: fixed K neighbours per node bounds memory
  [x] Mean aggregator: MEAN(h_j for j in Sample(N(i)))
  [x] Separate projections: W_self @ h_i + W_neigh @ h_agg
  [x] INDUCTIVE design: a learned aggregation FUNCTION can embed unseen
      nodes — this run trained on the full Cora graph, so it shows the
      mechanism; proving generalisation needs held-out nodes
  [x] Trained on {dataset_name}: {best_val:.1%} val accuracy, {best_test:.1%} test accuracy
  [x] Analysed sampling stochasticity: variance vs node degree
  [x] Compared with a neighbour-majority baseline: {improvement*100:+.1f} percentage points

  THREE-WAY COMPARISON (so far):
  - GCN: fixed weights, full graph, fast, simple
  - GAT: learned attention, full graph, interpretable
  - GraphSAGE: sampling + learned aggregation, SCALABLE, INDUCTIVE

  WHEN TO USE GRAPHSAGE:
  - Graph has 100K+ nodes (GCN/GAT memory-bound)
  - New nodes arrive at inference time (inductive requirement)
  - Mini-batch training needed (can't fit full graph on GPU)
  - Recommendation systems, dynamic social networks, evolving knowledge graphs

  Next: Exercise 6.4 — Link Prediction: predict missing edges with a
  dot-product decoder (foundation of recommendation systems)...
"""
)

# Clean up
asyncio.run(conn.close())
