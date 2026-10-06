# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 6.4: Link Prediction with GNN Encoder-Decoder
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why link prediction matters (knowledge graphs, social networks, recs)
#   - Encoder-decoder architecture: GNN encoder + dot-product decoder
#   - Positive vs negative edge sampling for training
#   - Splitting EDGES into train/val/test so the score measures unseen links
#   - AUC metric for ranking quality evaluation
#   - Train a link predictor on the Cora citation network
#   - Track training with kailash-ml ExperimentTracker
#
# PREREQUISITES: M5/ex_6.1 (GCN layer implementation).
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_6 import (
    OUTPUT_DIR,
    device,
    load_graph_data,
    normalise_adjacency,
    plot_training_curves,
    register_model,
    setup_engines,
)
from kailash_ml.types import MetricSpec

import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why Link Prediction Matters
# ════════════════════════════════════════════════════════════════════════
#
# Node classification asks "what is this node?" Link prediction asks
# "should these two nodes be connected?" This is the foundation of:
#
# 1. KNOWLEDGE GRAPHS: A medical knowledge graph has nodes for drugs,
#    diseases, proteins, and genes. Edges represent known interactions
#    (drug X treats disease Y). Link prediction discovers MISSING edges
#    — potential new drug-disease interactions that haven't been tested.
#
# 2. SOCIAL NETWORKS: "People you may know" = predict missing friendship
#    edges based on mutual connections and profile features.
#
# 3. RECOMMENDATION: "Papers you should cite" = predict missing citation
#    edges based on content similarity and citation patterns.
#
# The approach: ENCODER-DECODER architecture
#   - ENCODER: A GNN (like our GCN) that produces node embeddings z_i
#   - DECODER: dot-product similarity between embeddings
#     score(i, j) = sigmoid( z_i^T z_j )
#
# If two nodes have similar embeddings, the decoder predicts an edge
# between them. Training uses known edges as positives and random
# non-edges as negatives — binary classification on edge existence.
print("=" * 70)
print("  PHASE 1 — THEORY: Link Prediction on Graphs")
print("=" * 70)
print(
    """
  LINK PREDICTION: "Should these two nodes be connected?"

  THREE MAJOR APPLICATIONS:
  1. Knowledge Graphs: discover new drug-disease interactions
  2. Social Networks: "people you may know" suggestions
  3. Citation Networks: "papers you should cite" recommendations

  ENCODER-DECODER APPROACH:
  - ENCODER (GNN): learns node embeddings z_i from features + structure
  - DECODER (dot product): score(i,j) = sigmoid(z_i^T z_j)
  - High similarity in embedding space -> predict edge exists

  TRAINING DATA:
  - Split the real edges: 85% train / 5% validation / 10% test
  - The encoder passes messages over TRAINING edges only — a held-out
    edge must not be visible in the graph it is asked to predict
  - Positive samples: training edges (label = 1)
  - Negative samples: random non-edges, re-drawn every epoch (label = 0)
  - Loss: binary cross-entropy on edge predictions
  - Metric: AUC on HELD-OUT edges — how well do we rank unseen real
    edges above non-edges?
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: GNN Encoder + Dot-Product Decoder
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: Link Prediction Model")
print("=" * 70)

# Load graph data and set up engines
graph_data = load_graph_data()
conn, tracker, exp_name, registry, has_registry = setup_engines()

X = graph_data["X"]
A = graph_data["A"]
A_np = graph_data["A_np"]
y_np = graph_data["y_np"]
edge_index_np = graph_data["edge_index_np"]
N = graph_data["N"]
F_dim = graph_data["F_dim"]
n_classes = graph_data["n_classes"]
dataset_name = graph_data["dataset_name"]

HIDDEN_DIM = 16 if dataset_name == "Karate Club" else 64
LINK_EPOCHS = 80


# Reuse GCN layer for the encoder
class GCNLayer(nn.Module):
    """GCN layer: H' = A_norm @ (H @ W)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.W(h)


class LinkPredictor(nn.Module):
    """Encoder-decoder for link prediction.

    Encoder: MLP -> two GCN layers that produce node embeddings.
    Decoder: dot product between node pairs -> edge probability.

    The encoder first projects features through an MLP (to handle
    high-dimensional bag-of-words features), then applies two GCN
    layers to incorporate graph structure into the embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        # TODO: a two-layer MLP (in_dim -> hidden -> hidden, ReLU between)
        #       followed by two GCNLayers (hidden -> hidden)
        # Hint: torch.nn.Sequential chains modules
        self.encoder = ____
        self.gcn1 = ____
        self.gcn2 = ____

    def encode(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """Produce node embeddings: MLP -> GCN -> GCN."""
        # TODO: MLP, then GCN1 + ReLU, then GCN2 (no activation: the
        #       decoder needs signed embeddings)
        h = ____
        h = ____
        h = ____
        return h

    def decode(
        self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        """Dot-product decoder: score(i,j) = z_i^T z_j."""
        # TODO: one score per (src, dst) pair — the dot product of their
        #       embeddings, without building the full N x N matrix
        return ____

    def forward(
        self,
        h: torch.Tensor,
        a_norm: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(h, a_norm)
        return self.decode(z, src, dst)


link_model = LinkPredictor(F_dim, HIDDEN_DIM).to(device)
n_params = sum(p.numel() for p in link_model.parameters())

print(f"\n  Link Prediction architecture:")
print(
    f"    Encoder MLP: Linear({F_dim}->{HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM}->{HIDDEN_DIM})"
)
print(f"    GCN Layer 1: GCNLayer({HIDDEN_DIM} -> {HIDDEN_DIM})")
print(f"    GCN Layer 2: GCNLayer({HIDDEN_DIM} -> {HIDDEN_DIM})")
print(f"    Decoder: dot_product(z_i, z_j) -> edge score")
print(f"    Total parameters: {n_params:,}")

# ── Build Checkpoint ────────────────────────────────────────────────
assert isinstance(link_model, nn.Module), "LinkPredictor should be an nn.Module"
assert n_params > 0, "LinkPredictor should have learnable parameters"
print("\n--- Build checkpoint passed --- LinkPredictor architecture created\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Link Prediction on Cora
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"  PHASE 3 — TRAIN: Link Prediction on {dataset_name}")
print("=" * 70)

# ── Split the EDGES before training ─────────────────────────────────
# The question is "can we predict links we have NOT seen?", so some real
# edges are hidden from training AND from the graph the encoder passes
# messages over. Scoring on training edges that also sit in A_norm would
# measure memorisation plus leakage, not link prediction.
und_src, und_dst = np.where(np.triu(A_np, k=1) > 0)  # each undirected edge once
rng_link = np.random.default_rng(42)
perm = rng_link.permutation(len(und_src))
n_val_edges = int(0.05 * len(perm))
n_test_edges = int(0.10 * len(perm))
val_idx = perm[:n_val_edges]
test_idx = perm[n_val_edges : n_val_edges + n_test_edges]
train_idx = perm[n_val_edges + n_test_edges :]


def _edge_tensors(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(und_src[idx]).to(device),
        torch.from_numpy(und_dst[idx]).to(device),
    )


pos_src, pos_dst = _edge_tensors(train_idx)  # training positives
val_src, val_dst = _edge_tensors(val_idx)
test_src, test_dst = _edge_tensors(test_idx)
n_pos = len(pos_src)

# Message passing uses the TRAINING edges only
A_train = torch.zeros(N, N, device=device)
A_train[pos_src, pos_dst] = 1.0
A_train[pos_dst, pos_src] = 1.0
# TODO: the propagation matrix the encoder may use — built from the
#       TRAINING edges only (shared.mlfp05.ex_6 has the normaliser)
A_norm_train = ____
A_train_np = A_train.cpu().numpy()


def sample_non_edges(
    n: int, adjacency: np.ndarray, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw n random node pairs (s != d) that are NOT edges of `adjacency`."""
    src: list[int] = []
    dst: list[int] = []
    while len(src) < n:
        s = rng.integers(0, N, size=2 * n)
        d = rng.integers(0, N, size=2 * n)
        # TODO: keep pairs that are not self-loops and not edges
        keep = ____
        src.extend(s[keep].tolist())
        dst.extend(d[keep].tolist())
    return (
        torch.tensor(src[:n], dtype=torch.long, device=device),
        torch.tensor(dst[:n], dtype=torch.long, device=device),
    )


# Fixed evaluation negatives: pairs that are not edges ANYWHERE in the graph
val_neg_src, val_neg_dst = sample_non_edges(len(val_src), A_np, rng_link)
test_neg_src, test_neg_dst = sample_non_edges(len(test_src), A_np, rng_link)


def auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """ROC AUC: P(a random real edge outscores a random non-edge); ties = 0.5."""
    # TODO: compare EVERY positive score with EVERY negative score
    # Hint: broadcasting a column against a row gives all pairs at once;
    #       wins count 1, ties count 0.5, then average
    ____


print(f"  Undirected edges: {len(perm):,}")
print(f"    train: {n_pos:,}  (positives + message passing)")
print(f"    val:   {len(val_src):,}  (model selection)")
print(f"    test:  {len(test_src):,}  (reported once, at the end)")
print(f"  Negatives: {n_pos:,} fresh non-edges of the training graph per epoch;")
print(f"    fixed non-edges of the full graph for val ({len(val_neg_src)}) and test")

# Train
link_opt = torch.optim.Adam(link_model.parameters(), lr=1e-2, weight_decay=1e-4)
link_losses: list[float] = []
link_aucs: list[float] = []  # VALIDATION AUC per epoch
best = {"auc": -1.0, "epoch": 0, "state": {}}


async def _train_link_predictor_async() -> float:
    """Train under a tracker.track(...) context; return the held-out test AUC."""
    async with tracker.track(experiment=exp_name, run_name="link_prediction") as run:
        await run.log_params(
            {
                "task": "link_prediction",
                "hidden_dim": str(HIDDEN_DIM),
                "epochs": str(LINK_EPOCHS),
                "n_train_edges": str(n_pos),
                "n_val_edges": str(len(val_src)),
                "n_test_edges": str(len(test_src)),
            }
        )

        for epoch in range(LINK_EPOCHS):
            link_model.train()
            link_opt.zero_grad()

            # Fresh negatives every epoch: non-edges of the TRAINING graph
            neg_src, neg_dst = sample_non_edges(n_pos, A_train_np, rng_link)
            # TODO: score the training positives and this epoch's negatives
            #       (message passing over the TRAINING graph only)
            pos_scores = ____
            neg_scores = ____

            # Binary cross-entropy loss
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat(
                [
                    torch.ones(n_pos, device=device),
                    torch.zeros(len(neg_src), device=device),
                ]
            )
            # TODO: BCE on raw scores (the decoder returns logits)
            loss = ____
            loss.backward()
            link_opt.step()
            link_losses.append(loss.item())

            # Validation AUC: held-out edges vs held-out non-edges
            link_model.eval()
            with torch.no_grad():
                z = link_model.encode(X, A_norm_train)
                # TODO: AUC of the validation edges vs validation non-edges
                val_auc = ____
            link_aucs.append(val_auc)
            if val_auc > best["auc"]:
                best.update(
                    auc=val_auc,
                    epoch=epoch,
                    state=copy.deepcopy(link_model.state_dict()),
                )

            await run.log_metrics(
                {"link_loss": loss.item(), "val_auc": val_auc},
                step=epoch + 1,
            )

            if (epoch + 1) % 20 == 0:
                print(
                    f"  [LinkPred] epoch {epoch+1:3d}  "
                    f"loss={loss.item():.4f}  val_auc={val_auc:.3f}"
                )

        # Restore the best-validation weights; score the test edges ONCE
        link_model.load_state_dict(best["state"])
        link_model.eval()
        with torch.no_grad():
            z = link_model.encode(X, A_norm_train)
            test_auc = auc(
                link_model.decode(z, test_src, test_dst),
                link_model.decode(z, test_neg_src, test_neg_dst),
            )
        await run.log_metrics(
            {
                "best_val_auc": best["auc"],
                "best_val_epoch": float(best["epoch"] + 1),
                "test_auc": test_auc,
            }
        )
    return test_auc


test_auc = asyncio.run(_train_link_predictor_async())

# ── Train Checkpoint ────────────────────────────────────────────────
assert len(link_losses) == LINK_EPOCHS, "Link prediction should train for all epochs"
assert link_losses[-1] < link_losses[0], "Link prediction loss should decrease"
best_val_auc = best["auc"]
assert (
    best_val_auc > 0.55
), f"Validation AUC {best_val_auc:.3f} should exceed random (0.5)"
print(f"\n  Link Prediction Results:")
print(f"    Final training loss:  {link_losses[-1]:.4f}")
print(f"    Best validation AUC:  {best_val_auc:.4f} (epoch {best['epoch'] + 1})")
print(f"    TEST AUC (unseen):    {test_auc:.4f}")
print(f"    Random baseline:      0.5000")
# INTERPRETATION: The test AUC is the probability that a citation the
# model has never seen — not as a training label and not as a message-
# passing edge — scores above a random non-citation. That is the honest
# link-prediction number. Scoring the TRAINING edges instead (with them
# also inside A_norm) gives a much rosier figure that only measures how
# well the model memorised edges it could already see.
print("\n--- Train checkpoint passed --- link prediction trained\n")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Prescription Pad before Visualise
# ══════════════════════════════════════════════════════════════════
# Replays the REAL training objective — BCE over training edges vs fresh
# non-edges, message passing over training edges only — in batches of
# training edges. No weights are updated.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _link_loss(m, batch):
    src, dst = batch
    neg_s, neg_d = sample_non_edges(len(src), A_train_np, rng_link)
    z = m.encode(X, A_norm_train)
    scores = torch.cat([m.decode(z, src, dst), m.decode(z, neg_s, neg_d)])
    labels = torch.cat(
        [torch.ones(len(src), device=device), torch.zeros(len(neg_s), device=device)]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


edge_batches = [
    (pos_src[i : i + 1024], pos_dst[i : i + 1024]) for i in range(0, n_pos, 1024)
]
diag, findings = run_diagnostic_checkpoint(
    link_model,
    edge_batches,
    _link_loss,
    title="Link Prediction — GCN encoder",
    n_batches=4,
    train_losses=link_losses,
    show=False,
)
print_prescription_pad(findings, "Link Prediction — GCN encoder")
# HOW TO READ IT (your readings depend on your run):
#  GRADIENT FLOW — the dot-product decoder can push embedding norms up
#     to sharpen scores; exploding readings in the GCN layers suggest
#     lowering the learning rate or adding weight decay.
#  DEAD NEURONS — the encoder MLP's ReLU is an nn.ReLU module, so this
#     reading is live: a large inactive fraction means many hidden units
#     never fire for these papers.
#  LOSS TREND — training loss only. Over-fitting shows up as the
#     validation AUC (Phase 4 plot) stalling or falling while this keeps
#     improving — the reason we select the epoch by validation AUC.
# ══════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Edge Scores + Embedding Similarity
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: Link Prediction Analysis")
print("=" * 70)

link_model.eval()
with torch.no_grad():
    # Node embeddings from the TRAINING graph (what the model can see)
    z = link_model.encode(X, A_norm_train).cpu().numpy()

    # Score distributions for HELD-OUT test edges vs test non-edges
    pos_final_scores = link_model(X, A_norm_train, test_src, test_dst).cpu().numpy()
    neg_final_scores = link_model(
        X, A_norm_train, test_neg_src, test_neg_dst
    ).cpu().numpy()

# Plot 1: Score distributions — held-out real edges vs non-edges
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(
    pos_final_scores[:2000],
    bins=60,
    alpha=0.7,
    color="green",
    edgecolor="white",
    label="Held-out real edges",
    density=True,
)
axes[0].hist(
    neg_final_scores[:2000],
    bins=60,
    alpha=0.7,
    color="red",
    edgecolor="white",
    label="Non-edges",
    density=True,
)
axes[0].set_xlabel("Edge Score (before sigmoid)", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title(
    "Test Score Distribution: Held-out Edges vs Non-Edges",
    fontsize=13,
    fontweight="bold",
)
axes[0].legend(fontsize=10)

# Plot 2: training loss and VALIDATION AUC over training
epochs_range = list(range(1, LINK_EPOCHS + 1))
ax_loss = axes[1]
ax_loss.plot(epochs_range, link_losses, color="steelblue", label="Loss")
ax_loss.set_xlabel("Epoch", fontsize=11)
ax_loss.set_ylabel("BCE Loss", fontsize=11, color="steelblue")
ax_loss.tick_params(axis="y", labelcolor="steelblue")

ax_auc = ax_loss.twinx()
ax_auc.plot(epochs_range, link_aucs, color="coral", label="Validation AUC")
ax_auc.set_ylabel("Validation AUC", fontsize=11, color="coral")
ax_auc.tick_params(axis="y", labelcolor="coral")
axes[1].set_title("Link Prediction Training Progress", fontsize=13, fontweight="bold")

fig.tight_layout()
filepath = OUTPUT_DIR / "link_prediction_analysis.png"
fig.savefig(filepath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {filepath}")

# Plot 3: Heatmap of similarity in embedding space for a subgraph
n_sub = min(50, N)
rng = np.random.default_rng(42)
sub_idx = rng.choice(N, n_sub, replace=False)
sub_idx = np.sort(sub_idx)
z_sub = z[sub_idx]
similarity = z_sub @ z_sub.T  # dot-product similarity
sub_A = A_np[np.ix_(sub_idx, sub_idx)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(sub_A, cmap="Blues", aspect="auto")
axes[0].set_title(f"True Adjacency ({n_sub} nodes)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Node")
axes[0].set_ylabel("Node")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

im2 = axes[1].imshow(similarity, cmap="RdBu_r", aspect="auto")
axes[1].set_title(f"Predicted Similarity (z_i^T z_j)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Node")
axes[1].set_ylabel("Node")
plt.colorbar(im2, ax=axes[1], shrink=0.8)

fig.suptitle(
    f"Link Prediction: True Edges vs Learned Similarities — {dataset_name}",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
filepath = OUTPUT_DIR / "link_prediction_similarity.png"
fig.savefig(filepath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {filepath}")

# Training curves via ModelVisualizer
plot_training_curves(
    metrics_dict={
        "Link pred loss": link_losses,
        "Link pred validation AUC": link_aucs,
    },
    title="Link Prediction Training",
    y_label="Value",
    filename="link_prediction_curves.html",
)

# Score statistics
pos_mean = pos_final_scores.mean()
neg_mean = neg_final_scores.mean()
print(f"\n  Score analysis:")
print(f"    Held-out real edges — mean score: {pos_mean:+.4f}")
print(f"    Non-edges           — mean score: {neg_mean:+.4f}")
print(f"    Separation:  {pos_mean - neg_mean:.4f}")
print(f"    -> Separation on UNSEEN edges is what a link predictor is for")

# ── Visualise Checkpoint ────────────────────────────────────────────
assert z.shape == (N, HIDDEN_DIM), f"Embedding shape should be ({N}, {HIDDEN_DIM})"
assert pos_mean > neg_mean, "Held-out real edges should score higher on average"
print("\n--- Visualise checkpoint passed --- link prediction analysis plotted\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: Knowledge Graph Completion for a Hospital
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 5 — APPLY: Knowledge Graph Completion for a Hospital")
print("=" * 70)
print(
    """
  SCENARIO (illustrative): You're building a drug-disease interaction
  predictor for a Singapore public hospital using a medical knowledge graph.

  THE KNOWLEDGE GRAPH:
  - Drug nodes: ~5K approved drugs (features: molecular weight, targets, ATC code)
  - Disease nodes: ~10K conditions (features: ICD-10 code, organ system, prevalence)
  - Protein nodes: ~20K proteins (features: function, pathway, expression level)
  - Edges: known interactions (drug-treats-disease, drug-binds-protein,
    protein-associated-with-disease)

  LINK PREDICTION TASK: Discover new drug-disease edges
  - Known: drug X treats disease Y (from clinical trials)
  - Unknown: does drug X also treat disease Z? (drug repurposing)
  - Validation: withhold known edges (and keep them out of the graph the
    encoder sees), then check the model ranks them highly

  HOW IT WORKS:
  1. Encode all nodes with GNN: each drug gets an embedding that
     captures its molecular features AND its known interactions
  2. Score all (drug, disease) pairs with dot product
  3. Rank by score — top-k are candidate interactions for lab testing
  4. Validate: do withheld known interactions appear in top-k?
"""
)

# Demonstrate with Cora: rank EVERY candidate pair, as the lab would, and
# check how many of the top-k are held-out real citations.
TOP_K = 100
print("  Demonstration: candidate ranking")
print(f"  (Score every unlinked pair; how many of the top {TOP_K} are real")
print("   held-out test citations the model never saw?)\n")

with torch.no_grad():
    z_all = link_model.encode(X, A_norm_train)
    pair_scores = z_all @ z_all.T  # (N, N) dot-product scores

# Candidates: pairs i < j that are neither training nor validation edges
candidates = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
candidates &= A_train == 0
candidates[val_src, val_dst] = False
is_test_edge = torch.zeros(N, N, dtype=torch.bool, device=device)
is_test_edge[test_src, test_dst] = True

cand_scores = pair_scores[candidates]
cand_is_test = is_test_edge[candidates]
top = torch.topk(cand_scores, TOP_K).indices
# TODO: how many of the top-k candidates are held-out test edges?
hits = ____
precision_at_k = hits / TOP_K
base_rate = cand_is_test.float().mean().item()
lift = precision_at_k / base_rate

print(f"    Candidate pairs scored:          {int(candidates.sum()):,}")
print(f"    Held-out real edges among them:  {int(cand_is_test.sum())}")
print(f"    Real edges in the top {TOP_K}:       {hits}  (precision@{TOP_K} = {precision_at_k:.2f})")
print(f"    Expected by random picking:      {base_rate * TOP_K:.3f}")
print(f"    Lift over random:                {lift:,.0f}x")
print(
    "    -> In a hospital setting the top-k list is what the pharmacology\n"
    "       team reviews; precision@k says how much of their time is well spent."
)

print(
    """
  CLINICAL DEPLOYMENT:
  1. Build the KG from public drug-target, disease-gene and protein-
     interaction databases plus the hospital's own records
  2. Split known drug-disease edges into train / val / test and train the
     link predictor with message passing over the training edges only
  3. Score all (drug, disease) pairs without known interactions
  4. Top-k candidates reviewed by pharmacology team for literature evidence
  5. Promising candidates enter pre-clinical or retrospective cohort studies
  6. Track predictions with ExperimentTracker — validate against new trial results
  7. Register model version in ModelRegistry with AUC and recall@k metrics
"""
)

# Register the link predictor
if has_registry:
    version = register_model(
        registry=registry,
        name="m5_gnn_link_predictor",
        model=link_model,
        metrics=[
            MetricSpec(name="best_val_auc", value=best_val_auc),
            MetricSpec(name="test_auc", value=test_auc),
            MetricSpec(name="final_link_loss", value=link_losses[-1]),
            MetricSpec(name=f"precision_at_{TOP_K}", value=precision_at_k),
        ],
    )
    print(
        f"  Registered link_predictor: version={version.version}, "
        f"test_auc={test_auc:.4f}"
    )

# ── Apply Checkpoint ────────────────────────────────────────────────
assert test_auc > 0.5, "Held-out test AUC should beat random"
assert 0 <= hits <= TOP_K, "Hits must be a count within the top-k list"
print("\n--- Apply checkpoint passed --- knowledge graph completion demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Link Prediction")
print("=" * 70)
print(
    f"""
  LINK PREDICTION WITH GNN ENCODER-DECODER:
  [x] Encoder: GCN layers produce node embeddings from features + structure
  [x] Decoder: dot-product similarity — score(i,j) = z_i^T z_j
  [x] Training: positive edges (real) vs negative edges (sampled non-edges)
  [x] Edge split: train / val / test edges; the encoder sees train edges only
  [x] Held-out test AUC: {test_auc:.1%} — ranks UNSEEN real edges above non-edges
  [x] Candidate ranking: {hits}/{TOP_K} top-ranked pairs were held-out real
      edges ({lift:,.0f}x random)
  [x] Visualised score distributions and similarity heatmaps

  LINK PREDICTION vs NODE CLASSIFICATION:
  - Node classification: "what IS this node?" (label prediction)
  - Link prediction: "should these nodes be CONNECTED?" (edge prediction)
  - Same GNN encoder; different decoder (classifier vs dot product)
  - Link prediction is the foundation of recommendation systems

  APPLICATIONS:
  - Knowledge graphs: discover new drug-disease interactions
  - Social networks: "people you may know"
  - Citation networks: "papers you should cite"
  - E-commerce: "products frequently bought together"

  Next: Exercise 6.5 — Architecture Comparison: systematic side-by-side
  evaluation of GCN vs GAT vs GraphSAGE on the same dataset...
"""
)

# Clean up
asyncio.run(conn.close())
