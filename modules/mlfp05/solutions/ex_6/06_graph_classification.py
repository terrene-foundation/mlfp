# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 6.6 — Graph CLASSIFICATION with GIN (and GCN) on
# TUDataset: whole-graph labels, not node labels
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Distinguish NODE classification (one big graph, label each node —
#     what 01-05 did on Cora) from GRAPH classification (many small
#     graphs, one label per graph — molecules, programs, molecules)
#   - Explain why GIN (Xu et al. 2019) is the maximally expressive
#     message-passing GNN: SUM aggregation + an MLP update matches the
#     Weisfeiler-Lehman test's power
#   - Build a graph classifier: message passing + global_mean_pool +
#     linear head, batched with torch_geometric's DataLoader
#   - Evaluate honestly: accuracy against the majority-class baseline,
#     plus per-class precision/recall for the class that costs money
#   - Apply to pre-synthesis mutagenicity screening at a specialty
#     chemicals firm
#
# PREREQUISITES: M5/ex_6/01_gcn.py (message passing, adjacency
#   normalisation). torch_geometric is used for real here — dataset,
#   GINConv/GCNConv layers, global_mean_pool, and the PyG DataLoader.
# ESTIMATED TIME: ~35 min
#
# DATASET: TUDataset MUTAG — 188 small molecular graphs, 7 node
#   features (atom types), 2 classes (mutagenic or not). If the lab
#   network blocks the TUDataset download, the file falls back to
#   torch_geometric's built-in FakeDataset (no download) and SAYS SO —
#   random labels cap accuracy at chance; the pipeline is still real.
#
# PHASES:
#   1. THEORY  — node vs graph tasks; why GIN; pooling
#   2. BUILD   — dataset + GINClassifier + GCNClassifier
#   3. TRAIN   — both models, best-val selection, ExperimentTracker
#   4. VISUALISE — curves, graph-embedding PCA, sample molecules
#   5. APPLY   — pre-synthesis screening with per-class costs
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

from shared.mlfp05.ex_6 import (
    OUTPUT_DIR,
    device,
    register_model,
    setup_engines,
)
from kailash_ml.types import MetricSpec

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: From Labelling Nodes to Labelling Whole Graphs
# ════════════════════════════════════════════════════════════════════════
# Everything in 01-05 was NODE classification: ONE citation graph (Cora),
# label each paper. The graph is the WORLD; nodes are the instances.
#
# GRAPH CLASSIFICATION inverts the framing: each instance IS a graph.
# A molecule: atoms are nodes, bonds are edges, and the LABEL belongs to
# the whole molecule (mutagenic / not). You get a DATASET of graphs —
# 188 small worlds — and must embed each whole graph into one vector.
#
# Two new ingredients make it work:
#
#   READOUT (pooling): after message passing, each node holds an
#   embedding. global_mean_pool averages the node embeddings WITHIN each
#   graph (the PyG batch vector says which graph each node belongs to),
#   yielding one graph vector per molecule. Sum-pooling counts (size
#   leaks in); mean-pooling compares shape regardless of size.
#
#   GIN (Graph Isomorphism Network, Xu et al. 2019): the paper asked
#   "how powerful CAN a message-passing GNN be?" Answer: at most as
#   powerful as the Weisfeiler-Lehman graph-isomorphism test — and GIN
#   reaches that ceiling. The recipe is two choices:
#     - Aggregate neighbour features by SUM (not mean): mean cannot
#       distinguish {a, b} from {a/2, a/2, b/2, b/2} — the multiset
#       information is destroyed. Sum preserves the multiset.
#     - Update with an MLP (not a single linear layer): the update must
#       be INJECTIVE over multisets, and an MLP can approximate that.
#   GCN (01) uses mean-style normalised aggregation — expressive enough
#   for citations; GIN's sum+MLP matters when fine structural
#   differences (a ring of 5 vs 6 atoms) carry the label.

print("=" * 70)
print("  PHASE 1 — THEORY: graph classification, readout, and GIN")
print("=" * 70)
print(
    """
  NODE classification (01-05): one big graph, label each node.
  GRAPH classification (here):  each instance IS a graph.

  READOUT:  global_mean_pool averages node embeddings within each
            graph (PyG's batch vector tracks graph membership).

  GIN (Xu et al. 2019) — the WL ceiling for message passing:
    SUM aggregation keeps the neighbour multiset; mean destroys it.
    MLP update can be injective over multisets; a linear layer cannot.
    {atom, atom, atom} and {atom} differ by SUM, not by MEAN — and in
    molecules that difference IS the chemistry.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: dataset, GINClassifier, GCNClassifier
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: TUDataset (MUTAG) + the two classifiers")
print("=" * 70)

DATA_DIR_GC = OUTPUT_DIR.parents[1] / "data" / "mlfp05" / "tudataset"
DATA_DIR_GC.mkdir(parents=True, exist_ok=True)


def load_graph_classification_dataset():
    """TUDataset MUTAG; FakeDataset fallback if the download is blocked."""
    from torch_geometric.datasets import FakeDataset, TUDataset

    try:
        dataset = TUDataset(root=str(DATA_DIR_GC), name="MUTAG")
        return dataset, "MUTAG", True
    except Exception as exc:
        print(
            f"  TUDataset download unavailable ({type(exc).__name__}: {exc})\n"
            "  Falling back to torch_geometric's built-in FakeDataset — no\n"
            "  download needed. Labels are RANDOM: accuracy is capped at\n"
            "  chance and this run demonstrates the PIPELINE, not chemistry."
        )
        dataset = FakeDataset(
            num_graphs=200, avg_num_nodes=20, avg_degree=3.0,
            num_channels=7, num_classes=2,
        )
        return dataset, "FakeDataset (random labels)", False


dataset, DATASET_NAME, IS_REAL = load_graph_classification_dataset()
N_GRAPH_CLASSES = int(dataset.num_classes)
IN_FEATURES = int(dataset.num_features)
n_graphs = len(dataset)
print(
    f"\nDataset: {DATASET_NAME} — {n_graphs} graphs, "
    f"{IN_FEATURES} node features, {N_GRAPH_CLASSES} classes"
)

labels = np.array([int(dataset[i].y.item()) for i in range(n_graphs)])
class_counts = {c: int((labels == c).sum()) for c in range(N_GRAPH_CLASSES)}
majority_rate = max(class_counts.values()) / n_graphs
print(f"  class counts: {class_counts} | majority baseline: {majority_rate:.1%}")
sizes = np.array([int(dataset[i].num_nodes) for i in range(n_graphs)])
print(
    f"  graph sizes: mean {sizes.mean():.1f} nodes "
    f"(min {sizes.min()}, max {sizes.max()})"
)

# Stratified 80/20 split (seeded)
rng = np.random.default_rng(42)
train_idx, val_idx = [], []
for c in range(N_GRAPH_CLASSES):
    idx = np.where(labels == c)[0]
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_idx.extend(idx[:cut].tolist())
    val_idx.extend(idx[cut:].tolist())
train_set = [dataset[i] for i in sorted(train_idx)]
val_set = [dataset[i] for i in sorted(val_idx)]
train_loader = PyGDataLoader(train_set, batch_size=32, shuffle=True)
val_loader = PyGDataLoader(val_set, batch_size=128)
print(f"  split: {len(train_set)} train / {len(val_set)} val (stratified)")


class GINClassifier(nn.Module):
    """GIN graph classifier: two GINConv layers + mean readout + head.

    Each GINConv wraps an MLP (Linear -> ReLU -> Linear): the update must
    be injective over neighbour multisets, and a single linear map cannot.
    """

    def __init__(self, in_dim: int = IN_FEATURES, hidden: int = 64,
                 n_classes: int = N_GRAPH_CLASSES):
        super().__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
        )
        self.head = nn.Linear(hidden, n_classes)

    def embed(self, data) -> torch.Tensor:
        """Node embeddings pooled to one vector per graph."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.embed(data))


class GCNClassifier(nn.Module):
    """GCN graph classifier — same shape as GIN, mean-style aggregation."""

    def __init__(self, in_dim: int = IN_FEATURES, hidden: int = 64,
                 n_classes: int = N_GRAPH_CLASSES):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def embed(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.embed(data))


# ── Checkpoint 1: dataset sane + batched shapes correct ───────────────
assert n_graphs >= 100, f"Too few graphs ({n_graphs}) for a classification demo"
assert N_GRAPH_CLASSES == 2
_probe = GCNClassifier().to(device)
_batch = next(iter(PyGDataLoader([dataset[0], dataset[1]], batch_size=2))).to(device)
with torch.no_grad():
    _out = _probe(_batch)
    _emb = _probe.embed(_batch)
assert _out.shape == (2, N_GRAPH_CLASSES), (
    f"2 graphs -> (2, {N_GRAPH_CLASSES}) logits, got {_out.shape}"
)
assert _emb.shape == (2, 64), f"graph embeddings should be (2, 64), got {_emb.shape}"
print(f"\n  batch of 2 molecular graphs -> logits {tuple(_out.shape)}, "
      f"embeddings {tuple(_emb.shape)}")
print("\n--- Checkpoint 1 passed --- dataset + architectures verified\n")
del _probe, _batch, _out, _emb


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: GIN and GCN with best-validation model selection
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: GIN vs GCN on whole-graph labels")
print("=" * 70)

GC_EPOCHS = 80
conn, tracker, exp_name, registry, has_registry = setup_engines()


def evaluate_graphs(model: nn.Module, loader) -> tuple[float, np.ndarray, np.ndarray]:
    """Accuracy over a PyG loader + the predictions and truths."""
    model.eval()
    correct, total = 0, 0
    preds_all, true_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(-1)
            preds_all.append(pred.cpu().numpy())
            true_all.append(batch.y.cpu().numpy())
            correct += int((pred == batch.y).sum().item())
            total += int(batch.y.numel())
    return (
        correct / max(total, 1),
        np.concatenate(preds_all),
        np.concatenate(true_all),
    )


async def train_graph_classifier(model: nn.Module, name: str):
    """Train with CE over graph labels; restore best-validation weights."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    n_params = sum(p.numel() for p in model.parameters())
    train_losses, val_accs = [], []
    best_val, best_state = -1.0, None

    async with tracker.track(experiment=exp_name, run_name=name) as run:
        await run.log_params(
            {
                "model_type": name,
                "dataset": DATASET_NAME,
                "hidden": "64",
                "epochs": str(GC_EPOCHS),
                "n_params": str(n_params),
                "task": "graph_classification",
            }
        )
        for epoch in range(GC_EPOCHS):
            model.train()
            batch_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                loss = F.cross_entropy(model(batch), batch.y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            train_losses.append(float(np.mean(batch_losses)))
            v_acc, _, _ = evaluate_graphs(model, val_loader)
            val_accs.append(v_acc)
            if v_acc > best_val:
                best_val, best_state = v_acc, copy.deepcopy(model.state_dict())
            await run.log_metrics(
                {"train_loss": train_losses[-1], "val_accuracy": v_acc},
                step=epoch + 1,
            )
            if (epoch + 1) % 20 == 0:
                print(
                    f"  [{name}] epoch {epoch+1:3d}  loss={train_losses[-1]:.4f}  "
                    f"val_acc={v_acc:.3f}"
                )
        model.load_state_dict(best_state)
        await run.log_metrics(
            {
                "final_train_loss": train_losses[-1],
                "best_val_accuracy": best_val,
            }
        )
    print(f"  [{name}] {n_params:,} params, best val_acc={best_val:.3f}")
    return train_losses, val_accs, best_val


torch.manual_seed(42)
gin = GINClassifier()
gin_losses, gin_accs, gin_best = asyncio.run(train_graph_classifier(gin, "GIN"))

torch.manual_seed(42)
gcn = GCNClassifier()
gcn_losses, gcn_accs, gcn_best = asyncio.run(train_graph_classifier(gcn, "GCN"))

# ── Checkpoint 2: training converged; honest bar on real data ─────────
assert gin_losses[-1] < gin_losses[0], "GIN train loss should decrease"
assert gcn_losses[-1] < gcn_losses[0], "GCN train loss should decrease"
if IS_REAL:
    assert gin_best > majority_rate, (
        f"GIN best val acc {gin_best:.3f} must beat the majority-class "
        f"baseline {majority_rate:.3f} on real data"
    )
else:
    print(
        "  NOTE: FakeDataset labels are random — accuracy is capped at "
        "chance. This run proves the pipeline, not predictive power."
    )

delta_gin = gin_best - gcn_best
print(f"\n{'=' * 58}")
print(f"  GRAPH CLASSIFICATION RESULTS (measured, {DATASET_NAME})")
print(f"{'=' * 58}")
print(f"  Majority-class baseline:   {majority_rate:.3f}")
print(f"  GCN best val accuracy:     {gcn_best:.3f}")
print(f"  GIN best val accuracy:     {gin_best:.3f}")
print(
    f"  GIN - GCN delta:           {delta_gin:+.3f}  "
    + (
        "(sum aggregation + MLP update pays off)"
        if delta_gin > 0.01
        else "(comparable at this dataset size — the WL advantage needs "
        "fine structural signal and enough data to show)"
    )
)
print("\n--- Checkpoint 2 passed --- both graph classifiers trained\n")

best_model = gin if gin_best >= gcn_best else gcn
best_name = "GIN" if gin_best >= gcn_best else "GCN"
best_acc = max(gin_best, gcn_best)
if has_registry:
    register_model(
        registry,
        f"m5_graph_classifier_{best_name.lower()}",
        best_model,
        [
            MetricSpec(name="best_val_accuracy", value=float(best_acc)),
            MetricSpec(name="majority_baseline", value=float(majority_rate)),
            MetricSpec(name="n_graphs", value=float(n_graphs)),
        ],
    )


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: curves, graph embeddings, sample molecules
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: what the whole-graph embedding learned")
print("=" * 70)

# (a) Training curves
fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig_curves.suptitle(f"Graph classification on {DATASET_NAME}", fontsize=13)
ax1.plot(range(1, GC_EPOCHS + 1), gin_losses, label="GIN", color="#2196F3")
ax1.plot(range(1, GC_EPOCHS + 1), gcn_losses, label="GCN", color="#FF5722")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train loss (CE)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(range(1, GC_EPOCHS + 1), gin_accs, label="GIN", color="#2196F3")
ax2.plot(range(1, GC_EPOCHS + 1), gcn_accs, label="GCN", color="#FF5722")
ax2.axhline(majority_rate, color="gray", linestyle="--",
            label=f"majority baseline ({majority_rate:.2f})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation accuracy")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
fig_curves.tight_layout()
fig_curves.savefig(str(OUTPUT_DIR / "06_graph_classification_curves.png"), dpi=150)
plt.close(fig_curves)
print(f"  Saved: {OUTPUT_DIR / '06_graph_classification_curves.png'}")

# (b) Graph-embedding PCA: one POINT PER GRAPH, coloured by class
best_model.eval()
embs, emb_labels = [], []
with torch.no_grad():
    for batch in PyGDataLoader([dataset[i] for i in range(n_graphs)],
                               batch_size=64):
        embs.append(best_model.embed(batch.to(device)).cpu().numpy())
        emb_labels.append(batch.y.cpu().numpy())
embs = np.concatenate(embs)
emb_labels = np.concatenate(emb_labels)
centred = embs - embs.mean(axis=0, keepdims=True)
Vt = np.linalg.svd(centred, full_matrices=False)[2]
coords = centred @ Vt.T[:, :2]

fig_emb, ax = plt.subplots(figsize=(9, 7))
for c, colour, label in (
    (0, "#2196F3", "class 0 (non-mutagenic)" if IS_REAL else "class 0"),
    (1, "#F44336", "class 1 (mutagenic)" if IS_REAL else "class 1"),
):
    mask = emb_labels == c
    ax.scatter(coords[mask, 0], coords[mask, 1], c=colour, s=30, alpha=0.7,
               label=label)
ax.set_title(
    f"Whole-graph embeddings ({best_name}, {DATASET_NAME}) — PCA projection"
)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend()
ax.grid(True, alpha=0.3)
fig_emb.tight_layout()
fig_emb.savefig(str(OUTPUT_DIR / "06_graph_embeddings_pca.png"), dpi=150)
plt.close(fig_emb)
print(f"  Saved: {OUTPUT_DIR / '06_graph_embeddings_pca.png'}")
print("  Each point is ONE MOLECULE — graph classification embeds whole graphs.")

# (c) Two sample molecules drawn as graphs
from torch_geometric.utils import to_networkx
import networkx as nx

fig_mol, axes = plt.subplots(1, 2, figsize=(12, 5))
for col, gi in enumerate([0, 1]):
    data_i = dataset[gi]
    g = to_networkx(data_i, to_undirected=True)
    pos = nx.spring_layout(g, seed=42)
    nx.draw(
        g, pos, ax=axes[col], node_size=120,
        node_color=data_i.x.argmax(dim=-1).numpy(), cmap="tab10",
        edge_color="gray", alpha=0.9,
    )
    axes[col].set_title(
        f"graph {gi}: {int(data_i.num_nodes)} atoms, label={int(data_i.y.item())}",
        fontsize=11,
    )
fig_mol.suptitle("What one 'instance' looks like in graph classification",
                 fontsize=13)
fig_mol.tight_layout()
fig_mol.savefig(str(OUTPUT_DIR / "06_sample_graphs.png"), dpi=150)
plt.close(fig_mol)
print(f"  Saved: {OUTPUT_DIR / '06_sample_graphs.png'}")

# ── Checkpoint 3: artefacts exist ─────────────────────────────────────
import os

for artefact in (
    "06_graph_classification_curves.png",
    "06_graph_embeddings_pca.png",
    "06_sample_graphs.png",
):
    assert os.path.exists(OUTPUT_DIR / artefact), f"Missing: {artefact}"
print("\n--- Checkpoint 3 passed --- visual proof generated\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: pre-synthesis mutagenicity screening
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a specialty-chemicals firm
# evaluates candidate compounds BEFORE committing to synthesis. A
# synthesis run costs ~$8,000 in materials and bench time; shipping a
# mutagenic compound to a client costs far more. The screening model
# answers: "does this molecular STRUCTURE look mutagenic?"
#
# The label lives on the whole molecule, so node-level tricks do not
# apply — this is exactly graph classification. The metric that matters
# is not accuracy: it is the confusion split on the MUTAGENIC class.
#   - Miss a mutagen (false negative): unsafe compound advances ($$$$).
#   - Flag a safe compound (false positive): wasted synthesis slot ($).
#
# We compute both rates from the validation predictions — measured, not
# assumed — and let the asymmetry of the costs set the operating point.

print("=" * 70)
print("  PHASE 5 — APPLY: pre-synthesis screening with per-class costs")
print("=" * 70)

val_acc_best, val_preds, val_true = evaluate_graphs(best_model, val_loader)
mutagenic = 1  # class index of the costly-to-miss class
tp = int(((val_preds == mutagenic) & (val_true == mutagenic)).sum())
fn = int(((val_preds != mutagenic) & (val_true == mutagenic)).sum())
fp = int(((val_preds == mutagenic) & (val_true != mutagenic)).sum())
tn = int(((val_preds != mutagenic) & (val_true != mutagenic)).sum())
recall = tp / max(tp + fn, 1)      # mutagens caught
precision = tp / max(tp + fp, 1)   # flags that were real

SYNTHESIS_COST = 8_000
MISS_COST = 250_000  # illustrative liability exposure per missed mutagen
avoided = tp * MISS_COST
wasted = fp * SYNTHESIS_COST

print(
    f"""
  SCREENING REPORT — {best_name} on {DATASET_NAME} (measured, this run):

    Validation graphs:        {len(val_true)}
    Overall accuracy:         {val_acc_best:.1%}   (majority baseline {majority_rate:.1%})
    Mutagens caught (recall): {recall:.1%}   ({tp} caught, {fn} missed)
    Flag precision:           {precision:.1%}   ({fp} safe compounds flagged)

  COST READING (illustrative costs):
    Mutagens caught x $250K liability avoided:  ${avoided:,}
    False flags x $8K wasted synthesis:         ${wasted:,}
    Net asymmetric value:                       ${avoided - wasted:,}

    The recall row is the one a safety officer reads first. Accuracy can
    look healthy while every mutagen slips through — always split the
    confusion by class when the costs are asymmetric.

  STAKEHOLDER-READY OUTPUT:
    "On {len(val_true)} held-out compounds the graph classifier catches
    {recall:.0%} of mutagenic structures before synthesis, at a flag
    precision of {precision:.0%}. Each molecule is evaluated as a whole
    graph — atoms and bonds, not a feature vector — so structurally
    similar compounds get similar verdicts."
"""
)

# ── Checkpoint 4: confusion counts are consistent ─────────────────────
assert tp + fn + fp + tn == len(val_true), "confusion cells must sum to N"
assert 0.0 <= recall <= 1.0 and 0.0 <= precision <= 1.0
print("--- Checkpoint 4 passed --- screening application demonstrated\n")

asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] Node vs graph classification: the graph is the world vs the graph
      is the instance
  [x] Readout: global_mean_pool turns node embeddings into one vector
      per graph (PyG's batch vector tracks membership)
  [x] GIN (Xu 2019): SUM aggregation keeps the neighbour multiset, the
      MLP update can be injective — together reaching the WL ceiling

  BUILD + TRAIN:
  [x] TUDataset MUTAG loaded via torch_geometric ({n_graphs} graphs,
      {IN_FEATURES} node features, 2 classes){'' if IS_REAL else ' — FakeDataset fallback this run (download blocked)'}
  [x] GINClassifier and GCNClassifier with PyG DataLoader batching
  [x] Best-validation model selection; every epoch in ExperimentTracker
  [x] Measured: GIN {gin_best:.3f} vs GCN {gcn_best:.3f} vs majority
      baseline {majority_rate:.3f}

  VISUALISE (the proof):
  [x] Training curves with the majority-baseline line drawn
  [x] Graph-embedding PCA — one point per molecule, coloured by class
  [x] Sample molecular graphs drawn as graphs (the actual input)

  APPLY:
  [x] Pre-synthesis screening scored by the asymmetric metric: recall
      {recall:.1%} on the mutagenic class, flag precision {precision:.1%}
  [x] Cost reading in dollars, not just rates: ${avoided - wasted:,} net
      illustrative value on the validation set
  [x] Stated limit: accuracy can hide every missed mutagen — split the
      confusion by class when costs are asymmetric

  KEY INSIGHT: The architecture follows the label's LOCATION. Labels on
  nodes -> message passing + per-node head (Cora). Labels on whole
  graphs -> message passing + readout + per-graph head (MUTAG). And when
  the label is expensive to miss, the evaluation must follow the COST,
  not the convention — recall on the costly class first, accuracy last.
"""
)
