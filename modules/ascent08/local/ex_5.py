# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 5: Attention Mechanisms
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement scaled dot-product attention and multi-head
#   attention from scratch, then visualize attention patterns with
#   ModelVisualizer.
#
# TASKS:
#   1. Implement scaled dot-product attention
#   2. Build multi-head attention module
#   3. Visualize attention weights on sample sequences
#   4. Demonstrate how attention solves the context bottleneck
#   5. Compare attention vs LSTM on long sequences
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_product_reviews.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)
print(f"=== Dataset: {df.height} reviews ===")
print(summary)


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<unk>"] + [w for w, c in word_counts.most_common(2000) if c >= 2]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim = 32

# Random embeddings for demonstration
embeddings = [
    [random.gauss(0, 0.1) for _ in range(embed_dim)] for _ in range(vocab_size)
]

print(f"Vocabulary: {vocab_size}, embedding dim: {embed_dim}")


def text_to_embeddings(text: str, max_len: int = 20) -> list[list[float]]:
    """Convert text to a sequence of embedding vectors."""
    tokens = tokenize(text)[:max_len]
    indices = [word_to_idx.get(t, 1) for t in tokens]
    return [embeddings[idx] for idx in indices]


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Scaled dot-product attention
# ══════════════════════════════════════════════════════════════════════


def softmax(scores: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_s = max(scores) if scores else 0
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def scaled_dot_product_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Returns (output, attention_weights).
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    seq_len_q = len(Q)
    seq_len_k = len(K)

    # TODO: Compute Q K^T / sqrt(d_k) — the attention score matrix.
    # Hint: For each query i and key j, compute sum(Q[i][d] * K[j][d] for d in range(d_k)) / scale
    scores = []
    for i in range(seq_len_q):
        row = []
        for j in range(seq_len_k):
            # TODO: Compute dot product between Q[i] and K[j].
            # Hint: sum(Q[i][d] * K[j][d] for d in range(d_k))
            dot = ____
            row.append(dot / scale)
        scores.append(row)

    # Softmax over keys for each query
    weights = [softmax(row) for row in scores]

    # Weighted sum of values
    d_v = len(V[0])
    output = []
    for i in range(seq_len_q):
        out_vec = [0.0] * d_v
        for j in range(seq_len_k):
            for d in range(d_v):
                out_vec[d] += weights[i][j] * V[j][d]
        output.append(out_vec)

    return output, weights


# Demo: self-attention on a sample review
sample_text = corpus[0] if corpus else "this product is great for singapore weather"
sample_emb = text_to_embeddings(sample_text, max_len=10)
sample_tokens = tokenize(sample_text)[:10]

# Self-attention: Q = K = V = input embeddings
output, attn_weights = scaled_dot_product_attention(sample_emb, sample_emb, sample_emb)

print(f"\n--- Scaled Dot-Product Attention ---")
print(f"Input sequence: {sample_tokens}")
print(f"Q/K/V dim: {len(sample_emb)}x{embed_dim}")
print(f"Output shape: {len(output)}x{len(output[0])}")
print(f"\nAttention weights (first token attends to):")
for j, w in enumerate(attn_weights[0]):
    token = sample_tokens[j] if j < len(sample_tokens) else "?"
    bar = "#" * int(w * 40)
    print(f"  {token:<15} {w:.3f} {bar}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Multi-head attention
# ══════════════════════════════════════════════════════════════════════


def linear_projection(X: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    """Project X (seq_len x d_in) with W (d_in x d_out)."""
    d_out = len(W[0])
    result = []
    for x in X:
        row = [0.0] * d_out
        for j in range(d_out):
            for k in range(len(x)):
                row[j] += x[k] * W[k][j]
        result.append(row)
    return result


class MultiHeadAttention:
    """Multi-head attention: parallel attention heads, then concatenate + project."""

    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        scale = 1.0 / math.sqrt(self.d_k)
        # Per-head projections
        self.W_q = [
            [[random.gauss(0, scale) for _ in range(self.d_k)] for _ in range(d_model)]
            for _ in range(n_heads)
        ]
        self.W_k = [
            [[random.gauss(0, scale) for _ in range(self.d_k)] for _ in range(d_model)]
            for _ in range(n_heads)
        ]
        self.W_v = [
            [[random.gauss(0, scale) for _ in range(self.d_k)] for _ in range(d_model)]
            for _ in range(n_heads)
        ]
        # Output projection
        self.W_o = [
            [random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model)
        ]

    def forward(
        self, X: list[list[float]]
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        """Multi-head self-attention. Returns (output, all_head_weights)."""
        all_head_outputs = []
        all_head_weights = []

        for h in range(self.n_heads):
            # TODO: Project X to Q, K, V using per-head weight matrices.
            # Hint: linear_projection(X, self.W_q[h])
            Q_h = ____
            K_h = linear_projection(X, self.W_k[h])
            V_h = linear_projection(X, self.W_v[h])
            head_out, head_weights = scaled_dot_product_attention(Q_h, K_h, V_h)
            all_head_outputs.append(head_out)
            all_head_weights.append(head_weights)

        # Concatenate heads
        seq_len = len(X)
        concat = []
        for i in range(seq_len):
            row = []
            for h in range(self.n_heads):
                row.extend(all_head_outputs[h][i])
            concat.append(row)

        # Output projection
        output = linear_projection(concat, self.W_o)
        return output, all_head_weights


n_heads = 4
mha = MultiHeadAttention(d_model=embed_dim, n_heads=n_heads)
mha_output, head_weights = mha.forward(sample_emb)

print(f"\n--- Multi-Head Attention ({n_heads} heads) ---")
print(f"Input: {len(sample_emb)}x{embed_dim}")
print(f"Per-head dim: {embed_dim // n_heads}")
print(f"Output: {len(mha_output)}x{len(mha_output[0])}")

for h in range(n_heads):
    top_attn = sorted(
        range(len(head_weights[h][0])),
        key=lambda j: head_weights[h][0][j],
        reverse=True,
    )[:3]
    top_tokens = [sample_tokens[j] if j < len(sample_tokens) else "?" for j in top_attn]
    print(f"  Head {h}: token[0] attends most to {top_tokens}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Visualize attention weights
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# TODO: Visualize attention heatmap for the first head using plot_embeddings.
# Hint: viz.plot_embeddings(embeddings=[row for row in head_weights[0]], labels=sample_tokens, method="tsne", title="Attention Head 0 — Query-Key Relationships")
fig = ____

print(f"\n=== Attention visualization generated ===")
print(f"Each head learns different linguistic relationships:")
print(f"  Head 0: may capture syntactic dependencies")
print(f"  Head 1: may capture semantic similarity")
print(f"  Head 2: may capture positional patterns")
print(f"  Head 3: may capture entity co-reference")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Attention solves the context bottleneck
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Context Bottleneck Problem ---")
print(f"RNN/LSTM: entire sequence compressed into fixed-size vector h_T")
print(f"  Long sequences → early tokens are 'forgotten'")
print(f"  Bottleneck: all information must pass through hidden state")
print(f"\nAttention: each output position can directly attend to ALL inputs")
print(f"  No bottleneck — O(1) path from any input to any output")
print(f"  Trade-off: O(n^2) memory for attention matrix")

# Demonstrate: attention weights for distant tokens
if len(sample_tokens) >= 5:
    first_to_last = attn_weights[0][-1]
    last_to_first = attn_weights[-1][0]
    print(
        f"\n  Attention from '{sample_tokens[0]}' to '{sample_tokens[-1]}': {first_to_last:.4f}"
    )
    print(
        f"  Attention from '{sample_tokens[-1]}' to '{sample_tokens[0]}': {last_to_first:.4f}"
    )
    print(f"  → Direct connection regardless of distance!")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare attention vs LSTM on long sequences
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Attention vs LSTM: Long Sequence Scaling ---")
print(
    f"{'Seq Length':<12} {'LSTM path':<15} {'Attention path':<15} {'Attention memory'}"
)
print("-" * 60)
for seq_len in [10, 50, 100, 500, 1000]:
    lstm_path = seq_len  # Sequential path length
    attn_path = 1  # Direct connection
    attn_memory = seq_len * seq_len  # O(n^2) attention matrix
    print(f"  {seq_len:<12} {lstm_path:<15} {attn_path:<15} {attn_memory:,}")

print(f"\nLSTM: O(n) path length, O(n) memory — gradients vanish over long paths")
print(f"Attention: O(1) path length, O(n^2) memory — constant gradient path")
print(f"Transformers combine attention's O(1) paths with parallelism (no recurrence)")

print(
    "\n✓ Exercise 5 complete — scaled dot-product + multi-head attention, visualization"
)
