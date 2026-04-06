# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 6: Transformer Architecture
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a mini transformer encoder from components — positional
#   encoding, multi-head attention, feed-forward network, layer
#   normalization, and residual connections.
#
# TASKS:
#   1. Implement sinusoidal positional encoding
#   2. Build transformer encoder layer (attention + FFN + LayerNorm + residual)
#   3. Stack layers into full encoder
#   4. Train on text classification task
#   5. Visualize attention patterns per layer with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer, TrainingPipeline

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_news_articles.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)
print(f"=== Dataset: {df.height} articles ===")
print(summary)


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<cls>", "<unk>"] + [
    w for w, c in word_counts.most_common(2000) if c >= 2
]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
d_model = 32

# Random token embeddings
token_embeddings = [
    [random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)
]

print(f"Vocabulary: {vocab_size}, d_model: {d_model}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Sinusoidal positional encoding
# ══════════════════════════════════════════════════════════════════════


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> list[list[float]]:
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = []
    for pos in range(max_len):
        row = [0.0] * d_model
        for i in range(0, d_model, 2):
            # TODO: Compute the denominator for positional encoding.
            # Hint: 10000.0 ** (i / d_model)
            denom = ____
            row[i] = math.sin(pos / denom)
            if i + 1 < d_model:
                row[i + 1] = math.cos(pos / denom)
        pe.append(row)
    return pe


max_seq_len = 50
pos_enc = sinusoidal_positional_encoding(max_seq_len, d_model)

print(f"\n--- Positional Encoding ---")
print(f"Shape: {len(pos_enc)}x{len(pos_enc[0])}")
for pos in [0, 1, 10, 49]:
    print(
        f"  pos={pos}: [{pos_enc[pos][0]:.3f}, {pos_enc[pos][1]:.3f}, ..., {pos_enc[pos][-1]:.3f}]"
    )

print(f"\nProperties:")
print(f"  - Unique encoding per position (no learned parameters)")
print(f"  - Captures relative position via dot product")
print(f"  - Generalizes to longer sequences than seen during training")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Transformer encoder layer
# ══════════════════════════════════════════════════════════════════════


def softmax(scores: list[float]) -> list[float]:
    max_s = max(scores) if scores else 0
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def layer_norm(x: list[float], eps: float = 1e-6) -> list[float]:
    """Layer normalization: normalize across the feature dimension."""
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


def feed_forward(
    x: list[float],
    W1: list[list[float]],
    b1: list[float],
    W2: list[list[float]],
    b2: list[float],
) -> list[float]:
    """FFN(x) = ReLU(xW1 + b1)W2 + b2."""
    d_ff = len(W1[0])
    hidden = [0.0] * d_ff
    for j in range(d_ff):
        val = b1[j]
        for k in range(len(x)):
            val += x[k] * W1[k][j]
        hidden[j] = max(0.0, val)  # ReLU

    d_out = len(W2[0])
    output = [0.0] * d_out
    for j in range(d_out):
        val = b2[j]
        for k in range(d_ff):
            val += hidden[k] * W2[k][j]
        output[j] = val
    return output


def residual_add(x: list[float], sublayer_out: list[float]) -> list[float]:
    """Residual connection: x + sublayer(x)."""
    return [a + b for a, b in zip(x, sublayer_out)]


class TransformerEncoderLayer:
    """Single transformer encoder layer: self-attention + FFN + residuals + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = 1.0 / math.sqrt(self.d_k)

        # Attention weights
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
        self.W_o = [
            [random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model)
        ]

        # FFN weights
        ff_scale = 1.0 / math.sqrt(d_ff)
        self.W1 = [
            [random.gauss(0, ff_scale) for _ in range(d_ff)] for _ in range(d_model)
        ]
        self.b1 = [0.0] * d_ff
        self.W2 = [
            [random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_ff)
        ]
        self.b2 = [0.0] * d_model

    def self_attention(
        self, X: list[list[float]]
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        """Multi-head self-attention."""
        all_heads = []
        all_weights = []
        for h in range(self.n_heads):
            Q = [
                [
                    sum(X[i][k] * self.W_q[h][k][j] for k in range(self.d_model))
                    for j in range(self.d_k)
                ]
                for i in range(len(X))
            ]
            K = [
                [
                    sum(X[i][k] * self.W_k[h][k][j] for k in range(self.d_model))
                    for j in range(self.d_k)
                ]
                for i in range(len(X))
            ]
            V = [
                [
                    sum(X[i][k] * self.W_v[h][k][j] for k in range(self.d_model))
                    for j in range(self.d_k)
                ]
                for i in range(len(X))
            ]

            scale = math.sqrt(self.d_k)
            scores = [
                [
                    sum(Q[i][d] * K[j][d] for d in range(self.d_k)) / scale
                    for j in range(len(K))
                ]
                for i in range(len(Q))
            ]
            weights = [softmax(row) for row in scores]
            head_out = [
                [
                    sum(weights[i][j] * V[j][d] for j in range(len(V)))
                    for d in range(self.d_k)
                ]
                for i in range(len(Q))
            ]
            all_heads.append(head_out)
            all_weights.append(weights)

        # Concatenate + project
        concat = [[] for _ in range(len(X))]
        for i in range(len(X)):
            for h in range(self.n_heads):
                concat[i].extend(all_heads[h][i])

        output = [
            [
                sum(concat[i][k] * self.W_o[k][j] for k in range(self.d_model))
                for j in range(self.d_model)
            ]
            for i in range(len(X))
        ]
        return output, all_weights

    def forward(
        self, X: list[list[float]]
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        """Full encoder layer: attention → residual → LayerNorm → FFN → residual → LayerNorm."""
        attn_out, attn_weights = self.self_attention(X)
        # TODO: Apply residual connection + layer norm after attention.
        # Hint: [layer_norm(residual_add(X[i], attn_out[i])) for i in range(len(X))]
        normed1 = ____

        ffn_out = [
            feed_forward(normed1[i], self.W1, self.b1, self.W2, self.b2)
            for i in range(len(normed1))
        ]
        # TODO: Apply residual connection + layer norm after FFN.
        # Hint: [layer_norm(residual_add(normed1[i], ffn_out[i])) for i in range(len(normed1))]
        normed2 = ____

        return normed2, attn_weights


n_heads = 4
d_ff = 64
layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

# Test with sample input
sample_tokens = tokenize(corpus[0] if corpus else "singapore economy growth")[:10]
sample_indices = [word_to_idx.get(t, 2) for t in sample_tokens]
sample_input = [
    [token_embeddings[idx][d] + pos_enc[pos][d] for d in range(d_model)]
    for pos, idx in enumerate(sample_indices)
]

layer_out, layer_attn = layer.forward(sample_input)
print(f"\n--- Encoder Layer ---")
print(
    f"Input: {len(sample_input)}x{d_model}, Output: {len(layer_out)}x{len(layer_out[0])}"
)
print(f"Attention heads: {n_heads}, FFN dim: {d_ff}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Stack layers into full encoder
# ══════════════════════════════════════════════════════════════════════


class TransformerEncoder:
    """Stack of N transformer encoder layers."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int):
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ]

    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        """Forward through all layers, collecting attention weights."""
        all_layer_attn = []
        hidden = X
        for layer in self.layers:
            hidden, attn_weights = layer.forward(hidden)
            all_layer_attn.append(attn_weights)
        return hidden, all_layer_attn


n_layers = 3
encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff)
enc_output, all_attn = encoder.forward(sample_input)

print(f"\n--- Full Encoder ({n_layers} layers) ---")
print(f"Output shape: {len(enc_output)}x{len(enc_output[0])}")
print(f"Attention maps collected: {len(all_attn)} layers x {n_heads} heads")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train on text classification task
# ══════════════════════════════════════════════════════════════════════

# TODO: Configure TrainingPipeline for text classification on the "category" target.
# Hint: TrainingPipeline(model_type="text_classifier", target="category", features=["text"])
pipeline = ____

# TODO: Fit the pipeline on the dataframe.
# Hint: pipeline.fit(df)
result = ____
predictions = pipeline.predict(df)

print(f"\n=== TrainingPipeline Text Classification ===")
print(f"Training result: {result}")
print(f"Predictions shape: {predictions.height} rows")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualize attention patterns per layer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

for layer_idx in range(n_layers):
    for head_idx in range(min(2, n_heads)):  # Show first 2 heads per layer
        weights = all_attn[layer_idx][head_idx]
        fig = viz.plot_embeddings(
            embeddings=weights,
            labels=sample_tokens,
            method="tsne",
            title=f"Layer {layer_idx} Head {head_idx} Attention",
        )

print(f"\n=== Attention Pattern Analysis ===")
print(f"Layer 0: tends to capture local/syntactic patterns")
print(f"Layer 1: tends to capture broader semantic relationships")
print(f"Layer 2: tends to capture task-specific patterns")
print(f"\nThis hierarchy is why deeper transformers capture more complex patterns.")

print("\n✓ Exercise 6 complete — mini transformer encoder from scratch")
