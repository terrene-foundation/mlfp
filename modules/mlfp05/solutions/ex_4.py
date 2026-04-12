# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4: Transformers and Self-Attention
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive scaled dot-product attention from scratch using torch.einsum
#   - Explain why we divide by sqrt(d_k) (softmax saturation on large dims)
#   - Implement multi-head attention with nn.MultiheadAttention
#   - Build a small Transformer encoder with nn.TransformerEncoder
#   - Train a sequence classifier end-to-end on synthetic token data
#   - Apply sinusoidal positional encoding (transformers have no position sense)
#
# PREREQUISITES: M5/ex_3 (RNNs, sequence modelling, nn.Module training).
# ESTIMATED TIME: ~60 min
# DATASET: Synthetic sequence classification (majority-bit task).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kailash_ml import ModelVisualizer

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Scaled dot-product attention from scratch
# ════════════════════════════════════════════════════════════════════════
# For a single head:
#   Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
#
# Q (query): "what am I looking for?"   — shape (B, L_q, d_k)
# K (key):   "what can I offer?"        — shape (B, L_k, d_k)
# V (value): "what do I actually pass"  — shape (B, L_k, d_v)
#
# Dividing by sqrt(d_k) prevents the dot products from growing with the
# embedding dimension, which would push softmax into saturation and kill
# gradients for all non-maximal keys.
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.size(-1)
    # einsum makes the batched matmul explicit: (B, Lq, D) x (B, Lk, D)^T
    scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("bqk,bkd->bqd", weights, v)
    return out, weights


# Quick sanity check on attention: with scaled identity-style queries,
# each query should attend most strongly to its own key.
q_demo = torch.eye(4, 8).unsqueeze(0) * 3.0
k_demo = q_demo.clone()
v_demo = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)
_, attn = scaled_dot_product_attention(q_demo, k_demo, v_demo)
print("Demo attention weights (should peak on the diagonal):")
print(attn.squeeze(0).round(decimals=2))


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Multi-head attention (educational version using the kernel above)
# ════════════════════════════════════════════════════════════════════════
# Multi-head attention runs h attention operations in parallel over
# different learned projections of Q/K/V, then concatenates. Each head can
# specialise. nn.MultiheadAttention bundles this with the output projection.
class EducationalMultiHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, d = x.shape
        qkv = self.qkv(x).reshape(b, seq, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)                 # each: (b, seq, h, d_k)
        q = q.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        k = k.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        v = v.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        out, _ = scaled_dot_product_attention(q, k, v)
        out = out.reshape(b, self.n_heads, seq, self.d_k).transpose(1, 2).reshape(b, seq, d)
        return self.proj(out)


# ════════════════════════════════════════════════════════════════════════
# PART 3 — Sinusoidal positional encoding
# ════════════════════════════════════════════════════════════════════════
# Transformers have no inherent sense of order. We add a fixed sinusoidal
# encoding so the model can attend by position as well as by content:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ════════════════════════════════════════════════════════════════════════
# PART 4 — Full Transformer encoder classifier
# ════════════════════════════════════════════════════════════════════════
# nn.TransformerEncoder stacks TransformerEncoderLayer blocks. Each block:
# multi-head self-attention -> add & norm -> feed-forward -> add & norm.
# We add an embedding, positional encoding, stack two encoder layers, mean
# pool over time, and project to class logits.
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, n_classes: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.posenc(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


# ════════════════════════════════════════════════════════════════════════
# Synthetic sequence classification — majority-bit task
# ════════════════════════════════════════════════════════════════════════
# Each sequence is 16 tokens from a vocab of 4 symbols. Symbol 0 = "bit 0",
# symbol 1 = "bit 1", symbols 2/3 = "noise". The label is 1 if there are
# more 1-bits than 0-bits in the sequence (ignoring noise), else 0.
def make_majority_bits(n: int = 1200, seq_len: int = 16):
    tokens = np.random.randint(0, 4, size=(n, seq_len), dtype=np.int64)
    bit_mask = tokens < 2
    ones = ((tokens == 1) & bit_mask).sum(axis=1)
    zeros = ((tokens == 0) & bit_mask).sum(axis=1)
    labels = (ones > zeros).astype(np.int64)
    return tokens, labels


tokens_np, labels_np = make_majority_bits(n=1200, seq_len=16)
tokens = torch.from_numpy(tokens_np).to(device)
labels = torch.from_numpy(labels_np).to(device)

split = int(0.8 * len(tokens))
train_loader = DataLoader(TensorDataset(tokens[:split], labels[:split]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(tokens[split:], labels[split:]), batch_size=32)


# ── Train the transformer ─────────────────────────────────────────────
model = TransformerClassifier(vocab_size=4, d_model=64, n_heads=4, n_layers=2, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses: list[float] = []
val_accs: list[float] = []

print("\n── Training TransformerClassifier ──")
for epoch in range(8):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(float(np.mean(batch_losses)))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for xb, yb in val_loader:
            preds = model(xb).argmax(dim=-1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
        val_accs.append(correct / total)
    print(f"  epoch {epoch+1}  train_loss={train_losses[-1]:.4f}  val_acc={val_accs[-1]:.3f}")


# ════════════════════════════════════════════════════════════════════════
# PART 5 — Sanity-check the educational multi-head attention
# ════════════════════════════════════════════════════════════════════════
mha = EducationalMultiHead(d_model=64, n_heads=4).to(device)
dummy = torch.randn(2, 16, 64, device=device)
out = mha(dummy)
print(f"\nEducationalMultiHead output shape: {tuple(out.shape)}  (expected (2, 16, 64))")


# ════════════════════════════════════════════════════════════════════════
# PART 6 — Visualise training history
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()
fig = viz.training_history(
    metrics={"loss": train_losses, "val_acc": val_accs},
    x_label="Epoch",
    y_label="Value",
)
fig.write_html("ex_4_training.html")
print("Training history saved to ex_4_training.html")


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Derived scaled dot-product attention with torch.einsum
  [x] Explained the 1/sqrt(d_k) factor (prevents softmax saturation)
  [x] Wrote a hand-rolled multi-head attention wrapping the scratch kernel
  [x] Built a TransformerClassifier with nn.TransformerEncoder
  [x] Trained it end-to-end and watched the accuracy climb on a real task
"""
)
