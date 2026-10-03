# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.2: Transformer Encoder Classifier
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement multi-head attention with parallel learned projections
#   - Explain how different heads capture different relationship types
#   - Build sinusoidal positional encoding (giving transformers order)
#   - Construct a full Transformer encoder classifier with nn.TransformerEncoder
#   - Train the Transformer on AG News and log metrics with ExperimentTracker
#   - Check whether a trained classifier's labels fit a new task
#     (routing regulatory filings) before using it
#
# PREREQUISITES: ex_4/01_self_attention_from_scratch.py
# ESTIMATED TIME: ~30 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_4 import (
    CLASS_NAMES,
    DEVICE,
    EPOCHS_SCRATCH,
    MAX_LEN,
    VOCAB_SIZE,
    build_vocab,
    create_attention_heatmap,
    load_ag_news,
    prepare_dataloaders,
    scaled_dot_product_attention,
    setup_engines,
    text_to_indices,
    evaluate_accuracy,
    train_model,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Multi-Head Attention and Positional Encoding
# ════════════════════════════════════════════════════════════════════════
# A single attention head can only capture one type of relationship at a
# time. But language has many simultaneous relationship types:
#
#   "The bank raised interest rates to control inflation"
#
#   - SYNTACTIC: "bank" -> "raised" (subject-verb)
#   - SEMANTIC: "rates" -> "inflation" (economic cause-effect)
#   - COREFERENCE: "bank" -> "central bank" (if mentioned earlier)
#
# Multi-head attention runs h parallel attention operations, each with
# its own learned Q/K/V projections. Head 1 might learn syntax, head 2
# might learn semantics, head 3 might learn entity relationships. The
# outputs are concatenated and projected back to the model dimension.
#
# POSITIONAL ENCODING: Transformers process all tokens simultaneously
# (unlike RNNs, which process sequentially). This means they have no
# inherent sense of word order. "Dog bites man" and "Man bites dog"
# would look identical without positional information. We inject position
# using a sinusoidal signal:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#
# This scheme gives each position a unique signature and allows the model
# to learn relative positions (the offset between any two positions has
# a consistent geometric relationship in the PE space).
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up engines
# ════════════════════════════════════════════════════════════════════════
train_df, test_df = load_ag_news()
vocab = build_vocab(train_df["text"].to_list())
train_loader, val_loader, train_t, train_y, test_t, test_y = prepare_dataloaders(
    train_df, test_df, vocab
)
conn, tracker, exp_name, registry, has_registry, bridge = setup_engines()
print(f"  vocab size: {len(vocab)}, seq_len: {MAX_LEN}")
print(f"  ExperimentTracker ready, experiment: {exp_name}")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build: Educational Multi-Head Attention
# ════════════════════════════════════════════════════════════════════════
# This implementation wraps our from-scratch scaled_dot_product_attention
# with learned Q/K/V projections and multi-head splitting. PyTorch has
# nn.MultiheadAttention, but building it ourselves reveals the mechanics.
class EducationalMultiHead(nn.Module):
    """Multi-head attention built on our from-scratch attention kernel.

    Each head gets its own learned projection of the input into Q, K, V
    subspaces. The heads operate in parallel, then their outputs are
    concatenated and projected back to the model dimension.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Single linear layer produces Q, K, V for all heads at once
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning output and attention weights.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            (output, attention_weights) where output has shape
            (batch, seq_len, d_model) and attention_weights has shape
            (batch, n_heads, seq_len, seq_len).
        """
        b, seq, d = x.shape

        # TODO: Compute Q, K, V for all heads in one matrix multiply
        # Hint: self.qkv gives (b, seq, 3 * d_model); view it as
        #   (b, seq, 3, n_heads, d_k), then split the "3" axis into q, k, v,
        #   each (b, seq, n_heads, d_k).
        qkv = ...  # YOUR CODE HERE
        q, k, v = ...  # YOUR CODE HERE

        # TODO: Reshape for attention — merge batch and head dims
        # Hint: bring the head axis next to batch, then fold both into one
        #   axis -> (b * n_heads, seq, d_k). Same for q, k and v.
        q = ...  # YOUR CODE HERE
        k = ...  # YOUR CODE HERE
        v = ...  # YOUR CODE HERE

        # TODO: Apply scaled_dot_product_attention from helpers (every head
        #   is now just another batch element).
        out, weights = ...  # YOUR CODE HERE

        # Reshape weights to (b, n_heads, seq, seq) for visualisation
        attn_weights = weights.reshape(b, self.n_heads, seq, seq)

        # TODO: Concatenate heads and project back to d_model
        # Hint: undo the fold: (b * n_heads, seq, d_k) -> (b, seq, d_model),
        #   with each position's heads side by side. Return the projected
        #   output together with attn_weights.
        out = ...  # YOUR CODE HERE
        return ...  # YOUR CODE HERE


# Sanity check the educational multi-head attention
mha = EducationalMultiHead(d_model=64, n_heads=4).to(DEVICE)
dummy = torch.randn(2, 16, 64, device=DEVICE)
mha_out, mha_attn = mha(dummy)
print(
    f"\nEducationalMultiHead output shape: {tuple(mha_out.shape)}  (expected (2, 16, 64))"
)
print(f"Attention weights shape: {tuple(mha_attn.shape)}  (expected (2, 4, 16, 16))")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert mha_out.shape == (2, 16, 64), "Multi-head output should be (2, 16, 64)"
assert mha_attn.shape == (2, 4, 16, 16), "Attention should be (batch, heads, seq, seq)"
print("\n--- Checkpoint 1 passed --- multi-head attention architecture ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build: Positional Encoding + Transformer Classifier
# ════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Adds a fixed, non-learned signal that encodes position. Each dimension
    oscillates at a different frequency, creating a unique "fingerprint"
    for each position that the model can use to distinguish word order.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # TODO: Fill pe with sinusoidal values
        # Hint: even feature columns get sin(position * div), odd columns get
        #   cos(position * div) (slice the column axis with a step of 2).
        ...  # YOUR CODE HERE
        ...  # YOUR CODE HERE
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Full Transformer encoder for text classification.

    Architecture: embedding -> positional encoding -> stacked
    TransformerEncoderLayers -> mean pool -> classification head.

    Uses PyTorch's nn.TransformerEncoder for the stacked layers (which
    internally uses nn.MultiheadAttention), but the architecture mirrors
    our educational implementation above.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: Build the Transformer architecture
        # - embed: token embedding, d_model wide, id 0 is padding
        # - posenc: the PositionalEncoding above; emb_drop / head_drop: dropout
        # - layer: one nn.TransformerEncoderLayer — n_heads heads, feed-forward
        #   width 4 * d_model, batch-first tensors
        # - encoder: nn.TransformerEncoder stacking n_layers copies; pass
        #   enable_nested_tensor=False (the nested-tensor fast path fails on MPS)
        # - head: linear map from d_model to n_classes
        self.embed = ...  # YOUR CODE HERE
        self.posenc = ...  # YOUR CODE HERE
        self.emb_drop = ...  # YOUR CODE HERE
        layer = ...  # YOUR CODE HERE
        self.encoder = ...  # YOUR CODE HERE
        self.head_drop = ...  # YOUR CODE HERE
        self.head = ...  # YOUR CODE HERE

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Step 1: boolean pad mask (True where the token id is 0)
        # Step 2: embed -> positional encoding -> dropout
        # Step 3: encoder, telling it which keys are padding
        #         (src_key_padding_mask)
        # Step 4: mean-pool over the NON-pad positions only (zero the pads,
        #         divide by the real length, never by 0)
        # Step 5: dropout -> classification head -> logits (batch, n_classes)
        ...  # YOUR CODE HERE


# ── Checkpoint 2 ─────────────────────────────────────────────────────
tc_test = TransformerClassifier(vocab_size=100).to(DEVICE)
dummy_tokens = torch.randint(0, 100, (2, MAX_LEN), device=DEVICE)
tc_out = tc_test(dummy_tokens)
assert tc_out.shape == (2, 4), "TransformerClassifier should output (batch, 4 classes)"
print("--- Checkpoint 2 passed --- TransformerClassifier architecture ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train: Transformer on full AG News with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
print("\n== Training Transformer on full AG News ==")
# TODO: Create TransformerClassifier and train it
# - transformer_model: full vocab, d_model 128, 4 heads, 3 layers, 4 classes
# - transformer_losses, transformer_accs: from the train_model helper
#   (run name "transformer", the train/val loaders, tracker, exp_name,
#   EPOCHS_SCRATCH epochs)
transformer_model = ...  # YOUR CODE HERE
transformer_losses, transformer_accs = ...  # YOUR CODE HERE

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Transformer (attention + residual stack)
# ══════════════════════════════════════════════════════════════════
# Probes come from the training loader (the probe runs in train mode,
# with dropout active).
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _ce_loss(m, batch):
    """Cross-entropy on one (token_ids, labels) batch."""
    xb, yb = batch
    return F.cross_entropy(m(xb), yb)


print("\n── Diagnostic Report (Transformer Encoder) ──")
diag, findings = run_diagnostic_checkpoint(
    transformer_model,
    train_loader,
    _ce_loss,
    title="Transformer Encoder",
    train_losses=transformer_losses,
    show=False,
)
print_prescription_pad(findings, "Transformer Encoder")

# ══════ READING THE PRESCRIPTION PAD (key: see ex_1/01_standard_ae.py) ══════
# nn.TransformerEncoderLayer uses ReLU in its feed-forward block by
# default, so the dead-neuron check applies to those units. Residual
# connections plus LayerNorm give every layer a short gradient path —
# compare the gradient-flow reading with the LSTM in 03. Only training
# loss is passed in; the validation-accuracy curve below shows whether
# more epochs would help.
# ══════════════════════════════════════════════════════════════════

# train_model kept the epoch with the best VALIDATION accuracy (a holdout
# carved from the training split); the test split is measured once, here.
transformer_test_acc = evaluate_accuracy(transformer_model, test_t, test_y)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    len(transformer_losses) == EPOCHS_SCRATCH
), "Transformer should train for all epochs"
assert (
    transformer_test_acc > 0.60
), f"Transformer should reach >60% test accuracy, got {transformer_test_acc:.3f}"
# INTERPRETATION: The Transformer processes all tokens in parallel and uses
# self-attention to capture long-range dependencies. On AG News headlines,
# it can directly connect "tech" at position 1 with "stocks" at position 8
# without propagating through every intermediate token. This architectural
# advantage becomes more pronounced on longer documents.
print(f"\n  Transformer: best validation acc {max(transformer_accs):.3f} -> test acc {transformer_test_acc:.3f}")
print("\n--- Checkpoint 3 passed --- Transformer trained on AG News\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Multi-head attention patterns on sample headline
# ════════════════════════════════════════════════════════════════════════
print("\n== Visualising multi-head attention patterns ==")
transformer_model.eval()
sample_texts = test_df["text"].to_list()[:3]
sample_idx = torch.tensor(
    [text_to_indices(t, vocab, MAX_LEN) for t in sample_texts],
    dtype=torch.long,
    device=DEVICE,
)

def encoder_attention(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Per-head attention weights of the TRAINED first encoder layer.

    nn.TransformerEncoderLayer (post-norm, the default) feeds its input
    straight into self_attn, so we rebuild that input (embedding +
    positional encoding) and ask the layer's own attention module for its
    weights. Returns (batch, n_heads, seq, seq); padded keys get weight 0.
    """
    model.eval()
    pad_mask = tokens == 0
    x = model.posenc(model.embed(tokens))
    first_layer = model.encoder.layers[0]
    # TODO: Call first_layer.self_attn as SELF-attention on x, masking the
    #   padded keys, and ask for per-head weights (see the need_weights and
    #   average_attn_weights arguments of nn.MultiheadAttention.forward).
    _, weights = ____
    return weights


# The heatmaps must come from the layer that was TRAINED: a fresh
# EducationalMultiHead here would show random projections.
with torch.no_grad():
    attn_weights = encoder_attention(transformer_model, sample_idx[:1])  # (1, 4, seq, seq)

words = sample_texts[0].lower().split()[:MAX_LEN]
word_labels = words + ["<pad>"] * (MAX_LEN - len(words))

# Visualise each head's attention pattern
for head_idx in range(min(4, attn_weights.shape[1])):
    attn_np = attn_weights[0, head_idx].cpu().numpy()
    fig = create_attention_heatmap(
        attn_np,
        word_labels,
        title=f"Attention Head {head_idx} on: '{sample_texts[0][:50]}...'",
        max_tokens=12,
    )
    fig.write_html(f"ex_4_2_head_{head_idx}_attention.html")

print(f"  Saved 4 attention head heatmaps (ex_4_2_head_0..3_attention.html)")
print(f"  Compare the four heads: do they spread attention differently?")
print(f"  (Specialisation is something to LOOK FOR, not assume: heads in a")
print(f"  small model trained for a few epochs often look alike.)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert attn_weights.shape == (
    1,
    4,
    MAX_LEN,
    MAX_LEN,
), "Should have 4 heads of attention"
print("\n--- Checkpoint 4 passed --- multi-head attention visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: Routing Regulatory Filings (and the Label-Space Trap)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A financial regulator's compliance team receives thousands of
# filings a month and wants each one routed by the regulation it concerns:
#   - Banking Act (Cap. 19)
#   - Securities and Futures Act (Cap. 289)
#   - Payment Services Act 2019
#   - Insurance Act (Cap. 142)
#
# The Transformer you trained answers a different question: its labels
# are AG News topics (World, Sports, Business, Sci/Tech). Run on finance
# headlines it can only say "Business" or similar — never "Insurance
# Act". Routing by regulation needs the same architecture trained on
# filings labelled by Act. What this section CAN show is how to inspect
# which words the trained attention focused on — with the caveat that
# attention weights are a view into the model, not a faithful,
# audit-grade explanation of its decision.
#
# BUSINESS VALUE (illustrative assumptions): at 15-20 minutes of manual
# triage per filing and ~3,000 filings/month, first-pass routing costs
# 750-1,000 officer-hours a month — savings that only exist once a
# classifier is trained on the regulator's own routing labels.
print("\n== Application: routing filings with a news-topic model ==")

financial_headlines = [
    "Banks report higher profits amid rising interest rates",
    "New technology startups attract venture capital funding",
    "Stock market volatility increases as trade tensions rise",
    "Sports betting companies face new regulatory scrutiny",
    "Insurance companies adapt to climate change risks",
]

# TODO: Classify financial headlines with the trained transformer
# - fin_idx: token-id tensor (long, on DEVICE) built with text_to_indices
#   for every headline -> (n_headlines, MAX_LEN)
# - fin_logits / fin_probs: model output and its softmax over classes
# - fin_preds: the predicted class id per headline, as a Python list
transformer_model.eval()
with torch.no_grad():
    fin_idx = ...  # YOUR CODE HERE
    fin_logits = ...  # YOUR CODE HERE
    fin_probs = ...  # YOUR CODE HERE
    fin_preds = ...  # YOUR CODE HERE

print(f"\n  Finance headlines through the AG News Transformer (topics, not Acts):")
print(f"  {'Headline':<55} {'Topic':<12} {'Confidence':>10}")
print("  " + "-" * 79)
for text, pred, probs in zip(financial_headlines, fin_preds, fin_probs.cpu().tolist()):
    cls_name = CLASS_NAMES[pred]
    confidence = max(probs)
    print(f"  {text[:53]:<55} {cls_name:<12} {confidence:>10.1%}")

# TODO: Attention-based explanation for the first document: trained
#   first-layer attention (encoder_attention), averaged across heads.
with torch.no_grad():
    fin_attn = ____
    avg_attn = ____

fin_words = financial_headlines[0].lower().split()[:MAX_LEN]
fin_labels = fin_words + ["<pad>"] * (MAX_LEN - len(fin_words))
token_importance = avg_attn[: len(fin_words), : len(fin_words)].sum(axis=0)
token_importance = token_importance / token_importance.max()

print(f"\n  Attention-based explanation for: '{financial_headlines[0]}'")
print(f"  Token importance (which words drive the classification):")
for word, imp in sorted(zip(fin_words, token_importance), key=lambda x: -x[1])[:5]:
    bar = "#" * int(imp * 20)
    print(f"    {word:<15} {imp:.3f} {bar}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(fin_preds) == len(financial_headlines), "Should classify all headlines"
# INTERPRETATION: Every prediction above is a news topic, so none of them
# routes a filing to a regulation team. The token-importance list shows
# where the trained first layer's attention went; treat it as a debugging
# view, not as an explanation a regulator could audit (attention weights
# are not guaranteed to reflect what drove the output).
print("\n--- Checkpoint 5 passed --- routing check complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Transformer Encoder")
print("=" * 70)
print(
    f"""
  [x] Built multi-head attention wrapping the from-scratch attention kernel
  [x] Explained how different heads capture different relationship types
  [x] Implemented sinusoidal positional encoding (word order for transformers)
  [x] Built a full TransformerClassifier with nn.TransformerEncoder
  [x] Trained on full AG News (120K headlines), test acc: {transformer_test_acc:.1%}
  [x] Visualised per-head attention patterns
  [x] Checked a regulatory-routing use case against the model's label space

  KEY INSIGHT:
    Multi-head attention is like having multiple specialists read the same
    document simultaneously. One head notices syntax, another notices
    entities, another notices sentiment. Together they capture a richer
    understanding than any single attention computation could.

  Next: In 03_lstm_baseline.py, you'll build an LSTM baseline to see
  exactly what the Transformer's attention mechanism buys us compared
  to sequential processing.
"""
)
