# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.3: LSTM Baseline for Comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain why baselines are essential for evaluating new architectures
#   - Build a bidirectional LSTM text classifier for fair comparison
#   - Contrast sequential (LSTM) vs parallel (Transformer) processing
#   - Train the LSTM on the same data and compare training dynamics
#   - Check whether a trained classifier's LABELS fit a new business task
#     (airline feedback routing) before deploying it
#
# PREREQUISITES: ex_4/01_self_attention_from_scratch.py
# ESTIMATED TIME: ~20 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_4 import (
    CLASS_NAMES,
    DEVICE,
    EPOCHS_SCRATCH,
    MAX_LEN,
    build_vocab,
    load_ag_news,
    prepare_dataloaders,
    setup_engines,
    text_to_indices,
    evaluate_accuracy,
    train_model,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why We Need a Baseline
# ════════════════════════════════════════════════════════════════════════
# When evaluating a new architecture (like the Transformer), you need a
# fair comparison point. Without a baseline, you cannot tell whether the
# Transformer's accuracy comes from:
#   (a) The attention mechanism itself, or
#   (b) Having more parameters, better hyperparameters, or more training.
#
# The LSTM is the strongest pre-Transformer baseline for text. It processes
# tokens sequentially, maintaining a hidden state that accumulates context.
# The key differences:
#
#   LSTM (sequential):
#     - Processes tokens one at a time: O(n) sequential steps
#     - Information flows through a hidden state "bottleneck"
#     - Bidirectional LSTM reads forward AND backward, doubling the context
#     - Cannot be parallelised across positions during training
#
#   Transformer (parallel):
#     - Processes all tokens simultaneously: O(1) depth (but O(n^2) attention)
#     - Every token directly attends to every other token
#     - Positional encoding provides word order information
#     - Fully parallelisable during training (much faster on GPU)
#
# By training both on the same data with similar parameter counts, we
# isolate the architectural difference and measure what attention buys.
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


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build: Bidirectional LSTM Text Classifier
# ════════════════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification.

    Architecture: embedding -> bidirectional LSTM -> mean pool -> classifier.

    Bidirectional processing reads the sequence both forward and backward,
    giving each token context from both directions. This partially addresses
    the LSTM's sequential limitation, but information must still propagate
    through the hidden state chain -- unlike attention, which is direct.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head_drop = nn.Dropout(dropout)
        # Bidirectional doubles the hidden dimension
        self.head = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        lstm_out, _ = self.lstm(x)  # (B, L, 2*H)
        # Mean pool over non-pad positions
        pad_mask = tokens == 0
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        lstm_out = lstm_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = lstm_out.sum(dim=1) / lengths
        return self.head(self.head_drop(pooled))


# ── Checkpoint 1 ─────────────────────────────────────────────────────
lstm_test = LSTMClassifier(vocab_size=100).to(DEVICE)
dummy_tokens = torch.randint(0, 100, (2, MAX_LEN), device=DEVICE)
lstm_out = lstm_test(dummy_tokens)
assert lstm_out.shape == (2, 4), "LSTM should output (batch, 4 classes)"
param_count = sum(p.numel() for p in lstm_test.parameters())
print(f"\n  LSTMClassifier output shape: {tuple(lstm_out.shape)} (expected (2, 4))")
print(f"  Parameter count: {param_count:,}")
print("\n--- Checkpoint 1 passed --- LSTM architecture ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train: LSTM on full AG News with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
print("\n== Training LSTM baseline on full AG News ==")
lstm_model = LSTMClassifier(
    vocab_size=len(vocab), embed_dim=128, hidden_dim=128, n_layers=2, n_classes=4
)
lstm_losses, lstm_accs = train_model(
    lstm_model,
    "lstm_baseline",
    train_loader,
    val_loader,
    tracker,
    exp_name,
    epochs=EPOCHS_SCRATCH,
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — LSTM baseline (contrast with the Transformer in 02)
# ══════════════════════════════════════════════════════════════════
# Same data, same training loop as 02 — so differences between the two
# pads come from the architecture.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _ce_loss(m, batch):
    """Cross-entropy on one (token_ids, labels) batch."""
    xb, yb = batch
    return F.cross_entropy(m(xb), yb)


print("\n── Diagnostic Report (LSTM baseline) ──")
diag, findings = run_diagnostic_checkpoint(
    lstm_model,
    train_loader,
    _ce_loss,
    title="LSTM baseline",
    train_losses=lstm_losses,
    show=False,
)
print_prescription_pad(findings, "LSTM baseline")

# ══════ READING THE PRESCRIPTION PAD (key: see ex_1/01_standard_ae.py) ══════
# Compare with 02: the LSTM must carry information across MAX_LEN (40) tokens
# step by step, the Transformer looks at all positions at once. The
# pad reads per-layer health (weight_ih / weight_hh / head), not
# per-timestep decay, so judge the architectures mainly on the
# accuracy and speed comparisons below.
# ══════════════════════════════════════════════════════════════════

# train_model kept the epoch with the best VALIDATION accuracy (a holdout
# carved from the training split); the test split is measured once, here.
lstm_test_acc = evaluate_accuracy(lstm_model, test_t, test_y)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(lstm_losses) == EPOCHS_SCRATCH, "LSTM should train for all epochs"
assert (
    lstm_test_acc > 0.60
), f"LSTM should reach >60% test accuracy, got {lstm_test_acc:.3f}"
# INTERPRETATION: The LSTM provides a strong baseline. On short headlines
# (avg ~10 words), the LSTM's sequential bottleneck isn't as severe as it
# would be on longer documents. The real gap between LSTM and Transformer
# widens as sequence length increases -- on 512-token documents, the
# Transformer's direct attention outperforms LSTM by a wider margin.
print(f"\n  LSTM: best validation acc {max(lstm_accs):.3f} -> test acc {lstm_test_acc:.3f}")
print(f"  LSTM final loss: {lstm_losses[-1]:.4f}")
print("\n--- Checkpoint 2 passed --- LSTM baseline trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: LSTM training dynamics
# ════════════════════════════════════════════════════════════════════════
# The LSTM's training curve reveals its learning dynamics. Unlike the
# Transformer, which can leverage parallel attention from epoch 1, the
# LSTM must learn to propagate information through the hidden state chain.
from shared.mlfp05.ex_4 import get_viz

viz = get_viz()
fig_lstm = viz.training_history(
    metrics={
        "LSTM train_loss": lstm_losses,
        "LSTM val_accuracy": lstm_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig_lstm.write_html("ex_4_3_lstm_training_curves.html")
print("  LSTM training curves saved to ex_4_3_lstm_training_curves.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(lstm_accs) == EPOCHS_SCRATCH, "Should have accuracy for each epoch"
# INTERPRETATION: The LSTM learning curve typically shows:
#   - Rapid initial learning (epochs 1-3): learns common word-class associations
#   - Slower improvement (epochs 4-6): refining contextual understanding
#   - Plateau (epochs 7-8): limited by the sequential bottleneck
# The Transformer, by contrast, often shows faster initial convergence
# because attention provides immediate access to all positions.
print("\n--- Checkpoint 3 passed --- LSTM training dynamics visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Can the Baseline Route Airline Feedback?
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore-based airline collects thousands of customer
# reviews a month and wants each one routed by topic — service, seat,
# delay, food, in-flight tech — to the right operations team.
#
# The LSTM you trained is a baseline for AG News: its four labels are
# World, Sports, Business and Sci/Tech. A classifier can only answer the
# question its labels asked, so it cannot output "Delay" or "Food" — it
# maps every review onto a NEWS topic, however confident it looks. The
# table below puts the routing label you WANTED next to the label the
# model CAN give. Real routing needs the same training recipe on
# reviews labelled with the airline's own topics.
#
# WHY THE BASELINE STILL MATTERS: once routing labels exist, an LSTM
# baseline sets the accuracy floor a Transformer must beat to justify
# its extra compute — the comparison ex_4/05 makes on AG News.
print("\n== Application: airline feedback vs a news-topic baseline ==")

airline_reviews = [
    "World class service from cabin crew on long haul flight",
    "New business class seat design wins innovation award",
    "Flight delayed three hours due to technical issues at the airport",
    "Award winning food menu designed by celebrity chef",
    "Technology upgrade to in-flight entertainment system completed",
]
wanted_topics = ["Service", "Seat", "Delay", "Food", "Tech"]

lstm_model.eval()
with torch.no_grad():
    review_idx = torch.tensor(
        [text_to_indices(t, vocab, MAX_LEN) for t in airline_reviews],
        dtype=torch.long,
        device=DEVICE,
    )
    review_logits = lstm_model(review_idx)
    review_probs = F.softmax(review_logits, dim=-1)
    review_preds = review_logits.argmax(dim=-1).cpu().tolist()

print(f"\n  Airline reviews through the AG News LSTM:")
print(f"  {'Review':<55} {'Wanted':<8} {'Model says':<12} {'Confidence':>10}")
print("  " + "-" * 87)
for text, topic, pred, probs in zip(
    airline_reviews, wanted_topics, review_preds, review_probs.cpu().tolist()
):
    print(f"  {text[:53]:<55} {topic:<8} {CLASS_NAMES[pred]:<12} {max(probs):>10.1%}")
print("  None of the 'Model says' labels is a routing team: the label spaces differ.")

# Measure throughput on a batch of 128 sequences of random token ids
import time

lstm_model.eval()
batch_input = torch.randint(0, len(vocab), (128, MAX_LEN), device=DEVICE)
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(10):
        _ = lstm_model(batch_input)
    t1 = time.perf_counter()
    throughput = (128 * 10) / (t1 - t0)
    print(f"\n  LSTM throughput: {throughput:,.0f} sequences/second")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(review_preds) == len(airline_reviews), "Should classify all reviews"
assert all(0 <= p < len(CLASS_NAMES) for p in review_preds), "Outputs are AG News ids"
# BUSINESS IMPACT (illustrative assumptions): at ~5,000 reviews/month and
# 2-3 minutes of manual sorting each, routing costs 167-250 analyst hours
# a month. That saving only materialises after a classifier is trained
# on the airline's own routing labels; this news-topic model saves none.
print("\n--- Checkpoint 4 passed --- airline feedback check complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — LSTM Baseline")
print("=" * 70)
print(
    f"""
  [x] Understood why baselines are essential for fair evaluation
  [x] Built a bidirectional LSTM text classifier
  [x] Contrasted sequential (LSTM) vs parallel (Transformer) processing
  [x] Trained on full AG News (120K headlines), test acc: {lstm_test_acc:.1%}
  [x] Measured inference throughput for production sizing
  [x] Checked a routing use case against the model's label space

  KEY INSIGHT:
    The LSTM is a strong baseline, not a strawman. On short sequences
    (headlines, tweets), it's competitive with transformers. The gap
    widens on longer documents where the sequential bottleneck hurts.
    Always establish your baseline BEFORE claiming your new architecture
    is "better" -- the margin matters more than the absolute number.

  Next: In 04_bert_finetuning.py, you'll see what happens when you
  combine the Transformer architecture with massive pre-training.
  BERT doesn't just learn from your 120K headlines -- it brings
  knowledge from billions of words of pre-training.
"""
)
