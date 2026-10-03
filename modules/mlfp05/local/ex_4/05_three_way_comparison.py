# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.5: Three-Way Comparison + ONNX Export
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Compare LSTM vs Transformer vs BERT side by side on the same dataset
#   - Explain the accuracy hierarchy (BERT >> Transformer > LSTM) and why
#   - Register all models in the ModelRegistry with versioned metrics
#   - Export the best model to ONNX for portable deployment
#   - Visualise training curves for all three architectures
#   - Interpret model predictions with attention heatmaps
#
# PREREQUISITES: All previous ex_4 files (01-04).
# ESTIMATED TIME: ~25 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_4 import (
    BERT_BATCH_SIZE,
    BERT_MAX_LEN,
    BERT_MODEL_NAME,
    CLASS_NAMES,
    DEVICE,
    EPOCHS_SCRATCH,
    MAX_LEN,
    build_vocab,
    create_attention_heatmap,
    get_viz,
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
# THEORY — The Architecture Hierarchy
# ════════════════════════════════════════════════════════════════════════
# This exercise is the payoff of the entire Exercise 4 sequence. We've
# built three architectures with fundamentally different approaches:
#
#   1. LSTM (sequential, no pre-training):
#      Processes tokens one at a time through a hidden state. No
#      pre-trained knowledge -- learns everything from our 120K headlines.
#      The sequential bottleneck limits long-range dependency capture.
#
#   2. Transformer (parallel attention, no pre-training):
#      Processes all tokens simultaneously via self-attention. Same
#      training data as LSTM, but the attention mechanism provides
#      direct access to all positions. Still learns from scratch.
#
#   3. BERT (parallel attention + pre-training):
#      Same architecture as the Transformer, but starts with pre-trained
#      weights from billions of words. Fine-tuning adapts this vast
#      language knowledge to our specific task.
#
# The comparison isolates two variables:
#   LSTM -> Transformer: the value of ATTENTION (parallel vs sequential)
#   Transformer -> BERT: the value of PRE-TRAINING (scratch vs transfer)
#
# Together, these reveal why modern NLP is dominated by pre-trained
# transformers: attention + pre-training is the winning combination.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Train all three models (reusing architectures from 02-04)
# ════════════════════════════════════════════════════════════════════════
train_df, test_df = load_ag_news()
vocab = build_vocab(train_df["text"].to_list())
train_loader, val_loader, train_t, train_y, test_t, test_y = prepare_dataloaders(
    train_df, test_df, vocab
)
conn, tracker, exp_name, registry, has_registry, bridge = setup_engines()

# --- Model Architectures (defined here for standalone execution) ---


# TODO: Define LSTMClassifier — bidirectional LSTM for text classification
# Hint: Same architecture as 03_lstm_baseline.py
# - __init__: embed, lstm (bidirectional), head_drop, head (hidden_dim * 2 -> n_classes)
# - forward: embed -> lstm -> mean pool over non-pad -> dropout -> head
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification (same as 03_lstm_baseline)."""

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
        ...  # YOUR CODE HERE — nn.Embedding, nn.LSTM(bidirectional=True), nn.Dropout, nn.Linear

    def forward(
        self, tokens: torch.Tensor
    ) -> torch.Tensor: ...  # YOUR CODE HERE — embed -> lstm -> mean pool -> head


# TODO: Define EducationalMultiHead — multi-head attention wrapper
# Hint: Same architecture as 02_transformer_encoder.py
class EducationalMultiHead(nn.Module):
    """Multi-head attention (same as 02_transformer_encoder)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq, d = x.shape
        # TODO: Compute QKV, split heads, apply attention, concatenate, project
        # Hint: (b, seq, 3*d_model) -> q, k, v of (b, seq, n_heads, d_k)
        #   -> fold heads into batch (b*n_heads, seq, d_k) -> attention
        #   -> unfold to (b, seq, d_model) -> self.proj. Return (output,
        #   weights reshaped to (b, n_heads, seq, seq)).
        ...  # YOUR CODE HERE


# TODO: Define PositionalEncoding — sinusoidal positional encoding
# Hint: Same as 02_transformer_encoder.py
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as 02_transformer_encoder)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # TODO: sin(position * div) into the even feature columns,
        #   cos(position * div) into the odd ones.
        ...  # YOUR CODE HERE
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# TODO: Define TransformerClassifier — full Transformer encoder classifier
# Hint: Same as 02_transformer_encoder.py
class TransformerClassifier(nn.Module):
    """Transformer encoder classifier (same as 02_transformer_encoder)."""

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
        # TODO: Build architecture — embed, posenc, emb_drop, encoder, head_drop, head
        ...  # YOUR CODE HERE

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # TODO: embed -> posenc -> dropout -> encoder(with pad mask) -> mean pool -> head
        ...  # YOUR CODE HERE


# --- LSTM ---
print("\n== Training LSTM baseline ==")
# TODO: Create and train LSTM model
# Hint: same configuration and train_model call as 03_lstm_baseline.py
#   (embed/hidden 128, 2 layers, 4 classes, run name "lstm_baseline").
lstm_model = ...  # YOUR CODE HERE
lstm_losses, lstm_accs = ...  # YOUR CODE HERE

# --- Transformer ---
print("\n== Training Transformer ==")
# TODO: Create and train Transformer model
# Hint: same configuration and train_model call as 02_transformer_encoder.py
#   (d_model 128, 4 heads, 3 layers, 4 classes, run name "transformer").
transformer_model = ...  # YOUR CODE HERE
transformer_losses, transformer_accs = ...  # YOUR CODE HERE

# --- BERT ---
print(f"\n== Fine-tuning {BERT_MODEL_NAME} ==")
from transformers import BertTokenizer, BertForSequenceClassification

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4
).to(DEVICE)

# TODO: Freeze lower 8 of 12 layers
# Hint: as in 04_bert_finetuning.py — read the layer index out of each
#   "bert.encoder.layer.<k>..." parameter name; freeze k < 8 and the embeddings.
for name, param in bert_model.named_parameters():
    ...  # YOUR CODE HERE

trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in bert_model.parameters())


def tokenise_for_bert(
    texts: list[str], max_len: int = BERT_MAX_LEN
) -> tuple[torch.Tensor, torch.Tensor]:
    encoding = bert_tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


print("  Tokenising for BERT...")
bert_train_ids, bert_train_mask = tokenise_for_bert(train_df["text"].to_list())
bert_test_ids, bert_test_mask = tokenise_for_bert(test_df["text"].to_list())
bert_train_y = torch.tensor(train_df["label"].to_list(), dtype=torch.long)
bert_test_y = torch.tensor(test_df["label"].to_list(), dtype=torch.long)

bert_train_loader = DataLoader(
    TensorDataset(
        bert_train_ids.to(DEVICE), bert_train_mask.to(DEVICE), bert_train_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE,
    shuffle=True,
)
bert_test_loader = DataLoader(
    TensorDataset(
        bert_test_ids.to(DEVICE), bert_test_mask.to(DEVICE), bert_test_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE,
)


async def train_bert_async(model, train_loader, test_loader, epochs=3, lr=2e-5):
    # TODO: Implement BERT training loop with ExperimentTracker
    # Hint: AdamW over trainable parameters only (lr, weight decay 0.01);
    #   the LinearLR scheduler and tracker run below are provided.
    optimizer = ...  # YOUR CODE HERE
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=epochs,
    )
    train_losses, test_accs = [], []

    async with tracker.track(experiment=exp_name, run_name="bert_finetune") as run:
        await run.log_params(
            {
                "model_type": "bert_finetune",
                "base_model": BERT_MODEL_NAME,
                "epochs": str(epochs),
                "lr": str(lr),
                "frozen_layers": "0-7",
                "trainable_params": str(trainable),
                "dataset_size": str(len(train_loader.dataset)),
            }
        )
        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch_idx, (ids, mask, labels) in enumerate(train_loader):
                # TODO: Forward + backward + step
                # Hint: labels in -> outputs.loss out; clip grad norm to 1.0;
                #   record each batch loss in batch_losses.
                ...  # YOUR CODE HERE
                if (batch_idx + 1) % 500 == 0:
                    print(
                        f"    batch {batch_idx+1}/{len(train_loader)}  loss={np.mean(batch_losses[-500:]):.4f}"
                    )
            scheduler.step()
            epoch_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                correct = total_count = 0
                for ids, mask, labels in test_loader:
                    # TODO: Get predictions and accumulate accuracy
                    #   (argmax of .logits; update correct and total_count).
                    ...  # YOUR CODE HERE
                acc = correct / total_count
                test_accs.append(acc)

            await run.log_metrics(
                {"train_loss": epoch_loss, "test_accuracy": acc}, step=epoch + 1
            )
            print(
                f"  [BERT] epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}  test_acc={acc:.3f}"
            )

        await run.log_metrics(
            {"final_test_accuracy": test_accs[-1], "final_train_loss": train_losses[-1]}
        )
    return train_losses, test_accs


bert_losses, bert_accs = asyncio.run(
    train_bert_async(bert_model, bert_train_loader, bert_test_loader, epochs=3)
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — comparative Prescription Pad for all 3
# ══════════════════════════════════════════════════════════════════
# Same instruments, three architectures, same data. Probes come from the
# training loaders (the probe runs in train mode).
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _ce_loss(m, batch):
    """Cross-entropy on one (token_ids, labels) batch."""
    xb, yb = batch
    return F.cross_entropy(m(xb), yb)


def _bert_loss(m, ids, mask, labels):
    return m(input_ids=ids, attention_mask=mask, labels=labels).loss


def _bert_adapter(batch):
    # BERT batches are (ids, mask, labels), passed to the loss as three args
    return batch[0], batch[1], batch[2]


for _title, _model, _loader, _loss, _hist, _adapter, _n in [
    ("LSTM", lstm_model, train_loader, _ce_loss, lstm_losses, None, 8),
    ("Transformer", transformer_model, train_loader, _ce_loss, transformer_losses, None, 8),
    ("BERT fine-tune", bert_model, bert_train_loader, _bert_loss, bert_losses, _bert_adapter, 4),
]:
    print(f"\n── Diagnostic Report ({_title}) ──")
    _diag, _findings = run_diagnostic_checkpoint(
        _model,
        _loader,
        _loss,
        title=f"{_title} (3-way comparison)",
        n_batches=_n,
        train_losses=_hist,
        batch_adapter=_adapter,
        show=False,
    )
    print_prescription_pad(_findings, f"{_title} (3-way comparison)")

# ══════ READING THE THREE PADS (key: see ex_1/01_standard_ae.py) ══════
# Line the three pads up. They read optimisation health per layer and
# the training-loss trend; the accuracy, size and speed table below is
# what ranks the models. BERT's frozen layers 0-7 receive no gradient
# by design. Readings you can tie to the architecture (the LSTM's
# recurrent weights vs the Transformer's residual stack) are worth
# writing down; differences you cannot explain are worth re-running.
# ══════════════════════════════════════════════════════════════════

# From-scratch models: the validation-selected checkpoint, measured once
# on the test split. BERT: its per-epoch test numbers were only monitored
# (no selection), so report the final epoch.
lstm_test_acc = evaluate_accuracy(lstm_model, test_t, test_y)
transformer_test_acc = evaluate_accuracy(transformer_model, test_t, test_y)
bert_test_acc = bert_accs[-1]

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert lstm_test_acc > 0.60, f"LSTM should exceed 60%, got {lstm_test_acc:.3f}"
assert (
    transformer_test_acc > 0.60
), f"Transformer should exceed 60%, got {transformer_test_acc:.3f}"
assert bert_test_acc > 0.85, f"BERT should exceed 85%, got {bert_test_acc:.3f}"
print("\n--- Checkpoint 1 passed --- all three models trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Visualise: Side-by-side comparison table + training curves
# ════════════════════════════════════════════════════════════════════════
# TODO: Build results dictionary and print comparison table
# Hint: keys "LSTM", "Transformer", "BERT (fine-tuned)"; each value holds
#   "test_acc" (the *_test_acc above), "final_loss" (last training loss) and
#   "params" (total parameter count; BERT's is total_params).
results = ...  # YOUR CODE HERE

print("\n== 3-Way Model Comparison on AG News ==")
print(f"{'Model':<20} {'Best Acc':>10} {'Final Loss':>12} {'Params':>12}")
print("-" * 56)
for name, r in results.items():
    print(
        f"{name:<20} {r['test_acc']:>10.3f} {r['final_loss']:>12.4f} {r['params']:>12,}"
    )

# TODO: Create training curves comparison chart
# Hint: viz.training_history with all 6 series (loss and val accuracy for
#   each model) against "Epoch"; save as HTML to ex_4_5_training_curves.html.
viz = get_viz()
fig_curves = ...  # YOUR CODE HERE
...  # YOUR CODE HERE
print("\nTraining curves saved to ex_4_5_training_curves.html")

# TODO: Sample predictions from all three models
# Hint: tokenise sample texts, run through each model, compare predictions
sample_texts = test_df["text"].to_list()[:5]
sample_true = test_df["label"].to_list()[:5]
sample_idx = torch.tensor(
    [text_to_indices(t, vocab, MAX_LEN) for t in sample_texts],
    dtype=torch.long,
    device=DEVICE,
)

transformer_model.eval()
lstm_model.eval()
bert_model.eval()
with torch.no_grad():
    transformer_preds = (
        ...
    )  # YOUR CODE HERE — predicted class ids as a Python list
    lstm_preds = (
        ...
    )  # YOUR CODE HERE
    bert_sample_ids, bert_sample_mask = tokenise_for_bert(sample_texts)
    bert_preds = (
        ...
    )  # YOUR CODE HERE — BERT takes the bert_sample_* tensors (on DEVICE)

print(f"\n== Sample Predictions (all 3 models) ==")
print(f"{'Headline':<50} {'True':<10} {'LSTM':<10} {'Trans':<10} {'BERT':<10}")
print("-" * 90)
for i, text in enumerate(sample_texts):
    t = CLASS_NAMES[sample_true[i]]
    l = CLASS_NAMES[lstm_preds[i]]
    tr = CLASS_NAMES[transformer_preds[i]]
    b = CLASS_NAMES[bert_preds[i]]
    print(f"{text[:48]:<50} {t:<10} {l:<10} {tr:<10} {b:<10}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["test_acc"])
assert best_model_name == "BERT (fine-tuned)", (
    f"Expected BERT to be the best model, but {best_model_name} won. "
    "Pre-trained models should dominate on standard NLP benchmarks."
)
assert Path("ex_4_5_training_curves.html").exists(), "Training curves should be saved"
# INTERPRETATION: The 3-way comparison reveals a clear hierarchy:
#   BERT >> Transformer > LSTM
# BERT dominates because it starts with pre-trained language knowledge.
# The Transformer edges out the LSTM because attention captures long-range
# dependencies without the information bottleneck of a fixed-size hidden
# state. The LSTM still does respectably -- it is a strong baseline.
print("\n--- Checkpoint 2 passed --- 3-way comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Register all models in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
async def register_all_models():
    """Register all three models in the ModelRegistry with metrics."""
    if not has_registry:
        print("  ModelRegistry not available -- skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}
    models_to_register = [
        ("m5_bert_agnews", bert_model.state_dict(), bert_test_acc, "bert_finetune"),
        (
            "m5_transformer_agnews",
            transformer_model.state_dict(),
            transformer_test_acc,
            "transformer",
        ),
        ("m5_lstm_agnews", lstm_model.state_dict(), lstm_test_acc, "lstm_baseline"),
    ]

    # TODO: Register each model in the registry
    # Hint: the artifact is the pickled state_dict (bytes); register_model is
    #   async and takes a list of MetricSpec (at least test_accuracy). Store
    #   each returned version in model_versions under its model_type.
    for name, state_dict, test_acc, model_type in models_to_register:
        ...  # YOUR CODE HERE

    return model_versions


model_versions = asyncio.run(register_all_models())

# ── Checkpoint 3 ─────────────────────────────────────────────────────
if has_registry:
    assert len(model_versions) == 3, "Should register all 3 models"
print("\n--- Checkpoint 3 passed --- models registered in ModelRegistry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Export best model (BERT) to ONNX via OnnxBridge
# ════════════════════════════════════════════════════════════════════════
# In production, the ModelRegistry stores the winning model. OnnxBridge
# exports it to ONNX format for portable deployment (any language, any
# runtime, no PyTorch dependency). This is how production ML pipelines
# separate training (Python) from serving (any language).
onnx_path = Path("ex_4_bert_agnews.onnx")
bert_model.eval()


class BertLogits(nn.Module):
    """Single-input view of the classifier for export: token ids in, logits
    out. The attention mask is rebuilt from the ids ([PAD] has id 0)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0).long()
        return self.model(input_ids=input_ids, attention_mask=mask).logits


# Export on the CPU, then move BERT back. The sample is a batch of TWO
# headlines, not one: the exporter traces with torch.export, which treats
# size-1 dimensions as constants and can freeze the batch size at 1.
bert_model.cpu()
export_model = BertLogits(bert_model).eval()
# TODO: Export export_model with OnnxBridge ("torch" framework) to
#   onnx_path, tracing with the first two test headlines' token ids.
export_result = ____

# Check the graph on real headlines. OnnxBridge.validate feeds float32
# arrays, so it cannot drive an int64 token-id graph — compare directly.
import onnxruntime as ort

session = ort.InferenceSession(str(onnx_path))
check_ids = bert_test_ids[:16]
# TODO: logits for check_ids from the ONNX session (its input name is
#   session.get_inputs()[0].name; feed a numpy array) and from export_model.
onnx_logits = ____
with torch.no_grad():
    torch_logits = ____
bert_model.to(DEVICE)
max_diff = float(np.abs(onnx_logits - torch_logits).max())
agreement = float((onnx_logits.argmax(axis=1) == torch_logits.argmax(axis=1)).mean())
print(
    f"  OnnxBridge.export: success={export_result.success} -> {onnx_path} "
    f"({onnx_path.stat().st_size // 1024:,} KB)"
)
print(f"  ONNX vs PyTorch on 16 test headlines: max |logit diff| = {max_diff:.2e}, "
      f"class agreement = {agreement:.0%}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert export_result.success, f"OnnxBridge export failed: {export_result.error_message}"
assert agreement == 1.0 and max_diff < 1e-3, "ONNX graph drifted from PyTorch"
# INTERPRETATION: The ModelRegistry gives you a versioned record of every
# model experiment. The ONNX export makes the model portable -- it can run
# on a server without PyTorch installed, in a mobile app, or in a browser
# via ONNX.js. This is how production ML pipelines separate training
# (Python) from serving (any language).
print("\n--- Checkpoint 4 passed --- ONNX export complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Attention heatmap from trained Transformer
# ════════════════════════════════════════════════════════════════════════
# The attention heatmap is the Transformer's "explanation" -- it shows
# which words the model attends to when classifying a headline.
transformer_model.eval()


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
    # TODO: same call as in 02_transformer_encoder.py — the trained layer's
    #   self_attn on x, padded keys masked, per-head weights returned.
    _, weights = ____
    return weights


# Head 0 of the TRAINED first encoder layer (a fresh attention module
# would only show random projections).
# TODO: attention weights for sample_idx[:1], then head 0 as a numpy array
with torch.no_grad():
    attn_weights = ____
    attn_np = ____

words = sample_texts[0].lower().split()[:MAX_LEN]
word_labels = words + ["<pad>"] * (MAX_LEN - len(words))

fig_attn = create_attention_heatmap(
    attn_np,
    word_labels,
    title=f"Transformer Attention on: '{sample_texts[0][:50]}...'",
    max_tokens=15,
)
fig_attn.write_html("ex_4_5_attention_heatmap.html")
print("Attention heatmap saved to ex_4_5_attention_heatmap.html")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert attn_np.shape[0] == MAX_LEN, "Attention heatmap should cover full sequence"
assert Path("ex_4_5_attention_heatmap.html").exists(), "Heatmap should be saved"
print("\n--- Checkpoint 5 passed --- visualisations complete\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — one-call diagnostics
# ════════════════════════════════════════════════════════════════════════
# This lesson built attention from scratch, a Transformer encoder, an
# LSTM baseline and a fine-tuned BERT. kailash-ml's
# run_diagnostic_checkpoint is the one call behind each pad: it hooks
# every layer, runs a few probe batches (no weight updates), replays the
# loss history and returns findings plus a DLDiagnostics session whose
# plot_training_dashboard() draws the loss, gradient and activation
# panels. (km.diagnose(model, kind="dl") on its own only builds an
# un-instrumented session, so it has nothing to report.) It does not
# replace the accuracy, latency and deployment comparisons above.
diag, findings = run_diagnostic_checkpoint(
    transformer_model,
    train_loader,
    _ce_loss,
    title="Transformer (close)",
    train_losses=transformer_losses,
    show=False,
)
print_prescription_pad(findings, "Transformer (close)")
dashboard = diag.plot_training_dashboard()  # Plotly figure: dashboard.show()


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Complete Exercise 4")
print("=" * 70)
print(
    f"""
  [x] Derived scaled dot-product attention with torch.einsum
  [x] Explained the 1/sqrt(d_k) factor (prevents softmax saturation)
  [x] Wrote a hand-rolled multi-head attention wrapping the scratch kernel
  [x] Built a TransformerClassifier with nn.TransformerEncoder
  [x] Built an LSTM baseline for fair comparison
  [x] Trained all 3 models on FULL AG News (120K headlines)
  [x] Fine-tuned BERT ({BERT_MODEL_NAME}) -- test acc: {bert_test_acc:.1%}
  [x] Visualised attention heatmaps (what the model "looks at")
  [x] Tracked every run with ExperimentTracker (params, per-epoch metrics)
  [x] Registered models in ModelRegistry with versioned metrics
  [x] Exported the fine-tuned model to ONNX for portable deployment

  KEY INSIGHT — The Attention Hierarchy:
    LSTM test acc:        {lstm_test_acc:.1%}  (sequential, no pre-training)
    Transformer test acc: {transformer_test_acc:.1%}  (parallel attention, no pre-training)
    BERT test acc:        {bert_test_acc:.1%}  (parallel attention + pre-training)

  Pre-training is the single biggest lever in NLP. The Transformer
  architecture enables it, but the pre-trained weights are what make
  BERT dominate. This is why modern NLP is "pre-train then fine-tune."

  Next: In Exercise 5, you'll build generative models (DCGAN + WGAN-GP)
  that CREATE new data instead of classifying existing data.
"""
)
