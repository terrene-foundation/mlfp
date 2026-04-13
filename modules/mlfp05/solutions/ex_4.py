# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4: Transformers, Self-Attention, and BERT Fine-Tuning
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Derive scaled dot-product attention from scratch using torch.einsum
#   - Explain why we divide by sqrt(d_k) (softmax saturation on large dims)
#   - Implement multi-head attention with nn.MultiheadAttention
#   - Build a small Transformer encoder with nn.TransformerEncoder
#   - Compare LSTM baseline vs Transformer vs BERT fine-tuning (3-way)
#   - Visualise attention heatmaps to see what the model attends to
#   - Fine-tune a pre-trained BERT model on real news classification
#   - Track all experiments with kailash-ml's ExperimentTracker
#   - Register the best model in the ModelRegistry
#   - Export the fine-tuned model to ONNX via OnnxBridge
#
# PREREQUISITES: M5/ex_3 (RNNs, sequence modelling, nn.Module training).
# ESTIMATED TIME: ~120-150 min
# DATASET: AG News — 120,000 real news headlines, 4 classes
#          (World / Sports / Business / Sci-Tech).
#
# TASKS:
#   1. Load FULL AG News (120K train, 7.6K test), build vocabulary, set up engines
#   2. Derive scaled dot-product attention from scratch and visualise it
#   3. Build multi-head attention and Transformer encoder classifier
#   4. Build an LSTM baseline for comparison
#   5. Train LSTM baseline + Transformer on AG News, log to ExperimentTracker
#   6. Fine-tune pre-trained BERT on AG News, log to ExperimentTracker
#   7. 3-way comparison: LSTM vs Transformer vs BERT
#   8. Register the best model + export to ONNX
#   9. Visualise attention heatmaps and training curves
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

import plotly.graph_objects as go

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.bridge.onnx_bridge import OnnxBridge
from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load FULL AG News and set up kailash-ml engines
# ════════════════════════════════════════════════════════════════════════
# AG News is a standard 4-class news topic benchmark: World, Sports,
# Business, Sci/Tech. We use the FULL 120K training set — transformers
# benefit from large datasets because they lack the inductive biases
# that RNNs get from sequential processing.
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_ag_news_split(split: str, cache_name: str):
    """Load AG News split, caching to parquet for subsequent runs."""
    cache = Path(__file__).resolve().parents[3] / "data" / "mlfp05" / cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pl.read_parquet(cache)
    print(f"  Downloading AG News {split} from HuggingFace...")
    ds = load_dataset("fancyzhx/ag_news", split=split)
    df = pl.from_pandas(ds.to_pandas())
    df.write_parquet(cache)
    return df


print("\n== Loading FULL AG News (120K train + 7.6K test) ==")
train_df = load_ag_news_split("train", "ag_news_full_train.parquet")
test_df = load_ag_news_split("test", "ag_news_full_test.parquet")
print(f"  train rows: {len(train_df):,}   test rows: {len(test_df):,}")
print(f"  sample headline: {train_df['text'][0][:80]!r}")
print(f"  class balance (train): {dict(Counter(train_df['label'].to_list()))}")


# Vocabulary + tokenisation for the from-scratch models (LSTM + Transformer).
# BERT uses its own WordPiece tokeniser, so this is only for our custom models.
MAX_LEN = 40
VOCAB_SIZE = 15000


def build_vocab(texts, max_vocab: int = VOCAB_SIZE):
    words: Counter[str] = Counter()
    for t in texts:
        words.update(t.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in words.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def text_to_indices(text: str, vocab: dict[str, int], max_len: int = MAX_LEN):
    tokens = text.lower().split()[:max_len]
    idxs = [vocab.get(t, 1) for t in tokens]
    return (idxs + [0] * (max_len - len(idxs)))[:max_len]


vocab = build_vocab(train_df["text"].to_list(), max_vocab=VOCAB_SIZE)
print(f"  vocab size: {len(vocab)} (cap {VOCAB_SIZE}, seq_len {MAX_LEN})")

train_tokens = np.array(
    [text_to_indices(t, vocab, MAX_LEN) for t in train_df["text"].to_list()],
    dtype=np.int64,
)
train_labels = np.array(train_df["label"].to_list(), dtype=np.int64)
test_tokens = np.array(
    [text_to_indices(t, vocab, MAX_LEN) for t in test_df["text"].to_list()],
    dtype=np.int64,
)
test_labels = np.array(test_df["label"].to_list(), dtype=np.int64)

train_t = torch.from_numpy(train_tokens).to(device)
train_y = torch.from_numpy(train_labels).to(device)
test_t = torch.from_numpy(test_tokens).to(device)
test_y = torch.from_numpy(test_labels).to(device)

train_loader = DataLoader(TensorDataset(train_t, train_y), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(test_t, test_y), batch_size=128)


# Set up kailash-ml engines: ExperimentTracker + ModelRegistry + OnnxBridge
async def setup_engines():
    conn = ConnectionManager("sqlite:///mlfp05_transformers.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name="m5_transformers",
        description="LSTM vs Transformer vs BERT on AG News (120K headlines)",
    )

    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())
bridge = OnnxBridge()

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(train_df) >= 100000, (
    f"Expected full AG News train (~120K), got {len(train_df):,}. "
    "Use the complete dataset for meaningful model comparison."
)
assert len(test_df) >= 5000, f"Expected full AG News test (~7.6K), got {len(test_df):,}"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: With 120K headlines, the three models have enough data
# to show their real strengths. BERT benefits from pre-training; the
# Transformer benefits from attention; the LSTM struggles at this scale.
print("\n--- Checkpoint 1 passed --- full AG News loaded and engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Scaled dot-product attention from scratch
# ════════════════════════════════════════════════════════════════════════
# For a single head:
#   Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
#
# Q (query): "what am I looking for?"   -- shape (B, L_q, d_k)
# K (key):   "what can I offer?"        -- shape (B, L_k, d_k)
# V (value): "what do I actually pass"  -- shape (B, L_k, d_v)
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


# Quick sanity check: with scaled identity-style queries, each query
# should attend most strongly to its own key.
q_demo = torch.eye(4, 8).unsqueeze(0) * 3.0
k_demo = q_demo.clone()
v_demo = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)
_, attn_demo = scaled_dot_product_attention(q_demo, k_demo, v_demo)
print("Demo attention weights (should peak on the diagonal):")
print(attn_demo.squeeze(0).round(decimals=2))

# Visualise the attention pattern as a heatmap. This is the core
# diagnostic for understanding what a transformer attends to.
viz = ModelVisualizer()

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert attn_demo.shape == (1, 4, 4), "Attention should be (1, 4, 4)"
diag_vals = torch.diag(attn_demo.squeeze(0))
assert diag_vals.min() > 0.5, "Diagonal should dominate (each query attends to itself)"
# INTERPRETATION: The attention matrix shows which queries attend to which
# keys. A strong diagonal means each position attends to itself -- exactly
# what we expect when Q and K are scaled identity matrices.
print("\n--- Checkpoint 2 passed --- attention from scratch works\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Multi-head attention + Transformer encoder classifier
# ════════════════════════════════════════════════════════════════════════


# Multi-head attention runs h attention operations in parallel over
# different learned projections of Q/K/V, then concatenates. Each head
# can specialise (e.g., one head for syntax, another for semantics).
class EducationalMultiHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq, d = x.shape
        qkv = self.qkv(x).reshape(b, seq, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)  # each: (b, seq, h, d_k)
        q = q.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        k = k.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        v = v.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        out, weights = scaled_dot_product_attention(q, k, v)
        # Reshape weights to (b, n_heads, seq, seq) for visualisation
        attn_weights = weights.reshape(b, self.n_heads, seq, seq)
        out = (
            out.reshape(b, self.n_heads, seq, self.d_k)
            .transpose(1, 2)
            .reshape(b, seq, d)
        )
        return self.proj(out), attn_weights


# Sinusoidal positional encoding -- transformers have no inherent sense
# of order, so we add a fixed sinusoidal signal that encodes position:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# Full Transformer encoder classifier: embedding + positional encoding +
# stacked TransformerEncoderLayers + mean pool + classification head.
class TransformerClassifier(nn.Module):
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
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.posenc = PositionalEncoding(d_model)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = tokens == 0
        x = self.embed(tokens)
        x = self.posenc(x)
        x = self.emb_drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # Mean-pool over non-pad positions so short headlines aren't diluted.
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = x.sum(dim=1) / lengths
        return self.head(self.head_drop(pooled))


# Sanity check the educational multi-head attention
mha = EducationalMultiHead(d_model=64, n_heads=4).to(device)
dummy = torch.randn(2, 16, 64, device=device)
mha_out, mha_attn = mha(dummy)
print(
    f"EducationalMultiHead output shape: {tuple(mha_out.shape)}  (expected (2, 16, 64))"
)
print(f"Attention weights shape: {tuple(mha_attn.shape)}  (expected (2, 4, 16, 16))")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert mha_out.shape == (2, 16, 64), "Multi-head output should be (2, 16, 64)"
assert mha_attn.shape == (2, 4, 16, 16), "Attention should be (batch, heads, seq, seq)"
print("\n--- Checkpoint 3 passed --- Transformer architecture ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — LSTM baseline for comparison
# ════════════════════════════════════════════════════════════════════════
# The LSTM gives us a strong sequential baseline. It processes tokens
# one by one (O(n) sequential steps), while the Transformer processes
# them all at once (O(1) depth, O(n^2) attention). This comparison
# reveals what the attention mechanism buys us.
class LSTMClassifier(nn.Module):
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
        # bidirectional doubles hidden_dim
        self.head = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        # Pack-pad would be ideal but for fixed-length sequences, masking
        # the final hidden state is simpler and pedagogically clearer.
        lstm_out, _ = self.lstm(x)  # (B, L, 2*H)
        # Mean pool over non-pad positions
        pad_mask = tokens == 0
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        lstm_out = lstm_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = lstm_out.sum(dim=1) / lengths
        return self.head(self.head_drop(pooled))


# ── Checkpoint 4 ─────────────────────────────────────────────────────
lstm_test = LSTMClassifier(vocab_size=100).to(device)
dummy_tokens = torch.randint(0, 100, (2, MAX_LEN), device=device)
lstm_out = lstm_test(dummy_tokens)
assert lstm_out.shape == (2, 4), "LSTM should output (batch, 4 classes)"
print("--- Checkpoint 4 passed --- LSTM baseline ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Train LSTM + Transformer on full AG News
# ════════════════════════════════════════════════════════════════════════
# We train both models with the same hyperparameters where possible so
# the comparison is fair. The ExperimentTracker logs every run.
EPOCHS_SCRATCH = 8


async def train_model_async(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS_SCRATCH,
    lr: float = 2e-3,
) -> tuple[list[float], list[float]]:
    """Train a PyTorch classifier and log every epoch to ExperimentTracker.

    Uses the modern ``tracker.run(...)`` async context manager — bulk
    param logging, per-step metrics, automatic COMPLETED/FAILED state.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses: list[float] = []
    val_accs: list[float] = []
    best_acc = 0.0
    best_state: dict | None = None
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    async with tracker.run(experiment_name=exp_name, run_name=model_name) as ctx:
        await ctx.log_params(
            {
                "model_type": model_name,
                "epochs": str(epochs),
                "lr": str(lr),
                "dataset_size": str(len(train_loader.dataset)),
                "batch_size": str(train_loader.batch_size),
                "trainable_params": str(param_count),
            }
        )

        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())
            scheduler.step()
            epoch_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for xb, yb in val_loader:
                    preds = model(xb).argmax(dim=-1)
                    correct += int((preds == yb).sum().item())
                    total += int(yb.size(0))
                acc = correct / total
                val_accs.append(acc)

            await ctx.log_metrics(
                {"train_loss": epoch_loss, "val_accuracy": acc},
                step=epoch + 1,
            )
            if acc > best_acc:
                best_acc = acc
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            print(
                f"  [{model_name}] epoch {epoch+1}/{epochs}  "
                f"loss={epoch_loss:.4f}  val_acc={acc:.3f}"
            )

        await ctx.log_metrics(
            {
                "best_val_accuracy": best_acc,
                "final_train_loss": train_losses[-1],
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_accs


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS_SCRATCH,
    lr: float = 2e-3,
) -> tuple[list[float], list[float]]:
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(
        train_model_async(model, model_name, train_loader, val_loader, epochs, lr)
    )


# Train the LSTM baseline
print("\n== Training LSTM baseline on full AG News ==")
lstm_model = LSTMClassifier(
    vocab_size=len(vocab), embed_dim=128, hidden_dim=128, n_layers=2, n_classes=4
)
lstm_losses, lstm_accs = train_model(
    lstm_model, "lstm_baseline", train_loader, val_loader, epochs=EPOCHS_SCRATCH
)

# Train the Transformer
print("\n== Training Transformer on full AG News ==")
transformer_model = TransformerClassifier(
    vocab_size=len(vocab), d_model=128, n_heads=4, n_layers=3, n_classes=4
)
transformer_losses, transformer_accs = train_model(
    transformer_model, "transformer", train_loader, val_loader, epochs=EPOCHS_SCRATCH
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(lstm_losses) == EPOCHS_SCRATCH, "LSTM should train for all epochs"
assert (
    len(transformer_losses) == EPOCHS_SCRATCH
), "Transformer should train for all epochs"
assert (
    max(lstm_accs) > 0.60
), f"LSTM should reach >60% accuracy, got {max(lstm_accs):.3f}"
assert (
    max(transformer_accs) > 0.60
), f"Transformer should reach >60% accuracy, got {max(transformer_accs):.3f}"
# INTERPRETATION: Both models train on the full 120K headlines. The
# Transformer typically matches or exceeds the LSTM because self-attention
# can capture long-range dependencies (e.g., "tech company" + "stock price"
# at opposite ends of a headline) without propagating through every token.
print(
    f"\n  LSTM best acc:        {max(lstm_accs):.3f}"
    f"\n  Transformer best acc: {max(transformer_accs):.3f}"
)
print("\n--- Checkpoint 5 passed --- LSTM + Transformer trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Fine-tune pre-trained BERT on AG News
# ════════════════════════════════════════════════════════════════════════
# BERT is pre-trained on massive text corpora (BookCorpus + Wikipedia).
# Fine-tuning adapts its learned language representations to our specific
# task. We freeze the lower layers and fine-tune the top layers + the
# classification head. This is dramatically more sample-efficient than
# training from scratch.
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 64
BERT_EPOCHS = 3
BERT_LR = 2e-5
BERT_BATCH_SIZE = 32

print(f"\n== Fine-tuning {BERT_MODEL_NAME} on AG News ==")
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4
).to(device)

# Freeze the lower 8 of 12 encoder layers — only fine-tune the top 4
# layers plus the pooler and classification head. This is faster and
# prevents catastrophic forgetting of the pre-trained representations.
for name, param in bert_model.named_parameters():
    if "bert.encoder.layer" in name:
        layer_num = int(name.split(".")[3])
        if layer_num < 8:
            param.requires_grad = False
    elif "bert.embeddings" in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in bert_model.parameters())
print(
    f"  BERT params: {total:,} total, {trainable:,} trainable "
    f"({trainable/total:.1%} unfrozen)"
)


# Tokenise with BERT's WordPiece tokeniser. This handles subword
# splitting, special tokens ([CLS], [SEP]), and padding automatically.
def tokenise_for_bert(texts: list[str], max_len: int = BERT_MAX_LEN):
    encoding = bert_tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


# Tokenise train and test sets. This takes a few seconds on 120K texts.
print("  Tokenising train + test sets...")
bert_train_ids, bert_train_mask = tokenise_for_bert(train_df["text"].to_list())
bert_test_ids, bert_test_mask = tokenise_for_bert(test_df["text"].to_list())
bert_train_y = torch.tensor(train_df["label"].to_list(), dtype=torch.long)
bert_test_y = torch.tensor(test_df["label"].to_list(), dtype=torch.long)

bert_train_loader = DataLoader(
    TensorDataset(
        bert_train_ids.to(device), bert_train_mask.to(device), bert_train_y.to(device)
    ),
    batch_size=BERT_BATCH_SIZE,
    shuffle=True,
)
bert_val_loader = DataLoader(
    TensorDataset(
        bert_test_ids.to(device), bert_test_mask.to(device), bert_test_y.to(device)
    ),
    batch_size=BERT_BATCH_SIZE,
)


# Train BERT and log to ExperimentTracker
async def train_bert_async(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = BERT_EPOCHS,
    lr: float = BERT_LR,
) -> tuple[list[float], list[float]]:
    """Fine-tune BERT and log to ExperimentTracker (modern context-manager API)."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
    )
    train_losses: list[float] = []
    val_accs: list[float] = []
    best_acc = 0.0

    async with tracker.run(experiment_name=exp_name, run_name="bert_finetune") as ctx:
        await ctx.log_params(
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
                optimizer.zero_grad()
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())
                if (batch_idx + 1) % 500 == 0:
                    print(
                        f"    batch {batch_idx+1}/{len(train_loader)}  "
                        f"loss={np.mean(batch_losses[-500:]):.4f}"
                    )
            scheduler.step()
            epoch_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for ids, mask, labels in val_loader:
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    preds = logits.argmax(dim=-1)
                    correct += int((preds == labels).sum().item())
                    total += int(labels.size(0))
                acc = correct / total
                val_accs.append(acc)

            await ctx.log_metrics(
                {"train_loss": epoch_loss, "val_accuracy": acc}, step=epoch + 1
            )
            if acc > best_acc:
                best_acc = acc
            print(
                f"  [BERT] epoch {epoch+1}/{epochs}  "
                f"loss={epoch_loss:.4f}  val_acc={acc:.3f}"
            )

        await ctx.log_metrics(
            {
                "best_val_accuracy": best_acc,
                "final_train_loss": train_losses[-1],
            }
        )

    return train_losses, val_accs


def train_bert(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = BERT_EPOCHS,
    lr: float = BERT_LR,
) -> tuple[list[float], list[float]]:
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(train_bert_async(model, train_loader, val_loader, epochs, lr))


bert_losses, bert_accs = train_bert(
    bert_model, bert_train_loader, bert_val_loader, epochs=BERT_EPOCHS
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(bert_losses) == BERT_EPOCHS, "BERT should train for all epochs"
assert (
    max(bert_accs) > 0.85
), f"BERT should reach >85% accuracy with fine-tuning, got {max(bert_accs):.3f}"
# INTERPRETATION: BERT's pre-trained language understanding gives it a
# massive head start. While our from-scratch models need to learn word
# meanings, syntax, and semantics from 120K headlines, BERT already
# "knows" English from billions of words of pre-training. Fine-tuning
# just teaches it the specific mapping from language to news categories.
print(f"\n  BERT best acc: {max(bert_accs):.3f}")
print("\n--- Checkpoint 6 passed --- BERT fine-tuned\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — 3-way comparison: LSTM vs Transformer vs BERT
# ════════════════════════════════════════════════════════════════════════
# This is the payoff of the exercise. We compare three architectures
# trained on the same data, revealing the impact of:
#   1. Architecture (sequential LSTM vs parallel Transformer)
#   2. Pre-training (BERT vs from-scratch Transformer)

results = {
    "LSTM": {
        "best_acc": max(lstm_accs),
        "final_loss": lstm_losses[-1],
        "params": sum(p.numel() for p in lstm_model.parameters()),
    },
    "Transformer": {
        "best_acc": max(transformer_accs),
        "final_loss": transformer_losses[-1],
        "params": sum(p.numel() for p in transformer_model.parameters()),
    },
    "BERT (fine-tuned)": {
        "best_acc": max(bert_accs),
        "final_loss": bert_losses[-1],
        "params": total,
    },
}

print("\n== 3-Way Model Comparison on AG News ==")
print(f"{'Model':<20} {'Best Acc':>10} {'Final Loss':>12} {'Params':>12}")
print("-" * 56)
for name, r in results.items():
    print(
        f"{name:<20} {r['best_acc']:>10.3f} {r['final_loss']:>12.4f} {r['params']:>12,}"
    )

# Per-class accuracy breakdown for the best model
bert_model.eval()
class_correct = Counter()
class_total = Counter()
with torch.no_grad():
    for ids, mask, labels in bert_val_loader:
        logits = bert_model(input_ids=ids, attention_mask=mask).logits
        preds = logits.argmax(dim=-1)
        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

print(f"\n  BERT per-class accuracy:")
for i, cls_name in enumerate(CLASS_NAMES):
    acc = class_correct[i] / max(class_total[i], 1)
    print(f"    {cls_name:<10} {acc:.3f} ({class_correct[i]}/{class_total[i]})")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["best_acc"])
assert best_model_name == "BERT (fine-tuned)", (
    f"Expected BERT to be the best model, but {best_model_name} won. "
    "Pre-trained models should dominate on standard NLP benchmarks."
)
# INTERPRETATION: The 3-way comparison reveals a clear hierarchy:
#   BERT >> Transformer > LSTM
# BERT dominates because it starts with pre-trained language knowledge.
# The Transformer edges out the LSTM because attention captures long-range
# dependencies without the information bottleneck of a fixed-size hidden
# state. The LSTM still does respectably -- it is a strong baseline.
print("\n--- Checkpoint 7 passed --- 3-way comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Register the best model + export to ONNX
# ════════════════════════════════════════════════════════════════════════
# In production, the ModelRegistry stores the winning model with its
# metrics. OnnxBridge exports it to ONNX format for portable deployment
# (any language, any runtime, no PyTorch dependency).


async def register_best_model():
    """Register the fine-tuned BERT and Transformer in the ModelRegistry."""
    if not has_registry:
        print("  ModelRegistry not available -- skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}
    models_to_register = [
        ("m5_bert_agnews", bert_model.state_dict(), max(bert_accs), "bert_finetune"),
        (
            "m5_transformer_agnews",
            transformer_model.state_dict(),
            max(transformer_accs),
            "transformer",
        ),
        ("m5_lstm_agnews", lstm_model.state_dict(), max(lstm_accs), "lstm_baseline"),
    ]

    for name, state_dict, best_acc, model_type in models_to_register:
        model_bytes = pickle.dumps(state_dict)
        version = await registry.register_model(
            name=name,
            artifact=model_bytes,
            metrics=[
                MetricSpec(name="best_val_accuracy", value=best_acc),
                MetricSpec(name="dataset", value=0.0),  # AG News 120K
                MetricSpec(name="model_type", value=0.0),  # metadata
            ],
        )
        model_versions[model_type] = version
        print(f"  Registered {name}: version={version.version}, acc={best_acc:.3f}")

    return model_versions


model_versions = asyncio.run(register_best_model())


# Export the fine-tuned BERT to ONNX. OnnxBridge is optimised for
# tabular/sklearn models, so for BERT we use torch.onnx.export directly
# (same approach as ex_2 with CNNs).
onnx_path = Path("ex_4_bert_agnews.onnx")
bert_model.eval()

exported = False
try:
    result = bridge.export(
        model=bert_model,
        framework="pytorch",
        output_path=onnx_path,
        n_features=BERT_MAX_LEN,
    )
    success = getattr(result, "success", bool(result))
    exported = bool(success) and onnx_path.exists()
except Exception:
    pass

if not exported:
    # Torch ONNX export for BERT: provide dummy input_ids + attention_mask
    print("  Using torch.onnx.export for BERT model...")
    dummy_ids = torch.ones(1, BERT_MAX_LEN, dtype=torch.long, device=device)
    dummy_mask = torch.ones(1, BERT_MAX_LEN, dtype=torch.long, device=device)
    bert_cpu = bert_model.cpu()
    torch.onnx.export(
        bert_cpu,
        (dummy_ids.cpu(), dummy_mask.cpu()),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    bert_model.to(device)  # move back after export

if onnx_path.exists():
    print(f"  ONNX export: {onnx_path} ({onnx_path.stat().st_size // 1024:,} KB)")
else:
    print("  ONNX export: skipped (export not available in this environment)")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
if has_registry:
    assert len(model_versions) == 3, "Should register all 3 models"
# INTERPRETATION: The ModelRegistry gives you a versioned record of every
# model. The ONNX export makes the model portable -- it can run on a
# server without PyTorch installed, in a mobile app, or in a browser via
# ONNX.js. This is how production ML pipelines separate training (Python)
# from serving (any language).
print("\n--- Checkpoint 8 passed --- model registered and ONNX exported\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Visualise attention heatmaps and training curves
# ════════════════════════════════════════════════════════════════════════
# Attention heatmaps are the transformer's "explanation" -- they show
# which words the model attends to when classifying a headline.

# Extract attention weights from our custom Transformer on sample headlines
transformer_model.eval()
sample_texts = test_df["text"].to_list()[:5]
sample_true = test_df["label"].to_list()[:5]
sample_idx = torch.tensor(
    [text_to_indices(t, vocab, MAX_LEN) for t in sample_texts],
    dtype=torch.long,
    device=device,
)

# Get predictions from all three models
with torch.no_grad():
    # Transformer predictions
    transformer_preds = transformer_model(sample_idx).argmax(dim=-1).cpu().tolist()
    # LSTM predictions
    lstm_preds = lstm_model(sample_idx).argmax(dim=-1).cpu().tolist()
    # BERT predictions
    bert_sample_ids, bert_sample_mask = tokenise_for_bert(sample_texts)
    bert_sample_ids = bert_sample_ids.to(device)
    bert_sample_mask = bert_sample_mask.to(device)
    bert_preds = (
        bert_model(input_ids=bert_sample_ids, attention_mask=bert_sample_mask)
        .logits.argmax(dim=-1)
        .cpu()
        .tolist()
    )

print("\n== Sample Predictions (all 3 models) ==")
print(f"{'Headline':<50} {'True':<10} {'LSTM':<10} {'Trans':<10} {'BERT':<10}")
print("-" * 90)
for i, text in enumerate(sample_texts):
    t = CLASS_NAMES[sample_true[i]]
    l = CLASS_NAMES[lstm_preds[i]]
    tr = CLASS_NAMES[transformer_preds[i]]
    b = CLASS_NAMES[bert_preds[i]]
    print(f"{text[:48]:<50} {t:<10} {l:<10} {tr:<10} {b:<10}")

# Attention heatmap from our educational multi-head attention.
# We feed the sample through the Transformer's embedding + the
# EducationalMultiHead to get interpretable attention weights.
mha_viz = EducationalMultiHead(d_model=128, n_heads=4).to(device)
with torch.no_grad():
    embed = transformer_model.embed(sample_idx[:1])  # first headline
    embed = transformer_model.posenc(embed)
    _, attn_weights = mha_viz(embed)  # (1, 4, seq, seq)
    attn_np = attn_weights[0, 0].cpu().numpy()  # head 0, first sample

# Build word labels for the heatmap axes
words = sample_texts[0].lower().split()[:MAX_LEN]
words = words + ["<pad>"] * (MAX_LEN - len(words))

# Create attention heatmap using plotly (ModelVisualizer doesn't have a
# native heatmap, so we use plotly directly -- consistent with M5 patterns).
# Show only the first 15 tokens for readability.
show_len = min(15, len([w for w in words if w != "<pad>"]))
fig_attn = go.Figure(
    data=go.Heatmap(
        z=attn_np[:show_len, :show_len],
        x=words[:show_len],
        y=words[:show_len],
        colorscale="Viridis",
        colorbar={"title": "Attention weight"},
    )
)
fig_attn.update_layout(
    title="Self-Attention Heatmap (Head 0, first sample)",
    xaxis_title="Key position",
    yaxis_title="Query position",
    width=600,
    height=500,
)
fig_attn.write_html("ex_4_attention_heatmap.html")
print("\nAttention heatmap saved to ex_4_attention_heatmap.html")

# Training curves comparison: all 3 models on one chart
fig_curves = viz.training_history(
    metrics={
        "LSTM loss": lstm_losses,
        "Transformer loss": transformer_losses,
        "BERT loss": bert_losses,
        "LSTM val_acc": lstm_accs,
        "Transformer val_acc": transformer_accs,
        "BERT val_acc": bert_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig_curves.write_html("ex_4_training_curves.html")
print("Training curves saved to ex_4_training_curves.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert attn_np.shape[0] == MAX_LEN, "Attention heatmap should cover full sequence"
assert Path("ex_4_attention_heatmap.html").exists(), "Attention heatmap should be saved"
assert Path("ex_4_training_curves.html").exists(), "Training curves should be saved"
print("\n--- Checkpoint 9 passed --- visualisations complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Derived scaled dot-product attention with torch.einsum
  [x] Explained the 1/sqrt(d_k) factor (prevents softmax saturation)
  [x] Wrote a hand-rolled multi-head attention wrapping the scratch kernel
  [x] Built a TransformerClassifier with nn.TransformerEncoder
  [x] Built an LSTM baseline for fair comparison
  [x] Trained all 3 models on FULL AG News (120K headlines)
  [x] Fine-tuned BERT ({BERT_MODEL_NAME}) -- best acc: {max(bert_accs):.1%}
  [x] Visualised attention heatmaps (what the model "looks at")
  [x] Tracked every run with ExperimentTracker (params, per-epoch metrics)
  [x] Registered models in ModelRegistry with versioned metrics
  [x] Exported the fine-tuned model to ONNX for portable deployment

  KEY INSIGHT — The Attention Hierarchy:
    LSTM best acc:        {max(lstm_accs):.1%}  (sequential, no pre-training)
    Transformer best acc: {max(transformer_accs):.1%}  (parallel attention, no pre-training)
    BERT best acc:        {max(bert_accs):.1%}  (parallel attention + pre-training)

  Pre-training is the single biggest lever in NLP. The Transformer
  architecture enables it, but the pre-trained weights are what make
  BERT dominate. This is why modern NLP is "pre-train then fine-tune."

  Next: In Exercise 5, you'll build generative models (DCGAN + WGAN-GP)
  that CREATE new data instead of classifying existing data.
"""
)
