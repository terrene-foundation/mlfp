# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 4: Tiny Transformer Text Classification (REFERENCE SOLUTION)

From-scratch tiny transformer encoder (embedding + 2-layer self-attention encoder +
mean-pool + linear) on the bundled AG News slice. Deterministic, CPU-only, < 35s.
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared import MLFPDataLoader

MAX_LEN = 40
MAX_VOCAB = 8000
N_CLASSES = 4
SEED = 5
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def make_dataset():
    """Deterministic AG News encoding — identical to starter."""
    train_df = MLFPDataLoader().load("mlfp05", "ag_news.parquet")
    test_df = MLFPDataLoader().load("mlfp05", "ag_news_test.parquet")
    train_texts = train_df["text"].to_list()
    test_texts = test_df["text"].to_list()
    y_train = train_df["label"].to_numpy().astype(np.int64)
    y_test = test_df["label"].to_numpy().astype(np.int64)

    # Vocab from TRAINING text only, by descending frequency (deterministic).
    from collections import Counter

    counts: Counter = Counter()
    for t in train_texts:
        counts.update(_tokenize(t))
    # 0 = PAD, 1 = UNK; most-common tokens fill the rest.
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counts.most_common(MAX_VOCAB - 2):
        vocab[tok] = len(vocab)

    def encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = _tokenize(t)[:MAX_LEN]
            for j, tok in enumerate(toks):
                out[i, j] = vocab.get(tok, 1)
        return out

    X_train = encode(train_texts)
    X_test = encode(test_texts)
    return X_train, y_train, X_test, y_test, len(vocab)


def solve() -> dict:
    torch.manual_seed(SEED)
    X_train, y_train, X_test, y_test, vocab_size = make_dataset()

    class TinyTransformer(nn.Module):
        def __init__(self, vocab: int, dim: int = 96, heads: int = 4) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim, padding_idx=0)
            self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, dim))
            self.dropout = nn.Dropout(0.1)
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 2,
                dropout=0.1,
                batch_first=True,
            )
            # enable_nested_tensor=False: the padding-mask fast path emits a
            # prototype-API UserWarning; disabling it keeps output identical and
            # the run warning-free (deterministic on CPU).
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=2, enable_nested_tensor=False
            )
            self.head = nn.Linear(dim, N_CLASSES)

        def forward(self, x):
            pad_mask = x == 0  # (B, L) True where padding
            h = self.dropout(self.embed(x) + self.pos)
            h = self.encoder(h, src_key_padding_mask=pad_mask)
            # Mean-pool over non-pad tokens.
            mask = (~pad_mask).unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.head(pooled)

    model = TinyTransformer(vocab_size)
    uses_attention = any(
        isinstance(
            m,
            (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerEncoder),
        )
        for m in model.modules()
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _epoch in range(12):
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).argmax(dim=1).cpu().numpy().astype(np.int64)

    return {
        "model": model,
        "preds": preds,
        "y_test": y_test,
        "uses_attention": uses_attention,
    }


if __name__ == "__main__":
    out = solve()
    acc = (out["preds"] == out["y_test"]).mean()
    yt = out["y_test"]
    majority = np.bincount(yt).max() / len(yt)
    print(f"transformer  test_acc={acc:.2f}  majority={majority:.2f}")
