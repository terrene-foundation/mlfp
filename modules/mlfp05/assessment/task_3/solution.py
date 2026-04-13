#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP05 Task 3 reference solution — STI walk-forward LSTM regression.

Predicts the HORIZON-day-ahead z-score-normalised close from a SEQ_LEN-day
window. Naive baseline = "close HORIZON days ahead equals close today".
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
STI_CACHE = REPO_ROOT / "data" / "mlfp05" / "sti" / "sti_close.parquet"

SEQ_LEN = 20
HORIZON = 5
FEATURES = ["Close", "High", "Low", "Volume"]


def _fetch_sti() -> pl.DataFrame:
    if STI_CACHE.exists():
        return pl.read_parquet(STI_CACHE)
    import yfinance as yf

    STI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df = yf.download(
        "^STI", start="2010-01-01", end="2024-12-31", progress=False, auto_adjust=True
    )
    if df is None or len(df) == 0:
        df = yf.download(
            "AAPL",
            start="2010-01-01",
            end="2024-12-31",
            progress=False,
            auto_adjust=True,
        )
    assert df is not None
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    out = pl.from_pandas(df)
    out.write_parquet(STI_CACHE)
    return out


def _train_stats(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    data = df.select(FEATURES).to_numpy().astype(np.float32)
    split = int(0.8 * len(data))
    train = data[:split]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def _windowed(
    data: np.ndarray, seq_len: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n_windows = len(data) - seq_len - horizon + 1
    X = np.stack([data[i : i + seq_len] for i in range(n_windows)])
    y = data[seq_len + horizon - 1 : seq_len + horizon - 1 + n_windows, 0]
    return X.astype(np.float32), y.astype(np.float32)


class SequenceLSTM(nn.Module):
    def __init__(self, input_dim: int = len(FEATURES), hidden_dim: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _out, (h_n, _c_n) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def solve() -> tuple[nn.Module, Callable[[pl.DataFrame], torch.Tensor]]:
    torch.manual_seed(42)
    np.random.seed(42)

    df = _fetch_sti()
    mean, std = _train_stats(df)
    data = df.select(FEATURES).to_numpy().astype(np.float32)
    data_norm = (data - mean) / std

    split = int(0.8 * len(data_norm))
    train_norm = data_norm[:split]
    X_train, y_train = _windowed(train_norm, SEQ_LEN, HORIZON)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
    )

    model = SequenceLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _epoch in range(40):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    model.eval()

    def predict(frame: pl.DataFrame) -> torch.Tensor:
        raw = frame.select(FEATURES).to_numpy().astype(np.float32)
        mean_p, std_p = _train_stats(frame)
        normed = (raw - mean_p) / std_p
        windows, _ = _windowed(normed, SEQ_LEN, HORIZON)
        with torch.no_grad():
            preds = model(torch.from_numpy(windows))
        return preds.float()

    return model, predict


if __name__ == "__main__":
    model, predict = solve()
    df = _fetch_sti()
    preds = predict(df)
    print(f"STI rows: {len(df)}  predictions: {tuple(preds.shape)}")
