#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP05 Task 3 starter — STI walk-forward LSTM regression.

Implement `solve()` per problem.md. Your model MUST contain an nn.LSTM or
nn.GRU child and MUST call nn.utils.clip_grad_norm_() during training.
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

SEQ_LEN = 10
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
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    out = pl.from_pandas(df)
    out.write_parquet(STI_CACHE)
    return out


def solve() -> tuple[nn.Module, Callable[[pl.DataFrame], torch.Tensor]]:
    torch.manual_seed(42)
    np.random.seed(42)

    df = _fetch_sti()

    # TODO: compute train-only z-score stats (first 80% of rows) and
    #       normalise ALL rows with those train stats. Do NOT use val
    #       statistics.

    # TODO: build (N, SEQ_LEN, 4) windows from the training portion and
    #       a matching (N,) target vector (next-day normalised close).

    # TODO: define a recurrent model. It MUST contain nn.LSTM or nn.GRU.
    # Hint: nn.LSTM(4, 32, num_layers=1, batch_first=True), then a
    #       nn.Linear(32, 1) head that consumes the last hidden state.

    # TODO: train for ~20-25 epochs, Adam(lr=1e-3), MSE loss.
    #       Apply nn.utils.clip_grad_norm_(model.parameters(), 1.0) every step.

    # TODO: put model in eval mode and write predict(frame) that:
    #         1. Recomputes z-score stats on the first 80% of `frame` only
    #         2. Normalises `frame` with those stats
    #         3. Builds rolling windows of length SEQ_LEN
    #         4. Runs them through model and returns a (len(frame) - SEQ_LEN,)
    #            float tensor

    model: nn.Module = ...  # type: ignore
    predict: Callable[[pl.DataFrame], torch.Tensor] = ...  # type: ignore
    return model, predict
