# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 3: GRU Time-Series Forecasting (REFERENCE SOLUTION)

Small 1-layer GRU on a windowed series with genuine temporal structure; beats the
naive last-value baseline on held-out test MSE. Deterministic, CPU-only, < 25s.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN = 20
SEED = 13


def make_dataset():
    """Deterministic windowed forecasting set with learnable structure — identical to starter.

    A pure random walk (raw STI returns) cannot be beaten by ANY model, so we use a
    synthetic series with genuine temporal structure: a damped AR(2) oscillator plus a
    short seasonal cycle plus modest noise. The next value depends on the SHAPE of the
    recent window (not just the last value), so a GRU clears the naive last-value
    baseline with a real margin while the baseline stays honestly hard.
    """
    rng = np.random.default_rng(SEED)
    n = 3000
    a1, a2 = 1.35, -0.55  # complex-conjugate roots => oscillation
    series = np.zeros(n, dtype=np.float64)
    noise = rng.normal(0.0, 0.5, size=n)
    for t in range(2, n):
        season = 0.6 * np.sin(2.0 * np.pi * t / 11.0)
        series[t] = a1 * series[t - 1] + a2 * series[t - 2] + season + noise[t]
    series = series.astype(np.float32)

    xs, ys = [], []
    for i in range(len(series) - SEQ_LEN - 1):
        xs.append(series[i : i + SEQ_LEN])
        ys.append(series[i + SEQ_LEN])
    X = np.array(xs, dtype=np.float32)[:, :, None]  # (N, SEQ_LEN, 1)
    y = np.array(ys, dtype=np.float32)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    naive_pred = X_test[:, -1, 0].astype(np.float32)  # last observed value in window
    return X_train, y_train, X_test, y_test, naive_pred


def solve() -> dict:
    torch.manual_seed(SEED)
    X_train, y_train, X_test, y_test, naive_pred = make_dataset()

    class GRUForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = GRUForecaster()
    uses_recurrent = any(
        isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)) for m in model.modules()
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _epoch in range(60):
        for xb, yb in loader:
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(X_test)).cpu().numpy().astype(np.float32)

    return {
        "model": model,
        "test_pred": test_pred,
        "y_test": y_test,
        "naive_pred": naive_pred,
        "uses_recurrent": uses_recurrent,
    }


if __name__ == "__main__":
    out = solve()
    yt = out["y_test"]
    mse = float(((out["test_pred"] - yt) ** 2).mean())
    naive = float(((out["naive_pred"] - yt) ** 2).mean())
    print(
        f"gru  test_mse={mse:.2e}  naive_mse={naive:.2e}  "
        f"ratio={mse / max(naive, 1e-12):.2f}"
    )
