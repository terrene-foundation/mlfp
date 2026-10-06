# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 3: One-Step-Ahead Load Forecast (Reference Solution)

Withheld from students. Verified to pass grader.py across seeds.

A one-layer GRU (16 hidden units) over a 24-step window — one full seasonal
period — trained with Adam on the documented training series. The seasonal
component is exactly what the naive "tomorrow = today" forecast cannot
exploit, so a trained GRU clears the 0.97x bar with a wide margin.

Raw torch.nn per the module's DL rules.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(2)

WINDOW = 24  # one daily period of the generator's seasonal component
SEED = 7


def make_series(seed: int, n: int = 1200) -> np.ndarray:
    """The load-series generator — identical to starter.py."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float64)
    x[0], x[1] = rng.normal(0.0, 0.3, size=2)
    for t in range(2, n):
        x[t] = (
            0.75 * x[t - 1]
            - 0.20 * x[t - 2]
            + 0.35 * np.sin(2 * np.pi * t / 24)
            + 0.15 * rng.normal()
        )
    return x.astype(np.float32)


def make_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(series, dtype=np.float32)
    xs = np.stack([s[i : i + window] for i in range(len(s) - window)])
    ys = s[window:]
    return xs.reshape(-1, window, 1), ys


class GRUForecaster(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _h = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def solve() -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    series = make_series(SEED)
    X, y = make_windows(series, WINDOW)
    n_train = 900
    loader = DataLoader(
        TensorDataset(torch.tensor(X[:n_train]), torch.tensor(y[:n_train])),
        batch_size=64,
        shuffle=True,
    )

    model = GRUForecaster()
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()
    for _epoch in range(20):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    return {"model": model, "window": WINDOW}


if __name__ == "__main__":
    out = solve()
    model, window = out["model"], out["window"]
    dev = make_series(99)
    X, y = make_windows(dev, window)
    with torch.no_grad():
        preds = model(torch.tensor(X)).reshape(-1).numpy()
    mse = float(((preds - y) ** 2).mean())
    naive = float(((X.reshape(len(X), window)[:, -1] - y) ** 2).mean())
    print(f"dev MSE {mse:.4f} vs naive {naive:.4f} (ratio {mse / naive:.3f})")
