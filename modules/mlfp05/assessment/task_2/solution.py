# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 2: Tiny CNN for Image Classification (REFERENCE SOLUTION)

Two-block CNN built from scratch on bundled 8x8 digits. Deterministic, CPU-only, <25s.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

N_CLASSES = 10
SEED = 42


def make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 8x8 digit split — identical to starter."""
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float32)[:, None, :, :]
    y = digits.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    return X_train, y_train, X_test, y_test


def solve() -> dict:
    torch.manual_seed(SEED)
    X_train, y_train, X_test, y_test = make_dataset()

    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8 -> 4
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 4 -> 2
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 2 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, N_CLASSES),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = TinyCNN()
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _epoch in range(25):
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        preds = logits.argmax(dim=1).cpu().numpy()

    return {
        "model": model,
        "preds": preds,
        "y_test": y_test,
        "n_conv": n_conv,
    }


if __name__ == "__main__":
    out = solve()
    acc = (out["preds"] == out["y_test"]).mean()
    print(f"conv_layers={out['n_conv']}  test_acc={acc:.3f}")
