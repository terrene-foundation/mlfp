# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 1: Handwritten Postcode Reader (Reference Solution)

Withheld from students. Verified to pass grader.py.

A two-block Conv2d network trained with Adam on the documented training
split. ~1437 8x8 images, batch 64, 25 epochs — under a minute on CPU.

sklearn appears here ONLY to load the bundled digits dataset and to draw the
documented stratified split (data plumbing, not model work). The model itself
is raw torch.nn, per the module's DL rules.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(2)

SEED = 7


def load_digits_train() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The documented training split — identical to starter.py."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
    return train_test_split(
        x, y.astype(np.int64), test_size=0.2, stratify=y, random_state=7
    )


class PostcodeCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8 -> 4
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4 -> 2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def solve() -> dict:
    x_train, x_val, y_train, y_val = load_digits_train()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = PostcodeCNN()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    xv, yv = torch.tensor(x_val), torch.tensor(y_val)
    for _epoch in range(25):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            vloss = float(F.cross_entropy(model(xv), yv))
        history["train_loss"].append(total / n)
        history["val_loss"].append(vloss)

    model.eval()
    return {"model": model, "history": history}


if __name__ == "__main__":
    out = solve()
    h = out["history"]
    print(f"train loss {h['train_loss'][0]:.4f} -> {h['train_loss'][-1]:.4f}")
    print(f"val loss   {h['val_loss'][0]:.4f} -> {h['val_loss'][-1]:.4f}")
