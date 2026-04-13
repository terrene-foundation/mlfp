#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP05 Task 1 reference solution — Fashion-MNIST CNN classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "fashion_mnist"


class FashionCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)  # 32x14x14
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)  # 64x7x7
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)  # 128x3x3
        x = self.dropout(x.flatten(1))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def solve() -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_set = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    N_TRAIN = 20000
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_set), size=N_TRAIN, replace=False)
    X_train = torch.stack([train_set[int(i)][0] for i in idx])
    y_train = torch.tensor([train_set[int(i)][1] for i in idx], dtype=torch.long)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    model = FashionCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(6):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

    model.eval()

    def predict(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = model(images.to(device))
            return logits.argmax(dim=1).cpu().to(torch.int64)

    return model, predict


if __name__ == "__main__":
    model, predict = solve()
    n = sum(p.numel() for p in model.parameters())
    print(f"parameters={n}")
    demo = predict(torch.zeros(2, 1, 28, 28))
    print(f"predict(zeros) -> {demo.tolist()}  dtype={demo.dtype}")
