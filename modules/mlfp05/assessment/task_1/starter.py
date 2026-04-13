#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP05 Task 1 starter — Fashion-MNIST CNN classifier.

Fill in the TODOs. Your submission MUST define `solve()` returning
(model, predict) per problem.md.
"""
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


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        # TODO: build conv layers and fc head. Aim for 50k-500k parameters.
        # Hint: Conv2d(1, 16, 3, padding=1) -> pool -> Conv2d(16, 32, 3, padding=1) -> pool -> fc
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass. Output raw logits shape (N, 10).
        ...


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

    # TODO: subsample the train_set (e.g. 12000 images) and build X_train, y_train.
    # Keep memory small so grader runs under 6 minutes on laptop CPU.

    # TODO: build DataLoader, model, optimizer, loss.
    # TODO: train for ~3 epochs with Adam lr=1e-3.

    # TODO: put model in eval mode and define predict(images) -> (N,) int64 labels.

    model: nn.Module = ...  # type: ignore
    predict = ...  # type: ignore
    return model, predict


if __name__ == "__main__":
    model, predict = solve()
    print("starter.py is a skeleton — fill in the TODOs.")
