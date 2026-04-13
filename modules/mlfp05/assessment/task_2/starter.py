#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP05 Task 2 starter — transfer learning + ONNX.

Implement `solve(onnx_path)` per problem.md. predict() MUST use onnxruntime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cifar10"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def solve(onnx_path: Path) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # TODO: subsample 2000 images from train_set and build X_train, y_train.

    # TODO: load torchvision.models.resnet18 with ImageNet weights.
    # TODO: freeze the backbone (requires_grad_ = False for every parameter).
    # TODO: replace backbone.fc with nn.Linear(512, 10).
    # TODO: wrap so forward() applies ImageNet normalisation before the backbone.

    # TODO: train the new head for 1-2 epochs with Adam(lr=1e-3).

    # TODO: torch.onnx.export the wrapped model to onnx_path with a dynamic
    #       batch axis. Use opset_version=17.

    # TODO: build an onnxruntime.InferenceSession and define predict(images)
    #       that runs the ONNX model (NOT the torch model) and returns int64
    #       labels of shape (N,).

    model: nn.Module = ...  # type: ignore
    predict: Callable[[torch.Tensor], torch.Tensor] = ...  # type: ignore
    return model, predict
