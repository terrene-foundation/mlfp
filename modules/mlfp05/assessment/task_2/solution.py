#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP05 Task 2 reference solution — transfer learning + ONNX."""
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


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def _build_transfer_model() -> nn.Module:
    backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.fc = nn.Linear(backbone.fc.in_features, 10)

    class TransferNet(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            up = nn.functional.interpolate(
                x, size=(96, 96), mode="bilinear", align_corners=False
            )
            return self.net(_normalize(up))

    return TransferNet(backbone)


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

    N_TRAIN = 2000
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_set), size=N_TRAIN, replace=False)
    X_train = torch.stack([train_set[int(i)][0] for i in idx])
    y_train = torch.tensor([train_set[int(i)][1] for i in idx], dtype=torch.long)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = _build_transfer_model().to(device)
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(head_params, lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(4):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

    model.eval()

    dummy = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
        dynamo=False,
    )

    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    def predict(images: torch.Tensor) -> torch.Tensor:
        arr = images.detach().cpu().numpy().astype(np.float32)
        logits = session.run(["logits"], {"input": arr})[0]
        return torch.from_numpy(logits).argmax(dim=1).to(torch.int64)

    return model, predict


if __name__ == "__main__":
    out = Path(__file__).with_name("reference.onnx")
    if out.exists():
        out.unlink()
    model, predict = solve(out)
    print(f"onnx size: {out.stat().st_size / (1024 * 1024):.1f} MB")
    sample = torch.rand(4, 3, 32, 32)
    print(f"predict(sample) -> {predict(sample).tolist()}")
