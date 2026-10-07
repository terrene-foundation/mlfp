# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 4: Ship the Postcode Reader (Reference Solution)

Withheld from students. Verified to pass grader.py.

A two-layer MLP trained with Adam, exported through the kailash-ml
OnnxBridge with framework="torch" and a (1, 64) sample input. The bridge
opens the dynamic batch dimension, so the artefact serves any batch size.

sklearn appears ONLY to load the bundled digits and draw the documented
split (data plumbing); the model is raw torch.nn per the module's DL rules.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(2)

SEED = 7


def load_digits_flat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The documented training split — identical to starter.py."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).astype(np.float32)
    return train_test_split(
        x, y.astype(np.int64), test_size=0.2, stratify=y, random_state=7
    )


class PostcodeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def solve() -> dict:
    x_train, x_val, y_train, y_val = load_digits_flat()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = PostcodeMLP()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _epoch in range(30):
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()

    from kailash_ml import OnnxBridge

    onnx_path = Path(tempfile.mkdtemp(prefix="mlfp05_task4_")) / "postcode_mlp.onnx"
    sample = torch.zeros(1, 64)  # serving shape; values are placeholders
    export_result = OnnxBridge().export(
        model, "torch", output_path=onnx_path, sample_input=sample
    )
    if not export_result.success:
        raise RuntimeError(f"OnnxBridge export failed: {export_result}")

    return {
        "model": model,
        "onnx_path": onnx_path,
        "export_result": export_result,
    }


if __name__ == "__main__":
    out = solve()
    print("exported:", out["onnx_path"], "success:", out["export_result"].success)
