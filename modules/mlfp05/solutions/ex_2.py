# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 2: CNNs and Computer Vision
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build CNNs with torch.nn.Conv2d, BatchNorm2d, and MaxPool2d
#   - Implement a ResNet-style residual block and explain why skip
#     connections keep gradients healthy in deep networks
#   - Add a Squeeze-and-Excitation (SE) block — a modern channel-attention
#     enhancement used in many production CV models
#   - Train with a PyTorch Lightning LightningModule (clean, testable loops)
#   - Export the trained model to ONNX via kailash-ml's OnnxBridge
#
# PREREQUISITES: M5/ex_1 (autoencoders, PyTorch training basics).
# ESTIMATED TIME: ~60 min
# DATASET: Synthetic 28x28 grayscale shape classification (3 classes).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

from kailash_ml import ModelVisualizer
from kailash_ml.bridge.onnx_bridge import OnnxBridge

torch.manual_seed(42)
np.random.seed(42)
pl.seed_everything(42, workers=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# Synthetic image data — 3 shape classes on 28x28 grayscale
# ════════════════════════════════════════════════════════════════════════
# Real CV tutorials often use Fashion-MNIST; for speed we generate 1500
# synthetic images drawn procedurally with numpy. Three classes:
#   0 = filled square, 1 = filled circle, 2 = diagonal bar
def make_shapes(n_per_class: int = 500, size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(n_per_class):  # SQUARES
        s = np.random.randint(6, 12)
        cx = np.random.randint(s, size - s)
        cy = np.random.randint(s, size - s)
        img = ((np.abs(xx - cx) < s) & (np.abs(yy - cy) < s)).astype(np.float32)
        images.append(img)
        labels.append(0)

    for _ in range(n_per_class):  # CIRCLES
        r = np.random.randint(5, 10)
        cx = np.random.randint(r + 2, size - r - 2)
        cy = np.random.randint(r + 2, size - r - 2)
        img = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).astype(np.float32)
        images.append(img)
        labels.append(1)

    for _ in range(n_per_class):  # DIAGONAL BARS
        offset = np.random.randint(-6, 6)
        thickness = np.random.randint(2, 4)
        img = (np.abs(xx - yy + offset) < thickness).astype(np.float32)
        images.append(img)
        labels.append(2)

    X = np.stack(images)[:, None, :, :]        # (N, 1, 28, 28)
    X += 0.05 * np.random.randn(*X.shape).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    perm = np.random.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


X_np, y_np = make_shapes(n_per_class=500)
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

split = int(0.8 * len(X))
train_ds = TensorDataset(X[:split], y[:split])
val_ds = TensorDataset(X[split:], y[split:])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Plain CNN with BatchNorm and MaxPool
# ════════════════════════════════════════════════════════════════════════
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ════════════════════════════════════════════════════════════════════════
# PART 2 — ResNet-style residual block
# ════════════════════════════════════════════════════════════════════════
# A ResBlock computes y = F(x) + x. The shortcut adds the input straight
# to the output, giving gradients a direct path during backprop. This is
# what allowed ResNet to train 50- and 152-layer networks in 2015.
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)     # skip connection


# ════════════════════════════════════════════════════════════════════════
# PART 3 — Squeeze-and-Excitation (SE) block
# ════════════════════════════════════════════════════════════════════════
# SE blocks recalibrate channel-wise feature responses. Intuition: "which
# feature maps matter for this specific input?" We squeeze spatial info
# to a single number per channel with global average pooling, learn a
# small MLP to produce a per-channel gate, and multiply the feature map
# by that gate. Adds <1% params, measurable accuracy boost.
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)     # squeeze
        w = self.fc(s).view(b, c, 1, 1)                # excite
        return x * w


class ResNetSE(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 28 -> 14
        )
        self.block1 = ResBlock(32)
        self.se1 = SEBlock(32)
        self.block2 = ResBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.se1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ════════════════════════════════════════════════════════════════════════
# PART 4 — LightningModule wrapping the model
# ════════════════════════════════════════════════════════════════════════
# pytorch_lightning gives us a clean training/validation loop, automatic
# device placement, and a trainer.fit() that handles gradient accumulation,
# checkpointing, and logging out of the box.
class LitCNN(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_losses: list[float] = []
        self.val_accs: list[float] = []
        self._batch_losses: list[float] = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self._batch_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        if self._batch_losses:
            self.train_losses.append(float(np.mean(self._batch_losses)))
            self._batch_losses = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        self.val_accs.append(acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def train_lightning(model: nn.Module, name: str, epochs: int = 5) -> tuple[list[float], list[float]]:
    lit = LitCNN(model)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(lit, train_loader, val_loader)
    print(f"  [{name}] final train loss={lit.train_losses[-1]:.4f}  val acc={lit.val_accs[-1]:.3f}")
    return lit.train_losses, lit.val_accs


# ── Train both models ──────────────────────────────────────────────────
print("\n── SimpleCNN ──")
simple = SimpleCNN()
simple_losses, simple_accs = train_lightning(simple, "SimpleCNN", epochs=4)

print("\n── ResNetSE ──")
resnet_se = ResNetSE()
resnet_losses, resnet_accs = train_lightning(resnet_se, "ResNetSE", epochs=4)


# ════════════════════════════════════════════════════════════════════════
# PART 5 — Export the best model to ONNX with OnnxBridge
# ════════════════════════════════════════════════════════════════════════
bridge = OnnxBridge()
resnet_se.eval()
onnx_path = Path("ex_2_resnet_se.onnx")

exported = False
try:
    result = bridge.export(
        model=resnet_se,
        framework="pytorch",
        output_path=onnx_path,
        n_features=28 * 28,
    )
    success = getattr(result, "success", bool(result))
    print(f"\nOnnxBridge.export success: {success}")
    exported = bool(success) and onnx_path.exists()
except Exception as exc:
    print(f"\nOnnxBridge.export raised {type(exc).__name__}: {exc}")

if not exported:
    # OnnxBridge is optimised for tabular models; for CNNs we fall back
    # to raw torch.onnx. Either path produces a .onnx file that can be
    # loaded by any ONNX Runtime.
    print("Falling back to torch.onnx.export for Conv2D graph ...")
    sample = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        resnet_se,
        sample,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

print(f"Wrote {onnx_path} ({onnx_path.stat().st_size // 1024} KB)")


# ════════════════════════════════════════════════════════════════════════
# PART 6 — Visualise training curves
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "SimpleCNN loss": simple_losses,
        "ResNetSE loss": resnet_losses,
    },
    x_label="Epoch",
    y_label="Cross-entropy",
)
fig.write_html("ex_2_training.html")
print("Training history saved to ex_2_training.html")


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Built Conv2d + BatchNorm + MaxPool CNNs in PyTorch
  [x] Implemented a ResNet residual block (y = F(x) + x)
  [x] Added a Squeeze-and-Excitation channel attention block
  [x] Trained with pytorch_lightning.LightningModule and Trainer
  [x] Exported the trained model to ONNX for portable deployment
"""
)
