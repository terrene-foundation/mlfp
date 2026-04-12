# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7: Transfer Learning with a Pre-trained ResNet
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Load a pre-trained torchvision ResNet-18 and adapt it to a new task
#   - Freeze the convolutional backbone and train only a new classifier head
#   - Explain why transfer learning needs far fewer examples than scratch
#   - Compare "from scratch" vs "transfer" on the same small dataset
#   - Export the transferred model to ONNX for deployment
#
# PREREQUISITES: M5/ex_2 (CNNs and PyTorch Lightning).
# ESTIMATED TIME: ~60 min
# DATASET: Synthetic 3-class 32x32 RGB images (upscaled from ex_2 shapes).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as T

from kailash_ml import ModelVisualizer

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# Synthetic RGB image data — 3 classes, 32x32 (upscaled later to 64x64)
# ════════════════════════════════════════════════════════════════════════
def make_rgb_shapes(n_per_class: int = 120, size: int = 32):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(n_per_class):                # class 0: red square
        s = np.random.randint(6, 12)
        cx = np.random.randint(s, size - s)
        cy = np.random.randint(s, size - s)
        mask = ((np.abs(xx - cx) < s) & (np.abs(yy - cy) < s)).astype(np.float32)
        img = np.stack([mask, 0.1 * mask, 0.1 * mask], axis=0)
        images.append(img)
        labels.append(0)

    for _ in range(n_per_class):                # class 1: green circle
        r = np.random.randint(5, 10)
        cx = np.random.randint(r + 2, size - r - 2)
        cy = np.random.randint(r + 2, size - r - 2)
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).astype(np.float32)
        img = np.stack([0.1 * mask, mask, 0.1 * mask], axis=0)
        images.append(img)
        labels.append(1)

    for _ in range(n_per_class):                # class 2: blue diagonal bar
        offset = np.random.randint(-6, 6)
        thickness = np.random.randint(2, 4)
        mask = (np.abs(xx - yy + offset) < thickness).astype(np.float32)
        img = np.stack([0.1 * mask, 0.1 * mask, mask], axis=0)
        images.append(img)
        labels.append(2)

    X = np.stack(images).astype(np.float32)
    X += 0.02 * np.random.randn(*X.shape).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


X_np, y_np = make_rgb_shapes(n_per_class=120, size=32)
print(f"Dataset: {X_np.shape[0]} images, shape={X_np.shape[1:]}")

# Upscale to 64x64 (ResNet expects larger inputs; we use 64 for speed)
X_t = torch.from_numpy(X_np)
X_t = F.interpolate(X_t, size=(64, 64), mode="bilinear", align_corners=False)

# ImageNet normalisation (important for torchvision pre-trained models)
normalise = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
X_t = normalise(X_t)
y_t = torch.from_numpy(y_np)

split = int(0.7 * len(X_t))
train_loader = DataLoader(TensorDataset(X_t[:split], y_t[:split]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_t[split:], y_t[split:]), batch_size=32)


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Load a pre-trained ResNet-18 and adapt it
# ════════════════════════════════════════════════════════════════════════
# torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT) loads the
# ImageNet-pretrained checkpoint. We swap the final fc layer for a fresh
# 3-class head, freeze everything else, and only train the new head.
# In real projects you often unfreeze the last conv block too ("fine-tuning").
def build_transfer_resnet(n_classes: int = 3, freeze_backbone: bool = True) -> nn.Module:
    try:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        print(f"Loaded pre-trained ResNet-18 (weights={weights})")
    except Exception as exc:
        # Offline fallback: random weights. Training "from pre-trained" in
        # this branch is really training from scratch, but the code path
        # remains identical — this is how a real offline environment is handled.
        print(f"Pre-trained weights unavailable ({type(exc).__name__}: {exc})")
        print("Falling back to randomly initialised ResNet-18.")
        model = torchvision.models.resnet18(weights=None)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    # Replace the final fc with a fresh head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model


def build_scratch_cnn(n_classes: int = 3) -> nn.Module:
    """Baseline: a small CNN trained from random init for comparison."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, n_classes),
    )


# ════════════════════════════════════════════════════════════════════════
# Training harness
# ════════════════════════════════════════════════════════════════════════
def train_model(model: nn.Module, name: str, epochs: int = 4, lr: float = 1e-3):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n── {name} ──  trainable params: {n_trainable:,} / {n_total:,}")

    opt = torch.optim.Adam(params, lr=lr)
    train_losses: list[float] = []
    val_accs: list[float] = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=-1)
                correct += int((preds == yb).sum().item())
                total += int(yb.size(0))
        val_accs.append(correct / total)
        print(f"  epoch {epoch+1}  loss={train_losses[-1]:.4f}  val_acc={val_accs[-1]:.3f}")
    return train_losses, val_accs


# ── Train both ─────────────────────────────────────────────────────────
transfer_model = build_transfer_resnet()
transfer_losses, transfer_accs = train_model(transfer_model, "Transfer (frozen ResNet-18 + new head)", epochs=4)

scratch_model = build_scratch_cnn()
scratch_losses, scratch_accs = train_model(scratch_model, "Scratch (small CNN)", epochs=4)


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Export the transferred model to ONNX
# ════════════════════════════════════════════════════════════════════════
transfer_model.eval()
onnx_path = Path("ex_7_transfer_resnet.onnx")
sample = torch.randn(1, 3, 64, 64, device=device)
torch.onnx.export(
    transfer_model,
    sample,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    dynamo=False,                       # use the stable TorchScript exporter
)
print(f"\nExported to {onnx_path} ({onnx_path.stat().st_size // 1024} KB)")


# ════════════════════════════════════════════════════════════════════════
# PART 3 — Visualise training histories
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "transfer loss": transfer_losses,
        "transfer val acc": transfer_accs,
        "scratch loss": scratch_losses,
        "scratch val acc": scratch_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig.write_html("ex_7_training.html")
print("Training history saved to ex_7_training.html")


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Loaded a pre-trained torchvision.models.resnet18
  [x] Froze the backbone and trained only a new classifier head
  [x] Compared transfer learning vs training a CNN from scratch
  [x] Exported the transferred model to ONNX with torch.onnx.export
  [x] Applied ImageNet mean/std normalisation for pre-trained inputs

  Architecture-selection guide (consolidated across M5):
    Images    -> CNN / ViT  + transfer learning (ImageNet pre-trained)
    Text      -> Transformer + transfer learning (BERT / GPT pre-trained)
    Sequences -> LSTM / Transformer (sometimes transfer)
    Graphs    -> GNN (task-specific; transfer rarely used)
    Tabular   -> Gradient boosting (train from scratch, fast and reliable)
"""
)
