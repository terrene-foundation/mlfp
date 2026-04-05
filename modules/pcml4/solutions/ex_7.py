# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT4 — Exercise 7: Deep Learning Foundations
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build intuition for deep learning from a simple linear
#   network up to a CNN with ResBlocks. Train with LR scheduling and
#   gradient monitoring. Export to ONNX via OnnxBridge for deployment.
#
# TASKS:
#   1. Simple linear network — forward pass, backprop by hand
#   2. Build CNN architecture with residual connections (ResBlock)
#   3. Train with cosine annealing LR scheduler
#   4. Monitor gradients and training dynamics
#   5. Export to ONNX with OnnxBridge
#   6. Validate ONNX model matches PyTorch predictions
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kailash_ml import ModelVisualizer
from kailash_ml.bridge.onnx_bridge import OnnxBridge

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Simple linear network — forward pass and backprop by hand
# ══════════════════════════════════════════════════════════════════════
# Before building a CNN, understand the building blocks:
# a single linear layer followed by sigmoid — binary classification.
#
# Forward pass:  z = Wx + b,  ŷ = σ(z)
# Loss:          L = -y log(ŷ) - (1-y) log(1-ŷ)   (binary cross-entropy)
# Backward pass: ∂L/∂W = (ŷ - y) x'  ,  ∂L/∂b = (ŷ - y)
# Update:        W ← W - η ∂L/∂W

rng_simple = np.random.default_rng(42)
n_simple = 200
n_feats_simple = 4

# XOR-like binary classification problem
X_simple = rng_simple.standard_normal((n_simple, n_feats_simple)).astype(np.float32)
y_simple = ((X_simple[:, 0] > 0) ^ (X_simple[:, 1] > 0)).astype(np.float32)

print(f"=== Simple Linear Network (by hand) ===")
print(
    f"Task: XOR-like binary classification ({n_simple} samples, {n_feats_simple} features)"
)
print(f"Class balance: {y_simple.mean():.2f}")

# PyTorch single-layer network
simple_net = nn.Linear(n_feats_simple, 1)
simple_opt = torch.optim.SGD(simple_net.parameters(), lr=0.1)
simple_crit = nn.BCEWithLogitsLoss()

X_t = torch.from_numpy(X_simple)
y_t = torch.from_numpy(y_simple).unsqueeze(1)

simple_losses = []
for epoch in range(50):
    simple_opt.zero_grad()
    logits = simple_net(X_t)
    loss = simple_crit(logits, y_t)
    loss.backward()
    simple_opt.step()
    simple_losses.append(loss.item())

with torch.no_grad():
    preds = torch.sigmoid(simple_net(X_t)).numpy().flatten()
    acc = ((preds > 0.5) == y_simple).mean()

print(f"After 50 epochs: loss={simple_losses[-1]:.4f}, accuracy={acc:.4f}")
print(f"  Note: linear model cannot learn XOR — needs hidden layers (non-linearity)")

# Add one hidden layer — now it CAN learn XOR
hidden_net = nn.Sequential(
    nn.Linear(n_feats_simple, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
hidden_opt = torch.optim.Adam(hidden_net.parameters(), lr=0.01)

hidden_losses = []
for epoch in range(100):
    hidden_opt.zero_grad()
    loss = simple_crit(hidden_net(X_t), y_t)
    loss.backward()
    hidden_opt.step()
    hidden_losses.append(loss.item())

with torch.no_grad():
    preds_h = torch.sigmoid(hidden_net(X_t)).numpy().flatten()
    acc_h = ((preds_h > 0.5) == y_simple).mean()

print(f"Hidden layer (16 units): loss={hidden_losses[-1]:.4f}, accuracy={acc_h:.4f}")
print(f"  ReLU non-linearity enables learning non-linear boundaries like XOR")
print(f"\nKey insight:")
print(f"  Linear: z = Wx + b  (hyperplane decision boundary)")
print(f"  ReLU:   max(0, z)   (piecewise-linear, can approximate any function)")
print(f"  Depth:  stacking layers = composing functions = exponential expressivity")


# ══════════════════════════════════════════════════════════════════════
# TASK 1b: Load and prepare image data
# ══════════════════════════════════════════════════════════════════════
# Using synthetic data matching ChestX-ray14 characteristics
# In production, load from ascent04/chest_xray_subset/

loader = ASCENTDataLoader()

# Load pre-extracted features (flattened images or embeddings)
# For the exercise, we create synthetic data matching medical image properties
n_samples = 5000
n_channels = 1  # Grayscale X-rays
img_size = 64  # Downsampled for training speed
n_classes = 5  # Multi-label: 5 conditions

rng = np.random.default_rng(42)
X_images = rng.standard_normal((n_samples, n_channels, img_size, img_size)).astype(
    np.float32
)
# Multi-label: each sample can have multiple conditions
y_labels = (rng.random((n_samples, n_classes)) > 0.85).astype(np.float32)

print(f"=== Medical Image Data ===")
print(f"Images: {X_images.shape} (N, C, H, W)")
print(f"Labels: {y_labels.shape} (N, classes)")
print(f"Positive rates per class: {y_labels.mean(axis=0).round(3)}")

# Split
split = int(0.8 * n_samples)
X_train, X_test = X_images[:split], X_images[split:]
y_train, y_test = y_labels[:split], y_labels[split:]

# DataLoaders
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build CNN with residual connections
# ══════════════════════════════════════════════════════════════════════


class ResBlock(nn.Module):
    """Residual block: skip connection preserves gradient flow."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)  # Skip connection


class MedicalCNN(nn.Module):
    """CNN for multi-label medical image classification."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(64),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = MedicalCNN(n_classes=n_classes).to(device)
print(f"\n=== Model Architecture ===")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable: {trainable_params:,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train with cosine annealing LR
# ══════════════════════════════════════════════════════════════════════

criterion = nn.BCEWithLogitsLoss()  # Multi-label: BCE per class
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine annealing: LR decays from max to min following cosine curve
n_epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs, eta_min=1e-6
)

# Training loop with gradient monitoring
history = {
    "train_loss": [],
    "val_loss": [],
    "lr": [],
    "grad_norm": [],
}

for epoch in range(n_epochs):
    # Train
    model.train()
    train_losses = []
    epoch_grad_norms = []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient monitoring: track gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        epoch_grad_norms.append(total_norm)

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            val_losses.append(criterion(logits, y_batch).item())

    # Record
    history["train_loss"].append(np.mean(train_losses))
    history["val_loss"].append(np.mean(val_losses))
    history["lr"].append(scheduler.get_last_lr()[0])
    history["grad_norm"].append(np.mean(epoch_grad_norms))

    scheduler.step()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"train_loss={history['train_loss'][-1]:.4f}, "
            f"val_loss={history['val_loss'][-1]:.4f}, "
            f"lr={history['lr'][-1]:.6f}, "
            f"grad_norm={history['grad_norm'][-1]:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate model
# ══════════════════════════════════════════════════════════════════════

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(y_batch.numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_labels)

print(f"\n=== Model Evaluation ===")
class_names = [
    "Condition_A",
    "Condition_B",
    "Condition_C",
    "Condition_D",
    "Condition_E",
]
for i, name in enumerate(class_names):
    if y_true[:, i].sum() > 0:
        from sklearn.metrics import roc_auc_score as auc_fn

        auc = auc_fn(y_true[:, i], y_pred[:, i])
        print(f"  {name}: AUC={auc:.4f}, prevalence={y_true[:, i].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export to ONNX with OnnxBridge
# ══════════════════════════════════════════════════════════════════════

bridge = OnnxBridge()

# Check compatibility
compat = bridge.check_compatibility(model, framework="pytorch")
print(f"\n=== ONNX Compatibility ===")
print(f"Compatible: {compat.compatible}")
print(f"Confidence: {compat.confidence}")

# Export
export_result = bridge.export(
    model=model,
    framework="pytorch",
    output_path="medical_cnn.onnx",
    n_features=None,  # Not needed for CNN (uses sample input internally)
)

print(f"\nExport result: {export_result.success}")
if export_result.onnx_path:
    print(f"ONNX path: {export_result.onnx_path}")
    if export_result.model_size_bytes:
        print(f"Model size: {export_result.model_size_bytes / 1024:.1f} KB")
    print(f"Export time: {export_result.export_time_seconds:.2f}s")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Validate ONNX model
# ══════════════════════════════════════════════════════════════════════

if export_result.success and export_result.onnx_path:
    sample_input = torch.from_numpy(X_test[:10]).to(device)

    validation = bridge.validate(
        model=model,
        onnx_path=export_result.onnx_path,
        sample_input=sample_input,
        tolerance=1e-4,
    )

    print(f"\n=== ONNX Validation ===")
    print(f"Valid: {validation.valid}")
    print(f"Max difference: {validation.max_diff:.8f}")
    print(f"Mean difference: {validation.mean_diff:.8f}")
    print(f"Samples tested: {validation.n_samples}")


# ══════════════════════════════════════════════════════════════════════
# Visualise training dynamics
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Loss curves
fig_loss = viz.training_history(
    {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]},
    x_label="Epoch",
)
fig_loss.update_layout(title="Training and Validation Loss")
fig_loss.write_html("ex5_loss_curves.html")

# LR schedule
fig_lr = viz.training_history(
    {"Learning Rate": history["lr"]},
    x_label="Epoch",
)
fig_lr.update_layout(title="Cosine Annealing LR Schedule")
fig_lr.write_html("ex5_lr_schedule.html")

# Gradient norms
fig_grad = viz.training_history(
    {"Gradient Norm": history["grad_norm"]},
    x_label="Epoch",
)
fig_grad.update_layout(title="Gradient Norm During Training")
fig_grad.write_html("ex5_gradient_norms.html")

print("\nSaved: ex5_loss_curves.html, ex5_lr_schedule.html, ex5_gradient_norms.html")

print("\n✓ Exercise 7 complete — CNN training + ONNX export")
print("  Next: Exercise 8 deploys this ONNX model via InferenceServer + Nexus")
