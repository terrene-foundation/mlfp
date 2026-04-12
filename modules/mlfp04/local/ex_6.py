# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6: Deep Learning with ONNX Export
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a CNN with residual connections (ResBlock) in PyTorch
#   - Explain why skip connections solve the vanishing gradient problem
#   - Train with cosine annealing LR scheduling and gradient clipping
#   - Monitor gradient norms to diagnose training stability
#   - Export a trained CNN to ONNX format via OnnxBridge for deployment
#
# PREREQUISITES:
#   - MLFP04 Exercise 4 (production monitoring — deployment context)
#   - MLFP03 Exercise 1 (feature engineering — same mindset for image features)
#
# ESTIMATED TIME: 60-75 minutes
#
# TASKS:
#   1. Load and prepare image data
#   2. Build CNN architecture with residual connections
#   3. Train with cosine annealing LR scheduler
#   4. Monitor gradients and training dynamics
#   5. Export to ONNX with OnnxBridge
#   6. Validate ONNX model matches PyTorch predictions
#
# DATASET: Synthetic medical image data (matching ChestX-ray14 properties)
#   Format: (N, 1, 64, 64) grayscale images, multi-label classification
#   Task: detect 5 medical conditions simultaneously (binary per class)
#   Real-world note: in production, load from mlfp03/chest_xray_subset/
#
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

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load and prepare data
# ══════════════════════════════════════════════════════════════════════
# Using synthetic data matching ChestX-ray14 characteristics
# In production, load from mlfp03/chest_xray_subset/

loader = MLFPDataLoader()

# Synthetic data matching medical image properties
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
        # TODO: pass x through conv1 -> bn1 -> relu
        out = ____  # Hint: torch.relu(self.bn1(self.conv1(x)))
        # TODO: pass out through conv2 -> bn2 (no relu yet)
        out = ____  # Hint: self.bn2(self.conv2(out))
        # TODO: return relu(out + residual) — the skip connection
        return ____  # Hint: torch.relu(out + residual)


class MedicalCNN(nn.Module):
    """CNN for multi-label medical image classification."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        # TODO: define self.features as nn.Sequential with:
        #   Conv2d(1->32, 3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2)
        #   -> ResBlock(32)
        #   -> Conv2d(32->64, 3, padding=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2)
        #   -> ResBlock(64)
        #   -> AdaptiveAvgPool2d(4)
        self.features = nn.Sequential(
            ____,  # Hint: nn.Conv2d(1, 32, 3, padding=1)
            ____,  # Hint: nn.BatchNorm2d(32)
            ____,  # Hint: nn.ReLU()
            ____,  # Hint: nn.MaxPool2d(2)
            ____,  # Hint: ResBlock(32)
            ____,  # Hint: nn.Conv2d(32, 64, 3, padding=1)
            ____,  # Hint: nn.BatchNorm2d(64)
            ____,  # Hint: nn.ReLU()
            ____,  # Hint: nn.MaxPool2d(2)
            ____,  # Hint: ResBlock(64)
            ____,  # Hint: nn.AdaptiveAvgPool2d(4)
        )
        # TODO: define self.classifier as nn.Sequential:
        #   Flatten -> Linear(64*4*4 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> n_classes)
        self.classifier = nn.Sequential(
            ____,  # Hint: nn.Flatten()
            ____,  # Hint: nn.Linear(64 * 4 * 4, 128)
            ____,  # Hint: nn.ReLU()
            ____,  # Hint: nn.Dropout(0.3)
            ____,  # Hint: nn.Linear(128, n_classes)
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

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert total_params > 0, "Model should have trainable parameters"
assert X_images.shape == (n_samples, n_channels, img_size, img_size), \
    f"Image tensor shape should be (N, C, H, W), got {X_images.shape}"
# Verify ResBlock preserves spatial dimensions
_dummy = torch.zeros(1, 32, 16, 16)
_res_block = ResBlock(32)
_out = _res_block(_dummy)
assert _out.shape == _dummy.shape, "ResBlock should preserve spatial dimensions"
# INTERPRETATION: ResBlocks (residual blocks) solve the vanishing gradient
# problem in deep networks. The skip connection y = F(x) + x lets gradients
# flow directly from loss to early layers. Without skip connections, deep
# networks often train worse than shallow ones — the residual connection
# ensures each layer only needs to learn the residual (improvement over
# the identity), which is easier to optimise.
print("\n✓ Checkpoint 1 passed — CNN architecture built, ResBlock verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train with cosine annealing LR
# ══════════════════════════════════════════════════════════════════════

criterion = nn.BCEWithLogitsLoss()  # Multi-label: BCE per class
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine annealing: LR decays from max to min following cosine curve
n_epochs = 20
# TODO: create CosineAnnealingLR scheduler with T_max=n_epochs, eta_min=1e-6
scheduler = ____  # Hint: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

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

        # TODO: zero gradients, forward pass, compute loss, backprop
        ____  # Hint: optimizer.zero_grad()
        logits = ____  # Hint: model(X_batch)
        loss = ____  # Hint: criterion(logits, y_batch)
        ____  # Hint: loss.backward()

        # TODO: compute total gradient norm (sum of squared L2 norms, then sqrt)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                # TODO: add squared L2 norm of p.grad.data to total_norm
                total_norm += ____  # Hint: p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        epoch_grad_norms.append(total_norm)

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # TODO: update model weights
        ____  # Hint: optimizer.step()
        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # TODO: forward pass through model
            logits = ____  # Hint: model(X_batch)
            val_losses.append(criterion(logits, y_batch).item())

    # Record
    history["train_loss"].append(np.mean(train_losses))
    history["val_loss"].append(np.mean(val_losses))
    history["lr"].append(scheduler.get_last_lr()[0])
    history["grad_norm"].append(np.mean(epoch_grad_norms))

    # TODO: step the LR scheduler at end of each epoch
    ____  # Hint: scheduler.step()

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

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(history["train_loss"]) == n_epochs, "Should have one loss per epoch"
assert history["train_loss"][-1] < history["train_loss"][0], \
    "Training loss should decrease over epochs"
assert history["lr"][-1] < history["lr"][0], \
    "Cosine annealing should reduce LR over training"
# INTERPRETATION: Cosine annealing starts at lr=1e-3 and smoothly decays
# to lr=1e-6 following a cosine curve. This avoids the sharp LR drops of
# step schedulers — the smooth decay keeps the model in a good region of
# the loss landscape rather than oscillating around sharp minima.
print("\n✓ Checkpoint 2 passed — training complete, LR schedule verified\n")

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

# TODO: export model to ONNX using bridge.export()
#       with framework="pytorch", output_path="medical_cnn.onnx", n_features=None
export_result = bridge.export(
    model=____,  # Hint: model
    framework=____,  # Hint: "pytorch"
    output_path=____,  # Hint: "medical_cnn.onnx"
    n_features=____,  # Hint: None (not needed for CNN)
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert compat.compatible, "Model should be ONNX-compatible with pytorch framework"
# INTERPRETATION: ONNX (Open Neural Network Exchange) is a vendor-neutral
# model interchange format. OnnxBridge checks that all operations in the
# model have ONNX equivalents — custom ops without ONNX support would fail
# here. ONNX compatibility is the first step toward cross-framework deployment.
print("\n✓ Checkpoint 3 passed — ONNX compatibility verified\n")

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

    # TODO: validate ONNX model against PyTorch using bridge.validate()
    #       with onnx_path=export_result.onnx_path, sample_input=sample_input,
    #       tolerance=1e-4
    validation = bridge.validate(
        model=____,  # Hint: model
        onnx_path=____,  # Hint: export_result.onnx_path
        sample_input=____,  # Hint: sample_input
        tolerance=____,  # Hint: 1e-4
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

print("\n✓ Exercise 6 complete — CNN training + ONNX export")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(f"""
  ✓ ResBlock: skip connections preserve gradient flow in deep networks
  ✓ BCEWithLogitsLoss: numerically stable multi-label loss (fused sigmoid)
  ✓ CosineAnnealingLR: smooth LR decay avoids sharp scheduler transitions
  ✓ Gradient monitoring: norm tracking catches exploding/vanishing gradients
  ✓ OnnxBridge: PyTorch → ONNX export with numerical fidelity validation

  KEY INSIGHT: Residual connections (y = F(x) + x) solve depth:
    Without skip: gradient must flow through every multiplication → vanishes
    With skip: identity path carries gradient directly → stable deep training
    ResNets proved: depth helps, but only with proper gradient highways.

  TRAINING DYNAMICS CHECKLIST:
    ✓ Loss decreasing → model learning
    ✓ LR following cosine curve → scheduler working
    ✓ Grad norm stable (not exploding) → clipping effective
    ✓ Val loss > train loss → overfitting (acceptable here, small data)

  ONNX DEPLOYMENT CHAIN:
    PyTorch (training) → OnnxBridge.export() → .onnx file
    → OnnxBridge.validate() → numerical parity check
    → InferenceServer (M5) → production serving

  NEXT: Exercise 8 is the M4 capstone — deep learning foundations
  from first principles. You'll prove why hidden layers beat linear
  on XOR, then build the full CNN architecture from scratch.
""")
print("═" * 70)
