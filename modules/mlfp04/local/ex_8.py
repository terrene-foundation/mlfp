# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8: Deep Learning Foundations
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Prove why hidden layers are necessary for non-linear problems (XOR)
#   - Implement the forward pass, loss, and backprop for a simple network
#   - Build a CNN with residual connections (ResBlock) from scratch
#   - Train with AdamW, cosine annealing LR, and gradient clipping
#   - Export to ONNX with OnnxBridge and validate numerical fidelity
#
# PREREQUISITES:
#   - MLFP04 Exercise 6 (deep learning with ONNX — this is the capstone)
#   - MLFP03 Exercise 8 (production pipeline — same deployment context)
#
# ESTIMATED TIME: 75-90 minutes
#
# TASKS:
#   1. Simple linear network — forward pass, backprop by hand
#   2. Build CNN architecture with residual connections (ResBlock)
#   3. Train with cosine annealing LR scheduler
#   4. Monitor gradients and training dynamics
#   5. Export to ONNX with OnnxBridge
#   6. Validate ONNX model matches PyTorch predictions
#
# DATASET: Synthetic XOR + synthetic medical images (matching ChestX-ray14)
#   XOR task: proves linear networks cannot learn non-linear boundaries
#   Image task: multi-label classification on 5 medical conditions
#   USML bridge: deep learning = automated feature engineering with error feedback
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

# TODO: Create a single-layer PyTorch network for binary classification.
#   nn.Linear takes (in_features, out_features). For binary output: out_features=1.
simple_net = ____  # Hint: nn.Linear(n_feats_simple, 1)

# TODO: Create SGD optimizer with lr=0.1 for simple_net parameters.
simple_opt = ____  # Hint: torch.optim.SGD(simple_net.parameters(), lr=0.1)

# TODO: Choose the correct loss for binary classification with raw logits.
simple_crit = ____  # Hint: nn.BCEWithLogitsLoss()

X_t = torch.from_numpy(X_simple)
y_t = torch.from_numpy(y_simple).unsqueeze(1)

simple_losses = []
for epoch in range(50):
    # TODO: Zero the gradients before each forward pass.
    ____  # Hint: simple_opt.zero_grad()

    # TODO: Run forward pass through simple_net.
    logits = ____  # Hint: simple_net(X_t)

    # TODO: Compute the loss using simple_crit.
    loss = ____  # Hint: simple_crit(logits, y_t)

    # TODO: Backpropagate the loss.
    ____  # Hint: loss.backward()

    # TODO: Update the parameters.
    ____  # Hint: simple_opt.step()

    simple_losses.append(loss.item())

with torch.no_grad():
    preds = torch.sigmoid(simple_net(X_t)).numpy().flatten()
    acc = ((preds > 0.5) == y_simple).mean()

print(f"After 50 epochs: loss={simple_losses[-1]:.4f}, accuracy={acc:.4f}")
print(f"  Note: linear model cannot learn XOR — needs hidden layers (non-linearity)")

# Add one hidden layer — now it CAN learn XOR
# TODO: Define a 2-layer network using nn.Sequential.
#   Layer 1: Linear(n_feats_simple, 16) + ReLU
#   Layer 2: Linear(16, 1)
hidden_net = nn.Sequential(
    ____,  # Hint: nn.Linear(n_feats_simple, 16)
    ____,  # Hint: nn.ReLU()
    ____,  # Hint: nn.Linear(16, 1)
)

# TODO: Create Adam optimizer with lr=0.01 for hidden_net parameters.
hidden_opt = ____  # Hint: torch.optim.Adam(hidden_net.parameters(), lr=0.01)

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

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert acc_h > acc, \
    f"Hidden-layer network ({acc_h:.4f}) should beat linear ({acc:.4f}) on XOR"
assert hidden_losses[-1] < hidden_losses[0], \
    "Hidden network loss should decrease over training"
# INTERPRETATION: XOR (x1 XOR x2) is NOT linearly separable — no single
# hyperplane can separate the two classes. A linear model is theoretically
# incapable of learning it. Adding one hidden layer with 16 ReLU units
# gives the model enough capacity to approximate any piecewise-linear
# boundary. This is the Universal Approximation Theorem in action.
print("\n✓ Checkpoint 1 passed — hidden layers beat linear on XOR (proven)\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1b: Load and prepare image data
# ══════════════════════════════════════════════════════════════════════
# Using synthetic data matching ChestX-ray14 characteristics
# In production, load from mlfp03/chest_xray_subset/

loader = MLFPDataLoader()

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
        # TODO: Define two Conv2d layers (3x3 kernel, same padding) and
        #   two BatchNorm2d layers. Both convolutions keep channels constant.
        self.conv1 = ____  # Hint: nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = ____  # Hint: nn.BatchNorm2d(channels)
        self.conv2 = ____  # Hint: nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = ____  # Hint: nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        # TODO: Apply conv1 → bn1 → relu, then conv2 → bn2.
        out = ____  # Hint: torch.relu(self.bn1(self.conv1(x)))
        out = ____  # Hint: self.bn2(self.conv2(out))
        # TODO: Add the skip connection (residual) and apply relu.
        return ____  # Hint: torch.relu(out + residual)


class MedicalCNN(nn.Module):
    """CNN for multi-label medical image classification."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        # TODO: Define the feature extractor using nn.Sequential.
        #   Block: Conv2d(1→32, 3x3, pad=1) → BN → ReLU → MaxPool2d(2)
        #          → ResBlock(32)
        #          → Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool2d(2)
        #          → ResBlock(64)
        #          → AdaptiveAvgPool2d(4)
        self.features = nn.Sequential(
            ____,  # Hint: nn.Conv2d(1, 32, 3, padding=1)
            ____,  # Hint: nn.BatchNorm2d(32)
            nn.ReLU(),
            ____,  # Hint: nn.MaxPool2d(2)
            ____,  # Hint: ResBlock(32)
            ____,  # Hint: nn.Conv2d(32, 64, 3, padding=1)
            ____,  # Hint: nn.BatchNorm2d(64)
            nn.ReLU(),
            ____,  # Hint: nn.MaxPool2d(2)
            ____,  # Hint: ResBlock(64)
            ____,  # Hint: nn.AdaptiveAvgPool2d(4)
        )
        # TODO: Define the classifier head.
        #   Flatten → Linear(64*4*4, 128) → ReLU → Dropout(0.3) → Linear(128, n_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            ____,  # Hint: nn.Linear(64 * 4 * 4, 128)
            nn.ReLU(),
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

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert total_params > 0, "Model should have parameters"
# Verify ResBlock shape preservation
_dummy = torch.zeros(1, 32, 16, 16)
_res_block = ResBlock(32)
_out = _res_block(_dummy)
assert _out.shape == _dummy.shape, "ResBlock should preserve spatial dimensions"
# INTERPRETATION: ResBlock preserves shape (H, W, C unchanged) because the
# skip connection adds input to output element-wise. This requires exact
# shape matching. The block only learns the residual improvement F(x),
# not the full transformation — which is far easier to optimise.
print("\n✓ Checkpoint 2 passed — ResBlock architecture verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train with cosine annealing LR
# ══════════════════════════════════════════════════════════════════════

# TODO: Choose the correct multi-label loss (BCE per class, with raw logits).
criterion = ____  # Hint: nn.BCEWithLogitsLoss()

# TODO: Create AdamW optimizer with lr=1e-3, weight_decay=1e-4.
optimizer = ____  # Hint: optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine annealing: LR decays from max to min following cosine curve
n_epochs = 20
# TODO: Create CosineAnnealingLR scheduler. T_max=n_epochs, eta_min=1e-6.
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

        # TODO: Clip gradients to max_norm=1.0 to prevent exploding gradients.
        ____  # Hint: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

    # TODO: Step the LR scheduler at the end of each epoch.
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

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(history["train_loss"]) == n_epochs, "Should have one loss per epoch"
assert history["train_loss"][-1] < history["train_loss"][0], \
    "Training loss should decrease over training"
assert history["lr"][-1] < history["lr"][0], \
    "LR should decrease with cosine annealing"
# INTERPRETATION: Gradient monitoring tracks ||∇L|| (the L2 norm of all
# gradients concatenated). Explosive gradients (norm >> 1) cause instability.
# Vanishing gradients (norm → 0) cause learning to stall. Gradient clipping
# at max_norm=1.0 prevents explosions. If clipping fires constantly, the
# learning rate is too high or the architecture needs adjustment.
print("\n✓ Checkpoint 3 passed — training dynamics verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export to ONNX with OnnxBridge
# ══════════════════════════════════════════════════════════════════════

# TODO: Instantiate OnnxBridge.
bridge = ____  # Hint: OnnxBridge()

# TODO: Check model compatibility using bridge.check_compatibility.
#   Pass framework="pytorch".
compat = ____  # Hint: bridge.check_compatibility(model, framework="pytorch")
print(f"\n=== ONNX Compatibility ===")
print(f"Compatible: {compat.compatible}")
print(f"Confidence: {compat.confidence}")

# TODO: Export the model to ONNX using bridge.export.
#   output_path="medical_cnn.onnx", framework="pytorch", n_features=None (CNN).
export_result = ____  # Hint: bridge.export(model=model, framework="pytorch", output_path="medical_cnn.onnx", n_features=None)

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

    # TODO: Validate the ONNX model against PyTorch using bridge.validate.
    #   Pass model, onnx_path, sample_input, tolerance=1e-4.
    validation = ____  # Hint: bridge.validate(model=model, onnx_path=export_result.onnx_path, sample_input=sample_input, tolerance=1e-4)

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

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert export_result.success, "ONNX export should succeed"
# INTERPRETATION: OnnxBridge.validate() runs the same sample through both
# PyTorch and the ONNX runtime, comparing outputs. max_diff < 1e-3 means
# the ONNX model is numerically identical to the PyTorch model. This is
# the critical quality gate before production deployment — ONNX runtimes
# (TensorRT, CoreML, ONNX Runtime) may fuse operations differently,
# producing slightly different float values. The tolerance sets the
# acceptable bound for this numerical drift.
if export_result.success and export_result.onnx_path:
    print(f"ONNX export: {export_result.onnx_path}")
print("\n✓ Checkpoint 4 passed — ONNX export and validation complete\n")

print("\n✓ Exercise 8 complete — deep learning foundations + ONNX deployment")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED — MODULE 4 CAPSTONE")
print("═" * 70)
print(f"""
  ✓ XOR proof: linear models cannot learn non-linear boundaries
  ✓ Universal Approximation: one hidden layer + ReLU = any function
  ✓ ResBlock: skip connections solve vanishing gradients in deep networks
  ✓ AdamW + CosineAnnealingLR: the standard modern training recipe
  ✓ OnnxBridge: PyTorch → ONNX with numerical fidelity validation

  MODULE 4 MASTERY — UNSUPERVISED ML:
    Ex 1: Clustering — finding structure without labels
    Ex 2: Anomaly detection — finding the unusual without supervision
    Ex 3: Topic modeling — discovering themes in unstructured text
    Ex 4: Drift monitoring — detecting when the world changes
    Ex 6: Deep learning — automated feature engineering with CNNs
    Ex 8: DL foundations — from linear regression to residual networks

  THE DEEP LEARNING = FEATURE ENGINEERING CONNECTION:
    Traditional ML: you design features → model uses them
    Deep learning:  model LEARNS features from raw data
    Each hidden layer = one level of the feature hierarchy
    Error signal (loss) trains the hierarchy automatically
    This is what "end-to-end learning" means.

  MODULE 5 PREVIEW: LLMs, AI AGENTS AND RAG SYSTEMS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  M5 takes deep learning to its logical extreme: language models.
  You'll build production AI agents with Kaizen, implement RAG
  pipelines, and serve them via Nexus and MCP integration.

  See you in Module 5.
""")
print("═" * 70)
