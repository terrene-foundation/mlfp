# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 2: CNNs, Computer Vision, and Production ML Pipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build CNNs with torch.nn.Conv2d, BatchNorm2d, and MaxPool2d
#   - Implement a ResNet-style residual block and explain why skip
#     connections keep gradients healthy in deep networks
#   - Add a Squeeze-and-Excitation (SE) block — a modern channel-attention
#     enhancement used in many production CV models
#   - Train with a PyTorch Lightning LightningModule (clean, testable loops)
#   - Track every training run with ExperimentTracker (hyperparams, per-epoch
#     metrics) and register trained models in the ModelRegistry
#   - Export the trained model to ONNX via kailash-ml's OnnxBridge
#   - Serve predictions through InferenceServer with warm cache
#   - Compare hyperparameter configurations (learning rates, architectures)
#     using tracked experiments
#   - Understand mixed-precision training and when to use it
#
# PREREQUISITES: M5/ex_1 (autoencoders, PyTorch training basics,
#   ExperimentTracker and ModelRegistry setup).
# ESTIMATED TIME: ~120-150 min
#
# DATASET: CIFAR-10 — 50,000 real 32x32 colour photos across 10 classes
#   (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
#   Downloaded automatically by torchvision and cached to data/mlfp05/cifar10/.
#   We use the FULL 50K training set — CNNs need large datasets to learn
#   generalisable spatial features without overfitting.
#
# TASKS:
#   1. Load full CIFAR-10, set up ExperimentTracker and ModelRegistry
#   2. Build a plain CNN with BatchNorm and MaxPool
#   3. Build ResNet residual blocks and SE attention blocks
#   4. Train SimpleCNN with Lightning and log to ExperimentTracker
#   5. Train ResNetSE and compare against SimpleCNN
#   6. Register both models in the ModelRegistry
#   7. Export the best model to ONNX with OnnxBridge
#   8. Serve predictions through InferenceServer
#   9. Hyperparameter comparison: learning rates and data augmentation
#  10. Visualise all training curves with ModelVisualizer
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
import torchvision

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.bridge.onnx_bridge import OnnxBridge
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.inference_server import InferenceServer
from kailash_ml.engines.model_registry import ModelRegistry
from shared.kailash_helpers import get_device, setup_environment

setup_environment()

# ── Reproducibility ─────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
pl.seed_everything(42, workers=True)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load FULL CIFAR-10 and set up kailash-ml engines
# ════════════════════════════════════════════════════════════════════════
# CNNs need large datasets to learn general spatial filters. Sub-sampling
# to 6K images means the filters memorise patch-level noise instead of
# learning generalisable edge, texture, and shape detectors. We use ALL
# 50K training images and 10K validation images.

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cifar10"
DATA_DIR.mkdir(parents=True, exist_ok=True)

train_set = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
val_set = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Use the FULL dataset — no sub-sampling
X_train = torch.stack(
    [train_set[i][0] for i in range(len(train_set))]
)  # (50000, 3, 32, 32)
y_train = torch.tensor(
    [train_set[i][1] for i in range(len(train_set))], dtype=torch.long
)
X_val = torch.stack([val_set[i][0] for i in range(len(val_set))])  # (10000, 3, 32, 32)
y_val = torch.tensor([val_set[i][1] for i in range(len(val_set))], dtype=torch.long)

# Per-channel normalisation using CIFAR-10 population statistics
cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
cifar_std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
X_train = (X_train - cifar_mean) / cifar_std
X_val = (X_val - cifar_mean) / cifar_std

print(
    f"CIFAR-10: train {tuple(X_train.shape)}, val {tuple(X_val.shape)}, "
    f"classes={len(train_set.classes)}: {train_set.classes}"
)

BATCH_SIZE = 128
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

N_CLASSES = 10
EPOCHS = 8


# Set up kailash-ml engines: ExperimentTracker + ModelRegistry.
# Both share a ConnectionManager for SQLite-backed persistence.
async def setup_engines():
    conn = ConnectionManager("sqlite:///mlfp05_cnns.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name="m5_cnns",
        description="CNN architectures on full CIFAR-10 (50K images)",
    )

    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train.shape[0] == 50000, (
    f"Expected full 50K CIFAR-10 training set, got {X_train.shape[0]}. "
    "CNNs need the full dataset to learn general spatial features."
)
assert X_val.shape[0] == 10000, f"Expected 10K validation, got {X_val.shape[0]}"
assert X_train.shape[1:] == (3, 32, 32), "CIFAR-10 images should be 3x32x32"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: We use all 50K images because CNNs learn hierarchical
# filters — early layers detect edges, deeper layers compose them into
# textures and parts. With only 6K images, the network has too few
# examples to learn general-purpose filters and instead memorises
# training-specific patterns. The ExperimentTracker will log every run
# so we can compare architectures quantitatively.
print("\n--- Checkpoint 1 passed --- data loaded and engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Plain CNN with BatchNorm and MaxPool
# ════════════════════════════════════════════════════════════════════════
# This baseline CNN stacks Conv2d + BatchNorm + ReLU + MaxPool layers.
# BatchNorm stabilises training by normalising activations within each
# mini-batch; MaxPool reduces spatial dimensions by 2x.
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — ResNet-style residual block + Squeeze-and-Excitation
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
        return F.relu(out + identity)  # skip connection


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
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)  # squeeze
        w = self.fc(s).view(b, c, 1, 1)  # excite
        return x * w


class ResNetSE(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
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


# ── Checkpoint 2 ─────────────────────────────────────────────────────
simple_test = SimpleCNN()
resnet_test = ResNetSE()
dummy_input = torch.randn(2, 3, 32, 32)
assert simple_test(dummy_input).shape == (
    2,
    N_CLASSES,
), "SimpleCNN output should be (batch, 10)"
assert resnet_test(dummy_input).shape == (
    2,
    N_CLASSES,
), "ResNetSE output should be (batch, 10)"

simple_params = sum(p.numel() for p in simple_test.parameters())
resnet_params = sum(p.numel() for p in resnet_test.parameters())
print(f"SimpleCNN params: {simple_params:,}")
print(f"ResNetSE params:  {resnet_params:,}")
# INTERPRETATION: ResNetSE has more parameters due to the SE MLP and extra
# ResBlock convolutions, but the skip connections make gradient flow more
# efficient so the network trains faster per-parameter.
print("\n--- Checkpoint 2 passed --- both architectures build correctly\n")
del simple_test, resnet_test


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — LightningModule + train SimpleCNN with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# PyTorch Lightning gives us clean training/validation loops, automatic
# device placement, and a trainer.fit() that handles gradient accumulation,
# checkpointing, and logging out of the box.
#
# MIXED-PRECISION AWARENESS: On GPUs with Tensor Cores (A100, RTX 30xx/40xx),
# you can enable mixed precision by passing precision="16-mixed" to the
# Trainer. This uses float16 for forward/backward (2x faster, 50% less
# memory) while keeping master weights in float32 for numerical stability.
# On CPU or MPS (Mac), mixed precision provides no benefit — the hardware
# lacks float16 compute units. We detect this automatically.
class LitCNN(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_losses: list[float] = []
        self.val_accs: list[float] = []
        self._batch_losses: list[float] = []
        self._val_correct = 0
        self._val_total = 0

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
        self._val_correct += int((logits.argmax(dim=-1) == y).sum().item())
        self._val_total += int(y.size(0))

    def on_validation_epoch_end(self):
        if self._val_total > 0:
            self.val_accs.append(self._val_correct / self._val_total)
            self._val_correct = 0
            self._val_total = 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def get_precision_setting() -> str:
    """Detect whether mixed-precision training is beneficial."""
    if torch.cuda.is_available():
        # Check for Tensor Core support (compute capability >= 7.0)
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 7:
            print("  GPU with Tensor Cores detected — using 16-mixed precision")
            return "16-mixed"
        print("  Older GPU detected — using 32-bit precision")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Apple MPS detected — using 32-bit precision (no fp16 compute units)")
    else:
        print("  CPU detected — using 32-bit precision")
    return "32"


precision = get_precision_setting()


def get_accelerator() -> str:
    """Return the Lightning accelerator string for the current device."""
    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


accelerator = get_accelerator()


async def train_lightning_async(
    model: nn.Module,
    name: str,
    lr: float = 1e-3,
    epochs: int = EPOCHS,
) -> tuple[list[float], list[float]]:
    """Train a CNN with Lightning and log metrics to ExperimentTracker.

    Uses the modern ``tracker.run(...)`` async context manager. On normal
    exit the run is marked COMPLETED; on exception it is marked FAILED.
    """
    lit = LitCNN(model, lr=lr)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        precision=precision,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        await ctx.log_params(
            {
                "architecture": name,
                "lr": str(lr),
                "epochs": str(epochs),
                "batch_size": str(BATCH_SIZE),
                "dataset_size": str(len(train_ds)),
                "precision": precision,
                "accelerator": accelerator,
            }
        )

        # Train (Lightning's trainer.fit is synchronous — this blocks
        # the event loop for the duration, which is fine for a CLI script).
        trainer.fit(lit, train_loader, val_loader)

        # Log per-epoch metrics
        for epoch_idx, loss in enumerate(lit.train_losses):
            await ctx.log_metric("train_loss", loss, step=epoch_idx + 1)
        for epoch_idx, acc in enumerate(lit.val_accs):
            await ctx.log_metric("val_accuracy", acc, step=epoch_idx + 1)

        await ctx.log_metrics(
            {
                "final_train_loss": lit.train_losses[-1],
                "final_val_accuracy": lit.val_accs[-1],
            }
        )

    print(
        f"  [{name}] final train loss={lit.train_losses[-1]:.4f}  "
        f"val acc={lit.val_accs[-1]:.3f}"
    )
    return lit.train_losses, lit.val_accs


def train_lightning(
    model: nn.Module,
    name: str,
    lr: float = 1e-3,
    epochs: int = EPOCHS,
) -> tuple[list[float], list[float]]:
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(train_lightning_async(model, name, lr, epochs))


# ── Train SimpleCNN ────────────────────────────────────────────────────
print("\n== SimpleCNN ==")
simple_cnn = SimpleCNN()
simple_losses, simple_accs = train_lightning(simple_cnn, "SimpleCNN", epochs=EPOCHS)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(simple_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert simple_losses[-1] < simple_losses[0], "Loss should decrease during training"
assert simple_accs[-1] > 0.4, (
    f"SimpleCNN val accuracy {simple_accs[-1]:.3f} too low — "
    "expected > 0.4 on full CIFAR-10 after 8 epochs"
)
# INTERPRETATION: With 50K images the SimpleCNN learns real convolutional
# filters: edge detectors in layer 1, texture detectors in layer 2. The
# accuracy above ~55% shows the model is learning meaningful spatial
# features, not just memorising. BatchNorm is crucial here — without it,
# training would be much slower and less stable.
print("\n--- Checkpoint 3 passed --- SimpleCNN trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Train ResNetSE and compare against SimpleCNN
# ════════════════════════════════════════════════════════════════════════
print("\n== ResNetSE ==")
resnet_se = ResNetSE()
resnet_losses, resnet_accs = train_lightning(resnet_se, "ResNetSE", epochs=EPOCHS)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(resnet_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert resnet_losses[-1] < resnet_losses[0], "ResNetSE loss should decrease"
assert resnet_accs[-1] > 0.4, (
    f"ResNetSE val accuracy {resnet_accs[-1]:.3f} too low — "
    "expected > 0.4 on full CIFAR-10"
)

# Print comparison
print(f"\n=== Architecture Comparison ===")
print(f"{'Model':>15} {'Final Loss':>12} {'Val Accuracy':>14}")
print("-" * 45)
print(f"{'SimpleCNN':>15} {simple_losses[-1]:>12.4f} {simple_accs[-1]:>13.3f}")
print(f"{'ResNetSE':>15} {resnet_losses[-1]:>12.4f} {resnet_accs[-1]:>13.3f}")
# INTERPRETATION: ResNetSE should achieve higher accuracy than SimpleCNN.
# The skip connections let gradients flow directly through the network,
# and the SE block re-weights channels so the most informative feature
# maps get amplified. The improvement is modest on shallow networks but
# becomes dramatic as depth increases (ResNet-50 vs VGG-19, for example).
print("\n--- Checkpoint 4 passed --- ResNetSE trained and compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Register both models in the ModelRegistry
# ════════════════════════════════════════════════════════════════════════
# The ModelRegistry provides versioned model storage with metrics. In
# production, this is how you track which model version is deployed and
# what performance it achieved at training time.


async def register_models():
    """Register trained CNN models with their metrics in the registry."""
    if not has_registry:
        print("  ModelRegistry not available — skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}

    variants = [
        ("simple_cnn_cifar10", simple_cnn, simple_losses[-1], simple_accs[-1]),
        ("resnet_se_cifar10", resnet_se, resnet_losses[-1], resnet_accs[-1]),
    ]

    for name, model, final_loss, final_acc in variants:
        model_bytes = pickle.dumps(model.state_dict())
        version = await registry.register_model(
            name=name,
            artifact=model_bytes,
            metrics=[
                MetricSpec(name="final_loss", value=final_loss),
                MetricSpec(name="val_accuracy", value=final_acc),
                MetricSpec(name="epochs", value=float(EPOCHS)),
                MetricSpec(name="batch_size", value=float(BATCH_SIZE)),
            ],
        )
        model_versions[name] = version
        print(f"  Registered {name}: version={version.version}, acc={final_acc:.3f}")

    return model_versions


model_versions = asyncio.run(register_models())

# ── Checkpoint 5 ─────────────────────────────────────────────────────
if has_registry:
    assert len(model_versions) == 2, "Should register both models"
    assert "resnet_se_cifar10" in model_versions, "ResNetSE should be registered"
    assert "simple_cnn_cifar10" in model_versions, "SimpleCNN should be registered"
# INTERPRETATION: The ModelRegistry gives you a single source of truth
# for model artifacts. Instead of saving .pt files to random directories,
# every model is versioned, tagged with metrics, and queryable. When you
# need to deploy or compare models, the registry has full provenance.
print("\n--- Checkpoint 5 passed --- models registered in ModelRegistry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Export the best model to ONNX with OnnxBridge
# ════════════════════════════════════════════════════════════════════════
# ONNX (Open Neural Network Exchange) is a portable format that lets you
# train in PyTorch and deploy anywhere — ONNX Runtime, TensorRT, mobile.
# OnnxBridge is kailash-ml's wrapper for ONNX export and validation.
bridge = OnnxBridge()
resnet_se.eval()
onnx_path = Path("ex_2_resnet_se.onnx")

exported = False
try:
    result = bridge.export(
        model=resnet_se,
        framework="pytorch",
        output_path=onnx_path,
        n_features=3 * 32 * 32,
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
    sample = torch.randn(1, 3, 32, 32)
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

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert onnx_path.exists(), "ONNX file should exist after export"
onnx_size_kb = onnx_path.stat().st_size // 1024
print(f"Wrote {onnx_path} ({onnx_size_kb} KB)")
# INTERPRETATION: ONNX export freezes the model's computation graph into
# a portable format. The .onnx file contains the architecture AND the
# learned weights. Any ONNX Runtime can load and execute it without
# needing PyTorch installed — key for production deployment.
print("\n--- Checkpoint 6 passed --- model exported to ONNX\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Serve predictions through InferenceServer
# ════════════════════════════════════════════════════════════════════════
# InferenceServer wraps the ModelRegistry for production serving. It
# loads models from the registry, caches them in memory (LRU), and serves
# predictions via predict() and predict_batch(). For CNN models with
# image input, we demonstrate the server setup pattern and use the
# model directly for inference (InferenceServer is optimised for
# tabular models with feature dicts; CNN inference uses raw tensors).
#
# The production deployment path for CNNs:
#   1. Export to ONNX (done in Task 7)
#   2. Load with ONNX Runtime
#   3. Serve via InferenceServer or custom Nexus endpoint
#
# Here we demonstrate the InferenceServer setup and show how CNN
# predictions work through the production pipeline.


async def serve_predictions():
    """Demonstrate InferenceServer setup and CNN prediction pipeline."""
    if not has_registry:
        print("  ModelRegistry not available — skipping InferenceServer demo")
        return None

    # Set up InferenceServer with the ModelRegistry
    server = InferenceServer(registry=registry, cache_size=5)

    # Warm the cache — pre-loads models so first prediction is fast
    try:
        await server.warm_cache(["resnet_se_cifar10"])
        print("  InferenceServer cache warmed with resnet_se_cifar10")
    except Exception as e:
        print(f"  Cache warm skipped (expected for PyTorch models): {e}")

    # For CNN models, we demonstrate the direct inference pipeline
    # that InferenceServer would wrap in production with ONNX Runtime
    print("\n  Direct CNN inference on validation samples:")
    resnet_se.eval()
    with torch.no_grad():
        # Pick 5 random validation images
        indices = [0, 100, 500, 2000, 5000]
        sample_images = X_val[indices]
        logits = resnet_se(sample_images)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        class_names = train_set.classes
        for i, idx in enumerate(indices):
            pred_class = class_names[preds[i].item()]
            true_class = class_names[y_val[idx].item()]
            confidence = probs[i][preds[i]].item()
            status = "CORRECT" if preds[i].item() == y_val[idx].item() else "WRONG"
            print(
                f"    Sample {idx}: pred={pred_class:>10s} "
                f"(conf={confidence:.2f}) | true={true_class:>10s} [{status}]"
            )

    # ONNX Runtime inference — the production path
    print("\n  ONNX Runtime inference (production path):")
    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(str(onnx_path))
        input_name = ort_session.get_inputs()[0].name

        sample_np = sample_images.numpy().astype(np.float32)
        ort_outputs = ort_session.run(None, {input_name: sample_np})
        ort_logits = ort_outputs[0]
        ort_preds = np.argmax(ort_logits, axis=-1)

        for i, idx in enumerate(indices):
            pred_class = class_names[ort_preds[i]]
            true_class = class_names[y_val[idx].item()]
            print(
                f"    ONNX Sample {idx}: pred={pred_class:>10s} | true={true_class:>10s}"
            )

        # Verify ONNX matches PyTorch
        pytorch_preds_np = preds.numpy()
        match_rate = np.mean(pytorch_preds_np == ort_preds)
        print(f"\n  PyTorch vs ONNX agreement: {match_rate:.0%}")
    except ImportError:
        print("  onnxruntime not installed — skipping ONNX inference demo")

    return server


server = asyncio.run(serve_predictions())

# ── Checkpoint 7 ─────────────────────────────────────────────────────
# Verify the model produces reasonable predictions on validation data
resnet_se.eval()
with torch.no_grad():
    all_logits = resnet_se(X_val[:1000])
    all_preds = all_logits.argmax(dim=-1)
    batch_acc = (all_preds == y_val[:1000]).float().mean().item()
assert (
    batch_acc > 0.3
), f"ResNetSE batch accuracy {batch_acc:.3f} on validation is too low"
print(f"  Validation batch accuracy (first 1000): {batch_acc:.3f}")
# INTERPRETATION: The InferenceServer pattern — register model, load from
# registry, serve via predict() — separates model training from model
# serving. In production, the training pipeline writes to the registry
# and the inference server reads from it. This decoupling means you can
# update the model without restarting the server.
print("\n--- Checkpoint 7 passed --- inference pipeline demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Hyperparameter comparison: learning rates and architectures
# ════════════════════════════════════════════════════════════════════════
# How does the learning rate affect CNN training? Too high and the model
# oscillates; too low and it converges slowly. We train ResNetSE with
# three different learning rates and compare convergence curves.
# Every trial is logged to ExperimentTracker for systematic comparison.

LR_SWEEP = [5e-4, 1e-3, 3e-3]
HP_EPOCHS = 6  # Fewer epochs for the sweep — enough to see the trend
hp_results: dict[str, dict] = {}


async def train_lr_sweep_async(lr: float) -> tuple[list[float], list[float]]:
    """Run one LR sweep trial, logged under its own tracker run."""
    hp_model = ResNetSE()
    lit = LitCNN(hp_model, lr=lr)
    trainer = pl.Trainer(
        max_epochs=HP_EPOCHS,
        accelerator=accelerator,
        precision=precision,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    async with tracker.run(
        experiment_name=exp_name, run_name=f"hp_sweep_lr_{lr}"
    ) as ctx:
        await ctx.log_params(
            {
                "architecture": "ResNetSE",
                "lr": str(lr),
                "epochs": str(HP_EPOCHS),
                "sweep_type": "learning_rate",
            }
        )

        trainer.fit(lit, train_loader, val_loader)

        for epoch_idx, loss in enumerate(lit.train_losses):
            await ctx.log_metric("train_loss", loss, step=epoch_idx + 1)
        for epoch_idx, acc in enumerate(lit.val_accs):
            await ctx.log_metric("val_accuracy", acc, step=epoch_idx + 1)

        await ctx.log_metrics(
            {
                "final_train_loss": lit.train_losses[-1],
                "final_val_accuracy": lit.val_accs[-1],
            }
        )

    return lit.train_losses, lit.val_accs


print("\n== Learning Rate Comparison ==")
for lr in LR_SWEEP:
    name = f"resnet_se_lr{lr}"
    print(f"\n  Training ResNetSE with lr={lr}...")
    sweep_losses, sweep_accs = asyncio.run(train_lr_sweep_async(lr))
    hp_results[name] = {
        "lr": lr,
        "losses": sweep_losses,
        "accs": sweep_accs,
    }
    print(
        f"    lr={lr}: final_loss={sweep_losses[-1]:.4f}, val_acc={sweep_accs[-1]:.3f}"
    )

# Print comparison table
print("\n=== Learning Rate Comparison ===")
print(f"{'Learning Rate':>15} {'Final Loss':>12} {'Val Accuracy':>14}")
print("-" * 45)
for name, result in hp_results.items():
    print(
        f"{result['lr']:>15.4f} {result['losses'][-1]:>12.4f} {result['accs'][-1]:>13.3f}"
    )

best_config = max(hp_results.items(), key=lambda x: x[1]["accs"][-1])
print(f"\nBest LR: {best_config[1]['lr']} (val_acc={best_config[1]['accs'][-1]:.3f})")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(hp_results) == len(
    LR_SWEEP
), f"Should have results for all {len(LR_SWEEP)} learning rates"
# The very high LR should not beat the very low LR consistently
# (both are suboptimal in different ways), but all should learn something
for name, result in hp_results.items():
    assert result["accs"][-1] > 0.2, (
        f"{name} val_acc={result['accs'][-1]:.3f} is too low — even suboptimal "
        "LR should learn basic features from 50K images"
    )
# INTERPRETATION: The learning rate is the most important hyperparameter
# in deep learning. lr=5e-4 converges slowly but stably. lr=1e-3 is
# usually the sweet spot for Adam. lr=3e-3 converges faster initially
# but may oscillate near the optimum. The ExperimentTracker records
# let you make this comparison quantitatively instead of guessing.
print("\n--- Checkpoint 8 passed --- learning rate sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 10 — Visualise all training curves with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════
# Three plots: (1) architecture comparison losses, (2) architecture
# comparison accuracies, (3) learning rate sweep. All use ModelVisualizer
# for consistent professional-quality charts.

viz = ModelVisualizer()

# Plot 1: Architecture comparison — training losses
fig_loss = viz.training_history(
    metrics={
        "SimpleCNN loss": simple_losses,
        "ResNetSE loss": resnet_losses,
    },
    x_label="Epoch",
    y_label="Training Loss",
)
fig_loss.write_html("ex_2_arch_losses.html")
print("Architecture loss comparison saved to ex_2_arch_losses.html")

# Plot 2: Architecture comparison — validation accuracies
fig_acc = viz.training_history(
    metrics={
        "SimpleCNN val acc": simple_accs,
        "ResNetSE val acc": resnet_accs,
    },
    x_label="Epoch",
    y_label="Validation Accuracy",
)
fig_acc.write_html("ex_2_arch_accuracies.html")
print("Architecture accuracy comparison saved to ex_2_arch_accuracies.html")

# Plot 3: Learning rate sweep
fig_lr = viz.training_history(
    metrics={f"lr={r['lr']} loss": r["losses"] for r in hp_results.values()},
    x_label="Epoch",
    y_label="Training Loss",
)
fig_lr.write_html("ex_2_lr_sweep.html")
print("Learning rate sweep saved to ex_2_lr_sweep.html")

# Plot 4: Learning rate sweep — accuracies
fig_lr_acc = viz.training_history(
    metrics={f"lr={r['lr']} acc": r["accs"] for r in hp_results.values()},
    x_label="Epoch",
    y_label="Validation Accuracy",
)
fig_lr_acc.write_html("ex_2_lr_sweep_acc.html")
print("Learning rate accuracy sweep saved to ex_2_lr_sweep_acc.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
import os

assert os.path.exists("ex_2_arch_losses.html"), "Architecture loss HTML should exist"
assert os.path.exists(
    "ex_2_arch_accuracies.html"
), "Architecture accuracy HTML should exist"
assert os.path.exists("ex_2_lr_sweep.html"), "LR sweep HTML should exist"
assert os.path.exists("ex_2_lr_sweep_acc.html"), "LR sweep accuracy HTML should exist"
print("\n--- Checkpoint 9 passed --- visualisations saved\n")


# ── Print summary ─────────────────────────────────────────────────────
print("\n=== Experiment Summary ===")
print(f"Experiment: {exp_name}")
print(f"Dataset: CIFAR-10 ({len(train_ds):,} training, {len(val_ds):,} validation)")
print(f"Main training epochs: {EPOCHS}")
print(f"Precision: {precision}")
print(f"Accelerator: {accelerator}")
print(f"\nArchitecture comparison:")
print(f"  SimpleCNN: loss={simple_losses[-1]:.4f}, acc={simple_accs[-1]:.3f}")
print(f"  ResNetSE:  loss={resnet_losses[-1]:.4f}, acc={resnet_accs[-1]:.3f}")
print(f"\nLearning rate sweep (ResNetSE, {HP_EPOCHS} epochs):")
for name, r in hp_results.items():
    print(f"  lr={r['lr']}: loss={r['losses'][-1]:.4f}, acc={r['accs'][-1]:.3f}")
print(f"\nBest config: {best_config[1]['lr']} (acc={best_config[1]['accs'][-1]:.3f})")
print(f"ONNX model: {onnx_path} ({onnx_size_kb} KB)")

# Clean up database connection
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  CNN ARCHITECTURES:
  [x] SimpleCNN: Conv2d + BatchNorm + MaxPool — the baseline that learns
      edge and texture detectors from raw pixels
  [x] ResBlock: y = F(x) + x skip connections give gradients a highway
      through deep networks, solving the degradation problem
  [x] SEBlock: channel attention re-weights feature maps by importance —
      "which filters matter for THIS specific image?"
  [x] ResNetSE: combining residual learning with channel attention for
      measurably higher accuracy at modest parameter cost

  ML ENGINEERING PIPELINE:
  [x] Trained on FULL CIFAR-10 (50K images, not a subsample) — real
      CNNs need large datasets to learn generalisable spatial features
  [x] Tracked every run with ExperimentTracker (hyperparams, per-epoch
      loss and accuracy, sweep metadata) for reproducible research
  [x] Registered trained models in ModelRegistry with versioned metrics —
      the single source of truth for deployment decisions
  [x] Exported to ONNX with OnnxBridge for portable deployment
  [x] Demonstrated InferenceServer setup and CNN prediction pipeline —
      registry-backed model loading with LRU cache
  [x] Compared learning rates (5e-4, 1e-3, 3e-3) systematically with
      tracked experiments, not by eyeballing print output

  PRODUCTION AWARENESS:
  [x] Mixed precision: fp16 on Tensor Core GPUs (2x speed, 50% less
      memory), fp32 on CPU/MPS — detected automatically
  [x] ONNX Runtime inference: the production path for serving CNNs
      without PyTorch installed
  [x] Model versioning: train -> register -> serve -> update without
      restarting the serving infrastructure

  KEY INSIGHT: The gap between a working CNN and a production CNN is
  not the architecture — it is the engineering. ExperimentTracker for
  reproducibility, ModelRegistry for versioning, OnnxBridge for
  portability, InferenceServer for serving. The model is just one
  artifact in a pipeline of artifacts.

  Next: In Exercise 3, you'll build RNNs, LSTMs, and GRUs for sequence
  modelling on real Singapore stock-market data, learning why recurrent
  architectures handle variable-length sequences that CNNs cannot...
"""
)
