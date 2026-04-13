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
#   9. Hyperparameter comparison: learning rates
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

torch.manual_seed(42)
np.random.seed(42)
pl.seed_everything(42, workers=True)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load FULL CIFAR-10 and set up kailash-ml engines
# ════════════════════════════════════════════════════════════════════════
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

X_train = torch.stack(
    [train_set[i][0] for i in range(len(train_set))]
)  # (50000, 3, 32, 32)
y_train = torch.tensor(
    [train_set[i][1] for i in range(len(train_set))], dtype=torch.long
)
X_val = torch.stack([val_set[i][0] for i in range(len(val_set))])
y_val = torch.tensor([val_set[i][1] for i in range(len(val_set))], dtype=torch.long)

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


async def setup_engines():
    # TODO: Create ConnectionManager("sqlite:///mlfp05_cnns.db"), initialize,
    #       create ExperimentTracker and ModelRegistry (same pattern as ex_1)
    # Hint: ConnectionManager(url) -> conn.initialize() -> ExperimentTracker(conn)
    conn = ____  # Hint: ConnectionManager("sqlite:///mlfp05_cnns.db")
    await conn.initialize()

    tracker = ____  # Hint: ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name=____,  # Hint: "m5_cnns"
        description=____,  # Hint: "CNN architectures on full CIFAR-10 (50K images)"
    )

    try:
        registry = ____  # Hint: ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (
    X_train.shape[0] == 50000
), f"Expected full 50K CIFAR-10 training set, got {X_train.shape[0]}."
assert X_val.shape[0] == 10000, f"Expected 10K validation, got {X_val.shape[0]}"
assert X_train.shape[1:] == (3, 32, 32), "CIFAR-10 images should be 3x32x32"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: We use all 50K images because CNNs learn hierarchical
# filters. The ExperimentTracker logs every run for quantitative comparison.
print("\n--- Checkpoint 1 passed --- data loaded and engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Plain CNN with BatchNorm and MaxPool
# ════════════════════════════════════════════════════════════════════════
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        # TODO: Build self.features as nn.Sequential:
        #   Conv2d(3->32, k=3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2) ->
        #   Conv2d(32->64, k=3, padding=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2)
        # Hint: nn.Conv2d(in_ch, out_ch, kernel_size, padding=1) for same-size output
        self.features = ____

        # TODO: Build self.head as nn.Sequential:
        #   Flatten -> Linear(64*8*8 -> 128) -> ReLU -> Linear(128 -> n_classes)
        # Hint: after two MaxPool2d(2) on 32x32 input, spatial size is 8x8
        self.head = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — ResNet-style residual block + Squeeze-and-Excitation
# ════════════════════════════════════════════════════════════════════════
# ResBlock computes y = F(x) + x. The skip connection gives gradients
# a direct path during backprop — what enabled 100+ layer networks.
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # TODO: Define conv1, bn1, conv2, bn2
        # Hint: Conv2d(channels, channels, k=3, padding=1) + BatchNorm2d(channels)
        self.conv1 = ____
        self.bn1 = ____
        self.conv2 = ____
        self.bn2 = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # TODO: conv1 -> bn1 -> ReLU, then conv2 -> bn2, then add identity
        # Hint: F.relu(out + identity) as the return
        out = ____
        out = ____
        return F.relu(out + identity)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        # TODO: Build self.fc: Linear(channels->hidden)->ReLU->Linear(hidden->channels)->Sigmoid
        # Hint: squeeze=global avg pool, excite=this MLP, scale=multiply with x
        self.fc = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # TODO: Squeeze to (b, c) with adaptive_avg_pool2d(x, 1).view(b, c)
        # Then excite with self.fc(s).view(b, c, 1, 1) and multiply
        s = ____
        w = self.fc(s).view(b, c, 1, 1)
        return x * w


class ResNetSE(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # TODO: Instantiate block1 = ResBlock(32), se1 = SEBlock(32), block2 = ResBlock(32)
        self.block1 = ____
        self.se1 = ____
        self.block2 = ____
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
assert simple_test(dummy_input).shape == (2, N_CLASSES), "SimpleCNN output shape wrong"
assert resnet_test(dummy_input).shape == (2, N_CLASSES), "ResNetSE output shape wrong"
print(f"SimpleCNN params: {sum(p.numel() for p in simple_test.parameters()):,}")
print(f"ResNetSE params:  {sum(p.numel() for p in resnet_test.parameters()):,}")
# INTERPRETATION: ResNetSE has more parameters but skip connections make
# gradient flow more efficient — faster convergence per parameter.
print("\n--- Checkpoint 2 passed --- both architectures build correctly\n")
del simple_test, resnet_test


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — LightningModule + train SimpleCNN with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
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
        # TODO: Compute cross-entropy loss
        # Hint: F.cross_entropy(logits, y)
        loss = ____
        self._batch_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        if self._batch_losses:
            self.train_losses.append(float(np.mean(self._batch_losses)))
            self._batch_losses = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        # TODO: Accumulate correct prediction count
        # Hint: int((logits.argmax(dim=-1) == y).sum().item())
        self._val_correct += ____
        self._val_total += int(y.size(0))

    def on_validation_epoch_end(self):
        if self._val_total > 0:
            self.val_accs.append(self._val_correct / self._val_total)
            self._val_correct = 0
            self._val_total = 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def get_precision_setting() -> str:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 7:
            print("  GPU with Tensor Cores — using 16-mixed precision")
            return "16-mixed"
    return "32"


def get_accelerator() -> str:
    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


precision = get_precision_setting()
accelerator = get_accelerator()


async def train_lightning_async(
    model: nn.Module,
    name: str,
    lr: float = 1e-3,
    epochs: int = EPOCHS,
) -> tuple[list[float], list[float]]:
    """Train a CNN with Lightning and log metrics to ExperimentTracker."""
    lit = LitCNN(model, lr=lr)
    # TODO: Create pl.Trainer with max_epochs=epochs, accelerator, precision,
    #       and disable progress bar / checkpointing / model summary / logger
    # Hint: pl.Trainer(max_epochs=..., accelerator=..., precision=..., enable_progress_bar=False, ...)
    trainer = ____

    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        # TODO: Log params as a dict with ctx.log_params({...})
        # Hint: await ctx.log_params({"architecture": name, "lr": str(lr), ...})
        await ctx.log_params(
            {
                "architecture": ____,  # Hint: name
                "lr": ____,  # Hint: str(lr)
                "epochs": ____,  # Hint: str(epochs)
                "batch_size": ____,  # Hint: str(BATCH_SIZE)
                "dataset_size": ____,  # Hint: str(len(train_ds))
                "precision": ____,  # Hint: precision
                "accelerator": ____,  # Hint: accelerator
            }
        )

        # TODO: Run trainer.fit with the Lightning module and data loaders
        # Hint: trainer.fit(lit, train_loader, val_loader)
        ____

        for epoch_idx, loss in enumerate(lit.train_losses):
            await ctx.log_metric(____, loss, step=epoch_idx + 1)  # Hint: "train_loss"
        for epoch_idx, acc in enumerate(lit.val_accs):
            await ctx.log_metric(____, acc, step=epoch_idx + 1)  # Hint: "val_accuracy"

        await ctx.log_metrics(
            {
                ____: lit.train_losses[-1],  # Hint: "final_train_loss"
                ____: lit.val_accs[-1],  # Hint: "final_val_accuracy"
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


print("\n== SimpleCNN ==")
simple_cnn = SimpleCNN()
simple_losses, simple_accs = train_lightning(simple_cnn, "SimpleCNN", epochs=EPOCHS)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(simple_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert simple_losses[-1] < simple_losses[0], "Loss should decrease during training"
assert (
    simple_accs[-1] > 0.4
), f"SimpleCNN val accuracy {simple_accs[-1]:.3f} too low — expected > 0.4 on CIFAR-10"
# INTERPRETATION: BatchNorm is crucial — it normalises activations within
# each mini-batch, stabilising training and allowing higher learning rates.
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
assert resnet_accs[-1] > 0.4, f"ResNetSE val acc {resnet_accs[-1]:.3f} too low"

print(f"\n=== Architecture Comparison ===")
print(f"{'Model':>15} {'Final Loss':>12} {'Val Accuracy':>14}")
print("-" * 45)
print(f"{'SimpleCNN':>15} {simple_losses[-1]:>12.4f} {simple_accs[-1]:>13.3f}")
print(f"{'ResNetSE':>15} {resnet_losses[-1]:>12.4f} {resnet_accs[-1]:>13.3f}")
# INTERPRETATION: ResNetSE improves over SimpleCNN via skip connections
# (direct gradient flow) and SE blocks (channel-wise attention). The
# SE block adds <1% parameters but measurably boosts accuracy.
print("\n--- Checkpoint 4 passed --- ResNetSE trained and compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Register both models in the ModelRegistry
# ════════════════════════════════════════════════════════════════════════
async def register_models():
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
        # TODO: Serialize with pickle.dumps(model.state_dict()) and register
        # Hint: registry.register_model(name=name, artifact=model_bytes, metrics=[...])
        model_bytes = ____
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
# INTERPRETATION: ModelRegistry decouples training from deployment —
# training pipeline writes; inference server reads.
print("\n--- Checkpoint 5 passed --- models registered in ModelRegistry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Export the best model to ONNX with OnnxBridge
# ════════════════════════════════════════════════════════════════════════
bridge = OnnxBridge()
resnet_se.eval()
onnx_path = Path("ex_2_resnet_se.onnx")

exported = False
try:
    # TODO: Export with bridge.export(model=resnet_se, framework="pytorch",
    #       output_path=onnx_path, n_features=3*32*32)
    # Hint: OnnxBridge is optimised for tabular; CNN fallback uses torch.onnx
    result = ____
    success = getattr(result, "success", bool(result))
    print(f"\nOnnxBridge.export success: {success}")
    exported = bool(success) and onnx_path.exists()
except Exception as exc:
    print(f"\nOnnxBridge.export raised {type(exc).__name__}: {exc}")

if not exported:
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
# INTERPRETATION: The .onnx file contains architecture AND weights. Any
# ONNX Runtime loads it without PyTorch — key for production deployment.
print("\n--- Checkpoint 6 passed --- model exported to ONNX\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Serve predictions through InferenceServer
# ════════════════════════════════════════════════════════════════════════
async def serve_predictions():
    if not has_registry:
        print("  ModelRegistry not available — skipping InferenceServer demo")
        return None

    # TODO: Create InferenceServer(registry=registry, cache_size=5)
    # Hint: InferenceServer wraps ModelRegistry for LRU-cached model serving
    server = ____

    try:
        # TODO: Warm the cache with ["resnet_se_cifar10"]
        # Hint: await server.warm_cache([model_name])
        await server.warm_cache(____)
        print("  InferenceServer cache warmed with resnet_se_cifar10")
    except Exception as e:
        print(f"  Cache warm skipped (expected for PyTorch models): {e}")

    print("\n  Direct CNN inference on validation samples:")
    resnet_se.eval()
    with torch.no_grad():
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

    print("\n  ONNX Runtime inference (production path):")
    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(str(onnx_path))
        input_name = ort_session.get_inputs()[0].name
        sample_np = sample_images.numpy().astype(np.float32)
        ort_outputs = ort_session.run(None, {input_name: sample_np})
        ort_preds = np.argmax(ort_outputs[0], axis=-1)

        for i, idx in enumerate(indices):
            print(
                f"    ONNX Sample {idx}: pred={class_names[ort_preds[i]]:>10s} | "
                f"true={class_names[y_val[idx].item()]:>10s}"
            )

        match_rate = np.mean(preds.numpy() == ort_preds)
        print(f"\n  PyTorch vs ONNX agreement: {match_rate:.0%}")
    except ImportError:
        print("  onnxruntime not installed — skipping ONNX inference demo")

    return server


server = asyncio.run(serve_predictions())

# ── Checkpoint 7 ─────────────────────────────────────────────────────
resnet_se.eval()
with torch.no_grad():
    all_logits = resnet_se(X_val[:1000])
    all_preds = all_logits.argmax(dim=-1)
    batch_acc = (all_preds == y_val[:1000]).float().mean().item()
assert (
    batch_acc > 0.3
), f"ResNetSE batch accuracy {batch_acc:.3f} on validation is too low"
print(f"  Validation batch accuracy (first 1000): {batch_acc:.3f}")
# INTERPRETATION: InferenceServer separates training from serving. In
# production, training writes to the registry; inference reads from it.
print("\n--- Checkpoint 7 passed --- inference pipeline demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Hyperparameter comparison: learning rates
# ════════════════════════════════════════════════════════════════════════
LR_SWEEP = [5e-4, 1e-3, 3e-3]
HP_EPOCHS = 6
hp_results: dict[str, dict] = {}


async def train_lr_sweep_async(lr: float) -> tuple[list[float], list[float]]:
    """Run one LR sweep trial, logged under its own tracker run."""
    # TODO: Create a fresh ResNetSE() and wrap in LitCNN(hp_model, lr=lr)
    # Hint: hp_model = ResNetSE()
    hp_model = ____
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
                "architecture": ____,  # Hint: "ResNetSE"
                "lr": ____,  # Hint: str(lr)
                "epochs": ____,  # Hint: str(HP_EPOCHS)
                "sweep_type": ____,  # Hint: "learning_rate"
            }
        )

        trainer.fit(lit, train_loader, val_loader)

        for epoch_idx, loss in enumerate(lit.train_losses):
            await ctx.log_metric(____, loss, step=epoch_idx + 1)  # Hint: "train_loss"
        for epoch_idx, acc in enumerate(lit.val_accs):
            await ctx.log_metric(____, acc, step=epoch_idx + 1)  # Hint: "val_accuracy"

        await ctx.log_metrics(
            {
                ____: lit.train_losses[-1],  # Hint: "final_train_loss"
                ____: lit.val_accs[-1],  # Hint: "final_val_accuracy"
            }
        )

    return lit.train_losses, lit.val_accs


print("\n== Learning Rate Comparison ==")
for lr in LR_SWEEP:
    name = f"resnet_se_lr{lr}"
    print(f"\n  Training ResNetSE with lr={lr}...")
    sweep_losses, sweep_accs = asyncio.run(train_lr_sweep_async(lr))
    hp_results[name] = {"lr": lr, "losses": sweep_losses, "accs": sweep_accs}
    print(
        f"    lr={lr}: final_loss={sweep_losses[-1]:.4f}, val_acc={sweep_accs[-1]:.3f}"
    )

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
), f"Should have results for all {len(LR_SWEEP)} LRs"
for name, result in hp_results.items():
    assert result["accs"][-1] > 0.2, f"{name} val_acc={result['accs'][-1]:.3f} too low"
# INTERPRETATION: lr=1e-3 is usually the Adam sweet spot. Too high
# oscillates; too low converges slowly. ExperimentTracker makes this
# comparison quantitative instead of intuition.
print("\n--- Checkpoint 8 passed --- learning rate sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 10 — Visualise all training curves with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()

# TODO: Plot architecture losses and accuracies with viz.training_history
# Hint: viz.training_history(metrics={name: losses_list, ...}, x_label=..., y_label=...)
fig_loss = viz.training_history(
    metrics={"SimpleCNN loss": ____, "ResNetSE loss": ____},
    x_label="Epoch",
    y_label="Training Loss",
)
fig_loss.write_html("ex_2_arch_losses.html")

fig_acc = viz.training_history(
    metrics={"SimpleCNN val acc": ____, "ResNetSE val acc": ____},
    x_label="Epoch",
    y_label="Validation Accuracy",
)
fig_acc.write_html("ex_2_arch_accuracies.html")

fig_lr = viz.training_history(
    metrics={f"lr={r['lr']} loss": r["losses"] for r in hp_results.values()},
    x_label="Epoch",
    y_label="Training Loss",
)
fig_lr.write_html("ex_2_lr_sweep.html")

fig_lr_acc = viz.training_history(
    metrics={f"lr={r['lr']} acc": r["accs"] for r in hp_results.values()},
    x_label="Epoch",
    y_label="Validation Accuracy",
)
fig_lr_acc.write_html("ex_2_lr_sweep_acc.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
import os

assert os.path.exists("ex_2_arch_losses.html"), "Architecture loss HTML should exist"
assert os.path.exists(
    "ex_2_arch_accuracies.html"
), "Architecture accuracy HTML should exist"
assert os.path.exists("ex_2_lr_sweep.html"), "LR sweep HTML should exist"
assert os.path.exists("ex_2_lr_sweep_acc.html"), "LR sweep accuracy HTML should exist"
print("\n--- Checkpoint 9 passed --- visualisations saved\n")

print("\n=== Experiment Summary ===")
print(f"Experiment: {exp_name}")
print(f"Dataset: CIFAR-10 ({len(train_ds):,} training, {len(val_ds):,} validation)")
print(f"\nFinal results:")
print(f"  SimpleCNN:  loss={simple_losses[-1]:.4f}, val_acc={simple_accs[-1]:.3f}")
print(f"  ResNetSE:   loss={resnet_losses[-1]:.4f}, val_acc={resnet_accs[-1]:.3f}")

asyncio.run(conn.close())


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print(
    """
What you've mastered:
  ✓ CNNs with BatchNorm and MaxPool for spatial feature learning
  ✓ ResNet skip connections — why H(x) = F(x) + x keeps gradients healthy
  ✓ SE blocks — channel-wise attention with <1% parameter overhead
  ✓ PyTorch Lightning for clean, device-agnostic training loops
  ✓ OnnxBridge for portable model export (train in PyTorch, deploy anywhere)
  ✓ InferenceServer for registry-backed production serving

Next: In Exercise 3, you'll tackle SEQUENCES with LSTM/GRU and attention
mechanisms — the temporal counterpart to the spatial features CNNs learned.
"""
)
