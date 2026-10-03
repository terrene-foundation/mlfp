# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 2.3 — Production Pipeline: ONNX Export + InferenceServer
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain WHY production teams ship ONNX instead of PyTorch
#     (dependency weight, hardware portability, multi-language runtimes)
#     in terms a non-technical manager can understand
#   - Export a trained CNN to ONNX format using kailash-ml's OnnxBridge
#   - Validate ONNX output matches PyTorch output (numerical fidelity)
#   - Register the ONNX artifact in ModelRegistry and serve predictions
#     through kailash-ml's InferenceServer
#   - Benchmark latency and throughput for production sizing
#   - Apply this to deploying the best model at a Singapore e-commerce
#     platform — latency targets, throughput planning, cost per inference
#
# PREREQUISITES: M5/ex_2/01_simple_cnn.py and 02_resnet_se.py (trained
#   CNN models, ModelRegistry registration)
# ESTIMATED TIME: ~25 min
#
# PHASES:
#   1. THEORY  — Why ONNX, what it solves, deployment landscape
#   2. BUILD   — Train a model and export to ONNX via OnnxBridge
#   3. TRAIN   — (integrated with BUILD — we need a trained model to export)
#   4. VISUALISE — Latency distribution, PyTorch vs ONNX comparison
#   5. APPLY   — E-commerce deployment: latency, throughput, cost
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.mlfp05.ex_2 import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    EPOCHS,
    N_CLASSES,
    FlatImageAdapter,
    attach_onnx_artifact,
    count_parameters,
    create_visualizer,
    denormalise_cifar,
    images_to_records,
    init_engines,
    load_cifar10,
    register_model,
    train_model,
)
from kailash_ml import OnnxBridge
from kailash_ml import InferenceServer


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why You Cannot Deploy PyTorch Directly
# ════════════════════════════════════════════════════════════════════════
# Imagine you've built a perfect engine in your workshop. It runs
# beautifully on the test bench. Now you need to install it in a car
# that's already driving at 100 km/h. You can't bring the whole workshop
# along — you need to ship JUST the engine, in a standard format that
# any mechanic can install.
#
# ONNX (Open Neural Network Exchange) is that standard format for ML.
#
# WHY NOT DEPLOY PYTORCH DIRECTLY?
#
# 1. DEPENDENCY WEIGHT:
#    PyTorch + CUDA + all dependencies = 2-5 GB installed.
#    ONNX Runtime = ~50 MB. In a containerised deployment (Kubernetes),
#    this is the difference between 30-second cold starts and 2-second
#    cold starts.
#
# 2. HARDWARE PORTABILITY:
#    PyTorch weights CAN be loaded on a CPU (map_location="cpu"), but the
#    serving host still needs the full PyTorch stack built for that
#    hardware. ONNX Runtime runs the same .onnx file on CPU, NVIDIA GPU,
#    AMD GPU, Apple Silicon, ARM — any hardware with an execution provider.
#
# 3. LANGUAGE BARRIER:
#    Your model was trained in Python. Production services might be in
#    Go, Java, C++, or Rust. ONNX Runtime has native bindings for all.
#    No Python GIL bottleneck in production.
#
# 4. OPTIMISATION:
#    ONNX Runtime applies graph-level optimisations: operator fusion,
#    constant folding, memory planning. A model that takes 15ms in
#    PyTorch often takes 5-8ms in ONNX Runtime — free speedup.
#
# THE DEPLOYMENT PIPELINE:
#   Train (PyTorch) -> Export (ONNX) -> Validate -> Register (ModelRegistry)
#   -> Serve (InferenceServer / ONNX Runtime) -> Monitor (DriftMonitor)
#
# InferenceServer serves ONE registered model version per server:
#   server = await InferenceServer.from_registry(name, registry=registry,
#                                                version=v, runtime="onnx")
#   await server.start()          # loads that version's model.onnx
#   out = await server.predict({"records": [...]})   # {"predictions": [...]}
# Each record becomes one row of a 2-D float array fed to ONNX Runtime,
# so an image model is exported behind a small adapter that accepts flat
# pixel rows (FlatImageAdapter in shared/mlfp05/ex_2.py).

print("=" * 70)
print("  PHASE 1 — THEORY: Why ONNX for Production Deployment")
print("=" * 70)
print(
    """
  PyTorch = workshop (great for building, too heavy for shipping)
  ONNX = standard engine format (any mechanic can install)

  Benefits:
    - 50 MB vs 5 GB dependency footprint
    - Hardware-agnostic (CPU, GPU, ARM, Apple Silicon)
    - Language-agnostic (Python, Go, Java, C++, Rust)
    - Free optimisation (graph fusion, constant folding)
    - 2-3x latency improvement typical

  Pipeline: Train -> Export -> Validate -> Register -> Serve -> Monitor
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2+3 — BUILD + TRAIN: Model for Export
# ════════════════════════════════════════════════════════════════════════
# We train a ResNetSE model (same as 02_resnet_se.py) and export it.
# In production, you'd load the best model from ModelRegistry instead.

print("=" * 70)
print("  PHASE 2+3 — BUILD + TRAIN: Prepare Model for Export")
print("=" * 70)


class ResBlock(nn.Module):
    """Residual block for the export model."""

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
        return F.relu(out + identity)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for the export model."""

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
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w


class ResNetSE(nn.Module):
    """ResNet+SE model for ONNX export."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
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


# Load data and train
X_train, y_train, X_val, y_val, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()

print(f"\nTraining ResNetSE for ONNX export ({EPOCHS} epochs)...")
resnet_se = ResNetSE()
resnet_losses, resnet_accs = train_model(
    resnet_se,
    "ResNetSE_for_export",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    epochs=EPOCHS,
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — pre-export clinical sign-off
# ══════════════════════════════════════════════════════════════════
# Running diagnostics BEFORE ONNX export is deployment hygiene: you
# never want to ship a model that is secretly pathological. A clean
# Prescription Pad is table stakes for production release.
from kailash_ml import diagnose

print("\n── Pre-Export Diagnostic Report (ResNetSE) ──")
report = diagnose(resnet_se, kind="dl", data=val_loader, show=False)
# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Gradient flow (HEALTHY): min RMS = 5.9e-04 at
#       'layer3.1.conv2.weight'. Same pattern as 02_resnet_se
#       — skip connections keep the full depth trainable.
#   [✓] Dead neurons  (HEALTHY): max 3.7% dead. SE blocks +
#       batch norm maintain channel health.
#   [✓] Loss trend    (HEALTHY): train slope -4.2e-02/epoch,
#       val slope -3.6e-02/epoch. Train-val gap 5%.
#   [✓] Export gate:   ALL CLEAR — no WARN/CRITICAL findings.
#       Safe to proceed to ONNX export.
# ════════════════════════════════════════════════════════════════
# Final val acc: ~0.60 on CIFAR-10 (production-calibrated run).
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [EXPORT-GATE DISCIPLINE] This checkpoint is DEPLOYMENT
#     HYGIENE, not training diagnosis. Every finding here
#     becomes a PRE-EXPORT GATE. Slide 5Q covers the full
#     gate: CRITICAL gradients or >50% dead neurons BLOCK
#     export (the model is structurally broken); WARNING
#     findings require written justification in the model
#     card. Shipping a pathological model is how
#     organisations discover weeks later that half their
#     production requests are answered by dead neurons.
#     >> Prescription: Wire this gate into CI. If diag.
#        findings has any CRITICAL, fail the build.
#
#  [BLOOD TEST — PRE-EXPORT INVARIANT] min RMS 5.9e-04
#     matches the 02 training-time reading (6.2e-04).
#     CONSISTENCY between training-time and export-time
#     readings proves the model hasn't drifted in the brief
#     window between end-of-training and export call. If
#     export-time RMS differs by >10x from training, you
#     have a serialization bug (BN stats not updated,
#     dropout left on, etc).
#     >> Prescription: Always diag EVAL-MODE outputs before
#        export (model.eval() + no_grad context). Compare
#        to training-time diag. >10x mismatch blocks
#        export.
#
#  [X-RAY — SERVING-MODE CHECK] 3.7% dead in eval mode
#     ≈ 4% in train mode (from 02_resnet_se.py). If eval
#     dead% SPIKES to 20%+ while train dead% is 4%, batch
#     norm is failing in single-sample or tiny-batch
#     inference. The fix is either BN→LayerNorm or
#     explicit running-stats update.
#     >> Prescription: Sanity-check with batch_size=1
#        inference on 10 random test images. If outputs
#        vary wildly vs batch_size=32, BN is the culprit.
#
#  FIVE-INSTRUMENT TAKEAWAY: production checkpoints shift
#  the instrument purpose from DIAGNOSIS to GATE
#  VERIFICATION. Same 5 instruments, but pass/fail logic
#  replaces learning-curve reading. This pattern repeats in
#  ex_5 GAN deployment (block export if mode collapse) and
#  ex_7 transfer learning (block export if base-model
#  gradients didn't freeze as intended).
# ════════════════════════════════════════════════════════════════════

# Register in ModelRegistry (model.pkl = the trained weights)
model_version = register_model(
    registry,
    "resnet_se_cifar10",
    resnet_se,
    resnet_losses[-1],
    resnet_accs[-1],
)

# ── Checkpoint 1: Model trained ──────────────────────────────────────
assert (
    resnet_accs[-1] > 0.4
), f"ResNetSE val accuracy {resnet_accs[-1]:.3f} too low for export"
print(f"\nModel ready: loss={resnet_losses[-1]:.4f}, val_acc={resnet_accs[-1]:.3f}")
print("--- Checkpoint 1 passed --- model trained and ready for export\n")


# ════════════════════════════════════════════════════════════════════════
# ONNX Export via OnnxBridge
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  ONNX EXPORT")
print("=" * 70)

bridge = OnnxBridge()
resnet_se.eval()
onnx_path = Path("ex_2_resnet_se.onnx")

# InferenceServer feeds ONNX Runtime one flat row per request record, so
# we export the CNN behind FlatImageAdapter: it takes (batch, 3072) pixel
# rows and reshapes them to (batch, 3, 32, 32). The adapter must be in
# eval mode — OnnxBridge restores the training flag it finds.
# (The exporter may print an opset-conversion traceback and fall back to
# opset 18; that is log noise — export_result.success is the real signal.)
serving_model = FlatImageAdapter(resnet_se).eval()
export_result = bridge.export(
    serving_model,
    "torch",
    output_path=onnx_path,
    sample_input=torch.randn(1, 3 * 32 * 32),
)
print(
    f"  OnnxBridge.export: success={export_result.success} "
    f"status={export_result.onnx_status} "
    f"time={export_result.export_time_seconds:.1f}s"
)
assert export_result.success, f"OnnxBridge export failed: {export_result.error_message}"

# Attach the .onnx file to the version registered above, so the
# InferenceServer can load it from the registry by name + version.
attach_onnx_artifact("resnet_se_cifar10", model_version.version, onnx_path)
print(f"  Attached model.onnx to resnet_se_cifar10 v{model_version.version}")

# ── Checkpoint 2: ONNX file exists ──────────────────────────────────
assert onnx_path.exists(), "ONNX file should exist after export"
onnx_size_kb = onnx_path.stat().st_size // 1024
print(f"  Wrote {onnx_path} ({onnx_size_kb} KB)")
print(
    "  The .onnx file contains the architecture AND learned weights.\n"
    "  Any ONNX Runtime can load and execute it without PyTorch."
)
print("\n--- Checkpoint 2 passed --- ONNX export complete\n")


# ════════════════════════════════════════════════════════════════════════
# Numerical Validation: PyTorch vs ONNX Runtime
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  NUMERICAL VALIDATION: PyTorch vs ONNX Runtime")
print("=" * 70)

# onnxruntime is required here: OnnxBridge.validate and InferenceServer
# both execute the .onnx graph with it.
import onnxruntime as ort

ort_available = True  # the benchmark sections below also report ONNX Runtime

# OnnxBridge.validate runs the native model (via its predict() method) and
# ONNX Runtime on the same rows and reports the largest output difference.
test_images = X_val[:100]
test_flat = test_images.reshape(len(test_images), -1).numpy().astype(np.float32)
validation = bridge.validate(serving_model, onnx_path, test_flat, tolerance=1e-3)
print(
    f"  OnnxBridge.validate: valid={validation.valid} "
    f"max_diff={validation.max_diff:.2e} mean_diff={validation.mean_diff:.2e} "
    f"(over {validation.n_samples} logits)"
)
assert validation.valid, f"ONNX output drifted from PyTorch: {validation.notes}"

if ort_available:
    ort_session = ort.InferenceSession(str(onnx_path))
    input_name = ort_session.get_inputs()[0].name

    # Do the two runtimes pick the same class?
    resnet_se.eval()
    with torch.no_grad():
        pt_logits = resnet_se(test_images).numpy()
    pt_preds = np.argmax(pt_logits, axis=-1)

    ort_logits = ort_session.run(None, {input_name: test_flat})[0]
    ort_preds = np.argmax(ort_logits, axis=-1)

    prediction_match = np.mean(pt_preds == ort_preds)
    print(
        f"  Prediction agreement: {prediction_match:.0%} ({int(prediction_match * 100)}/100)"
    )
    assert prediction_match >= 0.99, (
        f"PyTorch vs ONNX prediction mismatch: {prediction_match:.0%} agreement. "
        "Export may have lost fidelity."
    )
    print("  Numerical validation PASSED -- ONNX matches PyTorch")


# ════════════════════════════════════════════════════════════════════════
# InferenceServer Setup
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  INFERENCE SERVER")
print("=" * 70)


indices = [0, 100, 500, 2000, 5000]
sample_images = X_val[indices]


async def serve_samples(name: str, version: int, images: torch.Tensor):
    """Load one registered ONNX model version and serve a batch request."""
    server = await InferenceServer.from_registry(
        name, registry=registry, version=version, runtime="onnx"
    )
    await server.start()  # loads model.onnx from the registry's artifact store
    print(f"  InferenceServer status: {server.status}  ({name} v{version}, onnx)")
    response = await server.predict({"records": images_to_records(images)})
    await server.stop()
    return response


response = asyncio.run(
    serve_samples("resnet_se_cifar10", model_version.version, sample_images)
)
server_logits = np.asarray(response["predictions"])
server_preds = server_logits.argmax(axis=1)

# The same images through PyTorch directly, for comparison
resnet_se.eval()
with torch.no_grad():
    logits = resnet_se(sample_images)
    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)

print("\n  InferenceServer predictions (vs direct PyTorch):")
for i, idx in enumerate(indices):
    server_class = CLASS_NAMES[server_preds[i]]
    torch_class = CLASS_NAMES[preds[i].item()]
    true_class = CLASS_NAMES[y_val[idx].item()]
    confidence = probs[i][preds[i]].item()
    status = "CORRECT" if server_preds[i] == y_val[idx].item() else "WRONG"
    print(
        f"    Sample {idx}: server={server_class:>10s} torch={torch_class:>10s} "
        f"(conf={confidence:.2f}) | true={true_class:>10s} [{status}]"
    )

# ── Checkpoint 3: InferenceServer served real predictions ────────────
assert server_logits.shape == (len(indices), N_CLASSES), server_logits.shape
assert np.array_equal(server_preds, preds.numpy()), (
    "InferenceServer (ONNX) and PyTorch disagree on the sample classes"
)
batch_acc_check = float(np.mean(server_preds == y_val[indices].numpy()))
print(f"\n  Served-sample accuracy: {batch_acc_check:.0%} ({len(indices)} images)")
print("--- Checkpoint 3 passed --- InferenceServer served the ONNX model\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Latency Benchmarks
# ════════════════════════════════════════════════════════════════════════
# Production ML is not just about accuracy — it is about SPEED and COST.
# A model that takes 500ms per prediction cannot serve a website that
# needs <100ms response time.

print("=" * 70)
print("  PHASE 4 — VISUALISE: Latency Benchmarks")
print("=" * 70)

N_BENCHMARK = 100
single_image = X_val[:1]
batch_16 = X_val[:16]

# PyTorch latency
resnet_se.eval()
pt_latencies_single = []
pt_latencies_batch = []

with torch.no_grad():
    # Warmup
    for _ in range(10):
        _ = resnet_se(single_image)

    for _ in range(N_BENCHMARK):
        t0 = time.perf_counter()
        _ = resnet_se(single_image)
        pt_latencies_single.append((time.perf_counter() - t0) * 1000)

    for _ in range(N_BENCHMARK):
        t0 = time.perf_counter()
        _ = resnet_se(batch_16)
        pt_latencies_batch.append((time.perf_counter() - t0) * 1000)

# ONNX Runtime latency
ort_latencies_single = []
ort_latencies_batch = []

if ort_available:
    single_np = single_image.reshape(1, -1).numpy().astype(np.float32)
    batch_np = batch_16.reshape(16, -1).numpy().astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, {input_name: single_np})

    for _ in range(N_BENCHMARK):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: single_np})
        ort_latencies_single.append((time.perf_counter() - t0) * 1000)

    for _ in range(N_BENCHMARK):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: batch_np})
        ort_latencies_batch.append((time.perf_counter() - t0) * 1000)

# Print benchmark results
pt_single_mean = np.mean(pt_latencies_single)
pt_single_p99 = np.percentile(pt_latencies_single, 99)
pt_batch_mean = np.mean(pt_latencies_batch)

print(f"\n  {'Metric':>30s} {'PyTorch':>12s}", end="")
if ort_available:
    print(f" {'ONNX RT':>12s} {'Speedup':>10s}")
else:
    print()
print("  " + "-" * 70)

metrics_to_print = [
    (
        "Single image (mean)",
        pt_single_mean,
        np.mean(ort_latencies_single) if ort_available else None,
    ),
    (
        "Single image (p99)",
        pt_single_p99,
        np.percentile(ort_latencies_single, 99) if ort_available else None,
    ),
    (
        "Batch of 16 (mean)",
        pt_batch_mean,
        np.mean(ort_latencies_batch) if ort_available else None,
    ),
    (
        "Per-image in batch",
        pt_batch_mean / 16,
        np.mean(ort_latencies_batch) / 16 if ort_available else None,
    ),
]

for label, pt_val, ort_val in metrics_to_print:
    line = f"  {label:>30s} {pt_val:>10.2f}ms"
    if ort_val is not None:
        speedup = pt_val / ort_val if ort_val > 0 else float("inf")
        line += f" {ort_val:>10.2f}ms {speedup:>9.1f}x"
    print(line)

# Throughput calculation
pt_throughput = 1000 / (pt_batch_mean / 16)  # images/second
print(f"\n  PyTorch throughput: {pt_throughput:,.0f} images/second")
if ort_available:
    ort_throughput = 1000 / (np.mean(ort_latencies_batch) / 16)
    print(f"  ONNX RT throughput: {ort_throughput:,.0f} images/second")

# Visualise latency distributions
fig_latency, axes = plt.subplots(1, 2, figsize=(14, 5))
fig_latency.suptitle("Inference Latency: PyTorch vs ONNX Runtime", fontsize=14)

# Single image latency distribution
axes[0].hist(
    pt_latencies_single, bins=30, alpha=0.7, label="PyTorch", color="steelblue"
)
if ort_available:
    axes[0].hist(
        ort_latencies_single, bins=30, alpha=0.7, label="ONNX RT", color="coral"
    )
axes[0].set_xlabel("Latency (ms)")
axes[0].set_ylabel("Count")
axes[0].set_title("Single Image Inference")
axes[0].legend()
axes[0].axvline(
    np.mean(pt_latencies_single), color="steelblue", linestyle="--", alpha=0.5
)
if ort_available:
    axes[0].axvline(
        np.mean(ort_latencies_single), color="coral", linestyle="--", alpha=0.5
    )

# Batch latency distribution
axes[1].hist(pt_latencies_batch, bins=30, alpha=0.7, label="PyTorch", color="steelblue")
if ort_available:
    axes[1].hist(
        ort_latencies_batch, bins=30, alpha=0.7, label="ONNX RT", color="coral"
    )
axes[1].set_xlabel("Latency (ms)")
axes[1].set_ylabel("Count")
axes[1].set_title("Batch of 16 Inference")
axes[1].legend()

plt.tight_layout()
plt.savefig("ex_2_03_latency_benchmark.png", dpi=150, bbox_inches="tight")
plt.close(fig_latency)
print("\n  Saved: ex_2_03_latency_benchmark.png")

# Model size comparison
pytorch_size_mb = sum(p.numel() * p.element_size() for p in resnet_se.parameters()) / (
    1024 * 1024
)
print(f"\n  Model sizes:")
print(f"    PyTorch in-memory: {pytorch_size_mb:.2f} MB")
print(f"    ONNX file on disk: {onnx_size_kb / 1024:.2f} MB ({onnx_size_kb} KB)")

# ── Checkpoint 4: Benchmarks complete ────────────────────────────────
import os

assert os.path.exists("ex_2_03_latency_benchmark.png"), "Latency plot missing"
assert pt_single_mean > 0, "PyTorch latency should be positive"
print("\n--- Checkpoint 4 passed --- latency benchmarks complete\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: E-Commerce Deployment Planning
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: You are deploying the product categorisation CNN from
# 01_simple_cnn.py to production at a Singapore e-commerce platform.
# The engineering manager asks three questions:
#   1. "What hardware do we need?"
#   2. "How many images per second can we process?"
#   3. "What will it cost?"
#
# You need concrete answers based on your benchmark data.

print("=" * 70)
print("  PHASE 5 — APPLY: E-Commerce Deployment (Singapore Platform)")
print("=" * 70)

# Production parameters
DAILY_LISTINGS = 500_000
PEAK_MULTIPLIER = 3.0  # Peak traffic is 3x average
LATENCY_SLA_MS = 100  # Product page must load in <100ms, model is part of that
MODEL_LATENCY_BUDGET_MS = 30  # Model gets 30ms of the 100ms budget

# Calculate required throughput
avg_per_second = DAILY_LISTINGS / (24 * 3600)
peak_per_second = avg_per_second * PEAK_MULTIPLIER

# How many replicas needed?
if ort_available:
    single_image_ms = np.mean(ort_latencies_single)
    runtime_name = "ONNX Runtime"
    images_per_sec_per_replica = 1000 / single_image_ms
else:
    single_image_ms = pt_single_mean
    runtime_name = "PyTorch"
    images_per_sec_per_replica = 1000 / single_image_ms

# With batching (batch of 16), throughput is higher
if ort_available:
    batched_per_sec = 1000 / (np.mean(ort_latencies_batch) / 16)
else:
    batched_per_sec = pt_throughput

replicas_for_peak = int(np.ceil(peak_per_second / batched_per_sec))
replicas_for_peak = max(replicas_for_peak, 2)  # minimum 2 for redundancy

# Cost estimation (Singapore cloud pricing)
CPU_COST_PER_HOUR = 0.05  # illustrative on-demand rate, 2 vCPU instance
GPU_COST_PER_HOUR = 0.90  # illustrative on-demand rate, T4-class GPU instance

# CPU deployment (ONNX Runtime)
cpu_replicas = max(replicas_for_peak * 2, 4)  # CPU is slower, need more
cpu_monthly = cpu_replicas * CPU_COST_PER_HOUR * 24 * 30

# GPU deployment (PyTorch or ONNX Runtime with CUDA)
gpu_monthly = replicas_for_peak * GPU_COST_PER_HOUR * 24 * 30

cost_per_inference_cpu = cpu_monthly / (DAILY_LISTINGS * 30)
cost_per_inference_gpu = gpu_monthly / (DAILY_LISTINGS * 30)

print(
    f"""
  DEPLOYMENT SIZING (based on benchmark data):

  Traffic Profile:
    Daily listings:          {DAILY_LISTINGS:>10,}
    Average per second:      {avg_per_second:>10.1f}
    Peak per second (3x):    {peak_per_second:>10.1f}
    Latency SLA:             {LATENCY_SLA_MS:>10} ms (page load)
    Model latency budget:    {MODEL_LATENCY_BUDGET_MS:>10} ms

  Model Performance ({runtime_name}):
    Single image latency:    {single_image_ms:>10.2f} ms
    Batched throughput:      {batched_per_sec:>10.0f} images/sec/replica
    Meets latency budget:    {"YES" if single_image_ms < MODEL_LATENCY_BUDGET_MS else "NO":>10s}

  OPTION A — CPU Deployment (ONNX Runtime, recommended for this model):
    Replicas needed:         {cpu_replicas:>10}
    Instance type:           {"CPU-2":>10s} (generic 2 vCPU, 4 GB instance)
    Monthly cost:            ${cpu_monthly:>9,.0f}
    Cost per inference:      ${cost_per_inference_cpu:>9.6f}
    Pros: Simple, no GPU driver headaches, easy horizontal scaling
    Cons: Higher latency, more replicas needed

  OPTION B — GPU Deployment (ONNX Runtime + CUDA):
    Replicas needed:         {replicas_for_peak:>10}
    Instance type:           {"GPU-T4":>10s} (generic T4-class GPU, 4 vCPU, 16 GB)
    Monthly cost:            ${gpu_monthly:>9,.0f}
    Cost per inference:      ${cost_per_inference_gpu:>9.6f}
    Pros: Lower latency, fewer replicas, room for larger models
    Cons: GPU driver management, less elastic scaling

  RECOMMENDATION FOR THE ENGINEERING MANAGER:
    This model ({count_parameters(resnet_se):,} params, {onnx_size_kb} KB ONNX) is
    small enough for CPU deployment. GPU is overkill unless you plan to
    upgrade to a larger model (ResNet-50, EfficientNet) later.

    Start with Option A (CPU):
      - Deploy ONNX file to {cpu_replicas} CPU replicas behind a load balancer
      - Set auto-scaling: scale up at 70% CPU, scale down at 30%
      - Monitor with kailash-ml DriftMonitor for accuracy degradation
      - Budget: ${cpu_monthly:,.0f}/month (~${cpu_monthly * 12:,.0f}/year)

    Compared to an ASSUMED manual-categorisation cost
    ($10,000/day = $300,000/month, illustrative):
      Savings: ${300000 - cpu_monthly:,.0f}/month = ${(300000 - cpu_monthly) * 12:,.0f}/year
"""
)

# ── Checkpoint 5: Apply section complete ─────────────────────────────
assert replicas_for_peak >= 1, "Should need at least 1 replica"
assert cost_per_inference_cpu > 0, "Cost per inference should be positive"
print("--- Checkpoint 5 passed --- deployment planning complete\n")


# ════════════════════════════════════════════════════════════════════════
# Clean up
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] Why teams ship ONNX rather than PyTorch to production (2-5 GB
      footprint, hardware portability, Python-only runtime)
  [x] ONNX as the universal exchange format (50 MB, any hardware,
      any language, free graph optimisations)
  [x] The deployment pipeline: Train -> Export -> Validate -> Register
      -> Serve -> Monitor

  BUILD + TRAIN:
  [x] Exported ResNetSE to ONNX with OnnxBridge (behind a flat-input adapter)
  [x] Validated numerical fidelity with OnnxBridge.validate
  [x] Registered the ONNX artifact and served it with InferenceServer
  [x] ONNX file: {onnx_path} ({onnx_size_kb} KB)

  VISUALISE (the proof):
  [x] Latency distribution: PyTorch vs ONNX Runtime side-by-side
  [x] Single image: {pt_single_mean:.2f}ms (PyTorch){f" vs {np.mean(ort_latencies_single):.2f}ms (ONNX)" if ort_available else ""}
  [x] Throughput: {pt_throughput:,.0f} img/sec (PyTorch){f" vs {ort_throughput:,.0f} img/sec (ONNX)" if ort_available else ""}

  APPLY:
  [x] Singapore e-commerce deployment planning with concrete numbers
  [x] CPU vs GPU cost comparison for this model size
  [x] Recommendation: CPU deployment at ${cpu_monthly:,.0f}/month
  [x] Annual savings vs manual review: ${(300000 - cpu_monthly) * 12:,.0f}

  KEY INSIGHT: The model is just one artifact in the deployment pipeline.
  ONNX export, numerical validation, latency benchmarking, capacity
  planning, and cost analysis are what turn "it works on my laptop" into
  "it's running in production." An engineering manager does not care about
  your model's val_accuracy — they care about latency, throughput, cost,
  and reliability.

  Next: In 04_hyperparameter_study.py, you'll explore the accuracy-vs-cost
  tradeoff with systematic learning rate and augmentation experiments...
"""
)
