# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7, Part 5: Production Deployment (ONNX + Serving)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Export a fine-tuned transfer model to ONNX format with OnnxBridge
#   - Understand why ONNX matters for production (portability, speed)
#   - Serve predictions with kailash-ml InferenceServer from the registry
#   - Compare PyTorch and ONNX Runtime latency and throughput
#   - Deploy the best model for a medical imaging use case with
#     concrete latency benchmarks and serving cost analysis
#
# PREREQUISITES: Parts 1-4 (all transfer learning techniques).
# ESTIMATED TIME: ~20 min
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from kailash_ml import InferenceServer, OnnxBridge
from kailash_ml.diagnostics import run_diagnostic_checkpoint

from shared.mlfp05.diagnostics import print_prescription_pad
from shared.mlfp05.ex_7 import (
    CLASS_NAMES,
    EPOCHS,
    INPUT_SIZE,
    N_CLASSES,
    N_PIXELS,
    OUTPUT_DIR,
    FlatImageAdapter,
    attach_onnx_artifact,
    classifier_diag_loss,
    count_params,
    device,
    images_to_records,
    init_engines,
    load_cifar10,
    register_model,
    train_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — From Experiment to Production
# ════════════════════════════════════════════════════════════════════════
# Training a model is only half the job. The other half is deploying it
# so that real users can get predictions. This requires:
#
# 1. MODEL EXPORT — Convert from PyTorch (Python-specific) to a portable
#    format. ONNX (Open Neural Network Exchange) is the industry standard:
#      - Runs on any ONNX runtime (CPU, GPU, mobile, edge devices)
#      - Often faster on CPU than eager PyTorch (the runtime fuses and
#        optimises the graph) — but measure it: Task 4 does
#      - Language-agnostic: serve from C++, Java, C#, not just Python
#      - Hardware-agnostic: same model on NVIDIA, AMD, Intel, Apple
#
# 2. MODEL SERVING — Wrap the model in an inference server that handles:
#      - Batch prediction (process multiple inputs efficiently)
#      - Caching (don't recompute identical predictions)
#      - Monitoring (track latency, throughput, error rates)
#      - Version management (roll back to a previous model version)
#
# 3. LATENCY BUDGETS — Production systems have strict latency
#    requirements. A medical imaging system might need:
#      - < 500ms per image for interactive use (clinician waiting)
#      - < 100ms per image for batch processing (overnight screening)
#      - < 50ms per image for real-time video analysis
#
# kailash-ml's OnnxBridge exports the model, ModelRegistry versions it,
# and InferenceServer loads the registered ONNX artifact and serves it —
# the full pipeline from experiment to production.
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART 5: Production Deployment (ONNX + InferenceServer)")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data, engines, and train the production model
# ════════════════════════════════════════════════════════════════════════

train_set, val_set, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()

# One name for the whole lifecycle: register, attach ONNX, serve.
PROD_MODEL_NAME = "production_resnet18_transfer"


def build_transfer_resnet(
    n_classes: int = N_CLASSES,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a ResNet-18 with frozen backbone and fresh classifier head."""
    # No fallback to random weights: if the ImageNet download fails this
    # raises, because a random frozen backbone would make every "transfer"
    # number below meaningless.
    # TODO: Load the ImageNet ResNet-18, optionally freeze it, replace the head
    # Hint: same builder as Parts 2-4 — torchvision's ResNet18_Weights enum,
    #   requires_grad on every parameter, and a fresh nn.Linear for model.fc
    weights = ____  # TODO: the ImageNet-1K weights enum
    model = ____  # TODO: build resnet18 with those weights

    if freeze_backbone:
        for p in model.parameters():
            ____  # TODO: freeze this parameter

    in_features = model.fc.in_features
    model.fc = ____  # TODO: new classifier head for n_classes
    return model


# Train the production transfer model
print("\nTraining production transfer model...")
prod_model = build_transfer_resnet()
prod_losses, prod_accs, prod_train_accs = train_model(
    prod_model,
    "production_resnet18",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    epochs=EPOCHS,
)
best_prod_acc = max(prod_accs)

# Register in ModelRegistry (model.pkl = the trained weights)
prod_version = register_model(
    registry,
    PROD_MODEL_NAME,
    prod_model,
    best_prod_acc,
    prod_losses[-1],
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert best_prod_acc > 0.50, f"Production model acc {best_prod_acc:.3f} too low"
# INTERPRETATION: This is the model we're deploying. It's tracked in
# ExperimentTracker, registered in ModelRegistry, and now we'll export
# it to ONNX for portable, optimised inference.
print(f"\n  Production model val_acc: {best_prod_acc:.3f}")
print("--- Checkpoint 1 passed --- production model trained\n")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — pre-export gate
# ══════════════════════════════════════════════════════════════════
# Before shipping, run kailash-ml's instrumented diagnostic pass on the
# model we are about to export: a few real forward/backward passes over
# validation batches (no optimiser step) with gradient, activation and
# dead-neuron hooks attached, plus the real training-loss history.
# The checkpoint puts the model in train mode, which updates BatchNorm
# running statistics, so we diagnose a COPY and leave the model we are
# about to export untouched.
print("── Diagnostic Report (Production model — pre-export gate) ──")
# TODO: Run the diagnostic checkpoint on a COPY of prod_model over val_loader
# Hint: run_diagnostic_checkpoint(model, loader, loss_fn, title=..., ...);
#   copy.deepcopy keeps BatchNorm statistics of the real model untouched;
#   the shared classifier_diag_loss is the loss_fn
diag, findings = run_diagnostic_checkpoint(
    ____,  # TODO: a copy of the production model
    val_loader,
    ____,  # TODO: loss function
    title="Production model — pre-export gate",
    n_batches=8,
    train_losses=prod_losses,
    show=False,
)
print_prescription_pad(findings, "Production model — pre-export gate")
# HOW TO READ THE PRESCRIPTION PAD FOR THIS MODEL:
#  Gradient flow — only the new fc head is trainable; the frozen backbone
#     produces no parameter gradients by design, so judge this reading by
#     the fc layer. A CRITICAL "exploding" reading on fc means the head's
#     updates are large relative to its weights: lower the learning rate
#     before retraining. A "vanishing" reading on fc means the head
#     stopped learning.
#  Dead neurons — the ReLUs belong to the frozen ImageNet backbone. A
#     high dead fraction here means many pretrained features are silent
#     on CIFAR-10 images; that is a domain-gap signal (consider unfreezing
#     the last stage or an adapter, Part 4), not a training bug.
#  Loss trend — read from the real per-epoch training losses. A
#     non-converging trend is a reason to block the export and retrain.
#  If any reading is UNKNOWN, the library could not compute it from this
#  run; the message says why.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Export to ONNX via OnnxBridge
# ════════════════════════════════════════════════════════════════════════
# ONNX export traces the model's computation graph with a sample input,
# then serialises graph + weights to one portable file. OnnxBridge marks
# the batch dimension dynamic (any batch size at inference) and requests
# opset 17, a widely supported operator set; recent PyTorch exporters
# may upgrade it to 18 and print a version-conversion traceback — that
# is log noise, export_result.success is the real signal.
#
# InferenceServer's ONNX runtime turns each request record into ONE row
# of a 2-D float array, so we export the network behind FlatImageAdapter:
# it takes (batch, 3*96*96) rows of preprocessed pixels and reshapes them
# to (batch, 3, 96, 96).

print("-- Exporting to ONNX with OnnxBridge --")
prod_model.eval()
prod_model_cpu = prod_model.cpu()
# TODO: Wrap the CPU model so it accepts flat pixel rows, in eval mode
serving_adapter = ____

onnx_path = OUTPUT_DIR / "transfer_resnet18.onnx"
bridge = OnnxBridge()
# TODO: Export the adapter with OnnxBridge
# Hint: bridge.export(model, framework, output_path=..., sample_input=...) —
#   the framework string for PyTorch modules is "torch", and the sample
#   input is ONE flat row of N_PIXELS random values
export_result = ____
print(
    f"  OnnxBridge.export: success={export_result.success} "
    f"status={export_result.onnx_status}"
)
assert export_result.success, f"OnnxBridge export failed: {export_result.error_message}"

onnx_size_kb = onnx_path.stat().st_size // 1024
print(f"  Exported to {onnx_path} ({onnx_size_kb} KB)")

# Numerical check: OnnxBridge.validate runs the native model (through the
# adapter's predict()) and ONNX Runtime on the same rows.
val_x, val_y = next(iter(val_loader))
check_rows = val_x[:16].reshape(16, -1).numpy().astype(np.float32)
# TODO: Compare PyTorch and ONNX Runtime outputs on check_rows
# Hint: bridge.validate(...) with a tolerance of 1e-3
validation = ____
print(
    f"  OnnxBridge.validate: valid={validation.valid} "
    f"max_diff={validation.max_diff:.2e}"
)
assert validation.valid, f"ONNX output drifted from PyTorch: {validation.notes}"

# Attach the .onnx file to the version registered in Task 1, so the
# InferenceServer can load it from the registry by name + version.
# TODO: Attach onnx_path to the registered version (same name the server uses)
____
print(f"  Attached model.onnx to {PROD_MODEL_NAME} v{prod_version.version}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert onnx_path.exists(), "ONNX file should be exported"
assert onnx_path.stat().st_size > 1000, "ONNX file should not be empty"
# INTERPRETATION: The ONNX file is a self-contained model artifact.
# It contains the full computation graph and all weights. You can
# deploy this file to any server with an ONNX runtime — no Python
# or PyTorch required. This is how models go from "laptop experiment"
# to "production API serving millions of requests".
print("--- Checkpoint 2 passed --- ONNX export complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Serve predictions with InferenceServer
# ════════════════════════════════════════════════════════════════════════
# InferenceServer.from_registry resolves a registered model by name and
# version; start() loads its model.onnx from the registry's artifact
# store; predict() takes {"records": [...]} and returns the model output
# per record.

print("-- Serving with InferenceServer --")

N_SERVE = 8
sample_x = val_x[:N_SERVE]
sample_y = val_y[:N_SERVE]


async def serve_predictions(name: str, version: int, images: torch.Tensor) -> dict:
    """Load one registered ONNX model version and serve a batch request."""
    # TODO: Build the server from the registry, start it, predict, stop it
    # Hint: InferenceServer.from_registry is a coroutine (await it); pass
    #   registry=, version= and the "onnx" runtime. predict() takes a dict
    #   with a "records" list — images_to_records builds it.
    server = ____
    ____  # TODO: start the server
    print(f"  InferenceServer status: {server.status}  ({name} v{version}, onnx)")
    response = ____
    ____  # TODO: stop the server
    return response


response = asyncio.run(
    serve_predictions(PROD_MODEL_NAME, prod_version.version, sample_x)
)
server_logits = np.asarray(response["predictions"], dtype=np.float32)
server_preds = ____  # TODO: predicted class per served image
server_conf = F.softmax(torch.from_numpy(server_logits), dim=-1).max(dim=-1).values

# The same images through PyTorch directly, for comparison
with torch.no_grad():
    torch_preds = prod_model_cpu(sample_x).argmax(dim=-1).numpy()

print("\n  === InferenceServer Predictions (vs direct PyTorch) ===")
print(
    f"  {'#':<4} {'True':>12} {'Served':>12} {'PyTorch':>12} "
    f"{'Confidence':>12} {'Correct':>8}"
)
print("  " + "-" * 64)
for i in range(N_SERVE):
    true_cls = CLASS_NAMES[int(sample_y[i])]
    served_cls = CLASS_NAMES[int(server_preds[i])]
    torch_cls = CLASS_NAMES[int(torch_preds[i])]
    correct = "Y" if server_preds[i] == int(sample_y[i]) else "N"
    print(
        f"  {i + 1:<4} {true_cls:>12} {served_cls:>12} {torch_cls:>12} "
        f"{float(server_conf[i]):>12.3f} {correct:>8}"
    )
n_correct = int((server_preds == sample_y.numpy()).sum())
print(f"\n  Served-sample accuracy: {n_correct}/{N_SERVE}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert server_logits.shape == (N_SERVE, N_CLASSES), server_logits.shape
assert np.array_equal(
    server_preds, torch_preds
), "InferenceServer (ONNX) and PyTorch disagree on the sample classes"
# INTERPRETATION: The predictions above came back from InferenceServer
# running the registered ONNX artifact — not from PyTorch — and they
# match the PyTorch classes image for image. That agreement is the
# deployment contract: what you validated in the notebook is what the
# server returns. The server is also where production concerns live:
# request batching, monitoring (P50/P99 latency, throughput) and
# version pinning (we asked for an explicit version above).
print("\n--- Checkpoint 3 passed --- InferenceServer served the ONNX model\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Latency benchmarks: PyTorch vs ONNX Runtime
# ════════════════════════════════════════════════════════════════════════
# Production systems need to know: how fast is this model? We benchmark
# single-image and batch latency for eager PyTorch (on the training
# device) and for ONNX Runtime on CPU (the serving runtime above).

print("-- Latency Benchmarks --")

serving_model = prod_model.to(device).eval()
ort_session = ort.InferenceSession(str(onnx_path))
ort_input = ort_session.get_inputs()[0].name
n_warmup = 5
n_bench = 50


def time_pytorch(batch: torch.Tensor) -> list[float]:
    """Return per-call latencies (ms) of the PyTorch model on `device`."""
    batch = batch.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            serving_model(batch)
        latencies = []
        for _ in range(n_bench):
            # TODO: time one forward pass in milliseconds
            # Hint: time.perf_counter() before and after; on CUDA, call
            #   torch.cuda.synchronize() first so you time the GPU work
            ____
    return latencies


def time_onnx(batch: torch.Tensor) -> list[float]:
    """Return per-call latencies (ms) of the ONNX graph in ONNX Runtime (CPU)."""
    rows = batch.reshape(len(batch), -1).numpy().astype(np.float32)
    for _ in range(n_warmup):
        ort_session.run(None, {ort_input: rows})
    latencies = []
    for _ in range(n_bench):
        # TODO: time one ort_session.run call in milliseconds
        ____
    return latencies


single_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
batch_input = torch.randn(32, 3, INPUT_SIZE, INPUT_SIZE)

single_latencies = time_pytorch(single_input)
batch_latencies = time_pytorch(batch_input)
onnx_single_latencies = time_onnx(single_input)
onnx_batch_latencies = time_onnx(batch_input)

# TODO: P50/P99 latencies and P50 throughput
# Hint: np.percentile; throughput = images per batch / seconds per batch
single_p50 = ____
single_p99 = ____
batch_p50 = ____
batch_p99 = ____
throughput = ____  # images/sec at P50
onnx_single_p50 = np.percentile(onnx_single_latencies, 50)
onnx_batch_p50 = np.percentile(onnx_batch_latencies, 50)
onnx_speedup = single_p50 / onnx_single_p50

print(f"\n  === Latency Benchmarks (PyTorch on {device}, ONNX Runtime on CPU) ===")
print(f"  {'Metric':<34} {'Value':>15}")
print("  " + "-" * 51)
print(f"  {'PyTorch single image P50':<34} {single_p50:>12.1f} ms")
print(f"  {'PyTorch single image P99':<34} {single_p99:>12.1f} ms")
print(f"  {'PyTorch batch (32) P50':<34} {batch_p50:>12.1f} ms")
print(f"  {'PyTorch batch (32) P99':<34} {batch_p99:>12.1f} ms")
print(f"  {'PyTorch throughput (P50)':<34} {throughput:>12.0f} img/s")
print(f"  {'ONNX Runtime single image P50':<34} {onnx_single_p50:>12.1f} ms")
print(f"  {'ONNX Runtime batch (32) P50':<34} {onnx_batch_p50:>12.1f} ms")
print(f"  {'ONNX model size':<34} {onnx_size_kb:>12,} KB")

# Visualise latency distribution
fig_latency = go.Figure()
for name, values, colour in [
    ("PyTorch single image", single_latencies, "#2196F3"),
    ("PyTorch batch (32)", batch_latencies, "#4CAF50"),
    ("ONNX Runtime single image", onnx_single_latencies, "#FF9800"),
    ("ONNX Runtime batch (32)", onnx_batch_latencies, "#9C27B0"),
]:
    fig_latency.add_trace(
        go.Histogram(x=values, name=name, marker_color=colour, opacity=0.6, nbinsx=20)
    )
fig_latency.update_layout(
    title=f"Inference Latency Distribution (PyTorch on {device}, ONNX Runtime on CPU)",
    xaxis_title="Latency (ms)",
    yaxis_title="Count",
    template="plotly_white",
    barmode="overlay",
)
latency_path = OUTPUT_DIR / "05_latency_distribution.html"
fig_latency.write_html(str(latency_path))
print(f"\n  Saved: {latency_path}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert single_p50 > 0, "Should have measured single-image latency"
assert batch_p50 > 0, "Should have measured batch latency"
assert throughput > 0, "Should have positive throughput"
assert onnx_single_p50 > 0, "Should have measured ONNX Runtime latency"
# INTERPRETATION: These benchmarks tell you whether the model meets
# production latency requirements. Batch processing is cheaper per image
# because the hardware parallelises across the batch. Whether ONNX
# Runtime beats PyTorch depends on hardware: on a GPU, eager PyTorch can
# win; on CPU-only servers ONNX Runtime usually does.
print(
    f"\n  Single-image P50: PyTorch {single_p50:.1f} ms vs ONNX Runtime "
    f"{onnx_single_p50:.1f} ms (ONNX speed-up x{onnx_speedup:.2f}); "
    f"budget for interactive use: 500 ms "
    f"({'within' if single_p50 < 500 else 'OVER'} budget)"
)
print("\n--- Checkpoint 4 passed --- latency benchmarks complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Medical Imaging Production Deployment
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): Deploy the transfer model for the public
# dermatology clinic use case from Part 2. The volumes below are
# illustrative planning figures, not data from a real clinic:
#   - 200 patients/day, ~5 images per patient = 1,000 images/day
#   - Interactive mode: a dermatologist reviews predictions in real time
#   - Batch mode: overnight screening of new submissions

print("\n" + "=" * 70)
print("  APPLY: Medical Imaging Deployment — Dermatology Clinic (illustrative)")
print("=" * 70)

PATIENTS_PER_DAY = 200
IMAGES_PER_PATIENT = 5
DAILY_IMAGES = PATIENTS_PER_DAY * IMAGES_PER_PATIENT

# Interactive mode: dermatologist reviews in real-time
# TODO: daily processing time in minutes for both modes
# Hint: interactive = one image at a time at the single-image P50;
#   batch = DAILY_IMAGES split into batches of 32 at the batch P50
interactive_time_per_image = ____  # ms
interactive_total_ms = ____
interactive_total_min = ____

# Batch mode: overnight screening
batch_n_batches = ____
batch_total_ms = ____
batch_total_min = ____

# Cost analysis (illustrative cloud GPU price — check your provider)
GPU_HOURLY_COST = 1.20  # S$/hr, illustrative
interactive_hours = interactive_total_min / 60
batch_hours = batch_total_min / 60

print(f"\n  === Deployment Profile ===")
print(
    f"  Daily volume: {PATIENTS_PER_DAY} patients x {IMAGES_PER_PATIENT} images = {DAILY_IMAGES:,} images/day"
)
print(f"  Model: ResNet-18 transfer ({best_prod_acc:.1%} accuracy)")
print(f"  ONNX size: {onnx_size_kb} KB")
print()
print(f"  {'Mode':<20} {'Per Image':>12} {'Daily Total':>14} {'GPU Cost':>12}")
print("  " + "-" * 60)
print(
    f"  {'Interactive':<20} "
    f"{interactive_time_per_image:>10.1f}ms "
    f"{interactive_total_min:>12.1f}min "
    f"S${interactive_hours * GPU_HOURLY_COST:>9.2f}"
)
print(
    f"  {'Batch (32)':<20} "
    f"{batch_p50 / 32:>10.1f}ms "
    f"{batch_total_min:>12.1f}min "
    f"S${batch_hours * GPU_HOURLY_COST:>9.2f}"
)
print()
print(f"  DEPLOYMENT RECOMMENDATION:")
if onnx_speedup > 1.0:
    print(
        f"  1. Serve the ONNX artifact: ONNX Runtime was x{onnx_speedup:.2f} faster "
        f"than PyTorch per image on this machine"
    )
else:
    print(
        f"  1. ONNX Runtime was not faster here (x{onnx_speedup:.2f}); serve ONNX for "
        f"portability, and benchmark again on the target server"
    )
print(f"  2. Use batch mode for overnight screening ({batch_total_min:.1f} min/day)")
print(
    f"  3. Interactive mode for real-time consultation ({interactive_time_per_image:.0f}ms/image)"
)
print(
    f"  4. Monthly GPU cost (illustrative, 1 h/day minimum): "
    f"~S${30 * max(interactive_hours, 1) * GPU_HOURLY_COST:.0f}"
)
print(f"  5. Register model in ModelRegistry for version tracking and rollback")

# Model comparison summary across all parts
print(f"\n  === Exercise 7 Complete Model Comparison ===")
print(f"  {'Approach':<30} {'Val Accuracy':>14} {'Trainable':>14} {'Use When':>25}")
print("  " + "-" * 85)

n_frozen_trainable = count_params(prod_model, trainable_only=True)

print(
    f"  {'From scratch (Part 1)':<30} "
    f"{'see Part 1':>14} "
    f"{'all params':>14} "
    f"{'Abundant data, unique domain':>25}"
)
print(
    f"  {'Frozen head (Part 2/5)':<30} "
    f"{best_prod_acc:>14.1%} "
    f"{n_frozen_trainable:>14,} "
    f"{'Quick start, limited compute':>25}"
)
print(
    f"  {'Adapter (Part 4)':<30} "
    f"{'see Part 4':>14} "
    f"{'see Part 4':>14} "
    f"{'Multi-tenant, balanced':>25}"
)
print(
    f"  {'LoRA (Module 6)':<30} "
    f"{'coming in M6':>14} "
    f"{'~1-5%':>14} "
    f"{'LLMs, billions of params':>25}"
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert onnx_path.exists(), "ONNX model should be deployed"
assert best_prod_acc > 0.50, "Production model should have reasonable accuracy"
# INTERPRETATION: The full pipeline — train, track, register, export,
# serve, benchmark — is what production ML looks like. Every step is
# logged and versioned: you can trace from a prediction back to the
# exact model version, training run, and dataset that produced it.
# This audit trail is essential for regulated industries like healthcare.
print("\n--- Checkpoint 5 passed --- deployment analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXERCISE 7 COMPLETE — What You've Mastered")
print("=" * 70)
print(
    f"""
  PART 1 — From-Scratch Baseline:
    [x] Trained CNN from random init, quantified the data bottleneck
    [x] Visualised noisy learned filters and overlapping t-SNE clusters

  PART 2 — Transfer Learning:
    [x] Loaded pre-trained ResNet-18, froze backbone, trained classifier head
    [x] Visualised structured activations, Grad-CAM attention maps
    [x] Applied to medical imaging (a public dermatology clinic, illustrative)

  PART 3 — Data Efficiency:
    [x] Measured accuracy at 10/25/50/100% of training data
    [x] Plotted efficiency curves, identified the labelling sweet spot
    [x] Answered "how many images do we need?" for a ride-hailing platform

  PART 4 — Adapter Modules:
    [x] Built bottleneck adapters with zero-init skip connections
    [x] Compared parameter efficiency: scratch vs frozen vs adapter
    [x] Analysed multi-tenant serving savings (50 clients)

  PART 5 — Production Deployment:
    [x] Exported to ONNX with OnnxBridge ({onnx_size_kb} KB portable model)
    [x] Served predictions with InferenceServer from the ModelRegistry
    [x] Benchmarked: {single_p50:.1f}ms single, {throughput:.0f} img/s throughput
    [x] Designed a medical imaging deployment for a dermatology clinic

  ARCHITECTURE-SELECTION GUIDE (consolidated across M5):
    Images    -> CNN / ViT + transfer learning (ImageNet pre-trained)
    Text      -> Transformer + transfer learning (BERT / GPT pre-trained)
    Sequences -> LSTM / Transformer (sometimes transfer)
    Tabular   -> Gradient boosting (train from scratch, fast and reliable)

  TRANSFER LEARNING SPECTRUM:
    Frozen head -> Adapter/LoRA -> Partial fine-tune -> Full fine-tune
    (fewest params)                                   (most params)
    (fastest training)                                (highest capacity)
    (safest from forgetting)                          (risk of forgetting)

  NEXT: Exercise 8 covers Reinforcement Learning (DQN + PPO).
  Then Module 6 uses LoRA and adapters for LLM fine-tuning — the same
  concept you explored here, applied to language models with billions
  of parameters.
"""
)
