# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Lesson 5.0 (Prelude): The Destination — kailash-ml in 5 lines
# ════════════════════════════════════════════════════════════════════════
#
# WHY THIS LESSON COMES FIRST:
#   The next 8 lessons teach you how to build neural networks from the
#   ground up — autoencoders, CNNs, RNNs, transformers, GANs, GNNs,
#   transfer learning, RL. By the end of M5 you will know what every
#   line of a PyTorch training loop does and why.
#
#   Before that journey, see the destination: what a training call looks
#   like through the unified `kailash-ml` surface. The engine detects
#   your compute backend (Apple Silicon MPS, CUDA, ROCm, Intel XPU or
#   CPU) and its precision for you, with no flags.
#
#   Note on async: `km.train()` is async (kailash-ml 1.0+). In a CLI
#   script (this file) wrap it with `asyncio.run(...)`; in a notebook
#   with `nest_asyncio`, top-level `await km.train(...)` works.
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Read the compute backend and precision kailash-ml auto-selected
#     (`km.device()`)
#   - Train a classifier end-to-end with one call (`km.train`)
#   - Construct the `MLEngine` that `km.train` runs on and confirm it
#     uses the same backend
#
# WHAT YOU WON'T DO YET:
#   Build the model. Compute the loss. Write a training loop. Backprop.
#   That comes next, in Lesson 5.1. You will return to this prelude in
#   the M5 reflection at the end and recognise every line.
#
# PREREQUISITES: M4.8 (neural-network basics)
# ESTIMATED TIME: ~5 min (<30s of compute)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from sklearn.datasets import make_classification

import kailash_ml as km


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — `km.device()`: the backend the rest of the SDK picks
# ════════════════════════════════════════════════════════════════════════
#
# kailash-ml's BackendInfo answers four questions every M5 lesson asks:
#   * which compute backend?           (mps / cuda / rocm / xpu / cpu)
#   * which Lightning accelerator?     (mps / gpu / cpu)
#   * which precision?                 (16-mixed / bf16-mixed / 32)
#   * what capabilities?               (fp16, bf16, tensor cores, …)
#
# On an Apple Silicon Mac you will typically see `backend=mps`; on an
# NVIDIA GPU, `backend=cuda`; on a CPU-only laptop, `backend=cpu`.
#
# TODO: call km.device() and print the four BackendInfo fields below.
print("=" * 72)
print("TASK 1 — Compute backend (auto-detected)")
print("=" * 72)
backend = ____
print(f"  backend       : {backend.____}")
print(f"  accelerator   : {backend.____}")
print(f"  precision     : {backend.____}")
print(f"  capabilities  : {sorted(backend.____)}")
print(f"  device count  : {backend.device_count}")
print()

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert backend.backend in {
    "mps",
    "cuda",
    "cpu",
    "rocm",
    "xpu",
}, "backend should be one of mps/cuda/cpu/rocm/xpu"
print("✓ Checkpoint 1 passed — backend detected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — `km.train(df, target='y')`: training in one call
# ════════════════════════════════════════════════════════════════════════
#
# A binary classification problem with 10 features and 800 rows. One
# call runs the pipeline: data split, candidate model families, default
# hyperparameters, evaluation and metrics packaging. With
# family="auto" (the default) kailash-ml compares the model families
# available in your install and returns the winner's TrainingResult.
# Lessons 5.1+ build the deep architectures by hand.
#
# TODO: train on df with target column 'y' (remember km.train is async).
print("=" * 72)
print("TASK 2 — km.train() zero-config training")
print("=" * 72)

X, y = make_classification(
    n_samples=800, n_features=10, n_informative=6, random_state=42
)
df = pl.DataFrame({**{f"f{i}": X[:, i] for i in range(10)}, "y": y})
print(f"  dataset shape : {df.shape}  (polars-native, no pandas)")

# km.train is async — wrap with asyncio.run() in a CLI script.
result = ____

print(f"  result type   : {type(result).__name__}")
print(f"  metrics       : {result.metrics}")
print()

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert "accuracy" in result.metrics, "training should report an accuracy metric"
assert (
    result.metrics["accuracy"] > 0.8
), "make_classification with n_informative=6 should be easy (>0.8)"
print("✓ Checkpoint 2 passed — model trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — `MLEngine()`: the surface `km.train()` runs on
# ════════════════════════════════════════════════════════════════════════
#
# `km.train()` is a convenience wrapper around a default `MLEngine`. The
# engine is where feature stores, model registries, experiment tracking,
# hyperparameter search and serving plug in. The M5 lessons mostly use
# those pieces directly (ExperimentTracker, ModelRegistry, OnnxBridge,
# InferenceServer, diagnostics) rather than building their own engine,
# but every one of them resolves the same compute backend you see here.
#
# TODO: construct an MLEngine with no arguments (accelerator='auto').
print("=" * 72)
print("TASK 3 — MLEngine: the unified surface")
print("=" * 72)
engine = ____
print(f"  engine.accelerator  : {engine.accelerator}")
print(
    f"  engine.backend_info : {engine.backend_info.backend}/"
    f"{engine.backend_info.precision}"
)
print(f"  engine.store_url    : {engine.store_url}")
print(f"  engine.tenant_id    : {engine.tenant_id}")
print()

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert (
    engine.accelerator == backend.accelerator
), "MLEngine should pick the same backend as km.device()"
print("✓ Checkpoint 3 passed — engine wired to detected backend\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("Reflection")
print("=" * 72)
print(
    "What you've seen:\n"
    "  ✓ Compute backend selected automatically (no env vars, no flags)\n"
    "  ✓ A full training run in one call (km.train returns metrics)\n"
    "  ✓ The MLEngine surface km.train runs on\n\n"
    "Next: In Lesson 5.1 you will build an autoencoder by hand — every\n"
    "layer, every loss function, every gradient update. The destination\n"
    "will make sense once you have walked the journey to it.\n"
)
