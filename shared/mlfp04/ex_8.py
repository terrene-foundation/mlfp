# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 8 — Deep Learning Foundations.

Contains: synthetic XOR data, synthetic chest-film-style triage images
(labels caused by drawn shapes), reusable training loops, gradient
monitoring and AUC helpers, ModelVisualizer output paths. Technique-specific code (model classes, per-file training
loops, scenario narratives) does NOT belong here — it lives per file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kailash_ml import ModelVisualizer

from shared import MLFPDataLoader
from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT — seeds, device, output dir
# ════════════════════════════════════════════════════════════════════════
setup_environment()
torch.manual_seed(42)
np.random.seed(42)
device = get_device()

OUTPUT_DIR = Path("outputs") / "ex8_deep_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared hyperparameters
N_FEATS_XOR = 4
N_XOR_SAMPLES = 200
N_IMG_SAMPLES = 5000
IMG_SIZE = 64
N_CHANNELS = 1
N_CLASSES = 5
BATCH_SIZE = 64

# Kailash visualiser (used by every phase 4 block)
viz = ModelVisualizer()

# ════════════════════════════════════════════════════════════════════════
# DATA — XOR toy problem (Tasks 1-3)
# ════════════════════════════════════════════════════════════════════════


def make_xor_data(
    n_samples: int = N_XOR_SAMPLES, n_features: int = N_FEATS_XOR, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate a synthetic XOR classification task.

    Label is XOR of the sign of features 0 and 1. Features 2..n-1 are noise.
    Returns (X_tensor, y_tensor, y_numpy) on CPU.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float32)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)
    return X_t, y_t, y


# ════════════════════════════════════════════════════════════════════════
# DATA — Synthetic chest-film-style triage images (Tasks 4-10)
# ════════════════════════════════════════════════════════════════════════
# Scenario: chest-film triage at a Singapore public hospital. Real films
# are 512x512 DICOMs that cannot be shipped with a course, so this exercise
# draws SYNTHETIC 64x64 images in which each "finding" is a simple shape
# placed on a noisy background. The labels are caused by what is drawn, so
# a CNN can genuinely learn them — but these are NOT medical images, and
# nothing learned here says anything about real radiology.
#
#   pneumonia   -> a large, diffuse bright blob ("opacity")
#   effusion    -> a bright horizontal band across the base of the image
#   atelectasis -> a thin vertical bright streak
#   nodule      -> a small, sharp, very bright dot
#   normal      -> none of the four findings drawn
#
# Findings are independent (multi-label), each present in ~25% of images.

SG_HOSPITAL_CLASSES = [
    "pneumonia",
    "effusion",
    "atelectasis",
    "nodule",
    "normal",
]
FINDING_PREVALENCE = 0.25


def make_sg_imaging_data(
    n_samples: int = N_IMG_SAMPLES, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_images, y_labels) as float32 numpy arrays.

    X: (N, 1, 64, 64) — synthetic single-channel images: Gaussian background
       noise, a random per-image brightness offset, plus the drawn findings.
    y: (N, 5) — multi-label targets; columns follow SG_HOSPITAL_CLASSES and
       each finding column is 1 exactly when that shape was drawn.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    X = rng.normal(0.0, 0.5, size=(n_samples, N_CHANNELS, IMG_SIZE, IMG_SIZE))
    X += rng.normal(0.0, 0.3, size=(n_samples, 1, 1, 1))  # exposure jitter
    present = rng.random((n_samples, 4)) < FINDING_PREVALENCE

    for i in range(n_samples):
        img = X[i, 0]
        if present[i, 0]:  # pneumonia: diffuse blob
            cy, cx = rng.uniform(16, 48, size=2)
            radius = rng.uniform(6, 10)
            img += 1.2 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
        if present[i, 1]:  # effusion: bright band at the base
            top = int(rng.integers(46, 56))
            img[top:, :] += 1.0
        if present[i, 2]:  # atelectasis: thin vertical streak
            col = int(rng.integers(8, 56))
            row0 = int(rng.integers(4, 24))
            length = int(rng.integers(20, 36))
            img[row0 : row0 + length, col : col + 2] += 1.5
        if present[i, 3]:  # nodule: small, sharp, bright dot
            cy, cx = rng.uniform(8, 56, size=2)
            radius = rng.uniform(2.0, 3.5)
            img += 2.5 * (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2)

    y = np.zeros((n_samples, N_CLASSES), dtype=np.float32)
    y[:, :4] = present
    y[:, 4] = ~present.any(axis=1)
    return X.astype(np.float32), y


def build_sg_loaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Produce (train_loader, test_loader, X_test_np, y_test_np) for the CNN tasks."""
    X, y = make_sg_imaging_data()
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader, X_te, y_te


# ════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ════════════════════════════════════════════════════════════════════════


def train_xor_net(
    net: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimiser: torch.optim.Optimizer,
    n_epochs: int = 100,
    criterion: nn.Module | None = None,
) -> list[float]:
    """Fit a small binary classifier to XOR data. Returns per-epoch loss."""
    crit = criterion or nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(n_epochs):
        optimiser.zero_grad()
        loss = crit(net(X), y)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return losses


def xor_accuracy(net: nn.Module, X: torch.Tensor, y_np: np.ndarray) -> float:
    """Binary accuracy on XOR data (threshold at 0.5)."""
    net.eval()
    with torch.no_grad():
        probs = torch.sigmoid(net(X)).numpy().flatten()
    return float(((probs > 0.5) == y_np).mean())


def grad_norm(model: nn.Module) -> float:
    """L2 norm of the concatenated gradient vector."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return float(total**0.5)


def train_cnn_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    clip_value: float | None = None,
) -> tuple[float, float]:
    """Train for one epoch on the synthetic triage-image loader.

    Returns (mean_loss, mean_grad_norm). If ``clip_value`` is set, the grad
    norm is measured pre-clipping and ``clip_grad_norm_`` is applied.
    """
    model.train()
    losses: list[float] = []
    grads: list[float] = []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        grads.append(grad_norm(model))
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimiser.step()
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.mean(grads))


def eval_cnn(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Return mean validation loss across the loader."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            losses.append(criterion(model(X_b), y_b).item())
    return float(np.mean(losses))


def eval_cnn_auc(
    model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 256
) -> dict[str, float]:
    """Per-class ROC AUC on held-out images, plus their macro average.

    Loss values are hard to interpret on their own; AUC answers "does the
    model rank images WITH a finding above images WITHOUT it?" (0.5 =
    chance, 1.0 = perfect).
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    probs = np.concatenate(scores)
    aucs = {
        name: float(roc_auc_score(y[:, k], probs[:, k]))
        for k, name in enumerate(SG_HOSPITAL_CLASSES)
    }
    aucs["macro"] = float(np.mean(list(aucs.values())))
    return aucs


# ════════════════════════════════════════════════════════════════════════
# CNN BUILDING BLOCKS (reused across files 03, 04, 05)
# ════════════════════════════════════════════════════════════════════════


class ResBlock(nn.Module):
    """Residual block: skip connection preserves gradient flow."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class TriageCNN(nn.Module):
    """CNN for multi-label Singapore hospital triage.

    Architecture: Conv32 -> ResBlock -> Conv64 -> ResBlock -> AdaptiveAvgPool
    -> Dropout -> Linear. Designed for the multi-label BCEWithLogitsLoss
    setup used throughout Exercise 8.
    """

    def __init__(self, n_classes: int = N_CLASSES, dropout_rate: float = 0.3) -> None:
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
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ════════════════════════════════════════════════════════════════════════
# DATA LOADER ENTRY POINT
# ════════════════════════════════════════════════════════════════════════
# We expose an MLFPDataLoader handle so student files have a single import
# path even though the images are generated on the fly. Real image datasets
# for CNN training and fine-tuning live in Module 5.
loader = MLFPDataLoader()
