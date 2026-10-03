# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.5: Contractive Autoencoder (Smooth Latent Space)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a contractive AE with a Jacobian penalty on the encoder
#   - Understand WHY smoothness in latent space matters
#   - Visualise latent interpolation proving smooth transitions
#   - Apply to medical image anomaly detection at a Singapore hospital
#   - Quantify workload reduction for radiologists
#
# PREREQUISITES: 04_sparse_ae.py
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build Contractive AE with explicit encoder weight access
#   2. Train with Frobenius norm penalty on encoder Jacobian
#   3. Visualise reconstruction + latent interpolation
#   4. Apply: chest X-ray anomaly screening at a Singapore hospital
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_1 import (
    INPUT_DIM,
    LATENT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    show_reconstruction,
    show_latent_interpolation,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Jacobian Penalty for Smooth Representations
# ════════════════════════════════════════════════════════════════════════
# The Jacobian penalty discourages the encoder from being too sensitive
# to small input perturbations. If two similar images map to very
# different latent codes, the latent space is "bumpy". The contractive
# penalty smooths it out.
#
# Analogy: A GPS system that jumps wildly when you move one metre is
# useless for navigation. You want small physical movements to produce
# small GPS changes. The Jacobian penalty is the "anti-jitter" filter
# for the encoder's latent coordinates.
#
# WHY THIS MATTERS: In manufacturing quality control, two photos of
# the same product taken at slightly different angles should map to
# similar latent codes. In medical imaging, a tumour that's 1mm
# larger should not cause the encoder to map to a completely different
# region of latent space.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Contractive AE
# ════════════════════════════════════════════════════════════════════════


class ContractiveAE(nn.Module):
    """Autoencoder whose encoder() is exposed so the penalty can differentiate it."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Define 3 explicit Linear layers for the encoder
        #       (not nn.Sequential — we need weight access for Jacobian)
        #       enc1: Linear(input_dim, 256)
        #       enc2: Linear(256, 64)
        #       enc3: Linear(64, latent_dim)
        self.enc1 = ____
        self.enc2 = ____
        self.enc3 = ____

        # TODO: Build decoder — nn.Sequential:
        #       Linear(latent_dim, 64), ReLU, Linear(64, 256), ReLU,
        #       Linear(256, input_dim), Sigmoid
        self.decoder = ____

    def encoder(self, x):
        # TODO: Forward through enc1->ReLU->enc2->ReLU->enc3
        # Use F.relu() for activations
        ____

    def forward(self, x):
        # TODO: Encode then decode. Return (reconstruction, latent_code)
        ____


CONTRACTIVE_WEIGHT = 1e-3


def jacobian_penalty(encode_fn, xb):
    """Mean squared Frobenius norm of the encoder Jacobian dz/dx (Rifai et al., 2011).

    torch.func.jacrev differentiates encode_fn for ONE sample — a
    (input_dim,) vector in, a (latent_dim,) code out — giving the
    (latent_dim, input_dim) Jacobian. vmap does this for every sample in
    the batch. The result is differentiable, so the penalty trains the
    encoder. Unlike a squared-weight sum (plain L2 weight decay), it depends
    on the input: it measures how much THIS image's code moves when its
    pixels move.
    """
    # TODO: Per-sample Jacobians with torch.func.jacrev, batched with
    #   torch.func.vmap -> shape (B, latent, input). Square, sum over the
    #   two matrix dims, then average over the batch.
    jac = ____
    return ____


def contractive_ae_loss(model, xb):
    """MSE + lambda * ||dz/dx||_F^2 — the contractive penalty."""
    # TODO: Reconstruction MSE plus CONTRACTIVE_WEIGHT times the Jacobian
    #   penalty of the ENCODER (pass the encoder function, not the model —
    #   the penalty is about dz/dx, not dx_hat/dx). Return (loss, {}).
    ____


print("\n" + "=" * 70)
print("  Contractive AE — Jacobian Penalty")
print("=" * 70)
print("  Smooth latent space: similar inputs -> similar latent codes.")

# TODO: Create ContractiveAE(INPUT_DIM, LATENT_DIM) and train
contractive_model = ____
contractive_losses = ____

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments (Jacobian-penalty stack)
# ══════════════════════════════════════════════════════════════════
# Contractive AEs penalise ‖∂z/∂x‖_F — the Jacobian Frobenius norm.
# This SHRINKS gradients near the bottleneck by design, so the Blood
# Test will look "low" — the question is whether it is CRITICALLY low
# (vanishing) or just REGULARISED low (intended).
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = contractive_ae_loss(m, xb)
    return loss


print("\n── Diagnostic Report (Contractive AE) ──")
diag, findings = run_diagnostic_checkpoint(
    contractive_model,
    flat_loader,
    _diag_loss,
    title=f"Contractive AE (lambda={CONTRACTIVE_WEIGHT})",
    n_batches=8,
    train_losses=contractive_losses,
    show=False,
)
print_prescription_pad(findings, f"Contractive AE (lambda={CONTRACTIVE_WEIGHT})")

# ══════ READING THE PRESCRIPTION PAD (key: see 01_standard_ae.py) ══════
# The Jacobian penalty limits how strongly the code reacts to small
# input changes, so lower encoder gradient RMS than in 01/02 is
# expected and is not automatically "vanishing" — the pad's own
# WARNING/CRITICAL thresholds decide that. If the reconstruction loss
# stalls well above 02's, CONTRACTIVE_WEIGHT is too strong.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise Reconstruction + Latent Interpolation
# ════════════════════════════════════════════════════════════════════════

# TODO: show_reconstruction and show_latent_interpolation
____
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(contractive_losses) == EPOCHS
assert contractive_losses[-1] < contractive_losses[0]
# INTERPRETATION: The latent interpolation plot is the key visual.
# As we morph from image A to image B through latent space, the
# contractive AE produces SMOOTH transitions — no abrupt jumps.
print("\n--- Checkpoint passed --- contractive AE trained\n")

if has_registry:
    register_model(
        registry, "contractive_ae", contractive_model, contractive_losses[-1]
    )


# ════════════════════════════════════════════════════════════════════════
# APPLY — Medical Image Anomaly Detection at a Singapore Hospital
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are an ML engineer at a large Singapore
# public hospital building a screening tool for chest X-rays.
# Radiologists are overwhelmed — 500 scans/day, each needing 5-10
# minutes of expert review. Your goal: automatically flag scans that
# look "abnormal" so radiologists focus on the hardest cases.
#
# WHY CONTRACTIVE AE: The smooth latent space means similar-looking
# anatomy maps to similar codes. Anomalous regions (tumours, opacities)
# produce codes far from the normal manifold, creating clear anomaly
# signals with pixel-level error heatmaps.

print("\n" + "=" * 70)
print("  APPLICATION: Medical Image Anomaly Detection (hospital)")
print("=" * 70)

# --- Generate synthetic medical images ---
MED_IMG_SIZE = 64
N_NORMAL = 3000
N_ANOMALOUS = 300
med_rng = np.random.default_rng(42)


# TODO: Implement generate_normal_image(rng_local)
# Create a 64x64 image with: vertical gradient background,
# two symmetric "lung" ellipses, a central "mediastinum" ellipse,
# small noise. Return clipped [0, 1].
def generate_normal_image(rng_local):
    ____


# TODO: Implement generate_anomalous_image(rng_local)
# Start from normal, add 1-3 bright circular blobs (simulated lesions).
# Return (image, anomaly_mask).
def generate_anomalous_image(rng_local):
    ____


# TODO: Generate datasets
normal_images = ____
anomalous_data = ____
anomalous_images = np.stack([d[0] for d in anomalous_data])
anomaly_masks = np.stack([d[1] for d in anomalous_data])

n_train_med = int(N_NORMAL * 0.8)
train_med_tensor = torch.tensor(normal_images[:n_train_med, None, :, :], device=device)
test_normal_tensor = torch.tensor(
    normal_images[n_train_med:, None, :, :], device=device
)
test_anomalous_tensor = torch.tensor(anomalous_images[:, None, :, :], device=device)
med_train_loader = DataLoader(
    TensorDataset(train_med_tensor), batch_size=64, shuffle=True
)

print(f"Normal: {N_NORMAL}, Anomalous: {N_ANOMALOUS}")


class MedicalConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Build encoder — 3 Conv2d layers (same pattern as SparseConvAE)
        self.encoder = ____

        # TODO: Build decoder — 3 ConvTranspose2d layers, ending with Sigmoid
        self.decoder = ____

    def forward(self, x):
        # TODO: Return decoder(encoder(x))
        ____


# TODO: Create model, optimizer, MSE criterion. Train 40 epochs.
med_model = ____
med_opt = ____
med_criterion = nn.MSELoss()

print("\nTraining medical image anomaly detector...")
for epoch in range(40):
    med_model.train()
    epoch_loss, n_batches = 0.0, 0
    for (batch,) in med_train_loader:
        # TODO: Forward, loss, backprop
        ____
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/40: loss = {epoch_loss/n_batches:.6f}")

# --- Evaluate ---
med_model.eval()
with torch.no_grad():
    recon_normal = med_model(test_normal_tensor)
    recon_anomalous = med_model(test_anomalous_tensor)
    normal_errors = (
        ((test_normal_tensor - recon_normal) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    )
    anomalous_errors = (
        ((test_anomalous_tensor - recon_anomalous) ** 2)
        .mean(dim=(1, 2, 3))
        .cpu()
        .numpy()
    )
    pixel_errors = (
        ((test_anomalous_tensor - recon_anomalous) ** 2).squeeze(1).cpu().numpy()
    )

# --- Visualisation: Anomaly heatmaps ---
# TODO: Create 4x5 grid: input, reconstructed, error heatmap, ground truth
# Save to OUTPUT_DIR / "ex1_medical_anomaly_heatmaps.png"
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_medical_anomaly_heatmaps.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- ROC curve ---
# TODO: Compute ROC curve (TPR vs FPR) and AUC
all_med_errors = np.concatenate([normal_errors, anomalous_errors])
all_med_labels = np.concatenate(
    [np.zeros(len(normal_errors)), np.ones(len(anomalous_errors))]
)
thresholds = np.linspace(all_med_errors.min(), all_med_errors.max(), 300)
tpr_list, fpr_list = [], []
for t in thresholds:
    pred = all_med_errors > t
    tp = np.sum(pred & (all_med_labels == 1))
    fp = np.sum(pred & (all_med_labels == 0))
    fn = np.sum(~pred & (all_med_labels == 1))
    tn = np.sum(~pred & (all_med_labels == 0))
    tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
tpr_arr, fpr_arr = np.array(tpr_list), np.array(fpr_list)
sorted_idx = np.argsort(fpr_arr)
auc = np.trapezoid(
    tpr_arr[sorted_idx], fpr_arr[sorted_idx]
)  # np.trapz removed in NumPy 2.0+

# TODO: Plot ROC curve. Save to OUTPUT_DIR / "ex1_medical_roc_curve.png"
fig, ax = plt.subplots(figsize=(8, 6))
____
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_medical_roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Business Impact ---
DAILY_SCANS = 500  # illustrative scenario figures
MINUTES_PER_REVIEW = 7.5
RADIOLOGIST_HOURLY_RATE = 250
target_tpr = 0.90
best_idx = np.argmin(np.abs(tpr_arr - target_tpr))
operating_fpr = fpr_arr[best_idx]
operating_tpr = tpr_arr[best_idx]

# TODO: Compute workload reduction metrics
anomaly_rate = 0.15
daily_anomalous = int(DAILY_SCANS * anomaly_rate)
daily_normal = DAILY_SCANS - daily_anomalous
flagged_true = int(daily_anomalous * operating_tpr)
flagged_false = int(daily_normal * operating_fpr)
total_flagged = flagged_true + flagged_false
scans_saved = DAILY_SCANS - total_flagged
time_saved_hours = scans_saved * MINUTES_PER_REVIEW / 60
cost_saved_annual = time_saved_hours * RADIOLOGIST_HOURLY_RATE * 260

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — Chest X-Ray Screening (illustrative)")
print("=" * 64)
print(f"\nDaily scan volume:               {DAILY_SCANS:>12}")
print(f"Conv AE detection AUC:           {auc:>12.3f}")
print(f"At {operating_tpr:.0%} sensitivity:")
print(f"  Scans auto-cleared/day:        {scans_saved:>12}")
print(f"  Workload reduction:            {scans_saved/DAILY_SCANS:>11.0%}")
print(f"  Hours saved/day:               {time_saved_hours:>12.1f}")
print(f"  Cost saved/year:               {'S$' + f'{cost_saved_annual:,.0f}':>12}")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a contractive AE with a per-sample Jacobian penalty ||dz/dx||_F^2
  [x] Visualised smooth latent interpolation — gradual morphing
  [x] Applied to medical image anomaly detection at a Singapore hospital
  [x] Generated pixel-level error heatmaps showing WHERE anomalies are
  [x] Computed ROC curve with AUC metric
  [x] Quantified radiologist workload reduction in hours and S$

  KEY INSIGHT: The Jacobian penalty ensures that small input changes
  produce small latent changes. This makes the latent space navigable:
  you can interpolate, cluster, and measure distances meaningfully.
  For medical imaging, this means the anomaly signal (high reconstruction
  error) is reliable — it comes from genuine structural differences,
  not from encoder sensitivity to irrelevant variations.

  Next: 06_convolutional_ae.py preserves spatial structure with Conv2d...
"""
)
