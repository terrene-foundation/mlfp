# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.6: Convolutional Autoencoder (Spatial Hierarchy)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a Conv AE that preserves spatial locality with Conv2d/ConvTranspose2d
#   - Understand WHY conv layers beat flat MLPs for image data
#   - Observe sharper reconstructions than any flat variant
#   - Apply to e-commerce image compression at Shopee (Conv AE vs JPEG)
#   - Quantify bandwidth cost savings for 50M images/day
#
# PREREQUISITES: 05_contractive_ae.py
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build Conv AE: 1x28x28 -> 16x14x14 -> 32x7x7 -> latent -> reconstruct
#   2. Train on Fashion-MNIST (image format, not flattened)
#   3. Compare reconstruction sharpness to flat variants
#   4. Apply: image compression rate-distortion vs JPEG
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from shared.mlfp05.ex_1 import (
    LATENT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    show_reconstruction,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Spatial Hierarchy via Convolution
# ════════════════════════════════════════════════════════════════════════
# Conv layers preserve SPATIAL LOCALITY that flat MLPs destroy. A Conv2d
# filter detects patterns (edges, textures) at each spatial position.
# The encoder progressively downsamples: 28x28 -> 14x14 -> 7x7.
#
# Analogy: A flat MLP treats every pixel independently — like reading
# a newspaper by cutting out individual letters and sorting them
# alphabetically. A Conv layer reads the newspaper as-is, detecting
# words, sentences, and paragraphs in their spatial context.
#
# WHY THIS MATTERS: For any image data (product photos, satellite
# imagery, medical scans), spatial relationships carry meaning. A
# button next to a collar means "shirt"; the same button floating
# in space means nothing. Conv AEs preserve these relationships.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Convolutional AE
# ════════════════════════════════════════════════════════════════════════


class ConvAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        # TODO: Build encoder — nn.Sequential:
        #       Conv2d(1, 16, kernel_size=3, stride=2, padding=1), ReLU,
        #       Conv2d(16, 32, kernel_size=3, stride=2, padding=1), ReLU,
        #       Flatten(), Linear(32*7*7, latent_dim)
        self.encoder = ____

        # TODO: Build decoder — nn.Sequential:
        #       Linear(latent_dim, 32*7*7), ReLU,
        #       Unflatten(1, (32, 7, 7)),
        #       ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), ReLU,
        #       ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), Sigmoid
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode then decode. Return (reconstruction, latent_code)
        ____


def conv_ae_loss(model, xb):
    # TODO: Forward, MSE loss. Return (loss, {})
    ____


print("\n" + "=" * 70)
print("  Convolutional AE — Spatial Hierarchy")
print("=" * 70)
print("  Conv2d preserves spatial structure. Expect sharper reconstructions.")

# TODO: Create ConvAE(LATENT_DIM) and train on img_loader (not flat_loader!)
conv_model = ____
conv_losses = ____

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments (convolutional stack)
# ══════════════════════════════════════════════════════════════════
# First CONV model in the course. The Blood Test now reports grad
# RMS per Conv2d kernel; the X-ray monitors per-channel dead
# fractions. Healthy Conv nets typically have far FEWER dead
# channels than dense nets at equal depth thanks to weight sharing.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = conv_ae_loss(m, xb)
    return loss


print("\n── Diagnostic Report (Convolutional AE) ──")
diag, findings = run_diagnostic_checkpoint(
    conv_model,
    img_loader,
    _diag_loss,
    title="Convolutional AE",
    n_batches=8,
    train_losses=conv_losses,
    show=False,
)

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Gradient flow (HEALTHY): min RMS = 8.7e-04 at
#       'encoder.4.weight' (Conv2d). Convolutional weight sharing
#       keeps gradients uniform — spread < 10x across 6 Conv layers.
#   [!] Dead neurons  (WARNING): 'encoder.1' (relu): 14% dead
#       channels. Each dead channel is an unused FILTER — worse
#       than a dead Linear neuron because it wastes spatial capacity.
#   [✓] Loss trend    (HEALTHY): train slope -3.4e-03/epoch.
#       Final loss ~0.0048 — lower than dense AEs at matched
#       latent size because spatial priors make the task easier.
# ════════════════════════════════════════════════════════════════
# Final train loss: ~0.0048 after 10 epochs, bottleneck=64 channels.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [BLOOD TEST — CONV-SPECIFIC] Gradient spread <10x across all
#     Conv layers is the CNN health signature. Slide 5J shows
#     why: weight sharing means each filter receives gradient
#     from every spatial position, so vanishing is intrinsically
#     harder than in dense nets. Contrast 07_stacked where a
#     5-layer DENSE net routinely spans 1000x in RMS.
#     >> Prescription: If you see a >100x spread, check the
#        pooling/stride layout. A stride-2 Conv followed by a
#        stride-2 Conv halves spatial dims twice, starving deep
#        filters of gradient contributors. Replace with one
#        stride-2 + one stride-1.
#
#  [X-RAY — CONV-SPECIFIC] 14% dead channels means 14% of
#     filters are permanently off. Each dead filter is an
#     unused 3x3 kernel (9 parameters + activations) — wasted
#     both in FLOPs and in representation. Worse than dead
#     Linear neurons because a CNN's whole premise is that
#     each filter specialises in one feature.
#     >> Prescription: GELU or LeakyReLU for the encoder stack.
#        Or: reduce bottleneck_channels if capacity is excess
#        (fewer filters, fewer dead ones). You'll see this
#        fix applied in ex_2's ResNet-SE (variant 02).
#
#  [STETHOSCOPE] Final loss ~0.0048 is LOWER than 02 undercomplete
#     (~0.025) and LOWER than 07 stacked (~0.018). Why? The 2D
#     convolutional prior (translation invariance, local
#     connectivity) matches the spatial structure of Fashion-MNIST.
#     Lesson: architecture encodes assumptions — Conv says "pixels
#     near each other are correlated".
#     >> Prescription: No fix. This is the reward for matching
#        inductive bias to data geometry.
#
#  FIVE-INSTRUMENT TAKEAWAY: Conv-AEs show what "healthy deep
#  network" looks like when the architecture matches the data —
#  uniform gradients, low dead%, low loss. You will use this
#  reference when comparing to the pathological patterns in
#  ex_3 (RNN gradient collapse) and ex_6 (GNN over-smoothing).
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise
# ════════════════════════════════════════════════════════════════════════

# TODO: show_reconstruction with is_conv=True
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(conv_losses) == EPOCHS
assert conv_losses[-1] < conv_losses[0]
# INTERPRETATION: Compare to the undercomplete AE. The Conv version
# preserves EDGES and TEXTURES better — sharper outlines of shirts,
# shoes, bags. This is because Conv2d filters share parameters across
# spatial positions, learning translation-invariant features.
print("\n--- Checkpoint passed --- convolutional AE trained\n")

if has_registry:
    register_model(registry, "conv_ae", conv_model, conv_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — E-Commerce Image Compression (Shopee)
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are an ML engineer at a Singapore e-commerce
# platform (Shopee/Lazada). The platform serves 50M product images per
# day. Bandwidth costs are S$300K/month. Your VP asks: "Can ML-based
# compression reduce bandwidth costs while maintaining image quality?"

print("\n" + "=" * 70)
print("  APPLICATION: Image Compression vs JPEG (Shopee)")
print("=" * 70)

IMG_SIZE = 28
ORIGINAL_BYTES = IMG_SIZE * IMG_SIZE


def compute_ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq, sigma2_sq = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return float(
        ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
        / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    )


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")


# --- JPEG baseline ---
test_images_np = X_test_img[:200].cpu().numpy()[:, 0]
jpeg_qualities = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]
jpeg_results = []

print("\nJPEG compression baseline...")
for quality in jpeg_qualities:
    ssim_vals, psnr_vals, byte_sizes = [], [], []
    for img in test_images_np[:100]:
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        compressed_size = buf.tell()
        buf.seek(0)
        decompressed = np.array(Image.open(buf)).astype(np.float32) / 255.0
        ssim_vals.append(compute_ssim(img, decompressed))
        psnr_vals.append(compute_psnr(img, decompressed))
        byte_sizes.append(compressed_size)
    ratio = ORIGINAL_BYTES / np.mean(byte_sizes)
    jpeg_results.append(
        (ratio, np.mean(ssim_vals), np.mean(psnr_vals), np.mean(byte_sizes), quality)
    )
    print(f"  JPEG q={quality:2d}: ratio={ratio:.1f}x, SSIM={np.mean(ssim_vals):.4f}")


# --- Conv AE at multiple bottleneck sizes ---
class CompressionAE(nn.Module):
    def __init__(self, bottleneck_channels: int):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        # TODO: Build encoder — Conv2d(1,16,3,stride=2,padding=1), ReLU,
        #       Conv2d(16,32,3,stride=2,padding=1), ReLU,
        #       Conv2d(32, bottleneck_channels, 3, padding=1), ReLU
        self.encoder = ____

        # TODO: Build decoder — ConvTranspose2d mirroring encoder, end with Sigmoid
        self.decoder = ____

    def forward(self, x):
        # TODO: Return decoder(encoder(x))
        ____

    @property
    def compressed_bytes(self):
        return self.bottleneck_channels * 7 * 7

    @property
    def compression_ratio(self):
        return ORIGINAL_BYTES / self.compressed_bytes


bottleneck_configs = [1, 2, 4, 8, 16]
ae_results = []
ae_models = {}

print("\nTraining Conv AE at different bottleneck sizes...")
for bn_ch in bottleneck_configs:
    # TODO: Create CompressionAE(bn_ch), train 30 epochs on img_loader
    # Evaluate SSIM/PSNR on test set. Store results.
    comp_model = ____
    comp_opt = ____
    for epoch in range(30):
        comp_model.train()
        for (batch,) in img_loader:
            # TODO: Forward, MSE loss, backprop
            ____
    comp_model.eval()
    with torch.no_grad():
        test_recon = comp_model(X_test_img[:200]).cpu().numpy()[:, 0]
    ssim_vals = [compute_ssim(test_images_np[i], test_recon[i]) for i in range(100)]
    psnr_vals = [compute_psnr(test_images_np[i], test_recon[i]) for i in range(100)]
    ae_results.append(
        (
            comp_model.compression_ratio,
            np.mean(ssim_vals),
            np.mean(psnr_vals),
            comp_model.compressed_bytes,
            bn_ch,
        )
    )
    ae_models[bn_ch] = comp_model
    print(
        f"  AE bn={bn_ch:2d}ch: ratio={comp_model.compression_ratio:.1f}x, SSIM={np.mean(ssim_vals):.4f}"
    )

# --- Visualisation 1: Rate-distortion curve ---
# TODO: Plot SSIM and PSNR vs compression ratio for JPEG and AE
# Save to OUTPUT_DIR / "ex1_compression_rate_distortion.png"
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_compression_rate_distortion.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: Visual comparison grid ---
# TODO: 3-row grid: Original, JPEG at matched ratio, AE at 4ch
# Save to OUTPUT_DIR / "ex1_compression_visual_comparison.png"
fig, axes = plt.subplots(3, 8, figsize=(18, 7))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_compression_visual_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Business Impact ---
DAILY_IMAGES = 50_000_000
MONTHLY_BANDWIDTH_COST = 300_000
ae_4ch_ssim = [r[1] for r in ae_results if r[4] == 4][0]
jpeg_matched_ssim = jpeg_results[jpeg_idx][1]
savings_pct = 0.15
monthly_savings = MONTHLY_BANDWIDTH_COST * savings_pct
annual_savings = monthly_savings * 12

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — E-Commerce Image Compression")
print("=" * 64)
print(f"\nDaily product images served:     {DAILY_IMAGES:>14,}")
print(f"Monthly bandwidth cost:          {'S$' + f'{MONTHLY_BANDWIDTH_COST:,}':>12}")
print(f"\nAt ~{target_ratio:.0f}x compression:")
print(f"  JPEG SSIM:  {jpeg_matched_ssim:.4f}")
print(f"  AE SSIM:    {ae_4ch_ssim:.4f}  (+{ae_4ch_ssim - jpeg_matched_ssim:.4f})")
print(f"\nBandwidth savings/year:          {'S$' + f'{annual_savings:,.0f}':>12}")
print(f"  AE: smoother blur artifacts (preserves edges)")
print(f"  JPEG: blocky 8x8 grid artifacts")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a convolutional AE with stride-2 downsampling/upsampling
  [x] Observed sharper reconstructions than flat MLPs (spatial locality)
  [x] Applied to image compression: Conv AE vs JPEG rate-distortion
  [x] Compared artifact types: AE blur vs JPEG blockiness
  [x] Quantified bandwidth savings for 50M images/day platform

  KEY INSIGHT: Conv2d filters share parameters across spatial positions,
  learning translation-invariant features. A button pattern detected
  at position (5,5) is also detected at (20,20). This parameter sharing
  makes conv AEs dramatically more efficient for image data than flat
  MLPs, which must learn separate weights for each spatial position.

  Next: 07_stacked_ae.py adds depth for hierarchical features...
"""
)
