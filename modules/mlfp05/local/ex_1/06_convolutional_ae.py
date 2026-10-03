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
#   - Apply to e-commerce image compression (Conv AE vs JPEG)
#   - Read a quality-at-equal-size comparison correctly before
#     promising bandwidth savings
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
        #       two 3x3 convolutions with stride 2 and padding 1, channels
        #       1 -> 16 -> 32, ReLU after each (spatial 28 -> 14 -> 7), then
        #       flatten the (32, 7, 7) maps and project them to latent_dim
        self.encoder = ____

        # TODO: Build decoder — nn.Sequential, the reverse path:
        #       project latent_dim back to 32*7*7 values (ReLU), reshape them
        #       to (32, 7, 7) feature maps, then two transposed 3x3
        #       convolutions that each double the spatial size (stride 2,
        #       padding 1, output_padding 1), channels 32 -> 16 -> 1, ReLU
        #       between and Sigmoid at the end (7 -> 14 -> 28)
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

# TODO: conv_model — a ConvAE with the module latent size; train it with
#       train_variant as run "conv_ae" on img_loader (image-shaped batches,
#       not flat_loader!)
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
from shared.mlfp05.diagnostics import print_prescription_pad


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
print_prescription_pad(findings, "Convolutional AE")

# ══════ READING THE PRESCRIPTION PAD (key: see 01_standard_ae.py) ══════
# First convolutional model: readings are per Conv2d/ConvTranspose2d
# parameter tensor, and the dead-neuron check covers each ReLU's
# channels. Compare the dead-unit shares with the dense AEs (01-05) —
# does weight sharing keep more units active?
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise
# ════════════════════════════════════════════════════════════════════════

# TODO: show_reconstruction on the IMAGE-shaped test set (X_test_img),
#       titled "Convolutional AE", telling the helper the model is
#       convolutional (see its is_conv flag)
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
# APPLY — E-Commerce Image Compression
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are an ML engineer at a Singapore e-commerce
# platform that serves ~50M product images a day with bandwidth costs
# of ~S$300K/month (illustrative scenario figures). Your VP asks: "Can ML-based
# compression reduce bandwidth costs while maintaining image quality?"

print("\n" + "=" * 70)
print("  APPLICATION: Image Compression vs JPEG")
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
        # TODO: Build encoder — two stride-2 3x3 convolutions (padding 1),
        #       1 -> 16 -> 32 channels (28 -> 14 -> 7), then a stride-1 3x3
        #       convolution (padding 1) down to bottleneck_channels; ReLU after
        #       each. The bottleneck is (bottleneck_channels, 7, 7).
        self.encoder = ____

        # TODO: Build decoder — transposed convolutions mirroring the encoder:
        #       a stride-1 3x3 layer bottleneck_channels -> 32 (padding 1), then
        #       two stride-2 3x3 layers (padding 1, output_padding 1),
        #       32 -> 16 -> 1, ReLU between, Sigmoid at the end
        self.decoder = ____

    def forward(self, x):
        # TODO: Return the reconstruction only (encode, then decode)
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
    # TODO: comp_model — a CompressionAE for this bottleneck size on the
    # device; comp_opt — Adam, lr 1e-3. The 30-epoch loop on img_loader and
    # the SSIM/PSNR evaluation below are given.
    comp_model = ____
    comp_opt = ____
    for epoch in range(30):
        comp_model.train()
        for (batch,) in img_loader:
            # TODO: Reconstruct the batch, MSE against the batch itself,
            #       optimiser step
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
ae_compare = ae_models[4]
ae_compare.eval()
target_ratio = ae_compare.compression_ratio
jpeg_idx = np.argmin([abs(r[0] - target_ratio) for r in jpeg_results])
jpeg_q = jpeg_results[jpeg_idx][4]

# TODO: 3-row grid of the first 8 test images: Original, JPEG at quality
#   jpeg_q (the JPEG setting closest to target_ratio), and ae_compare's
#   reconstruction. Save to OUTPUT_DIR / "ex1_compression_visual_comparison.png"
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

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — E-Commerce Image Compression")
print("=" * 64)
print(f"\nDaily product images served:     {DAILY_IMAGES:>14,}")
print(f"Monthly bandwidth cost:          {'S$' + f'{MONTHLY_BANDWIDTH_COST:,}':>12}")
print(f"\nAt ~{target_ratio:.0f}x compression:")
print(f"  JPEG SSIM:  {jpeg_matched_ssim:.4f}")
print(f"  AE SSIM:    {ae_4ch_ssim:.4f}  (+{ae_4ch_ssim - jpeg_matched_ssim:.4f})")
print(f"  AE: smoother blur artifacts; JPEG: blocky 8x8 grid artifacts")
print("\nReading this correctly: both codecs were compared at the SAME size,")
print("so this comparison saves no bandwidth by itself. If the AE's SSIM is")
print("higher, the saving would come from running it at a HIGHER ratio until")
print("its SSIM drops to JPEG's — measure that before quoting a number, and")
print("count the decoder's compute cost on every page view.")
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
  [x] Compared AE and JPEG quality at equal size, without inventing savings

  KEY INSIGHT: Conv2d filters share parameters across spatial positions,
  learning translation-invariant features. A button pattern detected
  at position (5,5) is also detected at (20,20). This parameter sharing
  makes conv AEs dramatically more efficient for image data than flat
  MLPs, which must learn separate weights for each spatial position.

  Next: 07_stacked_ae.py adds depth for hierarchical features...
"""
)
