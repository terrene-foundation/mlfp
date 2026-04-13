# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1 Application: Medical Image Anomaly Detection
# ════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are an ML engineer at Singapore General Hospital (SGH)
#   building a screening tool for chest X-rays. Radiologists are
#   overwhelmed — 500 scans/day, each needing 5-10 minutes of
#   expert review. Your goal: automatically flag scans that look
#   "abnormal" so radiologists focus on the hardest cases.
#
# TECHNIQUE: Convolutional Autoencoder
#   Train a Conv AE on normal images only. The encoder learns
#   spatial features of healthy anatomy. Abnormal regions
#   reconstruct poorly, producing a pixel-level error heatmap
#   that highlights WHERE the anomaly is — not just IF one exists.
#
# WHAT YOU'LL SEE:
#   - Normal image vs reconstruction (close match)
#   - Anomalous image vs reconstruction (visible mismatch)
#   - Pixel-level error heatmaps showing anomaly location
#   - ROC curve for detection performance
#   - Workload reduction calculation for SGH
#
# ESTIMATED TIME: ~30-45 min (run and interpret)
# ════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")

OUTPUT_DIR = Path("outputs/mlfp05/app_medical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. Generate synthetic medical-like images (64x64 grayscale)
# ════════════════════════════════════════════════════════════════════════
# In production you would use real chest X-ray datasets (ChestX-ray14,
# MIMIC-CXR). Here we generate synthetic grayscale images with structure
# that mimics medical imaging: smooth background, organ-like elliptical
# regions, and anomalies as localised bright/dark spots.

IMG_SIZE = 64
N_NORMAL = 3000
N_ANOMALOUS = 300


def generate_normal_image(rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic 'normal' medical image with organ structures."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    # Background gradient (like chest cavity)
    y_grad = np.linspace(0.1, 0.3, IMG_SIZE).reshape(-1, 1)
    img += y_grad

    # Two elliptical "lung" regions
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    for cx, cy, rx, ry in [(22, 32, 12, 18), (42, 32, 12, 18)]:
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0
        img[mask] += rng.uniform(0.15, 0.25)

    # Central "mediastinum"
    med_mask = ((xx - 32) / 6) ** 2 + ((yy - 32) / 20) ** 2 < 1.0
    img[med_mask] += rng.uniform(0.2, 0.35)

    # Subtle texture noise
    img += rng.normal(0, 0.02, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    return np.clip(img, 0, 1)


def generate_anomalous_image(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate anomalous image with localised defect + ground truth mask."""
    img = generate_normal_image(rng)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Add 1-3 anomalous regions (nodules, opacities)
    n_anomalies = rng.integers(1, 4)
    for _ in range(n_anomalies):
        cx = rng.integers(15, IMG_SIZE - 15)
        cy = rng.integers(15, IMG_SIZE - 15)
        radius = rng.integers(3, 10)
        intensity = rng.uniform(0.2, 0.5)

        yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        blob = np.exp(-(dist**2) / (2 * (radius / 2) ** 2)) * intensity
        img += blob.astype(np.float32)
        mask[dist < radius] = 1.0

    return np.clip(img, 0, 1), mask


rng = np.random.default_rng(42)

normal_images = np.stack([generate_normal_image(rng) for _ in range(N_NORMAL)])
anomalous_data = [generate_anomalous_image(rng) for _ in range(N_ANOMALOUS)]
anomalous_images = np.stack([d[0] for d in anomalous_data])
anomaly_masks = np.stack([d[1] for d in anomalous_data])

print(
    f"Normal images: {normal_images.shape} (min={normal_images.min():.3f}, max={normal_images.max():.3f})"
)
print(f"Anomalous images: {anomalous_images.shape}")

# ════════════════════════════════════════════════════════════════════════
# 2. Prepare data — train on normal only
# ════════════════════════════════════════════════════════════════════════
n_train = int(N_NORMAL * 0.8)
train_normal = normal_images[:n_train]
test_normal = normal_images[n_train:]

# Add channel dim: (N, 1, 64, 64)
train_tensor = torch.tensor(train_normal[:, None, :, :], device=device)
test_normal_tensor = torch.tensor(test_normal[:, None, :, :], device=device)
test_anomalous_tensor = torch.tensor(anomalous_images[:, None, :, :], device=device)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

print(f"Training set: {len(train_normal)} normal images")
print(f"Test set: {len(test_normal)} normal + {len(anomalous_images)} anomalous")


# ════════════════════════════════════════════════════════════════════════
# 3. Convolutional Autoencoder for medical images
# ════════════════════════════════════════════════════════════════════════
class MedicalConvAE(nn.Module):
    """Conv AE for medical image anomaly detection.

    Encoder: 1->16->32->64 channels with stride-2 convolutions.
    Decoder: 64->32->16->1 channels with transposed convolutions.
    Bottleneck at 64 channels x 8x8 = 4096 dims (vs 64x64=4096 input).
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16->8
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, 3, stride=2, padding=1, output_padding=1
            ),  # 32->64
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


model = MedicalConvAE().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ════════════════════════════════════════════════════════════════════════
# 4. Train on normal images only
# ════════════════════════════════════════════════════════════════════════
EPOCHS = 40
losses = []

print("\nTraining medical image anomaly detector...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for (batch,) in train_loader:
        recon = model(batch)
        loss = criterion(recon, batch)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {avg_loss:.6f}")

# ════════════════════════════════════════════════════════════════════════
# 5. Compute per-image reconstruction errors
# ════════════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    recon_normal = model(test_normal_tensor)
    recon_anomalous = model(test_anomalous_tensor)

    # Per-image mean squared error
    normal_errors = (
        ((test_normal_tensor - recon_normal) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    )
    anomalous_errors = (
        ((test_anomalous_tensor - recon_anomalous) ** 2)
        .mean(dim=(1, 2, 3))
        .cpu()
        .numpy()
    )

    # Per-pixel error maps for anomalous images (for heatmaps)
    pixel_errors = (
        ((test_anomalous_tensor - recon_anomalous) ** 2).squeeze(1).cpu().numpy()
    )

print(f"\nReconstruction errors:")
print(f"  Normal:    mean={normal_errors.mean():.6f}, std={normal_errors.std():.6f}")
print(
    f"  Anomalous: mean={anomalous_errors.mean():.6f}, std={anomalous_errors.std():.6f}"
)

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 1: Normal vs reconstruction, Anomalous vs reconstruction
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 6, figsize=(16, 8))

# Row 1: Normal images — original vs reconstruction
for i in range(6):
    if i < 3:
        axes[0, i].imshow(test_normal[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Normal #{i+1}\n(Original)", fontsize=9)
    else:
        recon_img = recon_normal[i - 3, 0].cpu().numpy()
        axes[0, i].imshow(recon_img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Normal #{i-2}\n(Reconstructed)", fontsize=9)
    axes[0, i].axis("off")

# Row 2: Anomalous images — original vs reconstruction
for i in range(6):
    if i < 3:
        axes[1, i].imshow(anomalous_images[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Anomalous #{i+1}\n(Original)", fontsize=9)
    else:
        recon_img = recon_anomalous[i - 3, 0].cpu().numpy()
        axes[1, i].imshow(recon_img, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Anomalous #{i-2}\n(Reconstructed)", fontsize=9)
    axes[1, i].axis("off")

# Row 3: Error heatmaps for anomalous images
for i in range(6):
    if i < 3:
        axes[2, i].imshow(pixel_errors[i], cmap="hot", vmin=0)
        axes[2, i].set_title(f"Error Heatmap #{i+1}", fontsize=9)
    else:
        axes[2, i].imshow(anomaly_masks[i - 3], cmap="hot", vmin=0, vmax=1)
        axes[2, i].set_title(f"Ground Truth #{i-2}", fontsize=9)
    axes[2, i].axis("off")

fig.suptitle(
    "Convolutional Autoencoder: Medical Image Anomaly Detection\n"
    "Row 1: Normal (good reconstruction) | Row 2: Anomalous (poor reconstruction) | "
    "Row 3: Error heatmaps vs ground truth",
    fontsize=12,
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "medical_reconstructions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'medical_reconstructions.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 2: Detailed anomaly localisation heatmaps
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for i in range(5):
    # Original
    axes[0, i].imshow(anomalous_images[i], cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"Original #{i+1}", fontsize=10)
    axes[0, i].axis("off")

    # Reconstruction
    axes[1, i].imshow(recon_anomalous[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title(f"Reconstructed", fontsize=10)
    axes[1, i].axis("off")

    # Error heatmap
    axes[2, i].imshow(pixel_errors[i], cmap="hot")
    axes[2, i].set_title(f"Error Heatmap", fontsize=10)
    axes[2, i].axis("off")

    # Ground truth mask
    axes[3, i].imshow(anomaly_masks[i], cmap="hot", vmin=0, vmax=1)
    axes[3, i].set_title(f"Ground Truth", fontsize=10)
    axes[3, i].axis("off")

axes[0, 0].set_ylabel("Input", fontsize=12, rotation=0, labelpad=60)
axes[1, 0].set_ylabel("Recon", fontsize=12, rotation=0, labelpad=60)
axes[2, 0].set_ylabel("Error", fontsize=12, rotation=0, labelpad=60)
axes[3, 0].set_ylabel("Truth", fontsize=12, rotation=0, labelpad=60)

fig.suptitle(
    "Pixel-Level Anomaly Localisation\n"
    "Error heatmaps highlight WHERE the anomaly is, not just IF one exists",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "medical_anomaly_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'medical_anomaly_heatmaps.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 3: ROC curve for anomaly detection
# ════════════════════════════════════════════════════════════════════════
all_errors = np.concatenate([normal_errors, anomalous_errors])
all_labels = np.concatenate(
    [np.zeros(len(normal_errors)), np.ones(len(anomalous_errors))]
)

thresholds = np.linspace(all_errors.min(), all_errors.max(), 300)
tpr_list = []
fpr_list = []

for t in thresholds:
    predicted_positive = all_errors > t
    tp = np.sum(predicted_positive & (all_labels == 1))
    fp = np.sum(predicted_positive & (all_labels == 0))
    fn = np.sum(~predicted_positive & (all_labels == 1))
    tn = np.sum(~predicted_positive & (all_labels == 0))

    tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

tpr_arr = np.array(tpr_list)
fpr_arr = np.array(fpr_list)

# AUC via trapezoidal rule (sort by FPR)
sorted_idx = np.argsort(fpr_arr)
auc = np.trapz(tpr_arr[sorted_idx], fpr_arr[sorted_idx])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    fpr_arr, tpr_arr, color="#673AB7", linewidth=2, label=f"Conv AE (AUC = {auc:.3f})"
)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title(
    "ROC Curve: Medical Image Anomaly Detection\n"
    "Conv AE trained on normal images only",
    fontsize=13,
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "medical_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'medical_roc_curve.png'}")

# ════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT ANALYSIS
# ════════════════════════════════════════════════════════════════════════
SGH_DAILY_SCANS = 500
MINUTES_PER_MANUAL_REVIEW = 7.5
RADIOLOGIST_HOURLY_RATE_SGD = 250

# At 90% sensitivity (find threshold)
target_tpr = 0.90
best_idx = np.argmin(np.abs(tpr_arr - target_tpr))
operating_fpr = fpr_arr[best_idx]
operating_tpr = tpr_arr[best_idx]

# Scans that pass automated screening (predicted normal) don't need review
# Only flagged scans (predicted anomalous) get radiologist attention
assumed_anomaly_rate = 0.15  # ~15% of scans have findings
daily_anomalous = int(SGH_DAILY_SCANS * assumed_anomaly_rate)
daily_normal = SGH_DAILY_SCANS - daily_anomalous

# True positives: correctly flagged anomalous scans
flagged_true_anomalous = int(daily_anomalous * operating_tpr)
# False positives: normal scans incorrectly flagged
flagged_false_normal = int(daily_normal * operating_fpr)
# Total flagged for review
total_flagged = flagged_true_anomalous + flagged_false_normal
# Scans skipped (normal, correctly classified)
scans_saved = SGH_DAILY_SCANS - total_flagged

time_saved_minutes = scans_saved * MINUTES_PER_MANUAL_REVIEW
time_saved_hours = time_saved_minutes / 60
cost_saved_daily = time_saved_hours * RADIOLOGIST_HOURLY_RATE_SGD
cost_saved_annual = cost_saved_daily * 260  # working days

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — SGH Chest X-Ray Screening")
print("=" * 64)
print(f"\nSGH daily scan volume:           {SGH_DAILY_SCANS:>12}")
print(f"Manual review time per scan:     {MINUTES_PER_MANUAL_REVIEW:>10.1f} min")
print(f"Conv AE detection AUC:           {auc:>12.3f}")
print(
    f"\nAt {operating_tpr:.0%} sensitivity (catching {operating_tpr:.0%} of anomalies):"
)
print(f"  False positive rate:           {operating_fpr:>11.1%}")
print(f"  Scans flagged for review/day:  {total_flagged:>12}")
print(f"  Scans auto-cleared/day:        {scans_saved:>12}")
print(f"  Workload reduction:            {scans_saved/SGH_DAILY_SCANS:>11.0%}")
print(f"\nRadiologist time saved:")
print(f"  Minutes saved/day:             {time_saved_minutes:>12.0f}")
print(f"  Hours saved/day:               {time_saved_hours:>12.1f}")
print(f"  Cost saved/day:                {'S$' + f'{cost_saved_daily:,.0f}':>12}")
print(f"  Cost saved/year (260 days):    {'S$' + f'{cost_saved_annual:,.0f}':>12}")
print(f"\nKey insight: Automated screening at {operating_tpr:.0%} sensitivity reduces")
print(
    f"radiologist workload by {scans_saved/SGH_DAILY_SCANS:.0%}, saving {time_saved_hours:.1f} hours/day."
)
print(f"The error heatmaps show radiologists WHERE to look, further")
print(f"reducing time-to-diagnosis on flagged scans.")
print("=" * 64)
