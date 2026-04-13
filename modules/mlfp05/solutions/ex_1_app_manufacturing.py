# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1 Application: Manufacturing Defect Detection
# ════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are an ML engineer at a semiconductor fabrication plant
#   in Singapore (GlobalFoundries/SSMC). Visual inspection of
#   silicon wafers is the quality bottleneck — manual inspection
#   catches 82% of defects at 15 seconds per wafer. A missed
#   defect costs S$5,000 in downstream rework. Your plant manager
#   asks: "Can we automate inspection to catch more defects faster?"
#
# TECHNIQUE: Sparse Autoencoder
#   Train a Sparse AE on images of good wafers. The sparsity
#   constraint forces the encoder to learn efficient, interpretable
#   features (edge patterns, die structure). Defective wafers
#   activate different patterns and reconstruct poorly.
#
# WHAT YOU'LL SEE:
#   - Good wafer: original vs reconstruction (near-identical)
#   - Defective wafer: original vs reconstruction (mismatch at defect)
#   - Error heatmap highlighting the defect region
#   - Detection rate vs false alarm curve
#   - Cost-benefit analysis vs manual inspection
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

OUTPUT_DIR = Path("outputs/mlfp05/app_manufacturing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. Generate synthetic wafer images
# ════════════════════════════════════════════════════════════════════════
# Real wafer images would come from the fab's inspection cameras.
# We generate 64x64 grayscale images with a circular die pattern
# (grid of rectangular dies within a circular wafer boundary).

IMG_SIZE = 64
N_GOOD = 3000
N_DEFECTIVE = 400


def generate_wafer_base(rng: np.random.Generator) -> np.ndarray:
    """Generate base wafer image with circular boundary and die grid."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    center = IMG_SIZE / 2

    # Circular wafer boundary
    radius = IMG_SIZE * 0.42
    wafer_mask = ((xx - center) ** 2 + (yy - center) ** 2) < radius**2
    img[wafer_mask] = 0.3 + rng.uniform(-0.02, 0.02)

    # Die grid pattern (rectangular dies arranged in rows/columns)
    die_size = 6
    die_gap = 1
    for dy in range(5, IMG_SIZE - 5, die_size + die_gap):
        for dx in range(5, IMG_SIZE - 5, die_size + die_gap):
            die_center_x = dx + die_size / 2
            die_center_y = dy + die_size / 2
            if ((die_center_x - center) ** 2 + (die_center_y - center) ** 2) < (
                radius - 3
            ) ** 2:
                die_val = 0.6 + rng.uniform(-0.03, 0.03)
                img[dy : dy + die_size, dx : dx + die_size] = die_val

    # Subtle edge ring
    edge_mask = (
        ((xx - center) ** 2 + (yy - center) ** 2) > (radius - 2) ** 2
    ) & wafer_mask
    img[edge_mask] = 0.25

    # Light background noise
    img += rng.normal(0, 0.01, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    return np.clip(img, 0, 1)


def generate_good_wafer(rng: np.random.Generator) -> np.ndarray:
    """Generate a defect-free wafer image."""
    return generate_wafer_base(rng)


def generate_defective_wafer(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a defective wafer with one of several defect types."""
    img = generate_wafer_base(rng)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    defect_type = rng.choice(["scratch", "particle", "edge_chip", "hotspot"])
    center = IMG_SIZE / 2
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]

    if defect_type == "scratch":
        # Linear scratch across the wafer
        angle = rng.uniform(0, np.pi)
        length = rng.integers(20, 45)
        cx, cy = center + rng.uniform(-10, 10), center + rng.uniform(-10, 10)
        for t in np.linspace(-length / 2, length / 2, length * 3):
            px = int(cx + t * np.cos(angle))
            py = int(cy + t * np.sin(angle))
            if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                w = rng.integers(1, 3)
                img[
                    max(0, py - w) : min(IMG_SIZE, py + w),
                    max(0, px - w) : min(IMG_SIZE, px + w),
                ] = rng.uniform(0.7, 0.9)
                mask[
                    max(0, py - w) : min(IMG_SIZE, py + w),
                    max(0, px - w) : min(IMG_SIZE, px + w),
                ] = 1.0

    elif defect_type == "particle":
        # Random bright spots (contamination)
        n_particles = rng.integers(3, 10)
        for _ in range(n_particles):
            px = rng.integers(10, IMG_SIZE - 10)
            py = rng.integers(10, IMG_SIZE - 10)
            r = rng.integers(1, 4)
            dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
            particle_mask = dist < r
            img[particle_mask] = rng.uniform(0.75, 0.95)
            mask[particle_mask] = 1.0

    elif defect_type == "edge_chip":
        # Damage at the wafer edge
        angle = rng.uniform(0, 2 * np.pi)
        radius = IMG_SIZE * 0.42
        ex = center + radius * np.cos(angle)
        ey = center + radius * np.sin(angle)
        chip_r = rng.integers(4, 10)
        dist = np.sqrt((xx - ex) ** 2 + (yy - ey) ** 2)
        chip_mask = dist < chip_r
        img[chip_mask] = rng.uniform(0.05, 0.15)
        mask[chip_mask] = 1.0

    elif defect_type == "hotspot":
        # Thermal hotspot (dark region indicating process issue)
        hx = center + rng.uniform(-15, 15)
        hy = center + rng.uniform(-15, 15)
        hr = rng.integers(5, 12)
        dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
        blob = np.exp(-(dist**2) / (2 * (hr / 2) ** 2)) * rng.uniform(0.2, 0.4)
        img -= blob.astype(np.float32)
        mask[dist < hr] = 1.0

    return np.clip(img, 0, 1), mask, defect_type


rng = np.random.default_rng(42)

good_wafers = np.stack([generate_good_wafer(rng) for _ in range(N_GOOD)])
defect_data = [generate_defective_wafer(rng) for _ in range(N_DEFECTIVE)]
defective_wafers = np.stack([d[0] for d in defect_data])
defect_masks = np.stack([d[1] for d in defect_data])
defect_types = [d[2] for d in defect_data]

type_counts = {t: defect_types.count(t) for t in set(defect_types)}
print(f"Good wafers: {N_GOOD}")
print(f"Defective wafers: {N_DEFECTIVE}")
print(f"  Defect types: {type_counts}")

# ════════════════════════════════════════════════════════════════════════
# 2. Prepare data — train on good wafers only
# ════════════════════════════════════════════════════════════════════════
n_train = int(N_GOOD * 0.8)
train_good = good_wafers[:n_train]
test_good = good_wafers[n_train:]

train_tensor = torch.tensor(train_good[:, None, :, :], device=device)
test_good_tensor = torch.tensor(test_good[:, None, :, :], device=device)
test_defective_tensor = torch.tensor(defective_wafers[:, None, :, :], device=device)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

print(f"\nTraining on {len(train_good)} good wafers")
print(f"Test: {len(test_good)} good + {len(defective_wafers)} defective")


# ════════════════════════════════════════════════════════════════════════
# 3. Sparse Convolutional Autoencoder
# ════════════════════════════════════════════════════════════════════════
class SparseConvAE(nn.Module):
    """Conv AE with L1 sparsity penalty on the bottleneck.

    Sparsity forces the encoder to represent each wafer using
    only a few active features — making defect signatures stand out.
    """

    def __init__(self, sparsity_weight: float = 1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def loss(
        self, recon: torch.Tensor, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        mse = nn.functional.mse_loss(recon, x)
        sparsity = self.sparsity_weight * torch.mean(torch.abs(z))
        return mse + sparsity


model = SparseConvAE(sparsity_weight=1e-3).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

# ════════════════════════════════════════════════════════════════════════
# 4. Train on good wafers only
# ════════════════════════════════════════════════════════════════════════
EPOCHS = 50
losses = []

print("\nTraining sparse autoencoder on good wafers...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for (batch,) in train_loader:
        recon, z = model(batch)
        loss = model.loss(recon, batch, z)
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
# 5. Compute reconstruction errors
# ════════════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    recon_good, z_good = model(test_good_tensor)
    recon_defective, z_defective = model(test_defective_tensor)

    good_errors = (
        ((test_good_tensor - recon_good) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    )
    defective_errors = (
        ((test_defective_tensor - recon_defective) ** 2)
        .mean(dim=(1, 2, 3))
        .cpu()
        .numpy()
    )
    pixel_errors_defective = (
        ((test_defective_tensor - recon_defective) ** 2).squeeze(1).cpu().numpy()
    )

print(f"\nReconstruction errors:")
print(f"  Good wafers:    mean={good_errors.mean():.6f}, std={good_errors.std():.6f}")
print(
    f"  Defective:      mean={defective_errors.mean():.6f}, std={defective_errors.std():.6f}"
)
print(f"  Separation:     {defective_errors.mean() / good_errors.mean():.1f}x")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 1: Good vs defective — reconstruction comparison
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 6, figsize=(16, 11))

# Row 1-2: Good wafers (original + reconstruction)
for i in range(6):
    if i < 3:
        axes[0, i].imshow(test_good[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Good #{i+1}\n(Original)", fontsize=9)
    else:
        axes[0, i].imshow(
            recon_good[i - 3, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[0, i].set_title(f"Good #{i-2}\n(Reconstructed)", fontsize=9)
    axes[0, i].axis("off")

# Row 2: Good wafer error maps (should be nearly zero)
for i in range(6):
    if i < 3:
        err = np.abs(test_good[i] - recon_good[i, 0].cpu().numpy())
        axes[1, i].imshow(err, cmap="hot", vmin=0, vmax=0.15)
        axes[1, i].set_title(f"Error (MSE={good_errors[i]:.5f})", fontsize=8)
    else:
        axes[1, i].imshow(defective_wafers[i - 3], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Defective #{i-2}\n({defect_types[i-3]})", fontsize=9)
    axes[1, i].axis("off")

# Row 3: Defective reconstructions
for i in range(6):
    axes[2, i].imshow(recon_defective[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2, i].set_title(f"Reconstructed #{i+1}", fontsize=9)
    axes[2, i].axis("off")

# Row 4: Error heatmaps for defective wafers
for i in range(6):
    axes[3, i].imshow(pixel_errors_defective[i], cmap="hot", vmin=0)
    axes[3, i].set_title(f"Error Heatmap\n({defect_types[i]})", fontsize=8)
    axes[3, i].axis("off")

fig.suptitle(
    "Sparse Autoencoder: Wafer Defect Detection\n"
    "Good wafers reconstruct well; defects produce high-error regions",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "manufacturing_reconstructions.png", dpi=150, bbox_inches="tight"
)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'manufacturing_reconstructions.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 2: Detailed defect localisation (5 examples)
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for i in range(5):
    axes[0, i].imshow(defective_wafers[i], cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"Defective ({defect_types[i]})", fontsize=10)
    axes[0, i].axis("off")

    axes[1, i].imshow(recon_defective[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title("Reconstructed", fontsize=10)
    axes[1, i].axis("off")

    axes[2, i].imshow(pixel_errors_defective[i], cmap="hot")
    axes[2, i].set_title(f"Error (MSE={defective_errors[i]:.5f})", fontsize=9)
    axes[2, i].axis("off")

    axes[3, i].imshow(defect_masks[i], cmap="hot", vmin=0, vmax=1)
    axes[3, i].set_title("Ground Truth", fontsize=10)
    axes[3, i].axis("off")

axes[0, 0].set_ylabel("Input", fontsize=12, rotation=0, labelpad=55)
axes[1, 0].set_ylabel("Recon", fontsize=12, rotation=0, labelpad=55)
axes[2, 0].set_ylabel("Error", fontsize=12, rotation=0, labelpad=55)
axes[3, 0].set_ylabel("Truth", fontsize=12, rotation=0, labelpad=55)

fig.suptitle(
    "Defect Localisation: Error Heatmaps Pinpoint Defect Location\n"
    "Automated system shows operators WHERE to look",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "manufacturing_defect_localisation.png", dpi=150, bbox_inches="tight"
)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'manufacturing_defect_localisation.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 3: Detection rate vs false alarm curve
# ════════════════════════════════════════════════════════════════════════
all_errors = np.concatenate([good_errors, defective_errors])
all_labels = np.concatenate(
    [np.zeros(len(good_errors)), np.ones(len(defective_errors))]
)

thresholds = np.linspace(all_errors.min(), np.percentile(all_errors, 99.5), 200)
detection_rates = []
false_alarm_rates = []

for t in thresholds:
    predicted_defective = all_errors > t
    tp = np.sum(predicted_defective & (all_labels == 1))
    fp = np.sum(predicted_defective & (all_labels == 0))
    fn = np.sum(~predicted_defective & (all_labels == 1))
    tn = np.sum(~predicted_defective & (all_labels == 0))

    detection_rates.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    false_alarm_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

detection_rates = np.array(detection_rates)
false_alarm_rates = np.array(false_alarm_rates)

# Find operating point: 94% detection rate
target_dr = 0.94
best_idx = np.argmin(np.abs(detection_rates - target_dr))
operating_far = false_alarm_rates[best_idx]
operating_dr = detection_rates[best_idx]
operating_threshold = thresholds[best_idx]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(false_alarm_rates * 100, detection_rates * 100, color="#673AB7", linewidth=2)
ax.scatter(
    [operating_far * 100],
    [operating_dr * 100],
    color="#F44336",
    s=120,
    zorder=5,
    label=f"Operating point\nDR={operating_dr:.0%}, FAR={operating_far:.1%}",
)
ax.plot([0, 100], [82, 82], "k--", alpha=0.5, label="Manual inspection (82%)")
ax.set_xlabel("False Alarm Rate (%)", fontsize=12)
ax.set_ylabel("Detection Rate (%)", fontsize=12)
ax.set_title(
    "Defect Detection Rate vs False Alarm Rate\n"
    "Sparse AE significantly outperforms manual inspection",
    fontsize=13,
)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 50)
ax.set_ylim(50, 102)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "manufacturing_detection_curve.png", dpi=150, bbox_inches="tight"
)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'manufacturing_detection_curve.png'}")

# ════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT ANALYSIS
# ════════════════════════════════════════════════════════════════════════
WAFERS_PER_DAY = 5000
MANUAL_DETECTION_RATE = 0.82
MANUAL_SECONDS_PER_WAFER = 15
AE_SECONDS_PER_WAFER = 0.3
COST_PER_MISSED_DEFECT = 5000
DEFECT_RATE = 0.03  # 3% of wafers have defects
INSPECTOR_ANNUAL_COST = 60_000
INSPECTORS_NEEDED = 4  # for 5000 wafers/day at 15s each
WORKING_DAYS = 260

daily_defective = int(WAFERS_PER_DAY * DEFECT_RATE)

# Manual system
manual_caught = int(daily_defective * MANUAL_DETECTION_RATE)
manual_missed = daily_defective - manual_caught
manual_cost_missed = manual_missed * COST_PER_MISSED_DEFECT

# AE system
ae_caught = int(daily_defective * operating_dr)
ae_missed = daily_defective - ae_caught
ae_cost_missed = ae_missed * COST_PER_MISSED_DEFECT
ae_false_alarms_day = int((WAFERS_PER_DAY - daily_defective) * operating_far)

# Time savings
manual_hours_day = WAFERS_PER_DAY * MANUAL_SECONDS_PER_WAFER / 3600
ae_hours_day = WAFERS_PER_DAY * AE_SECONDS_PER_WAFER / 3600

annual_missed_cost_manual = manual_cost_missed * WORKING_DAYS
annual_missed_cost_ae = ae_cost_missed * WORKING_DAYS
annual_savings_defects = annual_missed_cost_manual - annual_missed_cost_ae
annual_savings_labour = (
    INSPECTORS_NEEDED * INSPECTOR_ANNUAL_COST - 1 * INSPECTOR_ANNUAL_COST
)  # Keep 1 for review

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — Semiconductor Wafer Inspection")
print("=" * 64)
print(f"\nPlant capacity: {WAFERS_PER_DAY:,} wafers/day, {DEFECT_RATE:.0%} defect rate")
print(f"Daily defective wafers: {daily_defective}")
print(f"\nManual inspection (current):")
print(f"  Detection rate:        {MANUAL_DETECTION_RATE:>10.0%}")
print(f"  Defects caught/day:    {manual_caught:>10}")
print(f"  Defects missed/day:    {manual_missed:>10}")
print(f"  Time per wafer:        {MANUAL_SECONDS_PER_WAFER:>10}s")
print(f"  Inspectors needed:     {INSPECTORS_NEEDED:>10}")
print(f"  Inspector hours/day:   {manual_hours_day:>10.1f}h")
print(f"\nSparse AE inspection (proposed):")
print(f"  Detection rate:        {operating_dr:>10.0%}")
print(f"  Defects caught/day:    {ae_caught:>10}")
print(f"  Defects missed/day:    {ae_missed:>10}")
print(f"  False alarms/day:      {ae_false_alarms_day:>10}")
print(f"  Time per wafer:        {AE_SECONDS_PER_WAFER:>10.1f}s")
print(f"  Processing hours/day:  {ae_hours_day:>10.1f}h")
print(f"\nAnnual financial impact:")
print(
    f"  Missed defect cost (manual):  {'S$' + f'{annual_missed_cost_manual:,.0f}':>14}"
)
print(f"  Missed defect cost (AE):      {'S$' + f'{annual_missed_cost_ae:,.0f}':>14}")
print(f"  Savings (fewer missed):       {'S$' + f'{annual_savings_defects:,.0f}':>14}")
print(f"  Labour savings (3 FTE):       {'S$' + f'{annual_savings_labour:,.0f}':>14}")
print(
    f"  Total annual savings:         {'S$' + f'{annual_savings_defects + annual_savings_labour:,.0f}':>14}"
)
print(f"\nKey insight: Automated inspection catches {operating_dr:.0%} of defects")
print(
    f"(vs {MANUAL_DETECTION_RATE:.0%} manual), reduces inspection time from {MANUAL_SECONDS_PER_WAFER}s to"
)
print(
    f"{AE_SECONDS_PER_WAFER}s per wafer, and saves {INSPECTORS_NEEDED - 1} FTE inspectors"
)
print(f"= S${annual_savings_labour:,}/year. The error heatmaps show the remaining")
print(f"inspector exactly WHERE to look on flagged wafers.")
print("=" * 64)
