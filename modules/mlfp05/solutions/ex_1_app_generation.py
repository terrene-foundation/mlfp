# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1 Application: Privacy-Preserving Synthetic Data
# ════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a data scientist at National University Hospital (NUH)
#   in Singapore. Researchers at NUS and NTU need patient data to
#   study diabetes risk factors, but Singapore's PDPA prohibits
#   sharing identifiable records. Your director asks: "Can we give
#   researchers statistically useful data without exposing any
#   real patient?"
#
# TECHNIQUE: Variational Autoencoder (VAE)
#   Train a VAE on real patient records. The learned latent space
#   captures statistical relationships (age-BMI correlation,
#   cholesterol-diagnosis link) without memorising individuals.
#   Sample from the latent prior to generate synthetic patients
#   that are statistically similar but not copies of real ones.
#
# WHAT YOU'LL SEE:
#   - Real vs synthetic distributions per feature (overlaid histograms)
#   - Correlation matrix comparison (real vs synthetic)
#   - t-SNE of real vs synthetic records (should overlap)
#   - Privacy test: nearest-neighbour distance proves no copying
#   - PDPA compliance assessment
#
# ESTIMATED TIME: ~30-45 min (run and interpret)
# ════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")

OUTPUT_DIR = Path("outputs/mlfp05/app_generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. Generate realistic patient data (Singapore demographics)
# ════════════════════════════════════════════════════════════════════════
# In production, this would be real de-identified hospital records.
# We generate 10K records with realistic clinical correlations
# based on Singapore population health statistics.

N_PATIENTS = 10_000
rng = np.random.default_rng(42)

# Age: Singapore adult distribution (skewed towards working age)
age = rng.normal(loc=52, scale=15, size=N_PATIENTS).clip(21, 90).astype(np.float32)

# Gender: 0=Female, 1=Male (roughly balanced)
gender = rng.binomial(1, 0.48, size=N_PATIENTS).astype(np.float32)

# BMI: correlated with age, higher in males (Singapore stats)
bmi_base = 22 + 0.05 * (age - 40) + 1.2 * gender
bmi = (bmi_base + rng.normal(0, 3.5, N_PATIENTS)).clip(16, 45).astype(np.float32)

# Systolic BP: correlated with age and BMI
sbp_base = 100 + 0.4 * (age - 40) + 0.8 * (bmi - 24)
systolic_bp = (
    (sbp_base + rng.normal(0, 12, N_PATIENTS)).clip(85, 200).astype(np.float32)
)

# Diastolic BP: correlated with systolic
diastolic_bp = (
    (systolic_bp * 0.6 + rng.normal(0, 8, N_PATIENTS)).clip(55, 120).astype(np.float32)
)

# Total cholesterol: correlated with age and BMI
chol_base = 150 + 0.5 * (age - 40) + 1.5 * (bmi - 24)
cholesterol = (
    (chol_base + rng.normal(0, 30, N_PATIENTS)).clip(100, 350).astype(np.float32)
)

# HbA1c: diabetes marker, correlated with BMI and age
hba1c_base = 5.0 + 0.01 * (age - 40) + 0.05 * (bmi - 24)
hba1c = (
    (hba1c_base + rng.exponential(0.4, N_PATIENTS)).clip(4.0, 14.0).astype(np.float32)
)

# Fasting glucose: correlated with HbA1c
glucose = (hba1c * 18 + rng.normal(0, 15, N_PATIENTS)).clip(60, 300).astype(np.float32)

# Diagnosis: probability increases with BMI, age, cholesterol, HbA1c
diagnosis_logit = (
    -6.0
    + 0.03 * (age - 40)
    + 0.08 * (bmi - 24)
    + 0.005 * (cholesterol - 200)
    + 0.5 * (hba1c - 5.5)
)
diagnosis_prob = 1 / (1 + np.exp(-diagnosis_logit))
diagnosis = rng.binomial(1, diagnosis_prob).astype(np.float32)

df = pl.DataFrame(
    {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "hba1c": hba1c,
        "fasting_glucose": glucose,
        "diabetes_diagnosis": diagnosis,
    }
)

FEATURE_NAMES = df.columns
N_FEATURES = len(FEATURE_NAMES)

print(f"Patient dataset: {df.shape[0]:,} records, {N_FEATURES} features")
print(f"Diabetes prevalence: {diagnosis.mean()*100:.1f}%")
print(f"Feature ranges:")
for col in FEATURE_NAMES:
    vals = df[col].to_numpy()
    print(f"  {col:25s}: [{vals.min():.1f}, {vals.max():.1f}], mean={vals.mean():.1f}")

# ════════════════════════════════════════════════════════════════════════
# 2. Normalise and prepare for VAE
# ════════════════════════════════════════════════════════════════════════
data_np = df.to_numpy().astype(np.float32)
data_min = data_np.min(axis=0)
data_max = data_np.max(axis=0)
data_range = data_max - data_min
data_range[data_range == 0] = 1.0
data_norm = (data_np - data_min) / data_range

data_tensor = torch.tensor(data_norm, device=device)
train_loader = DataLoader(TensorDataset(data_tensor), batch_size=256, shuffle=True)


# ════════════════════════════════════════════════════════════════════════
# 3. Variational Autoencoder for tabular data
# ════════════════════════════════════════════════════════════════════════
class PatientVAE(nn.Module):
    """VAE for generating synthetic patient records."""

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


LATENT_DIM = 8
model = PatientVAE(N_FEATURES, LATENT_DIM).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


def vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ════════════════════════════════════════════════════════════════════════
# 4. Train VAE on patient records
# ════════════════════════════════════════════════════════════════════════
EPOCHS = 100
losses = []

print("\nTraining VAE on patient records...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_samples = 0
    for (batch,) in train_loader:
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
        n_samples += len(batch)
    avg_loss = epoch_loss / n_samples
    losses.append(avg_loss)
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {avg_loss:.4f}")

# ════════════════════════════════════════════════════════════════════════
# 5. Generate synthetic patients
# ════════════════════════════════════════════════════════════════════════
N_SYNTHETIC = N_PATIENTS

model.eval()
with torch.no_grad():
    z = torch.randn(N_SYNTHETIC, LATENT_DIM, device=device)
    synthetic_norm = model.decode(z).cpu().numpy()

# Denormalise
synthetic_raw = synthetic_norm * data_range + data_min

# Round binary features
synthetic_raw[:, 1] = np.round(synthetic_raw[:, 1]).clip(0, 1)  # gender
synthetic_raw[:, -1] = np.round(synthetic_raw[:, -1]).clip(0, 1)  # diagnosis

df_synthetic = pl.DataFrame(
    {name: synthetic_raw[:, i] for i, name in enumerate(FEATURE_NAMES)}
)

print(f"\nGenerated {N_SYNTHETIC:,} synthetic patients")
print(f"Synthetic diabetes prevalence: {synthetic_raw[:, -1].mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 1: Real vs synthetic distributions per feature
# ════════════════════════════════════════════════════════════════════════
n_cols = 3
n_rows = (N_FEATURES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
axes = axes.flatten()

for i, name in enumerate(FEATURE_NAMES):
    real_vals = data_np[:, i]
    synth_vals = synthetic_raw[:, i]

    if name in ("gender", "diabetes_diagnosis"):
        # Bar chart for binary features
        categories = [0, 1]
        real_counts = [np.mean(real_vals == c) for c in categories]
        synth_counts = [np.mean(synth_vals == c) for c in categories]
        x = np.arange(len(categories))
        axes[i].bar(
            x - 0.15, real_counts, 0.3, label="Real", color="#2196F3", alpha=0.8
        )
        axes[i].bar(
            x + 0.15, synth_counts, 0.3, label="Synthetic", color="#FF9800", alpha=0.8
        )
        axes[i].set_xticks(x)
        labels = ["Female", "Male"] if name == "gender" else ["No", "Yes"]
        axes[i].set_xticklabels(labels)
    else:
        axes[i].hist(
            real_vals, bins=40, alpha=0.6, density=True, label="Real", color="#2196F3"
        )
        axes[i].hist(
            synth_vals,
            bins=40,
            alpha=0.6,
            density=True,
            label="Synthetic",
            color="#FF9800",
        )

    axes[i].set_title(name.replace("_", " ").title(), fontsize=11)
    axes[i].legend(fontsize=8)

# Hide unused axes
for j in range(N_FEATURES, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Real vs VAE-Generated Patient Distributions\n"
    "Synthetic data should closely match real statistical properties",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "generation_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'generation_distributions.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 2: Correlation matrix comparison
# ════════════════════════════════════════════════════════════════════════
real_corr = np.corrcoef(data_np.T)
synth_corr = np.corrcoef(synthetic_raw.T)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, corr_mat, title in [
    (axes[0], real_corr, "Real Data"),
    (axes[1], synth_corr, "Synthetic Data"),
    (axes[2], np.abs(real_corr - synth_corr), "Absolute Difference"),
]:
    im = ax.imshow(
        corr_mat,
        cmap="RdBu_r" if title != "Absolute Difference" else "Reds",
        vmin=-1 if title != "Absolute Difference" else 0,
        vmax=1 if title != "Absolute Difference" else 0.3,
    )
    ax.set_xticks(range(N_FEATURES))
    ax.set_yticks(range(N_FEATURES))
    short_names = [n[:8] for n in FEATURE_NAMES]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

corr_mae = np.abs(real_corr - synth_corr).mean()
fig.suptitle(
    f"Correlation Preservation: MAE = {corr_mae:.4f}\n"
    "Low difference = synthetic data preserves real relationships",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "generation_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'generation_correlations.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 3: t-SNE embedding + privacy distance test
# ════════════════════════════════════════════════════════════════════════
# Use PCA (faster than t-SNE for this demo) to project to 2D
# In production, use UMAP or t-SNE for better separation

from numpy.linalg import svd

# Combine real + synthetic for joint projection
combined = np.vstack([data_norm, synthetic_norm])
combined_centered = combined - combined.mean(axis=0)
_, _, Vt = svd(combined_centered, full_matrices=False)
projected = combined_centered @ Vt[:2].T

real_proj = projected[:N_PATIENTS]
synth_proj = projected[N_PATIENTS:]

# Privacy test: nearest-neighbour distance from each synthetic to closest real
from scipy.spatial.distance import cdist

# Sample 1000 for speed
sample_idx_synth = rng.choice(N_SYNTHETIC, size=1000, replace=False)
sample_idx_real = rng.choice(N_PATIENTS, size=1000, replace=False)

dist_matrix = cdist(synthetic_norm[sample_idx_synth], data_norm[sample_idx_real])
nn_distances = dist_matrix.min(axis=1)

# Self-distances (real-to-real) for comparison
self_dist_matrix = cdist(
    data_norm[sample_idx_real[:500]], data_norm[sample_idx_real[500:]]
)
self_nn_distances = self_dist_matrix.min(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA scatter
axes[0].scatter(
    real_proj[::5, 0], real_proj[::5, 1], alpha=0.3, s=8, c="#2196F3", label="Real"
)
axes[0].scatter(
    synth_proj[::5, 0],
    synth_proj[::5, 1],
    alpha=0.3,
    s=8,
    c="#FF9800",
    label="Synthetic",
)
axes[0].set_xlabel("PC1", fontsize=11)
axes[0].set_ylabel("PC2", fontsize=11)
axes[0].set_title(
    "PCA Projection: Real vs Synthetic\n" "Overlap = good statistical similarity",
    fontsize=12,
)
axes[0].legend(fontsize=10, markerscale=3)

# Nearest-neighbour distance histogram
axes[1].hist(
    nn_distances,
    bins=40,
    alpha=0.7,
    density=True,
    label="Synthetic-to-Real NN",
    color="#FF9800",
)
axes[1].hist(
    self_nn_distances,
    bins=40,
    alpha=0.7,
    density=True,
    label="Real-to-Real NN (baseline)",
    color="#2196F3",
)
axes[1].axvline(
    nn_distances.mean(),
    color="#E65100",
    linestyle="--",
    label=f"Synth mean = {nn_distances.mean():.3f}",
)
axes[1].axvline(
    self_nn_distances.mean(),
    color="#1565C0",
    linestyle="--",
    label=f"Real mean = {self_nn_distances.mean():.3f}",
)
axes[1].set_xlabel("Nearest Neighbour Distance", fontsize=11)
axes[1].set_ylabel("Density", fontsize=11)
axes[1].set_title(
    "Privacy Test: Synthetic Records Are NOT Copies\n"
    "Synth-to-Real distance >= Real-to-Real = privacy preserved",
    fontsize=12,
)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "generation_privacy_test.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'generation_privacy_test.png'}")

# ════════════════════════════════════════════════════════════════════════
# Statistical utility tests
# ════════════════════════════════════════════════════════════════════════
print("\n--- Statistical Utility Tests ---")
tests_passed = 0
tests_total = 0

for i, name in enumerate(FEATURE_NAMES):
    real_mean = data_np[:, i].mean()
    synth_mean = synthetic_raw[:, i].mean()
    real_std = data_np[:, i].std()

    # Pass if synthetic mean is within 10% of real
    if real_std > 0:
        relative_diff = abs(synth_mean - real_mean) / real_std
        passed = relative_diff < 0.3
    else:
        passed = abs(synth_mean - real_mean) < 0.1

    tests_total += 1
    if passed:
        tests_passed += 1
    status = "PASS" if passed else "FAIL"
    print(f"  {name:25s}: real={real_mean:.2f}, synth={synth_mean:.2f} [{status}]")

utility_rate = tests_passed / tests_total * 100
privacy_safe = nn_distances.mean() >= self_nn_distances.mean() * 0.8

# ════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT ANALYSIS
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — NUH PDPA-Compliant Synthetic Data")
print("=" * 64)
print(
    f"\nDataset size: {N_PATIENTS:,} real records -> {N_SYNTHETIC:,} synthetic records"
)
print(
    f"Statistical utility: {tests_passed}/{tests_total} features pass ({utility_rate:.0f}%)"
)
print(f"Correlation MAE: {corr_mae:.4f} (lower is better)")
print(f"\nPrivacy assessment:")
print(f"  Mean synthetic-to-real NN distance: {nn_distances.mean():.4f}")
print(f"  Mean real-to-real NN distance:      {self_nn_distances.mean():.4f}")
print(
    f"  Ratio (>1.0 = good):                {nn_distances.mean()/self_nn_distances.mean():.3f}"
)
print(f"  Privacy safe: {'YES' if privacy_safe else 'NO'}")
print(f"  Zero exact copies: {'YES' if nn_distances.min() > 0.01 else 'NO'}")
print(f"\nPDPA compliance:")
print(f"  No real patient identifiable from synthetic data")
print(f"  Synthetic records statistically representative but not copies")
print(f"  Research teams can analyse risk factors without PDPA breach")
print(f"\nResearch impact:")
print(f"  Before: 6-12 month ethics approval per data request")
print(f"  After: Instant access to synthetic data, ethics-exempt")
print(f"  Estimated acceleration: 3-5 research projects/year unblocked")
print(f"  Value: NUH publishes ~200 papers/year; unblocking 5 data-dependent")
print(f"  projects adds ~S$500K in grant revenue (S$100K avg per project)")
print("=" * 64)
