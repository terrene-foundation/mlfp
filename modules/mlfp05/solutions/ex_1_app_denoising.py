# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1 Application: Sensor Data Cleaning (DAE)
# ════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are an IoT engineer at SMRT (Singapore MRT). Vibration
#   and temperature sensors on MRT trains generate readings every
#   second. Sensor noise — electrical interference, sensor drift,
#   dust on contacts — corrupts the signal. Noisy signals trigger
#   false maintenance alerts (costly) or mask real faults (dangerous).
#
# TECHNIQUE: Denoising Autoencoder (DAE)
#   Train the DAE to reconstruct clean signals from noisy inputs.
#   During training, we deliberately corrupt clean signals with
#   Gaussian noise and teach the network to remove it. At inference,
#   it filters real sensor noise the same way.
#
# WHAT YOU'LL SEE:
#   - 3-panel time-series: original clean, noisy, DAE-cleaned
#   - Signal-to-noise ratio (SNR) improvement bar chart
#   - Before/after frequency spectrum comparison
#   - Predictive maintenance accuracy improvement
#
# ESTIMATED TIME: ~30-40 min (run and interpret)
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

OUTPUT_DIR = Path("outputs/mlfp05/app_denoising")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. Generate realistic MRT sensor time-series data
# ════════════════════════════════════════════════════════════════════════
# 10 sensors (vibration, temperature, current, etc.), 1000 timesteps each.
# We generate 5000 windows of 100 timesteps for training.

N_SENSORS = 10
WINDOW_SIZE = 100
N_WINDOWS = 5000
NOISE_SIGMA = 0.4

rng = np.random.default_rng(42)

sensor_names = [
    "Vibration_X",
    "Vibration_Y",
    "Vibration_Z",
    "Temperature_Bearing",
    "Temperature_Motor",
    "Current_Draw",
    "Voltage",
    "Brake_Pressure",
    "Door_Actuator",
    "HVAC_Flow",
]


def generate_clean_window(rng: np.random.Generator) -> np.ndarray:
    """Generate one clean sensor reading window (100 timesteps x 10 sensors)."""
    t = np.linspace(0, 2 * np.pi, WINDOW_SIZE)
    window = np.zeros((WINDOW_SIZE, N_SENSORS), dtype=np.float32)

    for s in range(N_SENSORS):
        # Base frequency varies per sensor type
        freq = rng.uniform(0.5, 4.0)
        amp = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        # Primary signal
        signal = amp * np.sin(freq * t + phase)
        # Add harmonics for realism
        signal += 0.3 * amp * np.sin(2 * freq * t + rng.uniform(0, np.pi))
        signal += 0.1 * amp * np.sin(3 * freq * t + rng.uniform(0, np.pi))
        # Slow drift component
        signal += 0.2 * np.sin(0.1 * t + rng.uniform(0, np.pi))
        window[:, s] = signal

    return window


# Generate clean windows
clean_windows = np.stack([generate_clean_window(rng) for _ in range(N_WINDOWS)])

# Create noisy versions
noise = rng.normal(0, NOISE_SIGMA, clean_windows.shape).astype(np.float32)
noisy_windows = clean_windows + noise

# Normalise per sensor (across all windows)
feat_mean = clean_windows.reshape(-1, N_SENSORS).mean(axis=0)
feat_std = clean_windows.reshape(-1, N_SENSORS).std(axis=0)
feat_std[feat_std == 0] = 1.0

clean_norm = ((clean_windows - feat_mean) / feat_std).astype(np.float32)
noisy_norm = ((noisy_windows - feat_mean) / feat_std).astype(np.float32)

print(
    f"Generated {N_WINDOWS} sensor windows: {WINDOW_SIZE} timesteps x {N_SENSORS} sensors"
)
print(f"Noise level: sigma = {NOISE_SIGMA}")

# ════════════════════════════════════════════════════════════════════════
# 2. Prepare training data
# ════════════════════════════════════════════════════════════════════════
n_train = int(N_WINDOWS * 0.8)
# Flatten windows for the AE: (N, 100*10) = (N, 1000)
train_noisy = torch.tensor(noisy_norm[:n_train].reshape(n_train, -1), device=device)
train_clean = torch.tensor(clean_norm[:n_train].reshape(n_train, -1), device=device)
test_noisy = torch.tensor(
    noisy_norm[n_train:].reshape(N_WINDOWS - n_train, -1), device=device
)
test_clean = torch.tensor(
    clean_norm[n_train:].reshape(N_WINDOWS - n_train, -1), device=device
)

train_loader = DataLoader(
    TensorDataset(train_noisy, train_clean), batch_size=128, shuffle=True
)

INPUT_DIM = WINDOW_SIZE * N_SENSORS  # 1000
print(f"Training set: {n_train} windows, input dim = {INPUT_DIM}")


# ════════════════════════════════════════════════════════════════════════
# 3. Denoising Autoencoder architecture
# ════════════════════════════════════════════════════════════════════════
class SensorDenoisingAE(nn.Module):
    """DAE that takes noisy sensor readings and outputs clean ones."""

    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


model = SensorDenoisingAE(INPUT_DIM).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ════════════════════════════════════════════════════════════════════════
# 4. Train: input = noisy, target = clean
# ════════════════════════════════════════════════════════════════════════
EPOCHS = 60
losses = []

print("\nTraining denoising autoencoder...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for noisy_batch, clean_batch in train_loader:
        recon = model(noisy_batch)
        loss = criterion(recon, clean_batch)  # Target is CLEAN, not noisy
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)
    if (epoch + 1) % 15 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {avg_loss:.6f}")

# ════════════════════════════════════════════════════════════════════════
# 5. Evaluate: clean the test set
# ════════════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    cleaned_test = model(test_noisy).cpu().numpy()

test_noisy_np = test_noisy.cpu().numpy()
test_clean_np = test_clean.cpu().numpy()


def compute_snr(clean: np.ndarray, signal: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    noise = signal - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# SNR per sensor
snr_noisy_per_sensor = []
snr_cleaned_per_sensor = []
for s in range(N_SENSORS):
    clean_s = test_clean_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[:, :, s].ravel()
    noisy_s = test_noisy_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[:, :, s].ravel()
    cleaned_s = cleaned_test.reshape(-1, WINDOW_SIZE, N_SENSORS)[:, :, s].ravel()

    snr_noisy_per_sensor.append(compute_snr(clean_s, noisy_s))
    snr_cleaned_per_sensor.append(compute_snr(clean_s, cleaned_s))

snr_noisy = np.mean(snr_noisy_per_sensor)
snr_cleaned = np.mean(snr_cleaned_per_sensor)
snr_improvement = snr_cleaned - snr_noisy

print(f"\nSignal quality:")
print(f"  Noisy SNR:   {snr_noisy:.1f} dB")
print(f"  Cleaned SNR: {snr_cleaned:.1f} dB")
print(f"  Improvement: +{snr_improvement:.1f} dB")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 1: 3-panel time-series comparison
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Pick one test window, one sensor for clarity
window_idx = 5
sensor_idx = 0  # Vibration_X
t = np.arange(WINDOW_SIZE)

clean_signal = test_clean_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[
    window_idx, :, sensor_idx
]
noisy_signal = test_noisy_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[
    window_idx, :, sensor_idx
]
cleaned_signal = cleaned_test.reshape(-1, WINDOW_SIZE, N_SENSORS)[
    window_idx, :, sensor_idx
]

axes[0].plot(t, clean_signal, color="#4CAF50", linewidth=1.5)
axes[0].set_title("Original Clean Signal (Ground Truth)", fontsize=12)
axes[0].set_ylabel(f"{sensor_names[sensor_idx]}", fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noisy_signal, color="#F44336", linewidth=1, alpha=0.8)
axes[1].plot(
    t, clean_signal, color="#4CAF50", linewidth=1.5, alpha=0.4, label="Ground truth"
)
axes[1].set_title(
    f"Noisy Sensor Reading (SNR = {compute_snr(clean_signal, noisy_signal):.1f} dB)",
    fontsize=12,
)
axes[1].set_ylabel(f"{sensor_names[sensor_idx]}", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, cleaned_signal, color="#2196F3", linewidth=1.5)
axes[2].plot(
    t, clean_signal, color="#4CAF50", linewidth=1.5, alpha=0.4, label="Ground truth"
)
axes[2].set_title(
    f"DAE-Cleaned Signal (SNR = {compute_snr(clean_signal, cleaned_signal):.1f} dB)",
    fontsize=12,
)
axes[2].set_ylabel(f"{sensor_names[sensor_idx]}", fontsize=11)
axes[2].set_xlabel("Timestep", fontsize=11)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    "Denoising Autoencoder: MRT Sensor Signal Cleaning\n"
    "Top: clean truth | Middle: corrupted by noise | Bottom: DAE-cleaned",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sensor_denoising_timeseries.png", dpi=150, bbox_inches="tight"
)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'sensor_denoising_timeseries.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 2: SNR improvement per sensor
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(N_SENSORS)
width = 0.35

bars_noisy = ax.bar(
    x - width / 2,
    snr_noisy_per_sensor,
    width,
    label="Noisy",
    color="#F44336",
    alpha=0.8,
)
bars_cleaned = ax.bar(
    x + width / 2,
    snr_cleaned_per_sensor,
    width,
    label="DAE-Cleaned",
    color="#2196F3",
    alpha=0.8,
)

ax.set_xlabel("Sensor", fontsize=12)
ax.set_ylabel("Signal-to-Noise Ratio (dB)", fontsize=12)
ax.set_title("SNR Improvement Per Sensor After DAE Cleaning", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(sensor_names, rotation=45, ha="right", fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

# Add improvement labels
for i in range(N_SENSORS):
    improvement = snr_cleaned_per_sensor[i] - snr_noisy_per_sensor[i]
    ax.annotate(
        f"+{improvement:.1f}dB",
        xy=(x[i] + width / 2, snr_cleaned_per_sensor[i]),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#1565C0",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensor_snr_improvement.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'sensor_snr_improvement.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 3: Frequency spectrum comparison (before/after)
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# FFT of clean, noisy, and cleaned signals
for ax, signal, title, color in [
    (axes[0], clean_signal, "Clean (Ground Truth)", "#4CAF50"),
    (axes[1], noisy_signal, "Noisy", "#F44336"),
    (axes[2], cleaned_signal, "DAE-Cleaned", "#2196F3"),
]:
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))
    ax.plot(freqs[1:], fft_vals[1:], color=color, linewidth=1.5)
    ax.fill_between(freqs[1:], 0, fft_vals[1:], color=color, alpha=0.2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_ylabel("Magnitude", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(np.abs(np.fft.rfft(clean_signal))[1:]) * 1.5)

fig.suptitle(
    "Frequency Spectrum: DAE Removes High-Frequency Noise\n"
    "Clean signal peaks preserved, noise floor reduced",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensor_frequency_spectrum.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'sensor_frequency_spectrum.png'}")

# ════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT ANALYSIS
# ════════════════════════════════════════════════════════════════════════
# SMRT operates ~150 trains, each with ~40 sensors.
# Noisy signals cause: (1) false maintenance alerts, (2) missed real faults.
# A missed fault on average costs S$200K (track closure + delay penalties).

SMRT_TRAINS = 150
SENSORS_PER_TRAIN = 40
FALSE_ALERT_RATE_NOISY = 0.05  # 5% false alert rate with noisy data
FALSE_ALERT_RATE_CLEAN = 0.008  # 0.8% with cleaned data
MISSED_FAULT_RATE_NOISY = 0.28  # misses 28% of real faults
MISSED_FAULT_RATE_CLEAN = 0.11  # misses 11% with cleaned signals
REAL_FAULTS_PER_QUARTER = 12
COST_PER_MISSED_FAULT = 200_000
COST_PER_FALSE_ALERT = 5_000  # technician dispatch + inspection

total_readings_per_day = SMRT_TRAINS * SENSORS_PER_TRAIN * 86400  # per second

false_alerts_noisy_q = int(
    SMRT_TRAINS * SENSORS_PER_TRAIN * 90 * FALSE_ALERT_RATE_NOISY
)
false_alerts_clean_q = int(
    SMRT_TRAINS * SENSORS_PER_TRAIN * 90 * FALSE_ALERT_RATE_CLEAN
)
missed_faults_noisy = int(REAL_FAULTS_PER_QUARTER * MISSED_FAULT_RATE_NOISY)
missed_faults_clean = int(REAL_FAULTS_PER_QUARTER * MISSED_FAULT_RATE_CLEAN)

savings_false_alerts = (
    false_alerts_noisy_q - false_alerts_clean_q
) * COST_PER_FALSE_ALERT
savings_missed_faults = (
    missed_faults_noisy - missed_faults_clean
) * COST_PER_MISSED_FAULT
total_quarterly_savings = savings_false_alerts + savings_missed_faults

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — SMRT Predictive Maintenance")
print("=" * 64)
print(f"\nSMRT fleet: {SMRT_TRAINS} trains x {SENSORS_PER_TRAIN} sensors")
print(f"DAE signal improvement: +{snr_improvement:.1f} dB average across sensors")
print(f"\nFalse maintenance alerts per quarter:")
print(f"  With noisy data:    {false_alerts_noisy_q:>10,}")
print(f"  With DAE-cleaned:   {false_alerts_clean_q:>10,}")
print(
    f"  Reduction:          {false_alerts_noisy_q - false_alerts_clean_q:>10,} ({(1 - false_alerts_clean_q/false_alerts_noisy_q):.0%})"
)
print(f"\nMissed real faults per quarter (of {REAL_FAULTS_PER_QUARTER}):")
print(
    f"  With noisy data:    {missed_faults_noisy:>10} ({MISSED_FAULT_RATE_NOISY:.0%})"
)
print(
    f"  With DAE-cleaned:   {missed_faults_clean:>10} ({MISSED_FAULT_RATE_CLEAN:.0%})"
)
print(f"\nQuarterly cost savings:")
print(f"  False alert reduction: {'S$' + f'{savings_false_alerts:,.0f}':>14}")
print(f"  Fewer missed faults:   {'S$' + f'{savings_missed_faults:,.0f}':>14}")
print(f"  Total per quarter:     {'S$' + f'{total_quarterly_savings:,.0f}':>14}")
print(f"  Total per year:        {'S$' + f'{total_quarterly_savings * 4:,.0f}':>14}")
print(f"\nKey insight: Cleaned signals improve predictive maintenance accuracy")
print(
    f"from {1-MISSED_FAULT_RATE_NOISY:.0%} to {1-MISSED_FAULT_RATE_CLEAN:.0%}, preventing {missed_faults_noisy - missed_faults_clean} unplanned"
)
print(f"track closures per quarter at S${COST_PER_MISSED_FAULT:,} each.")
print("=" * 64)
