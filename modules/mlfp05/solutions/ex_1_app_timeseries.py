# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1 Application: Financial Time-Series Anomaly
# ════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a quantitative analyst at a Singapore hedge fund
#   monitoring SGX equities for regime changes. Markets shift
#   between calm and crisis states — COVID crash, rate hikes,
#   earnings shocks. Your PM asks: "Can we detect regime changes
#   early enough to reduce portfolio drawdown?"
#
# TECHNIQUE: Recurrent (LSTM) Autoencoder
#   Train an LSTM-AE on "normal" market periods. Feed sequences
#   of daily returns. High reconstruction error = the market is
#   behaving unlike anything in training = regime change.
#
# WHAT YOU'LL SEE:
#   - Stock price with reconstruction error overlay (shaded anomaly regions)
#   - Detected anomaly dates vs actual market events
#   - Portfolio drawdown analysis: reduced exposure saves capital
#   - Dollar-value impact for a S$100M portfolio
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

OUTPUT_DIR = Path("outputs/mlfp05/app_timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. Generate realistic SGX equity data
# ════════════════════════════════════════════════════════════════════════
# In production, use yfinance or pre-downloaded parquet for real SGX data.
# Here we generate 5 Singapore stocks with realistic return distributions,
# correlations, and embedded crisis periods (COVID, rate hikes, etc.).

N_DAYS = 1500  # ~6 years of trading days
N_STOCKS = 5
STOCK_NAMES = ["DBS", "OCBC", "Singtel", "CapitaLand", "Keppel"]

rng = np.random.default_rng(42)

# Base parameters per stock (annualised)
base_returns = np.array([0.08, 0.07, 0.04, 0.06, 0.05])  # annual drift
base_vols = np.array([0.18, 0.16, 0.22, 0.20, 0.24])  # annual vol

# Correlation matrix (Singapore banks are highly correlated)
corr = np.array(
    [
        [1.00, 0.85, 0.45, 0.55, 0.50],
        [0.85, 1.00, 0.40, 0.50, 0.45],
        [0.45, 0.40, 1.00, 0.35, 0.30],
        [0.55, 0.50, 0.35, 1.00, 0.60],
        [0.50, 0.45, 0.30, 0.60, 1.00],
    ]
)
L = np.linalg.cholesky(corr)

# Generate daily returns with regime changes
daily_returns = np.zeros((N_DAYS, N_STOCKS), dtype=np.float32)
regime_labels = np.zeros(N_DAYS, dtype=np.int32)  # 0=normal, 1=crisis

# Define crisis periods (day ranges)
crisis_periods = [
    (300, 360, "COVID Crash (Mar 2020)"),
    (600, 640, "Rate Hike Shock (2022)"),
    (900, 930, "Banking Stress (2023)"),
    (1200, 1230, "Geopolitical Crisis"),
]

for day in range(N_DAYS):
    # Check if we're in a crisis
    in_crisis = False
    for start, end, _ in crisis_periods:
        if start <= day < end:
            in_crisis = True
            break

    if in_crisis:
        regime_labels[day] = 1
        # Crisis: higher vol, negative drift, higher correlation
        crisis_vol = base_vols * 3.0
        crisis_drift = -base_returns * 2.0
        z = rng.standard_normal(N_STOCKS)
        corr_z = L @ z
        daily_returns[day] = crisis_drift / 252 + crisis_vol / np.sqrt(252) * corr_z
    else:
        regime_labels[day] = 0
        z = rng.standard_normal(N_STOCKS)
        corr_z = L @ z
        daily_returns[day] = base_returns / 252 + base_vols / np.sqrt(252) * corr_z

# Compute cumulative prices (start at 100)
prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))

print(f"Generated {N_DAYS} trading days for {N_STOCKS} SGX stocks")
print(f"Crisis days: {regime_labels.sum()} ({regime_labels.mean()*100:.1f}%)")
for start, end, name in crisis_periods:
    print(f"  Day {start}-{end}: {name}")

# ════════════════════════════════════════════════════════════════════════
# 2. Create windowed sequences for LSTM-AE
# ════════════════════════════════════════════════════════════════════════
SEQ_LEN = 20  # 20-day windows (one trading month)

# Features per timestep: daily returns + 5-day rolling vol for each stock
rolling_vol = np.zeros_like(daily_returns)
for i in range(5, N_DAYS):
    rolling_vol[i] = daily_returns[i - 5 : i].std(axis=0)

features = np.concatenate([daily_returns, rolling_vol], axis=1)  # (N_DAYS, 10)
N_FEATURES = features.shape[1]

# Normalise features
feat_mean = features.mean(axis=0)
feat_std = features.std(axis=0)
feat_std[feat_std == 0] = 1.0
features_norm = ((features - feat_mean) / feat_std).astype(np.float32)

# Create sequences
sequences = []
seq_labels = []
seq_days = []
for i in range(N_DAYS - SEQ_LEN):
    sequences.append(features_norm[i : i + SEQ_LEN])
    # Label: 1 if any day in window is crisis
    seq_labels.append(int(regime_labels[i : i + SEQ_LEN].any()))
    seq_days.append(i + SEQ_LEN)  # end day of window

sequences = np.array(sequences, dtype=np.float32)
seq_labels = np.array(seq_labels)
seq_days = np.array(seq_days)

# Split: train on normal-only, test on all
normal_mask = seq_labels == 0
train_seqs = sequences[normal_mask][: int(normal_mask.sum() * 0.8)]
test_seqs = sequences
test_labels_arr = seq_labels

train_tensor = torch.tensor(train_seqs, device=device)
test_tensor = torch.tensor(test_seqs, device=device)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)

print(
    f"\nSequences: {len(sequences)} total, {normal_mask.sum()} normal, {(~normal_mask).sum()} crisis"
)
print(f"Training on {len(train_seqs)} normal-only sequences")
print(f"Sequence shape: ({SEQ_LEN}, {N_FEATURES})")


# ════════════════════════════════════════════════════════════════════════
# 3. LSTM Autoencoder architecture
# ════════════════════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, c) = self.lstm(x)
        return h, c


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        # Repeat the last input across seq_len
        x_repeated = x[:, -1:, :].repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x_repeated, (h, c))
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, c = self.encoder(x)
        return self.decoder(x, h, c)


HIDDEN_DIM = 32
model = LSTMAutoencoder(N_FEATURES, HIDDEN_DIM, SEQ_LEN).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ════════════════════════════════════════════════════════════════════════
# 4. Train on normal market periods only
# ════════════════════════════════════════════════════════════════════════
EPOCHS = 60
losses = []

print("\nTraining LSTM autoencoder on normal market periods...")
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
    if (epoch + 1) % 15 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {avg_loss:.6f}")

# ════════════════════════════════════════════════════════════════════════
# 5. Compute reconstruction errors on all sequences
# ════════════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    # Process in chunks to avoid OOM
    chunk_size = 512
    all_errors = []
    for i in range(0, len(test_tensor), chunk_size):
        chunk = test_tensor[i : i + chunk_size]
        recon = model(chunk)
        chunk_errors = ((chunk - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
        all_errors.append(chunk_errors)
    recon_errors = np.concatenate(all_errors)

normal_errors = recon_errors[test_labels_arr == 0]
crisis_errors = recon_errors[test_labels_arr == 1]

print(f"\nReconstruction errors:")
print(
    f"  Normal:  mean={normal_errors.mean():.4f}, p95={np.percentile(normal_errors, 95):.4f}"
)
print(
    f"  Crisis:  mean={crisis_errors.mean():.4f}, p95={np.percentile(crisis_errors, 95):.4f}"
)
print(f"  Separation: {crisis_errors.mean() / normal_errors.mean():.1f}x")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 1: Stock price with anomaly score overlay
# ════════════════════════════════════════════════════════════════════════
# Set anomaly threshold at 95th percentile of normal reconstruction error
threshold = np.percentile(normal_errors, 95)
is_anomaly = recon_errors > threshold

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})

# Top: DBS price with crisis shading
ax1 = axes[0]
days = np.arange(N_DAYS)
ax1.plot(days, prices[:, 0], color="#1565C0", linewidth=1.2, label="DBS Price")

# Shade actual crisis periods
for start, end, name in crisis_periods:
    ax1.axvspan(start, end, alpha=0.15, color="#F44336", label=name)

ax1.set_ylabel("Price (S$)", fontsize=12)
ax1.set_title("DBS Group — Price with Market Regime Detection", fontsize=14)
ax1.legend(fontsize=9, loc="upper left", ncol=2)
ax1.grid(True, alpha=0.3)

# Bottom: Anomaly score
ax2 = axes[1]
ax2.fill_between(
    seq_days, 0, recon_errors, alpha=0.4, color="#9E9E9E", label="Anomaly Score"
)
ax2.fill_between(
    seq_days,
    0,
    recon_errors,
    where=is_anomaly,
    alpha=0.7,
    color="#F44336",
    label="Detected Anomaly",
)
ax2.axhline(
    threshold,
    color="#FF9800",
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold (p95) = {threshold:.4f}",
)
ax2.set_ylabel("Reconstruction Error", fontsize=12)
ax2.set_xlabel("Trading Day", fontsize=12)
ax2.set_title("LSTM-AE Anomaly Score — Spikes During Regime Changes", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "timeseries_anomaly_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'timeseries_anomaly_overlay.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 2: Detection performance — detected vs actual events
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

# Plot per-event detection
event_names = [name for _, _, name in crisis_periods]
event_detected = []
event_lead_days = []

for start, end, name in crisis_periods:
    # Check if any anomaly was detected in the window [start-30, end]
    window_mask = (seq_days >= start - 30) & (seq_days <= end)
    window_anomalies = is_anomaly[window_mask]

    detected = window_anomalies.any()
    event_detected.append(detected)

    if detected:
        # First detection relative to crisis start
        first_detect_idx = np.where(window_anomalies)[0][0]
        first_detect_day = seq_days[window_mask][first_detect_idx]
        lead = start - first_detect_day
        event_lead_days.append(max(lead, 0))
    else:
        event_lead_days.append(0)

colors = ["#4CAF50" if d else "#F44336" for d in event_detected]
bars = ax.barh(range(len(event_names)), event_lead_days, color=colors, height=0.5)

for i, (detected, lead) in enumerate(zip(event_detected, event_lead_days)):
    status = f"Detected ({lead}d early)" if detected else "MISSED"
    ax.text(
        max(event_lead_days) * 0.02 + lead,
        i,
        status,
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1B5E20" if detected else "#B71C1C",
    )

ax.set_yticks(range(len(event_names)))
ax.set_yticklabels(event_names, fontsize=11)
ax.set_xlabel("Days of Early Warning Before Crisis Start", fontsize=12)
ax.set_title(
    "Event Detection: LSTM-AE Catches Regime Changes\n"
    "Green = detected early, Red = missed",
    fontsize=13,
)
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "timeseries_event_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'timeseries_event_detection.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 3: Portfolio drawdown — with vs without anomaly signal
# ════════════════════════════════════════════════════════════════════════
# Strategy: equal-weight portfolio. When anomaly detected, reduce exposure to 20%.
PORTFOLIO_VALUE = 100_000_000  # S$100M

# Equal-weight daily returns
portfolio_returns = daily_returns.mean(axis=1)

# Passive: always fully invested
passive_cum = np.cumprod(1 + portfolio_returns) * PORTFOLIO_VALUE

# Anomaly-adjusted: reduce to 20% exposure when anomaly detected
# Map anomaly signal back to trading days
anomaly_by_day = np.zeros(N_DAYS, dtype=bool)
for i, day in enumerate(seq_days):
    if is_anomaly[i] and day < N_DAYS:
        # Anomaly persists for the next 5 trading days
        anomaly_by_day[day : min(day + 5, N_DAYS)] = True

adjusted_returns = np.where(anomaly_by_day, portfolio_returns * 0.2, portfolio_returns)
adjusted_cum = np.cumprod(1 + adjusted_returns) * PORTFOLIO_VALUE

# Drawdowns
passive_peak = np.maximum.accumulate(passive_cum)
passive_dd = (passive_cum - passive_peak) / passive_peak

adjusted_peak = np.maximum.accumulate(adjusted_cum)
adjusted_dd = (adjusted_cum - adjusted_peak) / adjusted_peak

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})

# Portfolio value
axes[0].plot(
    days,
    passive_cum / 1e6,
    color="#F44336",
    linewidth=1.5,
    label=f"Passive (final: S${passive_cum[-1]/1e6:.1f}M)",
)
axes[0].plot(
    days,
    adjusted_cum / 1e6,
    color="#4CAF50",
    linewidth=1.5,
    label=f"Anomaly-Adjusted (final: S${adjusted_cum[-1]/1e6:.1f}M)",
)
axes[0].set_ylabel("Portfolio Value (S$M)", fontsize=12)
axes[0].set_title(
    "S$100M Equal-Weight SGX Portfolio: Passive vs Anomaly-Adjusted", fontsize=14
)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Drawdown
axes[1].fill_between(
    days, passive_dd * 100, alpha=0.5, color="#F44336", label="Passive"
)
axes[1].fill_between(
    days, adjusted_dd * 100, alpha=0.5, color="#4CAF50", label="Anomaly-Adjusted"
)
axes[1].set_ylabel("Drawdown (%)", fontsize=12)
axes[1].set_xlabel("Trading Day", fontsize=12)
axes[1].set_title("Maximum Drawdown Comparison", fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "timeseries_portfolio_drawdown.png", dpi=150, bbox_inches="tight"
)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'timeseries_portfolio_drawdown.png'}")

# ════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT ANALYSIS
# ════════════════════════════════════════════════════════════════════════
max_dd_passive = passive_dd.min() * 100
max_dd_adjusted = adjusted_dd.min() * 100
dd_improvement = max_dd_passive - max_dd_adjusted

n_events_detected = sum(event_detected)
n_events_total = len(crisis_periods)

# Dollar savings at worst drawdown
passive_worst_loss = PORTFOLIO_VALUE * abs(passive_dd.min())
adjusted_worst_loss = PORTFOLIO_VALUE * abs(adjusted_dd.min())
dollar_saved = passive_worst_loss - adjusted_worst_loss

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — SGX Regime Detection (S$100M Fund)")
print("=" * 64)
print(f"\nMarket event detection: {n_events_detected}/{n_events_total} events caught")
for i, (_, _, name) in enumerate(crisis_periods):
    status = f"Detected {event_lead_days[i]}d early" if event_detected[i] else "MISSED"
    print(f"  {name}: {status}")
print(f"\nPortfolio performance:")
print(f"  Passive max drawdown:     {max_dd_passive:>10.1f}%")
print(f"  Adjusted max drawdown:    {max_dd_adjusted:>10.1f}%")
print(f"  Drawdown improvement:     {dd_improvement:>10.1f} pp")
print(f"\nDollar impact at worst drawdown:")
print(f"  Passive worst loss:       {'S$' + f'{passive_worst_loss:,.0f}':>14}")
print(f"  Adjusted worst loss:      {'S$' + f'{adjusted_worst_loss:,.0f}':>14}")
print(f"  Capital preserved:        {'S$' + f'{dollar_saved:,.0f}':>14}")
print(f"\nFinal portfolio value (S$100M initial):")
print(f"  Passive:                  {'S$' + f'{passive_cum[-1]:,.0f}':>14}")
print(f"  Anomaly-adjusted:         {'S$' + f'{adjusted_cum[-1]:,.0f}':>14}")
print(f"\nKey insight: Anomaly-triggered position reduction would have")
print(f"preserved S${dollar_saved:,.0f} at the worst drawdown point.")
print(
    f"The LSTM-AE detects regime changes {np.mean(event_lead_days):.0f} days early on average,"
)
print(f"giving the PM time to reduce exposure before the crash deepens.")
print("=" * 64)
