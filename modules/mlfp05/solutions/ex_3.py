# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3: RNNs, LSTM, and GRU for Sequence Modelling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build vanilla RNN, LSTM, and GRU networks with torch.nn
#   - Explain all six LSTM gate equations and why they solve vanishing gradients
#   - Compare gradient magnitudes across RNN / LSTM / GRU on long sequences
#   - Train sequence-to-one regressors on synthetic time-series data
#   - Apply gradient clipping (standard practice for recurrent networks)
#
# PREREQUISITES: M5/ex_2 (CNNs, PyTorch training loops, batch norm).
# ESTIMATED TIME: ~45 min
# DATASET: Synthetic multi-variate time-series (vectorised numpy generator).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kailash_ml import ModelVisualizer

# ── Reproducibility ─────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Synthetic time-series ───────────────────────────────────────────────
# Goal: given a window of 20 timesteps with 3 features, predict the next value
# of a target signal built from trends, seasonality, and noise.
def make_timeseries(n_samples: int = 800, seq_len: int = 20, n_features: int = 3):
    """Vectorised multi-variate time-series generator (no Python loops)."""
    t = np.arange(seq_len + 1, dtype=np.float32)
    phases = np.random.rand(n_samples, n_features).astype(np.float32) * 2 * np.pi
    freqs = 0.1 + 0.3 * np.random.rand(n_samples, n_features).astype(np.float32)
    trends = 0.05 * np.random.randn(n_samples, n_features).astype(np.float32)
    # (n_samples, n_features, seq_len+1)
    signal = (
        np.sin(freqs[:, :, None] * t[None, None, :] + phases[:, :, None])
        + trends[:, :, None] * t[None, None, :]
    )
    noise = 0.1 * np.random.randn(*signal.shape).astype(np.float32)
    signal = signal + noise
    # Features: (n_samples, seq_len, n_features); Target: next value of feature 0
    X = signal[:, :, :seq_len].transpose(0, 2, 1).astype(np.float32)
    y = signal[:, 0, seq_len].astype(np.float32)
    return X, y


X_np, y_np = make_timeseries(n_samples=800, seq_len=20, n_features=3)
X = torch.from_numpy(X_np).to(device)
y = torch.from_numpy(y_np).to(device)
print(f"X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Vanilla RNN
# ════════════════════════════════════════════════════════════════════════
# A vanilla RNN applies: h_t = tanh(W_hh h_{t-1} + W_xh x_t + b).
# It struggles on long sequences because the gradient passes through tanh
# at every step — small eigenvalues shrink the signal to zero (vanishing),
# large eigenvalues blow it up (exploding). torch.nn.RNN wraps this for us.
class VanillaRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)        # (batch, seq, hidden)
        return self.head(out[:, -1]).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════
# PART 2 — LSTM: six gate equations, cell state highway
# ════════════════════════════════════════════════════════════════════════
# LSTM solves vanishing gradients with a separate cell state C_t that passes
# through additive updates rather than multiplicative ones.
#
#   f_t = sigma(W_f [h_{t-1}, x_t] + b_f)     (forget gate)
#   i_t = sigma(W_i [h_{t-1}, x_t] + b_i)     (input gate)
#   g_t = tanh (W_g [h_{t-1}, x_t] + b_g)     (candidate cell)
#   C_t = f_t * C_{t-1} + i_t * g_t           (cell update)
#   o_t = sigma(W_o [h_{t-1}, x_t] + b_o)     (output gate)
#   h_t = o_t * tanh(C_t)                     (hidden state)
#
# torch.nn.LSTM implements all six in optimised C++/CUDA.
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# For a transparent educational view, here is one step of the LSTM in pure
# torch — still vectorised across the batch dimension. Production code should
# use nn.LSTM; this is only to make the gate equations concrete.
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # One Linear produces all four gate pre-activations at once
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([x_t, h_prev], dim=-1)
        pre = self.gates(combined)
        i, f, g, o = pre.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ════════════════════════════════════════════════════════════════════════
# PART 3 — GRU: simpler, two gates
# ════════════════════════════════════════════════════════════════════════
# GRU merges forget + input into a single update gate and drops the cell
# state. Fewer parameters (roughly 75% of LSTM), similar performance in
# practice.
class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════
# Training harness
# ════════════════════════════════════════════════════════════════════════
def train_model(
    model: nn.Module,
    name: str,
    epochs: int = 8,
    lr: float = 1e-3,
    clip: float = 1.0,
) -> tuple[list[float], list[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            # Gradient clipping — essential for RNNs to prevent explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            vb_losses = [F.mse_loss(model(xb), yb).item() for xb, yb in val_loader]
        val_losses.append(float(np.mean(vb_losses)))
        print(f"  [{name}] epoch {epoch+1:2d}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")
    return train_losses, val_losses


# ── Train all three models ──────────────────────────────────────────────
print("\n── Training VanillaRNN ────────────────────────────────────────")
rnn = VanillaRNN(input_dim=3, hidden_dim=32)
rnn_train, rnn_val = train_model(rnn, "RNN", epochs=8)

print("\n── Training LSTM ──────────────────────────────────────────────")
lstm = LSTMRegressor(input_dim=3, hidden_dim=32)
lstm_train, lstm_val = train_model(lstm, "LSTM", epochs=8)

print("\n── Training GRU ───────────────────────────────────────────────")
gru = GRURegressor(input_dim=3, hidden_dim=32)
gru_train, gru_val = train_model(gru, "GRU", epochs=8)


# ════════════════════════════════════════════════════════════════════════
# PART 4 — Gradient decay across time (vanishing gradients demo)
# ════════════════════════════════════════════════════════════════════════
# We take one vanilla RNN layer, hand-roll it step by step with PyTorch,
# and look at how the gradient of the final loss w.r.t. the hidden state
# at time t shrinks as t moves backward in time. Tanh + multiplicative
# recurrence produces near-geometric decay — this is THE reason LSTMs and
# GRUs exist. Their additive cell state / update gate provides a path
# where the gradient can flow without repeated multiplication.
def rnn_gradient_decay(seq_len: int = 60) -> list[float]:
    torch.manual_seed(0)
    hidden_dim = 16
    W_xh = (torch.randn(3, hidden_dim, device=device) * 0.5).requires_grad_(True)
    W_hh = (torch.randn(hidden_dim, hidden_dim, device=device) * 0.5).requires_grad_(True)
    b = torch.zeros(hidden_dim, device=device, requires_grad=True)

    x = torch.randn(1, seq_len, 3, device=device)
    h = torch.zeros(1, hidden_dim, device=device, requires_grad=True)
    hiddens: list[torch.Tensor] = []
    for t in range(seq_len):
        h = torch.tanh(x[:, t] @ W_xh + h @ W_hh + b)
        h.retain_grad()
        hiddens.append(h)

    loss = hiddens[-1].pow(2).sum()
    loss.backward()
    norms = [float(h.grad.norm().item()) if h.grad is not None else 0.0 for h in hiddens]
    return norms


rnn_decay = rnn_gradient_decay(seq_len=60)
print("\nRNN gradient norm of loss w.r.t. hidden_t as t moves backward:")
print(f"  t=last (step 59): {rnn_decay[-1]:.4e}")
print(f"  t=mid  (step 30): {rnn_decay[30]:.4e}")
print(f"  t=first (step 0): {rnn_decay[0]:.4e}")
ratio = rnn_decay[0] / max(rnn_decay[-1], 1e-12)
print(f"  first/last ratio: {ratio:.4e}  (<< 1 means early steps contribute almost nothing to the loss gradient)")


# ════════════════════════════════════════════════════════════════════════
# PART 5 — Visualise training histories
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "RNN train": rnn_train,
        "RNN val": rnn_val,
        "LSTM train": lstm_train,
        "LSTM val": lstm_val,
        "GRU train": gru_train,
        "GRU val": gru_val,
    },
    x_label="Epoch",
    y_label="MSE",
)
fig.write_html("ex_3_training.html")
print("\nTraining history saved to ex_3_training.html")


# ════════════════════════════════════════════════════════════════════════
# PART 6 — Sanity-check the hand-rolled LSTM cell
# ════════════════════════════════════════════════════════════════════════
cell = LSTMCellFromScratch(input_dim=3, hidden_dim=16).to(device)
h = torch.zeros(4, 16, device=device)
c = torch.zeros(4, 16, device=device)
x_seq = torch.randn(4, 20, 3, device=device)
for t in range(x_seq.size(1)):       # Loop over TIME ONLY; not over hidden units
    h, c = cell(x_seq[:, t], h, c)
print(f"\nHand-rolled LSTMCell produced hidden state shape: {tuple(h.shape)}")


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Built VanillaRNN, LSTM, and GRU using torch.nn modules
  [x] Trained all three on a multivariate time-series regression task
  [x] Wrote the six LSTM gate equations as vectorised torch operations
  [x] Compared gradient magnitudes to see vanishing gradients in action
  [x] Applied gradient clipping (clip_grad_norm_) as standard RNN practice
"""
)
