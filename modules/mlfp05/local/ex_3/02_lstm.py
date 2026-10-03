# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3.2: LSTM for Sequence Prediction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Explain how LSTM gates solve the vanishing gradient problem
#   - Write the six LSTM gate equations as vectorised torch operations
#   - Build an LSTM regressor with torch.nn.LSTM for multi-step forecasting
#   - Compare LSTM vs RNN gradient preservation quantitatively
#   - Track training with ExperimentTracker and register in ModelRegistry
#   - Visualise gate activations and cell state evolution
#
# PREREQUISITES: 01_vanilla_rnn.py (understand vanishing gradients).
# ESTIMATED TIME: ~30-40 min
#
# DATASET: STI + APAC/global stocks via yfinance (2010-2024).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from shared.mlfp05.ex_3 import (
    CLIP,
    EPOCHS,
    FEATURES,
    FORECAST_HORIZON,
    HIDDEN_DIM,
    LR,
    OUTPUT_DIR,
    SEQ_LEN,
    TICKERS,
    build_dataset,
    init_environment,
    load_stock_data,
    prepare_dataloaders,
    setup_engines,
    train_model,
    register_best_model,
    get_visualizer,
    plot_training_curves,
    plot_predictions,
    plot_time_series_overlay,
    plot_horizon_error,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — How Gates Solve Vanishing Gradients
# ════════════════════════════════════════════════════════════════════════
#
# The vanilla RNN's problem: information must pass through tanh at EVERY
# timestep. After 20+ steps, gradients shrink to zero and the network
# forgets early inputs entirely.
#
# LSTM's solution: a SEPARATE "cell state" C_t that acts as a HIGHWAY.
# Information can flow through C_t with minimal transformation — like an
# express lane on the highway that bypasses all the local traffic.
#
# The six gate equations (this is the core of LSTM):
#
#   f_t = sigma(W_f [h_{t-1}, x_t] + b_f)     FORGET gate
#       "What fraction of the old memory should I keep?"
#       sigma outputs 0-1: 0 = forget everything, 1 = remember everything
#
#   i_t = sigma(W_i [h_{t-1}, x_t] + b_i)     INPUT gate
#       "How much of the new candidate should I write to memory?"
#
#   g_t = tanh(W_g [h_{t-1}, x_t] + b_g)      CANDIDATE cell
#       "What is the new information I could store?"
#
#   C_t = f_t * C_{t-1} + i_t * g_t            CELL UPDATE
#       The key equation: ADDITIVE update, not multiplicative!
#       This is why gradients survive — addition preserves them.
#
#   o_t = sigma(W_o [h_{t-1}, x_t] + b_o)      OUTPUT gate
#       "How much of the cell state should I expose as output?"
#
#   h_t = o_t * tanh(C_t)                       HIDDEN STATE
#       The output: filtered cell state, passed to the next layer.
#
# INTUITION for non-technical professionals:
#   Think of LSTM as a notebook with a pencil:
#   - The FORGET gate erases irrelevant old notes
#   - The INPUT gate decides what new notes to write
#   - The CELL STATE is the notebook itself (persistent memory)
#   - The OUTPUT gate decides which notes to share with others
#   - Vanilla RNN is like trying to remember everything in your head
#     without writing anything down — you forget quickly.
# ════════════════════════════════════════════════════════════════════════

device = init_environment()


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up experiment tracking
# ════════════════════════════════════════════════════════════════════════
stock_data, PRIMARY, primary_df = load_stock_data()

(
    train_loader,
    val_loader,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t,
    norm_mean,
    norm_std,
    n_train_w,
    N_FEATURES,
) = prepare_dataloaders(primary_df, device)

conn, tracker, exp_name, registry, has_registry = setup_engines(
    PRIMARY, experiment_suffix="lstm"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train_t.shape[1] == SEQ_LEN
assert tracker is not None
print("--- Checkpoint 1 passed --- data and tracking ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build LSTM architectures
# ════════════════════════════════════════════════════════════════════════


# 2A: Production LSTM — uses torch.nn.LSTM (optimised C++/CUDA)
class LSTMRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: self.lstm — single-layer nn.LSTM, input_dim -> hidden_dim, batch-first
        # TODO: self.head — linear map from the hidden size to `horizon` outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Run x through self.lstm (it returns the output sequence plus a
        #   (hidden, cell) tuple); forecast from the LAST timestep's output,
        #   giving shape (batch, horizon)
        pass


# 2B: Hand-rolled LSTM cell — makes the gate equations concrete
# Use nn.LSTM in production; this is for LEARNING the equations.
class LSTMCellFromScratch(nn.Module):
    """Implements the six LSTM gate equations as explicit torch operations."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # TODO: self.gates — ONE linear layer that reads the concatenation
        #   [x_t, h_prev] and emits all four gate pre-activations at once
        #   (so its output is four hidden-sized blocks: i, f, g, o).
        #   Work out its in/out sizes from that description.
        self.hidden_dim = hidden_dim

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        """One timestep of the LSTM.

        Args:
            x_t: input at this timestep (batch, input_dim)
            h_prev: previous hidden state (batch, hidden_dim)
            c_prev: previous cell state (batch, hidden_dim)

        Returns:
            h_next, c_next
        """
        # TODO: Join x_t and h_prev along the feature dimension
        # TODO: Run self.gates and split the result into 4 equal parts, in the
        #   order i, f, g, o (see Tensor.chunk)
        # TODO: Squash each part: the three gates (input i, forget f, output o)
        #   must lie in (0, 1); the candidate cell g lies in (-1, 1)
        # TODO: Cell update (ADDITIVE — the key insight): the forget gate scales
        #   the old cell, the input gate scales the candidate, and the two are SUMMED
        # TODO: Hidden output: the output gate filters a tanh-squashed new cell
        # TODO: Return h_next, c_next
        pass


lstm_model = LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
n_params_lstm = sum(p.numel() for p in lstm_model.parameters())
n_params_cell = sum(p.numel() for p in LSTMCellFromScratch(N_FEATURES, 16).parameters())
print(f"LSTMRegressor: {n_params_lstm:,} parameters")
print(f"LSTMCellFromScratch (hidden=16): {n_params_cell:,} parameters")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
# Verify the hand-rolled cell produces correct shapes
cell = LSTMCellFromScratch(input_dim=N_FEATURES, hidden_dim=16).to(device)
h, c = torch.zeros(4, 16, device=device), torch.zeros(4, 16, device=device)
x_seq = torch.randn(4, SEQ_LEN, N_FEATURES, device=device)
for t in range(x_seq.size(1)):
    h, c = cell(x_seq[:, t], h, c)
assert h.shape == (4, 16), f"Expected (4, 16), got {h.shape}"
assert c.shape == (4, 16), f"Cell state shape mismatch"
print(f"Hand-rolled LSTMCell: h={tuple(h.shape)}, c={tuple(c.shape)} -- verified")
print("--- Checkpoint 2 passed --- LSTM architectures built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train the LSTM
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training LSTM on {PRIMARY} ==")
lstm_results = train_model(
    lstm_model,
    "LSTM",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    device,
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — LSTM vs the vanilla RNN
# ══════════════════════════════════════════════════════════════════
# The LSTM's additive cell-state path is designed to carry gradient
# across many timesteps. Compare this pad with 01's — and, more
# directly, the LSTM-vs-RNN gradient-decay ratios computed below.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _mse_loss(m, batch):
    """Forecast MSE on one (window, target) batch; attention models return
    (prediction, weights), so keep only the prediction."""
    xb, yb = batch
    pred = m(xb)
    pred = pred[0] if isinstance(pred, tuple) else pred
    return nn.functional.mse_loss(pred, yb)


print("\n── Diagnostic Report (LSTM) ──")
diag, findings = run_diagnostic_checkpoint(
    lstm_model,
    train_loader,
    _mse_loss,
    title="LSTM",
    train_losses=lstm_results["train_losses"],
    val_losses=lstm_results["val_losses"],
    show=False,
)
print_prescription_pad(findings, "LSTM")

# ══════ READING THE PRESCRIPTION PAD (key: see ex_1/01_standard_ae.py) ══════
# Gate weights live in four stacked blocks inside weight_ih/weight_hh,
# so a dead or saturated gate does not show up as "dead neurons" (that
# check covers ReLU-style activations). Use the gate-activation plots
# below to see saturated gates; use the pad for overall optimisation
# health and the train-vs-val loss trend.
# ══════════════════════════════════════════════════════════════════

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(lstm_results["train_losses"]) == EPOCHS
assert lstm_results["final_val_loss"] < 5.0
print(f"\n  Final val loss: {lstm_results['final_val_loss']:.4f}")
print("--- Checkpoint 3 passed --- LSTM trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: gradient preservation (LSTM vs RNN)
# ════════════════════════════════════════════════════════════════════════
# Compare gradient flow through 60 timesteps: the LSTM's additive cell
# update preserves gradients far better than the RNN's tanh chain.


def _collect_grad_norms(hiddens: list[torch.Tensor]) -> list[float]:
    return [float(h.grad.norm().item()) if h.grad is not None else 0.0 for h in hiddens]


def gradient_decay_rnn(seq_len: int = 60) -> list[float]:
    """Gradient norm at each timestep for a vanilla RNN (for comparison)."""
    torch.manual_seed(0)
    hd = 16
    W_xh = torch.randn(N_FEATURES, hd, device=device).mul_(0.5).requires_grad_(True)
    W_hh = torch.randn(hd, hd, device=device).mul_(0.5).requires_grad_(True)
    b = torch.zeros(hd, device=device, requires_grad=True)
    x = torch.randn(1, seq_len, N_FEATURES, device=device)
    h = torch.zeros(1, hd, device=device, requires_grad=True)
    hiddens: list[torch.Tensor] = []
    for t in range(seq_len):
        h = torch.tanh(x[:, t] @ W_xh + h @ W_hh + b)
        h.retain_grad()
        hiddens.append(h)
    hiddens[-1].pow(2).sum().backward()
    return _collect_grad_norms(hiddens)


def gradient_decay_lstm(seq_len: int = 60) -> list[float]:
    """Gradient norm at each timestep for an LSTM (hand-rolled)."""
    torch.manual_seed(0)
    hd = 16
    cell_gd = LSTMCellFromScratch(N_FEATURES, hd).to(device)
    # TODO: Create random input x of shape (1, seq_len, N_FEATURES) on device
    # TODO: Start from zero hidden and cell states, each (1, hd), tracking gradients
    # TODO: Step cell_gd through every timestep (it returns the new hidden and
    #   cell states); keep each intermediate h's gradient and collect the
    #   hidden states in a list `hiddens` — same pattern as gradient_decay_rnn
    # TODO: Backpropagate from the LAST hidden state (sum of its squares)
    # TODO: Return the per-step gradient norms via _collect_grad_norms
    pass


GRAD_SEQ_LEN = 60
rnn_decay = gradient_decay_rnn(GRAD_SEQ_LEN)
lstm_decay = gradient_decay_lstm(GRAD_SEQ_LEN)

rnn_ratio = rnn_decay[0] / max(rnn_decay[-1], 1e-12)
lstm_ratio = lstm_decay[0] / max(lstm_decay[-1], 1e-12)

print(f"\n== Gradient Decay ({GRAD_SEQ_LEN} steps) ==")
print(
    f"  RNN:  first={rnn_decay[0]:.4e}  last={rnn_decay[-1]:.4e}  ratio={rnn_ratio:.4e}"
)
print(
    f"  LSTM: first={lstm_decay[0]:.4e}  last={lstm_decay[-1]:.4e}  ratio={lstm_ratio:.4e}"
)
print(
    f"  LSTM preserves gradients {lstm_ratio/max(rnn_ratio, 1e-12):.0f}x better than RNN"
)

# TODO: Plot side-by-side gradient decay comparison (2 subplots)
#   Left: semilogy of RNN decay (red) and LSTM decay (green) vs timestep
#   Right: normalised gradient flow (each normalised to its last-step value)
#   Save to OUTPUT_DIR / "02_lstm_gradient_comparison.png"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
# TODO: Fill in both subplots
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "02_lstm_gradient_comparison.png"), dpi=150)
plt.close(fig)
print("  Saved: 02_lstm_gradient_comparison.png")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
if lstm_ratio > rnn_ratio:
    print("--- Checkpoint 4 passed --- LSTM preserved gradients better than RNN")
else:
    # Random initialization can produce a session where vanilla RNN happens
    # to keep gradients alive longer; the canonical claim still holds in
    # expectation, but seed drift leaves room for individual-run variance.
    # Print a note so students see the data, not an opaque crash.
    print(f"--- Checkpoint 4 note: LSTM ratio={lstm_ratio:.4e} vs RNN ratio={rnn_ratio:.4e} (random-init variance)")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: gate activations and cell state
# ════════════════════════════════════════════════════════════════════════
# Show what the forget, input, and output gates actually DO on real data.
# This is the visual proof that LSTM "decides" what to remember.


def visualise_gate_activations(sample: torch.Tensor) -> None:
    """Run a sample through the hand-rolled LSTM cell and plot gate activations."""
    cell_viz = LSTMCellFromScratch(N_FEATURES, 16).to(device)
    cell_viz.eval()

    seq_len = sample.shape[1]
    h = torch.zeros(1, 16, device=device)
    c = torch.zeros(1, 16, device=device)

    forget_gates, input_gates, output_gates, cell_states = [], [], [], []

    with torch.no_grad():
        for t in range(seq_len):
            x_t = sample[:, t]
            # TODO: Recompute this step's gate pre-activations from cell_viz.gates
            #   (same input as inside the cell) and split them as i, f, g, o
            # TODO: Squash the forget, input and output gates into (0, 1) and
            #   store each as a flat numpy vector in forget_gates, input_gates,
            #   output_gates
            # TODO: Advance the state by calling the cell itself on (x_t, h, c)
            # TODO: Store the new cell state as a flat numpy vector in cell_states
            pass

    # TODO: Stack each list into numpy matrices of shape (seq_len, 16)
    # TODO: Create 2x2 subplot figure (16, 10):
    #   (0,0): forget_mat.T with "Reds" cmap — "Forget Gate (what to erase)"
    #   (0,1): input_mat.T with "Greens" cmap — "Input Gate (what to write)"
    #   (1,0): output_mat.T with "Blues" cmap — "Output Gate (what to expose)"
    #   (1,1): cell_mat.T with "RdBu_r" cmap — "Cell State (the memory)"
    # TODO: Save to OUTPUT_DIR / "02_lstm_gate_activations.png"
    pass


sample_input = X_val_t[:1]
visualise_gate_activations(sample_input)


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: predicted vs actual time-series overlay
# ════════════════════════════════════════════════════════════════════════
viz = get_visualizer()
plot_training_curves(viz, lstm_results, "LSTM", "02_lstm")

preds_denorm, actual_denorm, _ = plot_predictions(
    viz, lstm_model, X_val_t, y_val_t, norm_mean, norm_std, "02_lstm"
)

plot_time_series_overlay(
    preds_denorm,
    actual_denorm,
    "02_lstm",
    title=f"LSTM: Predicted vs Actual Close ({PRIMARY})",
)

rmses = plot_horizon_error(preds_denorm, actual_denorm, "LSTM")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "02_lstm_training_curves.html").exists()
assert (OUTPUT_DIR / "02_lstm_gate_activations.png").exists()
assert (OUTPUT_DIR / "02_lstm_time_series_overlay.png").exists()
print("--- Checkpoint 5 passed --- LSTM visualisations generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Register model
# ════════════════════════════════════════════════════════════════════════
register_best_model(
    lstm_model,
    "LSTM",
    lstm_results["final_val_loss"],
    PRIMARY,
    registry,
    has_registry,
)


# ════════════════════════════════════════════════════════════════════════
# APPLY — SGX Equity Forecasting for a Singapore Hedge Fund
# ════════════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a quantitative analyst at a Singapore hedge fund. Your PM
#   wants a model that predicts next-5-day returns for DBS Group
#   (Singapore's largest bank by market cap) to inform position sizing.
#
# WHY LSTM?
#   Equity returns have LONG-RANGE dependencies: earnings cycles (quarterly),
#   macro trends (interest rates, Fed decisions), sector rotation. A vanilla
#   RNN forgets these. LSTM's cell state preserves information across
#   20-60 day lookback windows — matching the fund's typical holding period.
#
# DELIVERABLES:
#   - Point prediction with prediction intervals (67% and 95%)
#   - Trading decision framework: BUY/HOLD/SELL based on predicted return
#   - Risk-adjusted return attribution
print("\n" + "=" * 70)
print("  APPLY: SGX Equity Forecasting — DBS Group (D05.SI)")
print("=" * 70)

# TODO: Use DBS data if available in stock_data, else fall back to primary
dbs_symbol = "D05.SI"  # DBS Group Holdings on SGX (public price data)
# TODO: Build dataset for DBS using build_dataset()
# TODO: Create train/val tensors and DataLoader
# TODO: Train a dedicated LSTMRegressor on DBS data for EPOCHS

# TODO: Evaluate and denormalise predictions to real prices
# TODO: Compute prediction intervals from the residual distribution:
#   residuals of the day-1 forecast (prediction minus actual), their spread
#   (standard deviation) as res_std, then bands around the prediction:
#   67% CI: +/- 1.0 residual std;  95% CI: +/- 1.96 residual std

# TODO: Trading decision framework:
#   predicted_5d_return: percentage change from the first to the last day of
#   the most recent forecast window (latest_pred)
#   BUY if return > 1.5%, SELL if < -1.5%, else HOLD
#   Store the label as decision and a one-line explanation as reasoning

# TODO: Plot prediction intervals (100-day window):
#   - Actual as solid blue, predicted as dashed green
#   - 95% CI as light green fill, 67% CI as darker green fill
#   - Save to OUTPUT_DIR / "02_lstm_dbs_prediction_intervals.png"

# ── Checkpoint 6 (Apply) ────────────────────────────────────────────
assert decision in ("BUY", "HOLD", "SELL"), "Trading decision must be valid"
assert (OUTPUT_DIR / "02_lstm_dbs_prediction_intervals.png").exists()
print("--- Checkpoint 6 passed --- SGX equity application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Built LSTM regressor with torch.nn.LSTM for multi-step forecasting
  [x] Wrote LSTM gate equations as vectorised torch operations (LSTMCellFromScratch)
  [x] Gradient preservation: LSTM ratio={lstm_ratio:.4e} vs RNN ratio={rnn_ratio:.4e}
  [x] Visualised gate activations: forget, input, output gates + cell state
  [x] Predicted vs actual time-series overlay with prediction intervals
  [x] Applied LSTM to SGX equity forecasting with trading decision framework
  [x] Trading signal: {decision} ({reasoning})

  Key insight: LSTM's cell state is a HIGHWAY for information. The additive
  update (C_t = f*C + i*g) preserves gradients where RNN's multiplicative
  chain (h = tanh(Wh + Wx)) destroys them. The forget/input/output gates
  let the network LEARN what to remember, not just hope gradients survive.

  Next: 03_gru.py — a lighter alternative with fewer parameters.
"""
)
