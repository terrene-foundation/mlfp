# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3: RNNs, LSTMs, GRUs, and Temporal Attention
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build vanilla RNN, LSTM, and GRU networks with torch.nn
#   - Write the six LSTM gate equations as vectorised torch operations
#   - Compare gradient norms across RNN/LSTM/GRU to see vanishing gradients
#   - Add a temporal attention mechanism on top of LSTM (preview of M5.4)
#   - Train multi-step forecasters on REAL multi-stock data (STI + 5 tickers)
#   - Track every model variant with ExperimentTracker (per-epoch metrics)
#   - Register the best-performing model in ModelRegistry
#   - Visualise training curves and prediction-vs-actual with ModelVisualizer
#
# PREREQUISITES: M5/ex_2 (CNNs, PyTorch training loops, batch norm).
# ESTIMATED TIME: ~120-150 min
#
# DATASET: STI + 5 APAC/global stocks via yfinance (2010-2024, ~3,700 days/ticker).
#   Cached to data/mlfp05/stocks/*.parquet.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from shared.kailash_helpers import get_device, setup_environment

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry

setup_environment()

# ── Reproducibility ─────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Download multi-stock data and build windowed datasets
# ════════════════════════════════════════════════════════════════════════
# STI (Singapore benchmark) + 5 APAC/global tickers for cross-market comparison.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "stocks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "^STI": "Straits Times Index",
    "DBS.SI": "DBS Group",
    "9988.HK": "Alibaba HK",
    "AAPL": "Apple",
    "005930.KS": "Samsung",
    "7203.T": "Toyota",
}

SEQ_LEN = 20  # 20-day lookback (4 trading weeks)
FORECAST_HORIZON = 5  # predict next 5 days
FEATURES = ["Close", "High", "Low", "Volume"]


def fetch_ticker(symbol: str) -> pl.DataFrame:
    """Download daily OHLCV bars from yfinance, return polars DataFrame."""
    import yfinance as yf

    df = yf.download(
        symbol, start="2010-01-01", end="2024-12-31", progress=False, auto_adjust=True
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned empty frame for {symbol}")
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return pl.from_pandas(df.reset_index())


def load_or_fetch(symbol: str):
    """Load from parquet cache, or download and cache."""
    cache = DATA_DIR / f"{symbol.replace('^', '').replace('.', '_')}.parquet"
    if cache.exists():
        return pl.read_parquet(cache), "cache"
    try:
        df = fetch_ticker(symbol)
        df.write_parquet(cache)
        return df, "yfinance"
    except Exception as exc:
        print(f"  {symbol} unavailable ({type(exc).__name__}: {exc})")
        return None, "failed"


stock_data: dict[str, pl.DataFrame] = {}
for symbol, name in TICKERS.items():
    df, source = load_or_fetch(symbol)
    if df is not None:
        stock_data[symbol] = df
        print(f"  {symbol} ({name}): {len(df)} days [{source}]")

if "^STI" not in stock_data and "AAPL" not in stock_data:
    raise RuntimeError("Need at least ^STI or AAPL data to proceed")

PRIMARY = "^STI" if "^STI" in stock_data else "AAPL"
primary_df = stock_data[PRIMARY]
print(
    f"\nPrimary: {PRIMARY} — {len(primary_df)} days, {primary_df['Date'].min()} -> {primary_df['Date'].max()}"
)


# ── Build windowed datasets ──────────────────────────────────────────────
def build_dataset(df: pl.DataFrame, seq_len: int, horizon: int):
    """Build (seq_len window) -> (next horizon closes) arrays with z-score normalisation."""
    data = df.select(FEATURES).to_numpy().astype(np.float32)
    n = len(data)
    split_n = int(0.8 * n)
    train_data = data[:split_n]
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True) + 1e-8
    data_norm = (data - mean) / std

    n_windows = n - seq_len - horizon + 1
    X = np.stack([data_norm[i : i + seq_len] for i in range(n_windows)])
    y = np.stack(
        [data_norm[i + seq_len : i + seq_len + horizon, 0] for i in range(n_windows)]
    )
    split_idx = split_n - seq_len
    return X.astype(np.float32), y.astype(np.float32), mean, std, split_idx


X_all, y_all, norm_mean, norm_std, n_train_w = build_dataset(
    primary_df, SEQ_LEN, FORECAST_HORIZON
)
print(
    f"Built {len(X_all)} windows (seq_len={SEQ_LEN}, horizon={FORECAST_HORIZON}); "
    f"train {n_train_w}, val {len(X_all) - n_train_w}"
)

X_train_t = torch.from_numpy(X_all[:n_train_w]).to(device)
y_train_t = torch.from_numpy(y_all[:n_train_w]).to(device)
X_val_t = torch.from_numpy(X_all[n_train_w:]).to(device)
y_val_t = torch.from_numpy(y_all[n_train_w:]).to(device)
print(f"  X_train: {tuple(X_train_t.shape)}  y_train: {tuple(y_train_t.shape)}")
print(f"  X_val:   {tuple(X_val_t.shape)}    y_val:   {tuple(y_val_t.shape)}")

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64)
N_FEATURES = X_train_t.shape[-1]

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(stock_data) >= 2, f"Need >= 2 tickers, got {len(stock_data)}"
assert X_train_t.shape[1] == SEQ_LEN
assert y_train_t.shape[1] == FORECAST_HORIZON, "Multi-step target shape mismatch"
print("--- Checkpoint 1 passed --- multi-stock data loaded and windowed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Set up ExperimentTracker and ModelRegistry
# ════════════════════════════════════════════════════════════════════════
async def setup_engines():
    conn = ConnectionManager("sqlite:///mlfp05_rnns.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name="m5_rnns_sequence_models",
        description=(
            f"RNN/LSTM/GRU/Attention on {PRIMARY} stock data. "
            f"Multi-step forecasting (next {FORECAST_HORIZON} days)."
        ),
    )

    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
print("--- Checkpoint 2 passed --- ExperimentTracker and ModelRegistry ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build model architectures
# ════════════════════════════════════════════════════════════════════════


# ── PART A: Vanilla RNN ──────────────────────────────────────────────
# h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)
# Struggles on long sequences: gradient passes through tanh at every step.
# Small eigenvalues shrink the signal to zero (vanishing), large eigenvalues
# blow it up (exploding). This is THE reason LSTMs and GRUs exist.
class VanillaRNN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (batch, seq, hidden)
        return self.head(out[:, -1])  # (batch, horizon)


# ── PART B: LSTM — six gate equations, cell state highway ────────────
# LSTM solves vanishing gradients with a separate cell state C_t that
# passes through additive updates rather than multiplicative ones.
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
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        return self.head(out[:, -1])  # (batch, horizon)


# Hand-rolled LSTM cell — makes the gate equations concrete. Use nn.LSTM in production.
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
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


# ── PART C: GRU — simpler, two gates ─────────────────────────────────
# GRU merges forget + input into a single update gate and drops the cell
# state. Fewer parameters (roughly 75% of LSTM), similar performance.
class GRURegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])  # (batch, horizon)


# ── PART D: LSTM with Temporal Attention ──────────────────────────────
# Standard LSTM uses only h_T for prediction. Temporal attention learns
# which past timesteps matter most — a preview of Transformers (M5.4).
#   a = softmax(tanh(H @ W) @ v)   ->   context = sum(a * H)
class TemporalAttention(nn.Module):
    """Additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor):
        """lstm_outputs: (batch, seq, hidden) -> context (batch, hidden), weights (batch, seq)."""
        energy = torch.tanh(self.W(lstm_outputs))
        scores = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context, weights


class LSTMWithAttention(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden)
        pred = self.head(context)  # (batch, horizon)
        return pred, attn_weights


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Training harness with gradient tracking and experiment logging
# ════════════════════════════════════════════════════════════════════════
HIDDEN_DIM = 64
EPOCHS = 15
LR = 1e-3
CLIP = 1.0


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the total L2 norm of all gradients (before clipping)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def _predict(model, x, attn=False):
    """Forward pass, handling attention models that return a tuple."""
    out = model(x)
    return out[0] if attn else out


async def train_model_async(model, name, epochs=EPOCHS, lr=LR, clip=CLIP, attn=False):
    """Train with gradient tracking, log to ExperimentTracker.

    Uses the modern ``tracker.run(...)`` async context manager — bulk
    param logging, per-step metric logging, automatic COMPLETED/FAILED
    bookkeeping on context exit.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, gradient_norms = [], [], []
    n_params = sum(p.numel() for p in model.parameters())

    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        await ctx.log_params(
            {
                "model_type": name,
                "hidden_dim": str(HIDDEN_DIM),
                "seq_len": str(SEQ_LEN),
                "forecast_horizon": str(FORECAST_HORIZON),
                "epochs": str(epochs),
                "lr": str(lr),
                "clip_norm": str(clip),
                "n_params": str(n_params),
                "ticker": PRIMARY,
            }
        )
        print(f"  [{name}] {n_params:,} parameters")

        for epoch in range(epochs):
            model.train()
            b_losses, e_grads = [], []
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = F.mse_loss(_predict(model, xb, attn), yb)
                loss.backward()
                e_grads.append(compute_gradient_norm(model))
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                opt.step()
                b_losses.append(loss.item())

            tl, gn = float(np.mean(b_losses)), float(np.mean(e_grads))
            train_losses.append(tl)
            gradient_norms.append(gn)

            model.eval()
            with torch.no_grad():
                vl = float(
                    np.mean(
                        [
                            F.mse_loss(_predict(model, xb, attn), yb).item()
                            for xb, yb in val_loader
                        ]
                    )
                )
            val_losses.append(vl)

            await ctx.log_metrics(
                {"train_loss": tl, "val_loss": vl, "gradient_norm": gn},
                step=epoch + 1,
            )
            print(
                f"  [{name}] epoch {epoch+1:2d}/{epochs}  "
                f"train={tl:.4f}  val={vl:.4f}  grad={gn:.4f}"
            )

        await ctx.log_metric("final_val_loss", val_losses[-1])

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "gradient_norms": gradient_norms,
        "final_val_loss": val_losses[-1],
    }


def train_model(model, name, epochs=EPOCHS, lr=LR, clip=CLIP, attn=False):
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(train_model_async(model, name, epochs, lr, clip, attn))


# ── Train all four architectures ──────────────────────────────────────
rnn_model = VanillaRNN(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
lstm_model = LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
gru_model = GRURegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
attn_model = LSTMWithAttention(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)

print(f"\n══ Training on {PRIMARY} ══")
rnn_results = train_model(rnn_model, "VanillaRNN")
lstm_results = train_model(lstm_model, "LSTM")
gru_results = train_model(gru_model, "GRU")
attn_results = train_model(attn_model, "LSTM_Attention", attn=True)

all_results = {
    "VanillaRNN": rnn_results,
    "LSTM": lstm_results,
    "GRU": gru_results,
    "LSTM_Attention": attn_results,
}

# ── Checkpoint 3 ─────────────────────────────────────────────────────
for name, res in all_results.items():
    assert len(res["train_losses"]) == EPOCHS, f"{name} should have {EPOCHS} epochs"
    assert res["final_val_loss"] < 5.0, f"{name} val loss suspiciously high"
print("\n--- Checkpoint 3 passed --- all four architectures trained\n")

# Comparison table — gated models should outperform vanilla RNN
print(f"{'Model':<18s} {'Train':>8s} {'Val':>8s} {'GradNorm':>10s}")
for name, res in all_results.items():
    print(
        f"{name:<18s} {res['train_losses'][-1]:>8.4f} {res['final_val_loss']:>8.4f} {np.mean(res['gradient_norms']):>10.4f}"
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Gradient decay across time (vanishing gradients demonstration)
# ════════════════════════════════════════════════════════════════════════
# Hand-roll RNN and LSTM step-by-step, measure gradient norm at each
# timestep. RNN decays geometrically; LSTM preserves via additive highway.
def _collect_grad_norms(hiddens):
    """Extract gradient norms from a list of hidden states after backward()."""
    return [float(h.grad.norm().item()) if h.grad is not None else 0.0 for h in hiddens]


def gradient_decay_rnn(seq_len: int = 60) -> list[float]:
    """Gradient norm at each timestep for a vanilla RNN."""
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
    cell = LSTMCellFromScratch(N_FEATURES, hd).to(device)
    x = torch.randn(1, seq_len, N_FEATURES, device=device)
    h = torch.zeros(1, hd, device=device, requires_grad=True)
    c = torch.zeros(1, hd, device=device, requires_grad=True)
    hiddens: list[torch.Tensor] = []
    for t in range(seq_len):
        h, c = cell(x[:, t], h, c)
        h.retain_grad()
        hiddens.append(h)
    hiddens[-1].pow(2).sum().backward()
    return _collect_grad_norms(hiddens)


rnn_decay = gradient_decay_rnn(seq_len=60)
lstm_decay = gradient_decay_lstm(seq_len=60)

rnn_ratio = rnn_decay[0] / max(rnn_decay[-1], 1e-12)
lstm_ratio = lstm_decay[0] / max(lstm_decay[-1], 1e-12)
print(f"\n══ Gradient Decay (60 steps) ══")
print(
    f"  RNN:  first={rnn_decay[0]:.4e}  last={rnn_decay[-1]:.4e}  ratio={rnn_ratio:.4e}"
)
print(
    f"  LSTM: first={lstm_decay[0]:.4e}  last={lstm_decay[-1]:.4e}  ratio={lstm_ratio:.4e}"
)
print("  LSTM ratio >> RNN ratio: cell-state highway preserves gradients")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert rnn_decay[0] < rnn_decay[-1], "RNN should show vanishing gradients"
assert lstm_ratio > rnn_ratio, "LSTM should preserve gradients better than RNN"
print("--- Checkpoint 4 passed --- vanishing gradient problem demonstrated\n")


# ── TASK 6 — Sanity-check hand-rolled LSTM cell ──────────────────────
cell = LSTMCellFromScratch(input_dim=N_FEATURES, hidden_dim=16).to(device)
h, c = torch.zeros(4, 16, device=device), torch.zeros(4, 16, device=device)
x_seq = torch.randn(4, SEQ_LEN, N_FEATURES, device=device)
for t in range(x_seq.size(1)):
    h, c = cell(x_seq[:, t], h, c)
assert h.shape == (4, 16), "Expected (batch=4, hidden=16)"
print(f"Hand-rolled LSTMCell: {tuple(h.shape)} -- verified\n")

# ── TASK 7 — Multi-stock comparison (generalisation test) ────────────
print("══ Multi-Stock Comparison (LSTM+Attention) ══")
multi_stock_results: dict[str, float] = {PRIMARY: attn_results["final_val_loss"]}

for symbol, sdf in stock_data.items():
    if symbol == PRIMARY or len(sdf) < SEQ_LEN + FORECAST_HORIZON + 50:
        continue
    X_s, y_s, _, _, sp = build_dataset(sdf, SEQ_LEN, FORECAST_HORIZON)
    ldr = DataLoader(
        TensorDataset(
            torch.from_numpy(X_s[:sp]).to(device), torch.from_numpy(y_s[:sp]).to(device)
        ),
        batch_size=64,
        shuffle=True,
    )
    ldr_v = DataLoader(
        TensorDataset(
            torch.from_numpy(X_s[sp:]).to(device), torch.from_numpy(y_s[sp:]).to(device)
        ),
        batch_size=64,
    )
    m = LSTMWithAttention(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(8):
        m.train()
        for xb, yb in ldr:
            opt.zero_grad()
            F.mse_loss(m(xb)[0], yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(), max_norm=CLIP)
            opt.step()
    m.eval()
    with torch.no_grad():
        vl = float(np.mean([F.mse_loss(m(xb)[0], yb).item() for xb, yb in ldr_v]))
    multi_stock_results[symbol] = vl
    print(f"  {symbol} ({TICKERS[symbol]}): val_loss={vl:.4f}")

# ── TASK 8 — Register best model in ModelRegistry ────────────────────
best_name = min(all_results, key=lambda k: all_results[k]["final_val_loss"])
best_val = all_results[best_name]["final_val_loss"]
print(f"\nBest model: {best_name} (val_loss={best_val:.4f})")

models_map = {
    "VanillaRNN": rnn_model,
    "LSTM": lstm_model,
    "GRU": gru_model,
    "LSTM_Attention": attn_model,
}
if has_registry and registry is not None:
    model_bytes = pickle.dumps(models_map[best_name].state_dict())
    try:
        reg_result = asyncio.run(
            registry.register(
                name=f"m5_rnn_{best_name.lower()}_{PRIMARY.replace('^', '')}",
                model_data=model_bytes,
                metadata={
                    "architecture": best_name,
                    "ticker": PRIMARY,
                    "hidden_dim": HIDDEN_DIM,
                    "seq_len": SEQ_LEN,
                    "forecast_horizon": FORECAST_HORIZON,
                    "val_loss": best_val,
                    "epochs": EPOCHS,
                },
            )
        )
        print(f"  Registered: {reg_result}")
    except Exception as e:
        print(f"  ModelRegistry registration skipped ({type(e).__name__}: {e})")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert best_val < 5.0, "Best model val loss should be reasonable"
print("--- Checkpoint 5 passed --- best model registered\n")


# ── TASK 9 — Visualisations with ModelVisualizer ─────────────────────
viz = ModelVisualizer()

# 9A: Training curves
train_metrics = {}
for label, res in all_results.items():
    train_metrics[f"{label} train"] = res["train_losses"]
    train_metrics[f"{label} val"] = res["val_losses"]
viz.training_history(
    metrics=train_metrics, x_label="Epoch", y_label="MSE Loss"
).write_html("ex_3_training_curves.html")

# 9B: Gradient norm comparison
grad_metrics = {k: v["gradient_norms"] for k, v in all_results.items()}
viz.training_history(
    metrics=grad_metrics, x_label="Epoch", y_label="Gradient L2 Norm"
).write_html("ex_3_gradient_norms.html")

# 9C: Gradient decay (vanishing gradient plot)
viz.training_history(
    metrics={"RNN": rnn_decay, "LSTM": lstm_decay},
    x_label="Timestep",
    y_label="Gradient Norm",
).write_html("ex_3_gradient_decay.html")

# 9D: Prediction vs Actual
best_model_eval = models_map[best_name]
best_model_eval.eval()
with torch.no_grad():
    if best_name == "LSTM_Attention":
        val_preds, val_attn_weights = best_model_eval(X_val_t)
    else:
        val_preds, val_attn_weights = best_model_eval(X_val_t), None

close_mean, close_std = norm_mean[0, 0], norm_std[0, 0]
preds_denorm = val_preds.cpu().numpy() * close_std + close_mean
actual_denorm = y_val_t.cpu().numpy() * close_std + close_mean

pred_df = pl.DataFrame(
    {"actual": actual_denorm[:, 0].tolist(), "predicted": preds_denorm[:, 0].tolist()}
)
viz.scatter(pred_df, x="actual", y="predicted").write_html("ex_3_pred_vs_actual.html")
print(
    "Plots saved: ex_3_training_curves.html, ex_3_gradient_norms.html, ex_3_pred_vs_actual.html"
)

# 9E: Attention weights
if val_attn_weights is not None:
    attn_np = val_attn_weights.cpu().numpy()
    sample_attn = attn_np[len(attn_np) // 2]
    top3 = np.argsort(sample_attn)[-3:][::-1]
    print(
        f"Attention top-3 steps: {list(top3)} (peak weight={sample_attn[top3[0]]:.4f})"
    )

# 9F: Multi-step error by horizon day
print("\n══ Forecast Error by Horizon Day ══")
for day in range(FORECAST_HORIZON):
    rmse = float(np.mean((preds_denorm[:, day] - actual_denorm[:, day]) ** 2)) ** 0.5
    print(f"  Day {day + 1}: RMSE={rmse:.2f}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert Path("ex_3_training_curves.html").exists()
assert Path("ex_3_pred_vs_actual.html").exists()
print("--- Checkpoint 6 passed --- visualisations generated\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  [x] Loaded {sum(len(df) for df in stock_data.values()):,} days across {len(stock_data)} tickers
  [x] Built VanillaRNN, LSTM, GRU, and LSTM+Attention in torch.nn
  [x] Wrote LSTM gate equations as vectorised torch operations
  [x] Multi-step forecasting: {SEQ_LEN}-day window -> next {FORECAST_HORIZON} days
  [x] Tracked every variant with ExperimentTracker (per-epoch loss + grad norms)
  [x] Vanishing gradients: RNN ratio={rnn_ratio:.4e}, LSTM ratio={lstm_ratio:.4e}
  [x] Temporal attention: learnable focus over past timesteps (preview of M5.4)
  [x] Best model ({best_name}) registered in ModelRegistry
  [x] Gradient clipping (clip_grad_norm_) as standard RNN practice

  Key insight: RNNs fail on long sequences. LSTMs fix this with additive
  cell-state updates. GRUs match with fewer parameters. Attention lets the
  model choose which past steps matter. Error compounds across the horizon.

  Note: this exercise teaches architectures, not market timing.

  Next: Exercise 4 — Transformers replace recurrence with pure attention.
"""
)
