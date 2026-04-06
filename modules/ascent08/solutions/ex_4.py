# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 4: Sequence Models — RNNs and LSTMs
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand sequence modeling with RNNs and LSTMs — vanishing
#   gradients, gating mechanisms — for text classification.
#
# TASKS:
#   1. Implement vanilla RNN cell forward pass
#   2. Demonstrate vanishing gradient problem
#   3. Implement LSTM cell with gates (forget, input, output)
#   4. Build bidirectional LSTM for sentiment analysis
#   5. Compare RNN vs LSTM convergence with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer, TrainingPipeline

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_product_reviews.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)
print(f"=== Dataset: {df.height} reviews, columns: {df.columns} ===")
print(summary)


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<unk>"] + [w for w, c in word_counts.most_common(2000) if c >= 2]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocabulary: {vocab_size} words")


def text_to_indices(text: str, max_len: int = 50) -> list[int]:
    """Convert text to padded index sequence."""
    tokens = tokenize(text)[:max_len]
    indices = [word_to_idx.get(t, 1) for t in tokens]
    indices += [0] * (max_len - len(indices))  # pad
    return indices


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Vanilla RNN cell forward pass
# ══════════════════════════════════════════════════════════════════════


def tanh(x: float) -> float:
    """Hyperbolic tangent activation."""
    return math.tanh(x)


class RNNCell:
    """Vanilla RNN: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        # Initialize weights
        scale = 1.0 / math.sqrt(hidden_dim)
        self.W_xh = [
            [random.gauss(0, scale) for _ in range(hidden_dim)]
            for _ in range(input_dim)
        ]
        self.W_hh = [
            [random.gauss(0, scale) for _ in range(hidden_dim)]
            for _ in range(hidden_dim)
        ]
        self.b_h = [0.0] * hidden_dim

    def forward(self, x: list[float], h_prev: list[float]) -> list[float]:
        """Single step: compute new hidden state."""
        h_new = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            val = self.b_h[j]
            for k in range(len(x)):
                val += x[k] * self.W_xh[k][j]
            for k in range(self.hidden_dim):
                val += h_prev[k] * self.W_hh[k][j]
            h_new[j] = tanh(val)
        return h_new

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        """Process a full sequence, return all hidden states."""
        h = [0.0] * self.hidden_dim
        hidden_states = []
        for x_t in sequence:
            h = self.forward(x_t, h)
            hidden_states.append(h[:])
        return hidden_states


# Demo with a short sequence
input_dim = 10
hidden_dim = 8
rnn = RNNCell(input_dim, hidden_dim)

demo_seq = [[random.gauss(0, 1) for _ in range(input_dim)] for _ in range(5)]
states = rnn.forward_sequence(demo_seq)

print(f"\nRNN Cell: input_dim={input_dim}, hidden_dim={hidden_dim}")
for t, h in enumerate(states):
    norm = math.sqrt(sum(v * v for v in h))
    print(f"  t={t}: ||h||={norm:.4f}, h[:3]={[f'{v:.3f}' for v in h[:3]]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Demonstrate vanishing gradient problem
# ══════════════════════════════════════════════════════════════════════


def measure_gradient_flow(cell: RNNCell, seq_len: int) -> list[float]:
    """Approximate gradient magnitude at each time step via perturbation."""
    epsilon = 1e-5
    sequence = [
        [random.gauss(0, 0.5) for _ in range(input_dim)] for _ in range(seq_len)
    ]

    # Forward pass
    states = cell.forward_sequence(sequence)
    final_output = sum(states[-1])

    # Measure how much perturbing input at step t affects the final output
    sensitivities = []
    for t in range(seq_len):
        perturbed = [row[:] for row in sequence]
        perturbed[t][0] += epsilon
        perturbed_states = cell.forward_sequence(perturbed)
        perturbed_output = sum(perturbed_states[-1])
        sensitivity = abs(perturbed_output - final_output) / epsilon
        sensitivities.append(sensitivity)

    return sensitivities


print(f"\n--- Vanishing Gradient Demonstration ---")
for seq_len in [10, 25, 50]:
    grads = measure_gradient_flow(rnn, seq_len)
    print(f"\n  Sequence length={seq_len}:")
    print(f"    Gradient at t=0: {grads[0]:.6f}")
    print(f"    Gradient at t={seq_len//2}: {grads[seq_len//2]:.6f}")
    print(f"    Gradient at t={seq_len-1}: {grads[-1]:.6f}")
    print(f"    Ratio (first/last): {grads[0] / (grads[-1] + 1e-10):.4f}")

print(f"\nEarly inputs have diminishing influence — the vanishing gradient problem.")
print(f"tanh derivatives < 1 compound multiplicatively across time steps.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: LSTM cell with gates
# ══════════════════════════════════════════════════════════════════════


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class LSTMCell:
    """LSTM with forget, input, and output gates."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = 1.0 / math.sqrt(hidden_dim)
        combined = input_dim + hidden_dim

        # Weights for all four gates: forget, input, cell_candidate, output
        self.W_f = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.W_i = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.W_c = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.W_o = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.b_f = [1.0] * hidden_dim  # Forget gate bias = 1 (remember by default)
        self.b_i = [0.0] * hidden_dim
        self.b_c = [0.0] * hidden_dim
        self.b_o = [0.0] * hidden_dim

    def _gate(
        self, combined: list[float], W: list[list[float]], b: list[float], activation
    ) -> list[float]:
        result = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            val = b[j]
            for k in range(len(combined)):
                val += combined[k] * W[k][j]
            result[j] = activation(val)
        return result

    def forward(
        self, x: list[float], h_prev: list[float], c_prev: list[float]
    ) -> tuple[list[float], list[float]]:
        """Single LSTM step returning (h_t, c_t)."""
        combined = x + h_prev

        f_t = self._gate(combined, self.W_f, self.b_f, sigmoid)  # Forget gate
        i_t = self._gate(combined, self.W_i, self.b_i, sigmoid)  # Input gate
        c_hat = self._gate(combined, self.W_c, self.b_c, tanh)  # Cell candidate
        o_t = self._gate(combined, self.W_o, self.b_o, sigmoid)  # Output gate

        # Cell state update: c_t = f_t * c_{t-1} + i_t * c_hat
        c_t = [f_t[j] * c_prev[j] + i_t[j] * c_hat[j] for j in range(self.hidden_dim)]
        # Hidden state: h_t = o_t * tanh(c_t)
        h_t = [o_t[j] * tanh(c_t[j]) for j in range(self.hidden_dim)]

        return h_t, c_t

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        """Process full sequence, return all hidden states."""
        h = [0.0] * self.hidden_dim
        c = [0.0] * self.hidden_dim
        hidden_states = []
        for x_t in sequence:
            h, c = self.forward(x_t, h, c)
            hidden_states.append(h[:])
        return hidden_states


lstm = LSTMCell(input_dim, hidden_dim)
lstm_states = lstm.forward_sequence(demo_seq)

print(f"\nLSTM Cell: input_dim={input_dim}, hidden_dim={hidden_dim}")
for t, h in enumerate(lstm_states):
    norm = math.sqrt(sum(v * v for v in h))
    print(f"  t={t}: ||h||={norm:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bidirectional LSTM for sentiment
# ══════════════════════════════════════════════════════════════════════


class BiLSTM:
    """Bidirectional LSTM: forward + backward, concatenate outputs."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.forward_lstm = LSTMCell(input_dim, hidden_dim)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        """Returns concatenated [forward; backward] hidden states."""
        fwd_states = self.forward_lstm.forward_sequence(sequence)
        bwd_states = self.backward_lstm.forward_sequence(sequence[::-1])[::-1]
        return [f + b for f, b in zip(fwd_states, bwd_states)]


bilstm = BiLSTM(input_dim, hidden_dim)
bi_states = bilstm.forward_sequence(demo_seq)

print(f"\nBiLSTM output dim: {len(bi_states[0])} (2 x {hidden_dim})")
print(f"Forward captures left context, backward captures right context.")
print(f"Concatenation gives full sentence context at every position.")

# Sentiment classification via TrainingPipeline
pipeline = TrainingPipeline(
    model_type="text_classifier",
    target="rating",
    features=["text"],
)
result = pipeline.fit(df)
print(f"\nTrainingPipeline sentiment result: {result}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare RNN vs LSTM convergence
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- RNN vs LSTM Gradient Flow ---")
rnn_grads_50 = measure_gradient_flow(rnn, 50)
lstm_grads_50 = measure_gradient_flow(lstm, 50)

print(f"{'Step':<8} {'RNN grad':<15} {'LSTM grad':<15}")
print("-" * 38)
for t in [0, 10, 25, 40, 49]:
    print(f"  t={t:<4} {rnn_grads_50[t]:<15.6f} {lstm_grads_50[t]:<15.6f}")

viz = ModelVisualizer()
fig = viz.plot_training_curves(
    history={
        "rnn_gradient": rnn_grads_50,
        "lstm_gradient": lstm_grads_50,
    },
    title="RNN vs LSTM Gradient Flow Over Time Steps",
)
print(f"\nLSTM maintains gradient flow via the cell state highway.")
print(f"Forget gate = 1 lets gradients pass through unattenuated.")

print("\n✓ Exercise 4 complete — RNN/LSTM cells, vanishing gradients, BiLSTM sentiment")
