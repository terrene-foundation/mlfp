# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.2: Activations and Weight Initialisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compare ReLU, GELU, Tanh, and SiLU on the same task
#   - See why zero initialisation and unscaled normal fail silently
#   - Apply Xavier/Glorot (for Sigmoid/Tanh) and Kaiming/He (for ReLU)
#   - Recognise "dying ReLU" and the fixes that exist for it
#
# PREREQUISITES: 01_xor_proof.py
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — activations as universal approximators, init as variance control
#   2. Build — one network architecture, four activation swaps
#   3. Train — identical optimiser/lr/epoch budget across all variants
#   4. Visualise — loss curves + dead-unit fraction per initialisation
#   5. Apply — ride-hailing driver churn: matching init to activation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import plotly.graph_objects as go
import torch
import torch.nn as nn

from shared.mlfp04.ex_8 import (
    N_FEATS_XOR,
    OUTPUT_DIR,
    make_xor_data,
    train_xor_net,
    viz,
    xor_accuracy,
)

print("\n" + "=" * 70)
print("  Activations and Initialisation — The Two Knobs That Must Agree")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why activation and init come as a pair
# ════════════════════════════════════════════════════════════════════════
# A ReLU neuron outputs zero (and passes zero gradient) whenever its
# pre-activation is negative. With weights initialised so pre-activations
# are centred on zero, each neuron is "off" for roughly half of the
# inputs — that is normal and fine. A neuron is DEAD only when it is off
# for EVERY input, so it never receives a gradient again. Initialisation
# controls the spread of pre-activations: too large and gradients
# explode; too small and they vanish; all-zero and every neuron is
# identical (and, for ReLU, off everywhere).
#
# Xavier/Glorot chose Var(W) = 2/(fan_in + fan_out) to keep pre-activation
# variance stable for tanh/sigmoid. Kaiming/He adjusted this to
# Var(W) = 2/fan_in specifically because ReLU kills half the signal, so
# the surviving half needs twice the variance to keep the post-activation
# variance stable across layers.
#
# Rule of thumb:
#   ReLU / LeakyReLU / GELU   -> Kaiming/He
#   Sigmoid / Tanh            -> Xavier/Glorot
#   Everything else           -> the PyTorch default (Kaiming uniform)
#   Zeros                     -> broken by symmetry — every neuron learns
#                                the same thing

X, y, y_np = make_xor_data()

# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD a reusable factory for the comparison grid
# ════════════════════════════════════════════════════════════════════════


def build_net(activation: nn.Module) -> nn.Sequential:
    """Two-hidden-layer MLP with a swappable activation."""
    # TODO: Same 32 -> 16 -> 1 MLP as 01, with ``activation`` after each
    # hidden layer
    return ____


def apply_init(net: nn.Sequential, init_fn) -> None:
    """Apply an initialisation to every linear layer."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # TODO: initialise the weight with init_fn and zero the bias
            ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the activation grid
# ════════════════════════════════════════════════════════════════════════
print("\n--- Activation comparison (matched init, Adam, 80 epochs) ---")

# Each activation is paired with the init that matches its gain:
# ReLU-family -> Kaiming/He; Tanh -> Xavier/Glorot with the tanh gain.
kaiming_relu = lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu")  # noqa: E731
# TODO: Xavier/Glorot uniform init scaled by the tanh gain
# (nn.init.calculate_gain)
xavier_tanh = ____
activation_variants: dict[str, tuple[nn.Module, object]] = {
    "ReLU": (nn.ReLU(), kaiming_relu),
    "GELU": (nn.GELU(), kaiming_relu),
    "Tanh": (nn.Tanh(), xavier_tanh),
    "SiLU/Swish": (nn.SiLU(), kaiming_relu),
}

act_histories: dict[str, list[float]] = {}
for name, (act, init_fn) in activation_variants.items():
    net = build_net(act)
    apply_init(net, init_fn)
    losses = train_xor_net(
        net, X, y, torch.optim.Adam(net.parameters(), lr=0.01), n_epochs=80
    )
    acc = xor_accuracy(net, X, y_np)
    act_histories[name] = losses
    print(f"  {name:<12}: final_loss={losses[-1]:.4f}, accuracy={acc:.4f}")

# ── Checkpoint A ───────────────────────────────────────────────────────
assert all(
    h[-1] < h[0] for h in act_histories.values()
), "Task 3: every activation should have reduced its loss"

# Now the initialisation grid (ReLU fixed, init swapped)
print("\n--- Initialisation comparison (ReLU, Adam, 80 epochs) ---")

init_variants = {
    "Xavier/Glorot": lambda w: nn.init.xavier_uniform_(w),
    "Kaiming/He": lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"),
    "Normal(0,1)": lambda w: nn.init.normal_(w, mean=0.0, std=1.0),
    "Zeros": lambda w: nn.init.zeros_(w),
}

def dead_unit_fraction(net: nn.Sequential, X_in: torch.Tensor) -> float:
    """Fraction of first-hidden-layer ReLU units that output 0 for EVERY row."""
    with torch.no_grad():
        hidden = torch.relu(net[0](X_in))
    # TODO: a unit is dead if its largest output over all rows is <= 0;
    # return the fraction of such units
    return ____


init_histories: dict[str, list[float]] = {}
dead_before: dict[str, float] = {}
dead_after: dict[str, float] = {}
for name, init_fn in init_variants.items():
    net = build_net(nn.ReLU())
    apply_init(net, init_fn)
    dead_before[name] = dead_unit_fraction(net, X)
    # TODO: train with Adam (lr=0.01) for 80 epochs, as in the activation grid
    losses = ____
    dead_after[name] = dead_unit_fraction(net, X)
    init_histories[name] = losses
    print(
        f"  {name:<15}: init_loss={losses[0]:.4f}, final_loss={losses[-1]:.4f}, "
        f"dead units {dead_before[name]:.0%} -> {dead_after[name]:.0%}"
    )

# ── Checkpoint B ───────────────────────────────────────────────────────
assert (
    init_histories["Kaiming/He"][-1] < init_histories["Zeros"][-1] - 0.1
), "Task 3: Kaiming must beat zero init (zero init is symmetry-broken)"
print("\n[ok] Checkpoint passed — activation + init grid trained\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE both grids
# ════════════════════════════════════════════════════════════════════════
fig_act = viz.training_history(act_histories, x_label="Epoch")
fig_act.update_layout(title="Activation Comparison on XOR (matched init)")
act_path = OUTPUT_DIR / "02_activation_curves.html"
fig_act.write_html(act_path)
print(f"[viz] Activation curves: {act_path}")

fig_init = viz.training_history(init_histories, x_label="Epoch")
fig_init.update_layout(title="Initialisation Comparison on XOR (ReLU hidden)")
init_path = OUTPUT_DIR / "02_initialisation_curves.html"
fig_init.write_html(init_path)
print(f"[viz] Init curves: {init_path}")

# Dead-unit fraction per initialisation, before and after training
fig_dead = go.Figure()
fig_dead.add_trace(
    go.Bar(x=list(dead_before), y=list(dead_before.values()), name="at init")
)
fig_dead.add_trace(
    go.Bar(x=list(dead_after), y=list(dead_after.values()), name="after training")
)
fig_dead.update_layout(
    title="Dead ReLU units in the first hidden layer (off for every input)",
    yaxis_title="Fraction of units",
    yaxis_tickformat=".0%",
    barmode="group",
)
dead_path = OUTPUT_DIR / "02_dead_units.html"
fig_dead.write_html(dead_path)
print(f"[viz] Dead-unit fractions: {dead_path}")

# INTERPRETATION (computed from this run):
act_final = {n: h[-1] for n, h in act_histories.items()}
spread = max(act_final.values()) - min(act_final.values())
print(
    f"\nActivation grid: final losses span {spread:.3f} "
    f"(best {min(act_final, key=act_final.get)}, "
    f"worst {max(act_final, key=act_final.get)})."
)
init_first = {n: h[0] for n, h in init_histories.items()}
init_final = {n: h[-1] for n, h in init_histories.items()}
for name in init_variants:
    print(
        f"  {name:<15} epoch-1 loss {init_first[name]:7.3f} -> final "
        f"{init_final[name]:.3f}, dead units at init {dead_before[name]:.0%}"
    )
if dead_before["Zeros"] == 1.0:
    print(
        "  -> Zero init leaves EVERY ReLU unit off for every input: no "
        "gradient reaches the hidden weights, so only the output bias "
        "learns and the loss barely moves."
    )
if init_first["Normal(0,1)"] > 2 * init_first["Kaiming/He"]:
    print(
        "  -> Unscaled Normal(0,1) starts with a far larger loss: its "
        "activations are too big, so early predictions are confidently wrong."
    )

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Ride-Hailing Driver Churn Scoring (Singapore + SEA)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A regional ride-hailing platform's retention team scores 200K
# drivers nightly for churn risk. Suppose a re-architecture swaps the
# hidden trunk from Tanh+Xavier to ReLU+Kaiming. The numbers below are
# ILLUSTRATIVE assumptions showing how to cost such a change — not
# measurements from any real company.
#
# BEFORE (Tanh + Xavier):
#   - 14 epochs to converge on each nightly retrain
#   - 3.5 hours training time per night on a T4 GPU
#   - ~S$9,000/month compute
#
# AFTER (ReLU + Kaiming):
#   - 6 epochs to converge — half the iterations
#   - 1.5 hours training time per night
#   - ~S$3,800/month compute
#
# BUSINESS IMPACT: (S$9,000 - S$3,800) x 12 = ~S$62K/year in compute plus
# a faster nightly refresh. Whether the epoch count really halves is an
# empirical question — the activation grid above shows how small the
# differences can be on an easy task, so measure on your own data.
#
# LIMITATION: If you have skip connections or normalisation layers (see
# 03_cnn_residual.py), the init choice matters less — the downstream
# layers can re-scale the signal anyway.

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Compared ReLU, GELU, Tanh, and SiLU on the same task
  [x] Reproduced the Xavier vs Kaiming vs Normal vs Zero init comparison
  [x] Measured dead ReLU units per init and saw why zero init cannot learn
  [x] Produced loss-curve and dead-unit visualisations
  [x] Costed an (illustrative) ride-hailing retraining scenario for an
      activation + init change

  KEY INSIGHT: Activation and initialisation are a paired decision. Pick
  the init that matches the activation's gain (Kaiming for ReLU-family,
  Xavier for Sigmoid/Tanh) and the network learns. Mix them wrong and
  you're training noise for 10 extra epochs.

  Next: 03_cnn_residual.py — stack the layers into a CNN with a ResBlock
  and watch the gradient highway prevent vanishing gradients.
"""
)
