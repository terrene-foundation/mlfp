# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 3.2: The DPO Loss From Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive DPO from the RLHF objective (bypass the reward model)
#   - Implement the DPO loss in PyTorch and verify against known cases
#   - Understand beta as the KL-penalty coefficient (the anchor to pi_ref)
#   - Visualise how the DPO loss responds to policy preference margin
#   - Run a beta sensitivity sweep and interpret the curve
#   - Apply to MAS-compliant model alignment in Singapore finance
#
# PREREQUISITES: 01_preference_data.py (preference triple intuition).
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Derivation walkthrough: RLHF -> Bradley-Terry -> closed-form DPO
#   2. Implement dpo_loss() and verify with synthetic log-probabilities
#   3. Visualise loss vs policy preference margin
#   4. Beta sensitivity sweep on a fixed batch
#   5. Apply: pick beta for a MAS-regulated Singapore robo-advisor
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl
import torch
import torch.nn.functional as F

from shared.mlfp06.ex_3 import (
    OUTPUT_DIR,
    show_beta_sensitivity,
    show_dpo_loss_curve_by_margin,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — RLHF to DPO in one page
# ════════════════════════════════════════════════════════════════════════
# RLHF:    max_pi E[r(x,y)] - beta * KL(pi || pi_ref)
# Optimal: pi*(y|x) proportional to pi_ref(y|x) * exp(r/beta)
# Bradley-Terry + substitution -> Z(x) cancels -> closed-form DPO:
#   L = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
#                          - log pi(y_l|x)/pi_ref(y_l|x)))]
# The reward model is IMPLICIT in the policy. That is the big idea.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement dpo_loss() in the student file
#          (The shared module has the canonical reference, but here you
#           write it yourself so you understand each term.)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Implement and verify DPO loss")
print("=" * 70)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the DPO loss from scratch."""
    # TODO: Compute the chosen log-ratio: policy_chosen - ref_chosen
    chosen_log_ratio = ____
    # TODO: Compute the rejected log-ratio: policy_rejected - ref_rejected
    rejected_log_ratio = ____
    # TODO: Compute the DPO logits: beta * (chosen_log_ratio - rejected_log_ratio)
    logits = ____
    # TODO: Return -F.logsigmoid(logits).mean()
    return ____


torch.manual_seed(42)
batch_size = 16

# Clear mean separation (+1.0 vs -1.0) with small noise (×0.3) so the
# reversed-prefs ordering is unambiguous regardless of the RNG stream.
policy_chosen = torch.randn(batch_size) * 0.3 + 1.0
policy_rejected = torch.randn(batch_size) * 0.3 - 1.0
ref_chosen = torch.randn(batch_size) * 0.3
ref_rejected = torch.randn(batch_size) * 0.3

loss_val = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
print(f"DPO loss (synthetic batch): {loss_val.item():.4f}")

# TODO: Compute the "perfect alignment" loss: chosen=0.0, rejected=-10.0
perfect_chosen = ____
perfect_rejected = ____
loss_perfect = dpo_loss(
    perfect_chosen, perfect_rejected, ref_chosen, ref_rejected, beta=0.1
)
print(f"DPO loss (perfect alignment): {loss_perfect.item():.4f}")

# TODO: Compute the "reversed preferences" loss by swapping policy_chosen
#       and policy_rejected in the call.
loss_reversed = ____
print(f"DPO loss (reversed preferences): {loss_reversed.item():.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert loss_val.item() > 0, "DPO loss should be positive"
assert loss_perfect.item() < loss_val.item(), "Perfect alignment -> lower loss"
assert loss_reversed.item() > loss_val.item(), "Reversed prefs -> higher loss"
print("✓ Checkpoint 2 passed — DPO loss behaviour verified\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — DPO loss vs policy preference margin
# ════════════════════════════════════════════════════════════════════════

# TODO: Call show_dpo_loss_curve_by_margin(beta=0.1) from the shared module.
____
# ── Checkpoint Visual ───────────────────────────────────────────────────
assert (OUTPUT_DIR / "ex3_dpo_loss_curve.png").exists()
print("✓ Visual checkpoint passed — loss-vs-margin curve saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Beta sensitivity sweep
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Beta sensitivity sweep")
print("=" * 70)

betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# TODO: For each beta, compute dpo_loss() on the synthetic batch and
#       append to beta_losses as a float.
beta_losses = []
for b in betas:
    ____

beta_df = pl.DataFrame({"beta": betas, "dpo_loss": beta_losses})
print(beta_df)

# TODO: Call show_beta_sensitivity(betas, beta_losses)
____

print(
    f"""
  Reading the sweep: the batch's log-ratio margin is fixed, and DPO scores
  it as beta * margin. At beta={betas[-1]} the loss is {beta_losses[-1]:.3f};
  at beta={betas[0]} it is {beta_losses[0]:.3f} (log 2 = 0.693 is "no preference").
  A HIGH beta rewards even a small departure from pi_ref, so the optimiser
  has little reason to move far. A LOW beta leaves the loss near log 2
  until the policy moves its log-ratios a long way from pi_ref.

  Interpretation of beta (the KL-penalty coefficient, pi* ~ pi_ref * exp(r/beta)):
    0.01 - 0.05 : Weak KL anchor. The policy can drift far from the SFT
                  reference to fit the preference data. Risk: over-fitting
                  the preferences (verbosity, over-refusal, lost skills).
    0.1         : Standard default. Balanced for most tasks.
    0.2 - 0.5   : Strong KL anchor. Stays close to the SFT reference;
                  preferences nudge rather than rewrite behaviour.
    >= 1.0      : Very strong anchor. The policy barely moves — the
                  preference signal may not change behaviour at all.
"""
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(beta_losses) == len(betas)
assert (OUTPUT_DIR / "ex3_beta_sensitivity.png").exists()
print("✓ Checkpoint 4 passed — beta sensitivity sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Pick beta for a MAS-regulated Singapore robo-advisor
# ════════════════════════════════════════════════════════════════════════
# DECISION: what beta should you pick for DPO? (beta is the KL anchor)
#   - Too low (0.02): weak anchor, the policy drifts far from the SFT
#     base to fit the refusal-heavy preference data -> it refuses benign
#     questions like "what is a stock?" -> user frustration, churn
#   - Too high (1.0): strong anchor, the policy barely moves from the
#     SFT base -> it still hedges on risk warnings inconsistently ->
#     compliance risk and possible regulatory action
#
# RULE OF THUMB for regulated finance: start near the default
# (beta=0.1-0.2), measure risk-disclosure compliance AND over-refusal on
# held-out eval sets, then tune: raise beta if benign questions start
# getting refused, lower it if disclosures are still inconsistent.
# All figures below are ILLUSTRATIVE planning numbers, not measurements.

print("=" * 70)
print("APPLICATION — Robo-advisor DPO beta selection")
print("=" * 70)

# Illustrative planning figures (not measurements)
AUM_SGD = 450_000_000  # Assets under management
EXPECTED_ENFORCEMENT_COST = 1_200_000  # Expected value of a single enforcement
AVG_ANNUAL_REVENUE_PER_USER = 180  # SGD per user per year
USERS_LOST_OVER_REFUSAL = 3500  # Churn from over-refusing benign queries

cost_low_beta = USERS_LOST_OVER_REFUSAL * AVG_ANNUAL_REVENUE_PER_USER
cost_high_beta = EXPECTED_ENFORCEMENT_COST
cost_balanced = 0.2 * cost_low_beta + 0.25 * cost_high_beta

# TODO: Build a 3-row polars DataFrame comparing beta=0.02, 0.20, 1.00
#       with columns: beta, approx_annual_cost_sgd, risk.
#       Low beta (weak KL anchor) carries the over-refusal churn cost;
#       high beta (policy barely moves) carries the enforcement cost.
cost_df = ____
print(cost_df)

print(
    f"\nRecommended: beta = 0.20  "
    f"(estimated annual cost ~ S${cost_balanced:,.0f} vs worst-case "
    f"S${max(cost_low_beta, cost_high_beta):,.0f})"
)

# ── Checkpoint Application ──────────────────────────────────────────────
assert cost_df.height == 3
assert cost_balanced < max(cost_low_beta, cost_high_beta)
print("✓ Application checkpoint passed — beta selection justified\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Derived DPO from RLHF + Bradley-Terry in one page
  [x] Implemented and verified the DPO loss with three sanity cases
      (typical, perfect alignment, reversed preferences)
  [x] Visualised the loss as a smooth function of preference margin
  [x] Swept beta from 0.01 -> 1.0 and interpreted the curve
  [x] Applied beta selection to a MAS-regulated robo-advisor — framed
      the trade-off as expected-cost in S$

  KEY INSIGHT: Beta is the KL-penalty coefficient. HIGH beta keeps the
  policy close to the SFT reference; LOW beta lets it drift far to fit
  the preferences (and over-fit them). For regulated deployments, start
  near 0.1-0.2, measure disclosure compliance and over-refusal, and
  raise beta if the policy drifts into refusing benign questions.

  Next: 03_dpo_training.py runs the full AlignmentPipeline on UltraFeedback
  and registers the resulting adapter.
"""
)
