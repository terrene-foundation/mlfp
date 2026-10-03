# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.4: Model Merging Without Training
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - THEORY: why merging fine-tuned models works (task vectors)
#   - BUILD: TIES (Trim, Elect Sign, Merge) on synthetic deltas
#   - BUILD: SLERP (spherical linear interpolation) on weight tensors
#   - VISUALISE: weight-norm preservation across merge strategies
#   - APPLY: Singapore fintech — three LoRAs into one deployment
#
# PREREQUISITES: Exercises 2.1 and 2.2 (LoRA + adapters)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. THEORY: task vectors and the merge landscape
#   2. BUILD: TIES merge (trim -> elect sign -> average)
#   3. BUILD: SLERP merge (hypersphere interpolation)
#   4. VISUALISE: norm comparison across merge strategies
#   5. APPLY: Singapore fintech KYC + fraud + explainer merge
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import torch
from dotenv import load_dotenv

from shared.mlfp06.ex_2 import OUTPUT_DIR

load_dotenv()

torch.manual_seed(42)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Task Vectors and the Merge Landscape
# ════════════════════════════════════════════════════════════════════════
# A task vector is the *difference* between a fine-tuned model and its
# base:  tau = W_finetuned - W_base.  It captures everything the
# fine-tuning added, and nothing else.
#
# Task arithmetic (Ilharco et al., 2023):
#   - Add:      W_merged = W_base + alpha * tau_A + beta * tau_B
#   - Subtract: W_merged = W_base - alpha * tau_A    (remove a skill)
#   - Negate:   reverses the adaptation entirely
#
# But raw addition of task vectors causes *sign conflict*: two fine-tunes
# may update the same parameter in opposite directions, and averaging
# cancels both out into noise.  TIES and DARE solve this.
#
# TIES (Yadav et al., 2023), three steps per parameter:
#   TRIM  — keep only each task vector's top-k% largest-magnitude entries
#           (k = "density", 20% in the paper); the rest is treated as noise
#   ELECT — the merged sign is the sign of the SUMMED trimmed deltas, so
#           the direction with more total mass wins a conflict (it is not
#           zeroed out)
#   MERGE — average only the non-zero deltas that agree with that sign
#           (a "disjoint mean": tasks that were trimmed or disagree there
#           do not dilute the average)
#
# DARE (Yu et al., 2023): randomly DROP a fraction of delta parameters
# then RESCALE the survivors by 1/(1-drop).  Like dropout for merging.
#
# SLERP (Spherical Linear Interpolation): walks the shortest path on the
# hypersphere between two weight tensors rather than the straight chord.
# Preserves weight norms, which matters because transformer weights are
# calibrated to a specific norm at training time.
#
# Crucially: merging is FREE — zero extra training compute.  This makes
# it the cheapest way to combine capabilities from multiple fine-tunes.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — BUILD: TIES merge on synthetic 128x128 deltas
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: TIES merge — trim / elect sign / average")
print("=" * 70)

delta_A = torch.randn(128, 128) * 0.1  # Task A fine-tuned delta
delta_B = torch.randn(128, 128) * 0.1  # Task B fine-tuned delta

TRIM_DENSITY = 0.20  # keep the top 20% of each task vector by magnitude


def trim_top_k(delta: torch.Tensor, density: float) -> torch.Tensor:
    """Step 1 — TRIM: keep the top `density` fraction of |delta|, zero the rest."""
    k = max(1, int(delta.numel() * density))
    threshold = delta.abs().flatten().topk(k).values.min()
    return torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))


delta_A_trim = trim_top_k(delta_A, TRIM_DENSITY)
delta_B_trim = trim_top_k(delta_B, TRIM_DENSITY)

# Step 2 — ELECT SIGN: sign of the summed (mass-weighted) trimmed deltas.
# On a conflict the larger-magnitude update wins instead of both cancelling.
sign_A = delta_A_trim.sign()
sign_B = delta_B_trim.sign()
elected_sign = (delta_A_trim + delta_B_trim).sign()

# Step 3 — DISJOINT MERGE: average only the non-zero deltas whose sign
# agrees with the elected sign.
mask_A = ((sign_A == elected_sign) & (delta_A_trim != 0)).float()
mask_B = ((sign_B == elected_sign) & (delta_B_trim != 0)).float()
merged_delta = (delta_A_trim * mask_A + delta_B_trim * mask_B) / (
    mask_A + mask_B
).clamp(min=1.0)

both_kept = (delta_A_trim != 0) & (delta_B_trim != 0)
conflicts = both_kept & (sign_A != sign_B)
print(f"Original non-zero params (delta_A): {(delta_A != 0).sum().item():,}")
print(f"After TRIM (top {TRIM_DENSITY:.0%}):               {(delta_A_trim != 0).sum().item():,}")
print(f"Entries kept by BOTH tasks:         {both_kept.sum().item():,}")
print(f"  ...of which sign conflicts:       {conflicts.sum().item():,}")
print(f"Merged delta Frobenius norm:        {merged_delta.norm():.4f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert merged_delta.shape == delta_A.shape, "Task 1: TIES should preserve shape"
assert (delta_A_trim != 0).sum() < (delta_A != 0).sum(), "TRIM should zero some values"
print("✓ Checkpoint 1 passed — TIES merge complete\n")

# INTERPRETATION: TIES prevents sign cancellation during merging.
# With plain averaging, two task vectors that disagree on a parameter
# cancel each other there. TIES keeps the update with more mass for each
# conflicting entry and averages only agreeing, non-trimmed values.
# (These are random synthetic deltas, so roughly half of the overlapping
# entries conflict; real task vectors from one base usually agree more.)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: SLERP merge on weight tensors
# ════════════════════════════════════════════════════════════════════════


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors.

    t=0 -> v0; t=1 -> v1; t=0.5 -> midpoint on the hypersphere.
    Falls back to linear interpolation when the vectors are nearly
    parallel (theta < 1e-6) to avoid divide-by-zero.
    """
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()
    cos_theta = torch.dot(v0_flat, v1_flat) / (v0_flat.norm() * v1_flat.norm() + 1e-8)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta.abs() < 1e-6:
        return (1 - t) * v0 + t * v1
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    return w0 * v0 + w1 * v1


print("=" * 70)
print("TASK 2: SLERP vs linear interpolation — weight norm comparison")
print("=" * 70)

W_base = torch.randn(128, 128)
W_task_A = W_base + delta_A
W_task_B = W_base + delta_B

slerp_merged = slerp(0.5, W_task_A, W_task_B)
linear_merged = 0.5 * W_task_A + 0.5 * W_task_B

norms = {
    "W_task_A": W_task_A.norm().item(),
    "W_task_B": W_task_B.norm().item(),
    "SLERP midpoint": slerp_merged.norm().item(),
    "Linear midpoint": linear_merged.norm().item(),
}
for k, v in norms.items():
    print(f"  {k:<20} norm = {v:.4f}")

mean_input_norm = 0.5 * (norms["W_task_A"] + norms["W_task_B"])
slerp_gap = abs(norms["SLERP midpoint"] - mean_input_norm)
linear_gap = abs(norms["Linear midpoint"] - mean_input_norm)

print(f"\n  SLERP deviation from mean input norm:  {slerp_gap:.4f}")
print(f"  Linear deviation from mean input norm: {linear_gap:.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert slerp_merged.shape == W_task_A.shape, "Task 2: SLERP should preserve shape"
assert slerp_gap <= linear_gap + 1e-6, "SLERP should preserve norms at least as well"
print("✓ Checkpoint 2 passed — SLERP preserves norms\n")

# INTERPRETATION: Linear merging shrinks toward the origin whenever the
# two inputs are not perfectly aligned.  SLERP walks the arc on the
# hypersphere and keeps the interpolated norm close to the inputs.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — VISUALISE: norm comparison across merge strategies
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Visualise merge-strategy norm preservation")
print("=" * 70)

# Sweep the interpolation parameter t from 0 -> 1 and record norms
ts = [i / 10 for i in range(11)]
slerp_norms = [slerp(t, W_task_A, W_task_B).norm().item() for t in ts]
linear_norms = [((1 - t) * W_task_A + t * W_task_B).norm().item() for t in ts]

fname = OUTPUT_DIR / "ex2_slerp_vs_linear.png"
fname.unlink(missing_ok=True)  # the checkpoint must see THIS run's plot

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(ts, slerp_norms, "o-", color="steelblue", linewidth=2, label="SLERP")
ax.plot(ts, linear_norms, "s-", color="darkorange", linewidth=2, label="Linear")
ax.axhline(
    norms["W_task_A"], color="gray", linestyle=":", alpha=0.6, label="input norms"
)
ax.axhline(norms["W_task_B"], color="gray", linestyle=":", alpha=0.6)
ax.set_xlabel("Interpolation parameter t (0 = task A, 1 = task B)")
ax.set_ylabel("Frobenius norm of merged weights")
ax.set_title("SLERP vs Linear — weight norm preservation", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert fname.exists(), "Task 3: plot should exist"
print("✓ Checkpoint 3 passed — norm comparison visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — APPLY: Singapore fintech — merging three LoRAs into one
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): A Singapore digital bank has trained three independent LoRA
# adapters on top of the same 7B base:
#   tau_kyc     — KYC document extraction (NRIC, addresses, phone formats)
#   tau_fraud   — Transaction-narrative fraud flagging
#   tau_explain — Plain-English explanations for rejected applications
#
# Each adapter was trained by a different team over several weeks.  The
# platform team wants to ship a SINGLE deployed model that handles all
# three tasks instead of three separate inference pipelines.
#
# OPTIONS:
#   A. Three deployments (status quo): 3 * 14 GB GPU VRAM per region.
#      High latency for multi-step journeys (KYC -> fraud -> explain).
#   B. Train a multi-task adapter: ~2 weeks of additional GPU time to
#      re-train on pooled data, new evals needed for each capability.
#   C. TIES merge the three LoRAs into one: zero training, ~2 hours of
#      engineering to run the merge + eval.  Some quality drop on each
#      task but typically <2 percentage points.
#
# DECISION: TIES merge (option C).  The $/capability trade-off favours
# free merging unless the merged model drops >5 points on any eval.
#
# BUSINESS IMPACT (illustrative figures):
#   - Infra saving: drop from 3 * 14 GB to 1 * 14 GB per region (28 GB
#     freed in each of 3 regions).  At ~S$1.20/GB-month on managed GPU,
#     that is ~S$33/region/month * 3 = S$99/month of inference VRAM.
#   - Latency: end-to-end KYC -> fraud -> explain journey drops from
#     ~780 ms (three hops) to ~320 ms (single inference).  On the
#     bank's 180,000 onboarding journeys/month, that saves ~23 agent-
#     hours of wait-time monitoring, or ~S$4,100/month in operations
#     cost.
#   - Avoided retraining: option B would have consumed ~S$18,000 of
#     GPU time plus ~6 weeks of engineering effort.
#
# RISK: merged model degrades on one of the tasks (most likely fraud,
# which is the most narrowly-scoped).  Mitigation: re-run the three
# task evaluation suites and fall back to option A if any eval drops
# below the SLA threshold.

print("Singapore fintech — three-LoRA merge decision:")
rows = [
    ("Option A: three deployments", 3 * 14, 780, 0),
    ("Option B: retrain multi-task", 1 * 14, 320, 18_000),
    ("Option C: TIES merge", 1 * 14, 320, 120),
]
decision_df = pl.DataFrame(
    {
        "Option": [r[0] for r in rows],
        "VRAM_GB": [r[1] for r in rows],
        "Latency_ms": [r[2] for r in rows],
        "Upfront_cost_SGD": [r[3] for r in rows],
    }
)
print(decision_df)
monthly_saving_sgd = 99 + 4_100
annual_saving_sgd = monthly_saving_sgd * 12
print(f"\n  Monthly saving (option C vs A): S${monthly_saving_sgd:,}")
print(f"  Annual saving:                  S${annual_saving_sgd:,}")
print(f"  Recommended: TIES merge (option C) with per-task eval gate")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert annual_saving_sgd > 0, "Task 4: fintech should see positive savings"
print("✓ Checkpoint 4 passed — fintech merge decision analysed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built TIES merge: trim -> elect sign -> merge
  [x] Built SLERP merge: spherical interpolation on the hypersphere
  [x] Visualised that SLERP preserves weight norms better than linear
  [x] Applied merging to a Singapore fintech three-LoRA scenario
      (~S$50k/year saving vs separate deployments, zero retraining)

  KEY INSIGHT: merging is free compute.  TIES handles sign conflict,
  SLERP handles norm drift, and task arithmetic composes them for
  targeted capability addition or removal.

  Next: 05_quantisation.py surveys how we shrink the merged model for
  deployment on smaller hardware.
"""
)
