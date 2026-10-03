# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 3.4: GRPO and LLM-as-Judge Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand GRPO (Group Relative Policy Optimization) and when it beats DPO
#   - Compute std-normalised group-relative advantages and the clipped GRPO objective
#   - Visualise GRPO advantages as a reward heatmap
#   - Run an LLM-as-judge evaluation with Kaizen Delegate
#   - Measure two known biases (position, verbosity) and mitigate position
#     bias with a swap-averaged judge
#   - Survey standard benchmarks (MMLU, HellaSwag, HumanEval, MT-Bench, etc.)
#   - Apply to a Singapore public hospital's clinical RAG assistant
#
# PREREQUISITES: 03_dpo_training.py (you trained a DPO adapter).
# ESTIMATED TIME: ~50 min
#
# TASKS:
#   1. GRPO theory: group-relative advantages + clipped objective with KL
#   2. Visualise GRPO advantages
#   3. LLM-as-judge: compare two responses with Kaizen Delegate
#   4. Position bias test (swap A/B) and a swap-averaged judge
#   5. Verbosity bias test (concise vs padded response)
#   6. Benchmarks survey (MMLU, HellaSwag, HumanEval, MT-Bench, etc.)
#   7. Apply: hospital clinical RAG assistant evaluation plan
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json

import polars as pl
import torch

# Delegate construction routes through shared.mlfp06._ollama_bootstrap.
from shared.mlfp06._ollama_bootstrap import (
    make_delegate,
    preflight_ollama,
    run_delegate_text,
)
from shared.mlfp06.ex_3 import (
    MODEL_NAME,
    OUTPUT_DIR,
    grpo_advantages,
    show_grpo_advantages,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — GRPO vs DPO in one page
# ════════════════════════════════════════════════════════════════════════
# GRPO (Group Relative Policy Optimization) was introduced by Shao et al.
# (2024) in DeepSeekMath and later used to train DeepSeek-R1 (2025).
#
#   For each prompt x, sample K completions from the current policy:
#     y_1, y_2, ..., y_K ~ pi_old(y|x)
#
#   Score each completion with a reward function r(x, y_i).
#
#   Advantage = reward standardised WITHIN the group (no value network):
#     A_i = (r_i - mean(r_1..r_K)) / (std(r_1..r_K) + eps)
#
#   Objective = PPO-style clipped surrogate + KL penalty to the reference:
#     rho_i  = pi_theta(y_i|x) / pi_old(y_i|x)
#     L_GRPO = -E[ (1/K) sum_i min(rho_i * A_i,
#                                  clip(rho_i, 1-eps_c, 1+eps_c) * A_i) ]
#              + beta_kl * KL(pi_theta || pi_ref)
#   (The real objective averages per token; this exercise works per
#    completion to keep the arithmetic visible.)
#
# DPO vs GRPO:
#   DPO:   pairwise preferences (chosen vs rejected)
#          Best when: preference pairs are available
#          Simpler:   closed-form loss, no sampling
#   GRPO:  group-relative scoring over K samples
#          Best when: a verifiable reward function exists
#                     (math correctness, code execution, unit tests)
#          Flexible:  any reward function, not just pairwise
#
# Why the std division matters: subtracting the mean alone keeps the
# reward's units — multiply every reward by 10 and every update is 10x
# larger. Dividing by the group std makes the advantages scale-free, so
# only the RELATIVE quality inside the group drives the update. A group
# where every completion scores the same carries no signal (A_i = 0).

print("=" * 70)
print("TASK 1: GRPO — Group Relative Policy Optimization")
print("=" * 70)

torch.manual_seed(42)
K = 5
n_prompts = 8
rewards = torch.randn(n_prompts, K)
advantages = grpo_advantages(rewards)
advantages_scaled = grpo_advantages(rewards * 10.0)

print(f"  Prompts: {n_prompts}, Completions per prompt: K={K}")
print(f"  Rewards (prompt 0): {[round(r, 3) for r in rewards[0].tolist()]}")
print(f"  Group mean / std:   {rewards[0].mean().item():.4f} / {rewards[0].std().item():.4f}")
print(f"  Advantages (prompt 0): {[round(a, 3) for a in advantages[0].tolist()]}")
print(f"  Advantage sum per group ~ 0: {advantages.sum(dim=1).mean().item():.6f}")
print(f"  Advantage std per group ~ 1: {advantages.std(dim=1).mean().item():.4f}")
scale_gap = (advantages - advantages_scaled).abs().max().item()
print(f"  Max change when rewards are x10: {scale_gap:.2e}  (scale-free)")


def grpo_objective(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-completion GRPO loss: clipped surrogate + KL penalty to pi_ref.

    All inputs are [n_prompts, K]. Returns (loss, mean_kl). The KL uses the
    unbiased estimator from DeepSeekMath: pi_ref/pi - log(pi_ref/pi) - 1.
    """
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped).mean()
    log_ref_ratio = logp_ref - logp_new
    kl = (torch.exp(log_ref_ratio) - log_ref_ratio - 1.0).mean()
    return -surrogate + kl_beta * kl, kl


# Sanity case: before any update, pi_theta == pi_old == pi_ref.
logp_old = torch.randn(n_prompts, K) - 5.0
loss_start, kl_start = grpo_objective(logp_old, logp_old, logp_old, advantages)
# After a step that raises the log-prob of high-advantage completions:
logp_new = logp_old + 0.5 * advantages
loss_step, kl_step = grpo_objective(logp_new, logp_old, logp_old, advantages)
# A huge step is clipped — the surrogate cannot keep improving.
logp_big = logp_old + 5.0 * advantages
loss_big, kl_big = grpo_objective(logp_big, logp_old, logp_old, advantages)
print(f"\n  GRPO loss at start (ratio=1, KL=0): {loss_start.item():+.4f}  KL={kl_start.item():.4f}")
print(f"  GRPO loss after a small step:       {loss_step.item():+.4f}  KL={kl_step.item():.4f}")
print(f"  GRPO loss after a huge step:        {loss_big.item():+.4f}  KL={kl_big.item():.4f}")
print(
    "  A small step towards high-advantage completions lowers the loss. A huge\n"
    "  step gains nothing from the clipped surrogate and pays a large KL\n"
    "  penalty — which is exactly what keeps GRPO updates conservative."
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert advantages.shape == rewards.shape
assert abs(advantages.sum(dim=1).mean().item()) < 1e-5
assert abs(advantages.std(dim=1).mean().item() - 1.0) < 1e-3, "advantages must be std-normalised"
assert scale_gap < 1e-4, "GRPO advantages must not depend on the reward scale"
assert abs(loss_start.item()) < 1e-5 and kl_start.item() < 1e-6
assert loss_step.item() < loss_start.item(), "a step towards high-advantage completions lowers the loss"
assert loss_big.item() > loss_step.item(), "clip + KL must penalise an oversized step"
print("✓ Checkpoint 1 passed — GRPO advantages and objective verified\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Reward and advantage heatmaps
# ════════════════════════════════════════════════════════════════════════

show_grpo_advantages(rewards, advantages)
assert (OUTPUT_DIR / "ex3_grpo_advantages.png").exists()
print("✓ Visual checkpoint passed — GRPO advantage heatmap saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — LLM-as-judge evaluation
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: LLM-as-judge — compare two responses with Kaizen Delegate")
print("=" * 70)

# Fail loudly and early if the local LLM is not available. There is no
# offline fallback: a judge that cannot run must not produce verdicts.
preflight_ollama(required_models=[MODEL_NAME])


class JudgeParseError(ValueError):
    """The judge replied, but not with a usable JSON verdict."""


async def llm_judge(prompt: str, response_a: str, response_b: str) -> dict:
    """Ask an LLM to pick between two responses. Returns the parsed verdict.

    Raises JudgeParseError when the reply is not a valid verdict — the
    caller decides how to count it. It is NEVER silently turned into a tie.
    """
    # make_delegate() reads the model from OLLAMA_CHAT_MODEL and backs the
    # call with the local Ollama daemon (no API keys, no OpenAI fallback).
    delegate = make_delegate(temperature=0.0)
    judge_prompt = f"""You are an impartial judge evaluating two responses to a user query.

Query: {prompt[:500]}

Response A:
{response_a[:500]}

Response B:
{response_b[:500]}

Evaluate on: helpfulness, accuracy, clarity, safety.
Output ONLY a JSON object:
{{"winner": "A" or "B" or "tie", "score_a": 1-10, "score_b": 1-10, "reasoning": "..."}}"""

    response, *_ = await run_delegate_text(delegate, judge_prompt)

    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        verdict = json.loads(response[start:end])
        verdict["score_a"] = float(verdict["score_a"])
        verdict["score_b"] = float(verdict["score_b"])
    except (ValueError, KeyError, TypeError) as exc:
        raise JudgeParseError(f"unparseable verdict: {response[:200]!r}") from exc
    if verdict.get("winner") not in ("A", "B", "tie"):
        raise JudgeParseError(f"invalid winner {verdict.get('winner')!r}")
    return verdict


async def swap_averaged_judge(prompt: str, x: str, y: str) -> dict:
    """Mitigate position bias: judge (x, y) AND (y, x), average each answer's score.

    Each response is scored once in slot A and once in slot B, so a judge
    that favours one slot adds the same bonus to both — it cancels out.
    """
    xy = await llm_judge(prompt, x, y)
    yx = await llm_judge(prompt, y, x)
    score_x = (xy["score_a"] + yx["score_b"]) / 2
    score_y = (xy["score_b"] + yx["score_a"]) / 2
    winner = "x" if score_x > score_y else "y" if score_y > score_x else "tie"
    return {"winner": winner, "score_x": score_x, "score_y": score_y}


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Position bias test (swap A/B) and its mitigation
# ════════════════════════════════════════════════════════════════════════

JUDGE_PAIRS = [
    {
        "prompt": "What is the capital of Singapore?",
        "good": "Singapore is a city-state; the entire country is the capital.",
        "bad": "Probably somewhere in Asia. I don't remember exactly.",
    },
    {
        "prompt": "How does public key cryptography work?",
        "good": "Each party has a public key (shared) and a private key (secret). "
        "Messages encrypted with the public key can only be decrypted by "
        "the matching private key.",
        "bad": "It uses two keys somehow. One is public.",
    },
    {
        "prompt": "Summarise photosynthesis in one sentence.",
        "good": "Plants convert sunlight, water, and CO2 into glucose and oxygen "
        "using chlorophyll in their leaves.",
        "bad": "Plants eat sunlight.",
    },
]


async def measure_position_bias() -> dict:
    """Judge each pair in both orders.

    A pair is CONSISTENT only when the judge picks the good answer in BOTH
    orders. Unparseable verdicts are counted as failures, not as agreement.
    """
    print("\n  --- Position Bias Test ---")
    consistent = 0
    judged = 0
    failures = 0
    for i, p in enumerate(JUDGE_PAIRS, 1):
        try:
            ab = await llm_judge(p["prompt"], p["good"], p["bad"])
            ba = await llm_judge(p["prompt"], p["bad"], p["good"])
        except JudgeParseError as exc:
            failures += 1
            print(f"    Pair {i}: JUDGE FAILURE — {exc}")
            continue
        judged += 1
        consistent_this = ab["winner"] == "A" and ba["winner"] == "B"
        consistent += int(consistent_this)
        tag = "consistent" if consistent_this else "POSITION BIAS / WRONG PICK"
        print(f"    Pair {i}: AB={ab['winner']}, BA={ba['winner']}  [{tag}]")
    if judged == 0:
        raise RuntimeError(
            f"The judge produced no usable verdicts ({failures} failures). "
            "Check the model in OLLAMA_CHAT_MODEL follows JSON instructions."
        )
    rate = consistent / judged
    print(f"\n  Position consistency: {consistent}/{judged} judged pairs ({rate:.0%})")
    print(f"  Judge failures:       {failures}/{len(JUDGE_PAIRS)}")
    print(f"  Bias: {'LOW' if rate > 0.7 else 'HIGH'}")
    return {"rate": rate, "judged": judged, "failures": failures}


async def measure_swap_averaged_accuracy() -> dict:
    """How often does the swap-averaged judge pick the good answer?"""
    print("\n  --- Swap-averaged judge (mitigation) ---")
    correct = 0
    judged = 0
    failures = 0
    for i, p in enumerate(JUDGE_PAIRS, 1):
        try:
            v = await swap_averaged_judge(p["prompt"], p["good"], p["bad"])
        except JudgeParseError as exc:
            failures += 1
            print(f"    Pair {i}: JUDGE FAILURE — {exc}")
            continue
        judged += 1
        correct += int(v["winner"] == "x")
        print(
            f"    Pair {i}: good={v['score_x']:.1f}  bad={v['score_y']:.1f}  "
            f"-> {'good' if v['winner'] == 'x' else v['winner']}"
        )
    if judged == 0:
        raise RuntimeError(f"Swap-averaged judge produced no usable verdicts ({failures} failures).")
    accuracy = correct / judged
    print(f"  Swap-averaged accuracy: {correct}/{judged} ({accuracy:.0%}), failures: {failures}")
    return {"accuracy": accuracy, "judged": judged, "failures": failures}


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Verbosity bias test
# ════════════════════════════════════════════════════════════════════════


async def measure_verbosity_bias() -> dict:
    """Check if the judge prefers padded responses over concise-but-correct ones.

    Uses the swap-averaged judge so position bias cannot masquerade as
    verbosity bias.
    """
    print("\n  --- Verbosity Bias Test ---")
    test_prompt = "What is machine learning?"
    concise = (
        "Machine learning is a subset of AI where algorithms learn patterns from "
        "data to make predictions without explicit programming."
    )
    verbose = (
        "Machine learning is a very interesting and important field of study that "
        "has been gaining a lot of attention in recent years. It is essentially a "
        "subset of artificial intelligence. The basic idea is that instead of "
        "explicitly programming every rule, we let the computer learn from data. "
    ) * 3

    verdict = await swap_averaged_judge(test_prompt, concise, verbose)
    print(f"    Concise ({len(concise)} chars) avg score: {verdict['score_x']:.1f}")
    print(f"    Verbose ({len(verbose)} chars) avg score: {verdict['score_y']:.1f}")
    print(f"    Winner: {'concise' if verdict['winner'] == 'x' else 'verbose' if verdict['winner'] == 'y' else 'tie'}")
    print(f"    Bias: {'VERBOSITY BIAS' if verdict['winner'] == 'y' else 'OK'}")
    return verdict


position_result = asyncio.run(measure_position_bias())
position_consistency = position_result["rate"]
swap_result = asyncio.run(measure_swap_averaged_accuracy())
verbosity_verdict = asyncio.run(measure_verbosity_bias())

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert 0.0 <= position_consistency <= 1.0
assert position_result["judged"] + position_result["failures"] == len(JUDGE_PAIRS)
assert 0.0 <= swap_result["accuracy"] <= 1.0
assert verbosity_verdict["winner"] in ("x", "y", "tie")
print("\n✓ Checkpoint 5 passed — LLM-as-judge bias measurements complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Benchmarks survey
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Evaluation Benchmarks Survey")
print("=" * 70)

benchmarks = pl.DataFrame(
    {
        "Benchmark": [
            "MMLU",
            "HellaSwag",
            "HumanEval",
            "MT-Bench",
            "TruthfulQA",
            "GSM8K",
            "MBPP",
            "ARC-Challenge",
        ],
        "Domain": [
            "Multi-task knowledge",
            "Commonsense reasoning",
            "Code generation",
            "Multi-turn conversation",
            "Truthfulness",
            "Grade-school math",
            "Code generation",
            "Science reasoning",
        ],
        "Format": [
            "MCQ (57 subjects)",
            "4-way completion",
            "Code + unit tests",
            "Judge scoring (1-10)",
            "MCQ + generation",
            "Chain-of-thought",
            "Code + test cases",
            "MCQ (science)",
        ],
        "Measures": [
            "Breadth of knowledge",
            "Common sense",
            "Coding ability",
            "Conversation quality",
            "Factual accuracy",
            "Math reasoning",
            "Practical coding",
            "Scientific reasoning",
        ],
    }
)
print(benchmarks)

print(
    """
  lm-eval-harness (EleutherAI):
    Unified evaluation framework for the static benchmarks above
    (MMLU, HellaSwag, HumanEval, TruthfulQA, GSM8K, MBPP, ARC).
    MT-Bench is NOT an lm-eval task: it is a multi-turn set scored by an
    LLM judge — the same technique (and the same biases) as Task 3.
    Install: pip install lm-eval
    Usage:   lm_eval --model hf --model_args pretrained=MODEL \\
                     --tasks mmlu,hellaswag,gsm8k
    Reports: each task's own metric (acc, acc_norm, exact_match, pass@1).

  Pre- vs post-alignment expectation:
    helpfulness & safety should improve after DPO.
    raw knowledge (MMLU) should NOT drop significantly.
    if MMLU drops > 3pp, the policy drifted too far from the reference:
    RAISE beta (stronger KL anchor) or train for fewer steps.
"""
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert benchmarks.height >= 8
print("✓ Checkpoint 6 passed — benchmarks survey loaded\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Hospital clinical RAG assistant evaluation plan
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO (illustrative): a Singapore public hospital is
# evaluating a DPO-aligned LLM for a clinical RAG assistant used by
# junior doctors. You must define the evaluation plan.
#
# EVALUATION MATRIX:
#   1. Safety    -> swap-averaged LLM-as-judge + verbosity check
#                   (this exercise) on a held-out clinical adversarial set
#   2. Knowledge -> MMLU (medicine subset), a clinical QA benchmark
#   3. Reasoning -> GSM8K for diagnostic math, custom clinical CoT eval
#   4. Honesty   -> TruthfulQA + "I don't know" rate on unanswerable queries
#
# APPROVAL GATE (the hospital's clinical AI governance board, informed
# by IMDA's AI Verify testing framework and HSA guidance on software
# medical devices) — thresholds below are illustrative:
#   - Safety refusal >= 85% on clinical adversarial set
#   - MMLU medicine >= baseline - 2pp (no knowledge degradation)
#   - TruthfulQA >= 80% (no hallucination on unanswerable)
#   - Position-swap consistency >= 75% (judge itself is reliable)

print("=" * 70)
print("APPLICATION — Hospital clinical RAG assistant evaluation plan")
print("=" * 70)

eval_plan = pl.DataFrame(
    {
        "Dimension": ["Safety", "Knowledge", "Reasoning", "Honesty", "Judge Quality"],
        "Method": [
            "Swap-averaged LLM-as-judge",
            "MMLU (medicine subset)",
            "GSM8K + clinical CoT",
            "TruthfulQA + IDK rate",
            "Position-swap consistency",
        ],
        "Approval Gate": [
            ">= 85% refusal on clinical adversarial",
            ">= baseline - 2pp on MMLU medicine",
            ">= 70% on clinical CoT",
            ">= 80% on TruthfulQA",
            ">= 75% position consistency",
        ],
        "Current": [
            "measured in Ex 3.3 (aligned refusal)",
            "run lm-eval-harness",
            "run lm-eval-harness",
            "run lm-eval-harness",
            f"{position_consistency:.0%} consistent, "
            f"{position_result['failures']} judge failures (this session)",
        ],
    }
)
print(eval_plan)

# Illustrative planning figures (not measurements)
CLINICAL_RAG_USERS = 420  # junior doctors at the hospital
HOURS_SAVED_PER_USER_PER_WEEK = 3.5
DOCTOR_HOURLY_COST_SGD = 90
annual_hours_saved = CLINICAL_RAG_USERS * HOURS_SAVED_PER_USER_PER_WEEK * 52
annual_value_sgd = annual_hours_saved * DOCTOR_HOURLY_COST_SGD

print(f"\n  Target users (junior doctors):  {CLINICAL_RAG_USERS:,}")
print(f"  Hours saved/user/week:          {HOURS_SAVED_PER_USER_PER_WEEK}")
print(f"  Doctor hourly cost:             S${DOCTOR_HOURLY_COST_SGD}")
print(f"  Annual hours saved:             {annual_hours_saved:,.0f}")
print(f"  Annual value of time saved:     S${annual_value_sgd:,.0f}")
print(
    "\n  Value only materialises if ALL approval gates above are met —\n"
    "  otherwise the hospital deploys rule-based triage and the investment is lost."
)

# ── Checkpoint Application ──────────────────────────────────────────────
assert eval_plan.height == 5
print("\n✓ Application checkpoint passed — hospital evaluation plan ready\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] GRPO intuition: std-normalised group-relative advantages (scale-free)
  [x] Computed the clipped GRPO objective with its KL penalty to pi_ref
  [x] Visualised reward and advantage heatmaps
  [x] LLM-as-judge with Kaizen Delegate — structured JSON verdict, with
      parse failures counted, never turned into fake ties
  [x] Measured position bias via A/B swap and mitigated it with a
      swap-averaged judge
  [x] Measured verbosity bias with concise-vs-padded test
  [x] Surveyed the standard LLM benchmarks (MMLU, HellaSwag, HumanEval,
      MT-Bench, TruthfulQA, GSM8K, MBPP, ARC)
  [x] Drafted a hospital clinical RAG evaluation plan with approval gates
      and quantified business value

  KEY INSIGHT: Evaluation IS the product. An LLM without an evaluation
  plan is a prototype, not a system. DPO trains the behaviour; evaluation
  proves the behaviour. For regulated deployments (healthcare, finance),
  the evaluation plan is what gets signed off, not the model weights.

  NEXT: Exercise 4 (RAG) grounds LLM responses in retrieved documents.
  Instead of relying on training data alone, RAG retrieves relevant
  text at inference time — enabling up-to-date, verifiable answers.
"""
)
