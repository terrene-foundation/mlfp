# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 10 — AI-Resilient Assessment Questions

Alignment, RL & Governance
Covers: LoRA, QLoRA, SFT, DPO, GRPO, PPO, SAC, model merging,
        PACT GovernanceEngine, PactGovernedAgent, operating envelopes,
        D/T/R, AlignmentPipeline, AdapterRegistry, RLTrainer
"""

QUIZ = {
    "module": "ASCENT10",
    "title": "Alignment, RL & Governance",
    "questions": [
        # ── Section A: LoRA / SFT ───────────────────────────────────────
        {
            "id": "10.A.1",
            "lesson": "10.A",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 fine-tunes with LoRA (lora_r=16, target_modules=['q_proj', 'v_proj']). "
                "For a 7B model where q_proj has shape 4096×4096, calculate the total trainable "
                "parameters for q_proj and v_proj combined, and the reduction ratio vs full fine-tuning."
            ),
            "options": [
                "A) 131,072 trainable params per module × 2 modules = 262,144 per layer. Across 32 layers: 262K × 32 = 8.4M trainable. Full fine-tuning of q_proj+v_proj: 2 × 16.7M × 32 = 1.07B. Reduction for these layers: 128×. As a fraction of the full 7B model: 8.4M / 7B = 0.12% of total parameters are trainable.",
                "B) 65,536 trainable params total — LoRA only adds one matrix, not two",
                "C) 8,388,608 trainable params — LoRA trains 50% of the original parameters",
                "D) 16,384 trainable params — lora_r=16 means 16 parameters per module",
            ],
            "answer": "A",
            "explanation": (
                "Per target module: A matrix (4096 × 16) = 65,536 params + B matrix (16 × 4096) = 65,536. "
                "Total per module: 131,072. Two modules (q_proj + v_proj): 262,144 per layer. "
                "Across 32 transformer layers: 262,144 × 32 = 8,388,608 trainable parameters. "
                "Full fine-tuning: 2 × 4096 × 4096 × 32 = 1.07B parameters for q_proj + v_proj. "
                "Reduction: 1.07B / 8.39M ≈ 128×. "
                "Total model: 7B parameters, only 8.4M trainable = 0.12%. "
                "Optimizer memory: only 8.4M params need Adam states (m, v), saving ~95% GPU memory."
            ),
            "learning_outcome": "Calculate LoRA parameter reduction for a specific model architecture",
        },
        {
            "id": "10.A.2",
            "lesson": "10.A",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 1's AlignmentConfig crashes with OOM on a 16GB GPU. "
                "The config has batch_size=16 and max_seq_length=2048. What change fixes it?"
            ),
            "code": (
                "config = AlignmentConfig(\n"
                "    method='sft',\n"
                "    base_model=os.environ['SFT_BASE_MODEL'],\n"
                "    lora_r=16,\n"
                "    lora_alpha=32,\n"
                "    batch_size=16,          # Too large\n"
                "    max_seq_length=2048,\n"
                "    gradient_checkpointing=False,  # Not enabled\n"
                ")\n"
            ),
            "options": [
                "A) Reduce lora_r from 16 to 4 — this is the main memory consumer",
                "B) Reduce batch_size to 2-4 and enable gradient_checkpointing=True. Activation memory scales as batch_size × seq_length × hidden_dim. At batch=16 and seq=2048, activations alone can consume 12+ GB. Gradient checkpointing recomputes activations during backward pass instead of storing them, trading ~30% compute for ~50% memory savings.",
                "C) Reduce max_seq_length to 128 — shorter sequences always fit in memory",
                "D) Switch from SFT to DPO — DPO uses less memory",
            ],
            "answer": "B",
            "explanation": (
                "Memory breakdown for SFT on a 7B model with batch=16, seq=2048: "
                "Model weights (fp16): 14 GB. LoRA weights: ~16 MB (negligible). "
                "Activations: batch × seq × layers × hidden × 2 bytes ≈ 16 × 2048 × 32 × 4096 × 2 ≈ 8.6 GB. "
                "Optimizer states: ~32 MB (only LoRA params). Total: ~23 GB → OOM on 16GB. "
                "Fix 1: batch_size=4 → activations ≈ 2.1 GB → total ~16.5 GB (tight fit). "
                "Fix 2: gradient_checkpointing=True → activations ≈ 1 GB → total ~15 GB (fits). "
                "Both together: comfortably fits with headroom for variable-length sequences."
            ),
            "learning_outcome": "Debug OOM errors in AlignmentConfig by adjusting batch size and gradient checkpointing",
        },
        # ── Section B: DPO ──────────────────────────────────────────────
        {
            "id": "10.B.1",
            "lesson": "10.B",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 2 aligns with DPO using beta=0.1. You have 8,000 preference pairs "
                "and want to align a credit explanation model to be less technical. "
                "Should you use DPO or SFT + RLHF (PPO)?"
            ),
            "options": [
                "A) RLHF always produces better alignment than DPO",
                "B) DPO. With 8,000 preference pairs, DPO is more practical: it directly optimizes the preference objective without needing a separate reward model or PPO training loop. RLHF requires: (1) train a reward model, (2) run PPO against it — two unstable training stages. DPO collapses both into a single stable objective: L_DPO = -log σ(β(log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))).",
                "C) Neither — 8,000 pairs is too few for any alignment method",
                "D) SFT on the chosen responses only — preference pairs are unnecessary",
            ],
            "answer": "B",
            "explanation": (
                "DPO eliminates the reward model training step by showing that the optimal "
                "RLHF policy can be directly parameterized from preferences. "
                "Practical advantages: (1) One training run vs two (reward model + PPO). "
                "(2) No reward model architecture decisions. (3) More stable — PPO is notoriously "
                "sensitive to hyperparameters (clip range, KL coefficient, reward scaling). "
                "(4) 8,000 pairs is sufficient for DPO but marginal for training a good reward model. "
                "The beta parameter controls divergence from the reference policy: "
                "beta=0.1 allows moderate deviation. Higher beta → closer to reference."
            ),
            "learning_outcome": "Choose DPO over RLHF for stable preference alignment with limited data",
        },
        {
            "id": "10.B.2",
            "lesson": "10.B",
            "type": "output_interpretation",
            "difficulty": "advanced",
            "question": (
                "Exercise 2 evaluates DPO-aligned vs base model on safety prompts. "
                "The DPO model scores 4.2/5 on helpfulness (base: 4.5) but 4.8/5 on safety "
                "(base: 2.1). Is this trade-off acceptable?"
            ),
            "options": [
                "A) No — any decrease in helpfulness means the alignment failed",
                "B) Yes — this is the expected alignment tax. DPO teaches the model to prefer safe responses over maximally helpful ones. A 0.3-point helpfulness decrease for a 2.7-point safety increase is an excellent trade-off. The alignment tax (small helpfulness reduction for large safety gain) is inherent to all alignment methods and is the entire point of alignment.",
                "C) The safety score improvement is suspicious — DPO cannot improve safety, only helpfulness",
                "D) Re-train with higher beta to eliminate the helpfulness decrease entirely",
            ],
            "answer": "B",
            "explanation": (
                "The alignment tax is well-documented: aligned models sacrifice some capability "
                "for improved safety/harmlessness. A 7% helpfulness decrease (4.5→4.2) for "
                "a 129% safety increase (2.1→4.8) is an excellent ratio. "
                "This happens because the preference data teaches: 'When safety and helpfulness "
                "conflict, prefer safety.' The model learns to refuse or caveat harmful requests "
                "instead of being maximally helpful regardless of consequences. "
                "Higher beta would actually REDUCE the alignment effect (keeping model closer to "
                "the unaligned reference), not eliminate the tax."
            ),
            "learning_outcome": "Evaluate alignment tax trade-off between helpfulness and safety",
        },
        # ── Section C: RL ───────────────────────────────────────────────
        {
            "id": "10.C.1",
            "lesson": "10.C",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3 implements Q-learning on a grid world. The agent converges to "
                "a suboptimal policy: it takes the safe but long path (15 steps, no penalties) "
                "instead of the short path (5 steps through two cells with -1.0 penalty each). "
                "The goal gives +1.0. The discount factor is gamma=0.5. What is wrong?"
            ),
            "options": [
                "A) The learning rate is too high — the Q-values are oscillating",
                "B) gamma=0.5 is too low, making the distant goal nearly worthless. Short path value: -1.0 + 0.5×0 + 0.5²×0 + 0.5³×(-1.0) + 0.5⁴×1.0 = -1.0 - 0.125 + 0.0625 = -1.0625 (negative!). Long path: 0.5¹⁵ × 1.0 ≈ 0.00003 (tiny but non-negative). The agent rationally avoids the short path because penalties are immediate but the goal is heavily discounted. Fix: gamma=0.99 gives short path value ≈ -1.0 + 0.99³×(-1.0) + 0.99⁴×1.0 = -1.0 - 0.97 + 0.96 = -1.01 — still negative. With gamma=0.99 and a +10 goal: +6.1 (clearly positive).",
                "C) The epsilon-greedy exploration rate is too low — the agent never discovers the short path",
                "D) Q-learning cannot handle negative rewards — switch to SARSA",
            ],
            "answer": "B",
            "explanation": (
                "Discount factor gamma controls how much the agent values future rewards. "
                "At gamma=0.5, each step discounts by half. The short path traverses two -1.0 "
                "penalty cells: V_short = -1.0 + 0.5³×(-1.0) + 0.5⁴×1.0 = -1.0 - 0.125 + 0.0625 "
                "= -1.0625 (net negative). The long safe path: V_long = 0.5¹⁵ × 1.0 ≈ 0 (tiny but "
                "non-negative). The agent rationally prefers the long path because gamma=0.5 makes "
                "the goal too distant to offset the immediate penalties. "
                "Fix: increase gamma to 0.99 AND increase the goal reward so the discounted goal "
                "outweighs the penalties. The general principle: low gamma → myopic agents that "
                "avoid any short-term pain regardless of long-term gain."
            ),
            "learning_outcome": "Tune discount factor gamma for appropriate reward time horizon",
        },
        {
            "id": "10.C.2",
            "lesson": "10.C",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 trains PPO on inventory management. The reward curve shows: "
                "rapid improvement for 50 episodes, then a plateau at reward=850, while "
                "the heuristic baseline achieves 800. A colleague says 'only 6% better "
                "than a simple rule — RL is not worth it.' How do you respond?"
            ),
            "options": [
                "A) They're right — 6% is not worth the training complexity",
                "B) The 6% average improvement understates RL's value. Check the variance: PPO likely has lower reward variance (more consistent decisions). Also check edge cases: PPO may dramatically outperform the heuristic during demand spikes or supply disruptions where static rules fail. The heuristic uses a fixed threshold; PPO adapts its policy based on the current state.",
                "C) Train for more episodes — PPO always converges to the optimal solution eventually",
                "D) The reward function is wrong — redesign it to get a larger improvement",
            ],
            "answer": "B",
            "explanation": (
                "Average reward comparison misses two key metrics: "
                "(1) Variance: PPO with consistent decisions (std=20) vs heuristic with "
                "high variance (std=100) is operationally much better — fewer stockouts. "
                "(2) Tail performance: PPO adapts to unusual demand patterns (holiday spikes, "
                "supply chain disruptions) where the fixed heuristic threshold fails badly. "
                "The heuristic may average 800 but hit -500 during a supply crisis, "
                "while PPO adjusts ordering dynamically. "
                "In production, reliability matters more than average performance."
            ),
            "learning_outcome": "Evaluate RL policy beyond average reward using variance and tail performance",
        },
        # ── Section D: Model merging ────────────────────────────────────
        {
            "id": "10.D.1",
            "lesson": "10.D",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 merges SFT and DPO adapters. Linear merge gives F1=0.78, "
                "SLERP gives F1=0.82. Why does SLERP outperform linear interpolation?"
            ),
            "options": [
                "A) SLERP uses more computation during merging, producing a better result",
                "B) Linear interpolation W = α×W_sft + (1-α)×W_dpo can shrink weight magnitudes. At α=0.5, ||W|| ≤ 0.5×||W_sft|| + 0.5×||W_dpo||. This 'magnitude collapse' weakens the model. SLERP interpolates along the unit hypersphere, preserving ||W|| throughout the interpolation. The merged weights maintain their original scale, preserving model capacity.",
                "C) SLERP automatically selects the better model for each layer — it's not truly merging",
                "D) Linear merge requires the adapters to be trained on the same data; SLERP does not",
            ],
            "answer": "B",
            "explanation": (
                "Consider two unit vectors at 90°: linear interpolation at t=0.5 gives a vector "
                "with magnitude cos(45°) = 0.707 — a 30% magnitude reduction. "
                "SLERP maintains unit magnitude throughout the interpolation. "
                "For neural network weights, magnitude carries information: a shrunken weight "
                "matrix produces weaker activations, potentially losing learned patterns. "
                "SLERP preserves the 'energy' of both adapters while smoothly blending their "
                "directional information. Exercise 5 demonstrates this: linear merge's F1=0.78 "
                "reflects the capacity loss, while SLERP's F1=0.82 preserves full model strength."
            ),
            "learning_outcome": "Choose SLERP over linear merge to preserve weight magnitude during adapter merging",
        },
        {
            "id": "10.D.2",
            "lesson": "10.D",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 5 registers merged adapters in AdapterRegistry. A student tries to "
                "merge three adapters (SFT + DPO + domain) but gets poor results. The SFT and "
                "DPO adapters were trained on the same base model, but the domain adapter was "
                "trained on a different base model. What went wrong?"
            ),
            "options": [
                "A) Three-way merging is not supported — only two adapters can be merged",
                "B) Adapters from different base models occupy different weight spaces and cannot be meaningfully merged. SFT and DPO adapters are corrections (ΔW) relative to the SAME base model W₀. Merging ΔW_sft + ΔW_dpo makes sense because both modify the same W₀. A domain adapter from a different W₀' produces ΔW_domain that is meaningless when applied to W₀.",
                "C) The merge weights were not set correctly — use 0.33 for each adapter",
                "D) AdapterRegistry does not support three adapters — only register two at a time",
            ],
            "answer": "B",
            "explanation": (
                "LoRA adapters are residual corrections: W_final = W_base + ΔW. "
                "When merging ΔW_1 and ΔW_2, we assume both correct the SAME W_base. "
                "If ΔW_domain was trained against a different W_base', then: "
                "W_base + ΔW_sft + ΔW_dpo + ΔW_domain ≠ anything meaningful, because "
                "ΔW_domain 'expects' W_base' underneath it. "
                "Solution: retrain the domain adapter on the same base model, or merge "
                "SFT+DPO first, then fine-tune the merged result on domain data."
            ),
            "learning_outcome": "Ensure adapter base model compatibility before merging in AdapterRegistry",
        },
        # ── Section E: Governance ───────────────────────────────────────
        {
            "id": "10.E.1",
            "lesson": "10.E",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 6 defines a PACT organization with D/T/R delegations. The "
                "model_trainer agent (clearance=confidential) tries to access restricted-level "
                "audit logs. GovernanceEngine blocks the request. The student asks: "
                "'But the trainer needs to see audit logs to debug training issues.' "
                "What is the governance-correct solution?"
            ),
            "options": [
                "A) Elevate model_trainer's clearance to restricted",
                "B) Create a new delegation. The chief_risk_officer (who has restricted clearance authority) delegates a 'training_audit' task to model_trainer with a SCOPED envelope: read-only access to training-related audit entries only, not all audit logs. This maintains least-privilege while enabling the legitimate use case. The D/T/R chain: CRO → training_audit → model_trainer.",
                "C) Disable the clearance check for the audit logs endpoint",
                "D) Have the risk_assessor agent read the logs and send a summary to model_trainer",
            ],
            "answer": "B",
            "explanation": (
                "PACT's D/T/R grammar solves this elegantly: "
                "1. The need is legitimate — trainer needs training audit data. "
                "2. Elevating clearance (A) violates least-privilege — trainer gets ALL restricted data. "
                "3. Disabling checks (C) removes governance entirely. "
                "4. A new delegation from CRO creates a scoped access path: "
                "   D=chief_risk_officer, T=training_audit, R=model_trainer "
                "   Envelope: read_only=True, filter='training_*', max_rows=1000 "
                "5. This is auditable: the audit trail shows WHO authorized WHAT access for WHOM. "
                "Option D (agent proxy) is a workaround, not governance."
            ),
            "learning_outcome": "Use scoped D/T/R delegations to grant least-privilege access",
        },
        {
            "id": "10.E.2",
            "lesson": "10.E",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 wraps agents with PactGovernedAgent. The governed agent has "
                "max_budget_usd=2.0 but processes 5 queries at $0.50 each ($2.50 total) "
                "without being blocked. The audit trail shows all 5 as 'allowed'. "
                "What is wrong with the governance configuration?"
            ),
            "code": (
                "governed = PactGovernedAgent(\n"
                "    agent=base_agent,\n"
                "    governance_engine=engine,\n"
                "    role='analyst',\n"
                "    max_budget_usd=2.0,\n"
                "    allowed_tools=['read_data', 'analyze_text'],\n"
                "    # Bug: missing clearance_level\n"
                ")\n"
            ),
            "options": [
                "A) max_budget_usd is per-query, not cumulative — set to 0.40 for a $2.00 total limit",
                "B) The budget enforcement checks each query independently against max_budget_usd, not cumulative spend. To enforce cumulative limits, the GovernanceEngine must be configured with budget tracking enabled. Additionally, missing clearance_level defaults to the highest level, bypassing data access controls. Always specify clearance_level explicitly.",
                "C) PactGovernedAgent budget enforcement only works with Delegate, not ReActAgent",
                "D) The governance_engine was not compiled — call engine.compile_org() first",
            ],
            "answer": "B",
            "explanation": (
                "Two issues: (1) Budget tracking must be explicitly enabled in GovernanceEngine "
                "for cumulative enforcement. Without it, each request is checked independently "
                "against max_budget_usd. 5 × $0.50 < $2.00 per-request, so all pass. "
                "With tracking: running total $0.50, $1.00, $1.50, $2.00, $2.50 → 5th blocked. "
                "(2) Missing clearance_level is a security gap — the default may be permissive. "
                "Always specify clearance_level to enforce data access boundaries. "
                "The audit trail showing 'allowed' for all 5 is the diagnostic clue — "
                "cumulative enforcement would show the 5th as 'blocked'."
            ),
            "learning_outcome": "Configure PactGovernedAgent with cumulative budget tracking and explicit clearance",
        },
    ],
}
