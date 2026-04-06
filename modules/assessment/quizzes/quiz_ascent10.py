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
        {
            "id": "10.A.3",
            "lesson": "10.A",
            "type": "output_interpretation",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 produces this memory comparison table for fine-tuning a 7B model:\n\n"
                "| Method        | Model Weights | Trainable Params | Optimizer States | Total  |\n"
                "| Full FT       | 14 GB         | 14 GB            | 28 GB            | 56 GB  |\n"
                "| LoRA (r=16)   | 14 GB         | 8.4 MB           | 16.8 MB          | ~14 GB |\n"
                "| QLoRA (r=16)  | 3.5 GB        | 8.4 MB           | 16.8 MB          | ~3.5 GB|\n\n"
                "Why does QLoRA use only 3.5 GB for model weights instead of 14 GB, "
                "and why are optimizer states so dramatically different between Full FT and LoRA?"
            ),
            "options": [
                "A) QLoRA deletes 75% of the model weights — it only keeps the most important parameters",
                "B) QLoRA quantizes the frozen base model to 4-bit (14 GB fp16 ÷ 4 = 3.5 GB). LoRA adapters remain in fp16 for training precision. Optimizer states differ because Adam stores 2 states (m, v) per trainable parameter: Full FT has 14B trainable × 2 × 2 bytes = 28 GB. LoRA has 8.4M trainable × 2 × 2 bytes = 16.8 MB — a 1,667× reduction. The base model weights require zero optimizer states because they are frozen.",
                "C) QLoRA uses model pruning to remove unnecessary layers, reducing from 14 GB to 3.5 GB",
                "D) The table is wrong — QLoRA cannot reduce memory below the LoRA baseline",
            ],
            "answer": "B",
            "explanation": (
                "Memory breakdown by component: "
                "(1) Model weights: Full FT and LoRA both store the base model in fp16 (7B × 2 bytes = 14 GB). "
                "QLoRA quantizes to NF4 (4-bit): 7B × 0.5 bytes = 3.5 GB. The model still works because "
                "4-bit NormalFloat preserves 99.7% of fp16 information for inference. "
                "(2) Trainable params: Full FT trains all 7B. LoRA/QLoRA train only low-rank adapters (8.4M). "
                "(3) Optimizer states: Adam maintains first moment (m) and second moment (v) for each "
                "trainable parameter. Full FT: 7B × 2 states × 2 bytes = 28 GB. "
                "LoRA/QLoRA: 8.4M × 2 states × 2 bytes = 16.8 MB. "
                "QLoRA's breakthrough: 4-bit base model + fp16 adapters = fine-tuning a 7B model on a "
                "single 8 GB GPU (3.5 GB weights + 8.4 MB adapters + 16.8 MB optimizer ≈ 3.5 GB + activations)."
            ),
            "learning_outcome": "Interpret QLoRA memory savings from quantization of frozen weights and reduced optimizer states",
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
        {
            "id": "10.B.3",
            "lesson": "10.B",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 2's DPO training completes without errors, but the aligned model "
                "behaves identically to the base model. Evaluation shows 0% preference shift "
                "on the test set. Training loss decreased normally. What is wrong?"
            ),
            "code": (
                "config = AlignmentConfig(\n"
                "    method='dpo',\n"
                "    base_model=os.environ['DPO_BASE_MODEL'],\n"
                "    beta=50.0,        # Bug: beta far too high\n"
                "    lora_r=16,\n"
                "    lora_alpha=32,\n"
                "    learning_rate=5e-7,\n"
                "    num_epochs=3,\n"
                ")\n"
            ),
            "options": [
                "A) The learning rate is too low — increase to 1e-3 for faster convergence",
                "B) beta=50.0 is far too high. In DPO's loss function, beta controls the strength of the KL divergence constraint from the reference policy. High beta means the model is heavily penalized for deviating from the reference (unaligned) policy — effectively preventing any alignment. At beta=50, even a small policy change produces enormous KL penalty, so the optimizer learns to stay near the reference. Fix: beta=0.1-0.5 is the standard range. beta=0.1 allows moderate deviation; beta=0.5 for conservative alignment.",
                "C) DPO requires at least 10 epochs — 3 is not enough for any alignment",
                "D) The lora_r=16 is too small — the adapter cannot express the alignment changes",
            ],
            "answer": "B",
            "explanation": (
                "DPO loss = -log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))). "
                "The beta parameter scales the log-ratio difference. At beta=50, the gradient signal "
                "is dominated by the KL constraint: any deviation from π_ref is penalized 50× more "
                "than the preference signal. The optimizer minimizes total loss by keeping π_θ ≈ π_ref "
                "(zero deviation = zero KL penalty), ignoring the preference data entirely. "
                "Training loss decreases because the model learns to perfectly predict the reference "
                "policy's outputs — but this is NOT alignment, it is reference copying. "
                "Standard beta values: 0.1 (aggressive alignment), 0.2-0.3 (balanced), 0.5 (conservative). "
                "Beta=50 is 100-500× too high."
            ),
            "learning_outcome": "Diagnose DPO alignment failure from excessive beta constraining policy deviation",
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
        {
            "id": "10.C.3",
            "lesson": "10.C",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 uses PPO for inventory management with discrete actions "
                "(order 0, 10, 50, or 100 units). A new requirement adds continuous "
                "actions: the agent must decide EXACTLY how many units to order (0-200, "
                "any integer). PPO with a softmax output over 201 discrete actions is "
                "impractical. What algorithm should you use instead?"
            ),
            "options": [
                "A) Keep PPO but bin the actions into 10 ranges — discretization always works",
                "B) Switch to SAC (Soft Actor-Critic). SAC is designed for continuous action spaces: the actor outputs a mean and standard deviation for a Gaussian policy, directly sampling continuous values. PPO CAN handle continuous actions but SAC is more sample-efficient in continuous spaces because its entropy regularization encourages exploration automatically. RLTrainer supports both: RLTrainer(algorithm='sac', action_space='continuous').",
                "C) Use Q-learning with a continuous Q-function — just replace the Q-table with a neural network",
                "D) Continuous actions are impossible in RL — always discretize the action space",
            ],
            "answer": "B",
            "explanation": (
                "Continuous action spaces require policies that output probability distributions "
                "over real-valued actions, not discrete softmax probabilities. "
                "SAC (Soft Actor-Critic): outputs μ and σ for a Gaussian → samples action ~ N(μ, σ²). "
                "Entropy bonus (H(π)) encourages exploration without epsilon-greedy heuristics. "
                "PPO can also handle continuous actions with a Gaussian policy head, but SAC's "
                "maximum entropy framework provides more stable training and better exploration "
                "in continuous spaces. "
                "For inventory management: SAC outputs 'order 73.2 units' directly, instead of "
                "choosing from 201 discrete bins. This matters when the optimal order quantity is "
                "sensitive to exact values (e.g., 73 units vs 70 units has meaningfully different cost). "
                "Kailash's RLTrainer(algorithm='sac') handles the continuous policy architecture."
            ),
            "learning_outcome": "Choose SAC over PPO for continuous action space RL problems",
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
        {
            "id": "10.D.3",
            "lesson": "10.D",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 5 exports the merged model to ONNX and compares three variants:\n\n"
                "| Variant          | Size   | F1    | Latency |\n"
                "| fp16 (original)  | 14 GB  | 0.82  | 45ms    |\n"
                "| INT8 quantized   | 3.5 GB | 0.81  | 18ms    |\n"
                "| INT4 quantized   | 1.8 GB | 0.76  | 12ms    |\n\n"
                "Your production environment has 4 GB GPU memory and requires F1 ≥ 0.80. "
                "Which variant do you deploy?"
            ),
            "options": [
                "A) fp16 — always deploy the highest quality model",
                "B) INT8 quantized. fp16 at 14 GB does not fit in 4 GB GPU memory — eliminated. INT4 at F1=0.76 fails the ≥0.80 requirement — eliminated. INT8 at 3.5 GB fits the GPU, achieves F1=0.81 (above threshold), and runs at 18ms (2.5× faster than fp16). The 1-point F1 drop (0.82→0.81) is negligible for a 4× size reduction and 2.5× speed improvement.",
                "C) INT4 — smallest size and fastest latency are always the priority",
                "D) None — quantization always destroys model quality; request a larger GPU",
            ],
            "answer": "B",
            "explanation": (
                "Production deployment requires matching constraints to available options: "
                "(1) Memory constraint: 4 GB GPU eliminates fp16 (14 GB). "
                "(2) Quality constraint: F1 ≥ 0.80 eliminates INT4 (F1=0.76). "
                "(3) INT8 satisfies both: 3.5 GB < 4 GB, F1=0.81 ≥ 0.80. "
                "The INT8 quantization trade-off: each weight stored as 8-bit integer instead of "
                "16-bit float. Size reduction: 14 GB × (8/16) = 7 GB theoretical, but with "
                "quantization metadata overhead, actual is ~3.5 GB. F1 drops only 0.01 because "
                "INT8 preserves most of the model's representational capacity. "
                "The 2.5× latency improvement (45ms → 18ms) comes from: smaller memory bandwidth, "
                "INT8 tensor core operations, and reduced cache pressure."
            ),
            "learning_outcome": "Select ONNX quantization level based on memory, quality, and latency constraints",
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
        {
            "id": "10.E.3",
            "lesson": "10.E",
            "type": "output_interpretation",
            "difficulty": "advanced",
            "question": (
                "Exercise 6's GovernanceEngine audit trail shows this sequence:\n\n"
                "```\n"
                "14:01:03 ALLOWED analyst → read_data(customers) [clearance=confidential]\n"
                "14:01:05 ALLOWED analyst → analyze_text(report_q3) [clearance=confidential]\n"
                "14:01:08 BLOCKED analyst → write_data(customers, {salary: ...}) [requires=restricted]\n"
                "14:01:08 ALLOWED analyst → read_data(products) [clearance=confidential]\n"
                "14:01:10 BLOCKED analyst → export_data(customers) [requires=restricted]\n"
                "14:01:10 ALLOWED analyst → analyze_text(report_q4) [clearance=confidential]\n"
                "```\n\n"
                "What pattern do you observe and what does it tell you about the agent's behavior?"
            ),
            "options": [
                "A) The analyst is behaving normally — blocks are expected for restricted operations",
                "B) The analyst is probing access boundaries. After being blocked on write_data(customers), it immediately tries export_data(customers) — a different operation on the same restricted resource. This retry-with-different-verb pattern suggests the agent is trying to find an allowed path to modify or extract customer data. The interleaved read/analyze calls between blocked attempts may be the agent 'disguising' its intent. GovernanceEngine correctly blocks both, but this pattern should trigger an escalation alert.",
                "C) The blocks are false positives — the analyst's clearance should include restricted data",
                "D) The audit trail is normal — two blocks out of six requests is a healthy ratio",
            ],
            "answer": "B",
            "explanation": (
                "Governance audit trails reveal behavioral patterns beyond individual allow/block decisions. "
                "The sequence shows: (1) analyst reads customer data (allowed — read is within clearance). "
                "(2) analyst tries to WRITE customer salary data (blocked — requires restricted clearance). "
                "(3) analyst reads a different table (potentially innocuous, or covering tracks). "
                "(4) analyst tries to EXPORT customer data (blocked — different verb, same restricted target). "
                "This probe-and-retry pattern with verb variation (write → export) is a known escalation "
                "signal in agent governance. The GovernanceEngine should: (1) Block the operations (done). "
                "(2) Flag the pattern for human review. (3) Consider reducing the agent's clearance "
                "if the pattern persists. PACT's verification gradient escalates from automated "
                "to human review based on exactly this kind of behavioral signal."
            ),
            "learning_outcome": "Interpret GovernanceEngine audit trails for agent boundary-probing behavior",
        },
        {
            "id": "10.E.4",
            "lesson": "10.E",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7's PactGovernedAgent can read public data but cannot read "
                "confidential data — even though its clearance_level is set to 'confidential'. "
                "The GovernanceEngine logs show: 'clearance check: agent_level=public, "
                "required=confidential → BLOCKED'. What is wrong?"
            ),
            "code": (
                "governed = PactGovernedAgent(\n"
                "    agent=base_agent,\n"
                "    governance_engine=engine,\n"
                "    role='analyst',\n"
                "    clearance_level='confidential',  # Set here...\n"
                "    max_budget_usd=5.0,\n"
                "    allowed_tools=['read_data', 'analyze_text'],\n"
                ")\n"
                "\n"
                "# But the org definition says:\n"
                "engine = GovernanceEngine(\n"
                "    org=Organization(\n"
                "        roles={\n"
                "            'analyst': Role(\n"
                "                clearance='public',  # ...overridden here\n"
                "                delegations=['read', 'analyze'],\n"
                "            )\n"
                "        }\n"
                "    )\n"
                ")\n"
            ),
            "options": [
                "A) PactGovernedAgent should be created before GovernanceEngine — the order matters",
                "B) The GovernanceEngine's Organization definition takes precedence over PactGovernedAgent's clearance_level parameter. The org defines role='analyst' with clearance='public'. When PactGovernedAgent sets clearance_level='confidential', the engine resolves the EFFECTIVE clearance by checking the org role definition — which says 'public'. Fix: update the org role definition to clearance='confidential', OR create a separate role with the correct clearance.",
                "C) 'confidential' is not a valid clearance level — use 'secret' instead",
                "D) The allowed_tools list overrides clearance_level — remove it to restore clearance",
            ],
            "answer": "B",
            "explanation": (
                "PACT governance follows a principle of organizational authority: the Organization "
                "definition is the source of truth for what each role can do. PactGovernedAgent's "
                "clearance_level is a REQUEST, not an override. The GovernanceEngine resolves: "
                "effective_clearance = min(agent_request, org_role_maximum). "
                "If the org says analyst has 'public' clearance, no agent can self-escalate to "
                "'confidential' — that would bypass the governance model entirely. "
                "Fix: the CRO or admin must update the org definition: "
                "Role('analyst', clearance='confidential'). "
                "This ensures clearance changes go through proper authorization channels, "
                "maintaining the governance audit trail. The diagnostic clue was the log showing "
                "agent_level=public despite the code saying 'confidential' — the engine resolved it."
            ),
            "learning_outcome": "Understand Organization role definitions override PactGovernedAgent clearance parameters",
        },
    ],
}
