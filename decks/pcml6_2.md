---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.2: Preference Alignment

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why preference alignment goes beyond supervised fine-tuning
- Apply DPO (Direct Preference Optimisation) with `AlignmentPipeline`
- Use LLM-as-judge for automated preference data generation
- Compare SFT, DPO, and RLHF approaches

---

## Recap: Lesson 6.1

- Fine-tuning adapts LLMs to specific domains via LoRA
- SFT teaches format and knowledge from instruction-response pairs
- LoRA trains <1% of parameters for efficient adaptation
- SFT alone cannot teach preferences or prevent confident errors

---

## SFT vs Preference Alignment

```
SFT:  "Here is a good answer. Learn to produce this."
      → Model learns to imitate
      → All training examples treated equally

DPO:  "Answer A is better than Answer B. Learn the difference."
      → Model learns to PREFER better answers
      → Learns from contrasts, not just examples

Analogy:
  SFT  = Showing someone good essays
  DPO  = Showing two essays and explaining why one is better
```

---

## Preference Data Format

```python
preference_data = [
    {
        "prompt": "What is a fair price for a 4-room flat in Tampines?",
        "chosen": "Based on recent transactions, 4-room flats in Tampines "
                  "sell for $480k-$540k. Key factors include floor level, "
                  "remaining lease, and proximity to MRT stations.",
        "rejected": "A 4-room flat in Tampines costs about $500,000. "
                    "Prices vary."
    },
    {
        "prompt": "Should I buy an HDB flat now?",
        "chosen": "This depends on your financial situation. Consider: "
                  "1) Interest rates are currently X%, 2) Cooling measures "
                  "affect ABSD, 3) Your budget and timeline. I'd recommend "
                  "consulting a financial advisor for personalised advice.",
        "rejected": "Yes, now is a great time to buy! Prices are going up "
                    "so you should act fast."
    },
]
```

---

## Why "Chosen" Is Better

```
The chosen answer is preferred because:

1. ACCURACY: Cites specific data ranges, not vague numbers
2. NUANCE: Acknowledges factors and uncertainty
3. SAFETY: Does not give overconfident financial advice
4. HELPFULNESS: Provides actionable information
5. FORMAT: Structured, easy to follow

The rejected answer is worse because:
  → Vague ("about $500k")
  → No supporting detail
  → Overconfident ("great time to buy!")
  → Potentially harmful financial advice
```

---

## DPO: Direct Preference Optimisation

```
DPO loss:
  L = -log σ(β · (log π(chosen) - log π_ref(chosen)
                 - log π(rejected) + log π_ref(rejected)))

In English:
  "Increase the probability of chosen responses
   and decrease the probability of rejected responses,
   relative to the reference model."

No reward model needed (unlike RLHF).
No RL training loop (simpler, more stable).
```

---

## DPO with AlignmentPipeline

```python
from kailash_align import AlignmentPipeline

pipeline = AlignmentPipeline()
pipeline.configure(
    base_model="meta-llama/Llama-3-8B",
    method="dpo",

    # Start from SFT-tuned model
    adapter_path="./adapters/hdb_expert_v1",

    # DPO configuration
    preference_data=preference_data,
    beta=0.1,              # KL divergence weight

    # LoRA settings
    lora_rank=16,
    lora_alpha=32,

    # Training
    epochs=1,
    batch_size=2,
    learning_rate=5e-5,    # lower than SFT
)

result = pipeline.run()
```

---

## DPO Hyperparameters

| Parameter         | Description                | Typical Value |
| ----------------- | -------------------------- | ------------- |
| `beta`            | KL penalty weight          | 0.05-0.2      |
| `learning_rate`   | Step size (lower than SFT) | 1e-5 to 1e-4  |
| `epochs`          | Training passes            | 1-3           |
| `label_smoothing` | Prevent overconfidence     | 0.0-0.1       |

Lower beta = more aggressive preference learning.
Higher beta = stay closer to the reference model.

---

## LLM-as-Judge: Automated Preference Labels

Creating preference pairs manually is expensive. Use a strong LLM to judge.

```python
from kailash_kaizen import Delegate, Signature

judge = Delegate()
judge.configure(model="claude-sonnet", temperature=0.0)

judge_sig = Signature(
    input_fields={
        "prompt": "The original question",
        "response_a": "First response",
        "response_b": "Second response",
    },
    output_fields={
        "winner": "A or B",
        "reasoning": "Why the winner is better",
        "scores": {"accuracy": "1-5", "helpfulness": "1-5", "safety": "1-5"},
    },
)

result = judge.execute(judge_sig, inputs={
    "prompt": question,
    "response_a": answer_a,
    "response_b": answer_b,
})
```

---

## Building Preference Data at Scale

```python
# Generate multiple responses, then judge them
base_model = pipeline.load_base_model()
sft_model = pipeline.load_tuned_model()

preference_pairs = []
for prompt in evaluation_prompts:
    # Generate responses from different models
    response_base = base_model.generate(prompt)
    response_sft = sft_model.generate(prompt)

    # Judge which is better
    judgment = judge.execute(judge_sig, inputs={
        "prompt": prompt,
        "response_a": response_base,
        "response_b": response_sft,
    })

    chosen = response_sft if judgment.winner == "B" else response_base
    rejected = response_base if judgment.winner == "B" else response_sft

    preference_pairs.append({
        "prompt": prompt, "chosen": chosen, "rejected": rejected,
    })
```

---

## SFT → DPO Pipeline

```
Step 1: Prepare instruction-response data
        └→ Domain-specific Q&A pairs

Step 2: SFT fine-tuning
        └→ AlignmentPipeline(method="sft")
        └→ Model learns domain language and format

Step 3: Generate preference pairs
        └→ Multiple responses + LLM-as-judge scoring

Step 4: DPO alignment
        └→ AlignmentPipeline(method="dpo")
        └→ Model learns to prefer better answers

Step 5: Evaluate
        └→ Compare base vs SFT vs DPO on test questions
```

---

## Evaluation: Win Rate

```python
# Compare DPO model against SFT model
dpo_model = pipeline.load_tuned_model()
sft_model = pipeline.load_sft_model()

wins = {"dpo": 0, "sft": 0, "tie": 0}

for prompt in test_prompts:
    dpo_response = dpo_model.generate(prompt)
    sft_response = sft_model.generate(prompt)

    judgment = judge.execute(judge_sig, inputs={
        "prompt": prompt,
        "response_a": dpo_response,
        "response_b": sft_response,
    })

    if judgment.winner == "A":
        wins["dpo"] += 1
    elif judgment.winner == "B":
        wins["sft"] += 1
    else:
        wins["tie"] += 1

print(f"DPO wins: {wins['dpo']}, SFT wins: {wins['sft']}, Ties: {wins['tie']}")
```

---

## SFT vs DPO vs RLHF

| Method   | Data Needed          | Complexity | Quality            |
| -------- | -------------------- | ---------- | ------------------ |
| **SFT**  | Instruction-response | Low        | Good baseline      |
| **DPO**  | Preference pairs     | Medium     | Better preferences |
| **RLHF** | Reward model + RL    | High       | Best (but fragile) |

DPO achieves near-RLHF quality with much simpler training.

---

## Exercise Preview

**Exercise 6.2: Preference-Aligned HDB Expert**

You will:

1. Create preference pairs for HDB domain responses
2. Use LLM-as-judge for automated preference labelling
3. Train with DPO via `AlignmentPipeline`
4. Evaluate win rate: base vs SFT vs DPO

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                               | Fix                                       |
| ------------------------------------- | ----------------------------------------- |
| DPO without SFT first                 | Always SFT first, then DPO                |
| Preference pairs that are too similar | Chosen/rejected must be clearly different |
| LLM judge with position bias          | Randomise A/B order; test both positions  |
| Learning rate too high for DPO        | Use 5-10x lower LR than SFT               |
| Too many DPO epochs                   | 1-2 epochs is usually sufficient          |

---

## Summary

- Preference alignment teaches models to prefer better answers
- DPO directly optimises from preference pairs without a reward model
- LLM-as-judge automates preference data creation at scale
- The SFT-then-DPO pipeline produces well-aligned domain models
- Evaluate with win rate comparisons, not just loss curves

---

## Next Lesson

**Lesson 6.3: Reinforcement Learning**

We will learn:

- Bellman equation and value functions
- PPO (Proximal Policy Optimisation) for RL training
- `RLTrainer` for reward-based model improvement
