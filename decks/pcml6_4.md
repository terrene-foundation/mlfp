---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.4: Advanced Alignment

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Merge multiple fine-tuned models with TIES and DARE methods
- Combine LoRA adapters for multi-domain expertise
- Apply advanced alignment techniques for production systems
- Evaluate merged models against individual specialists

---

## Recap: Lesson 6.3

- RL learns policies from reward signals via Bellman equation
- PPO clips policy updates for stable training
- RLHF uses a learned reward model + PPO for LLM alignment
- DPO is simpler and often comparable to RLHF

---

## The Multi-Expert Problem

```
You have fine-tuned three specialists:
  Adapter A: HDB pricing expert (trained on pricing data)
  Adapter B: Policy/regulation expert (trained on policy docs)
  Adapter C: Investment advisor (trained on financial data)

Question: "Given the new cooling measures, is this flat
           a good investment at $520k?"

Needs ALL THREE experts. Options:
  1. Run all three separately → inconsistent, redundant
  2. Merge into one model → unified expertise
```

---

## Model Merging: The Concept

```
Model merging combines multiple fine-tuned models
into one model with combined capabilities.

   Base Model
   ┌────────┐
   │        │
   │ Llama  │──→ LoRA A (pricing) ──┐
   │  3-8B  │──→ LoRA B (policy)  ──┤──→ Merged Model
   │        │──→ LoRA C (finance) ──┘    (all three skills)
   └────────┘
```

No additional training needed -- merging is a weight arithmetic operation.

---

## TIES: Task Interference Elimination by Sign

```
Problem: Naively averaging weights creates interference.

  Weight from A: +0.5   (important for pricing)
  Weight from B: -0.3   (important for policy)
  Average: +0.1          (useless for both!)

TIES method:
  1. Trim: Remove small-magnitude changes (noise)
  2. Elect sign: For conflicting signs, majority wins
  3. Merge: Average only the surviving, sign-aligned weights
```

---

## TIES Step by Step

```
Step 1 - Trim (remove noise):
  ΔA: [+0.5, +0.02, -0.3, +0.01]  → [+0.5, 0, -0.3, 0]
  ΔB: [-0.1, +0.4, -0.2, +0.03]   → [-0.1, +0.4, -0.2, 0]

Step 2 - Elect sign (resolve conflicts):
  Position 0: +0.5 vs -0.1  → positive wins (higher magnitude)
  Position 2: -0.3 vs -0.2  → both negative, no conflict

Step 3 - Merge (average sign-aligned):
  Position 0: only +0.5 survives → +0.5
  Position 1: only +0.4 survives → +0.4
  Position 2: mean(-0.3, -0.2)   → -0.25
```

---

## DARE: Drop and REscale

```
DARE takes a different approach:

  1. Randomly DROP a fraction of weight changes (e.g., 90%)
  2. RESCALE remaining weights to compensate

Why this works:
  - Fine-tuned weights are highly redundant
  - Dropping forces the model to rely on shared knowledge
  - Rescaling preserves the total magnitude

DARE is simpler than TIES and often equally effective.
```

---

## Merging with AlignmentPipeline

```python
from kailash_align import AlignmentPipeline

pipeline = AlignmentPipeline()

# TIES merging
merged_model = pipeline.merge_adapters(
    base_model="meta-llama/Llama-3-8B",
    adapters=[
        {"path": "./adapters/pricing_v1", "weight": 0.4},
        {"path": "./adapters/policy_v1", "weight": 0.3},
        {"path": "./adapters/finance_v1", "weight": 0.3},
    ],
    method="ties",
    trim_threshold=0.1,    # remove changes < 10% of max
)

# Save merged adapter
pipeline.save_adapter(merged_model, "./adapters/hdb_merged_v1")
```

---

## DARE Merging

```python
merged_dare = pipeline.merge_adapters(
    base_model="meta-llama/Llama-3-8B",
    adapters=[
        {"path": "./adapters/pricing_v1", "weight": 0.4},
        {"path": "./adapters/policy_v1", "weight": 0.3},
        {"path": "./adapters/finance_v1", "weight": 0.3},
    ],
    method="dare",
    drop_rate=0.9,    # drop 90% of weight changes
)
```

---

## Evaluating Merged Models

```python
from kailash_kaizen import Delegate, Signature

# Test across all domains
test_questions = {
    "pricing": ["What is a fair price for...", ...],
    "policy": ["What are the latest cooling measures...", ...],
    "finance": ["Is this a good investment...", ...],
}

for domain, questions in test_questions.items():
    correct = 0
    for q in questions:
        merged_answer = merged_model.generate(q)
        specialist_answer = specialists[domain].generate(q)

        # Judge quality
        judgment = judge.execute(judge_sig, inputs={
            "prompt": q,
            "response_a": merged_answer,
            "response_b": specialist_answer,
        })
        if judgment.winner == "A":
            correct += 1

    print(f"{domain}: merged wins {correct}/{len(questions)}")
```

---

## Adapter Stacking

```python
# Alternative to merging: stack adapters at inference time
model = pipeline.load_base_model("meta-llama/Llama-3-8B")

# Load multiple adapters
model.load_adapter("./adapters/pricing_v1", adapter_name="pricing")
model.load_adapter("./adapters/policy_v1", adapter_name="policy")

# Activate specific adapter based on question type
model.set_active_adapter("pricing")
price_answer = model.generate("What is a fair price for...")

model.set_active_adapter("policy")
policy_answer = model.generate("What are the cooling measures...")
```

Stacking switches adapters at runtime without merging.

---

## Merging vs Stacking vs Multi-Agent

| Approach        | Pros                         | Cons                         |
| --------------- | ---------------------------- | ---------------------------- |
| **Merging**     | Single model, fast inference | May lose specialist quality  |
| **Stacking**    | Full specialist quality      | Must classify question first |
| **Multi-agent** | Maximum flexibility          | Slower, more complex         |

Choose based on latency requirements and quality needs.

---

## Advanced: Iterative DPO

```python
# Round 1: DPO on initial preferences
pipeline.configure(method="dpo", preference_data=round1_data)
model_v1 = pipeline.run()

# Round 2: Generate new responses with v1, get new preferences
new_responses = model_v1.generate_batch(prompts)
round2_data = create_preferences(new_responses)

# Round 3: DPO again with harder examples
pipeline.configure(method="dpo", preference_data=round2_data,
                   adapter_path=model_v1.adapter_path)
model_v2 = pipeline.run()

# Each round improves on harder examples
```

---

## Exercise Preview

**Exercise 6.4: Multi-Domain Model Merging**

You will:

1. Fine-tune specialist adapters for pricing, policy, and investment
2. Merge with TIES and DARE methods
3. Compare merged model vs individual specialists
4. Evaluate adapter stacking as an alternative

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                             | Fix                                    |
| ----------------------------------- | -------------------------------------- |
| Equal weights for all adapters      | Weight by importance or quality        |
| TIES trim threshold too aggressive  | Start at 0.1; increase carefully       |
| DARE drop rate too high             | 0.9 is typical; 0.95+ may lose quality |
| Not testing all domains after merge | Regression in one domain is common     |
| Merging incompatible LoRA configs   | Same rank and target modules required  |

---

## Summary

- Model merging combines multiple specialists into one model
- TIES resolves weight conflicts by trimming noise and electing signs
- DARE randomly drops weight changes and rescales for robustness
- Adapter stacking switches experts at runtime without merging
- Iterative DPO progressively improves alignment quality

---

## Next Lesson

**Lesson 6.5: AI Governance**

We will learn:

- PACT governance framework: Duties, Triggers, Rights
- `GovernanceEngine` for policy enforcement
- Building governed AI systems for enterprise deployment
