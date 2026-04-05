---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.2: Chain-of-Thought

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why step-by-step reasoning improves LLM accuracy
- Build `ChainOfThoughtAgent` for multi-step analysis
- Design reasoning chains for complex property valuation
- Evaluate reasoning quality beyond just the final answer

---

## Recap: Lesson 5.1

- LLMs predict text tokens from learned patterns
- `Signature` defines structured input/output contracts
- `Delegate` executes single-call LLM tasks
- Temperature 0.0 for deterministic extraction; higher for creativity

---

## The Problem with Direct Answers

```
Question: "Is this flat fairly priced at $520k?"
  Input: 4-room, Tampines, 95sqm, 70-year lease, high floor

Direct answer (Delegate):
  "Yes, this appears to be fairly priced."
  → No reasoning visible. How did it decide?

Chain-of-thought answer:
  "Step 1: Average 4-room price in Tampines is ~$490k
   Step 2: This flat is 95sqm (above average 90sqm) → +$15k
   Step 3: High floor premium → +$20k
   Step 4: 70-year lease is slightly below average → -$5k
   Conclusion: Fair value ~$520k. This is fairly priced."
  → Reasoning is transparent, verifiable, and more accurate.
```

---

## Why Chain-of-Thought Works

```
Without CoT:
  Complex question → [black box] → Answer
  Error rate: high on multi-step problems

With CoT:
  Complex question → Step 1 → Step 2 → Step 3 → Answer
  Each step is simple; errors are catchable

Analogy: solving 347 × 28 in your head vs on paper
  Head: error-prone
  Paper: each step is verifiable
```

---

## ChainOfThoughtAgent

```python
from kailash_kaizen import ChainOfThoughtAgent, Signature

agent = ChainOfThoughtAgent()
agent.configure(
    model="claude-sonnet",
    temperature=0.0,
    max_steps=5,
)

sig = Signature(
    input_fields={
        "listing": "Property details",
        "asking_price": "The listed price",
    },
    output_fields={
        "fair_value": "Estimated fair value as integer",
        "assessment": "overpriced, fair, or underpriced",
        "reasoning_steps": "List of reasoning steps taken",
    },
)

result = agent.execute(sig, inputs={
    "listing": "4-room, Tampines, 95sqm, lease 70yr, floor 15",
    "asking_price": 520_000,
})
```

---

## Inspecting the Reasoning Chain

```python
print(f"Fair value: ${result.fair_value:,}")
print(f"Assessment: {result.assessment}")
print(f"\nReasoning:")
for i, step in enumerate(result.reasoning_steps, 1):
    print(f"  Step {i}: {step}")
```

```
Fair value: $518,000
Assessment: fair

Reasoning:
  Step 1: Baseline 4-room Tampines price: ~$490,000
  Step 2: Floor area 95sqm (above avg 90sqm): +$15,000
  Step 3: High floor (15th): +$18,000
  Step 4: Lease 70yr (below avg 75yr): -$5,000
  Step 5: Fair value estimate: $518,000 vs asking $520,000 → fair
```

---

## Designing Reasoning Templates

```python
agent.configure(
    model="claude-sonnet",

    # Guide the reasoning structure
    reasoning_template=[
        "Identify the comparable market segment",
        "Assess size premium or discount vs average",
        "Evaluate location-specific factors",
        "Consider lease and age adjustments",
        "Synthesise into a fair value estimate",
    ],
)
```

Templates guide the agent through a specific analytical framework.

---

## CoT for Data Analysis

```python
analysis_sig = Signature(
    input_fields={
        "statistics": "Summary statistics of the dataset",
        "question": "The analytical question to answer",
    },
    output_fields={
        "answer": "The analytical conclusion",
        "reasoning": "Step-by-step reasoning",
        "caveats": "List of limitations or assumptions",
    },
)

result = agent.execute(analysis_sig, inputs={
    "statistics": str(df.describe()),
    "question": "Which town segment shows the strongest price growth?",
})
```

---

## CoT vs Delegate: When to Use Each

| Scenario              | Use       | Why                            |
| --------------------- | --------- | ------------------------------ |
| Simple extraction     | Delegate  | One-step, no reasoning needed  |
| Classification        | Delegate  | Map input to category directly |
| Multi-factor analysis | CoT Agent | Multiple factors to weigh      |
| Valuation/estimation  | CoT Agent | Step-by-step calculation       |
| Comparison            | CoT Agent | Needs structured evaluation    |
| Creative generation   | Delegate  | Reasoning chain adds overhead  |

---

## Evaluating Reasoning Quality

```python
# Not just the final answer — evaluate each step

evaluation_sig = Signature(
    input_fields={
        "reasoning_chain": "The steps taken by the agent",
        "known_facts": "Ground truth data for verification",
    },
    output_fields={
        "step_accuracy": "List of correct/incorrect per step",
        "logical_consistency": "Are steps logically connected?",
        "overall_quality": "Score from 1-5",
    },
)

eval_result = delegate.execute(evaluation_sig, inputs={
    "reasoning_chain": str(result.reasoning_steps),
    "known_facts": str(market_data),
})
```

---

## Self-Consistency: Multiple Chains

```python
# Run the same question multiple times
results = []
for _ in range(5):
    r = agent.execute(sig, inputs=inputs)
    results.append(r.fair_value)

import numpy as np
mean_value = np.mean(results)
std_value = np.std(results)

print(f"Estimates: {results}")
print(f"Mean: ${mean_value:,.0f} (std: ${std_value:,.0f})")

# High std → reasoning is unstable → question may be ambiguous
```

If different reasoning chains reach different conclusions, the question needs more constraints.

---

## CoT with Context Injection

```python
agent.configure(
    model="claude-sonnet",
    system_context="""
    You are a Singapore HDB property valuation expert.
    Use these market benchmarks for your reasoning:
    - Average 3-room: $350k
    - Average 4-room: $490k
    - Average 5-room: $580k
    - High floor premium: $15-25k
    - MRT proximity premium: $20-40k
    - Lease depreciation: ~$3k per year below 80yr
    """,
)
```

Context injection gives the reasoning chain factual grounding.

---

## Exercise Preview

**Exercise 5.2: Chain-of-Thought Property Valuation**

You will:

1. Build a `ChainOfThoughtAgent` for property valuation
2. Design reasoning templates for different property types
3. Evaluate reasoning quality with self-consistency checks
4. Compare CoT accuracy vs direct Delegate answers

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                        | Fix                                              |
| ------------------------------ | ------------------------------------------------ |
| Using CoT for simple tasks     | Delegate is faster for extraction/classification |
| Not inspecting reasoning steps | Always review the chain, not just the answer     |
| Too many steps (>10)           | Keep chains focused; 3-7 steps is optimal        |
| No ground truth for evaluation | Use known sales prices to validate estimates     |
| Trusting LLM calculations      | LLMs are poor at arithmetic; verify numbers      |

---

## Summary

- Chain-of-thought breaks complex problems into verifiable steps
- `ChainOfThoughtAgent` provides structured multi-step reasoning
- Reasoning templates guide the analytical framework
- Self-consistency (multiple chains) validates reasoning stability
- Evaluate each step, not just the final answer

---

## Next Lesson

**Lesson 5.3: ReAct Agents with Tools**

We will learn:

- ReAct pattern: Reasoning + Acting with external tools
- Giving agents access to databases, APIs, and calculators
- Building agents that gather information before answering
