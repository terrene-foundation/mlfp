---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.1: LLM Fundamentals

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain how large language models work at a conceptual level
- Use Kailash Kaizen `Signature` for structured LLM interactions
- Apply `Delegate` for direct LLM task execution
- Design effective prompts with input/output contracts

---

## Recap: Module 4

- Unsupervised learning: clustering, GMMs, dimensionality reduction
- NLP text processing, anomaly detection, ensemble methods
- Deep learning with PyTorch, drift monitoring
- InferenceServer for production model deployment

Module 5 adds **language intelligence** to your ML systems.

---

## What Is an LLM?

```
Input:   "What is the average HDB price in Tampines?"
Output:  "Based on recent data, the average resale price
          for HDB flats in Tampines is approximately $520,000."

Under the hood:
  - Billions of parameters learned from text
  - Predicts the next token (word/subword)
  - No "understanding" — statistical pattern matching
  - But the patterns are extraordinarily useful
```

---

## LLMs in the ML Pipeline

```
Traditional ML:     Data → Features → Model → Prediction
                    (structured, numerical, trained by you)

LLM-augmented ML:   Data → Features → Model → Prediction
                         ↑               ↑
                    LLM extracts     LLM explains
                    features from    predictions in
                    text data        natural language
```

LLMs complement, not replace, traditional ML.

---

## Kaizen: The Agent Framework

```
Kailash Kaizen provides:
  - Signature:  Define input/output contracts for LLM calls
  - Delegate:   Execute tasks with a single LLM call
  - Agents:     Multi-step reasoning (Lessons 5.2-5.3)
  - RAG:        Knowledge-augmented generation (Lesson 5.4)
  - Multi-agent: Orchestrated agent systems (Lesson 5.7)
```

We start with the simplest building blocks: Signature and Delegate.

---

## Signature: Structured LLM Contracts

```python
from kailash_kaizen import Signature

# Define what goes in and what comes out
sig = Signature(
    input_fields={
        "listing_description": "The property listing text",
        "town": "The HDB town name",
    },
    output_fields={
        "sentiment": "positive, neutral, or negative",
        "key_features": "List of highlighted features",
        "renovation_status": "renovated, original, or unknown",
    },
)
```

Signatures make LLM interactions **predictable and testable**.

---

## Why Signatures Matter

```
Without Signature:
  prompt = f"Analyse this listing: {text}"
  response = llm(prompt)
  # Returns... a string? JSON? Who knows?
  # Different every time. Unparseable.

With Signature:
  result = delegate.execute(sig, inputs={...})
  result.sentiment          # "positive"
  result.key_features       # ["near MRT", "renovated kitchen"]
  result.renovation_status  # "renovated"
  # Typed, structured, consistent.
```

---

## Delegate: Single-Call Execution

```python
from kailash_kaizen import Delegate, Signature

delegate = Delegate()
delegate.configure(
    model="claude-sonnet",
    temperature=0.0,     # deterministic
)

sig = Signature(
    input_fields={"description": "Property listing text"},
    output_fields={
        "sentiment": "positive, neutral, or negative",
        "highlights": "List of key selling points",
    },
)

result = delegate.execute(sig, inputs={
    "description": "Spacious 4-room flat near Tampines MRT, recently renovated"
})

print(result.sentiment)    # "positive"
print(result.highlights)   # ["near MRT", "recently renovated", "spacious"]
```

---

## Delegate for Data Enrichment

```python
import polars as pl

# Enrich listings with LLM-extracted features
enriched_rows = []
for row in df.iter_rows(named=True):
    result = delegate.execute(sig, inputs={
        "description": row["description"],
    })
    enriched_rows.append({
        **row,
        "sentiment": result.sentiment,
        "highlights": result.highlights,
    })

df_enriched = pl.DataFrame(enriched_rows)
```

LLM-extracted features can feed into traditional ML models.

---

## Prompt Engineering Principles

```
1. Be specific:
   ❌ "Analyse this text"
   ✅ "Extract the sentiment and key features from this listing"

2. Provide examples (few-shot):
   "Example: 'Near MRT, renovated' → sentiment: positive"

3. Constrain output:
   "Respond with exactly: positive, neutral, or negative"

4. Use Signatures:
   The contract defines the structure — no ambiguity
```

---

## Temperature and Determinism

```python
delegate.configure(
    model="claude-sonnet",
    temperature=0.0,  # deterministic (same input → same output)
)

# For creative tasks:
delegate.configure(
    model="claude-sonnet",
    temperature=0.7,  # more varied responses
)
```

| Temperature | Use Case                             |
| ----------- | ------------------------------------ |
| 0.0         | Classification, extraction, analysis |
| 0.3         | Summarisation, structured generation |
| 0.7         | Creative writing, brainstorming      |
| 1.0         | Maximum creativity (rarely used)     |

---

## Structured Output Types

```python
sig = Signature(
    input_fields={
        "listing": "Property listing text",
    },
    output_fields={
        "price_estimate": "Estimated price as integer",
        "confidence": "Float between 0 and 1",
        "features": "List of strings",
        "analysis": {
            "pros": "List of advantages",
            "cons": "List of disadvantages",
            "summary": "One sentence summary",
        },
    },
)
```

Signatures support strings, numbers, lists, and nested objects.

---

## Error Handling

```python
from kailash_kaizen import Delegate, DelegateError

delegate = Delegate()
delegate.configure(model="claude-sonnet")

try:
    result = delegate.execute(sig, inputs={"description": listing})

    # Validate output
    if result.sentiment not in ["positive", "neutral", "negative"]:
        print(f"Unexpected sentiment: {result.sentiment}")

except DelegateError as e:
    print(f"LLM call failed: {e}")
    # Fallback to rule-based classification
```

---

## Batch Processing with Delegate

```python
# Process multiple items efficiently
results = delegate.execute_batch(
    sig,
    inputs=[
        {"description": row["description"]}
        for row in df.head(100).iter_rows(named=True)
    ],
    max_concurrent=5,   # parallel requests
)

for i, result in enumerate(results):
    print(f"Listing {i}: {result.sentiment}")
```

---

## Exercise Preview

**Exercise 5.1: LLM-Powered Property Analysis**

You will:

1. Define Signatures for property listing analysis
2. Use Delegate to extract sentiment and features from listings
3. Enrich a DataFrame with LLM-extracted features
4. Compare model performance with and without LLM features

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                               | Fix                                            |
| ------------------------------------- | ---------------------------------------------- |
| Unstructured prompts                  | Use Signatures for predictable output          |
| Temperature too high for extraction   | Use 0.0 for deterministic tasks                |
| Calling LLM in a tight loop           | Use `execute_batch` with concurrency           |
| Not validating LLM output             | Always check that outputs match expected types |
| Using LLM for tasks ML handles better | LLMs for text; ML for numbers                  |

---

## Summary

- LLMs predict text tokens based on learned patterns from training data
- Kaizen `Signature` defines structured input/output contracts
- `Delegate` executes single-call LLM tasks with consistent output
- Temperature controls determinism (0.0 for extraction, higher for creativity)
- LLM-extracted features enrich traditional ML pipelines

---

## Next Lesson

**Lesson 5.2: Chain-of-Thought**

We will learn:

- `ChainOfThoughtAgent` for multi-step reasoning
- When and why step-by-step thinking improves results
- Designing reasoning chains for complex analysis
