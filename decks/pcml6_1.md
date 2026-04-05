---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.1: SFT Fine-Tuning

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why fine-tuning adapts LLMs to specific domains
- Apply LoRA (Low-Rank Adaptation) for parameter-efficient tuning
- Build a fine-tuning pipeline with Kailash `AlignmentPipeline`
- Prepare training data in the instruction-response format

---

## Recap: Module 5

- LLM fundamentals: Signature, Delegate, Chain-of-Thought
- ReAct agents with tools, RAG systems, MCP servers
- Multi-agent orchestration: Sequential, Parallel, Supervisor-Worker
- Production deployment with Nexus

Module 6 makes these systems **aligned, governed, and trustworthy**.

---

## Why Fine-Tune?

```
Base LLM:
  "What is a good price for a 4-room flat in Tampines?"
  → Generic answer, may hallucinate, no domain expertise

Fine-tuned LLM:
  "What is a good price for a 4-room flat in Tampines?"
  → "Based on recent HDB resale data, 4-room flats in Tampines
     typically range from $450k-$560k. A fair price is around
     $500k depending on floor, lease, and condition."
  → Domain-specific, calibrated, reliable
```

---

## Fine-Tuning Approaches

```
Full fine-tuning:
  Update ALL parameters (billions)
  → Very expensive, needs many GPUs
  → Risk of catastrophic forgetting

LoRA (Low-Rank Adaptation):
  Freeze original weights, add small trainable matrices
  → 0.1-1% of parameters trained
  → Single GPU is sufficient
  → Preserves base model knowledge

QLoRA:
  LoRA + 4-bit quantisation
  → Even less memory
  → Consumer GPU feasible
```

---

## LoRA: How It Works

```
Original layer:  y = Wx        (frozen, billions of params)

LoRA addition:   y = Wx + BAx  (only A, B are trained)

Where:
  W is the original weight matrix (d × d, frozen)
  A is a small matrix (d × r)
  B is a small matrix (r × d)
  r = rank (typically 8, 16, or 32)

Parameters trained: 2 × d × r  (instead of d × d)
If d = 4096 and r = 16:
  Original: 16,777,216 params
  LoRA:     131,072 params (0.8%)
```

---

## LoRA Visual

```
Input ──→ [Frozen W] ──→ Add ──→ Output
   │                       ↑
   └──→ [A] ──→ [B] ──────┘
         r×d    d×r

         (trainable)

The LoRA path learns a small correction to the frozen weights.
Like adding a specialisation layer on top of general knowledge.
```

---

## Preparing Training Data

```python
# Instruction-response format
training_data = [
    {
        "instruction": "What is a fair price for a 4-room flat in Tampines?",
        "response": "Based on recent resale data, 4-room flats in Tampines "
                    "typically sell for $480k-$540k. Key factors: floor level "
                    "(+$15-25k for high floors), remaining lease, and "
                    "renovation status."
    },
    {
        "instruction": "Compare HDB prices in Bedok vs Woodlands.",
        "response": "Bedok averages $420k-$500k for 4-room flats, while "
                    "Woodlands averages $380k-$450k. The $40-50k premium "
                    "in Bedok reflects its mature estate status and "
                    "MRT connectivity."
    },
    # ... hundreds of examples
]
```

---

## Data Quality Matters More Than Quantity

```
1,000 high-quality examples > 100,000 noisy examples

High quality means:
  ✅ Accurate information (verified against real data)
  ✅ Consistent format (same tone, structure, detail level)
  ✅ Diverse coverage (different towns, flat types, questions)
  ✅ Clear instruction-response mapping

Low quality:
  ❌ Hallucinated numbers
  ❌ Inconsistent format across examples
  ❌ Duplicate or near-duplicate examples
  ❌ Ambiguous instructions
```

---

## AlignmentPipeline: Fine-Tuning

```python
from kailash_align import AlignmentPipeline

pipeline = AlignmentPipeline()
pipeline.configure(
    base_model="meta-llama/Llama-3-8B",
    method="sft",                    # Supervised Fine-Tuning

    # LoRA configuration
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],

    # Training configuration
    training_data=training_data,
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
)

result = pipeline.run()
print(f"Training loss: {result.final_loss:.4f}")
```

---

## LoRA Hyperparameters

| Parameter        | Description                 | Typical Range       |
| ---------------- | --------------------------- | ------------------- |
| `lora_rank`      | Rank of adaptation matrices | 8-64                |
| `lora_alpha`     | Scaling factor              | 2x rank             |
| `lora_dropout`   | Regularisation              | 0.0-0.1             |
| `target_modules` | Which layers to adapt       | q,v,k,o projections |
| `learning_rate`  | Step size                   | 1e-4 to 5e-4        |
| `epochs`         | Training passes             | 1-5                 |

Higher rank = more capacity = more risk of overfitting.

---

## Evaluating the Fine-Tuned Model

```python
# Compare base vs fine-tuned
from kailash_align import AlignmentPipeline

base_model = pipeline.load_base_model()
tuned_model = pipeline.load_tuned_model()

test_questions = [
    "What affects HDB resale prices most?",
    "Is now a good time to buy in Queenstown?",
]

for q in test_questions:
    base_answer = base_model.generate(q)
    tuned_answer = tuned_model.generate(q)
    print(f"Q: {q}")
    print(f"  Base:  {base_answer[:100]}...")
    print(f"  Tuned: {tuned_answer[:100]}...")
```

---

## Saving and Loading Adapters

```python
# Save LoRA adapter (small file, ~10-50MB)
pipeline.save_adapter("./adapters/hdb_expert_v1")

# Load adapter onto base model later
from kailash_align import AlignmentPipeline

loaded = AlignmentPipeline()
loaded.load_adapter(
    base_model="meta-llama/Llama-3-8B",
    adapter_path="./adapters/hdb_expert_v1",
)

# Generate with the adapted model
answer = loaded.generate("What is the outlook for HDB prices?")
```

Adapters are small and portable -- the base model is not duplicated.

---

## SFT Limitations

```
SFT teaches the model WHAT to say, but not WHAT NOT to say.

SFT alone:
  ✅ Learns domain language and format
  ✅ Learns to follow instructions
  ❌ May still hallucinate confidently
  ❌ Cannot distinguish good answers from great ones
  ❌ No preference learning

Next step: preference alignment (DPO, RLHF) in Lesson 6.2
```

---

## Exercise Preview

**Exercise 6.1: Fine-Tune an HDB Expert**

You will:

1. Prepare instruction-response training data from HDB domain
2. Configure LoRA parameters and train with `AlignmentPipeline`
3. Evaluate base vs fine-tuned model on domain questions
4. Save and load the LoRA adapter

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                               | Fix                                            |
| ------------------------------------- | ---------------------------------------------- |
| Training data with factual errors     | Verify all numbers against real data           |
| LoRA rank too high for small datasets | Start with rank 8-16; increase if underfitting |
| No evaluation on held-out questions   | Always split data into train/eval              |
| Fine-tuning too many epochs           | Monitor eval loss; stop when it increases      |
| Forgetting to save the adapter        | Always save after successful training          |

---

## Summary

- Fine-tuning adapts base LLMs to specific domains
- LoRA trains <1% of parameters via low-rank adaptation matrices
- `AlignmentPipeline` with `method="sft"` handles the training loop
- Data quality matters more than quantity (hundreds of good examples)
- SFT teaches format and knowledge but not preferences

---

## Next Lesson

**Lesson 6.2: Preference Alignment**

We will learn:

- DPO (Direct Preference Optimisation) for learning preferences
- LLM-as-judge for automated preference labelling
- Moving beyond SFT to teach models what is "better"
