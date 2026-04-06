# Chapter 1: Signatures

## Overview

Signatures are the foundational abstraction in Kaizen's agent system. A Signature declares WHAT an agent does -- its inputs and outputs -- without specifying HOW it does it. The docstring becomes the agent's system instructions, and `InputField`/`OutputField` descriptors define the structured interface. This chapter covers class-based signatures, programmatic creation, inheritance, field metadata, and the metaclass machinery that powers the system.

## Prerequisites

- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Understanding of Python class syntax and metaclasses (conceptual)
- Familiarity with type annotations

## Concepts

### What Is a Signature?

A Signature is a declarative contract between the programmer and the AI agent. It specifies:

- **System prompt**: The class docstring becomes the agent's instructions.
- **Inputs**: Fields the agent receives (e.g., a question, context, data).
- **Outputs**: Fields the agent must produce (e.g., an answer, confidence score).

The agent framework handles the LLM call, prompt construction, and response parsing. You only declare the interface.

### Why Signatures?

Without signatures, agent code mixes prompt engineering, I/O parsing, and business logic. Signatures separate the interface declaration from the execution strategy, making agents:

- **Composable**: Swap execution strategies without changing the interface.
- **Testable**: Verify input/output contracts without calling an LLM.
- **Auditable**: Every field has a description, making the agent's capabilities transparent.

### How Does SignatureMeta Work?

The `Signature` class uses a metaclass (`SignatureMeta`) that processes class definitions at import time. It extracts `InputField` and `OutputField` instances from class annotations and stores them in `_signature_inputs` and `_signature_outputs` dictionaries.

### Intent and Guidelines

Beyond the docstring, signatures support:

- `__intent__`: A high-level statement of WHY the agent exists.
- `__guidelines__`: A list of behavioral constraints (HOW the agent should behave).

## Key API

| Method / Property                  | Parameters                                 | Returns                  | Description                              |
| ---------------------------------- | ------------------------------------------ | ------------------------ | ---------------------------------------- |
| `Signature` (class-based)          | class body with fields                     | Signature class          | Declare a signature via class definition |
| `Signature()` (programmatic)       | `inputs`, `outputs`, `name`, `description` | `Signature` instance     | Create a signature dynamically           |
| `InputField(description, default)` | `description: str`, `default` (optional)   | `InputField`             | Declare an input field                   |
| `OutputField(description)`         | `description: str`                         | `OutputField`            | Declare an output field                  |
| `cls._signature_inputs`            | --                                         | `dict[str, InputField]`  | All input fields                         |
| `cls._signature_outputs`           | --                                         | `dict[str, OutputField]` | All output fields                        |
| `cls._signature_intent`            | --                                         | `str`                    | The intent statement                     |
| `cls._signature_guidelines`        | --                                         | `list[str]`              | Behavioral guidelines                    |
| `field.required`                   | --                                         | `bool`                   | True if no default value                 |
| `field.default`                    | --                                         | `Any`                    | Default value (if set)                   |
| `field.metadata`                   | --                                         | `dict`                   | Extra keyword arguments                  |

## Code Walkthrough

### Step 1: Class-Based Signature (Recommended)

```python
from kaizen import Signature, InputField, OutputField

class QASignature(Signature):
    """You are a helpful question-answering assistant."""

    question: str = InputField(description="The question to answer")
    context: str = InputField(description="Additional context", default="")
    answer: str = OutputField(description="Clear, accurate answer")
    confidence: float = OutputField(description="Confidence score 0.0 to 1.0")
```

The docstring becomes the system prompt. `question` is required (no default); `context` is optional (has a default).

### Step 2: Verify Metaclass Processing

```python
assert "question" in QASignature._signature_inputs
assert "context" in QASignature._signature_inputs
assert "answer" in QASignature._signature_outputs
assert "confidence" in QASignature._signature_outputs

question_field = QASignature._signature_inputs["question"]
assert question_field.required is True

context_field = QASignature._signature_inputs["context"]
assert context_field.required is False
assert context_field.default == ""
```

### Step 3: Intent and Guidelines

```python
class CustomerSupportSignature(Signature):
    """You are a customer support agent for an e-commerce platform."""

    __intent__ = "Resolve customer issues efficiently and empathetically"
    __guidelines__ = [
        "Acknowledge concerns before proposing solutions",
        "Use empathetic language",
        "Escalate if unresolved in 3 turns",
    ]

    customer_message: str = InputField(description="Customer's message")
    order_history: str = InputField(description="Recent order history", default="")
    response: str = OutputField(description="Support response")
    action: str = OutputField(description="Action: resolve, escalate, transfer")

assert CustomerSupportSignature._signature_intent == (
    "Resolve customer issues efficiently and empathetically"
)
assert len(CustomerSupportSignature._signature_guidelines) == 3
```

### Step 4: Programmatic Creation

```python
sig = Signature(
    inputs=["query", "context"],
    outputs=["result"],
    name="search",
    description="Search for relevant information",
)
```

Use programmatic creation when inputs/outputs are determined at runtime.

### Step 5: Signature Inheritance

```python
class DetailedQA(QASignature):
    """Extended QA with source citations."""

    sources: str = OutputField(description="List of sources used")

assert "question" in DetailedQA._signature_inputs   # Inherited
assert "answer" in DetailedQA._signature_outputs     # Inherited
assert "sources" in DetailedQA._signature_outputs    # Added
```

Child fields merge with parent fields. On name collision, the child field wins.

### Step 6: Field Metadata

```python
class RichSignature(Signature):
    """Signature with rich field metadata."""

    data: str = InputField(
        description="Input data",
        format="json",
        max_length=10000,
    )
    result: str = OutputField(description="Processed result")

data_field = RichSignature._signature_inputs["data"]
assert data_field.metadata.get("format") == "json"
assert data_field.metadata.get("max_length") == 10000
```

Extra keyword arguments on `InputField`/`OutputField` are stored in `field.metadata`.

## Common Mistakes

| Mistake                     | Problem                        | Fix                                         |
| --------------------------- | ------------------------------ | ------------------------------------------- |
| Forgetting the docstring    | Agent has no system prompt     | Always include a descriptive docstring      |
| Making all fields required  | Agent fails on partial input   | Use `default=""` for optional fields        |
| Minimal output fields       | Agent response lacks structure | Include all expected output dimensions      |
| Putting logic in signatures | Signatures are data, not code  | Put logic in the agent's execution strategy |

## Exercises

1. **Sentiment Analyzer**: Create a `SentimentSignature` with inputs `text` (required) and `language` (optional, default `"en"`), and outputs `sentiment`, `confidence`, and `keywords`. Verify all fields via `_signature_inputs` and `_signature_outputs`.

2. **Inheritance Chain**: Create a base `AnalysisSignature` with `data` input and `result` output. Create `DetailedAnalysis` that adds `methodology` output. Create `PeerReviewedAnalysis` that adds `reviewer_notes` output. Verify the inheritance chain.

3. **Field Metadata**: Create a signature where one input field has `format="csv"`, `max_rows=1000`, and `encoding="utf-8"` metadata. Access the metadata dictionary and verify all three values.

## Key Takeaways

- Signatures declare WHAT an agent does (inputs/outputs), not HOW.
- The class docstring becomes the agent's system prompt.
- `InputField(description=, default=)` marks inputs; `OutputField(description=)` marks outputs.
- Fields without defaults are required; fields with defaults are optional.
- `__intent__` and `__guidelines__` provide high-level purpose and behavioral constraints.
- Signatures support inheritance -- child fields merge with parent fields.
- Extra keyword arguments on fields are stored in `field.metadata`.

## Next Chapter

[Chapter 2: Delegate](02_delegate.md) -- Use Delegate, the primary entry point for autonomous AI execution.
