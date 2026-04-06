# Chapter 2: Delegate

## Overview

Delegate is the single entry point for autonomous AI execution in Kaizen. It composes an AgentLoop with optional governance (GovernedSupervisor) and provides both streaming (`run()`) and synchronous (`run_sync()`) interfaces. This chapter covers Delegate construction with progressive disclosure (three layers), the typed event system for pattern matching, budget tracking, and cost estimation by model provider.

## Prerequisites

- Completed Chapter 1: Signatures
- Environment variables configured (`.env` with `DEFAULT_LLM_MODEL`)
- Understanding of async generators in Python

## Concepts

### Progressive Disclosure

Delegate uses three layers of configuration complexity:

| Layer                | What You Provide                               | What You Get              |
| -------------------- | ---------------------------------------------- | ------------------------- |
| Layer 1 (minimal)    | `model`                                        | Basic AI execution        |
| Layer 2 (configured) | `model`, `tools`, `system_prompt`, `max_turns` | Customized agent behavior |
| Layer 3 (governed)   | All above + `budget_usd`                       | Cost-controlled execution |

### The Event Stream

`Delegate.run()` is an async generator that yields typed `DelegateEvent` subclass instances. These are structured data for pattern matching, not raw strings:

| Event Type        | When Yielded                    | Key Fields                           |
| ----------------- | ------------------------------- | ------------------------------------ |
| `TextDelta`       | Incremental text from the model | `text`                               |
| `TurnComplete`    | Model finished responding       | `prompt_tokens`, `completion_tokens` |
| `BudgetExhausted` | Budget cap exceeded             | `budget_usd`, `consumed_usd`         |
| `ErrorEvent`      | An error occurred               | `error`, `details`                   |

### Cost Estimation

Delegate uses conservative per-1M-token cost estimates by model prefix:

| Model Prefix | Input Cost (per 1M) | Output Cost (per 1M) |
| ------------ | ------------------- | -------------------- |
| `claude-`    | $3.00               | $15.00               |
| `gpt-4o`     | $2.50               | $10.00               |
| `gemini-`    | $1.25               | $5.00                |

These are approximations for budget tracking, not billing data.

## Key API

| Method / Property                                              | Parameters    | Returns                         | Description                    |
| -------------------------------------------------------------- | ------------- | ------------------------------- | ------------------------------ | ----------------------------- |
| `Delegate(model, tools, system_prompt, max_turns, budget_usd)` | see below     | `Delegate`                      | Create a delegate              |
| `delegate.run(prompt)`                                         | `prompt: str` | `AsyncGenerator[DelegateEvent]` | Streaming async execution      |
| `delegate.run_sync(prompt)`                                    | `prompt: str` | `str`                           | Synchronous full-text response |
| `delegate._budget_usd`                                         | --            | `float                          | None`                          | Budget cap (None = unlimited) |
| `delegate._consumed_usd`                                       | --            | `float`                         | Cost consumed so far           |
| `delegate._config.max_turns`                                   | --            | `int`                           | Maximum turns per session      |

## Code Walkthrough

### Step 1: Layer 1 -- Minimal Delegate

```python
import os
from dotenv import load_dotenv
load_dotenv()

from kaizen_agents import Delegate

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

delegate = Delegate(model=model)

assert isinstance(delegate, Delegate)
assert delegate._budget_usd is None  # No budget cap
```

Model names always come from environment variables -- never hardcoded.

### Step 2: Layer 2 -- Configured Delegate

```python
configured = Delegate(
    model=model,
    tools=[],
    system_prompt="You are a helpful code reviewer.",
    max_turns=20,
)

assert configured._config.max_turns == 20
```

### Step 3: Layer 3 -- Governed Delegate

```python
governed = Delegate(
    model=model,
    budget_usd=5.0,
)

assert governed._budget_usd == 5.0
assert governed._consumed_usd == 0.0
```

### Step 4: Event Types

```python
from kaizen_agents.delegate.events import (
    DelegateEvent, TextDelta, TurnComplete,
    BudgetExhausted, ErrorEvent,
)

assert issubclass(TextDelta, DelegateEvent)
assert issubclass(TurnComplete, DelegateEvent)
assert issubclass(BudgetExhausted, DelegateEvent)
assert issubclass(ErrorEvent, DelegateEvent)
```

### Step 5: Streaming Consumption Pattern

```python
# Production pattern (requires LLM API key):
# async for event in delegate.run("What is Kailash?"):
#     if isinstance(event, TextDelta):
#         print(event.text, end="")
#     elif isinstance(event, TurnComplete):
#         print(f"\nTokens: {event.prompt_tokens} + {event.completion_tokens}")
#     elif isinstance(event, BudgetExhausted):
#         print("Budget exceeded")
#     elif isinstance(event, ErrorEvent):
#         print(f"Error: {event.error}")
```

### Step 6: Budget Validation

```python
try:
    Delegate(model=model, budget_usd=-1.0)
except ValueError as e:
    assert "non-negative" in str(e).lower()

try:
    Delegate(model=model, budget_usd=float("inf"))
except ValueError as e:
    assert "finite" in str(e).lower()
```

Budget must be finite and non-negative.

### Step 7: Model from Environment

```python
env_delegate = Delegate(model="")
assert env_delegate._config.model is not None
```

When `model=""`, Delegate reads `DEFAULT_LLM_MODEL` from the environment.

## Common Mistakes

| Mistake                             | Problem                          | Fix                                                 |
| ----------------------------------- | -------------------------------- | --------------------------------------------------- |
| Hardcoding model names              | Breaks on provider changes       | Use `os.environ.get("DEFAULT_LLM_MODEL")`           |
| Forgetting `load_dotenv()`          | Environment variables not loaded | Call `load_dotenv()` before any `os.environ` access |
| Ignoring `BudgetExhausted` events   | Agent runs beyond budget         | Always handle budget events in the event loop       |
| Using `run()` without async context | `run()` is an async generator    | Use `async for` or `run_sync()` for scripts         |

## Exercises

1. **Three Layers**: Create a Delegate at each of the three layers. Verify the defaults at Layer 1, the configuration at Layer 2, and the budget tracking at Layer 3.

2. **Budget Validation**: Test all invalid budget values: negative, infinity, NaN. Verify that each raises `ValueError` with an appropriate message.

3. **Cost Estimation**: Given a model prefix, calculate the estimated cost for 5000 prompt tokens and 2000 completion tokens. Verify the math against the cost table.

## Key Takeaways

- Delegate is the single entry point for AI execution, with three layers of progressive disclosure.
- `run()` yields typed `DelegateEvent` instances for pattern matching (not raw strings).
- `run_sync()` provides a synchronous convenience for scripts.
- `budget_usd` enables automatic cost tracking with `BudgetExhausted` events.
- Model names always come from environment variables, never hardcoded.
- Budget must be finite and non-negative; zero is valid (immediately exhausted).
- Cost estimation uses conservative per-1M-token rates by model prefix.

## Next Chapter

[Chapter 3: Agent Configuration](03_agent_config.md) -- Configure the Kaizen framework with KaizenConfig and AgentManager.
