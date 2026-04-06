# Chapter 4: LLM Providers

## Overview

Kaizen supports multiple LLM providers (Anthropic, OpenAI, Google) through a uniform environment-variable-based configuration system. This chapter covers how provider detection works from model name prefixes, the API key pairing rules, the cost estimation table, and the security pattern for managing secrets via `.env` files.

## Prerequisites

- Completed Chapters 1-3
- A `.env` file with at least `DEFAULT_LLM_MODEL` set
- Understanding of environment variable patterns

## Concepts

### Provider Detection

The SDK detects the provider from model name prefixes:

| Prefix                      | Provider  | Required API Key                     |
| --------------------------- | --------- | ------------------------------------ |
| `claude-`                   | Anthropic | `ANTHROPIC_API_KEY`                  |
| `gpt-`, `o1-`, `o3-`, `o4-` | OpenAI    | `OPENAI_API_KEY`                     |
| `gemini-`                   | Google    | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

### Environment-Only Pattern

API keys are always loaded from environment variables at runtime, never at import time and never hardcoded. The standard pattern is:

```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["ANTHROPIC_API_KEY"]
```

### Cost Estimation

Delegate tracks cost estimates per 1M tokens for budget enforcement:

| Model Prefix | Input Cost | Output Cost |
| ------------ | ---------- | ----------- |
| `claude-`    | $3.00      | $15.00      |
| `gpt-4o`     | $2.50      | $10.00      |
| `gpt-4`      | $30.00     | $60.00      |
| `gemini-`    | $1.25      | $5.00       |

## Key API

| Method / Property                     | Parameters | Returns | Description                       |
| ------------------------------------- | ---------- | ------- | --------------------------------- |
| `os.environ.get("DEFAULT_LLM_MODEL")` | --         | `str`   | Primary model for all agents      |
| `os.environ.get("OPENAI_PROD_MODEL")` | --         | `str`   | OpenAI-specific override          |
| `load_dotenv()`                       | --         | `None`  | Load `.env` file into environment |

## Code Walkthrough

### Step 1: Model from Environment

```python
import os
from dotenv import load_dotenv
load_dotenv()

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
assert isinstance(model, str)
assert len(model) > 0
```

### Step 2: Provider Detection

```python
def detect_provider(model_name: str) -> str:
    if model_name.startswith("claude-"):
        return "anthropic"
    elif model_name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    elif model_name.startswith("gemini-"):
        return "google"
    return "unknown"

assert detect_provider("claude-sonnet-4-20250514") == "anthropic"
assert detect_provider("gpt-4o") == "openai"
assert detect_provider("gemini-pro") == "google"
```

### Step 3: Security Verification

```python
import inspect

source = inspect.getsource(detect_provider)
assert "sk-" not in source, "No hardcoded API keys"
```

No API keys should ever appear in source code.

## Common Mistakes

| Mistake                      | Problem                                             | Fix                                         |
| ---------------------------- | --------------------------------------------------- | ------------------------------------------- |
| Hardcoding `model="gpt-4"`   | Breaks on deprecation, prevents env-based switching | Use `os.environ.get("DEFAULT_LLM_MODEL")`   |
| Forgetting `load_dotenv()`   | `os.environ` returns `None` for `.env` vars         | Always call `load_dotenv()` first           |
| Committing `.env` to git     | API keys leaked in history                          | Add `.env` to `.gitignore`                  |
| Mismatched model-key pairing | Opaque 401/403 errors                               | Verify the model prefix matches the API key |

## Exercises

1. **Provider Matrix**: Write a function that returns the required API key environment variable name for a given model name. Test with at least five different model names.

2. **Cost Calculator**: Write a function that estimates the cost for a given model, prompt token count, and completion token count. Test with multiple models and verify the math.

3. **Security Audit**: Scan a Python file for hardcoded API key patterns (strings starting with `sk-`, `key-`, etc.). Report any findings.

## Key Takeaways

- Provider is detected from model name prefixes: `claude-` (Anthropic), `gpt-` (OpenAI), `gemini-` (Google).
- API keys always come from environment variables via `load_dotenv()` + `os.environ`.
- `.env` files must be in `.gitignore` -- never committed to source control.
- Cost estimation uses conservative per-1M-token rates for budget enforcement.
- The SDK reads keys at runtime, not at import time.

## Next Chapter

[Chapter 5: Cost Tracking](05_cost_tracking.md) -- Track LLM costs and enforce budget limits on agents.
