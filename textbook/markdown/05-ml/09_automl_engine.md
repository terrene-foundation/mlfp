# Chapter 9: AutoMLEngine

## Overview

AutoMLEngine automates the full model selection process. It quick-trains multiple model families, ranks them by a target metric, runs deep hyperparameter search on the top candidate, and optionally augments the process with AI agent recommendations. The engine supports a "double opt-in" pattern for agent features: you must both install `kailash-ml[agents]` and set `agent=True` in the config.

This chapter covers:

- Configuring AutoML with `AutoMLConfig` (task type, metric, strategy, agent settings)
- Running AutoML for classification and regression
- Understanding `AutoMLResult` and `CandidateResult`
- The agent double opt-in pattern (`pip install kailash-ml[agents]` + `agent=True`)
- LLM cost tracking with `LLMCostTracker` and `LLMBudgetExceededError`
- Financial field validation (NaN/Inf/negative rejection)
- Baseline recommendations vs agent recommendations

## Prerequisites

| Requirement    | Details                                      |
| -------------- | -------------------------------------------- |
| Python         | 3.10+                                        |
| kailash-ml     | `pip install kailash-ml`                     |
| Prior chapters | 7-8 (TrainingPipeline, HyperparameterSearch) |
| Level          | Intermediate                                 |

## Concepts

### How AutoML Works

AutoMLEngine follows a three-phase process:

1. **Quick-train** -- Train multiple model families (RandomForest, GradientBoosting, LogisticRegression, etc.) with default hyperparameters to establish a ranking.
2. **Deep search** -- Run HyperparameterSearch on the top-ranked candidate to find optimal hyperparameters.
3. **Agent augmentation** (optional) -- If `agent=True`, AI agents suggest additional model families, feature strategies, and experiment interpretations.

### Agent Double Opt-In

Agent features require two explicit opt-ins:

1. **Install**: `pip install kailash-ml[agents]` (installs kailash-kaizen as a dependency)
2. **Config**: `agent=True` in `AutoMLConfig`

Without both, agent features are simply not available. The engine falls back to pure algorithmic mode gracefully. This design prevents accidental LLM costs.

### LLM Cost Tracking

When agents are enabled, `LLMCostTracker` tracks cumulative API costs. If spending exceeds `max_llm_cost_usd`, a `LLMBudgetExceededError` is raised, halting agent calls while allowing the algorithmic pipeline to continue.

### Baseline Recommendation

AutoMLResult always includes a `baseline_recommendation` -- the algorithmically determined best models. When `agent=True`, it also includes `agent_recommendation` alongside the baseline, so you can compare human-readable agent suggestions with raw metric rankings.

## Key API Reference

| Class / Method                                                 | Purpose                                                                                                                                                               |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AutoMLEngine(pipeline, hp_search, registry)`                  | Create with pipeline and search                                                                                                                                       |
| `engine.run(data, schema, config, eval_spec, experiment_name)` | Async. Run the full AutoML process                                                                                                                                    |
| `AutoMLConfig`                                                 | Fields: `task_type`, `metric_to_optimize`, `direction`, `search_strategy`, `search_n_trials`, `agent`, `auto_approve`, `max_llm_cost_usd`, `approval_timeout_seconds` |
| `AutoMLResult`                                                 | Fields: `best_model`, `best_metrics`, `all_candidates`, `total_time_seconds`, `baseline_recommendation`, `agent_recommendation`                                       |
| `CandidateResult`                                              | Fields: `model_class`, `rank`, `metrics`, `hyperparameters`                                                                                                           |
| `LLMCostTracker(max_budget_usd)`                               | Track LLM API costs                                                                                                                                                   |
| `LLMBudgetExceededError`                                       | Raised when budget exceeded                                                                                                                                           |

## Code Walkthrough

### Step 1: Set Up and Create Engine

```python
from kailash_ml.engines.automl_engine import AutoMLEngine, AutoMLConfig

engine = AutoMLEngine(pipeline, hp_search, registry=registry)
```

### Step 2: Configure for Classification

```python
config = AutoMLConfig(
    task_type="classification",
    metric_to_optimize="accuracy",
    direction="maximize",
    search_strategy="random",
    search_n_trials=2,
    agent=False,  # No agent augmentation
)
```

### Step 3: Run AutoML

```python
result = await engine.run(
    data=df,
    schema=schema,
    config=config,
    eval_spec=eval_spec,
    experiment_name="automl_classification",
)

assert result.best_model.rank == 1
assert "accuracy" in result.best_metrics
assert len(result.all_candidates) >= 1
assert result.baseline_recommendation is not None
assert result.agent_recommendation is None  # agent=False
```

### Step 4: Regression

```python
reg_config = AutoMLConfig(
    task_type="regression",
    metric_to_optimize="r2",
    direction="maximize",
    search_strategy="random",
    search_n_trials=2,
)

reg_result = await engine.run(
    data=regression_df,
    schema=reg_schema,
    config=reg_config,
    eval_spec=EvalSpec(metrics=["r2", "rmse"]),
    experiment_name="automl_regression",
)
```

### Step 5: LLM Cost Tracking

```python
from kailash_ml.engines.automl_engine import LLMCostTracker, LLMBudgetExceededError

tracker = LLMCostTracker(max_budget_usd=0.10)
tracker.record("test-model", input_tokens=100, output_tokens=50)
assert tracker.total_spent > 0.0

try:
    tracker.record("test-model", input_tokens=100000, output_tokens=100000)
except LLMBudgetExceededError:
    pass  # Budget exceeded
```

### Step 6: Agent Config (Double Opt-In)

```python
agent_config = AutoMLConfig(
    task_type="classification",
    agent=True,           # Opt-in 1: enable agents
    auto_approve=False,   # Human approval gate
    max_llm_cost_usd=5.0, # Cost budget
)
```

## Common Mistakes

| Mistake                                                     | What Happens                              | Fix                                           |
| ----------------------------------------------------------- | ----------------------------------------- | --------------------------------------------- |
| Setting `agent=True` without `kailash-ml[agents]` installed | Falls back to algorithmic mode (no error) | Install the agents extra                      |
| NaN/Inf/negative `max_llm_cost_usd`                         | `ValueError`                              | Use a positive finite number                  |
| Zero `approval_timeout_seconds`                             | `ValueError`                              | Use a positive integer                        |
| Invalid task type (e.g., "unsupervised")                    | `ValueError`                              | Use `classification` or `regression`          |
| Forgetting to check `baseline_recommendation`               | Missing actionable insights               | Always review baseline even when using agents |

## Exercises

1. **Classification vs regression**: Run AutoML on a classification dataset and a regression dataset. Compare the model families that are ranked highest for each task type.

2. **Cost budgeting**: Create an `LLMCostTracker` with a $0.01 budget. Record calls until the budget is exceeded. How many typical calls can you make?

3. **Agent comparison**: Run AutoML twice -- once with `agent=False` and once with `agent=True` (if agents are installed). Compare `baseline_recommendation` with `agent_recommendation`.

## Key Takeaways

- AutoMLEngine automates model family selection and hyperparameter optimization.
- The three-phase process (quick-train, deep search, optional agent augmentation) balances speed with quality.
- Agent features require double opt-in: install `kailash-ml[agents]` and set `agent=True`.
- LLMCostTracker enforces budget limits to prevent runaway API costs.
- Financial fields (costs, budgets) reject NaN, Inf, and negative values.
- `baseline_recommendation` is always available; `agent_recommendation` only when agents are enabled.

## Next Chapter

Chapter 10 covers **EnsembleEngine** -- the engine that combines multiple models using blending, stacking, bagging, and boosting.
