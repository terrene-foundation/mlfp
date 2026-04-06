# Chapter 8: HyperparameterSearch

## Overview

HyperparameterSearch wraps TrainingPipeline to systematically explore hyperparameter combinations. You define a `SearchSpace` with `ParamDistribution` entries, choose a `SearchConfig` strategy (grid, random, or Bayesian), and call `search()`. The engine runs multiple trials, tracks all results, and identifies the best configuration.

This chapter covers:

- Defining search spaces with `ParamDistribution` (uniform, log_uniform, int_uniform, categorical)
- Exhaustive grid search via `SearchSpace.sample_grid()`
- Random sampling via `SearchSpace.sample_random()`
- Running searches with `SearchConfig` (strategy, n_trials, direction)
- Inspecting `SearchResult` and `TrialResult` objects
- Maximization vs minimization
- Serialization round-trips

## Prerequisites

| Requirement   | Details                      |
| ------------- | ---------------------------- |
| Python        | 3.10+                        |
| kailash-ml    | `pip install kailash-ml`     |
| Prior chapter | Chapter 7 (TrainingPipeline) |
| Level         | Intermediate                 |

## Concepts

### ParamDistribution Types

| Type          | Behavior                                | Use Case                                   |
| ------------- | --------------------------------------- | ------------------------------------------ |
| `uniform`     | Continuous float between low and high   | General hyperparameters                    |
| `log_uniform` | Log-scale sampling between low and high | Learning rates (spans orders of magnitude) |
| `int_uniform` | Integer between low and high            | n_estimators, max_depth                    |
| `categorical` | Draws from a list of choices            | Activation functions, solvers              |

### Search Strategies

| Strategy   | Behavior                                                            |
| ---------- | ------------------------------------------------------------------- |
| `grid`     | Exhaustive enumeration of all combinations. Ignores `n_trials`.     |
| `random`   | Samples `n_trials` random combinations from the space.              |
| `bayesian` | Uses past results to guide future sampling (most sample-efficient). |

### Direction

- `maximize` -- Higher is better (accuracy, f1, r2)
- `minimize` -- Lower is better (loss, rmse, mae)

The best trial is selected based on the `metric_to_optimize` and `direction`.

## Key API Reference

| Class / Method                                                                                   | Purpose                                                                                                    |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `HyperparameterSearch(pipeline)`                                                                 | Create with a TrainingPipeline                                                                             |
| `search.search(data, schema, base_model_spec, search_space, config, eval_spec, experiment_name)` | Async. Run the search                                                                                      |
| `SearchSpace(params=[...])`                                                                      | Define the parameter space                                                                                 |
| `ParamDistribution(name, type, low, high, choices)`                                              | Single parameter definition                                                                                |
| `SearchConfig(strategy, n_trials, metric_to_optimize, direction)`                                | Search configuration                                                                                       |
| `SearchResult`                                                                                   | Fields: `strategy`, `all_trials`, `best_trial_number`, `best_metrics`, `best_params`, `total_time_seconds` |
| `TrialResult`                                                                                    | Fields: `trial_number`, `params`, `metrics`, `training_time_seconds`                                       |
| `SearchSpace.sample_grid()`                                                                      | Generate all grid combinations                                                                             |
| `SearchSpace.sample_random(n)`                                                                   | Generate n random samples                                                                                  |

## Code Walkthrough

### Step 1: Define a SearchSpace

```python
from kailash_ml.engines.hyperparameter_search import (
    HyperparameterSearch, SearchSpace, ParamDistribution, SearchConfig,
)

space = SearchSpace(params=[
    ParamDistribution(name="n_estimators", type="int_uniform", low=5, high=30),
    ParamDistribution(name="max_depth", type="categorical", choices=[3, 5, 7]),
])
```

### Step 2: Preview Samples

```python
# Exhaustive grid
grid = space.sample_grid()
assert len(grid) > 0
assert "n_estimators" in grid[0]

# Random samples
random_samples = space.sample_random(n=10)
for sample in random_samples:
    assert 5 <= sample["n_estimators"] <= 30
    assert sample["max_depth"] in [3, 5, 7]
```

### Step 3: Configure and Run Random Search

```python
config = SearchConfig(
    strategy="random",
    n_trials=3,
    metric_to_optimize="accuracy",
    direction="maximize",
)

result = await search.search(
    data=df,
    schema=schema,
    base_model_spec=base_model_spec,
    search_space=space,
    config=config,
    eval_spec=eval_spec,
    experiment_name="random_search_demo",
)

assert len(result.all_trials) == 3
assert result.best_trial_number >= 0
assert "accuracy" in result.best_metrics
```

### Step 4: Grid Search

```python
small_space = SearchSpace(params=[
    ParamDistribution(name="max_depth", type="categorical", choices=[3, 5]),
])

grid_config = SearchConfig(
    strategy="grid",
    n_trials=10,  # Ignored for grid -- uses full grid
    metric_to_optimize="accuracy",
    direction="maximize",
)

grid_result = await search.search(
    data=df, schema=schema,
    base_model_spec=base_model_spec,
    search_space=small_space,
    config=grid_config,
    eval_spec=eval_spec,
    experiment_name="grid_search_demo",
)

assert len(grid_result.all_trials) == 2  # 2 choices for max_depth
```

### Step 5: Minimization

```python
min_config = SearchConfig(
    strategy="random", n_trials=2,
    metric_to_optimize="accuracy",
    direction="minimize",
)

min_result = await search.search(...)

# Best trial has the lowest accuracy
best_acc = min_result.best_metrics["accuracy"]
for trial in min_result.all_trials:
    assert best_acc <= trial.metrics["accuracy"]
```

## Common Mistakes

| Mistake                                | What Happens                  | Fix                                     |
| -------------------------------------- | ----------------------------- | --------------------------------------- |
| Using `grid` with continuous params    | Enormous grid, slow search    | Use `categorical` or switch to `random` |
| Invalid strategy name                  | `ValueError`                  | Use `grid`, `random`, or `bayesian`     |
| Setting `n_trials=1` for random search | Only one configuration tested | Use at least 3-5 trials                 |
| Forgetting `metric_to_optimize`        | No best trial selected        | Always specify the target metric        |
| Mixing up maximize/minimize            | Best trial is actually worst  | Match direction to the metric semantics |

## Exercises

1. **Grid vs random**: Define a space with 3 categorical params (3 choices each = 27 grid points). Run grid search and random search with 10 trials. How do the best results compare?

2. **Log-uniform for learning rates**: Create a `log_uniform` param for learning rate (0.0001 to 0.1). Sample 20 values and verify they span multiple orders of magnitude.

3. **Search result analysis**: Run a 20-trial random search. Plot trial number vs metric value. Does the search find better results over time (suggesting a pattern), or is it uniformly random?

## Key Takeaways

- SearchSpace defines the parameter landscape with four distribution types.
- Grid search is exhaustive but expensive; random search is often more efficient.
- `sample_grid()` and `sample_random()` let you preview the search space before committing.
- Direction (maximize vs minimize) determines which trial is selected as best.
- All result types support serialization for logging and reproducibility.
- Grid search ignores `n_trials` -- it always evaluates the full grid.

## Next Chapter

Chapter 9 covers **AutoMLEngine** -- the engine that automates model selection and hyperparameter optimization with optional agent-based augmentation.
