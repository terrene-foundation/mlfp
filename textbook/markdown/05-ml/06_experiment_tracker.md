# Chapter 6: ExperimentTracker

## Overview

ExperimentTracker records every aspect of your ML experiments: parameters, step-based metrics, tags, and run status. It uses an async context manager pattern (`async with tracker.run(...) as ctx:`) that automatically marks runs as COMPLETED on normal exit and FAILED on exception. You can compare runs side-by-side, retrieve metric histories for training curves, and search runs by parameters.

This chapter covers:

- Creating experiments and tracking runs with the context manager pattern
- Logging parameters, metrics (scalar and step-based), and tags
- Automatic run status management (RUNNING, COMPLETED, FAILED)
- Comparing multiple runs with `compare_runs()`
- Retrieving step-based metric histories for training curves
- Searching runs by parameter filters
- Deleting runs and experiments
- Serialization round-trips for all data types

## Prerequisites

| Requirement        | Details                  |
| ------------------ | ------------------------ |
| Python             | 3.10+                    |
| kailash-ml         | `pip install kailash-ml` |
| kailash (core SDK) | For `ConnectionManager`  |
| Level              | Intermediate             |

## Concepts

### The Context Manager Pattern

The preferred way to track runs is `async with tracker.run(experiment_name, run_name) as ctx:`. This pattern:

1. Auto-creates the experiment if it does not exist.
2. Sets the run status to `RUNNING` on entry.
3. Marks `COMPLETED` on normal exit with `end_time` recorded.
4. Marks `FAILED` if an exception propagates out of the block.

This ensures every run has a definitive outcome, even if your training code crashes.

### Step-Based Metrics

`log_metric(key, value, step=N)` records a value at a specific training step. This enables training curve visualization -- you can retrieve the full history with `get_metric_history(run_id, "train_loss")` and plot loss over epochs.

### Run Comparison

`compare_runs([run_id_a, run_id_b])` returns a `RunComparison` with metrics and parameters aligned side-by-side. This is the quickest way to see which run performed better and what parameters differed.

### NaN/Inf Rejection

ExperimentTracker rejects `NaN` and `Inf` metric values. These indicate bugs in training code, and storing them would corrupt comparison logic.

## Key API Reference

| Class / Method                                   | Purpose                                                  |
| ------------------------------------------------ | -------------------------------------------------------- |
| `ExperimentTracker(conn, artifact_root)`         | Create tracker with DB connection and artifact directory |
| `tracker.run(experiment_name, run_name)`         | Async context manager for tracking a run                 |
| `ctx.log_params(dict)`                           | Log string parameters within a run                       |
| `ctx.log_metric(key, value, step)`               | Log a single metric at a step                            |
| `ctx.log_metrics(dict, step)`                    | Log multiple metrics at once                             |
| `ctx.set_tag(key, value)`                        | Set a tag on the current run                             |
| `tracker.get_run(run_id)`                        | Async. Retrieve a completed run                          |
| `tracker.get_experiment(name)`                   | Async. Retrieve experiment metadata                      |
| `tracker.list_experiments()`                     | Async. List all experiments                              |
| `tracker.list_runs(experiment, status=None)`     | Async. List runs, optionally filtered by status          |
| `tracker.compare_runs(run_ids)`                  | Async. Side-by-side metric/param comparison              |
| `tracker.get_metric_history(run_id, key)`        | Async. Step-based metric entries                         |
| `tracker.search_runs(experiment, filter_params)` | Async. Find runs by parameter values                     |
| `tracker.delete_run(run_id)`                     | Async. Remove a run                                      |
| `tracker.delete_experiment(name)`                | Async. Remove experiment and all its runs                |

### Data Types

| Type            | Fields                                                                        |
| --------------- | ----------------------------------------------------------------------------- |
| `Experiment`    | `id`, `name`, `created_at`                                                    |
| `Run`           | `id`, `name`, `status`, `start_time`, `end_time`, `params`, `metrics`, `tags` |
| `RunContext`    | `run_id`, `run` (the Run object)                                              |
| `MetricEntry`   | `key`, `value`, `step`, `timestamp`                                           |
| `RunComparison` | `run_ids`, `run_names`, `metrics` (dict of lists), `params` (dict of lists)   |

## Code Walkthrough

### Step 1: Set Up Infrastructure

```python
from kailash.db.connection import ConnectionManager
from kailash_ml.engines.experiment_tracker import ExperimentTracker

conn = ConnectionManager("sqlite:///:memory:")
await conn.initialize()

tracker = ExperimentTracker(conn, artifact_root="/tmp/artifacts")
```

### Step 2: Track a Run with the Context Manager

```python
async with tracker.run("churn_experiment", run_name="baseline") as ctx:
    # Log parameters
    await ctx.log_params({
        "model_class": "RandomForestClassifier",
        "n_estimators": "100",
        "max_depth": "10",
    })

    # Log step-based metrics (training curve)
    await ctx.log_metric("train_loss", 0.65, step=0)
    await ctx.log_metric("train_loss", 0.42, step=1)
    await ctx.log_metric("train_loss", 0.28, step=2)

    # Log scalar metrics
    await ctx.log_metrics({"accuracy": 0.92, "f1": 0.89}, step=0)

    # Set tags
    await ctx.set_tag("team", "ml-platform")

baseline_run_id = ctx.run_id
```

### Step 3: Verify Run Completed

```python
completed_run = await tracker.get_run(baseline_run_id)
assert completed_run.status == "COMPLETED"
assert completed_run.params["model_class"] == "RandomForestClassifier"
assert completed_run.metrics["accuracy"] == 0.92
```

### Step 4: Failed Run

```python
try:
    async with tracker.run("churn_experiment", run_name="failed_run") as ctx3:
        failed_run_id = ctx3.run_id
        raise RuntimeError("Training crashed")
except RuntimeError:
    pass

failed_run = await tracker.get_run(failed_run_id)
assert failed_run.status == "FAILED"
```

### Step 5: Compare Runs

```python
comparison = await tracker.compare_runs([baseline_run_id, improved_run_id])
assert comparison.metrics["accuracy"][0] == 0.92  # baseline
assert comparison.metrics["accuracy"][1] == 0.95  # improved
```

### Step 6: Retrieve Metric History

```python
history = await tracker.get_metric_history(baseline_run_id, "train_loss")
assert len(history) == 3  # steps 0, 1, 2
assert history[0].value == 0.65
assert history[2].value == 0.28
```

### Step 7: Search Runs by Parameters

```python
results = await tracker.search_runs(
    "churn_experiment",
    filter_params={"model_class": "GradientBoostingClassifier"},
)
assert len(results) == 1
```

### Step 8: Delete Operations

```python
await tracker.delete_run(failed_run_id)
await tracker.delete_experiment("throwaway")
```

## Common Mistakes

| Mistake                             | What Happens                           | Fix                                              |
| ----------------------------------- | -------------------------------------- | ------------------------------------------------ |
| Logging `NaN` or `Inf` metrics      | `ValueError`                           | Fix training bugs that produce non-finite values |
| Accessing a non-existent experiment | `ExperimentNotFoundError`              | Check name spelling or use `list_experiments()`  |
| Accessing a non-existent run        | `RunNotFoundError`                     | Verify run_id or use `list_runs()`               |
| Forgetting to use `async with`      | Run never gets COMPLETED/FAILED status | Always use the context manager pattern           |
| Logging params as non-strings       | Implicit conversion may lose precision | Pass string values explicitly                    |

## Exercises

1. **Training curve visualization**: Track a run with 20 steps of `train_loss` and `val_loss`. Retrieve the histories and plot them (using ModelVisualizer's `training_history()`).

2. **Run comparison**: Train three models with different hyperparameters. Use `compare_runs()` to create a side-by-side view. Which model performed best on which metric?

3. **Search and filter**: Create 10 runs with varying `learning_rate` and `batch_size` parameters. Use `search_runs()` to find all runs with a specific learning rate.

## Key Takeaways

- The `async with tracker.run()` context manager automatically manages run lifecycle (RUNNING -> COMPLETED/FAILED).
- Step-based metrics enable training curve tracking and visualization.
- `compare_runs()` provides instant side-by-side comparison of metrics and parameters.
- `search_runs()` filters runs by parameter values for finding specific configurations.
- NaN and Inf metric values are rejected to maintain data integrity.
- All data types support serialization round-trips via `to_dict()` / `from_dict()`.

## Next Chapter

Chapter 7 covers **TrainingPipeline** -- the engine that trains models using ModelSpec, EvalSpec, and FeatureSchema with automatic validation, splitting, evaluation, and model registration.
