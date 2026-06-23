# MLFP03 — Module Assessment: Supervised Machine Learning

This is the consolidated, auto-graded assessment for **Module 3 — Supervised
Machine Learning for Building and Deploying Models**. It covers the full
supervised lifecycle on a single Southeast-Asia e-commerce dataset: feature
engineering and selection, a multi-algorithm bake-off, honest evaluation under
class imbalance with interpretability, and a production registry + drift +
deploy pipeline.

Everything is **framework-first** (kailash-ml engines) and **polars-native**.

## Format

- **Total**: 100 marks across 4 tasks
- **Duration**: 3 hours
- **Conditions**: open-book, **no AI assistants**
- **Dataset**: `data/mlfp03/ecommerce_customers.parquet` (50,000 customers).
  The tasks derive a documented `premium_response` target (premium-membership
  upsell, ~25% positive — a realistic 3:1 class imbalance), because the native
  `churned` column is a near-deterministic function of recency and unusable for
  modelling practice.

## Tasks

| Task | Title | Weight | Focus |
| ---- | ----- | ------ | ----- |
| **Task 1** | Feature Engineering & Leakage-Free Selection | 20 | Engineer six exact features; rank them with `FeatureEngineer`, fit on the train split only |
| **Task 2** | The Model Zoo | 25 | Train and compare six algorithms through `TrainingPipeline` |
| **Task 3** | Evaluation, Class Imbalance & Interpretability | 25 | Baseline vs `class_weight="balanced"`; per-class recall via `km.diagnose`; SHAP via `ModelExplainer` |
| **Task 4** | Production Pipeline — Registry, Drift, Deploy | 30 | `TrainingPipeline` → `ModelRegistry` (promote to production) → `DriftMonitor` |

## What you submit

Each task folder contains:

- `problem.md` — the scenario, exact contract, performance target, and rules
- `starter.py` — the file you complete (implement `solve()`)

Edit **`starter.py`** in each task folder and submit the four completed starter
files to the portal. Do not rename `solve()` or change its return contract.

## How each task works

Each `starter.py` exposes a single `solve()` function with a strict,
deterministic return contract documented in that task's `problem.md`. Read the
contract carefully — every metric key, column name, and threshold is checked.

- All randomness is seeded; given a correct implementation your output is
  reproducible.
- Tasks that use async engines wrap everything inside `solve()` — you call
  `solve()` normally; no event loop required from the caller.
- Keep the provided helper blocks (the derived-target code, the holdout split,
  the drift shift) **exactly as written** — they define the ground truth your
  output is checked against.

## Grading

Submissions are graded automatically against an independent reference
implementation that re-derives the expected values from the data. Marks are
awarded on **outcomes** (held-out ROC-AUC floors, minority-class recall lift,
correct model count, registry promotion, drift detected on the shifted batch but
not the clean one) — not on code style. A task's marks are awarded only when
every one of its checks passes.

The automated graders are withheld. Focus on meeting each `problem.md` contract
and clearing the stated performance target.

## Environment

```bash
# from the repo root
.venv/bin/python modules/mlfp03/assessment/task_1/starter.py
```

Requires the MLFP environment (`kailash-ml`, `polars`, `lightgbm`, `shap`) and
the `shared` package on the path — both already configured in the course `.venv`.
