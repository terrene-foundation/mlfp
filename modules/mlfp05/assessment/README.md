# MLFP05 — Assessment: Deep Learning on Real Data

Three practical coding tasks. No multiple choice. Every task loads a real,
publicly-available dataset, trains a PyTorch model, and is graded on an
**outcome** (test accuracy, ONNX parity, MSE-below-baseline). The grader imports
your submission, runs it on hidden data, and prints a JSON report.

## Tasks

| Task | Weight | Difficulty | Dataset           | Skills                                                          |
| ---- | ------ | ---------- | ----------------- | --------------------------------------------------------------- |
| 1    | 20%    | Easy       | Fashion-MNIST     | Build + train a small CNN, beat a 85% test-accuracy threshold   |
| 2    | 35%    | Medium     | CIFAR-10          | Transfer learning with frozen ResNet-18, export to ONNX, parity |
| 3    | 45%    | Hard       | Straits Times Idx | Walk-forward LSTM/GRU regression, beat a naive-baseline MSE     |

Each task directory contains:

- `problem.md` — statement, dataset location, required function signature,
  performance target, visible sanity-check
- `starter.py` — skeleton with `# TODO` markers
- `solution.py` — instructor reference that passes every check
- `grader.py` — automated grader that loads your file and runs hidden checks

## How to run

```bash
cd modules/mlfp05/assessment/task_1
uv run python grader.py starter.py     # grade your attempt
uv run python grader.py solution.py    # verify the reference passes
```

Exit code `0` = passed, `1` = failed. The grader prints a JSON report:

```json
{
  "passed": true,
  "checks": {"trains_model": true, "test_accuracy_above_threshold": true, ...},
  "metrics": {"test_accuracy": 0.873, "model_parameters": 120234},
  "total": 5,
  "max": 5
}
```

## Rubric

| Task   | Weight | Pass threshold                  |
| ------ | ------ | ------------------------------- |
| Task 1 | 20%    | test accuracy ≥ 0.85            |
| Task 2 | 35%    | test acc ≥ 0.55 AND ONNX parity |
| Task 3 | 45%    | val MSE strictly below naive    |

A task is **passed** when every check in `grader.py` returns `true`.

## Rules

- Use PyTorch (`torch.nn`) for all model code — kailash-ml engines may wrap it.
- Use `polars` only — no pandas.
- Data files are provided by `torchvision` / `yfinance` on first run and cached
  under `data/mlfp05/`. The grader reuses the same cache.
- No internet access needed at grading time after the first run populates the
  cache; keep the exercise reproducible with a fixed seed.
- Graders cap wall time per task at 6 minutes on a 2024 laptop CPU. If you use
  a much larger model you may miss that budget.
- AI-resilient: the grader measures **outcomes on real data**, not code
  patterns. A language-model-generated skeleton that does not train a real
  model will not pass.
