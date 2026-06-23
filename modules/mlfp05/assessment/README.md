# MLFP05 — Module Assessment: Deep Learning (Vision & Sequences)

Four practical, auto-graded coding tasks covering the four pillars of Module 5:
**autoencoders, CNNs, RNNs/sequences, and transformers**. No multiple choice. Every
task builds and trains a real PyTorch model and is graded on an **outcome** (AUC,
test accuracy, MSE-below-baseline). Each grader imports your submission, re-derives
its data independently, re-runs your model on it (so hand-tuned output arrays are
caught), and prints a JSON report.

## No GPU required

Every task is **CPU-shaped**: tiny models, small or synthetic data, few epochs, fixed
seeds. Each reference solution runs to completion on a laptop CPU in **well under 60
seconds** (most in 5–25s). You can complete and pass all four tasks without any GPU.
No large pretrained backbones are downloaded (no ResNet/BERT) — where a topic was
GPU-heavy in the exercises, it is adapted to a small-from-scratch equivalent that
tests the same skill (documented in each `problem.md`).

## Tasks

| Task | Weight | Difficulty | Topic            | Dataset                                 | Skill graded                                       |
| ---- | ------ | ---------- | ---------------- | --------------------------------------- | -------------------------------------------------- |
| 1    | 25%    | Hard       | Autoencoders     | Synthetic sensor telemetry (in-process) | Undercomplete AE anomaly detection, ROC-AUC ≥ 0.90 |
| 2    | 25%    | Hard       | CNNs             | `sklearn` 8×8 digits (bundled)          | CNN from scratch, test accuracy ≥ 0.90             |
| 3    | 25%    | Hard       | RNNs / sequences | Synthetic AR(2) series (in-process)     | GRU forecast beats naive last-value (≤ 0.97× MSE)  |
| 4    | 25%    | Hard       | Transformers     | AG News slice (bundled parquet)         | Tiny transformer text classifier, accuracy ≥ 0.72  |

**Total: 25 + 25 + 25 + 25 = 100 marks.** Each task is worth 25 marks and passes only
when **every** check in its `grader.py` returns `true`.

### Why some datasets are synthetic

- **Task 1** plants off-manifold anomalies into a low-rank healthy signal so a
  bottleneck AE has something real to separate (deterministic, no download).
- **Task 3** uses a damped AR(2) + seasonal series because **real equity returns are
  a random walk that no model can beat** — grading "beat the baseline" on a random
  walk would be impossible. AR(2) has genuine autocorrelation a GRU can exploit.

Tasks 2 and 4 use **bundled** real data committed to the repo (`sklearn` digits ship
inside scikit-learn; AG News parquet lives under `data/mlfp05/`).

## Each task directory contains

- `problem.md` — scenario, weight, difficulty, dataset source, exact return
  contract, performance target, visible sanity check, grading checklist, rules, and
  any CPU adaptation notes.
- `starter.py` — light scaffold with numbered `# TODO` markers. The placeholder
  **fails** grading. This is the file you complete and submit.
- `solution.py` — instructor reference that passes every check (**withheld** from the
  student portal).
- `grader.py` — automated grader (**withheld** from the student portal).

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
  "checks": { "returns_dict": true, "auc_at_least_0p90": true, "...": true },
  "total": 8,
  "max": 8
}
```

A task is **passed** when `total == max` (every check `true`).

## Exam conditions

- **Duration**: 3 hours.
- **Open-book, no-AI**: you may consult the Module 5 exercises, the kailash-ml docs,
  and PyTorch docs. You may **not** use AI assistants — the graders measure outcomes
  on data they re-derive, and an AI-generated skeleton that does not train a real
  model will not pass.
- **Submit your completed `starter.py` files to the portal.** Graders are withheld;
  your submissions are run against them.

## Rules

- **No GPU** — CPU only; keep models tiny and seeds fixed.
- Raw **PyTorch** (`torch.nn`) is allowed throughout — Module 5 is the deep-learning
  module and its exercises build models directly in `torch.nn`.
- **No large pretrained backbones / downloads** (no ResNet, no BERT, no HuggingFace
  weights) — build models from scratch.
- **Polars** for any tabular/parquet work — **no pandas**.
- Fix all seeds (`torch.manual_seed`) for reproducibility. Where exact reproduction
  is impossible, tasks grade on outcome thresholds with margin.
- Never train on the held-out test labels. No hardcoded API keys or model names.
- **AI-resilient**: each grader re-derives its data, re-runs your returned model, and
  checks the model itself produces the claimed result — a faked output array fails.
