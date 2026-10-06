# MLFP05 — End-of-Module Assessment: Deep Learning (Vision, Sequences & Deployment)

Four practical coding tasks covering the module's working skills: training a
CNN that beats a classical baseline, reading failing training runs with the
DLDiagnostics toolkit, training a recurrent forecaster that beats the naive
forecast, and shipping a trained model as an ONNX artefact. There is no
multiple choice.

The tasks state **goals, data, constraints and acceptance criteria**. They do
not give architectures or training recipes. Choosing them is part of every
task.

**Duration**: 3 hours · **Total**: 100 marks · **Open book**: documentation is
allowed; AI assistants are **not** allowed.

## Tasks

| Task | Marks | Data                                       | What it assesses                                                                                          | Spec lessons |
| ---- | ----- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------ |
| 1    | 25    | 8×8 digits (bundled with scikit-learn)     | Build and train a CNN from scratch that beats the grader's own classical baseline on grader-held mail     | 5.2          |
| 2    | 25    | none — the grader hands you live models    | Triage training runs with kailash-ml DLDiagnostics: dead neurons, gradient flow, loss trend               | 5.1–5.4      |
| 3    | 25    | synthetic load series (generator provided) | Train a recurrent forecaster that beats the naive "tomorrow = today" forecast on series you never see     | 5.3          |
| 4    | 25    | 8×8 digits (bundled)                       | Train a classifier and ship it as an ONNX artefact via OnnxBridge; parity + accuracy scored by the grader | 5.2, 5.7     |

Each task directory contains:

- `problem.md`: the scenario, the interface, the acceptance criteria and the
  rules;
- `starter.py`: data loading, the function signatures and a local runner. You
  complete it and submit it.

Instructors also hold `solution.py` (the reference), `grader.py`, and the
shared `grading_harness.py`. These are not given to students.

## No GPU required

Every task is **CPU-shaped**: tiny models, small or bundled data, few epochs,
fixed seeds. Each reference solution runs in well under a couple of minutes on
a laptop CPU, and each grader in about the same.

## How grading works

Every grader measures **outcomes on ground truth you cannot influence**. No
number that a submission reports about itself is trusted. In particular:

- Task 1 draws its own stratified split of the digits with a fresh secret
  seed, trains its own logistic-regression baseline, and scores **your
  returned model's** predictions against its own labels — plus a forward-hook
  check that a `Conv2d` actually fires, and an intensity-jittered variant for
  generalisation.
- Task 2 builds the ward itself: fresh models with planted pathologies (a
  zeroed layer, a deep tiny-init tanh stack, a diverging recorded loss).
  Your labels are compared against the planted truth.
- Task 3 regenerates the load series with fresh secret seeds and computes the
  naive and mean forecasts itself, then runs your returned model.
- Task 4 loads your `.onnx` artefact with onnxruntime itself, checks
  numerical parity with your torch model on grader-built batches, and scores
  accuracy against grader-held labels.

Marks are awarded per check: weight × (checks passed / checks). Tasks 1, 3
and 4 have **gates** (a returned model honouring the input/output contract);
if a gate fails, the task scores 0. An untrained model, a constant predictor,
an echoed input or a hard-coded answer fails every task.

Run a grader (instructors):

```bash
python modules/mlfp05/assessment/task_1/grader.py path/to/submission.py
python modules/mlfp05/assessment/task_1/grader.py path/to/submission.py --seed 123  # replay
```

It prints a JSON report with each check, the marks, and diagnostic notes.

## How to work

1. Read `task_N/problem.md` in full.
2. Implement `solve()` / `diagnose_model()` in `task_N/starter.py`. Run it —
   the runner trains your model and prints a summary on documented dev data.
3. Submit your completed `starter.py` files. Do not rename the functions or
   change their contracts.

## Rules

- Raw PyTorch (`torch.nn`) throughout — Module 5 is the deep-learning module.
- **No pretrained weights or downloads**; CPU only; fix your seeds.
- Task 2 diagnoses with the kailash-ml DLDiagnostics instruments
  (`kailash_ml.diagnostics.run_diagnostic_checkpoint`); Task 4 exports with
  the kailash-ml `OnnxBridge` (`framework="torch"`, a `sample_input` in the
  serving shape; `export` returns a result object — read `.success`).
- Graders replay: the same `--seed` reproduces the same grading run.
