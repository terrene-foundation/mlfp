# MLFP05 — Task 1: Handwritten Postcode Reader That Beats the Classical Baseline

**Weight**: 25 marks · **Data**: the 8×8 digits dataset bundled with scikit-learn
(no download) · **Outcomes assessed**: convolutional architectures, training a
real model to a measurable outcome (5.2)

## Scenario

A mail-sorting contractor reads handwritten Singapore postal codes. Their
legacy pipeline — a classical model on raw pixels — is the system to beat.
You are asked to deliver a **convolutional neural network**, trained from
scratch, that outperforms it on held-out mail.

The 8×8 grayscale digits in `sklearn.datasets.load_digits` stand in for the
mail scans: 1,797 images, pixel intensities 0–16. Nobody is handing you an
architecture or a training recipe. Choosing them is the task.

Your model will be graded on **held-out mail you never see**: the grader makes
its own stratified split with a fresh secret seed, trains its own classical
baseline on its own training slice, and scores **your returned model's
predictions** against **its own labels**. No number your code reports about
itself is read. An untrained network, a constant predictor, or a model whose
convolution is declared but never used in the forward pass will fail.

## Interface

```python
def solve() -> dict: ...
```

Returns a dict with at least:

- `"model"`: your trained model, a `torch.nn.Module` in eval-ready state.
- `"history"`: `{"train_loss": [...], "val_loss": [...]}` — the recorded
  training log (for your own debugging; the grader does not score it).

**Model contract** (the serving contract the sorter's loader expects):

- input: a `(N, 1, 8, 8)` float32 tensor, pixels scaled to `[0, 1]`
  (raw intensity divided by 16);
- output: a `(N, 10)` tensor of class logits (raw scores, not probabilities).

`starter.py` provides `load_digits_train()` — the training mail (a fixed,
documented split) — and the `solve()` signature. Work only in PyTorch.

## Acceptance criteria (what the grader measures)

| #   | Check                                                                      |
| --- | -------------------------------------------------------------------------- |
| 1   | A `torch.nn.Module` is returned (gate)                                     |
| 2   | The model honours the input/output contract on grader-built batches (gate) |
| 3   | At least one `Conv2d` actually fires during the forward pass               |
| 4   | Eval-mode forward is deterministic (same input twice → identical logits)   |
| 5   | Accuracy on the grader's held-out split ≥ 0.90                             |
| 6   | Accuracy beats the grader's majority-class baseline                        |
| 7   | Accuracy ≥ grader's own classical baseline − 0.05 (trained by the grader)  |
| 8   | Accuracy on a grader-built intensity-jittered variant ≥ 0.80               |
| 9   | Predictions on held-out mail span at least 5 distinct classes              |

Marks = 25 × (non-gate checks passed / 7). If a gate fails, the task scores 0.

## Rules

- Raw PyTorch (`torch.nn`); CPU only; no pretrained weights or downloads.
- Fix every seed you use. The grader replays runs with `--seed`.
- Train time inside `solve()` must stay under ~3 minutes on a laptop CPU.
- **Never train on held-out labels you construct yourself** — the grader's
  split is not yours, and self-reported metrics are ignored.
- Self-check: run `starter.py` end-to-end and read your own validation
  accuracy before submitting.
