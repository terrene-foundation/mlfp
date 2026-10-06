# MLFP05 — Task 2: Triage a Ward of Failing Training Runs

**Weight**: 25 marks · **Data**: none — the grader hands you live models
· **Outcomes assessed**: diagnostic-driven iteration with the kailash-ml
DLDiagnostics toolkit — gradient flow, dead neurons, loss trend

## Scenario

You are the on-call ML engineer. Colleagues hand you models from overnight
training runs — each model arrives with its data loader, its loss function,
and the loss history the run recorded. Some runs are fine. Some are not.
Your job is an automated triage function that reads each patient and names
what is wrong.

You are given no list of which run has which problem. The grader builds the
ward itself — fresh models, fresh data, fresh seeds, with pathologies planted
by the grader — and checks your label for every patient against the planted
truth. A function that always answers "healthy" passes only the healthy
patients and fails the task.

## Interface

```python
def diagnose_model(
    model: torch.nn.Module,
    loader,                       # yields (x_batch, y_batch)
    loss_fn,                      # loss_fn(model, (x_batch, y_batch)) -> scalar
    *,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
) -> str: ...
```

Return exactly one of:

| Label                   | Meaning                                                                                                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"healthy"`             | Nothing is wrong: gradients flow, every layer activates, loss trended down.                                                                                           |
| `"dead_neurons"`        | A layer's units never activate — its output is all zeros — while the rest of the network is alive.                                                                    |
| `"vanishing_gradients"` | Gradients shrink by orders of magnitude from the output layers back to the input layers. The loss may have plateaued as a _symptom_; the disease is in the gradients. |
| `"diverging_loss"`      | The recorded loss **increases** across the run (the model itself can be intact — the run went wrong, e.g. the learning rate).                                         |

## Constraints

- Use the kailash-ml DLDiagnostics instruments
  (`kailash_ml.diagnostics.run_diagnostic_checkpoint` or a `DLDiagnostics`
  you drive yourself) to read each patient. Naming a pathology you did not
  measure is not a diagnosis.
- Return the label string only. Deterministic: the same patient twice must
  get the same label.
- Each call must finish in seconds — read a few batches, not the whole set.

## Acceptance criteria (what the grader measures)

The grader builds ten patients with a fresh secret seed — healthy runs,
runs with a dead layer, runs with vanishing gradients, and runs whose
recorded loss diverged — and calls your `diagnose_model` on each.

| #   | Check                                                            |
| --- | ---------------------------------------------------------------- |
| 1   | Every returned label is one of the four valid labels             |
| 2   | All healthy patients labelled `"healthy"`                        |
| 3   | All dead-layer patients labelled `"dead_neurons"`                |
| 4   | All vanishing-gradient patients labelled `"vanishing_gradients"` |
| 5   | All diverging-loss patients labelled `"diverging_loss"`          |
| 6   | Your labels across the ward use at least 3 distinct values       |
| 7   | The same patient examined twice gets the same label              |

Marks = 25 × (checks passed / 7).

## Rules

- Raw PyTorch for reading the models; no training inside `diagnose_model`.
- Do not mutate the handed model's parameters.
- Self-check: `starter.py` builds one example patient per class so you can
  watch your function work — those are _examples_, not the grader's ward.
