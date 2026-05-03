# [kailash-ml 0.13] Add `training_diagnostics` engine — during-training observability for PyTorch

## Summary

kailash-ml ships `model_explainer` for **post-training** attribution (SHAP, LIME, feature importance) but has **zero during-training** observability. Every PyTorch trainer that wants gradient-flow per layer, dead-neuron tracking, or activation saturation has to hand-roll forward/backward hooks, polars aggregation, and Plotly dashboards. This issue proposes a new engine — `kailash_ml.engines.training_diagnostics` — to close that gap.

## Evidence — 1,679 LOC already written

The Terrene Foundation's ML Foundations for Professionals course (MLFP) has been carrying a `DLDiagnostics` helper at `shared/mlfp05/diagnostics.py` for its deep-learning module. It is 1,679 LOC of Apache-2.0 code, Polars-native, Plotly-based, zero LLM coupling, built entirely on `torch.nn.Module` forward/backward hooks.

The helper is organised around the four failure modes a practising DL engineer must recognise:

| Instrument       | What it tracks                                 | Failure mode surfaced        |
| ---------------- | ---------------------------------------------- | ---------------------------- |
| **Stethoscope**  | Loss-curve shape per epoch                     | Under-fit / over-fit / noise |
| **Blood Test**   | Gradient norms per layer per step              | Vanishing / exploding grads  |
| **X-Ray**        | Activation statistics (mean / var / %dead)     | Saturation / dying ReLU      |
| **Prescription** | Rule-based auto-diagnosis with actionable text | Next-step guidance           |

Public surface:

```python
from kailash_ml import DLDiagnostics  # proposed import path

with DLDiagnostics(model) as diag:
    diag.track_gradients()
    diag.track_activations()
    diag.track_dead_neurons()
    for epoch in range(epochs):
        for batch in dataloader:
            loss = train_step(model, batch)
            diag.record_batch(loss=loss.item(), lr=opt.param_groups[0]["lr"])
        diag.record_epoch(val_loss=evaluate(model, val_loader))
    diag.plot_training_dashboard().show()
    report = diag.report()  # dict with diagnosis + suggestions
```

Accessors and plots (all Polars DataFrames, all Plotly figures):

- `gradients_df()`, `activations_df()`, `dead_neurons_df()`, `batches_df()`, `epochs_df()`
- `plot_loss_curves()`, `plot_gradient_flow()`, `plot_activation_stats()`, `plot_dead_neurons()`, `plot_training_dashboard()`, `plot_lr_vs_loss()`, `plot_weight_distributions()`, `plot_gradient_norms()`
- `grad_cam(...)` — Grad-CAM for conv layers
- `lr_range_test(...)` — Smith-style LR finder

Three convenience entry points:

- `run_diagnostic_checkpoint(...)` — one-shot during-training snapshot
- `diagnose_classifier(...)` — classification-specific presets
- `diagnose_regressor(...)` — regression-specific presets

## Why upstream

1. **Every PyTorch trainer in the Foundation ecosystem wants this.** MLFP wrote it; any other Foundation ML project will reinvent it. Shipping in kailash-ml 0.13 ends the reinvention loop.
2. **Parallel to `model_explainer`.** Same polars/plotly/torch stack, same SDK aesthetic, same "one engine, clear API" shape. Natural sibling.
3. **Pure infrastructure, zero domain coupling.** The helper has no pedagogy, no exercise-specific logic, no lesson-tied datasets — it's already a platform primitive hiding in a course repo.
4. **Apache-2.0 already.** No license negotiation, no IP transfer needed.

## Proposed placement

- **Package**: `kailash-ml`
- **Module**: `kailash_ml.engines.training_diagnostics`
- **Import**: `from kailash_ml import DLDiagnostics` (top-level re-export in `__all__`)
- **Tier**: sibling of `training_pipeline` — trainers call it; diagnostics is an observer pattern.
- **Target release**: `kailash-ml 0.13`

## What stays local to MLFP

The course also carries `shared/mlfp05/ex_*.py` modules (~2,000 LOC) that wire kailash-ml engines (ExperimentTracker, ModelRegistry) into lesson-specific orchestration. Those are **EXTENDS** — course pedagogy on top of SDK primitives, appropriately local. Only `DLDiagnostics` is a **GAP** — platform code hiding in the course. The upstream migration is this one module.

## Migration plan (MLFP side, once upstream lands)

1. MLFP removes `shared/mlfp05/diagnostics.py` (1,679 LOC)
2. MLFP updates `shared/mlfp05/ex_*.py` to `from kailash_ml import DLDiagnostics`
3. MLFP lesson notebooks regenerate (automated via `scripts/generate_selfcontained_notebook.py`)
4. Net MLFP LOC reduction: **1,679 lines**

## Non-goals

- No Lightning-specific wiring — `DLDiagnostics` is backend-agnostic (works with raw `torch.optim` loops; a Lightning `Callback` adapter can land later if needed).
- No distributed-training support in v1 — single-device first, DDP/FSDP in a follow-up.
- No GPU-profiling overlap with `torch.profiler` — we track statistics, not kernel timings.

## Source

- MLFP course DISCOVERY: `courses/mlfp/workspaces/curriculum/journal/.pending/1776442000000-0-DISCOVERY.md` (will be committed + renumbered on /codify)
- Current implementation: [`shared/mlfp05/diagnostics.py` in mlfp](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp05/diagnostics.py) (1,679 LOC)
- Uses only: `torch`, `torch.nn`, `polars`, `plotly`, `numpy`, `logging` — all already in kailash-ml's dep tree.

## For discussion

- Accept the API surface above, or want naming changes (`DLDiagnostics` → `TrainingDiagnostics` to match engine-module convention)?
- Same-PR or separate-PR for the Lightning `Callback` adapter?
- Should `grad_cam` and `lr_range_test` split out into their own engines or stay bundled (both are training-adjacent and share the hook-management machinery)?
