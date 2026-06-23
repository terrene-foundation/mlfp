# Engine-First Audit — kailash-ml 1.1.1 stack

## Status: MLFP is on Layer 1 (primitives), NOT Layer 0 (`km.*` engine-first)

**Current km.\* surface usage** in MLFP (real call sites, not false-positives):

| Function                                                        | Sites   | Where                                                       |
| --------------------------------------------------------------- | ------- | ----------------------------------------------------------- |
| `km.device()`                                                   | several | `shared/kailash_helpers.py`, M5 prelude — backend detection |
| `km.train()`                                                    | 1       | M5 ex_0 destination-first prelude                           |
| `km.register()`                                                 | 1       | M5 ex_0 destination-first prelude                           |
| `km.diagnose()`                                                 | **0**   | not used anywhere                                           |
| `km.dashboard()`                                                | **0**   | not used                                                    |
| `km.serve()`                                                    | **0**   | not used                                                    |
| `km.watch()`                                                    | **0**   | not used                                                    |
| `km.track()`                                                    | **0**   | not used                                                    |
| `km.lineage()`, `km.reproduce()`, `km.resume()`, `km.autolog()` | **0**   | not used                                                    |

**Primitive engine usage** (Layer 1 — what MLFP actually does):

| Engine                    | Instantiations | Wrapper available         |
| ------------------------- | -------------- | ------------------------- |
| `ExperimentTracker(conn)` | **22**         | `km.track(...)`           |
| `TrainingPipeline(...)`   | **18**         | `km.train(...)`           |
| `ModelRegistry(conn)`     | **13**         | `km.register(...)`        |
| `MLEngine()`              | **7**          | `km.train(...)`           |
| `InferenceServer(...)`    | **6**          | `km.serve(...)`           |
| `FeatureStore`            | 2              | (no km.\* shortcut)       |
| `DriftMonitor`            | 2              | `km.watch(...)`           |
| `AutoMLEngine`            | 1              | `km.train(family='auto')` |

## Pedagogical filter — what to keep on Layer 1

MLFP teaches by exposing primitives. **Keep** primitive layer for the teaching surfaces:

- **M3 ex_3 `fit_and_evaluate`** — explicit `estimator.fit(X, y)` is the lesson (model-family comparison). Students learn what each estimator does.
- **M3 ex_6 `train_credit_model`** — explicit LightGBM with `scale_pos_weight` for imbalanced data; SHAP needs the raw model. Pedagogical choice.
- **M3 ex_8 `train_calibrated_model`** — `CalibratedClassifierCV(base, method='isotonic', cv=5)` is the lesson (probability calibration). km.train would hide it.
- **M5 lesson bodies** — students write torch loops by hand. M5 ex_0 prelude says "this is the destination" (`km.train`); ex_1-8 walk the journey.

These are explicit choices, not drift.

## What's actually drifted from engine-first

### 1. `setup_engines()` boilerplate (highest-impact migration target)

7 of 8 M5 helpers ship a 22-32 LOC `_setup_engines()` block:

```python
async def _setup_engines():
    conn = ConnectionManager("sqlite:///mlfp05_autoencoders.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(name="m5_autoencoders", ...)
    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
    return conn, tracker, exp_name, registry, has_registry

def setup_engines() -> tuple:
    return asyncio.run(_setup_engines())
```

**Replace with**:

```python
def setup_engines() -> tuple:
    """km.track wraps ConnectionManager + ExperimentTracker + ModelRegistry."""
    return km.track(experiment="m5_autoencoders", db="sqlite:///mlfp05_autoencoders.db")
```

LOC reduction: ~22 × 8 = **176 LOC** across M5 helpers. Same primitives under the hood; teaching value intact (km.track docstring shows what it composes).

Same pattern in M3 (8 helpers) and M4 (5 helpers) — additional ~250 LOC.

### 2. `km.diagnose` at lesson tails (highest pedagogical leverage)

Every M5 lesson ends with manual evaluation. After students finish their hand-rolled torch loop, the lesson should demonstrate the SDK with one extra cell:

```python
# Lesson 5.1, after the manual training loop completes
report = km.diagnose(model, kind='auto', data=val_loader)
report.show()  # Plotly auto-dashboard — exact same instrument students just saw
```

This closes the destination-first loop: ex_0 shows `km.train` as the destination → ex_1-8 walk the journey → each lesson tail shows `km.diagnose` as the production observability primitive.

**Currently**: 0 lessons use km.diagnose.

### 3. `km.dashboard` for module wrap-ups

M5 reflection blocks compare runs across all lessons. Currently each lesson hand-rolls comparison plots. `km.dashboard()` shows everything tracked:

```python
# M5 wrap-up (after all 8 lessons)
km.dashboard(db_url="sqlite:///mlfp05_autoencoders.db", port=5000).show()
```

**Currently**: 0 modules use km.dashboard.

### 4. Import path inconsistency

Top-level imports (preferred 1.0 convention) vs deep-path imports — mixed:

```python
# Mix of patterns across MLFP:
from kailash_ml import ExperimentTracker            # ← preferred
from kailash_ml.engines.experiment_tracker import ExperimentTracker  # ← cleanup target
from kailash_ml.engines.automl_engine import AutoMLConfig, AutoMLEngine  # ← cleanup target
from kailash_ml.engines.drift_monitor import DriftSpec  # ← cleanup target
from kailash_ml.engines.hyperparameter_search import (...)  # ← cleanup target
```

Top-level should work for all canonical engines. Verify with `from kailash_ml import X` for each.

### 5. M5 Lesson 5.0 prelude (already aligned post-migration)

`modules/mlfp05/solutions/ex_0/00_destination_first.py` already uses km.train + km.register + km.MLEngine post the kailash-ml 1.0 async migration we did this session. **No drift here**.

## Recommended migration order (impact-ranked)

1. **`km.diagnose` at every M5 lesson tail** (8 lessons × 1 cell each ≈ +24 LOC, +~200 LOC pedagogical value) — highest pedagogical win, smallest code change. Most visible to students.
2. **`setup_engines() → km.track`** (~250-400 LOC reduction across M3/M4/M5) — biggest LOC win, no pedagogical loss.
3. **`km.dashboard` in M5/M3/M4 wrap-ups** (3 modules × 1 cell ≈ +9 LOC) — highest "kailash-as-platform" framing.
4. **Import path cleanup** (cosmetic, ~30 sites) — last; doesn't change behavior.
5. **Selective `km.train`/`km.serve`/`km.watch` adoption** in M3 helpers where pedagogy isn't lost (deferred — requires per-lesson judgment).

## What NOT to migrate

- M5 ex_1-8 lesson bodies (manual torch loops are the curriculum)
- M3 ex_3 fit_and_evaluate (explicit sklearn primitives are the curriculum)
- M3 ex_6 train_credit_model (LightGBM hyperparameter exposure is the lesson)
- M3 ex_8 train_calibrated_model (CalibratedClassifierCV is the lesson)
- DLDiagnostics class itself (already migrated to `kailash_ml.diagnostics` upstream — we just import it)

## Verdict

MLFP is structurally sound on Layer 1 for pedagogy but has **untapped Layer 0 surface**. The two highest-impact changes are zero-pedagogical-cost:

- **Adopt `km.diagnose` at lesson tails** (closes the destination-first loop)
- **Replace `setup_engines()` boilerplate with `km.track`** (~250-400 LOC reduction, same primitives underneath)

Estimated total work: ~3 sessions to migrate, regenerate notebooks via existing generator pipeline, and verify against the 90+ M5 lessons + ~40 M3/M4 lessons that consume `setup_engines`.
