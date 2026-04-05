# SDK Correctness Validation Report

**Date**: 2026-04-05
**Validated against**: kailash-py `main` branch (commit 0274a6f5)

---

## 1. Import Paths

### 1.1 kailash-ml top-level imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from kailash_ml import DataExplorer` | M1 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import PreprocessingPipeline` | M1 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import ModelVisualizer` | M1 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import FeatureStore` | M2 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import FeatureEngineer` | M2 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import ExperimentTracker` | M2 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import TrainingPipeline` | M3 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import HyperparameterSearch` | M3 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import ModelRegistry` | M3 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import AutoMLEngine` | M4 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import EnsembleEngine` | M4 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import DriftMonitor` | M4 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import InferenceServer` | M4 | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import OnnxBridge` | Engine ref | In lazy `__getattr__` map | **PASS** |
| `from kailash_ml import MLDashboard` | Engine ref | In lazy `__getattr__` map | **PASS** |

### 1.2 kailash-ml types (ModelSpec, EvalSpec)

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `ModelSpec` | M3 (used as `ModelSpec(...)`) | Defined in `kailash_ml.engines.training_pipeline`, NOT in `__init__.py` lazy map | **FAIL** |
| `EvalSpec` | M3 (used as `EvalSpec(...)`) | Defined in `kailash_ml.engines.training_pipeline`, NOT in `__init__.py` lazy map | **FAIL** |

**Impact**: `from kailash_ml import ModelSpec` will raise `AttributeError`. Students must use `from kailash_ml.engines.training_pipeline import ModelSpec, EvalSpec`.

**Remediation options**:
1. Fix the briefs to use the full import path.
2. Add `ModelSpec` and `EvalSpec` to the `_engine_map` in `kailash_ml/__init__.py` (SDK change).

### 1.3 kaizen-agents imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from kaizen_agents import Delegate` | M5 | Exported in `kaizen_agents/__init__.py` | **PASS** |
| `from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent` | M5 | File exists, class exists | **PASS** |
| `from kaizen_agents.agents.specialized.chain_of_thought import ChainOfThoughtAgent` | M5 | File exists, class exists | **PASS** |
| `from kaizen_agents.agents.specialized.react import ReActAgent` | M5 | File exists, class exists | **PASS** |
| `from kaizen_agents.agents.specialized.rag_research import RAGResearchAgent` | M5 | File exists, class exists | **PASS** |

### 1.4 kailash-kaizen imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from kaizen import Signature, InputField, OutputField` | M5 | All three in `kaizen/__init__.py` `__all__` | **PASS** |
| `from kaizen.core.base_agent import BaseAgent` | M5 | File exists, class `BaseAgent(Node)` at line 78 | **PASS** |

### 1.5 kailash-ml agents (all 6)

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from kailash_ml.agents.data_scientist import DataScientistAgent` | M5 | File exists, class at line 30 | **PASS** |
| `from kailash_ml.agents.model_selector import ModelSelectorAgent` | M5 | File exists, class at line 30 | **PASS** |
| `from kailash_ml.agents.feature_engineer import FeatureEngineerAgent` | M5 | File exists, class at line 30 | **PASS** |
| `from kailash_ml.agents.experiment_interpreter import ExperimentInterpreterAgent` | M5 | File exists, class at line 30 | **PASS** |
| `from kailash_ml.agents.drift_analyst import DriftAnalystAgent` | M5 | File exists, class at line 30 | **PASS** |
| `from kailash_ml.agents.retraining_decision import RetrainingDecisionAgent` | M5 | File exists, class at line 30 | **PASS** |

### 1.6 kailash-align imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from kailash_align import AlignmentConfig` | M6 | In lazy `__getattr__` map, in `__all__` | **PASS** |
| `from kailash_align import AlignmentPipeline` | M6 | In lazy `__getattr__` map, in `__all__` | **PASS** |
| `from kailash_align import AdapterRegistry` | M6 | In lazy `__getattr__` map, in `__all__` | **PASS** |

### 1.7 kailash-pact imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from pact import GovernanceEngine` | M6 | In `pact/__init__.py` `__all__` (re-exported from `kailash.trust.pact`) | **PASS** |
| `from pact import GovernanceContext` | M6 | In `pact/__init__.py` `__all__` (re-exported from `kailash.trust.pact`) | **PASS** |
| `from pact import PactGovernedAgent` | M6 | In `pact/__init__.py` `__all__` (re-exported from `kailash.trust.pact`) | **PASS** |
| `from pact import Address` | M6 | In `pact/__init__.py` `__all__` | **PASS** |
| `from pact import RoleEnvelope` | M6 | In `pact/__init__.py` `__all__` | **PASS** |
| `from pact import TaskEnvelope` | M6 | In `pact/__init__.py` `__all__` | **PASS** |
| `from pact import compile_org` | M6 | In `pact/__init__.py` `__all__` | **PASS** |
| `from pact import load_org_yaml` | M6 | In `pact/__init__.py` `__all__` | **PASS** |

### 1.8 kailash-ml RL imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from kailash_ml.rl.trainer import RLTrainer` | M6 | File exists, class `RLTrainer` at line 94 | **PASS** |

### 1.9 Core SDK imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from kailash.workflow.builder import WorkflowBuilder` | M3 | File exists, class `WorkflowBuilder` at line 20 | **PASS** |
| `from kailash.runtime import LocalRuntime` | M3 | Exported in `kailash/runtime/__init__.py` | **PASS** |

### 1.10 DataFlow imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from dataflow import DataFlow` | M3 | Exported in `dataflow/__init__.py` | **PASS** |

### 1.11 Nexus imports

| Import | Brief | SDK Status | Verdict |
|--------|-------|------------|---------|
| `from nexus import Nexus` | M4, M6 | Exported in `nexus/__init__.py` | **PASS** |

---

## 2. Engine Names

All 15 engine names referenced in course materials verified against actual class definitions in the SDK source.

| Engine Name | Module File | Class Found | Verdict |
|-------------|-------------|-------------|---------|
| DataExplorer | `engines/data_explorer.py` | Line exists | **PASS** |
| PreprocessingPipeline | `engines/preprocessing.py` | Line 104 | **PASS** |
| ModelVisualizer | `engines/model_visualizer.py` | Line 47 | **PASS** |
| FeatureStore | `engines/feature_store.py` | Line 37 | **PASS** |
| FeatureEngineer | `engines/feature_engineer.py` | Line 168 | **PASS** |
| ExperimentTracker | `engines/experiment_tracker.py` | Line 377 | **PASS** |
| TrainingPipeline | `engines/training_pipeline.py` | Line 177 | **PASS** |
| ModelRegistry | `engines/model_registry.py` | Line 398 | **PASS** |
| HyperparameterSearch | `engines/hyperparameter_search.py` | Line 244 | **PASS** |
| AutoMLEngine | `engines/automl_engine.py` | Line 298 | **PASS** |
| EnsembleEngine | `engines/ensemble.py` | Line 245 | **PASS** |
| DriftMonitor | `engines/drift_monitor.py` | Line 363 | **PASS** |
| InferenceServer | `engines/inference_server.py` | Line 127 | **PASS** |
| RLTrainer | `rl/trainer.py` | Line 94 | **PASS** |
| MLDashboard | `dashboard/__init__.py` | Line 26 | **PASS** |
| OnnxBridge | `bridge/onnx_bridge.py` | Line 131 | **PASS** |

---

## 3. API Patterns

### 3.1 Runtime execution pattern

| Pattern | Brief | SDK Status | Verdict |
|---------|-------|------------|---------|
| `runtime.execute(workflow.build())` | M3 | Correct. `LocalRuntime.execute()` takes built workflow. | **PASS** |
| `results, run_id = runtime.execute(...)` | M3 | Correct. Returns `(results, run_id)` tuple. | **PASS** |

### 3.2 DataFlow Express API

| Pattern | Brief | SDK Status | Verdict |
|---------|-------|------------|---------|
| `db.express.create("Model", {...})` | M3 | Correct. Express CRUD API. | **PASS** |

### 3.3 ModelSpec parameters

| Pattern | Brief | SDK Status | Verdict |
|---------|-------|------------|---------|
| `ModelSpec(model_class="lightgbm.LGBMClassifier", hyperparameters={...})` | M3 | `model_class: str` at line 56, `hyperparameters: dict` at line 57 | **PASS** |

### 3.4 EvalSpec parameters

| Pattern | Brief | SDK Status | Verdict |
|---------|-------|------------|---------|
| `EvalSpec(metrics=["accuracy", "f1", "auc_roc"], split_ratio=0.2)` | M3 | `metrics` field exists. **`split_ratio` does NOT exist** -- the actual field is `test_size: float = 0.2` | **FAIL** |

**Impact**: Code using `EvalSpec(split_ratio=0.2)` will raise `TypeError: unexpected keyword argument 'split_ratio'`. Must use `test_size=0.2`.

### 3.5 GovernanceContext(frozen=True)

| Pattern | Brief | SDK Status | Verdict |
|---------|-------|------------|---------|
| `GovernanceContext(frozen=True)` | M6 | `GovernanceContext` is `@dataclass(frozen=True)` -- ALL instances are frozen by the class definition. `frozen` is NOT a constructor parameter. | **FAIL** |

**Impact**: `GovernanceContext(frozen=True)` will raise `TypeError: __init__() got an unexpected keyword argument 'frozen'`. The class requires its actual fields: `role_address`, `posture`, `effective_envelope`, `clearance`, `effective_clearance_level`, `allowed_actions`, `compartments`, `org_id`, `created_at`.

**Remediation**: Fix the brief to show correct construction via `GovernanceEngine.create_context(role_address)` or by passing the required fields. The conceptual point (contexts are immutable) is correct but the code sample is wrong.

---

## 4. Package Versions

### 4.1 ASCENT pyproject.toml constraints vs SDK current versions

| Package | ASCENT Constraint | SDK Current | Compatible | Verdict |
|---------|----------------|-------------|------------|---------|
| `kailash` | `>=1.0` | `2.5.1` | Yes | **PASS** |
| `kailash-ml` | `>=0.4.0` | `0.4.0` | Yes (exact) | **PASS** |
| `kailash-dataflow` | `>=1.7.0` | `1.7.1` | Yes | **PASS** |
| `kailash-nexus` | `>=1.9.0` | `1.9.0` | Yes (exact) | **PASS** |
| `kailash-kaizen` | `>=2.5.0` | `2.5.0` | Yes (exact) | **PASS** |
| `kaizen-agents` | `>=0.6.0` | `0.6.0` | Yes (exact) | **PASS** |
| `kailash-pact` | `>=0.7.2` | `0.7.2` | Yes (exact) | **PASS** |
| `kailash-align` | `>=0.2.1` | `0.2.1` | Yes (exact) | **PASS** |

All version constraints are satisfied. Most pin to the current version as a minimum, which is tight but correct for a training course that depends on specific APIs.

---

## Summary

| Category | Total Checks | PASS | FAIL |
|----------|-------------|------|------|
| Import Paths | 35 | 33 | **2** |
| Engine Names | 16 | 16 | 0 |
| API Patterns | 5 | 3 | **2** |
| Package Versions | 8 | 8 | 0 |
| **Total** | **64** | **60** | **4** |

## Failures Requiring Action

### FAIL-1: ModelSpec / EvalSpec not importable from kailash_ml top-level

- **Location**: M3 brief, Key Patterns section
- **Problem**: `ModelSpec` and `EvalSpec` are not in `kailash_ml.__init__.py`'s lazy-load map
- **Code that breaks**: `from kailash_ml import ModelSpec, EvalSpec`
- **Fix (option A -- brief)**: Change imports in M3 brief to `from kailash_ml.engines.training_pipeline import ModelSpec, EvalSpec`
- **Fix (option B -- SDK)**: Add `"ModelSpec": "kailash_ml.engines.training_pipeline"` and `"EvalSpec": "kailash_ml.engines.training_pipeline"` to the `_engine_map` in `kailash_ml/__init__.py`

### FAIL-2: EvalSpec `split_ratio` parameter does not exist

- **Location**: M3 brief, Key Patterns code block (line 47)
- **Problem**: The actual field name is `test_size`, not `split_ratio`
- **Code that breaks**: `EvalSpec(metrics=["accuracy", "f1", "auc_roc"], split_ratio=0.2)`
- **Fix**: Change `split_ratio=0.2` to `test_size=0.2` in the M3 brief

### FAIL-3: GovernanceContext(frozen=True) is incorrect API usage

- **Location**: M6 brief, Exercise 6.4 description (line 59)
- **Problem**: `GovernanceContext` is `@dataclass(frozen=True)` at the class level. `frozen` is not a constructor parameter. The class requires 9 specific fields.
- **Code that breaks**: `GovernanceContext(frozen=True)`
- **Fix**: Replace with correct construction pattern. Example:
  ```python
  # Correct: GovernanceContext is always frozen (enforced by @dataclass(frozen=True))
  # Create via GovernanceEngine:
  ctx = engine.create_context(role_address="org/dept/role")
  # ctx is automatically immutable -- assignment raises FrozenInstanceError
  ```
  The exercise text should say "Demonstrate that `GovernanceContext` is immutable (frozen dataclass) -- agents cannot modify their own governance" rather than implying `frozen=True` is a parameter.

### FAIL-4: ModelSpec / EvalSpec not exported (same root cause as FAIL-1)

This is the same root cause as FAIL-1, but specifically affects the code pattern shown in M3:
```python
model_spec = ModelSpec(model_class="lightgbm.LGBMClassifier", hyperparameters={...})
eval_spec = EvalSpec(metrics=["accuracy", "f1", "auc_roc"], split_ratio=0.2)
```
Both the import path AND the `split_ratio` parameter name are wrong, making this code block doubly broken.

---

## Recommended Fix Priority

1. **FAIL-2** (split_ratio) -- single word change in M3 brief, highest student-impact (runtime TypeError)
2. **FAIL-3** (GovernanceContext) -- rewrite M6 exercise 6.4 description to show correct construction
3. **FAIL-1** (ModelSpec/EvalSpec import) -- either fix brief OR add to SDK exports (SDK fix is preferable since these are core user-facing types)
