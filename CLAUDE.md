# ASCENT — Professional Certificate in Machine Learning

Official Terrene Open Academy programme powered by the Kailash Python SDK, maintained by the Terrene Foundation. Open-source course materials from zero Python to masters-level ML engineering.

## Absolute Directives

### 0. Foundation Independence

Kailash Python SDK is a **Terrene Foundation project** (Singapore CLG). No commercial references. Do not compare with, reference, or design against any proprietary product. See `rules/independence.md`.

### 1. Framework-First

All ML content uses Kailash frameworks. Never raw sklearn/pandas/PyTorch without the Kailash wrapper. Before writing custom code, check if a kailash-ml engine handles it: `DataExplorer`, `PreprocessingPipeline`, `FeatureStore`, `FeatureEngineer`, `TrainingPipeline`, `AutoMLEngine`, `HyperparameterSearch`, `EnsembleEngine`, `ModelRegistry`, `InferenceServer`, `DriftMonitor`, `ExperimentTracker`, `ModelVisualizer`.

### 2. Polars-Native

No pandas in any exercise. kailash-ml is polars-native — students learn polars from Module 1. A pandas-to-polars cheatsheet lives in `docs/polars-cheatsheet.md`. Every data operation uses `polars` or `kailash_ml` APIs.

### 3. Three-Format Consistency

Every exercise must exist in three formats:

| Format  | Location                                | Data Loading         |
| ------- | --------------------------------------- | -------------------- |
| Local   | `modules/ascentNN/local/ex_N.py`        | `shared.data_loader` |
| Jupyter | `modules/ascentNN/notebooks/ex_N.ipynb` | `shared.data_loader` |
| Colab   | `modules/ascentNN/colab/ex_N.ipynb`     | Drive mount + gdown  |

Data loading via `shared/data_loader.py`. Setup differs by format; exercise code is identical.

### 4. Solution-First Authoring

Write complete solutions first in `modules/ascentNN/solutions/`, then strip to create exercises. The exercise-designer agent automates this. Solutions must always run correctly end-to-end.

### 5. Progressive Disclosure

| Module | Scaffolding | Code Provided |
| ------ | ----------- | ------------- |
| M1     | Heavy       | ~70%          |
| M2     | Moderate+   | ~60%          |
| M3     | Moderate    | ~50%          |
| M4     | Light+      | ~40%          |
| M5     | Light       | ~30%          |
| M6     | Minimal     | ~20%          |

Each module reduces hand-holding. By M6, students write most code from documentation.

## Course Structure

| Module   | Title                                   | Kailash Frameworks                                                                                          |
| -------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| ASCENT01 | Python & Data Fluency                   | kailash-ml: DataExplorer, PreprocessingPipeline, ModelVisualizer                                            |
| ASCENT02 | Statistical Foundations                 | kailash-ml: ExperimentTracker, ModelVisualizer                                                              |
| ASCENT03 | Feature Engineering & Experiment Design | kailash-ml: FeatureEngineer, FeatureStore, ExperimentTracker                                                |
| ASCENT04 | Supervised ML                           | kailash-ml: TrainingPipeline, ModelSpec, EvalSpec, HyperparameterSearch, ModelRegistry                      |
| ASCENT05 | ML Engineering & Production             | Core SDK WorkflowBuilder, DataFlow, kailash-ml: EnsembleEngine, ModelVisualizer                             |
| ASCENT06 | Unsupervised ML & Pattern Discovery     | kailash-ml: AutoMLEngine, DriftMonitor                                                                      |
| ASCENT07 | Deep Learning                           | kailash-ml: OnnxBridge, InferenceServer                                                                     |
| ASCENT08 | NLP & Transformers                      | kailash-ml: ModelVisualizer, AutoMLEngine (text)                                                            |
| ASCENT09 | LLMs, AI Agents & RAG                   | Kaizen: Delegate, BaseAgent, Signature; Nexus, MCP                                                          |
| ASCENT10 | Alignment, RL & Governance              | Align: AlignmentPipeline, AdapterRegistry; kailash-ml: RLTrainer; PACT: GovernanceEngine, PactGovernedAgent |

## Kailash Platform

| Framework | Purpose                                | Install                        |
| --------- | -------------------------------------- | ------------------------------ |
| Core SDK  | Workflow orchestration, 140+ nodes     | `pip install kailash`          |
| DataFlow  | Zero-config database operations        | `pip install kailash-dataflow` |
| Nexus     | Multi-channel deployment (API+CLI+MCP) | `pip install kailash-nexus`    |
| Kaizen    | AI agent framework                     | `pip install kailash-kaizen`   |
| PACT      | Organizational governance (D/T/R)      | `pip install kailash-pact`     |
| ML        | ML lifecycle (13 engines, polars)      | `pip install kailash-ml`       |
| Align     | LLM fine-tuning & serving              | `pip install kailash-align`    |

## Workspace Commands

| Command               | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| `/analyze`            | Load analysis phase for current workspace               |
| `/todos`              | Load todos phase; stops for human approval              |
| `/implement`          | Load implementation phase; repeat until todos done      |
| `/redteam`            | Load validation phase; red team exercises               |
| `/codify`             | Load codification phase; update agents & skills         |
| `/ws`                 | Read-only workspace status dashboard                    |
| `/wrapup`             | Write session notes before ending                       |
| `/build-module`       | Scaffold a new module (solutions, exercises, notebooks) |
| `/build-exercise`     | Create a single exercise across all three formats       |
| `/validate-notebooks` | Run all notebooks and verify outputs                    |

## Rules Index

| Concern                      | Rule File                     |
| ---------------------------- | ----------------------------- |
| Exercise authoring standards | `rules/exercise-standards.md` |
| Three-format consistency     | `rules/three-format.md`       |
| Assessment integrity         | `rules/domain-integrity.md`   |

## Agents

### Education (`agents/education/`)

- **exercise-designer** — Generates fill-in-blank exercises from solutions (progressive scaffolding)
- **kailash-tutor** — Maps traditional ML patterns (pandas/sklearn/PyCaret/CrewAI) to Kailash equivalents
- **dataset-curator** — Validates dataset quality (messiness, size, public availability, Singapore/APAC relevance)
- **quiz-designer** — Creates Kailash-pattern assessment questions (AI-resilient)

### Framework Specialists (`agents/frameworks/`)

Inherited from kailash-coc-claude-py: ml-specialist, dataflow-specialist, nexus-specialist, kaizen-specialist, pact-specialist, align-specialist, mcp-specialist

### Quality (`agents/quality/`)

- **reviewer** — Code review, exercise quality, solution correctness
- **notebook-validator** — Cross-validates 3 delivery formats for consistency
- **security-reviewer** — No hardcoded API keys, no exposed secrets

### Management (`agents/management/`)

- **todo-manager** — Task tracking across module development
- **gh-manager** — GitHub issue and project management

## Critical Execution Rules

```python
# Data loading — always use the shared loader
from shared import ASCENTDataLoader
loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdbprices.csv")

# kailash-ml — use engines, not raw libraries
from kailash_ml import DataExplorer, TrainingPipeline, AutoMLEngine

# Polars — never pandas
import polars as pl
df = pl.read_csv("data.csv")

# Environment — load before any operation
from dotenv import load_dotenv
load_dotenv()
```
