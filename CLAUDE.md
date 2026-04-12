# MLFP — ML Foundations for Professionals

Terrene Open Academy course powered by the Kailash Python SDK, maintained by the Terrene Foundation. Open-source course materials from zero Python to production ML engineering.

**Source program**: Derived from [ASCENT](../../programs/ascent/) (ML Engineering from Foundations to Mastery). ASCENT holds the canonical knowledge; this course packages it for professional delivery.

## Absolute Directives

### 0. Foundation Independence

Kailash Python SDK is a **Terrene Foundation project** (Singapore CLG). No commercial references. No institutional course codes, partner names, or funding body references in any material. See `rules/independence.md`.

### 1. Framework-First

All ML content uses Kailash frameworks. Never raw sklearn/pandas/PyTorch without the Kailash wrapper. Before writing custom code, check if a kailash-ml engine handles it: `DataExplorer`, `PreprocessingPipeline`, `FeatureStore`, `FeatureEngineer`, `TrainingPipeline`, `AutoMLEngine`, `HyperparameterSearch`, `EnsembleEngine`, `ModelRegistry`, `InferenceServer`, `DriftMonitor`, `ExperimentTracker`, `ModelVisualizer`.

### 2. Polars-Native

No pandas in any exercise. kailash-ml is polars-native — students learn polars from Module 1. A pandas-to-polars cheatsheet lives in `docs/polars-cheatsheet.md`. Every data operation uses `polars` or `kailash_ml` APIs.

### 3. Two-Format Consistency

Every exercise must exist in two formats:

| Format  | Location                             | Data Loading         |
| ------- | ------------------------------------ | -------------------- |
| VS Code | `modules/mlfpNN/local/ex_N.py`       | `shared.data_loader` |
| Colab   | `modules/mlfpNN/colab/ex_N.ipynb`    | Drive mount + gdown  |

Data loading via `shared/data_loader.py`. Setup differs by format; exercise code is identical.

### 4. Solution-First Authoring

Write complete solutions first in `modules/mlfpNN/solutions/`, then strip to create exercises. The exercise-designer agent automates this. Solutions must always run correctly end-to-end.

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

### 6. Naming Compliance

No references to any institutional partner, funding body, or prior course code in any material — code, comments, decks, notebooks, README, or metadata. This course is an independent Terrene Foundation publication. See `rules/independence.md`.

## Course Structure

| Module | Title                                                                                        | Kailash Frameworks                                                                                             |
| ------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| MLFP01 | Machine Learning Data Pipelines and Visualisation Mastery with Python                        | kailash-ml: DataExplorer, PreprocessingPipeline, ModelVisualizer                                               |
| MLFP02 | Statistical Mastery for Machine Learning and Artificial Intelligence (AI) Success             | kailash-ml: ExperimentTracker, FeatureEngineer, FeatureStore, ModelVisualizer                                  |
| MLFP03 | Supervised Machine Learning for Building and Deploying Models                                | kailash-ml: TrainingPipeline, HyperparameterSearch, ModelRegistry; Core SDK: WorkflowBuilder; DataFlow         |
| MLFP04 | Unsupervised Machine Learning and Advanced Techniques for Insights                           | kailash-ml: AutoMLEngine, EnsembleEngine, DriftMonitor, ModelVisualizer                                        |
| MLFP05 | Deep Learning and Machine Learning Mastery in Vision and Transfer Learning                   | kailash-ml: OnnxBridge, InferenceServer, TrainingPipeline, ModelVisualizer                                     |
| MLFP06 | Machine Learning with Language Models and Agentic Workflows | Kaizen: Delegate, BaseAgent, Signature; Align: AlignmentPipeline; PACT: GovernanceEngine; Nexus; MCP          |

### Certification Structure

- **Foundation Certificate**: Modules 1-4 (32 lessons, ~128 contact hours)
- **Advanced Certificate**: Modules 5-6 (16 lessons, ~64 contact hours)

### Source Program Mapping

Each MLFP module draws from one or more ASCENT program modules:

| MLFP   | ASCENT Source                      | Content Scope                                                           |
| ------ | ---------------------------------- | ----------------------------------------------------------------------- |
| MLFP01 | ASCENT01 (partial)                 | Python basics through data, Polars, visualisation, profiling, cleaning  |
| MLFP02 | ASCENT01 (stats) + ASCENT02        | Probability, inference, experiments, features, feature store            |
| MLFP03 | ASCENT03 + ASCENT05 (partial)      | Supervised ML, workflows, DataFlow, model registry, hyperparameters     |
| MLFP04 | ASCENT04                           | Clustering, dimensionality reduction, anomaly detection, NLP, drift     |
| MLFP05 | ASCENT07 + ASCENT08                | Neural networks, CNNs, vision, NLP transformers, transfer learning, ONNX |
| MLFP06 | ASCENT05 (agents) + ASCENT06 + ASCENT09 + ASCENT10 | LLMs, agents, RAG, alignment, governance, deployment            |

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
| `/build-exercise`     | Create a single exercise across both formats            |
| `/validate-notebooks` | Run all notebooks and verify outputs                    |

## Rules Index

| Concern                      | Rule File                     |
| ---------------------------- | ----------------------------- |
| Exercise authoring standards | `rules/exercise-standards.md` |
| Two-format consistency       | `rules/two-format.md`         |
| Assessment integrity         | `rules/domain-integrity.md`   |
| Foundation independence      | `rules/independence.md`       |
| Naming compliance            | `rules/terrene-naming.md`     |

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
- **notebook-validator** — Cross-validates delivery formats for consistency
- **security-reviewer** — No hardcoded API keys, no exposed secrets

### Management (`agents/management/`)

- **todo-manager** — Task tracking across module development
- **gh-manager** — GitHub issue and project management

## Critical Execution Rules

```python
# Data loading — always use the shared loader
from shared import MLFPDataLoader
loader = MLFPDataLoader()
df = loader.load("mlfp01", "hdbprices.csv")

# kailash-ml — use engines, not raw libraries
from kailash_ml import DataExplorer, TrainingPipeline, AutoMLEngine

# Polars — never pandas
import polars as pl
df = pl.read_csv("data.csv")

# Environment — load before any operation
from dotenv import load_dotenv
load_dotenv()
```
