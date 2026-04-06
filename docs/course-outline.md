# ASCENT — Practical Course in Machine Learning

**Institution**: Terrene Open Academy  
**Platform**: Kailash Python SDK (Terrene Foundation)  
**Audience**: Working professionals targeting senior data scientist / ML engineer roles  
**Format**: 6 modules (7h each: 3h lecture + 3h lab + 1h assessment), 3 delivery formats  
**Standard**: Georgia Tech OMSCS / Stanford CS229 depth, production-practice reality

---

## Course Philosophy

This is not an intro course. Students are working professionals who need to operate at **senior ML engineer level** — understanding theory deeply enough to debug production failures, make principled architecture decisions, and lead ML teams.

**Academic rigor**: Every technique taught with mathematical foundations. Bias-variance decomposition, EM algorithm, attention derivation, Bellman equations — students understand _why_, not just _how_.

**Production reality**: Every concept practiced on messy, large-scale, real-world data. Missing values, temporal leakage, class imbalance, schema drift. If the dataset is clean, it's not in this course.

**Kailash as production platform**: Theory with math, practice with Kailash engines. The SDK bridges textbook and production.

---

## Module Overview

| Module | Title                                   | Lecture Topics                                                                              | Lab (Kailash)                                                                    |
| ------ | --------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **1**  | Statistics, Probability & Data Fluency  | Bayesian estimation, MLE, hypothesis testing, bootstrapping (BCa), polars                   | DataExplorer, PreprocessingPipeline, ModelVisualizer                             |
| **2**  | Feature Engineering & Experiment Design | Feature selection theory, causal inference (Rubin/Pearl), A/B testing, CUPED, Double ML     | FeatureStore, FeatureEngineer, ExperimentTracker                                 |
| **3**  | Supervised ML — Theory to Production    | Bias-variance, regularization geometry, gradient boosting internals, SHAP/LIME, calibration | TrainingPipeline, HyperparameterSearch, ModelRegistry, WorkflowBuilder, DataFlow |
| **4**  | Unsupervised ML, NLP & Deep Learning    | Spectral clustering, EM/GMM, UMAP, BERTopic, attention mechanism, CNN/LSTM internals        | AutoMLEngine, DriftMonitor, InferenceServer, Nexus                               |
| **5**  | LLMs, AI Agents & RAG Systems           | Transformer internals, tokenization, RAG evaluation, agent architecture, multi-agent A2A    | Kaizen Delegate, ReActAgent, RAGResearchAgent, 6 ML agents                       |
| **6**  | Alignment, Governance, RL & Deployment  | LoRA/DPO theory, EU AI Act, PACT D/T/R, Bellman equations, PPO                              | AlignmentPipeline, GovernanceEngine, PactGovernedAgent, RLTrainer, Nexus         |

---

## Progressive Framework Introduction

```
Module 1: kailash-ml (3 engines: DataExplorer, PreprocessingPipeline, ModelVisualizer)
Module 2: kailash-ml (+3 engines: FeatureStore, FeatureEngineer, ExperimentTracker)
Module 3: Core SDK + DataFlow + kailash-ml (+3: TrainingPipeline, HyperparameterSearch, ModelRegistry)
Module 4: kailash-ml (+4 engines: AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer) + Nexus
Module 5: Kaizen (agents) + kailash-ml (6 ML agents)
Module 6: Align + PACT + kailash-ml (RLTrainer) + Nexus (full platform)
```

Cumulative: M1(3) → M2(6) → M3(9) → M4(13) → all 14 engines by M6 (RLTrainer).

No module introduces more than 2 new framework packages. Each builds on the prior.

---

## Three Delivery Formats

| Format               | Setup                                   | Data Loading     | Async             |
| -------------------- | --------------------------------------- | ---------------- | ----------------- |
| **Local** (.py)      | `uv sync`                               | gdown from Drive | `asyncio.run()`   |
| **Jupyter** (.ipynb) | `%pip install kailash-ml`               | gdown from Drive | Top-level `await` |
| **Colab** (.ipynb)   | `!pip install kailash-ml` + Drive mount | Drive mount path | Top-level `await` |

All formats use `shared/data_loader.py` (ASCENTDataLoader) which auto-detects the environment.

---

## Data Strategy

All datasets live on a shared Google Drive (`ascent_data`). Emphasis on:

- Complex, messy, real-world data (not toy CSVs)
- Singapore/APAC relevance where possible
- Publicly available for open-source distribution
- Multiple files requiring joins and cleaning

---

## Assessment

| Component                | Weight | Description                                                              |
| ------------------------ | ------ | ------------------------------------------------------------------------ |
| **Module Quizzes** (6)   | 20%    | 15 questions each: theory + code + interpretation                        |
| **Individual Portfolio** | 35%    | Extend one module exercise to production depth with model card           |
| **Team Capstone**        | 35%    | Multi-framework production system using 3+ Kailash packages              |
| **Peer Review**          | 10%    | Code review of another team's capstone (SHAP analysis, governance audit) |

---

## Prerequisites

- Python fundamentals (variables, loops, functions, classes)
- Basic statistics (mean, median, distributions)
- No prior ML experience required
- No prior Kailash experience required

---

## Installation

```bash
# Clone the repo
git clone https://github.com/terrene-foundation/ascent.git
cd ascent

# Set up environment
uv venv
uv sync

# Configure API keys (needed for Modules 5-6)
cp .env.example .env
# Edit .env with your keys

# Verify
uv run python -c "import kailash_ml; print(f'kailash-ml {kailash_ml.__version__}')"
```
