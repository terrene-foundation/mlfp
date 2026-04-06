# ASCENT — ML Engineering from Foundations to Mastery

The official Terrene Open Academy machine learning programme. 10 modules, 80 lessons, from zero Python to masters-level ML engineering. Hands-on exercises delivered in three formats (Python, Jupyter, Colab), powered by the open-source [Kailash Python SDK](https://github.com/terrene-foundation/kailash-py). Includes a supplementary [SDK textbook](textbook/) with 163 tutorials covering every Kailash engine in both Python and Rust.

**License**: Apache 2.0 | **Python**: 3.10+ | **Data**: [Polars](https://pola.rs)-native | **New to Polars?** See the [cheatsheet](docs/polars-cheatsheet.md)

---

## Who This Is For

Working professionals at any level — from complete beginners to experienced practitioners. **No prerequisites.** The programme starts from zero Python and progresses to masters-level content:

- **Module 1** teaches Python from scratch through real data exploration (variables, functions, loops — learned by using Polars and Kailash engines, not abstract exercises)
- **Modules 2-5** build statistical, feature engineering, and supervised ML foundations to production-grade systems
- **Modules 6-7** cover unsupervised ML as automated feature engineering, then deep learning as architecture-driven feature engineering
- **Modules 8-10** cover NLP/transformers, LLM agents, alignment, governance, and RL at an advanced level

You do **not** need prior experience with Python, statistics, Polars, Kailash, or ML. The course teaches everything from the ground up and ends at a masters-and-above level.

**What you get at the end**: A portfolio of completed exercises spanning Python basics through governed AI deployment, plus the architectural patterns to build ML systems that survive production.

---

## Why This Course Exists

Most ML courses teach you to call `model.fit()`. This course teaches you to build ML systems that survive production — where data is messy, models drift, stakeholders need explanations, and regulators need audit trails.

Every module has two layers:

- **Mathematical foundations** you can derive on a whiteboard: bias-variance decomposition, EM algorithm, SHAP axioms, DPO from Bradley-Terry, PPO clipped objective, Bellman equations
- **Production practice** on real-world data using the Kailash platform: data profiling, experiment tracking, model registry, drift monitoring, governed deployment

## What You Learn

| Module | Topic                               | Theory Depth                                                                  | Production Practice                                   |
| ------ | ----------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------- |
| **1**  | Python & Data Fluency               | Python from scratch, Polars, joins, windows, data profiling                   | DataExplorer, PreprocessingPipeline, ModelVisualizer  |
| **2**  | Statistical Foundations             | Probability, MLE, Bayesian inference, hypothesis testing, bootstrap           | ExperimentTracker, ModelVisualizer                    |
| **3**  | Feature Engineering & Experiments   | CUPED, DiD, causal inference, encoding, selection, leakage                    | FeatureEngineer, FeatureStore, ExperimentTracker      |
| **4**  | Supervised ML                       | Bias-variance, XGBoost internals, complete model zoo, calibration             | TrainingPipeline, HyperparameterSearch, ModelRegistry |
| **5**  | ML Engineering & Production         | SHAP/LIME/ALE, workflows, DataFlow, model lifecycle, ensembles                | WorkflowBuilder, DataFlow, EnsembleEngine             |
| **6**  | Unsupervised ML & Pattern Discovery | EM/GMM, PCA, spectral clustering, LDA, NMF, BERTopic, anomaly detection       | AutoMLEngine, DriftMonitor                            |
| **7**  | Deep Learning                       | Neural networks from linear regression, backprop, CNN, optimizers, embeddings | OnnxBridge, InferenceServer                           |
| **8**  | NLP & Transformers                  | Word2Vec, LSTM, attention, transformer architecture, BERT, GPT                | ModelVisualizer, AutoMLEngine (text)                  |
| **9**  | LLMs, AI Agents & RAG               | Scaling laws, RAG evaluation, MCP protocol, multi-agent A2A                   | Kaizen Delegate, ReActAgent, 6 ML agents, Nexus       |
| **10** | Alignment, RL & Governance          | LoRA/DPO/GRPO, EU AI Act, PACT D/T/R, PPO, Bellman equations                  | AlignmentPipeline, GovernanceEngine, RLTrainer        |

### Certification Structure

| Certificate           | Modules | Hours | Level                 |
| --------------------- | ------- | ----- | --------------------- |
| **Foundation Ascent** | M1-M5   | 160h  | Zero to production ML |
| **Summit Ascent**     | M6-M10  | 160h  | Advanced to masters   |

### The Feature Engineering Spectrum

The curriculum is organized around a central insight — the evolution of feature engineering:

```
M3-M5: Manual      →  M6: USML discovers    →  M7: DL learns        →  M8-M9: Transformers
Human designs         patterns within X         features via            learn semantic
features              n → 1 (clustering)        architecture            features from
                      n → k (dim reduction)     n → m (embeddings)      language
```

### After completing this course, you will be able to:

- Profile and clean messy real-world data at scale using polars and automated data quality tools
- Design and execute A/B tests with proper power analysis, variance reduction (CUPED), and causal inference
- Train, calibrate, and interpret production ML models with full SHAP explainability and conformal prediction
- Build and deploy unsupervised ML and deep learning systems with drift monitoring
- Build AI agent systems with structured outputs, tool use, RAG retrieval, and cost budgets
- Implement organizational governance that scales: access control, operating envelopes, and tamper-evident logging

---

## Learn Industry Standards, Not Vendor Lock-in

A fair question: _"Am I learning something I can only use with Kailash?"_

**No.** Kailash is a governance and orchestration layer that sits on top of the Python data science ecosystem. The core libraries are industry standards:

| What You Learn      | Industry Standard                                      | What Kailash Adds                                        |
| ------------------- | ------------------------------------------------------ | -------------------------------------------------------- |
| Data manipulation   | **Polars** (Apache Arrow)                              | DataExplorer: automated profiling with 8 alert types     |
| Classical ML        | **scikit-learn**, XGBoost, LightGBM, CatBoost          | TrainingPipeline: orchestrates training + model registry |
| Deep learning       | **PyTorch**                                            | OnnxBridge: exports to portable ONNX format              |
| NLP                 | **BERTopic**, sentence-transformers                    | ModelVisualizer: renders analysis as interactive Plotly  |
| Experiment tracking | Structured logging (run/params/metrics pattern)        | ExperimentTracker: async context manager with comparison |
| Model serving       | ONNX Runtime                                           | InferenceServer: serves ONNX with signature validation   |
| LLM agents          | OpenAI / Anthropic / Groq APIs (configurable via .env) | Kaizen Delegate: structured output with cost budgets     |

**Every concept has a mathematical derivation, an industry-standard implementation, AND a Kailash engine that adds production governance.** If you move to a different stack, you keep the math, the sklearn, the PyTorch, and the architectural patterns. Kailash teaches you _how to govern_ ML systems — that knowledge transfers everywhere.

The `kailash_ml.interop` module is the proof: `to_sklearn_input()`, `from_sklearn_output()`, `to_pandas()`, `polars_to_arrow()`. Kailash converts to and from standard formats at every boundary.

---

## Why Terrene Foundation

Kailash is maintained by the [Terrene Foundation](https://terrene.foundation), an independent non-profit (Singapore CLG). This matters for learners:

**Genuinely open source.** All intellectual property was irrevocably transferred to the Foundation under Apache 2.0. No contributor has exclusive rights, special access, or structural advantage. The Foundation's constitution explicitly prevents open-washing, rent-seeking, and commercial capture by any party.

**Open governance specifications.** The CARE, EATP, and CO specifications that underpin Kailash's governance features are published under CC BY 4.0. You can read, implement, and build commercial products on these specifications without restriction.

**Standard dependencies only.** PyPI packages with OSI-approved licenses. No proprietary SDKs, no vendor APIs, no commercial plug-ins required.

---

## Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager).

```bash
git clone https://github.com/terrene-foundation/ascent.git
cd ascent
uv venv && uv sync
cp .env.example .env  # API keys needed for Modules 9-10

# Run your first exercise
uv run python modules/ascent01/local/ex_1.py

# For advanced modules (deep learning, agents, RL, alignment):
uv sync --extra full
```

Three delivery formats for every exercise:

| Format       | Location                            | Best For                             |
| ------------ | ----------------------------------- | ------------------------------------ |
| Local Python | `modules/ascent*/local/*.py`        | Full async support, Nexus deployment |
| Jupyter      | `modules/ascent*/notebooks/*.ipynb` | Interactive exploration              |
| Google Colab | `modules/ascent*/colab/*.ipynb`     | Zero-install, GPU access             |

See [docs/setup-guide.md](docs/setup-guide.md) for detailed installation instructions.

---

## Course Structure

```
modules/
  ascent01/         Module 1: Python & Data Fluency
    solutions/    Complete, runnable solutions (instructor reference)
    local/        Python exercises (fill-in-the-blank)
    notebooks/    Jupyter notebooks (same exercises, notebook format)
    colab/        Colab notebooks (Drive mount, pip install)
  ascent02-10/      Modules 2-10 (same structure)

textbook/         Supplementary SDK textbook (163 tutorials)
  python/         83 Python tutorials (all 8 packages, basic→advanced)
  rust/           80 Rust tutorials (all 9 packages, basic→advanced)

data/             11 Singapore-context datasets (CSV/Parquet)
shared/           Data loader, Kailash helpers, async wrappers
docs/             Course outline, setup guide, Polars cheatsheet
decks/            10 Reveal.js lecture decks (1,333 slides total)
scripts/          Notebook converter, dataset generator, deck auditor
```

### Data

Datasets load automatically via `ASCENTDataLoader`, which detects your environment and pulls from Google Drive:

```python
from shared import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdb_resale.parquet")  # Singapore HDB resale prices
```

Singapore-focused datasets (HDB prices, taxi trips, economic indicators) combined with standard ML benchmarks (credit scoring, ecommerce, experiment data).

---

## Status

| Component               | Status   | Details                                        |
| ----------------------- | -------- | ---------------------------------------------- |
| Curriculum (80 lessons) | Complete | 10 modules × 8 lessons                         |
| Exercises               | Complete | Progressive scaffolding: 70% (M1) to 20% (M10) |
| SDK Textbook — Python   | Complete | 83 tutorials across all 8 packages             |
| SDK Textbook — Rust     | Complete | 80 tutorials across all 9 packages             |
| Datasets (11)           | Complete | Singapore-context CSV/Parquet                  |
| Lecture decks (10)      | Complete | Reveal.js, 1,333 slides, three-layer depth     |
| Quizzes (6)             | Complete | AI-resilient questions                         |
| Speaker notes (6)       | Complete | Per-slide timing + dual-audience tips          |

---

## Kailash Platform Packages

| Package                                                     | Purpose                                             | Install                        |
| ----------------------------------------------------------- | --------------------------------------------------- | ------------------------------ |
| [kailash](https://github.com/terrene-foundation/kailash-py) | Workflow orchestration (140+ nodes)                 | `pip install kailash`          |
| kailash-ml                                                  | ML lifecycle: 13 engines + RLTrainer, polars-native | `pip install kailash-ml`       |
| kailash-dataflow                                            | Zero-config database operations                     | `pip install kailash-dataflow` |
| kailash-nexus                                               | Deploy as API + CLI + MCP simultaneously            | `pip install kailash-nexus`    |
| kailash-kaizen                                              | AI agent framework (signatures, tools, A2A)         | `pip install kailash-kaizen`   |
| kailash-pact                                                | Governance: D/T/R accountability, audit chains      | `pip install kailash-pact`     |
| kailash-align                                               | LLM fine-tuning: SFT, DPO, QLoRA, GRPO              | `pip install kailash-align`    |

---

## For Instructors

See [docs/course-outline.md](docs/course-outline.md) for the full curriculum map, lecture topics, and assessment framework.

**Solution-first authoring**: Complete solutions in `modules/ascent*/solutions/` are the source of truth. Exercises are generated by stripping solutions to the appropriate scaffolding level. The `scripts/py_to_notebook.py` converter generates Jupyter and Colab formats from local Python files.

**Progressive scaffolding**: M1 provides ~70% of the code (students fill in key arguments). By M10, only imports and section headers are given (~20%) — students write nearly everything from hints and documentation.

## Contributing

This is a [Terrene Foundation](https://terrene.foundation) project. Contributions welcome via pull requests to [terrene-foundation/ascent](https://github.com/terrene-foundation/ascent).

Questions or feedback? Open an [issue](https://github.com/terrene-foundation/ascent/issues).

## License

Apache 2.0 — see [LICENSE](LICENSE).

Specifications (CARE, EATP, CO) are licensed under CC BY 4.0.
