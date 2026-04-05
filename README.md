# ASCENT — Professional Certificate in Machine Learning

The official Terrene Open Academy machine learning programme. 6 modules × 8 lessons = 48 lessons, from zero Python to masters-level ML engineering. 48 hands-on exercises delivered in three formats (Python, Jupyter, Colab), powered by the open-source [Kailash Python SDK](https://github.com/terrene-foundation/kailash-py). Includes a supplementary [SDK textbook](textbook/) with 163 tutorials covering every Kailash engine in both Python and Rust.

**License**: Apache 2.0 | **Python**: 3.10+ | **Data**: [Polars](https://pola.rs)-native | **New to Polars?** See the [cheatsheet](docs/polars-cheatsheet.md)

> All 48 exercises, 11 datasets, 48 lecture decks, and 6 quiz files are complete. See [Status](#status) for details.

---

## Who This Is For

Working professionals at any level — from complete beginners to experienced practitioners. **No prerequisites.** The programme starts from zero Python and progresses to masters-level content:

- **Module 1** teaches Python from scratch through real data exploration (variables, functions, loops — learned by using Polars and Kailash engines, not abstract exercises)
- **Modules 2-4** build statistical and ML foundations to production-grade supervised/unsupervised systems
- **Modules 5-6** cover LLM agents, alignment, governance, and RL at an advanced level

You do **not** need prior experience with Python, statistics, Polars, Kailash, or ML. The course teaches everything from the ground up and ends at a masters-and-above level.

**What you get at the end**: A portfolio of 48 completed exercises spanning Python basics through governed AI deployment, plus the architectural patterns to build ML systems that survive production.

---

## Why This Course Exists

Most ML courses teach you to call `model.fit()`. This course teaches you to build ML systems that survive production — where data is messy, models drift, stakeholders need explanations, and regulators need audit trails.

Every module has two layers:

- **Mathematical foundations** you can derive on a whiteboard: bias-variance decomposition, EM algorithm, SHAP axioms, DPO from Bradley-Terry, PPO clipped objective, Bellman equations
- **Production practice** on real-world data using the Kailash platform: data profiling, experiment tracking, model registry, drift monitoring, governed deployment

## What You Learn

| Module | Topic                                        | Theory Depth                                                                   | Production Practice                                  |
| ------ | -------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **1**  | Data Pipelines & Visualisation with Python   | Python from scratch, Polars, joins, windows, data profiling                    | DataExplorer, PreprocessingPipeline, ModelVisualizer |
| **2**  | Statistical Mastery for ML                   | Bayesian estimation, MLE, CUPED, DiD, bootstrap, causal inference              | ExperimentTracker, FeatureStore, FeatureEngineer     |
| **3**  | Supervised ML — Theory to Production         | Bias-variance, XGBoost internals, SHAP/LIME, calibration, conformal prediction | TrainingPipeline, ModelRegistry, WorkflowBuilder     |
| **4**  | Advanced ML ��� Unsupervised & Deep Learning | EM/GMM, PCA, spectral clustering, BERTopic, CNN/ResNet, ONNX                   | AutoMLEngine, DriftMonitor, InferenceServer          |
| **5**  | LLMs, AI Agents & Production Deployment      | Tokenization, scaling laws, RAG evaluation, MCP protocol, multi-agent A2A      | Kaizen Delegate, ReActAgent, 6 ML agents, Nexus      |
| **6**  | Alignment, Governance & Organisational AI    | LoRA/DPO/GRPO, EU AI Act, PACT D/T/R, PPO, Bellman equations                   | AlignmentPipeline, GovernanceEngine, RLTrainer       |

Each module has 8 lessons of ~4 hours each (lecture + lab). Total: ~192 contact hours. Foundation Certificate (M1-M4, 128h) + Advanced Certificate (M5-M6, 64h).

### After completing this course, you will be able to:

- Profile and clean messy real-world data at scale using polars and automated data quality tools
- Design and execute A/B tests with proper power analysis, variance reduction (CUPED), and causal inference
- Train, calibrate, and interpret production ML models with full SHAP explainability and conformal prediction
- Deploy models as governed APIs with drift monitoring, version control, and audit trails
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

The `kailash_ml.interop` module is the proof: `to_sklearn_input()`, `from_sklearn_output()`, `to_pandas()`, `polars_to_arrow()`. Kailash converts to and from standard formats at every boundary. The exercises directly import and use scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, and BERTopic — Kailash wraps these tools, it does not replace them.

---

## Why Terrene Foundation

Kailash is maintained by the [Terrene Foundation](https://terrene.foundation), an independent non-profit (Singapore CLG). This matters for learners:

**Genuinely open source.** All intellectual property was irrevocably transferred to the Foundation under Apache 2.0. No contributor has exclusive rights, special access, or structural advantage. The Foundation's constitution explicitly prevents open-washing, rent-seeking, and commercial capture by any party.

**Open governance specifications.** The CARE, EATP, and CO specifications that underpin Kailash's governance features are published under CC BY 4.0. You can read, implement, and build commercial products on these specifications without restriction. These are not proprietary standards locked behind a vendor — they are open protocols designed for the ecosystem.

**Standard dependencies only.** PyPI packages with OSI-approved licenses. No proprietary SDKs, no vendor APIs, no commercial plug-ins required. Kailash runs on the same infrastructure as any Python project.

---

## Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager).

```bash
git clone https://github.com/terrene-foundation/ascent.git
cd ascent
uv venv && uv sync
cp .env.example .env  # API keys needed for Modules 5-6

# Run your first exercise
uv run python modules/ascent01/local/ex_1.py

# For Modules 4-6 (deep learning, agents, RL, alignment):
uv sync --extra full
```

Three delivery formats for every exercise:

| Format       | Location                          | Best For                             |
| ------------ | --------------------------------- | ------------------------------------ |
| Local Python | `modules/ascent*/local/*.py`        | Full async support, Nexus deployment |
| Jupyter      | `modules/ascent*/notebooks/*.ipynb` | Interactive exploration              |
| Google Colab | `modules/ascent*/colab/*.ipynb`     | Zero-install, GPU access             |

**Recommended**: Local Python for classroom delivery. Use Colab if you cannot install software on your machine. Note: Nexus deployment exercises (M4, M6) are local-only.

See [docs/setup-guide.md](docs/setup-guide.md) for detailed installation instructions.

---

## Course Structure

```
modules/
  ascent01/          Module 1: Data Pipelines & Visualisation with Python
    solutions/    Complete, runnable solutions (instructor reference)
    local/        Python exercises (fill-in-the-blank)
    notebooks/    Jupyter notebooks (same exercises, notebook format)
    colab/        Colab notebooks (Drive mount, pip install)
    quiz/         Assessment questions
  ascent02-6/        Modules 2-6 (same structure)

textbook/         Supplementary SDK textbook (163 tutorials)
  python/         83 Python tutorials (all 8 packages, basic→advanced)
  rust/           80 Rust tutorials (all 9 packages, basic→advanced)
  PARITY.md       Cross-language parity matrix (20 known divergences)

data/             11 Singapore-context datasets (CSV/Parquet)
shared/           Data loader, Kailash helpers, async wrappers
docs/             Course outline, setup guide, Polars cheatsheet
decks/            48 Marp-format lecture slide decks
scripts/          Notebook converter, dataset generator, CI validation
```

### Data

Datasets load automatically via `ASCENTDataLoader`, which detects your environment and pulls from Google Drive:

```python
from shared import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdb_resale.parquet")  # Singapore HDB resale prices
```

Singapore-focused datasets (HDB prices, taxi trips, economic indicators) combined with standard ML benchmarks (credit scoring, fraud detection, medical imaging).

---

## Status

| Component                | Status   | Details                                                    |
| ------------------------ | -------- | ---------------------------------------------------------- |
| Curriculum (48 lessons)  | Complete | 6 modules × 8 lessons, red-team reviewed (v4)              |
| Solutions (48/48)        | Complete | All exercises written (16 new, 8 adapted, 2 moved)         |
| Exercises — local (48)   | Complete | Progressive scaffolding: 70% (M1) to 20% (M6)              |
| Exercises — Jupyter (48) | Complete | Auto-generated from local sources                          |
| Exercises — Colab (48)   | Complete | Auto-generated with Drive mount + pip install              |
| SDK Textbook — Python    | Complete | 83 tutorials across all 8 packages                         |
| SDK Textbook — Rust      | Complete | 80 tutorials across all 9 packages (core through RL)       |
| Datasets (11)            | Complete | Singapore-context CSV/Parquet (weather, HDB, credit, etc.) |
| Lecture decks (48)       | Complete | Marp-format slide decks, 20-30 slides each                 |
| Quizzes (6)              | Complete | AI-resilient questions, ~24-32 per module                  |
| CI/Testing               | Planned  | Needs Kailash packages on PyPI                             |

---

## Kailash Platform Packages

| Package                                                     | Purpose                                        | Install                        |
| ----------------------------------------------------------- | ---------------------------------------------- | ------------------------------ |
| [kailash](https://github.com/terrene-foundation/kailash-py) | Workflow orchestration (140+ nodes)            | `pip install kailash`          |
| kailash-ml                                                  | ML lifecycle: 13 engines, polars-native        | `pip install kailash-ml`       |
| kailash-dataflow                                            | Zero-config database operations                | `pip install kailash-dataflow` |
| kailash-nexus                                               | Deploy as API + CLI + MCP simultaneously       | `pip install kailash-nexus`    |
| kailash-kaizen                                              | AI agent framework (signatures, tools, A2A)    | `pip install kailash-kaizen`   |
| kailash-pact                                                | Governance: D/T/R accountability, audit chains | `pip install kailash-pact`     |
| kailash-align                                               | LLM fine-tuning: SFT, DPO, QLoRA, GRPO         | `pip install kailash-align`    |

---

## For Instructors

See [docs/course-outline.md](docs/course-outline.md) for the full curriculum map, lecture topics, and assessment framework (20% quizzes, 35% individual portfolio, 35% team capstone, 10% peer review).

**Solution-first authoring**: Complete solutions in `modules/ascent*/solutions/` are the source of truth. Exercises are generated by stripping solutions to the appropriate scaffolding level. The `scripts/py_to_notebook.py` converter generates Jupyter and Colab formats from local Python files.

**Progressive scaffolding**: M1 provides ~70% of the code (students fill in key arguments). By M6, only imports and section headers are given (~20%) — students write nearly everything from hints and documentation.

## Contributing

This is a [Terrene Foundation](https://terrene.foundation) project. Contributions welcome via pull requests to [terrene-foundation/ascent](https://github.com/terrene-foundation/ascent).

Questions or feedback? Open an [issue](https://github.com/terrene-foundation/ascent/issues).

## License

Apache 2.0 — see [LICENSE](LICENSE).

Specifications (CARE, EATP, CO) are licensed under CC BY 4.0.
