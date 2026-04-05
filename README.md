# ASCENT — Professional Certificate in Machine Learning

A rigorous, production-grade machine learning curriculum for working professionals. 34 hands-on exercises across 6 modules, delivered in three formats (Python, Jupyter, Colab), powered by the open-source [Kailash Python SDK](https://github.com/terrene-foundation/kailash-py).

**License**: Apache 2.0 | **Python**: 3.10+ | **Data**: [Polars](https://pola.rs)-native

> This curriculum is under active development. All 34 exercises and solutions are complete. Lecture slides, quizzes, and datasets are in progress. See [Status](#status) for details.

---

## Who This Is For

Working professionals targeting senior data scientist or ML engineer roles. You should have:

- **Python fluency** — comfortable writing functions, classes, async/await
- **Basic statistics** — mean, variance, probability distributions, hypothesis testing concepts
- **Some ML exposure** — have used sklearn or similar; understand train/test splits, overfitting

You do **not** need prior experience with Polars, Kailash, or production ML systems. The course teaches these from the ground up.

**What you get at the end**: A portfolio of 34 completed exercises spanning statistics through governed AI deployment, plus the architectural patterns to build ML systems that survive production. The exercises themselves — running on real data with proper governance — are the portfolio.

---

## Why This Course Exists

Most ML courses teach you to call `model.fit()`. This course teaches you to build ML systems that survive production — where data is messy, models drift, stakeholders need explanations, and regulators need audit trails.

Every module has two layers:

- **Mathematical foundations** you can derive on a whiteboard: bias-variance decomposition, EM algorithm, SHAP axioms, DPO from Bradley-Terry, PPO clipped objective, Bellman equations
- **Production practice** on real-world data using the Kailash platform: data profiling, experiment tracking, model registry, drift monitoring, governed deployment

## What You Learn

| Module | Topic                             | Theory Depth                                                                   | Production Practice                              |
| ------ | --------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------ |
| **1**  | Statistics & Data Fluency         | Bayesian estimation, MLE, BCa bootstrap, permutation tests                     | DataExplorer, PreprocessingPipeline              |
| **2**  | Feature Engineering & Experiments | Causal inference (DiD, CUPED), A/B testing, point-in-time correctness          | FeatureStore, FeatureEngineer, ExperimentTracker |
| **3**  | Supervised ML to Production       | Bias-variance, XGBoost internals, SHAP/LIME, calibration, conformal prediction | TrainingPipeline, ModelRegistry, WorkflowBuilder |
| **4**  | Unsupervised, NLP & Deep Learning | EM/GMM, spectral clustering, attention derivation, BERTopic                    | AutoMLEngine, DriftMonitor, InferenceServer      |
| **5**  | LLMs, AI Agents & RAG             | Transformer internals, RAG evaluation (RAGAS), agent safety                    | Kaizen Delegate, ReActAgent, 6 ML agents         |
| **6**  | Alignment, Governance & RL        | LoRA/DPO/GRPO, EU AI Act, PACT D/T/R, PPO, Bellman equations                   | AlignmentPipeline, GovernanceEngine, Nexus       |

Each module is designed for 7 hours (3h lecture + 3h lab + 1h assessment). Total: 42 hours of structured learning.

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

```bash
git clone https://github.com/terrene-foundation/ascent.git
cd ascent
uv venv && uv sync
cp .env.example .env  # API keys needed for Modules 5-6

# Run your first exercise
uv run python modules/ascent01/local/ex_1.py
```

Three delivery formats for every exercise:

| Format       | Location                          | Best For                             |
| ------------ | --------------------------------- | ------------------------------------ |
| Local Python | `modules/ascent*/local/*.py`        | Full async support, Nexus deployment |
| Jupyter      | `modules/ascent*/notebooks/*.ipynb` | Interactive exploration              |
| Google Colab | `modules/ascent*/colab/*.ipynb`     | Zero-install, GPU access             |

See [docs/setup-guide.md](docs/setup-guide.md) for detailed installation instructions.

---

## Course Structure

```
modules/
  ascent01/          Module 1: Statistics & Data Fluency
    solutions/    Complete, runnable solutions (instructor reference)
    local/        Python exercises (fill-in-the-blank)
    notebooks/    Jupyter notebooks (same exercises, notebook format)
    colab/        Colab notebooks (Drive mount, pip install)
    quiz/         Assessment questions
  ascent02-6/        Modules 2-6 (same structure)

shared/           Data loader, Kailash helpers (used by all exercises)
docs/             Course outline, setup guide, Polars cheatsheet
decks/            Reveal.js lecture slides
scripts/          Notebook converter, CI validation
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

| Component                | Status      | Details                                            |
| ------------------------ | ----------- | -------------------------------------------------- |
| Solutions (34)           | Complete    | All modules, red-team reviewed for SDK correctness |
| Exercises — local (34)   | Complete    | Progressive scaffolding: 70% (M1) to 20% (M6)      |
| Exercises — Jupyter (34) | Complete    | Auto-generated from local sources                  |
| Exercises — Colab (34)   | Complete    | Auto-generated with Drive mount + pip install      |
| Datasets                 | In progress | Synthetic + public datasets being prepared         |
| Lecture slides           | Planned     | Reveal.js decks per module                         |
| Quizzes                  | Planned     | 15 Kailash-pattern questions per module            |
| CI/Testing               | Planned     | Needs Kailash packages on PyPI                     |

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
