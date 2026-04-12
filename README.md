# ML Foundations for Professionals

### From Zero Python to Production ML Engineering

> _"The question isn't whether AI will transform your industry. The question is: will you be the one leading that transformation?"_

**6 modules. 48 lessons. 192 contact hours. Zero to production.**

ML Foundations for Professionals (MLFP) is an open-source ML engineering course from [Terrene Open Academy](https://terrene.foundation), powered by the [Kailash Python SDK](https://github.com/terrene-foundation/kailash-py). It takes working professionals from their first line of Python to production-grade ML systems with language models and agentic workflows — every step backed by rigorous mathematical derivations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![Data](https://img.shields.io/badge/Data-Polars%20Native-orange.svg)](https://pola.rs)

---

## The Course

|             | Foundation Certificate (M1-M4) | Advanced Certificate (M5-M6)           |
| ----------- | ------------------------------ | -------------------------------------- |
| **Level**   | Zero Python to production ML   | Deep learning to organisational AI     |
| **Hours**   | ~128h (32 lessons)             | ~64h (16 lessons)                      |
| **Outcome** | Deploy governed ML models      | Build LLM agents and transform organisations |

### Module Map

| #   | Module                                                                                        | What You Master                                                                                                          |
| --- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1   | **Machine Learning Data Pipelines and Visualisation Mastery with Python**                     | Python from scratch, Polars data pipelines, interactive visualisation, data profiling, automated cleaning                |
| 2   | **Statistical Mastery for Machine Learning and Artificial Intelligence (AI) Success**          | Probability, Bayesian inference, hypothesis testing, bootstrap, experiment design, feature engineering, feature stores    |
| 3   | **Supervised Machine Learning for Building and Deploying Models**                             | Model zoo (linear through CatBoost), bias-variance decomposition, workflow orchestration, model registry, hyperparameter search |
| 4   | **Unsupervised Machine Learning and Advanced Techniques for Insights**                        | Clustering (K-means through HDBSCAN), dimensionality reduction, anomaly detection, NLP text analysis, drift monitoring   |
| 5   | **Deep Learning and Machine Learning Mastery in Vision and Transfer Learning**                | Neural networks, backpropagation, CNNs, vision architectures, NLP transformers, transfer learning, ONNX deployment       |
| 6   | **Machine Learning with Language Models and Agentic Workflows** | LLM prompt engineering, fine-tuning (LoRA/DPO/GRPO), RAG systems, multi-agent orchestration, AI governance engineering, production deployment        |

### The Organising Principle: Feature Engineering Spectrum

Every module builds on a single insight — the evolution of how we create features:

```
M3                       M4                        M5                     M6
Manual                   USML discovers            DL learns features     LLMs learn
Human designs            patterns within X         via architecture       semantic features
features                 n -> 1 (clustering)       n -> m (embeddings)    from language
                         n -> k (dim reduction)
                         n -> k (topics)
```

**Unsupervised ML is automated feature engineering.** Supervised ML discovers combinations of X that predict y. Unsupervised ML discovers patterns _within_ X. Deep learning uses architecture to _learn_ features. LLMs learn semantic features from language. This progression — manual, algorithmic, architectural, semantic — is the spine of the entire curriculum.

---

## Three Layers Per Concept

Every concept is taught at three depths simultaneously:

| Layer           | Marker                     | Audience                      | Example (Bias-Variance)                                                                                                                      |
| --------------- | -------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intuition**   | :green_circle: FOUNDATIONS | Zero-background professionals | _"Imagine throwing darts at a target. Bias is how far the center of your throws is from the bullseye. Variance is how spread out they are."_ |
| **Mathematics** | :blue_circle: THEORY       | Intermediate practitioners    | E[(y-y_hat)^2] = Bias^2(y_hat) + Var(y_hat) + sigma^2 — derived step by step, each term colour-coded                                        |
| **Research**    | :purple_circle: ADVANCED   | Masters+ / PhD holders        | Double descent (Belkin et al., 2019): test error _decreases_ past the interpolation threshold in over-parameterised models                   |

A banker and a PhD sit in the same classroom. Both leave having learned something they didn't know.

---

## What's In The Box

| Component          | Details                                                       |
| ------------------ | ------------------------------------------------------------- |
| Lecture decks      | 6 Reveal.js HTML decks, three-layer depth, KaTeX math         |
| Speaker notes      | Per-slide timing, beginner tips, expert tangents              |
| Coding exercises   | Solutions + VS Code (.py) + Google Colab (.ipynb)             |
| Datasets           | Singapore-context: HDB resale, taxi trips, economic data      |
| End-of-module quizzes | AI-resilient questions (context-specific, not recall)       |

---

## Quick Start

```bash
git clone https://github.com/terrene-foundation/mlfp.git
cd mlfp
uv venv && uv sync
cp .env.example .env  # API keys for M6

# Your first exercise
uv run python modules/mlfp01/local/ex_1.py

# View lecture deck
open decks/mlfp01/deck.html

# Advanced modules (DL, agents, alignment)
uv sync --extra full
```

### Two Delivery Formats

| Format       | Location                         | Best For                     |
| ------------ | -------------------------------- | ---------------------------- |
| VS Code      | `modules/mlfp*/local/*.py`       | Full async, local development |
| Google Colab | `modules/mlfp*/colab/*.ipynb`    | Zero-install, GPU access     |

### Data Loading

```python
from shared import MLFPDataLoader

loader = MLFPDataLoader()
df = loader.load("mlfp01", "hdb_resale.parquet")
```

Auto-detects environment (local vs Colab) and pulls from Google Drive when needed.

---

## Repository Structure

```
mlfp/
  modules/
    mlfp01-06/          6 modules, each with:
      solutions/        Complete runnable solutions (instructor reference)
      local/            Python exercises (fill-in-the-blank)
      colab/            Colab notebooks (Drive mount)
  decks/
    mlfp01-06/          6 Reveal.js lecture decks
    assets/css/         Custom theme (teal/indigo, three-layer markers)
  data/                 Singapore-context datasets (CSV/Parquet)
  shared/               Data loader, Kailash helpers
  scripts/              Notebook converter, dataset generator
```

---

## Not Vendor Lock-in

Kailash is a governance and orchestration layer on top of industry standards:

| What You Learn | Industry Standard                             | What Kailash Adds                                        |
| -------------- | --------------------------------------------- | -------------------------------------------------------- |
| Data           | **Polars** (Apache Arrow)                     | DataExplorer: automated profiling, 8 alert types         |
| Classical ML   | **scikit-learn**, XGBoost, LightGBM, CatBoost | TrainingPipeline: orchestrated training + model registry |
| Deep learning  | **PyTorch**                                   | OnnxBridge: portable ONNX export                         |
| NLP            | **BERTopic**, sentence-transformers           | ModelVisualizer: interactive Plotly analysis             |
| LLM agents     | OpenAI / Anthropic / Groq APIs                | Kaizen Delegate: structured output with cost budgets     |
| Model serving  | ONNX Runtime                                  | InferenceServer: signature validation + caching          |
| Governance     | EU AI Act / Singapore AI Verify               | PACT: D/T/R accountability with operating envelopes      |

Every concept has a derivation, an industry-standard implementation, _and_ a Kailash engine. If you move to a different stack, you keep the math, the sklearn, the PyTorch, and the architectural patterns.

---

## Kailash Platform

| Package                                                     | Purpose                                  | Install                        |
| ----------------------------------------------------------- | ---------------------------------------- | ------------------------------ |
| [kailash](https://github.com/terrene-foundation/kailash-py) | Workflow orchestration, 140+ nodes       | `pip install kailash`          |
| kailash-ml                                                  | 13 ML engines + RLTrainer, polars-native | `pip install kailash-ml`       |
| kailash-dataflow                                            | Zero-config database operations          | `pip install kailash-dataflow` |
| kailash-nexus                                               | API + CLI + MCP deployment               | `pip install kailash-nexus`    |
| kailash-kaizen                                              | AI agent framework, signatures, A2A      | `pip install kailash-kaizen`   |
| kailash-pact                                                | D/T/R governance, operating envelopes    | `pip install kailash-pact`     |
| kailash-align                                               | SFT, DPO, QLoRA, GRPO fine-tuning        | `pip install kailash-align`    |

---

## For Instructors

**Solution-first authoring.** Solutions in `modules/mlfp*/solutions/` are the source of truth. Exercises are generated by stripping to the appropriate scaffolding level.

**Progressive scaffolding.** M1 provides ~70% of code. M6 provides ~20%. By the end, students write from documentation alone.

**Three-layer delivery.** Green/blue/purple markers let instructors calibrate depth in real time. Beginners follow green. Experts engage with purple. Everyone sees the same deck.

**Speaker notes.** Per-slide timing, common student questions, "if beginners look confused" alternatives, "if experts look bored" tangents.

---

## Why Terrene Foundation

[Terrene Foundation](https://terrene.foundation) is an independent non-profit (Singapore CLG).

**Genuinely open source.** All IP irrevocably transferred under Apache 2.0. No contributor has exclusive rights or structural advantage. The constitution prevents open-washing and commercial capture.

**Open specifications.** CARE, EATP, and CO published under CC BY 4.0. Build commercial products on them without restriction.

**Standard dependencies only.** PyPI packages, OSI-approved licenses. No proprietary SDKs required.

---

## Contributing

Contributions welcome via [pull request](https://github.com/terrene-foundation/mlfp/pulls). Questions via [issues](https://github.com/terrene-foundation/mlfp/issues).

## License

[Apache 2.0](LICENSE). Specifications (CARE, EATP, CO) under CC BY 4.0.
