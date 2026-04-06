# ASCENT

### ML Engineering from Foundations to Mastery

> _"The question isn't whether AI will transform your industry. The question is: will you be the one leading that transformation?"_

**1,333 lecture slides. 10 modules. 320 hours. Zero to masters.**

ASCENT is the open-source ML engineering programme from [Terrene Open Academy](https://terrene.foundation), powered by the [Kailash Python SDK](https://github.com/terrene-foundation/kailash-py). It takes working professionals from their first line of Python to production-grade ML systems with full governance — and every step is backed by rigorous mathematical derivations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Data](https://img.shields.io/badge/Data-Polars%20Native-orange.svg)](https://pola.rs)
[![Slides](https://img.shields.io/badge/Slides-1%2C333-purple.svg)](#lecture-decks)

---

## The Programme

|             | Foundation Ascent (M1-M5)    | Summit Ascent (M6-M10)         |
| ----------- | ---------------------------- | ------------------------------ |
| **Level**   | Zero Python to production ML | Advanced to masters            |
| **Hours**   | 160h (40 lessons)            | 160h (40 lessons)              |
| **Outcome** | Deploy governed ML models    | Build aligned AI agent systems |

### Module Map

| #   | Module                                  | What You Master                                                                                                                                                      | Slides    |
| --- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| 1   | **Python & Data Fluency**               | Python from scratch, Polars, data profiling, visualization                                                                                                           | 85        |
| 2   | **Statistical Foundations**             | 20+ distributions, MLE, Bayesian inference, hypothesis testing, bootstrap, information theory                                                                        | 131       |
| 3   | **Feature Engineering & Experiments**   | CUPED variance reduction, DiD, causal forests, Double ML, 9 encoding methods, Boruta, leakage detection                                                              | 99        |
| 4   | **Supervised ML**                       | Complete model zoo (linear through CatBoost), XGBoost 2nd-order Taylor, bias-variance decomposition, conformal prediction                                            | 83        |
| 5   | **ML Engineering & Production**         | SHAP axioms + TreeSHAP, LIME, ALE, fairness (impossibility theorem), workflows, DataFlow, model registry, ensembles                                                  | 150       |
| 6   | **Unsupervised ML & Pattern Discovery** | K-means through HDBSCAN, EM/GMM (full derivation), PCA-SVD connection, t-SNE, UMAP, LDA, NMF, BERTopic, anomaly detection                                            | 146       |
| 7   | **Deep Learning**                       | Linear regression as NN, backpropagation (full chain rule), parallelized training (data/model/pipeline/tensor), CNN, ResNet, Adam derivation                         | 100       |
| 8   | **NLP & Transformers**                  | BPE tokenization, Word2Vec (negative sampling derivation), LSTM gates, self-attention (why divide by sqrt d_k), transformer architecture, BERT, GPT, Flash Attention | 150       |
| 9   | **LLMs, AI Agents & RAG**               | LLM landscape Q1 2026, 7 RAG architectures, hybrid retrieval, RAGAS evaluation, ReAct/Reflexion agents, multi-agent A2A, MCP protocol, Nexus deployment              | 235       |
| 10  | **Alignment, RL & Governance**          | LoRA/QLoRA, DPO (5-step derivation from RLHF), GRPO, PPO (clipped objective + GAE), Bellman equations, EU AI Act, PACT D/T/R governance, full platform capstone      | 154       |
|     |                                         | **Total**                                                                                                                                                            | **1,333** |

### The Organizing Principle: Feature Engineering Spectrum

Every module builds on a single insight — the evolution of how we create features:

```
M3-M5                    M6                        M7                     M8-M9
Manual                   USML discovers            DL learns features     Transformers learn
Human designs            patterns within X         via architecture       semantic features
features                 n → 1 (clustering)        n → m (embeddings)     from language
                         n → k (dim reduction)
                         n → k (topics)
```

**Unsupervised ML is automated feature engineering.** Supervised ML discovers combinations of X that predict y. Unsupervised ML discovers patterns _within_ X and represents them as new features. Deep learning uses architecture to _learn_ features through gradient optimization. This progression — manual, algorithmic, architectural, semantic — is the spine of the entire curriculum.

---

## Three Layers Per Concept

Every concept is taught at three depths simultaneously:

| Layer           | Marker                     | Audience                      | Example (Bias-Variance)                                                                                                                      |
| --------------- | -------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intuition**   | :green_circle: FOUNDATIONS | Zero-background professionals | _"Imagine throwing darts at a target. Bias is how far the center of your throws is from the bullseye. Variance is how spread out they are."_ |
| **Mathematics** | :blue_circle: THEORY       | Intermediate practitioners    | E[(y-y_hat)^2] = Bias^2(y_hat) + Var(y_hat) + sigma^2 — derived step by step, each term color-coded                                          |
| **Research**    | :purple_circle: ADVANCED   | Masters+ / PhD holders        | Double descent (Belkin et al., 2019): test error _decreases_ past the interpolation threshold in over-parameterized models                   |

A banker and a PhD sit in the same classroom. Both leave having learned something they didn't know.

---

## What's In The Box

| Component          | Count         | Details                                                            |
| ------------------ | ------------- | ------------------------------------------------------------------ |
| Lecture decks      | 10            | Reveal.js HTML, three-layer depth, KaTeX math, speaker notes       |
| Slides             | 1,333         | Every equation derived, every algorithm stepped through            |
| Speaker notes      | 6             | Per-slide timing, beginner tips, expert tangents                   |
| Exercises          | 48+           | Solutions + local + Jupyter + Colab (three-format consistency)     |
| Datasets           | 11            | Singapore-context: HDB 15M, taxi 50K, credit 100K, experiment 500K |
| Quizzes            | 6             | AI-resilient questions (context-specific, not recall)              |
| SDK Textbook       | 163 tutorials | 83 Python + 80 Rust, basic to advanced, every Kailash engine       |
| Assessment rubrics | 6             | Portfolio, capstone, peer review, model card template              |

### Verified Mathematical Content

Every derivation was red-teamed by specialist reviewers:

- Bias-variance decomposition (squared loss)
- XGBoost 2nd-order Taylor expansion + split gain formula
- EM algorithm (E-step responsibilities, M-step MLE, convergence)
- PCA eigendecomposition + SVD connection
- DPO 5-step derivation (RLHF objective through Z(x) cancellation)
- PPO clipped objective + Generalized Advantage Estimation
- Bellman equations (expectation + optimality)
- SHAP Shapley axioms (efficiency, symmetry, dummy, linearity)
- LSTM gate equations (all 6)
- Transformer scaled dot-product attention (why divide by sqrt d_k)

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

## Quick Start

```bash
git clone https://github.com/terrene-foundation/ascent.git
cd ascent
uv venv && uv sync
cp .env.example .env  # API keys for M9-M10

# Your first exercise
uv run python modules/ascent01/local/ex_1.py

# View lecture deck
open decks/ascent01/deck.html

# Advanced modules (DL, agents, alignment)
uv sync --extra full
```

### Three Delivery Formats

| Format       | Location                            | Best For                     |
| ------------ | ----------------------------------- | ---------------------------- |
| Local Python | `modules/ascent*/local/*.py`        | Full async, Nexus deployment |
| Jupyter      | `modules/ascent*/notebooks/*.ipynb` | Interactive exploration      |
| Google Colab | `modules/ascent*/colab/*.ipynb`     | Zero-install, GPU access     |

### Data Loading

```python
from shared import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdb_resale.parquet")
```

Auto-detects environment (local vs Colab) and pulls from Google Drive when needed.

---

## Repository Structure

```
ascent/
  modules/
    ascent01-10/        10 modules, each with:
      solutions/        Complete runnable solutions (instructor reference)
      local/            Python exercises (fill-in-the-blank)
      notebooks/        Jupyter notebooks
      colab/            Colab notebooks (Drive mount)
  decks/
    ascent01-10/        10 Reveal.js lecture decks (1,333 slides)
    assets/css/         Custom theme (teal/indigo, three-layer markers)
  textbook/
    python/             83 tutorials (all 8 Kailash packages)
    rust/               80 tutorials (all 9 Kailash packages)
  data/                 11 Singapore-context datasets (CSV/Parquet)
  shared/               Data loader, async wrappers, Kailash helpers
  scripts/              Notebook converter, dataset generator, deck auditor
```

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

**Solution-first authoring.** Solutions in `modules/ascent*/solutions/` are the source of truth. Exercises are generated by stripping to the appropriate scaffolding level.

**Progressive scaffolding.** M1 provides ~70% of code. M10 provides ~20%. By the end, students write from documentation alone.

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

Contributions welcome via [pull request](https://github.com/terrene-foundation/ascent/pulls). Questions via [issues](https://github.com/terrene-foundation/ascent/issues).

## License

[Apache 2.0](LICENSE). Specifications (CARE, EATP, CO) under CC BY 4.0.
