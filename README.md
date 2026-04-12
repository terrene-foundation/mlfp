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

| Component                 | Details                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| **Master lecture decks**  | 6 Reveal.js HTML decks (541 slides total), three-layer depth, KaTeX math                 |
| **Master textbooks**      | 6 markdown chapters (~18,200 lines), full derivations, worked examples, glossaries       |
| **Master speaker notes**  | 6 markdown files (~7,300 lines), per-slide timing, beginner/expert cues                  |
| **Per-lesson HTML**       | 144 files (48 lessons × 3 formats): standalone `textbook.html`, `slides.html`, `notes.html` |
| **Coding exercises**      | 48 solutions + scaffolded `local/*.py` + `colab/*.ipynb` (144 exercise files)            |
| **Datasets**              | Singapore-context: HDB resale (50K), ICU patient data, weather, taxi, economic indicators |
| **Red-team reports**      | 6 independent audits (math, pedagogy, student UX, spec compliance, QA, naming)           |
| **End-of-module quizzes** | AI-resilient questions (context-specific, not recall)                                    |

### Exercise Validation

All exercises validated in a fresh `uv venv` with `kailash 2.8.4 + torch 2.11 + full ML stack`:

| Module | Exercises | Status | Notes                                    |
| ------ | --------- | ------ | ---------------------------------------- |
| M1     | 8/8       | ✅ PASS | Polars pipelines, no external deps       |
| M2     | 8/8       | ✅ PASS | Statistics, FeatureStore degrades gracefully |
| M3     | 8/8       | ✅ PASS | sklearn + XGBoost/LightGBM/CatBoost + SHAP |
| M4     | 8/8       | ✅ PASS | Clustering, NLP, recommenders, PyTorch DL |
| M5     | 8/8       | ✅ PASS | PyTorch nn.Module (runs in 34s total)    |
| M6     | —         | Requires API keys (OPENAI_API_KEY, HF_TOKEN) |

---

## Quick Start

```bash
git clone https://github.com/terrene-foundation/mlfp.git
cd mlfp
uv venv && uv sync
cp .env.example .env  # API keys for M6 (OPENAI_API_KEY, HF_TOKEN)

# First exercise (no setup — Polars + Kailash only)
uv run python modules/mlfp01/solutions/ex_1.py

# View the first lecture deck
open decks/mlfp01/deck.html

# Or read the textbook chapter for self-study
open decks/mlfp01/lessons/01/textbook.html
```

### Three Ways to Consume Each Lesson

Every lesson ships as **three independent HTML files**, so instructors, students, and self-learners each get the right tool:

| Format           | File                                       | Audience       | Layout    |
| ---------------- | ------------------------------------------ | -------------- | --------- |
| **Slides**       | `decks/mlfpNN/lessons/LL/slides.html`      | Lecturers      | Landscape (Reveal.js 1280×720) |
| **Textbook**     | `decks/mlfpNN/lessons/LL/textbook.html`    | Self-learners  | Portrait (long-form reading)    |
| **Speaker notes**| `decks/mlfpNN/lessons/LL/notes.html`       | Instructors    | Portrait (per-slide cues)       |

A module-level `deck.html` concatenates all 8 lessons into one Reveal.js deck for a continuous 3-hour session. A `textbook.md` / `speaker-notes.md` provides the markdown equivalents for instructors who prefer a single file.

### Exercise Delivery Formats

| Format           | Location                              | Best For                                |
| ---------------- | ------------------------------------- | --------------------------------------- |
| **Solutions**    | `modules/mlfpNN/solutions/ex_N.py`    | Complete reference (instructor)         |
| **Scaffolded**   | `modules/mlfpNN/local/ex_N.py`        | VS Code fill-in-the-blank (learner)     |
| **Colab**        | `modules/mlfpNN/colab/ex_N.ipynb`     | Zero-install, GPU access (learner)      |

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
    mlfp01-06/                    6 modules, 8 exercises each
      solutions/ex_1-8.py         Complete reference implementations
      local/ex_1-8.py             Scaffolded fill-in-the-blank
      colab/ex_1-8.ipynb          Colab notebooks
      notebooks/ex_1-8.ipynb      Jupyter notebooks
      README.md                   Module description
  decks/
    mlfp01-06/                    Per-module content
      deck.html                   Master 78-100 slide Reveal.js deck
      speaker-notes.md            Master instructor notes (markdown)
      textbook.md                 Master textbook chapter (markdown)
      index.html                  Module landing page with lesson TOC
      lessons/01-08/              Per-lesson HTML files
        textbook.html             Standalone textbook chapter (portrait)
        slides.html               Standalone slide deck (landscape)
        notes.html                Standalone speaker notes (portrait)
    assets/
      css/theme.css               Reveal.js theme (--mlfp-* variables)
      css/textbook.css            Portrait reading theme
      templates/                  HTML templates for lesson generation
  data/
    mlfp01-04/                    Singapore-context datasets
    mlfp_assessment/              Capstone assessment datasets
  shared/                         Data loader, Kailash helpers
  specs/                          Authoritative v2 course specs
    module-1.md .. module-6.md    Per-module lesson specs
    design-principles.md          Feature Engineering Spectrum, 3-layer system
    exercise-mapping.md           Exercise-to-lesson map
  workspaces/curriculum/
    04-validate/                  Red-team audit reports (rt1-rt6)
  scripts/
    generate-pdfs.sh              Build PDFs for all HTML deliverables
    generate_datasets.py          Regenerate datasets (reproducible seed)
  pdf/                            Generated PDFs (gitignored, build locally)
```

### Build PDFs

All 6 master decks, 6 textbook chapters, 6 speaker notes, and 144 per-lesson files can be rendered to PDF via Chrome headless:

```bash
./scripts/generate-pdfs.sh
# → pdf/decks/         (6 landscape master decks)
# → pdf/textbooks/     (6 portrait textbook chapters)
# → pdf/notes/         (6 portrait speaker notes)
# → pdf/lessons/       (144 per-lesson PDFs: textbook/slides/notes × 48 lessons)
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

Pinned stack (verified in fresh venv — see `pyproject.toml`):

| Package                                                     | Version | Purpose                                  |
| ----------------------------------------------------------- | ------- | ---------------------------------------- |
| [kailash](https://github.com/terrene-foundation/kailash-py) | 2.8.4   | Workflow orchestration, 140+ nodes       |
| kailash-ml                                                  | 0.9.0   | 13 ML engines + RLTrainer, polars-native |
| kailash-dataflow                                            | 2.0.6   | Zero-config database operations          |
| kailash-nexus                                               | 2.0.1   | API + CLI + MCP deployment               |
| kailash-kaizen                                              | 2.7.3   | AI agent framework, signatures, A2A      |
| kaizen-agents                                               | 0.9.2   | ReActAgent, RAGResearchAgent, etc.       |
| kailash-pact                                                | 0.8.1   | D/T/R governance, operating envelopes    |
| kailash-align                                               | 0.3.1   | SFT, DPO, QLoRA, GRPO fine-tuning        |

Plus the classical/deep ML stack (`pip install` or `uv sync`):
`torch 2.11`, `torchvision`, `pytorch-lightning`, `polars 1.x`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `imbalanced-learn`, `gymnasium`, `mlxtend`.

---

## For Instructors

**Solution-first authoring.** Solutions in `modules/mlfp*/solutions/` are the source of truth. Exercises are generated by stripping to the appropriate scaffolding level.

**Progressive scaffolding.** M1 provides ~70% of code. M6 provides ~20%. By the end, students write from documentation alone.

**Three-layer delivery.** Green/blue/purple markers let instructors calibrate depth in real time. Beginners follow green. Experts engage with purple. Everyone sees the same deck.

**Speaker notes.** Per-slide timing, common student questions, "if beginners look confused" alternatives, "if experts look bored" tangents. Available as both a per-module master file (`speaker-notes.md`) and per-lesson HTML (`notes.html`) with audience cues and transition lines.

**Red-team validated.** Six independent audits in `workspaces/curriculum/04-validate/`:
- `rt1-math-rigor.md` — 110+ formulas verified, full derivations checked
- `rt2-pedagogical-flow.md` — lesson progression, scaffolding consistency
- `rt3-student-experience.md` — zero-Python beginner accessibility
- `rt4-spec-compliance.md` — topic coverage across 456 spec items
- `rt5-exercise-alignment.md` — exercise ↔ textbook alignment (all 48 pairs)
- `rt6-naming-branding.md` — naming, cross-references, navigation

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
