# Design Principles

Authoritative design principles governing the MLFP curriculum (extracted from mlfp-curriculum-v2.md Part I).

---

## 1. Kailash-Only Stack

Every ML operation uses Kailash engines. No PyCaret (broken deps, slow install). No ydata_profiling (conflicts). No raw sklearn/pandas/PyTorch without Kailash wrapper.

| Instead of                 | Use                                              |
| -------------------------- | ------------------------------------------------ |
| ydata_profiling / Sweetviz | DataExplorer (8 alert types, profiling, compare) |
| PyCaret / AutoML           | AutoMLEngine, TrainingPipeline                   |
| pandas                     | polars                                           |
| sklearn Pipeline           | PreprocessingPipeline                            |
| MLflow                     | ExperimentTracker + ModelRegistry                |
| CrewAI / LangChain         | Kaizen (Delegate, BaseAgent, Signature)          |

## 2. Python Through Data

Every Python concept grounded in data manipulation from Lesson 1.1. No abstract exercises. Students never learn a `for` loop in isolation — they learn it by iterating over data.

## 3. Engines Before Infrastructure

Students use Kailash engines (DataExplorer, ModelVisualizer) in M1 before learning workflow orchestration (WorkflowBuilder, custom nodes) in M3. Motivation before machinery.

## 4. The Feature Engineering Spectrum

The organising spine of the entire curriculum (from R5 Deck 5B "Unsupervised meets Supervised Learning"):

```
M3:     Manual feature engineering — human designs features from domain knowledge
M4.1-6: USML discovers features independently — no labels, no error signal
M4.7:   Collaborative filtering learns embeddings via optimisation — THE PIVOT
M4.8:   DL generalises — hidden layers are automated feature engineering WITH error feedback
        2+ hidden layers can write any non-linear combination (representation learning)
        Node values = embeddings
M5:     Specialised architectures learn domain-specific features (vision, sequence, graph, generative)
M6:     LLMs learn semantic features from language at scale
```

## 5. Statistics Teaches Models, SML Teaches the Pipeline

Following R5 delivery (Deck 3A): regression, logistic regression, and ANOVA are taught as inferential statistics (M2) BEFORE the ML pipeline (M3). M3 does NOT re-teach these models — it builds on them with advanced ensembles and production engineering.

## 6. Engineering, Not Philosophy

Every concept is taught through implementation. Governance is access controls you code (PACT), not frameworks you discuss. Alignment is a training loop you run (DPO/GRPO), not a position paper you read.

## 7. Progressive Scaffolding

| Module | Scaffolding | Code Provided | Audience State               |
| ------ | ----------- | ------------- | ---------------------------- |
| M1     | Heavy       | ~70%          | Zero Python                  |
| M2     | Moderate+   | ~60%          | Can wrangle data             |
| M3     | Moderate    | ~50%          | Understands statistics       |
| M4     | Light+      | ~40%          | Can build supervised models  |
| M5     | Light       | ~30%          | Understands USML + DL basics |
| M6     | Minimal     | ~20%          | Fluent in DL architectures   |

## 8. Three Layers Per Concept

Every concept taught at three depths simultaneously:

| Layer       | Marker      | Audience        | How to Use                               |
| ----------- | ----------- | --------------- | ---------------------------------------- |
| Intuition   | FOUNDATIONS | Zero-background | Plain language, analogies, visualisation |
| Mathematics | THEORY      | Intermediate    | Derivation, step-by-step formula         |
| Research    | ADVANCED    | Masters+ / PhD  | Paper reference, frontier result         |

A banker and a PhD sit in the same classroom. Both leave having learned something they didn't know.

## 9. Delivery Architecture — One Deck, One Truth

### The Reading Trinity

Students receive three reading materials plus exercises:

| Material          | Format                       | Purpose                                                   |
| ----------------- | ---------------------------- | --------------------------------------------------------- |
| **Deck** (slides) | Reveal.js HTML               | Visual teaching, classroom delivery, instructor reference |
| **Textbook**      | Long-form document           | Deep reading, worked examples, extended explanations      |
| **Student notes** | Condensed guide              | Study aid, revision, key formula reference                |
| **Exercises**     | .py (local) + .ipynb (Colab) | Hands-on practice (two-format, see `rules/two-format.md`) |

The deck, textbook, and student notes form the complete reading materials. Exercises are separate practice artifacts.

### Deck Is the Source of Truth for Slides

Each module has ONE authoritative slide deck: `modules/mlfpNN/deck.html`. This deck contains ALL slides for all 8 lessons at full richness — three-layer markers, speaker notes, callout boxes, diagrams, Kailash bridge slides, and module-level preamble (overview, journey map, engine reference).

Individual lesson files (`modules/mlfpNN/lessons/NN/slides.html`) are **extractions** from the deck for per-lesson navigation. They are NEVER authored independently. The relationship is:

```
deck.html (source of truth, full richness)
  ├── extract → lessons/01/slides.html
  ├── extract → lessons/02/slides.html
  ├── ...
  └── extract → lessons/08/slides.html
```

**Improvement flow**: All slide improvements (diagrams, content, speaker notes, markers) go into `deck.html` first. Individual lesson files are re-extracted after the deck is updated. Never the reverse.

**BLOCKED**:

- Authoring lesson slides independently from the deck (creates divergence)
- Improving lesson slides without updating the deck (sparse copies diverge from rich source)
- Rebuilding the deck by concatenating lesson slides (destroys richness — speaker notes, markers, preamble)
- Maintaining two quality tiers (sparse lessons + rich deck)

### Every Slide Must Be Instructor-Ready

Every slide in the deck MUST include:

1. **Three-layer marker** — Foundations (green), Theory (blue), or Advanced (purple) badge indicating audience level
2. **Speaker notes** — Timing, talking points, transition cues, what to emphasize for different audience levels
3. **Slide footer** — Module and course identification
4. **Visual content** — Diagrams, illustrations, or code blocks that teach (see Redline 2). Text-only slides are acceptable only for reflection/summary.

### PDF Export

Decks export to PDF via `scripts/export-pdf.sh` using decktape. The PDF is the student-facing version (speaker notes excluded). PDF files live in `pdf/lessons/mlfpNN/NN/slides.pdf`.
