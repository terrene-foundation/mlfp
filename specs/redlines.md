# MLFP Redlines — Non-Negotiable Quality Standards

These 8 principles are ABSOLUTE. No deviations without explicit justification. All `/redteam` audits MUST validate against every redline. Any violation is a BLOCKING finding.

## Redline 1: 25-Hour Depth Per Module

Each module is 25 contact hours. Current coverage that can be completed in 10% of the time is FAILING. Content must fill 25 hours of hands-on professional instruction with depth, not padding.

**Audit test**: If a competent professional could complete all exercises, read all slides, and answer all quiz questions in under 20 hours, the module is underweight.

**BLOCKED**: Shallow exercises that take 10 minutes. Slides that summarize without teaching. Quiz questions that test recall instead of application.

## Redline 2: Illustrations, Diagrams, and Real-World Applications

Math without visual intuition kills the course on day 1. The target audience is non-science, non-tech domain experts (finance, healthcare, logistics, operations). Every mathematical concept MUST have:

1. **Visual illustration** — diagrams, plots, geometric intuitions
2. **Real-world application** — from the learner's domain, not toy academic examples
3. **Business framing** — why this matters for their decisions, not just how it works

**Audit test**: Show a slide to a finance director with no CS background. If they can't understand the intuition in 30 seconds, the slide fails.

**BLOCKED**: Equations without diagrams. Theory without business context. Academic framing ("consider the hypothesis space") instead of practical framing ("this tells you whether your marketing campaign actually worked").

## Redline 3: Visual Deck Quality — No Overflows

Slides MUST be visually inspected by an automated headless-browser check. Common failures:

- Text/code overflowing the bottom of the slide
- Content cut off or hidden below the viewport
- Fonts too small to read in a classroom
- Code blocks that don't fit the slide width
- Tables that overflow horizontally
- Diagrams added to dense slides pushing existing content out of frame

**Audit test (automated)**: Every slide rendered at 1280×720 in headless Chrome by `scripts/check-deck-overflow.js` (puppeteer). Any `<section>` whose `scrollHeight > 720` is BLOCKING. Run on every PR and before every release:

```bash
# Check every deck
node scripts/check-deck-overflow.js

# Check one module + save screenshots of overflowing slides
node scripts/check-deck-overflow.js modules/mlfp05 --screenshots

# JSON output for CI
node scripts/check-deck-overflow.js --json

# Wired into the Python redline runner (visual check runs by default):
.venv/bin/python scripts/redline-check.py            # visual + static
.venv/bin/python scripts/redline-check.py --no-visual  # static only
```

The script catches the failure mode where a slide _renders_ (no JavaScript error) but content is clipped at the bottom edge — invisible to a static HTML parse.

**Origin**: M5 deck overhaul session 2026-04-13. After adding 20 SVG diagrams + 8 application slides, two GNN slides clipped diagrams below the fold. The script was created to prevent regression and to surface this failure across all 6 module decks.

## Redline 4: Exercise Coverage = Deck Coverage

Every concept mentioned in the deck MUST have a corresponding exercise. If the deck covers 100 concepts, there must be 100 exercise coverage points. Exercises MUST be application-based and real-world.

**Audit test**: Extract every concept/technique/API from the deck. Map each to an exercise. Any unmapped concept = gap finding.

**BLOCKED**: Exercises that only cover "the easy 30%" of the deck. Exercises that demonstrate toy usage instead of real application.

## Redline 5: Massive, Dirty, Real-World Data

No toy datasets. No simulated data. All datasets MUST be:

1. **Massively large** — millions of rows, hundreds of features where applicable
2. **Sourced from the internet** — publicly available, downloadable, real
3. **Dirty** — missing values, outliers, duplicates, inconsistent formats, encoding issues, mixed types — because that's how real data arrives
4. **Multi-modal in M4-M6** — images, audio, video, language data MUST appear

If data must be generated (rare), it must be massive (millions of rows) and realistic (domain-appropriate noise, missing values, correlations that mirror real distributions).

**Audit test**: `df.shape` on every dataset. Under 100K rows for tabular = suspect. Under 1M for production exercises = weak. Clean data with no nulls = fake.

**BLOCKED**: Iris, Boston Housing, tips, titanic, wine quality, or any dataset that fits in memory in under 1 second. Synthetic data that's perfectly clean.

## Redline 6: Comprehensive 3-Hour Exam

The exam is 3 hours long. It MUST test coding ability to the maximum.

- **No MCQ** — multiple choice is banned
- **Coding-intensive** — students must write, debug, and optimize real code
- **Comprehensive** — covers the full module, not cherry-picked easy topics
- **Cannot be solved easily** — requires integration of multiple concepts, real problem-solving, and deep understanding

**Audit test**: Give the exam to a strong ML engineer. If they finish in under 90 minutes, it's too easy. If they can pass by memorizing API calls, it tests the wrong thing.

**BLOCKED**: MCQ. Fill-in-the-blank with one-line answers. Exercises that can be solved by copy-pasting from slides.

## Redline 7: Kailash Engines First, Always

Every ML operation MUST use the Kailash engine if one exists. The 13 kailash-ml engines, Kaizen agents, Align pipelines, PACT governance, DataFlow, Nexus — these are the primary tools.

For operations NOT covered by Kailash: use `numba`, `torch`, `pytorch-lightning`, and optimized production packages. No toy packages.

**BLOCKED packages**:

- `mlflow` (reimplemented as `ExperimentTracker` + `ModelRegistry`)
- `pycaret` (reimplemented as `AutoMLEngine`)
- `pandas` (replaced by `polars`)
- Any package whose capability exists in kailash-ml

**Audit test**: `grep -r "import mlflow\|import pycaret\|import pandas" modules/`. Any match = BLOCKING.

**Verification**: Before using a non-kailash package, delegate to ml-specialist/kaizen-specialist/align-specialist to confirm no kailash equivalent exists.

## Redline 8: GPU Usage Is Mandatory

All deep learning and compute-intensive operations MUST use GPU acceleration.

- macOS: `mps` (Metal Performance Shaders)
- Linux/Cloud: `cuda`
- Device selection must be automatic with graceful fallback

```python
# Standard device selection pattern
import torch
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
```

**Audit test**: Any `torch` training loop without device placement = BLOCKING. Any model that trains on CPU when GPU is available = BLOCKING.

## Redline 9: Seeing Is Believing — Visual Intuition and Real-World Application in Every Exercise

The target audience is senior professionals (finance, healthcare, logistics, operations) who learn by SEEING results and APPLYING techniques to their domain. They do not learn from watching loss curves fall or reading metric tables. Every exercise MUST build intuition through visual proof AND connect every technique to a real-world professional objective.

This redline has two inseparable components:

### 9A: Visual Proof — Show, Don't Tell

Every model, algorithm, or technique MUST include a "seeing is believing" visualization that makes the result tangible to a non-technical professional. Abstract metrics (loss, accuracy, F1) are necessary but NOT sufficient — the learner must SEE the model's actual behaviour.

**Required visualization patterns by technique type:**

| Technique                | MUST show                                                                                                                                         | NOT sufficient                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| Autoencoders             | Original vs reconstructed image grid (2×10 subplot). For denoising: 3 rows (original, noisy, cleaned). For VAE: sampled images from latent space. | Loss curve alone                |
| CNNs                     | Learned filters, feature maps at each layer, Grad-CAM heatmap on a test image showing WHERE the model looks                                       | Accuracy number alone           |
| RNNs/LSTM                | Predicted vs actual time series overlay, attention weight heatmap over timesteps                                                                  | Loss curve alone                |
| Transformers             | Self-attention matrix heatmap per head, token-level attention visualization on example sentence                                                   | Accuracy number alone           |
| GANs                     | Gallery of generated images (8×8 grid), latent space interpolation (walk between two generated images)                                            | FID score alone                 |
| GNNs                     | Graph with node embeddings coloured by predicted class, attention edge weights visualized                                                         | Node accuracy alone             |
| Transfer Learning        | Layer activation comparison (pretrained vs fine-tuned), data efficiency curve (accuracy vs training set size)                                     | Accuracy comparison table alone |
| RL                       | Agent behaviour visualization (episode replay or state trajectory), reward landscape                                                              | Reward curve alone              |
| Clustering               | Scatter plot with cluster assignments coloured, dendrogram, silhouette plot per sample                                                            | Silhouette score alone          |
| Regression               | Predicted vs actual scatter (45-degree line), residual plot, coefficient visualization                                                            | R² number alone                 |
| Classification           | Decision boundary plot, confusion matrix heatmap, ROC curve, calibration plot                                                                     | Accuracy number alone           |
| Dimensionality Reduction | 2D/3D scatter coloured by class, reconstruction error vs components plot                                                                          | Variance explained table alone  |
| Feature Engineering      | Feature importance bar chart, SHAP summary plot, partial dependence plot                                                                          | Feature list alone              |
| Statistics               | Distribution plot with confidence interval bands, bootstrap distribution histogram, prior→posterior animation                                     | p-value alone                   |

**Audit test**: For every exercise, identify the primary model/technique. Search for `plt.subplots`, `plt.imshow`, `fig.add_trace`, `viz.scatter`, `viz.histogram`, or equivalent visualization calls that render model BEHAVIOUR (not just training loss). Zero behaviour visualizations = BLOCKING.

**BLOCKED**: Exercises where the only visual output is a loss/accuracy curve or a printed metric table. These are debugging tools, not learning tools. They MUST exist (for engineering practice) but they are NOT the intuition-building visualization.

### 9B: Real-World Application — Every Technique Answers a Professional Question

Every exercise MUST include a section where the technique is applied to a realistic professional scenario with a business question that the learner's manager would ask. The application must be CONCRETE (named industry, named dataset, named decision) not GENERIC ("this is useful in many industries").

**Required per exercise:**

1. **Business scenario** (2-3 sentences): who you are, what problem you face, what decision depends on this analysis
2. **Domain dataset**: real or realistic data from the learner's world (Singapore HDB, bank transactions, hospital records, retail sales, logistics routes) — NOT academic benchmarks (MNIST, CIFAR, AG News) used in isolation
3. **Business interpretation**: after every key result, a paragraph explaining what this means for the professional decision — not "accuracy is 0.92" but "this model correctly identifies 92% of fraudulent transactions, which at DBS's transaction volume would catch an additional S$2.3M in fraud annually"
4. **Comparison to status quo**: "without this model, the bank's rule-based system catches 67% — the ML model adds 25 percentage points"
5. **Stakeholder-ready output**: at least one visualization or summary table that could be shown directly to a non-technical executive

**Application examples that MUST appear (or equivalent quality):**

- **Autoencoders**: credit card fraud detection (train on normal transactions, flag high reconstruction error), with a precision-recall analysis at different thresholds and a dollar-value impact estimate
- **CNNs**: medical image screening or manufacturing defect detection, with Grad-CAM showing what the model focuses on and a cost-benefit analysis of automated vs manual inspection
- **RNNs**: stock price or demand forecasting for a real Singapore equity or FMCG product, with prediction intervals and a trading/ordering decision framework
- **Clustering**: customer segmentation on real retail data, with segment profiles a marketing director can act on
- **Regression**: property price prediction on HDB data, with interpretable coefficients a property investor can understand

**Audit test**: For every exercise, search for a business scenario section. Check that it names a specific industry, a specific decision, and includes a quantified impact statement. Generic theory-only exercises = BLOCKING.

**BLOCKED**: Exercises that train on Fashion-MNIST/CIFAR/AG News and stop there. Academic benchmarks are acceptable for teaching the technique, but every exercise MUST ALSO apply the technique (or its results) to a professional scenario in the same exercise.

### Why this is a separate redline (not covered by R2 or R4)

Redline 2 requires "illustrations and diagrams" — that covers the DECK (slides), not the EXERCISES. Redline 4 requires "exercise coverage = deck coverage" — that ensures topics are covered, not HOW they are taught. Redline 9 mandates that the EXERCISES themselves build intuition through visual proof AND professional application. An exercise can satisfy R2 (the deck has a diagram), R4 (the exercise covers the topic), R7 (uses Kailash engines), and still FAIL R9 because the student never SEES the model work and never APPLIES it to their job.

**Origin**: Session 2026-04-13. Comparison of DIS PCML6-1 autoencoder notebook (10 variants, each with original-vs-reconstruction image grid) against MLFP M5 ex_1 (4 variants, zero image visualizations, only loss curves). The DIS notebook builds intuition in 24 cells that MLFP's 790-line exercise fails to build because it substitutes engineering metrics for visual proof. Extended to the full curriculum after audit revealed the same pattern across all 48 exercises.

---

## Red Team Protocol

Every `/redteam` run MUST validate ALL 9 redlines for the module(s) under review. The red team report MUST include a section for each redline with:

1. **Status**: PASS / FAIL / PARTIAL
2. **Evidence**: Specific files, line numbers, measurements
3. **Gaps**: What's missing
4. **Fix required**: Concrete action to remediate

A module with ANY redline FAIL cannot ship.
