# MLFP Redlines — Non-Negotiable Quality Standards

These 13 principles are ABSOLUTE. No deviations without explicit justification. All `/redteam` audits MUST validate against every redline. Any violation is a BLOCKING finding.

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

### Full CI Guard Suite

Overflow is one of three classes of shipping bug caught by automated guards. `/redteam` and `/release` MUST run all three:

```bash
# 1. Visual overflow — slide content fits in 1280×720
node scripts/check-deck-overflow.js

# 2. Notebook syntax — every code cell AST-parses (students allowed `# TODO` / `____` blanks)
.venv/bin/python scripts/check_notebook_syntax.py modules/

# 3. Deck content parity — rebuild deck PDFs; pdftotext MUST match committed baseline
scripts/check-deck-parity.sh
```

The three guards catch disjoint failure modes:

| Guard                      | Catches                                                                                                        | Origin                             |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `check-deck-overflow.js`   | Slide content clipped below the 720-pixel fold after richness edits                                            | M5 deck, 2026-04-13                |
| `check_notebook_syntax.py` | Notebooks that ship with orphan tuples / broken cells (e.g. commit `6b28127` shipped 84 unusable student nbs)  | M5 self-contained audit 2026-04-17 |
| `check-deck-parity.sh`     | KaTeX / rendering regressions invisible to HTML diff (e.g. a broken edit to `modules/assets/js/katex-init.js`) | KaTeX extraction 2026-04-17        |

Unacknowledged failure in ANY guard BLOCKS `/redteam` convergence.

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

## Redline 10: One Technique, One Script, Full Value Chain — Exercise Directories

Every exercise is a DIRECTORY, not a single file. Each file in the directory covers ONE technique and tells the COMPLETE story: theory → build → train → visualise → real-world application. The learner never leaves the file to understand why a technique matters or to see it applied.

This redline mandates exercise structure. R9 mandates exercise content (visual proof + application). Together they ensure the learner experiences a complete narrative arc per technique, in a single script, with no context-switching between files.

### Structure

```
modules/mlfpNN/solutions/ex_N/
  _shared.py                    — common utilities (viz, data loading, training helpers)
  01_technique_name.py          — full value chain for technique 1
  02_technique_name.py          — full value chain for technique 2
  ...
  NN_grand_comparison.py        — (optional) side-by-side comparison of all techniques
```

Each `NN_technique_name.py` file MUST contain, in this order:

1. **Theory** — Why this technique exists. What problem it solves. The intuition a finance director would understand.
2. **Build** — The model/algorithm implementation (torch.nn.Module, sklearn pipeline, kailash-ml engine call).
3. **Train** — Training on the canonical dataset with ExperimentTracker logging.
4. **Visualise** — R9A visual proof (reconstruction grids, feature maps, attention heatmaps — whatever the technique demands).
5. **Apply** — R9B real-world application with business scenario, domain dataset, business interpretation, and stakeholder-ready output. This is NOT a separate file. It is the climax of the narrative.

### Why This Is Separate From R9

R9 says every exercise needs visual proof and real-world application. It does not say HOW to organise them. Before R10, an exercise could satisfy R9 by having `ex_1.py` (technique + visuals) and `ex_1_app_fraud.py` (application) as separate files. That breaks the narrative: the student learns the technique, closes the file, opens another file, and must rebuild context to see why the technique matters.

R10 forbids this separation. The application IS the payoff. It lives in the same script, right after the visualisation, while the student's understanding is fresh. One technique, one script, one unbroken narrative from "why does this exist?" to "here's how it saves S$2.3M in fraud annually."

### Audit Test

For every exercise directory:

1. Verify it IS a directory (not a single file)
2. For each technique file, verify all 5 phases are present (search for section markers or structural indicators: class/function definitions for Build, training loops for Train, `plt.` or viz calls for Visualise, business scenario text for Apply)
3. Verify no `ex_N_app_*.py` files exist alongside the directory — applications MUST be inline, not separate

```bash
# Check exercise structure — every ex_N should be a directory
for ex in modules/mlfp*/solutions/ex_*; do
  if [ -f "$ex" ] && [[ "$ex" == *.py ]]; then
    echo "BLOCKING: $ex is a file, not a directory"
  fi
done

# Check no separated application files exist
find modules/ -name "ex_*_app_*.py" -type f
# Any matches = BLOCKING
```

### BLOCKED

- Single-file exercises for any topic with more than one technique/variant. A single technique MAY be a single file if it genuinely has only one narrative arc.
- Separate `ex_N_app_*.py` application files. Applications are inline in the technique file.
- Technique files that stop at "Train" or "Visualise" without the "Apply" phase. If a technique has no real-world application, it doesn't belong in a professional course.
- Shared utility files that contain domain logic. `_shared.py` holds ONLY reusable infrastructure (viz functions, data loaders, training helpers). Business scenarios, domain datasets, and application logic live in the technique files.

**Origin**: Session 2026-04-13. M5 ex_1 was a monolithic 1,919-line file covering 10 AE variants with 7 separate application scripts (`ex_1_app_fraud.py`, `ex_1_app_medical.py`, etc.). The separation broke learner UX — students had to context-switch between files to see why each technique matters. Restructured to one technique per file with inline applications.

---

## Red Team Protocol

## Redline 11: Self-Contained Colab Is the Canonical Notebook Format

Every exercise ships in exactly TWO formats — VS Code `.py` and self-contained Colab `.ipynb`. No classic `colab/` (requires git clone + FORK_URL), no `colab-instructor/` (parallel maintenance burden), no `notebooks/` (Jupyter %pip format). One student format, one instructor format, full stop.

### Contract

| Format                                          | Audience   | Generation                                                              |
| ----------------------------------------------- | ---------- | ----------------------------------------------------------------------- |
| `modules/mlfpNN/local/ex_N/*.py`                | Student    | Source of truth (hand-authored with `____` / `# TODO` scaffolds)        |
| `modules/mlfpNN/solutions/ex_N/*.py`            | Instructor | Source of truth (complete, runnable)                                    |
| `modules/mlfpNN/colab-selfcontained/`           | Student    | Generated from `local/` by `scripts/generate_selfcontained_notebook.py` |
| `modules/mlfpNN/colab-selfcontained-solutions/` | Instructor | Generated from `solutions/`                                             |

Cell structure of every generated notebook:

- **Cell 0** — `!pip install` + `nest_asyncio.apply()` + GPU check (no git clone, no FORK_URL)
- **Cell 1** — Inlined shared helpers (collapsible; starts with `# MLFP inlined helpers — DO NOT EDIT`)
- **Cell 2+** — Exercise content with every `from shared.*` stripped and the exercise cells rewritten for `await` top-level

### BLOCKED

- Authoring `.ipynb` files by hand — notebooks MUST be generated from `.py` source. Hand-edits silently drift from local.
- Re-introducing `modules/mlfpNN/colab/` or `modules/mlfpNN/notebooks/` directories — the consolidation mandate is in force.
- Shipping a notebook whose Cell 1 does not execute cleanly in a fresh Python process with repo root as CWD.
- Shipping a solution notebook whose Cell 1 defines fewer symbols than the original helper — indicates stripping / flattening dropped content.

### Audit Test

```bash
# Parity: every .py has a matching .ipynb under both output dirs (ignoring __init__.py)
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  src=$(find modules/$m/solutions -name "*.py" ! -name "__init__.py" | wc -l)
  out=$(find modules/$m/colab-selfcontained-solutions -name "*.ipynb" | wc -l)
  [ "$src" = "$out" ] || echo "PARITY FAIL: $m src=$src out=$out"
done

# Cell 1 exec smoke-test on every solution notebook
.venv/bin/python scripts/check_notebook_syntax.py modules/
```

**Origin**: Session 2026-04-17. Commit `6b28127` shipped 84 M5 self-contained notebooks with `SyntaxError` in Cell 3 because an un-committed generator stripped only the first line of multi-line `from shared.X import (\n  A,\n  B,\n)` imports, leaving an orphan tuple. Band-aid applied, then superseded by a proper committed generator (`scripts/generate_selfcontained_notebook.py`) with AST validation. Consolidation to self-contained-only landed in commit `8696560`.

## Redline 12: Class-Based Equation Markup and Shared Deck Assets

Every deck math expression uses class-based markup, not dollar-delimiter syntax. Shared deck assets (KaTeX renderer, CSS themes) live in `modules/assets/` and are sourced by every deck via relative path.

### Equation markup contract

```html
<!-- Display equation -->
<div class="equation-box">
  <span class="katex-display">E = mc^2</span>
</div>

<!-- Inline equation -->
as shown in <span class="katex-inline">\nabla f(x)</span> above.
```

**BLOCKED** (these break the idempotent renderer):

- `$E = mc^2$` / `$$E = mc^2$$` — dollar delimiters (triggers KaTeX auto-render, conflicts with the idempotent renderer)
- `<span class="katex-display">...</span>` outside an `.equation-box` parent (renderer contract requires the wrapper)
- Inline MathJax / custom KaTeX init blocks per-deck — use the shared script below

### Shared asset contract

All decks include the shared KaTeX renderer as the LAST `<script>` before `</body>`:

```html
<!-- Idempotent KaTeX renderer — shared across decks. See modules/assets/js/README.md. -->
<script src="../assets/js/katex-init.js"></script>
```

The renderer at `modules/assets/js/katex-init.js` handles display + inline equations idempotently (pre-captures LaTeX source via `data-mlfp-source`, marks rendered elements via `data-mlfp-rendered`) so repeated `slidechanged` events don't compound corruption. See `modules/assets/js/README.md` for the full contract.

### Audit test

```bash
# Delimiter-style math in any deck = BLOCKING (other than inside <pre><code>)
grep -nE '\$\$?[^$]+\$\$?' modules/mlfp*/deck.html | grep -v '<code>'

# Every deck using class-based markup must reference the shared renderer
for d in modules/mlfp*/deck.html; do
  grep -l "equation-box" "$d" >/dev/null && \
    grep -q 'katex-init.js' "$d" || echo "MISSING shared renderer: $d"
done
```

**Origin**: Session 2026-04-17. M6 deck crashed Chromium during `decktape` PDF export due to a double-render bug where authored `<span class="katex-display">` collided with the nested span KaTeX creates. Fixed with an idempotent renderer; extracted to `modules/assets/js/katex-init.js` in commit `bd7a1c0` so future decks inherit the fix without a per-deck copy.

## Redline 13: Shared Package Structure and Transitive Inlining

The `shared/` package is the canonical home for helper code. Exercises import FROM it; no exercise defines infrastructure locally. The self-contained notebook generator inlines the closure of `shared/` imports into Cell 1.

### Package layout

```
shared/
├── __init__.py                  # Re-exports MLFPDataLoader, get_device, run_profile
├── kailash_helpers.py           # Env setup, connection managers, common Kailash glue
├── data_loader.py               # MLFPDataLoader (Drive + local + gdown)
├── run_profile.py               # Alert / comparison helpers
├── mlfp02/                      # Module 2 helpers (polars-native)
│   ├── __init__.py
│   ├── ex_1.py                  # One helper module per exercise
│   ├── ex_2.py
│   └── ...
├── mlfp03/ … mlfp05/            # Same structure
└── mlfp06/                      # Module 6 — has a diagnostics subpackage
    ├── __init__.py
    ├── ex_1.py … ex_8.py
    └── diagnostics/             # LLM Observatory subpackage
        ├── __init__.py
        ├── _judges.py            # Leaf utilities (inlined first)
        ├── _plots.py
        ├── _traces.py
        ├── retrieval.py         # Higher-level lenses (inline after leaves)
        ├── output.py
        ├── alignment.py
        ├── interpretability.py
        ├── governance.py
        ├── agent.py
        └── observatory.py       # Top-level facade (inline last)
```

### Transitive inlining contract

The generator MUST walk the `shared.*` import graph to fixpoint. Missing any transitive dependency yields a runtime `NameError`.

- `from shared.mlfp02.ex_1 import load_customer_data` → inline `shared/mlfp02/ex_1.py`
- `shared/mlfp02/ex_1.py` imports `from shared.data_loader import MLFPDataLoader` → inline `shared/data_loader.py`
- `from shared import MLFPDataLoader` → resolve through `shared/__init__.py`, which re-exports from `shared.data_loader`

### Colab safety rewrites

Helpers written with `REPO_ROOT = Path(__file__).resolve().parents[N]` break in Colab (where `__file__` is unset during cell execution) and in tempfile test harnesses (where `.parents[N]` escapes the filesystem root). The generator MUST rewrite this pattern to `Path.cwd()` (which is `/content` in Colab, writable, consistent).

### Subpackage flattening (M6 diagnostics)

When inlining a subpackage, the generator MUST:

1. Strip `from . import x` and `from .x import y` (relative imports — the symbols are co-located after concatenation)
2. Flatten module-style references: `_plots.PRIMARY` → `PRIMARY` (since `_plots.py` contributes `PRIMARY` at top level after inlining)
3. Order inline by dependency: leaves first (`_judges`, `_plots`, `_traces`), then higher-level modules that reference them

### Audit test

```bash
# Every solution notebook Cell 1 runs clean in a fresh Python process (CWD = repo root)
python3 scripts/check_notebook_syntax.py modules/
# Plus a manual spot-check for each module: exec Cell 1 and confirm N symbols defined
```

**Origin**: Session 2026-04-17 red team. Three generator bugs surfaced by Cell 1 execution testing: (a) `NameError: MLFPDataLoader` on M2/M4 notebooks because transitive imports were not walked; (b) M6 `ImportError: attempted relative import with no known parent package` from flattened subpackage; (c) M5 ex_6/ex_7 `OSError: Read-only file system: '/data'` because `Path(__file__).parents[2]` escaped to `/`. All three fixed in commit `38549f1`; generator now validates every cell via AST + executes Cell 1 as part of the check.

---

## Red Team Protocol

Every `/redteam` run MUST validate ALL 13 redlines for the module(s) under review. The red team report MUST include a section for each redline with:

1. **Status**: PASS / FAIL / PARTIAL
2. **Evidence**: Specific files, line numbers, measurements
3. **Gaps**: What's missing
4. **Fix required**: Concrete action to remediate

A module with ANY redline FAIL cannot ship.
