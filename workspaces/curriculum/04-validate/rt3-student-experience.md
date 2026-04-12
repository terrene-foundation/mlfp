# RT3: Student Experience Audit — Zero-Python Beginner Perspective

**Auditor**: Quality Reviewer (zero-Python beginner persona)
**Date**: 2026-04-13
**Scope**: MLFP Module 1 (all 8 lessons), M2 L01 bridge, M3 L01 bridge

---

## Executive Summary

Module 1 is **exceptionally well-crafted** for zero-Python beginners. Every lesson opens with a motivating real-world question, introduces concepts with concrete Singapore examples, and builds on what came before. The progressive disclosure — from 12 rows of weather data to 500,000 rows of HDB transactions — is pedagogically sound. The callout system (Foundations, Theory, Advanced) gives three reading paths and lets beginners skip ahead without shame.

**Overall M1 score: Accessibility 4.2/5, Clarity 4.4/5**

The cross-module bridges (M2 L01, M3 L01) are where the difficulty spike hits hardest. M2 L01 is a steep but manageable jump; M3 L01 assumes comfort with sklearn and numpy that was never taught.

---

## Per-Lesson Findings

### Lesson 1.1 — Your First Data Exploration

**Accessibility: 5/5 | Clarity: 5/5**

This is a model lesson for beginners. It accomplishes something remarkable: teaching variables, types, DataFrames, and summary statistics in one sitting without feeling rushed.

**Strengths:**
- Opens with "In the next 45 minutes you will write your first lines of Python" — sets scope and time expectation immediately
- The three-step diagram (Load / Inspect / Summarise) gives a mental model before any code appears
- Variables explained through a concrete mental model ("a label stuck onto a value") not abstract CS theory
- The weather dataset has only 12 rows — small enough to trace every number by hand
- Every code block shows its output so students know what to expect
- The `describe()` output is read aloud line by line: "Read those values aloud: '12 rows, 3 columns...'"
- The Foundations/Theory/Advanced callout system is introduced naturally

**"I got lost here" moments:**
- *None for L1.1.* This is the gold standard for a first lesson.

**Minor note:** The f-string format spec table (`.0f`, `.2f`, `:,.0f`, etc.) is a lot of information to absorb in a first lesson. A beginner will just use `:.1f` and move on, which is fine — the table is reference material, not required knowledge.

---

### Lesson 1.2 — Filtering and Transforming Data

**Accessibility: 4/5 | Clarity: 5/5**

The jump from 12 rows to 500,000 is well-handled because the lesson immediately frames it as "your first job is to throw away the 899,700 you do not need." This makes the scale feel manageable rather than intimidating.

**Strengths:**
- Boolean operators table is crystal clear, with examples and results
- The Foundations callout re-explaining `=` vs `==` is exactly right — repetition of the #1 beginner error
- The diagram showing "Expression (a recipe)" vs "Method (applies recipe)" is a major conceptual unlock
- Method chaining is introduced with an excellent visual: rows shrink from 500k down to 32k at each step
- `when/then/otherwise` is explained by analogy to the familiar if/elif/else pattern
- Every "Common Mistakes" entry addresses a real beginner trap with a concrete fix

**"I got lost here" moments:**
1. **`pl.col()` is deferred evaluation** — The text says "On its own, an expression does nothing" but a true beginner may not grasp why this matters until they try to print `pl.col("town") == "BISHAN"` and get an opaque Expression object instead of True/False. A one-line "Try printing this — you will get an Expression object, not True/False" would help.
2. **`pl.lit()` in `when/then`** — The explanation is good ("Polars would interpret the string 'budget' as a column name") but a beginner encountering this error message for the first time will still be confused. The callout for Mistake 4 at the bottom catches this, but it could be flagged at first use too.
3. **Parentheses requirement for `&` combinations** — The precedence explanation is correct and important, but `pl.col("town") == ("BISHAN" & pl.col("flat_type")) == "4 ROOM"` is a meaningless parse to a beginner. Better: "Without parentheses, Python gets confused about what belongs to what. Always wrap each condition."

---

### Lesson 1.3 — Functions and Aggregation

**Accessibility: 4/5 | Clarity: 5/5**

The motivational frame ("26 towns = 26 filters = an afternoon of copy-paste" vs. "one group_by") is perfect. The SVG diagram contrasting "yesterday: one-at-a-time" with "today: all-at-once" is one of the best visuals in the entire module.

**Strengths:**
- Functions are introduced with a concrete helper (`format_sgd`) that gets immediately reused — not as an abstract concept
- The anatomy-of-a-function SVG with labeled arrows to `def`, name, parameter, type hint, and return type is excellent
- Lists and dictionaries get a brief but complete introduction — enough to read code, not so much that it derails the lesson
- The group_by SVG shows input, groups, and output side-by-side — visual learners can trace the entire operation
- The worked example builds incrementally: load, helper function, aggregation, iteration, cross-summary

**"I got lost here" moments:**
1. **Type hints introduced without full context** — "The colon after the parameter name is a type hint" — a beginner will wonder if they *need* type hints. The text says "Python does not enforce these hints at runtime" which is reassuring, but a beginner might still feel anxious about getting them "wrong." A brief "You can omit them and your code still works; we include them because they help you read the function later" would settle it.
2. **`.iter_rows(named=True)` in Step 4** — This is the first time iteration over a DataFrame appears. The explanation ("yields each row as a dictionary") is good, but a beginner who has only seen `for town in central_towns` may need a beat to understand that `row` is now a dict, not a string. The `row["town"]` syntax is explained but not contrasted with the list indexing from earlier.
3. **The Theory callout about hash tables and O(n)** — This is well-marked as Theory (blue) so beginners know to skip it, but it does appear in the middle of the core concepts. No action needed — the callout system works.
4. **Coefficient of variation (CV)** — Introduced in the mathematical foundations section. A zero-Python beginner taking this as a professional course may not need this concept yet. However, it's well-explained and clearly optional.

---

### Lesson 1.4 — Joins and Multi-Table Data

**Accessibility: 4/5 | Clarity: 4/5**

This lesson packs three big ideas: imports, if/elif/else, and joins. All three are essential and well-taught, but the density is higher than previous lessons.

**Strengths:**
- The opening question ("Is Bishan more expensive because it is closer to the MRT?") perfectly motivates why joins exist
- The Venn diagram for join types is clear, with the "use LEFT 95% of the time" rule of thumb boxed in yellow
- The key-overlap check before joining (Step 2 of the worked example) is a critical habit and it's taught early
- `if/elif/else` is introduced with a concrete function (`price_range_label`) and a flowchart SVG showing the evaluation path
- Null handling after joins gets its own section — not an afterthought

**"I got lost here" moments:**
1. **Three concepts in one lesson** — `import`, `if/elif/else`, and `join` are each worth a lesson in a pure Python course. The pacing works because each gets a focused section, but a slower learner may need to re-read this lesson. Not a flaw — just the densest lesson so far.
2. **`pl.struct([...]).map_elements(lambda x: ...)` in Step 6** — This is the most complex single code line in M1. The text explains it ("bundles two columns into a single argument for map_elements") but a beginner who has just learned what a function is two lessons ago will find `lambda x:` intimidating. The Foundations callout is missing here — a brief "lambda is a one-line anonymous function; think of it as a mini-def without a name" would help.
3. **Set operations (intersection, difference)** — Used in Step 2 for key overlap. Sets haven't been formally introduced. The code `hdb_towns & mrt_towns` and `hdb_towns - mrt_towns` will puzzle a beginner who only knows `&` as the Polars AND operator. A one-line note: "Python sets use `&` for intersection and `-` for difference — different from the Polars `&` you saw in 1.2" would prevent confusion.

---

### Lesson 1.5 — Window Functions and Trends

**Accessibility: 4/5 | Clarity: 4/5**

This is the first lesson where the concepts are genuinely harder, not just more code. Rolling means and YoY changes require understanding time ordering, partitioning, and null handling simultaneously. The lesson handles it well by leading with a strong visual motivation.

**Strengths:**
- The raw/3m/12m rolling mean SVG immediately communicates the idea before any code
- The group_by-vs-window comparison SVG is the key conceptual unlock and it's done beautifully
- "A rolling window on unsorted data is nonsense" — direct, memorable
- The responsiveness-smoothness trade-off table is practical and immediately usable
- Lazy frames get a focused introduction that's correctly scoped ("use when data is big or pipeline is long")

**"I got lost here" moments:**
1. **`.over("town")` as a window partition** — The concept is well-explained but the syntax is new and non-obvious. A beginner might wonder: "Why doesn't group_by already do this?" The text explains it ("group_by collapses; over does not") but the distinction is subtle. The SVG helps a lot; without it, this would be much harder.
2. **`shift(12)` for YoY** — The mental model of "move every value 12 positions forward" requires understanding that rows are sorted by time within each town. The text says "first 12 rows of every town will have null YoY — no 12-month history" which is correct but a beginner may not immediately see *why* 12 rows of nulls appear. A Foundations callout: "Imagine a new town with only 5 months of data — there is no 'same month last year' to compare to, so the first months are null" would ground it.
3. **Lazy frames** — Introduced well as a preview, but `pl.scan_parquet` is a new function that replaces `pl.read_csv` / `loader.load()` without a direct mapping. A beginner may wonder: "Do I still use MLFPDataLoader?" The worked example Step 5 uses `loader.path(...)` inside `scan_parquet`, which shows they coexist, but this connection could be called out.

---

### Lesson 1.6 — Data Visualisation

**Accessibility: 5/5 | Clarity: 5/5**

An outstanding lesson. The Gestalt principles and Cleveland-McGill ranking elevate this from "how to use Plotly" to "how to think about visual communication." The zero-Python beginner benefits enormously because the lesson is about *thinking*, not just API calls.

**Strengths:**
- The single-question rule ("every chart answers one question") is the most important takeaway and it's hammered home early
- The chart selection table (question -> chart type -> why) is immediately actionable
- Good chart / bad chart comparison grid is concrete
- "Title as takeaway, not header" is a career-level insight delivered in a callout
- ModelVisualizer's API is consistent (DataFrame + columns + title -> Figure) — easy to remember
- The Cleveland-McGill ranking SVG is excellent reference material

**"I got lost here" moments:**
1. **Step 7 (stacked bar) uses `import plotly.express as px` directly instead of ModelVisualizer** — The text acknowledges this ("Plotly Express handles the stacking when given long-format data") but a beginner may wonder: "When do I use ModelVisualizer vs Plotly directly?" A brief "ModelVisualizer covers the six common chart types; for specialist charts, drop to Plotly Express directly" would clarify.
2. **`.to_pandas()` in Step 7** — This is the first time pandas appears. The course is explicitly Polars-native, so this will confuse a student who was told "no pandas." The text doesn't flag this as an exception. A note: "Plotly Express expects pandas DataFrames, so we convert with .to_pandas() — this is one of the few places where the Polars-to-pandas bridge is needed" would acknowledge the contradiction.

---

### Lesson 1.7 — Automated Data Profiling

**Accessibility: 3/5 | Clarity: 4/5**

This is the hardest lesson in M1 for a beginner, not because the concepts are hard but because `async/await` is introduced for the first time and the DataExplorer API has more moving parts than anything seen before.

**Strengths:**
- The alert-type table with "meaning" and "fix" columns is immediately actionable
- "Alerts are not failures. They are decision prompts." — exactly the right framing
- AlertConfig tuning is taught with domain-specific rationale, not just API documentation
- The worked example uses deliberately messy economic data with three date formats — realistic and instructive
- try/except is introduced at the right time (just before a capstone that will need error handling)

**"I got lost here" moments:**
1. **`async def` / `await` / `asyncio.run()`** — This is a significant conceptual leap. The text says "treat the pattern as a template" which is the right advice, but a zero-Python beginner who learned `def` only three lessons ago will feel a spike of anxiety. The `shared.run_profile()` helper is mentioned in Mistake 4's callout but not shown in the main worked example. **Recommendation**: Lead with the synchronous helper (`run_profile(explorer, df)`) in the main example and put the async version in a Theory callout. This would keep the lesson accessible to beginners while preserving the async explanation for those who want it.
2. **AlertConfig parameter names** — `high_correlation_threshold`, `high_null_pct_threshold`, `skewness_threshold`, `zero_pct_threshold` — these are self-explanatory but verbose. A beginner won't know what "good" values are. The text provides domain-specific rationale ("Macro indicators are structurally correlated") which helps, but a Foundations callout with "If you don't know what threshold to set, start with the defaults and tune when you see too many false alarms" would reduce anxiety.
3. **Three date formats in one column** — The regex-based cleaning (`str.replace(r"^(\d{2})/(\d{4})$", "$2-$1-01")`) is advanced Python. A beginner will not understand regex. The text explains the chain at a high level ("walks through the three possible formats, converting each to YYYY-MM-01") which is sufficient for understanding, but a Foundations callout: "Regular expressions (regex) are pattern-matching rules — you do not need to memorise them now" would acknowledge the complexity gap.

---

### Lesson 1.8 — Data Pipelines and End-to-End Project

**Accessibility: 4/5 | Clarity: 4/5**

A satisfying capstone that ties together all seven prior lessons. The five-stage pipeline diagram is clean and the taxi trip dataset is a good choice — different from HDB, so students apply skills in a new context.

**Strengths:**
- The ETL diagram is clear and maps to the code structure
- Null handling gets its own section with three concrete strategies and a "never drop without checking" rule
- The REST API extraction section introduces httpx with a real Singapore API (OneMap)
- PreprocessingPipeline is introduced as a black box that automates what they did by hand — the right framing
- The project structure section (extract.py, transform.py, etc.) shows how to scale beyond a single file
- fit_transform vs transform distinction is called out clearly as a data leakage prevention measure

**"I got lost here" moments:**
1. **`from shared import run_profile`** — This import appears in Step 2 but wasn't introduced in the lesson. A student who has been following along would look for this in their shared module and potentially not find it. The relationship between `run_profile` and the async `DataExplorer.profile()` from L1.7 should be explicit.
2. **`pl.count().alias("trips")` in Step 5** — This uses `pl.count()` which hasn't been shown before; previous lessons used `pl.len()`. Both work, but the inconsistency will confuse a careful student. Standardise to `pl.len()`.
3. **REST API section** — httpx is introduced without installation instructions. The text imports it with `import httpx` but a beginner may not have it installed. A note: "httpx is included in the course environment" or "install with pip install httpx" would help.
4. **WorkflowBuilder preview** — The Advanced callout mentions WorkflowBuilder from Module 3, which is fine as a forward reference, but the text "node graph with automatic error handling, retry logic, and parallel execution" is jargon that a L1.8 student has no context for. Simplify to: "a more structured way to build the same pipeline."

---

## Cross-Module Bridge Quality

### M2 L01 — Probability and Bayesian Thinking

**Accessibility: 3/5 | Clarity: 4/5**

**Bridge quality: GOOD but steep**

The opening paragraph does the bridging work: "In Module 1 you learned how to look at data. Now we ask the harder question: how confident can you be in what you saw?" This directly connects the prior module to the new one.

**What works:**
- The prior -> likelihood -> posterior diagram is immediately intuitive
- The COVID ART test example is brilliant — it defeats most people's intuition and makes Bayes' theorem feel essential, not academic
- The "10,000-person ward" counting argument is the best possible Foundations callout — it makes the math concrete
- The HDB Bayesian update worked example reuses the familiar dataset and Polars patterns from M1

**"I got lost here" moments:**
1. **Mathematical notation spike** — M1 used LaTeX sparingly (mean formula, variance formula). M2 L01 introduces $P(A)$, $P(A \mid B)$, $P(A, B)$, conditional probability, joint probability, Bayes' theorem derivation, five probability distributions with parameter notation, expected value, and the variance derivation — all in one lesson. For a student who just learned what a function is 5 lessons ago, this is a wall of symbols. The Foundations callouts help, but the density of new notation is the #1 accessibility risk.
2. **Conjugate priors section** — The Normal-Normal and Beta-Binomial update formulas ($\mu_{\text{post}} = \frac{\sigma^2\,\mu_0 + n\,\sigma_0^2\,\bar{x}}{\sigma^2 + n\,\sigma_0^2}$) are algebraically dense. The intuition ("weighted average of prior and sample mean") is given but buried after the formula. **Recommendation**: Lead with the intuition, then show the formula.
3. **`precision_prior = 1 / sigma_0**2`** — The concept of "precision" (inverse variance) is not explained before use in the code. A beginner will read this and wonder "why 1 divided by variance?" The statistical motivation is there in the surrounding text but the code arrives first.
4. **Five probability distributions table** — Introduces Normal, Beta, Binomial, Poisson, Exponential all at once with mathematical notation. A Foundations callout: "You do not need to memorise all five today — Normal is the workhorse; the rest arrive when you need them" would reduce overwhelm.

### M3 L01 — Feature Engineering, ML Pipeline, and Feature Selection

**Accessibility: 2/5 | Clarity: 4/5**

**Bridge quality: ADEQUATE — but the difficulty cliff is real**

The bridge sentence ("Module 2 gave you statistics and a single linear model") correctly summarises the prior module. The seven-stage pipeline SVG is an excellent high-level map. However, this lesson makes assumptions about the student's comfort level that are not supported by M1-M2.

**What works:**
- "Data beats models, and features beat data" — memorable framing
- Point-in-time correctness is taught visually with a timeline diagram showing allowed vs leaky features
- The ICU mortality dataset is compelling and realistic
- The three feature-selection families are clearly compared (filter/wrapper/embedded) with a visual

**"I got lost here" moments:**
1. **sklearn appears without introduction** — `from sklearn.feature_selection import mutual_info_classif, RFE` — sklearn has not been used in M1 or M2. The course uses Kailash engines exclusively up to this point. A student will wonder: "What is sklearn? Why are we suddenly importing from it? Does this replace Kailash?" The transition needs an explicit callout: "scikit-learn (sklearn) is the most widely-used ML library in Python. Kailash's engines wrap many sklearn algorithms; here we use sklearn directly for feature selection."
2. **`.to_numpy()` appears without introduction** — `X.to_numpy()` converts a Polars DataFrame to a numpy array. Numpy has not been introduced. The student learned Polars; now they need to know that sklearn cannot accept Polars directly. This impedance mismatch should be flagged.
3. **ML objective formula** — $\min_{f} \mathbb{E}_{(x, y) \sim \mathcal{D}} [L(y, f(x))]$ — This uses $\mathbb{E}$ (expectation over a distribution), $\mathcal{D}$ (unknown distribution), $L$ (loss function), and $f$ (model as a function). A student who found Bayes' theorem challenging in M2 will find this impenetrable. The Theory callout helps but the formula lands in the Core Concepts section, not in a Theory block.
4. **Mutual information formula** — $I(X; Y) = \sum_{x}\sum_{y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$ — requires understanding joint probability, marginal probability, and logarithms of ratios. The intuition ("how much knowing X reduces uncertainty about Y") is great and should come *before* the formula.
5. **Lambda and sklearn idioms** — The code in Step 6 uses `sorted(zip(X.columns, mi), key=lambda t: -t[1])[:10]` which chains `sorted`, `zip`, `lambda`, negative indexing, and list slicing in a single line. This is intermediate Python. A beginner will need line-by-line decomposition.

---

## Systemic Findings

### Pattern 1: The callout system is the course's greatest pedagogical asset
Every lesson consistently uses four callout types:
- **Foundations** (green): Beginner-friendly intuition. Always safe to read.
- **Theory** (blue): Deeper understanding. Can skip without losing the plot.
- **Advanced** (purple): Expert-level. Read later.
- **Warning** (yellow): Common mistakes. Always read.

This three-track reading system means a zero-Python beginner can read *only* the Foundations callouts and still follow the lesson. **This is excellent design.**

### Pattern 2: Missing Foundations callouts at critical moments
Five specific places where a Foundations callout would prevent a beginner from getting lost:
1. L1.4: `lambda` function (first appearance)
2. L1.4: Python set operations `&` and `-` (different from Polars `&`)
3. L1.7: `async/await` (should lead with sync helper, put async in Theory)
4. M2 L01: "precision" concept before it's used in code
5. M3 L01: sklearn and numpy introduction

### Pattern 3: Code line explanations are excellent in M1, thin out in M2-M3
In L1.1-L1.3, nearly every code line has a prose explanation. By L1.7 and into M2-M3, multi-line blocks appear with only a paragraph-level gloss. The beginner is assumed to have more fluency than 7 lessons can provide.

### Pattern 4: The `.to_pandas()` exception is unaddressed
The course mandates Polars-only, but L1.6 Step 7 silently uses `.to_pandas()` for Plotly Express. This should be acknowledged as an intentional bridge, not left as an unexplained contradiction.

### Pattern 5: `pl.count()` vs `pl.len()` inconsistency
L1.8 uses `pl.count()` while all prior lessons use `pl.len()`. Both work but the inconsistency will catch a careful student.

---

## Recommendations (Priority Order)

| Priority | Location | Issue | Fix |
|----------|----------|-------|-----|
| High | L1.7 | async/await is a beginner barrier | Lead with `run_profile()` sync helper; put async in Theory callout |
| High | M3 L01 | sklearn/numpy introduced without context | Add Foundations callout explaining sklearn and the Polars-to-numpy bridge |
| Medium | L1.4 | `lambda` introduced without Foundations callout | Add 1-line: "lambda is a one-line mini-function" |
| Medium | L1.4 | Python set `&` vs Polars `&` | Add 1-line disambiguation |
| Medium | L1.6 | `.to_pandas()` contradicts Polars-only mandate | Add note explaining the Plotly Express bridge |
| Medium | L1.8 | `pl.count()` vs `pl.len()` | Standardise to `pl.len()` |
| Medium | M2 L01 | Conjugate prior formulas before intuition | Restructure: intuition first, formula second |
| Low | L1.2 | `pl.col()` deferred evaluation surprise | Add 1-line: "Try printing this expression — you'll get an Expression object, not True/False" |
| Low | L1.5 | `loader.path()` inside `scan_parquet` | Note that MLFPDataLoader still works with lazy frames |
| Low | L1.8 | httpx installation not mentioned | Add install note or mention course environment |

---

## Summary Scores

| Lesson | Accessibility | Clarity | "Lost" moments | Notes |
|--------|:---:|:---:|:---:|-------|
| L1.1 | 5 | 5 | 0 | Gold standard first lesson |
| L1.2 | 4 | 5 | 3 | Dense but well-explained; minor deferred-eval confusion |
| L1.3 | 4 | 5 | 2 | Type hints and iter_rows need brief context |
| L1.4 | 4 | 4 | 3 | Three big concepts; lambda and set ops need callouts |
| L1.5 | 4 | 4 | 3 | Window concept is hard; visuals carry the lesson |
| L1.6 | 5 | 5 | 2 | Outstanding; .to_pandas() is only concern |
| L1.7 | 3 | 4 | 3 | async/await is the barrier; regex is secondary |
| L1.8 | 4 | 4 | 4 | Good capstone; minor inconsistencies |
| M2 L01 | 3 | 4 | 4 | Steep notation cliff; COVID example is brilliant |
| M3 L01 | 2 | 4 | 5 | sklearn/numpy cliff; code density high |

**M1 Average: Accessibility 4.1, Clarity 4.5**
**Bridge Average: Accessibility 2.5, Clarity 4.0**
