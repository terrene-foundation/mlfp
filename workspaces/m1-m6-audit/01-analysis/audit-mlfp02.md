# MLFP Module 2 (mlfp02) — Correctness & Completeness Audit

Repo: /Users/esperie/repos/lyceum/courses/mlfp (read-only audit; kailash-ml 2.2.2 in `.venv`). Colab notebooks and the items listed as ALREADY KNOWN were excluded.

**Totals**: 20 BLOCKING · 35 MAJOR · 40 MINOR

Headline themes:
1. **Engine code shown to students does not run.** Every Kailash code block in deck.html, textbook.md and lessons/NN pages uses a fabricated or obsolete API (`ExperimentTracker()`, `create_experiment`, `TrainingPipeline().fit/summary`, `FeatureEngineer.add_*`, `FeatureStore().register*`, `ModelVisualizer.residual_plot/coefficient_plot`). The ex_8 FeatureStore calls are also invalid, so the module's feature-store outcome never actually executes. The deck/textbook also credit ExperimentTracker with analyses it does not perform (posterior, bootstrap, SRM, CUPED, DiD).
2. **Wrong core numbers and formulas**: the t cutoffs (1.6/1.8/1.97), the CUPED ρ=0.7 "51% reduction" (in deck, textbook, speaker notes and lesson 07), the textbook OLS worked example, the lesson 2.4 sample size (halved), the Bayesian expected-loss signs, a parallel-trends test that can never reject, peeking "theory" that assumes independent looks, a "Tukey HSD" that is really Bonferroni, CUPED using a post-treatment covariate, and AIC/BIC guidance that is backwards.
3. **Stale or divergent derivative material**: speaker-notes.md describes a superseded 83-slide module, and the lesson pages use different cases, datasets and APIs from the master deck.
4. **Exercise/assessment integrity**: four local checkpoints cannot pass with correct code; SRM failures are ignored (even with "SHIP"); assessment Task 4 is not point-in-time; Task 2 rewards analysing an SRM-contaminated cohort; and the assessment dictates all code while omitting MLE, ANOVA, power analysis and DiD, and never uses Kailash.

## Deck / textbook / speaker notes (parent-audited)

### [BLOCKING] Every Kailash Bridge code block in the master deck calls APIs that do not exist in the installed kailash-ml

**File**: modules/mlfp02/deck.html:465-475, 735-743, 1010-1023, 1226-1239, 1543-1559, 1853-1866, 2094-2113, 2213-2222, 2257-2269, 2312-2344 (and the prose claim at 2825)

**Evidence** (`.venv/bin/python`, kailash-ml 2.2.2):

```
ExperimentTracker() -> RuntimeError: ExperimentTracker cannot be constructed synchronously; use `await ExperimentTracker.create(store_url=...)`
ExperimentTracker.create_experiment / log_metric / analyze / did_analysis -> hasattr False
TrainingPipeline.__init__(self, feature_store, registry)   # both required
TrainingPipeline.fit / get_coefficients / summary -> hasattr False   (only calibrate/evaluate/retrain/train)
FeatureEngineer.add_polynomial / add_interaction / add_temporal -> hasattr False   (only generate/select)
FeatureStore.__init__(self, dataflow, *, default_tenant_id=None)  # dataflow required
FeatureStore.register_feature / store_features -> hasattr False; get_features(schema, timestamp=None, *, tenant_id, entity_ids) (no as_of kwarg; async)
ModelVisualizer.residual_plot / coefficient_plot -> hasattr False   (has residuals(y_true, y_pred), confusion_matrix(y_true, y_pred), roc_curve(y_true, y_scores))
DataExplorer.profile -> coroutine (must be awaited)
```

Deck 1859-1860 also uses pandas idioms on the result (`coeffs["odds_ratio"] = coeffs["estimate"].apply(lambda b: math.exp(b))`, `math` never imported), and 2319 hard-codes `pl.read_csv("data/wine_quality.csv")` (no such file exists anywhere under `data/`; course rule requires `MLFPDataLoader`).

**Problem**: All ten "Kailash Bridge" code slides — the only engine code students see in the deck — would raise on the first line (`ExperimentTracker()` raises `RuntimeError`; `TrainingPipeline()` raises `TypeError` for missing `feature_store`/`registry`; `FeatureStore()` raises for missing `dataflow`). The accompanying prose also attributes capabilities the engines do not have: ExperimentTracker "computes posterior probabilities" (462), "runs bootstrap, permutation, and power analysis in one call" (1007), "built-in SRM detection" (1223, 1238-1239), "applies CUPED adjustment and runs DiD analysis automatically" (2091), the engine map at 2363-2369, and ModelVisualizer "residual_plot ... creates residuals vs fitted, Q-Q, and scale-location plots in one call" (2825). ExperimentTracker is a run/metric logger; it performs none of these analyses.

**Fix**: Rewrite each bridge slide against the real API used by the module's own solutions (e.g. solutions/ex_7/02_bayesian_ab.py:246-260): `tracker = await ExperimentTracker.create(store_url=...)`; `async with tracker.track(experiment=..., run_name=...) as run: await run.log_metrics({...})`; statistics computed in student code (numpy/scipy/polars) and only *logged* by the tracker; `ModelVisualizer().residuals(y_true, y_pred)` / `.roc_curve(y_true, y_scores)`; `FeatureEngineer(...).generate(df, schema)` + `.select(...)`; `FeatureStore(dataflow)` + `await store.get_features(schema, timestamp=...)`. Remove the false capability claims (or say "you compute X; the tracker records it"). Replace the wine CSV path with an `MLFPDataLoader` call to a dataset that exists.

### [BLOCKING] Textbook code samples use non-existent / wrong-signature engine APIs throughout

**File**: modules/mlfp02/textbook.md:567-579, 678, 1260-1268, 1720, 2124-2155, 2606-2629, 3061-3075, 3460-3500, 3608-3617, 3630-3639, 3663-3674, 3708-3746

**Evidence**:

```
ExperimentTracker() -> RuntimeError (must use await ExperimentTracker.create(...))
ExperimentTracker.start_run is a coroutine (iscoroutinefunction True); signature start_run(self, experiment, *, tenant_id=None, parent_run_id=None, **params) — no `name=` param, and the result is only an async context manager (__aenter__), so `with tracker.start_run(name=...) as run:` fails
ExperimentRun.log_param / log_metric are coroutines -> un-awaited calls never execute
ModelVisualizer.line -> hasattr False;   ModelVisualizer.coefficient_plot -> hasattr False
TrainingPipeline(task=..., estimator=...) -> TypeError (__init__(self, feature_store, registry)); .fit/.summary/.r_squared_ do not exist
DataExplorer.explore -> hasattr False (method is async profile)
FeatureEngineer.add_temporal / add_distance_to_point -> hasattr False
FeatureSchema signature: (name, features: list[FeatureField], entity_id_column, timestamp_column=None, version=1)  — textbook passes a single dict
FeatureStore() -> requires dataflow; register_schema / ingest -> hasattr False
```

Also: textbook 2621, 3496, 3670, 3734 call `.to_pandas()` (course is Polars-only per CLAUDE.md directive 2), and reference columns absent from `hdb_resale.parquet` (`remaining_lease_years`, `storey_mid`, `year`, `lat`, `lon`; actual schema: month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, remaining_lease (string), resale_price).

**Problem**: The textbook claims (93-94) "You can run the code locally; imports and dataset names match `shared.MLFPDataLoader`." None of the engine snippets run; every ExperimentTracker block raises on construction, and the `with`/un-awaited pattern would silently do nothing even if construction succeeded.

**Fix**: Rewrite every engine block to the async pattern used in solutions (`await ExperimentTracker.create(store_url=...)`, `async with tracker.track(experiment=..., run_name=...) as run: await run.log_params({...}); await run.log_metrics({...})`, wrapped in `asyncio.run(main())`); replace `TrainingPipeline(task=..., estimator="ols").fit/summary` with the real OLS done in numpy (as ex_5 does) or with the real `TrainingPipeline(feature_store, registry).train(data, schema, model_spec, eval_spec, experiment_name)`; remove `.to_pandas()`; derive the missing columns explicitly (parse `remaining_lease`, `storey_range`) or use columns that exist.

### [BLOCKING] t-statistic critical values on the Lesson 2.5 slide are wrong

**File**: modules/mlfp02/deck.html:1372-1374

**Evidence**: `<tr><td>90%</td><td>&gt; 1.6</td>…<tr><td>95%</td><td>&gt; 1.8</td>…<tr><td>99%</td><td>&gt; 1.97</td>`

**Problem**: Large-sample two-sided cutoffs are 1.645 (90%), 1.960 (95%), 2.576 (99%) — as stated in specs/module-2.md (Lesson 2.5), in the same deck at line 643, and in textbook.md:2500-2502. A student using this table would call a coefficient with t = 1.9 "significant at 95%" and t = 2.0 "significant at 99%".

**Fix**: Replace with `> 1.645 *`, `> 1.960 **`, `> 2.576 ***`.

### [BLOCKING] CUPED variance-reduction figure for ρ = 0.7 is wrong (51% stated; correct is 49%)

**File**: modules/mlfp02/deck.html:1948; modules/mlfp02/textbook.md:3279-3281; modules/mlfp02/speaker-notes.md:469; modules/mlfp02/lessons/07/slides.html:84, lessons/07/textbook.html:88, lessons/07/notes.html:99

**Evidence**: deck: `If $\rho = 0.7$, variance reduction = $1 - 0.49 = 51\%$`; textbook: "If the correlation … is `ρ = 0.7`, variance drops by `1 − 0.49 = 0.51`: a **51% reduction**." speaker-notes: "If rho = 0.7, variance drops by 51%." lessons/07 table row: "0.7 | 51% | 2.04x".

**Problem**: With Var(Y_adj) = Var(Y)(1 − ρ²), the *remaining* variance is 1 − ρ² = 51%; the *reduction* is ρ² = 49%. The textbook's own table two lines later (3295) correctly says 49% for ρ = 0.7, and the deck's next bullet (1949) correctly says "75% remaining (25% reduction)" for ρ = 0.5, so the material contradicts itself on the module's headline 2.7 quantity ("CUPED reduces variance (quantified)" is a spec assessment criterion).

**Fix**: deck 1948 → "If ρ = 0.7, remaining variance = 1 − 0.49 = 51% (49% reduction)"; textbook 3280-3281 → "variance drops to 0.51 of the original: a **49% reduction**"; same correction in speaker-notes.md:469; lessons/07 table row → "0.7 | 49% | 1.96x" (1/0.51 = 1.96).

### [BLOCKING] Textbook OLS worked example reports coefficients and R² that are not the OLS solution

**File**: modules/mlfp02/textbook.md:2553-2599

**Evidence**: textbook states `β̂ ≈ [318, 3.0, −2.5]`, `SS_res ≈ 4173`, `R² ≈ 0.708`. Computing `np.linalg.lstsq` on the textbook's own X and y:

```
beta = [139.05, 5.263, -3.474]   SS_res = 16.84   SS_tot = 14280   R2 = 0.9988
with the textbook's beta [318, 3.0, -2.5]: residuals [-53,-25.5,-10.5,2,24.5], SS_res = 4173.75
```

**Problem**: The text tells students to "do this in NumPy"; they will get a completely different answer (and the stated residuals sum to −62.5, which is impossible for an OLS fit with an intercept). The interpretation bullets ("+SGD 3,000 per sqm", "−SGD 2,500 per year", "~71% of variance explained") are therefore wrong.

**Fix**: Replace with β̂ ≈ [139.1, 5.26, −3.47] (intercept SGD 139K, +SGD 5,263/sqm, −SGD 3,474/year of age), residuals ≈ y − Xβ̂ with SS_res ≈ 16.8, R² ≈ 0.999 — or change the toy data so the stated numbers are the actual OLS solution.

### [MAJOR] Textbook defines p/(1−p) as the "odds ratio"

**File**: modules/mlfp02/textbook.md:2813-2814

**Evidence**: "`p / (1 − p)` is the **odds ratio**: if `p = 0.8`, odds = 4 (i.e. 4 to 1 in favour)."

**Problem**: p/(1−p) is the *odds*; the odds ratio is the ratio of two odds (e^β per unit change) — exactly as the textbook's own glossary says (4105-4110) and as the spec's 2.6 learning objective "Compute and interpret odds ratios" requires students to distinguish.

**Fix**: "`p / (1 − p)` is the **odds**: if p = 0.8, odds = 4 (4 to 1 in favour). The ratio of two odds is the **odds ratio**."

### [MAJOR] Draft/authoring text left in the Lesson 2.3 worked example

**File**: modules/mlfp02/textbook.md:1738-1751

**Evidence**: "10,000 visitors split evenly … Variant A has 451 conversions (4.51%)…" then "p_a = 451 / 5000 = 0.0902         NO wait, we said 10k split evenly" and "Let me redo the setup: 5000 per group. … Let's make it realistic: 50_000 per group…"

**Problem**: Internal scratch reasoning shipped as student-facing content; the stated setup (10,000 visitors, 4.51%) is inconsistent with the counts and is then abandoned mid-example.

**Fix**: Replace lines 1738-1751 with a single clean setup: "100,000 visitors split evenly (50,000 per arm). A: 2,255 conversions (4.51%); B: 2,510 (5.02%)."

### [MAJOR] speaker-notes.md describes a different (superseded) module, not the current deck

**File**: modules/mlfp02/speaker-notes.md (whole file; e.g. lines 61, 99, 412, 450, 498, 668, 775, 908, 964)

**Evidence**: Lesson numbering in the notes: "Lessons 2.1-2.6 are statistics and experiments. Lessons 2.7-2.8 are feature engineering and the feature store" (61); headings "Slide 34: 2.4 Bootstrap", "Slide 37: 2.5 CUPED", "Slide 41: 2.6 Causal Inference", "Slide 55: 2.7 Feature Selection — Mutual Information", "Slide 63: 2.8 Feature Store"; opening case "The Feature That Killed a Clinical Trial" (66-77); "HDB resale: 15M+ rows" (908); "M2 (new): FeatureSchema, FeatureEngineer, FeatureStore, ExperimentTracker" (964). deck.html has 99 slides on a different plan (2.3 Bootstrapping & Hypothesis Testing, 2.4 A/B Testing, 2.5 Linear Regression, 2.6 Logistic & ANOVA, 2.7 CUPED & DiD, 2.8 Capstone); none of the notes' Pearl DAGs / Boruta / James-Stein / Double ML slides exist in the deck. `data/mlfp01/hdb_resale.parquet` has 50,150 rows.

**Problem**: An instructor following speaker-notes.md would teach a lesson order and content that contradicts the deck, the spec (specs/module-2.md), the textbook and the exercises — and would skip linear regression, logistic regression, ANOVA, SRM and A/B design entirely (no slide notes exist for them).

**Fix**: Regenerate speaker-notes.md from the deck's `<aside class="notes">` blocks (which are current) or rewrite it against the spec's 2.1-2.8 structure; delete the obsolete 83-slide plan.

### [MAJOR] Capstone option A (wine quality) has no dataset anywhere in the course

**File**: modules/mlfp02/deck.html:2171, 2319; specs/module-2.md (Lesson 2.8 project options)

**Evidence**: `ls data/` → mlfp01 (economic_indicators.csv, hdb_resale.parquet, …), mlfp02 (experiment_data.parquet, icu_*.parquet, sg_credit_scoring.parquet), no wine file; `grep -rli wine` over solutions/, assessment/, shared/mlfp02 → no hits. Deck code loads `pl.read_csv("data/wine_quality.csv")`.

**Problem**: The deck advertises Option A as "the most structured" capstone ("Start there if you feel uncertain", 2185), but there is no data, loader entry or exercise support for it.

**Fix**: Either add the wine-quality dataset to the data loader (and an ex_8 path for it) or drop Option A from the deck/lesson-8 pages and replace with an option backed by an existing dataset.

### [MINOR] "0.0005 = 5-sigma (particle physics)" is wrong

**File**: modules/mlfp02/deck.html:871

**Evidence**: `<td>0.0005</td><td>5-sigma</td><td>Particle physics</td>`

**Problem**: 5σ corresponds to p ≈ 2.9 × 10⁻⁷ (one-sided), as the textbook itself states (textbook.md:1530, "α = 5 × 10⁻⁷ (physics 'five sigma')"). p = 0.0005 is ≈ 3.5σ.

**Fix**: Either keep 0.0005 (spec value) and relabel it "Very strict / hard sciences", or change the row to "≈ 3 × 10⁻⁷ | 5-sigma | Particle physics".

### [MINOR] Conjugate-prior slide puts a Beta prior on the mean HDB price

**File**: modules/mlfp02/deck.html:406

**Evidence**: "Prior belief: mean HDB resale price is $\text{Beta}$ distributed around $500K (weak prior)"

**Problem**: Beta has support [0, 1]; it cannot be a prior on a price in SGD. The Normal-Normal conjugate (listed one bullet above, and used by the textbook 595 and ex_1) is the correct pairing for an unknown mean.

**Fix**: "Prior belief: mean HDB resale price ~ Normal(μ₀ = $500K, σ₀ = $25K)".

### [MINOR] Simpson's-paradox slide asserts a negative area–price relationship in HDB data that the course data does not show

**File**: modules/mlfp02/deck.html:1519, notes 1528

**Evidence**: "A simple regression shows area has a negative coefficient for price (larger flats cost less)…" ; on the course dataset `corr(floor_area_sqm, resale_price) = 0.469` (positive).

**Problem**: Presented as an HDB fact; students checking it in ex_5 will find the opposite.

**Fix**: Label it explicitly as a hypothetical illustration, or use a real reversal from the data.

### [MINOR] Power-analysis worked number is mis-rounded

**File**: modules/mlfp02/deck.html:2762-2763

**Evidence**: `(1.96+0.84)^2 × 2 × 16 / 0.25 = 7.84 × 32 / 0.25 = 1,003`

**Problem**: 7.84 × 32 / 0.25 = 1,003.52; sample sizes are always rounded *up* → 1,004 per arm (2,008 total).

**Fix**: "= 1,003.5 → 1,004 per group, ~2,008 total".

### [MINOR] Textbook "95% sure" phrasing for a frequentist coefficient CI contradicts the module's own CI rule

**File**: modules/mlfp02/textbook.md:2308-2309

**Evidence**: "…and we are 95% sure that effect is between `a` and `b`." vs textbook.md:642-644: "You can literally say 'I'm 95% sure the true mean is in this range' — which … you *cannot* say about a frequentist CI."

**Fix**: "…and the 95% confidence interval for that effect is [a, b]."

### [MINOR] Revenue/ROI arithmetic in the Lesson 2.3 worked example is doubled

**File**: modules/mlfp02/textbook.md:1799-1802

**Evidence**: "a 0.51 percentage-point lift at $10 per conversion and 100K weekly visitors is roughly 510 × 2 × $10 = $10,200 per week … ROI is about 5 weeks."

**Problem**: 100,000 × 0.0051 = 510 extra conversions/week × $10 = $5,100/week; the "× 2" has no basis. Payback on $50K is ~10 weeks, not 5.

**Fix**: "510 × $10 = $5,100 per week … payback about 10 weeks."

### [MINOR] Bayesian worked example narrates results the shown code cannot produce

**File**: modules/mlfp02/textbook.md:598-599, 686-691

**Evidence**: Narrative: "x̄ = 540_000 … s = 80_000", "a narrow bell centred at 540K … the data has overwhelmed the prior". The code below loads `hdb_resale.parquet` 4-ROOM 2024: n = 2064, mean = 871,673, sd = 452,366 (computed with `.venv/bin/python`); with the stated prior (500K, σ₀ = 25K) the posterior mean is ≈ 821K, i.e. the prior still pulls ~50K.

**Fix**: Either present the 540K/80K numbers as a stand-alone hypothetical (and drop the "Code:" claim that it reproduces them) or update the narrative to the dataset's real summary statistics.

### [MINOR] Employee-attrition table mixes standardised coefficients with raw-unit interpretations

**File**: modules/mlfp02/textbook.md:3087-3096

**Evidence**: "Logistic regression on standardised predictors" then "years_since_last_promotion +0.35 … Each additional year → 42% more churn odds"; "monthly_hours +0.20 … 10 extra hours/mo → 22% more".

**Problem**: With standardised predictors e^β is the odds multiplier per one standard deviation, not per year / per 10 hours.

**Fix**: Say "per 1 SD increase" or state the predictors are unstandardised.

### [MINOR] DiD cooling-measure example uses a treated group that cannot exist in the HDB resale market

**File**: modules/mlfp02/textbook.md:3425-3451, 3646-3648, 3687-3691

**Evidence**: "Treated: non-owner-occupier (investment) purchases. Control: first-time buyers" for HDB resale prices; capstone narrative reports "reduced investment-segment price growth by SGD 18K … (95% CI [SGD 12K, SGD 24K], p < 0.001)".

**Problem**: HDB resale flats must be owner-occupied (a household cannot hold two HDB flats), so there is no "investment" HDB segment; and the course dataset has no buyer-type column, so the capstone's reported result is not computable from the provided data, yet is presented as a finding with a CI and p-value. The two sections also contradict each other (−15K hypothetical in 2.7 vs −18K "result" in 2.8).

**Fix**: Reframe the example around a contrast that exists in the data (e.g. private vs HDB index, or flat types/towns differentially exposed to a measure) and mark any numbers as hypothetical.

### [MINOR] Broken internal cross-references in the textbook

**File**: modules/mlfp02/textbook.md:556-557, 2046-2047, 3758-3760

**Evidence**: "inverse propensity weights, which we'll meet in Lesson 2.7" (Lesson 2.7 has no IPW content); "Leakage … We'll cover this formally in Lesson 2.7" (leakage/point-in-time is Lesson 2.8 per spec and deck 2235-2269; textbook 2.7 has none); "Module 5 adds LLM-based synthesis … drafted by an agent" (LLMs/agents are Module 6 per specs/_index.md; M5 is deep learning/vision).

**Fix**: Point IPW to "reference material" (propensity methods are deferred per spec 2.7 design note), leakage to Lesson 2.8, and LLM synthesis to Module 6.

### [MINOR] Module 3 preview lesson list does not match the M3 spec

**File**: modules/mlfp02/deck.html:3405-3412

**Evidence**: Deck: "3.3 Decision Trees and Random Forests … 3.6 Hyperparameter Tuning, 3.7 Model Registry and Deployment, 3.8 Capstone: Full ML Pipeline". specs/module-3.md: 3.3 The Complete Supervised Model Zoo; 3.6 Interpretability and Fairness; 3.7 Workflow Orchestration, Model Registry, and Hyperparameter Search; 3.8 Production Pipeline — DataFlow, Drift, and Deployment.

**Fix**: Copy the eight lesson titles from specs/module-3.md.
## Exercises ex_1 – ex_4 (solutions, local scaffolds, shared/mlfp02/ex_1–4.py)

### [BLOCKING] ex_1/04 — following the local hint makes Checkpoint 1 unpassable

**File**: modules/mlfp02/local/ex_1/04_intervals.py:117 (hint), :140 (assert); cf. solutions/ex_1/04_intervals.py:101-106, 130-132

**Evidence**: Solution builds the coverage simulation's credible interval with a flat prior (`sim_prior_mu = true_mu`, `sim_prior_sigma = 10 * true_sigma`; comment: "An informative prior biased away from true_mu would deliberately undercover"). Local drops those variables and hints `normal_normal_posterior(sample, mu_0, sigma_0, sample.std(ddof=0))` with mu_0 = $500K, sigma_0 = $100K. Re-running that loop (1,000 sims, seed 42) gives Bayesian coverage **0.694**; the assert requires `0.90 < bayes_coverage < 1.0`.

**Problem**: A student following the hint exactly cannot pass the checkpoint.

**Fix**: Restore `sim_prior_mu`/`sim_prior_sigma` in local and change the hint to `normal_normal_posterior(sample, sim_prior_mu, sim_prior_sigma, sample.std(ddof=0))`.

### [BLOCKING] ex_2/04 — local generates different data from the solution; Checkpoint 5 fails and the conclusion flips

**File**: modules/mlfp02/local/ex_2/04_mle_failures.py:151 (assert :240) vs solutions/ex_2/04_mle_failures.py:178

**Evidence**: local `shock_data = rng_t.standard_t(df=3, size=100) * 2.0 + 2.5`; solution `standard_t(df=4, size=500) * 1.2 + 2.5` (comment: df=4 chosen to avoid outliers that "inflate the sample std and contaminate the Normal fit"). Recomputed (seed 77): local data P(x < −5) = 0.1206 under Normal vs 0.0336 under t; solution data 1.3e-05 vs 0.00235.

**Problem**: `assert prob_t > prob_normal` fails for a correct student answer, and the printed "Normal UNDERESTIMATES crisis probability" becomes false (ratio ≈ 0.28×).

**Fix**: Make local line 151 identical to the solution.

### [BLOCKING] ex_2/05 — AIC/BIC guidance is reversed

**File**: modules/mlfp02/local/ex_2/05_model_selection.py:58-59; solutions/ex_2/05_model_selection.py:132

**Evidence**: "When they disagree, prefer BIC for prediction, AIC for explanation." The solution contradicts itself at 66-67 ("AIC tends to select more complex models (better for prediction). BIC … (better for identification)"); local contains only the wrong version.

**Problem**: AIC (asymptotically efficient) is the predictive criterion; BIC (consistent) is for identifying the true/parsimonious model.

**Fix**: "Prefer AIC for prediction, BIC for identifying the true/parsimonious model" in both files.

### [BLOCKING] ex_1/02 — "most positives are false positives" is false for the scenario the file computes

**File**: modules/mlfp02/solutions/ex_1/02_bayes_theorem.py:56-59 and :287 (REFLECTION); local same text at 56-59, :281

**Evidence**: "A COVID test with 99.5% specificity … when prevalence is only 2%, most positive results are false positives". With the file's own numbers (sens 0.85, spec 0.995, prevalence 0.02): P(infected | +) = 0.017/0.0219 = **0.776** (the file's own comment at line 101 says "~77%").

**Problem**: At 2% prevalence ~78% of positives are true; "most positives false" only holds below ≈0.58% prevalence. The REFLECTION lists the wrong takeaway as mastered.

**Fix**: "At 2% prevalence about 1 in 4 positives is false; below ~0.5% prevalence most positives are false (e.g. 85% at 0.1%)" — in theory text and REFLECTION.

### [MAJOR] ex_1/02 — false discovery rate labelled "False Positive Rate"

**File**: solutions/ex_1/02_bayes_theorem.py:86, 157, 210 (local 84, 153, 206)

**Evidence**: `p_false_positive = 1 - p_infected_given_positive`, captioned "false positive rate among positives", plotted as "False Positive Rate vs Prevalence".

**Problem**: 1 − PPV is the false discovery rate; FPR = FP/(FP+TN) = 1 − specificity = 0.5%, which does not vary with prevalence. Conflicts with ex_6 / M3 confusion-matrix definitions.

**Fix**: Rename to "False discovery rate, P(not infected | +)" in variable comment, subplot title and axis label.

### [MAJOR] ex_1/01 Apply — standard error of the mean used as unit-price precision, and wrong population

**File**: solutions/ex_1/01_probability_mle.py:268-290 (local ~272-295)

**Evidence**: Scenario "4-room HDB launch in Queenstown"; code compares the national 4-room SE ($4,178) to a $20K margin and prints "List at $869,023 … range $849,023 – $889,023". Measured: national 4-room σ = $419,793 (n = 10,094); Queenstown 4-room mean = **$1,001,500**.

**Problem**: Confuses a CI for the mean with a prediction interval for one unit (the exact misconception M2 targets) and prices a Queenstown unit at the all-island mean (~$130K too low).

**Fix**: Filter to Queenstown; contrast SE of the mean with unit-level spread / a prediction interval; base the pricing decision on the spread.

### [MAJOR] ex_3 / ex_4 — SRM tested against 50/50 on a non-50/50 design; always fires and analysis proceeds (even "SHIP")

**File**: shared/mlfp02/ex_3.py:65-71 (pools all non-control arms into "treatment"); solutions/ex_3/01_bootstrap_power.py:62-64, 76; solutions/ex_4/02_srm_detection.py:150-152; solutions/ex_4/04_validity_adaptive.py:262-265, 344, 354

**Evidence**: Arms: control 188,732; treatment_a 165,413; treatment_b 70,855; variant_c 75,000. `srm_check(188732, 311268)` → χ² = 30030, p = 0.0 "SRM DETECTED"; ex_4 two-arm subset χ² = 1535.5, p = 0.0. ex_3/01 says "If SRM fires, do NOT trust downstream results" then runs every test; ex_4/04 report prints "SRM Check: … FAIL — results may be biased" beside "Decision: SHIP".

**Problem**: Expected ratio does not match the design; ex_3 pools three different treatments; students are taught to ignore their own validity gate (spec 2.4 criterion "SRM checked").

**Fix**: Test SRM against the designed allocation, analyse one treatment arm vs control, and make ex_4/04 return "DO NOT SHIP — investigate SRM" when SRM fails.

### [MAJOR] ex_3 — "conversion" defined as metric_value > 0 gives ~99% rates while the narrative assumes ~10% baseline

**File**: shared/mlfp02/ex_3.py:61-64; solutions/ex_3/01_bootstrap_power.py:313-315 (local :318)

**Evidence**: Measured conversion: control 0.9860, treatment 0.9916 (implied MDE ≈ 0.1pp). Comment: "With the current baseline conversion of ~10% and an MDE of ~2pp…", and un-interpolated template text "~{n_total:,} users per group".

**Fix**: Derive conversion from a meaningful binary event or rewrite the narrative to the real numbers; remove brace placeholders.

### [MAJOR] ex_3/02 — decision rule outputs "Need more data" for a highly significant result

**File**: solutions/ex_3/02_hypothesis_testing.py:343-351 (local ~349-357)

**Evidence**: z = 18.67, p = 0.0, Cohen's h = 0.0532 → code returns `"Need more data" if abs(h_conversion) > 0.05`; the 4-quadrant table directly above (330-336) maps p < 0.05 & |h| < 0.1 to "significant but negligible".

**Fix**: Implement the documented quadrants (branch on p, then |h| ≥ 0.1).

### [MAJOR] ex_3/04 — Mann-Whitney U called "parametric" and compared like-for-like with a mean-difference permutation test

**File**: solutions/ex_3/04_permutation_test.py:158, 195-203 (local 166, 207)

**Evidence**: "The parametric Mann-Whitney U test is robust to this"; table header "Parametric p" next to `mwu_p`.

**Problem**: MWU is non-parametric and tests stochastic ordering, not the mean difference the permutation test uses.

**Fix**: Call it non-parametric; compare the permutation test with Welch's t on revenue (or permute U).

### [MAJOR] ex_2/02 — "profile likelihood" CI holds σ fixed, so it is identical to the Wald CI

**File**: shared/mlfp02/ex_2.py:149-183; solutions/ex_2/02_mle_fisher.py:72-75, 134-145

**Evidence**: `loglik_values = [-neg_log_likelihood_normal([mu, np.log(sigma_hat)], x) …]` — σ never re-maximised per μ.

**Problem**: With σ fixed, 2Δℓ = n(x̄−μ)²/σ̂² — exactly Wald. The exercise claims profile CIs are "more accurate for small n", compares two identical intervals and concludes "the Normal approximation is adequate".

**Fix**: Profile out σ: σ̂²(μ) = mean((x−μ)²) at each grid point.

### [MAJOR] ex_4/03 Apply — cashback cost-benefit is wrong and flips the business decision

**File**: solutions/ex_4/03_welchs_ttest.py:259-262, 278-280 (local 255-256)

**Evidence**: `extra_cashback_cost = dbs_diff * 0.005 * 12` charges the extra 0.5% only on incremental spend; conclusion "Even the lower bound of the CI produces positive net revenue — so the decision is clear."

**Problem**: The 2% rate applies to all treatment spend: monthly incremental cashback 0.02×2350 − 0.015×2200 ≈ $14.0 vs incremental interchange 0.015×150 ≈ $2.25 → net ≈ **−$141/customer/year**.

**Fix**: Cost = full treatment-spend cashback minus control cashback.

### [MAJOR] ex_4/04 — novelty check splits by unsorted row order; report hard-codes "no novelty effect detected"

**File**: solutions/ex_4/04_validity_adaptive.py:162-175, 358 (local 160, 351)

**Evidence**: `early_treat = data.treat_values[:n_half]` on a string timestamp column that is not sorted (first rows 2024-01-03, 2024-01-13, 2024-01-02); report f-string always prints "no novelty effect detected".

**Fix**: Sort by parsed timestamp before splitting; print the verdict from `novelty_p`.

### [MAJOR] ex_4/02 Apply — the "device-type SRM" cannot be caught by the aggregate χ² test used

**File**: solutions/ex_4/02_srm_detection.py:271-318

**Evidence**: 45% iOS at 55% treatment, rest at 45% → overall 49.5%. Over 200 seeds the aggregate χ² detected SRM only **15%** of the time (median p = 0.14); text asserts "A device-type SRM appears".

**Fix**: Add a per-segment SRM check and state that the aggregate test misses compositional imbalance.

### [MAJOR] ex_4/04 Apply — adaptive-design narrative contradicted by its own output

**File**: solutions/ex_4/04_validity_adaptive.py:379-413

**Evidence**: `required_n_per_group(5.0, 1.0)` = 393/group → 786 of 2,000 available, code prints "Feasible? YES", then prints "the initial sigma overestimate would have required more participants than available"; achieved power 99.96%.

**Fix**: Choose parameters where the initial design is infeasible (e.g. σ = 8 → 1,005/group) or reword.

### [MAJOR] ex_2/02, ex_2/03 Apply narratives make false real-world claims

**File**: solutions/ex_2/03_map_estimation.py:184-194; solutions/ex_2/02_mle_fisher.py:177-197

**Evidence**: "GIC (Singapore's sovereign wealth fund) backstops the SME lending ecosystem" (GIC manages foreign reserves; no such role); scenario calls data a "default rate" but feeds GDP growth [2.1, 3.8, …]; ex_2/02 treats a CI of mean GDP growth as a bank's PD and prints "If MAS requires capital provisioning against the lower bound of the CI…" (no such requirement).

**Fix**: Rebuild as coherent proxies (e.g. planning baseline for growth) and drop fabricated regulatory/institutional claims.

### [MAJOR] Spec 2.4 — A/B exercise never uses ExperimentTracker

**File**: modules/mlfp02/solutions/ex_4/*.py, shared/mlfp02/ex_4.py

**Evidence**: specs/module-2.md 2.4 Exercise: "Design and analyse a complete A/B experiment using ExperimentTracker"; exercise-standards.md Train phase "with kailash-ml ExperimentTracker". `grep -rln "ExperimentTracker"` over solutions → only ex_7 and ex_8. Only Kailash usage in ex_1–ex_4 is `ModelVisualizer.histogram` (`viz` in ex_1/04:257 created and never used).

**Fix**: Log design params, SRM result, test statistics and decision to ExperimentTracker in ex_4.

### [MAJOR] Spec 2.1–2.3 techniques not exercised in ex_1–ex_4

**File**: modules/mlfp02/solutions/ex_1 … ex_4

**Evidence** (grep): no Poisson; Exponential only as a CLT population, never fitted to economic data (spec 2.2 exercise "Fit Normal and Exponential distributions to Singapore economic data"); no LLN demo; no parametric vs non-parametric bootstrap; no one-sample (`ttest_1samp`) or one-tailed test (spec 2.3 "Common test types"); spec 2.1 "Visualise prior, likelihood, and posterior" — ex_1/03 plots prior and posterior only.

**Fix**: ex_2 add Exponential MLE fit + AIC comparison and an LLN running-mean plot; ex_3 add one-sample/one-tailed test and a parametric bootstrap; ex_1/03 add the likelihood curve.

### [MINOR] ex_1/01 — Fisher information prints as 0.0000

**File**: solutions/ex_1/01_probability_mle.py:176

**Evidence/Problem/Fix**: I(μ) = n/σ² = 5.73e-08 printed with `:.4f`. Use `:.3e`.

### [MINOR] ex_1/01 local — TODOs on code that is already complete

**File**: local/ex_1/01_probability_mle.py:141, :284

**Evidence/Problem/Fix**: "Create price bands using pl.when()…" and "Compare mle.standard_error with margin and print the decision" are followed by the full solution code with no `____`. Blank the code or remove the TODO.

### [MINOR] ex_1/04 — stated numbers do not match the data

**File**: solutions/ex_1/04_intervals.py:403 (local :380), :106

**Evidence/Problem/Fix**: "prior contributes <0.01%" — computed prior weight 0.174%; "~1.35M SGD" — 10×σ = $4.2M. Print computed values.

### [MINOR] ex_2/01, ex_2/03 — comments contradict printed results

**File**: solutions/ex_2/01_clt_sampling.py:302 (local :258); solutions/ex_2/03_map_estimation.py:129

**Evidence/Problem/Fix**: "~40-60 quarterly observations … underestimates volatility by ~2%" — actual n = 101, ddof gap 0.50%. "With n=3, the prior has ~30% influence. With n=50+, the prior barely matters" — recomputed shrinkage 0.1% at n=3, 11.7% at n=50, 6.8% at n=101. Reference printed values.

### [MINOR] ex_3/02 — p-value definition omits "at least as extreme"

**File**: solutions/ex_3/02_hypothesis_testing.py:30, :380 (local 30, 386)

**Evidence/Problem/Fix**: "p-value = P(data | H0 true)"; spec requires "at least as extreme", which line 90 itself states. Fix both lines.

### [MINOR] ex_3/04 — permutation p-value can print exactly 0

**File**: solutions/ex_3/04_permutation_test.py:127, 168

**Evidence/Problem/Fix**: `np.mean(np.abs(perm) >= np.abs(obs))` → 0 when nothing exceeds observed (z ≈ 18.7). Use `(count + 1) / (N + 1)`.

### [MINOR] ex_3/01, ex_3/03 — business arithmetic inconsistencies

**File**: solutions/ex_3/01_bootstrap_power.py:329-331; solutions/ex_3/03_multiple_testing.py:381-399

**Evidence/Problem/Fix**: "need ~{n_total*2} users" doubles the existing n although MDE was computed for existing n; comment says "~17 false positives … S$750K" while code prints `int(50*fwer_8*0.5)` = 8. Align comments with code.

### [MINOR] ex_4/03 — wrong Cohen's d thresholds

**File**: solutions/ex_4/03_welchs_ttest.py:131-132

**Evidence/Problem/Fix**: `<0.2 small, <0.5 medium, else large`; Cohen's convention (used by ex_3 `interpret_magnitude`) is 0.2/0.5/0.8. Reuse the shared helper.

### [MINOR] ex_4/01 — PDPA claim overstated

**File**: solutions/ex_4/01_experiment_design.py:65-69

**Evidence/Problem/Fix**: "Pre-registration satisfies [PDPA purpose limitation]" — purpose limitation requires notification/consent; an internal pre-registration does not satisfy it. Reword.

## Exercises ex_5 – ex_8 (solutions, local scaffolds, shared/mlfp02/ex_5–8.py)

### [BLOCKING] ex_7/02 — Bayesian expected-loss formulas have the wrong signs

**File**: shared/mlfp02/ex_7.py:287-289 (used by solutions/ex_7/02_bayesian_ab.py decision rule)

**Evidence**: `exp_loss_treat = se*pdf(z) + lift*cdf(z)`, `exp_loss_ctrl = se*pdf(-z) - lift*cdf(-z)`, `z = -lift/se`. Monte-Carlo with posterior N(lift, se): lift = −1.111, se = 0.143 → code loss_treat = 0.0000, true 1.1107; lift = 0.967 → code loss_ctrl = 0.0000, true 0.9675.

**Problem**: Correct: E[max(0,−L)] = se·φ(lift/se) − lift·Φ(−lift/se); E[max(0,L)] = se·φ(lift/se) + lift·Φ(lift/se). The code reports $0 loss for shipping a worse treatment; the "ship if expected loss is tiny" advice relies on it.

**Fix**: `exp_loss_treat = se*pdf(z) - lift*cdf(z)`; `exp_loss_ctrl = se*pdf(z) + lift*cdf(-z)`.

### [BLOCKING] ex_7/04 — parallel-trends "bootstrap test" can never reject

**File**: shared/mlfp02/ex_7.py:526-545 (used at solutions/ex_7/04_diff_in_diff.py:140-150)

**Evidence**: Adds N(0, 1000) noise to the observed slopes and computes `mean(|s_c+e − (s_nc+e')| >= |slope_diff|)`. Re-run with Central growing an extra $30,000/period (clearly non-parallel): slope_diff = 31,045, p = 0.4952, passes = True; the parallel case gives the same p = 0.4952.

**Problem**: Null distribution centred on the observed difference → p ≈ 0.5 always; noise SD arbitrary (real SE ≈ 80,000/√200 ≈ 5,657); pre-period series simulated independently of the DiD cells (bases 530K vs 550K). Spec 2.7 criterion "DiD parallel trends tested before applying" is met only nominally.

**Fix**: Test on pre-period microdata of the same groups (group×time interaction t-test) or permute group labels (null centred at 0).

### [BLOCKING] ex_7/03 — peeking "theory" assumes independent peeks; contradicted by its own simulation

**File**: solutions/ex_7/03_sequential_testing.py:52-58, 149-155, 197; shared/mlfp02/ex_7.py:427

**Evidence**: "Each peek is an independent test… 1-(1-0.05)^20 = 64%"; bar labelled "Theory (20 peeks)"; `simulate_peeking()` returns rate_fixed_peek = 0.257 vs theoretical_inflated_rate 0.6415.

**Problem**: Peeks on accumulating data are correlated; actual inflation for 20 equally spaced looks ≈ 25% (matching the simulation). The 64% "theory" is wrong and propagates to the application.

**Fix**: Remove the independence claim and "theory" bar; cite the simulated ~25%.

### [BLOCKING] ex_7/01 — multi-covariate CUPED uses a post-treatment metric, biasing the effect ~79%

**File**: solutions/ex_7/01_cuped.py:195-203 (and local equivalent)

**Evidence**: `pre_features.append("metric_value")`; metric_value differs by arm (control 41.37, treatment_a 44.55, treatment_b 45.90); corr(revenue, metric_value) = 0.987. Lift: naive 3.81 (SE 0.143), single-covariate CUPED 3.79, multi-covariate CUPED **0.80** (SE 0.022).

**Problem**: CUPED covariates must be pre-experiment (the textbook says so, 3305-3307); this absorbs the treatment effect and presents a biased estimate as a precision gain. Checkpoint only checks SE.

**Fix**: Use only pre-period covariates (pre_metric_value + segment/platform/country dummies) and add a note on why post-treatment covariates are forbidden.

### [BLOCKING] ex_6/04 — "Tukey HSD" is actually Bonferroni-corrected pairwise t-tests

**File**: solutions/ex_6/04_calibration_anova.py:176-211, REFLECTION :373; local/ex_6/04_calibration_anova.py same block

**Evidence**: `# Post-hoc: Tukey HSD (Bonferroni-corrected pairwise t-tests)`; p-value = `2*(1-t.cdf)` × number of comparisons; column labelled q.

**Problem**: Tukey HSD uses the studentized range distribution; spec 2.6 requires Tukey's HSD. Students learn that Tukey = Bonferroni.

**Fix**: `scipy.stats.tukey_hsd(*anova_groups)` (or `stats.studentized_range.sf(q, k, df)`); keep Bonferroni as a separately labelled comparison.

### [BLOCKING] ex_8 — FeatureStore calls do not match the installed API; every FeatureStore step (and the 8.4 lineage logging) is silently skipped

**File**: shared/mlfp02/ex_8.py:89; solutions/ex_8/01_feature_schema.py:186-187; 02_point_in_time.py:107-118, 417; 03_rolling_features.py:184-185; 04_modeling_lineage.py:175-220

**Evidence**: Installed `FeatureStore.__init__(self, dataflow, *, default_tenant_id=None)`; public methods `get_features, materialize, serve_online, …` — no `register_features`, `store`, `get_training_set`. `setup_feature_store()` prints `[warn] FeatureStore backend unavailable (TypeError: FeatureStore.__init__() got an unexpected keyword argument 'table_prefix')` and returns `(None, None, None, False)`.

**Problem**: The module LO "manage features in a feature store" never executes; reflection teaches non-existent `get_training_set(start, end)`; Checkpoint 2 in 8.1/8.3 still prints "schema registered, features stored"; 8.4's tracker comes from the same failed tuple so lineage logging always takes the "[Manual lineage — ExperimentTracker unavailable]" path even though `ExperimentTracker.create(store_url=...)` works.

**Fix**: Construct a DataFlow instance → `FeatureStore(dataflow)`; use `materialize(group, df)` / `await get_features(schema, timestamp)`; create the tracker independently.

### [BLOCKING] local ex_5/01 — Checkpoint 5 cannot pass with correct code

**File**: modules/mlfp02/local/ex_5/01_ols_from_scratch.py:267

**Evidence**: `abs(fit["SST"] - fit["SSR"] - fit["SSE"]) < 1`; on the real data |SST−SSR−SSE| = 44.5 (SST ≈ 6.19e15). Solution uses `< 1e-6 * fit["SST"]`.

**Fix**: Copy the solution's relative tolerance.

### [BLOCKING] local ex_8/04 — Checkpoint 1 cannot pass with correct code

**File**: modules/mlfp02/local/ex_8/04_modeling_lineage.py:113

**Evidence**: `assert ols["r2"] > 0.3`; model on v2 features gives R² = 0.2248 (n = 47,610). Solution uses `> 0.2`.

**Fix**: `> 0.2`.

### [MAJOR] ex_7 — four-arm experiment pooled into two arms; SRM failure ignored

**File**: shared/mlfp02/ex_7.py:55-59, 85-90 (used by solutions/ex_7/01, 02, 03)

**Evidence**: control 188,732; treatment_a 165,413; variant_c 75,000; treatment_b 70,855. `split_groups` pools all non-control rows; `compute_srm` assumes 50/50 → p = 0.0 "SRM DETECTED", analysis continues.

**Problem**: Lift is a mixture of three arms; spec 2.7 criterion "SRM detection correctly identifies problematic experiments" is undermined because detection is ignored.

**Fix**: Analyse control vs one arm and test SRM against the designed allocation; stop/explain when SRM fires.

### [MAJOR] ex_8/03 — "trailing" rolling features include the current month (target leakage in the point-in-time lesson)

**File**: shared/mlfp02/ex_8.py:136-171; solutions/ex_8/03_rolling_features.py:112; solutions/ex_8/04_modeling_lineage.py:401 (local :413)

**Evidence**: `rolling_mean(window_size=6)` over monthly medians joined back on the same `transaction_date` month; for BISHAN the value first appears at 2015-06 using the Jan–Jun window (includes the row's own price); nulls cover 5 months, not 6 as stated; 8.4 report claims "Point-in-time correctness ensures no future data leaks."

**Fix**: `.shift(1).over("town")` before the rolling window; fix the "first 6 months" text; qualify the no-leak claim.

### [MAJOR] ex_8/02 — leakage demo shows ~no effect but narrative claims a large one

**File**: solutions/ex_8/02_point_in_time.py:243-246, 386-391

**Evidence**: Correct model RMSE 330,725 / R² 0.33955; leaked 330,676 / 0.33975 (gap $50, 0.015%); line 245 prints `abs(leakage_gap)` (hides sign); application hard-codes "Overestimates property values by ~5%".

**Fix**: Demonstrate a real leak (future-window or target-derived features); compute impact from output; print signed gap.

### [MAJOR] ex_5/03 — CI of the conditional mean presented as the uncertainty of one flat

**File**: solutions/ex_5/03_weighted_ls.py:285-321

**Evidence**: "This flat is worth $520K, 95% CI [$480K, $560K]… WLS CI better reflects the actual uncertainty for this property"; code uses `se = sqrt(σ² x'(X'X)⁻¹x)` → ±$8,373 (OLS), ±$8,060 (WLS) while σ̂ = $435,319.

**Problem**: Needs a prediction interval √(σ²(1 + x'(X'X)⁻¹x)) ≈ ±$853K.

**Fix**: Show both, label correctly, use the prediction interval for the valuation.

### [MAJOR] ex_5/03 — dense n×n weight matrix (~5 GB)

**File**: solutions/ex_5/03_weighted_ls.py:122-124; hint local/ex_5/03_weighted_ls.py:128 (`# Hint: np.diag(weights)`)

**Evidence**: n = 24,904 → `np.diag(weights)` = 24,904² × 8 B ≈ 4.96 GB; `X.T @ W @ X` is O(n²k).

**Fix**: `Xw = X * weights[:, None]; XtWX = Xw.T @ X; XtWy = Xw.T @ y`; update hint.

### [MAJOR] ex_6/02, ex_6/03 — threshold optimised under one cost matrix, reported "cost-optimal" under another

**File**: solutions/ex_6/02_interpretation.py:145-163, 283-312; solutions/ex_6/03_classification_metrics.py:113-121, 305-348

**Evidence**: Threshold chosen with FP=$30K/FN=$50K; applications score it with FP=$800/FN=$45,000 (6.2) or FP=$200/FN=$8,000 (6.3), still labelled "Cost-optimal threshold"; 6.3:348 "match[es] the 40:1 asymmetry".

**Fix**: Re-optimise with each application's cost matrix.

### [MAJOR] ex_7 / ex_8 scaffolds only call pre-built helpers; students never write CUPED/DiD maths

**File**: local/ex_7/*.py, local/ex_8/*.py, shared/mlfp02/ex_7.py

**Evidence**: Every `____` in ex_7 is a helper call (`cuped = ____  # Hint: single_cov_cuped(...)`, `did = ____  # Hint: diff_in_diff(cells)`); θ = Cov/Var, Y_adj, DiD estimate and mSPRT Λ live pre-written in shared/.

**Problem**: Spec 2.7 LO "Implement CUPED" is not exercised.

**Fix**: Move θ/Y_adj, DiD and expected-loss maths into the technique files as blanks.

### [MAJOR] Local scaffolds drifted from solutions (missing VISUALISE blocks and REFLECTIONs)

**File**: local/ex_5/01, 03, 04; local/ex_6/01, 02, 03

**Evidence**: local ex_5/01, 03, 04 lack the solutions' VISUALISE blocks (solution 01:276-323, 03:187-279, 04:345-433); local ex_6/01, 02, 03 have no REFLECTION (grep count 0) — exercise-standards.md requires REFLECTION in every exercise.

**Fix**: Regenerate locals from current solutions.

### [MAJOR] Spec 2.5–2.8 techniques missing from ex_5–ex_8

**File**: modules/mlfp02/solutions/ex_5 … ex_8

**Evidence** (grep): FeatureEngineer never used (8.3:114-123 hand-rolls month/quarter in polars, calls it "FeatureEngineer-equivalent", though `FeatureEngineer.generate/select` exist); no DiD placebo test; no multiclass logistic (OvR/multinomial); no log-linear ln(y) model; no k-fold CV (single 80/20 split); no lat/lon features (spec 2.5 exercise); spec's employee-attrition logistic regression replaced by an HDB above-median target; no capstone project options and no logistic model in capstone; DiD on hand-set simulated means, not cooling-measure data (spec 2.7 exercise). Also TrainingPipeline (listed among M2 engines in spec + deck) is never used in any M2 solution.

**Fix**: Add the missing pieces (or amend the spec): FeatureEngineer.generate/select in ex_8, placebo DiD and multinomial logit, ln(price) model + k-fold in ex_5, attrition data for ex_6.

### [MINOR] ex_6/04 — "Max Brier (random): 0.25"

**File**: solutions/ex_6/04_calibration_anova.py:110

**Evidence/Problem/Fix**: Maximum Brier score is 1; 0.25 is the score of always predicting 0.5. Relabel "Brier of constant 0.5 forecast: 0.25".

### [MINOR] ex_6/01 — "accuracy on clear cases" prints accuracy on all cases

**File**: solutions/ex_6/01_logistic_from_scratch.py:320. Fix the label or the computation.

### [MINOR] ex_5/04 — base-category rationale and curve comparison wrong

**File**: solutions/ex_5/04_model_enrichment.py:186, 350

**Evidence/Problem/Fix**: 3 ROOM called the "most common" type; 4 ROOM (10,094) outnumbers 3 ROOM (6,309). The "Linear" curve drops storey/lease terms while the polynomial curve fixes them at medians, so they are not comparable. Correct the rationale; hold covariates fixed in both curves.

### [MINOR] ex_5/02 — "estimates stay unbiased" when assumptions break

**File**: solutions/ex_5/02_diagnostics.py:27-28

**Evidence/Problem/Fix**: False for misspecified linearity (bias). Qualify: unbiased under heteroscedasticity/non-normality, biased under misspecification/omitted variables.

### [MINOR] ex_8/01 — stated feature ranges contradict data; impossible lease values pass the schema

**File**: solutions/ex_8/01_feature_schema.py:221-223

**Evidence/Problem/Fix**: "price_per_sqm should cluster around $4,000-8,000/sqm" — actual IQR $8,050–$9,582; remaining_lease_years reaches 107 (impossible on a 99-year lease) and passes unflagged. Use computed ranges; add a ≤ 99 bound.

### [MINOR] Solution code left inside local TODOs

**File**: local/ex_6/03_classification_metrics.py:259; local/ex_6/04_calibration_anova.py:245; local/ex_7/01_cuped.py:275; local/ex_7/04_diff_in_diff.py:164

**Fix**: Blank the leaked code.

### [MINOR] Stale references in ex_8 helpers and comments

**File**: shared/mlfp02/ex_8.py:13 (`04_modeling_with_features.py`, actual `04_modeling_lineage.py`), :71 ("kailash-ml 1.1.1", installed 2.2.2); solutions/ex_8/01:198 (ANOVA "from M3" — taught in M2 ex_6); ex_8/04:116 (dummies "foreshadowing M3" — done in 5.4). Update text.

### [MINOR] "p < 0.00e+00" printed

**File**: solutions/ex_5/01_ols_from_scratch.py:245; solutions/ex_8/04_modeling_lineage.py:138

**Fix**: Print `p < 1e-300` (or "p ≈ 0") when f_p underflows.

### [MINOR] Public agencies given invented roles/figures

**File**: solutions/ex_6/02:271 (SLA as valuation-anomaly monitor); ex_6/04:328-333 (invented HDB dispute stats); ex_7/04:76-78 ("evaluated using DiD by MAS and URA"); ex_8/04:66-70, 357-361 (FEAT as binding "MAS requirements", 10–20% capital buffer)

**Fix**: Mark as illustrative or remove; FEAT principles are guidance, not binding capital rules.

### [MINOR] Generic boilerplate REFLECTIONs in ex_6/01–03 solutions

**File**: solutions/ex_6/01, 02, 03 (end of file)

**Evidence/Fix**: "The concepts and implementation covered above" instead of the skill list exercise-standards.md requires. List concrete skills + "Next:" pointer.

### [MINOR] ex_7/01 savings arithmetic; ex_8/03 hard-coded "insight"

**File**: solutions/ex_7/01_cuped.py:305; solutions/ex_8/03_rolling_features.py:320, 365

**Evidence/Fix**: 200 experiments/quarter × fractional speedup labelled annual, comment "~S$5M/year"; "Bishan +3.2% vs Tampines +7.1%" printed as an insight but not computed. Compute from data or remove.
## Lesson pages, index.html, README.md

### [BLOCKING] Lesson-page worked examples load non-existent datasets / wrong module / missing columns

**File**: lessons/01/textbook.html:474, 478; lessons/02/slides.html:298; lessons/02/textbook.html:376; lessons/03/textbook.html:413-425; lessons/04/slides.html:205; lessons/04/textbook.html:330; lessons/05/slides.html:160-163; lessons/05/textbook.html:400-404; lessons/06/textbook.html:353-356, 424-430; lessons/07/textbook.html:127, 308, 326-341; lessons/08/slides.html:216; lessons/08/textbook.html:244 (all under modules/mlfp02/)

**Evidence**: `data/mlfp02/` contains only experiment_data, icu_*, sg_credit_scoring; `hdb_resale.parquet` and `economic_indicators.csv` are in `data/mlfp01/`; `ab_test.parquet`, `ab_test_cuped.parquet`, `wine_quality.csv` exist nowhere → `MLFPDataLoader.load` raises `FileNotFoundError("File not found: mlfp02/... (not in local data/mlfp02/)")` (shared/data_loader.py:258-261). hdb_resale has no `year` column (has `month: String`) yet 2.1/2.5 compute `pl.col("year") - ...`; icu_patients has no `outcome_30d`, `apache_score`, `emergency_admission`; 2.6's ANOVA then filters the ICU df on `flat_type`; 2.7 Step 4 uses undefined `treatment_adj`, `control_adj`, `Y_adj_all`, `Y_all` (NameError).

**Problem**: Every lesson's worked example crashes within its first lines.

**Fix**: Load `("mlfp01","hdb_resale.parquet")` / `("mlfp01","economic_indicators.csv")` and derive year from `month`; rewrite 2.3/2.4/2.7 on `experiment_data.parquet` (experiment_group, metric_value, pre_metric_value, revenue); point 2.6 at the HDB high_price setup ex_6 uses; point 2.8 at an existing dataset; complete the 2.7 CUPED code with pooled θ.

### [BLOCKING] Lesson-page Kailash code uses APIs absent from installed kailash-ml 2.2.2

**File**: lessons/04/textbook.html:279-291; lessons/05/textbook.html:373-374; lessons/08/slides.html:110-139, 238-267; lessons/08/textbook.html:101-184, 262-290

**Evidence**: `import kailash_ml.experiment_tracking` → ModuleNotFoundError; no `log_hypothesis`/`log_sample_size` on ExperimentTracker; FeatureEngineer has only `generate`/`select` (no `create_temporal_features`, `create_interactions`); FeatureStore(dataflow) has no `register`, and `get_features(schema, timestamp=None, *, tenant_id, entity_ids)` takes no `name`/`as_of`; TrainingPipeline(feature_store, registry) has only `.train(...)`; TrainingResult has no coefficients/t_stat/p_value/r_squared/significant_features. lessons/05 textbook claims TrainingPipeline "stores the coefficients with their standard errors" — training_pipeline.py has no SE/p-value code.

**Problem**: The capstone lesson's entire engine section and the 2.4 tracker example fail on import/construction; the 2.5 engine claim is false.

**Fix**: Rewrite against the real API (async `ExperimentTracker.create` + `track`; `FeatureSchema` + `FeatureEngineer.generate`; `FeatureStore(DataFlow)` + `get_features(schema, timestamp=...)`); drop the SE/p-value claim.

### [BLOCKING] Lesson 2.4 sample size halved (≈5,800 per arm stated; formula gives ≈11,554 per arm)

**File**: lessons/04/slides.html:125, 198; lessons/04/textbook.html:320, 291; lessons/04/notes.html:107-110, 190-192

**Evidence**: The lesson's own `int(np.ceil(2*(z_a+z_b)**2*0.08*0.92/0.01**2))` = **11554**; slides/textbook say "n ≈ 5,800 per arm"; notes: "≈11,600 — wait, per arm is half. Around 5,800 per arm."; textbook tracker call logs `n_per_arm=7_850`.

**Problem**: The formula already yields n per arm; halving it teaches running experiments at roughly half the required sample (and duration "12 days at 1,000 users/day" should be ~23).

**Fix**: ≈11,600 per arm (≈23,100 total) everywhere; make logged value and duration consistent; remove the "wait" draft text from the notes.

### [MAJOR] Lesson 2.1 Bayesian-update numbers wrong

**File**: lessons/01/slides.html:217-218; lessons/01/notes.html:224-227

**Evidence**: Slide inputs μ0 = 480k, σ0 = 40k, x̄ = 542k, σ = 60k, n = 30 → computed `537674.4, 10565.4`; slide states `mu_post ≈ 533,000`, `sigma_p ≈ 9,600`.

**Fix**: ≈537,700 and ≈10,600.

### [MAJOR] Lesson pages diverge from the master deck (different cases, data and code)

**File**: lessons/04/* vs deck.html:1063, 1217-1250; lessons/06/* vs deck.html:1600-1630; index.html:123; lessons/08/* vs deck.html:2257-2269

**Evidence**: Deck 2.4 case "Singapore Hawker Centre: Should We Raise Prices?"; lesson 04 pages use a BOGO case with a different (also invalid) tracker API. Deck 2.6 case "Will This Employee Resign?" (spec 2.6); lesson 06 pages + index card use "ICU mortality prediction"; ex_6 actually uses HDB `high_price` (shared/mlfp02/ex_6.py:56). Deck 2.8 `store.register_feature(...)` vs lessons `store.register(...)`.

**Problem**: Students get different cases/code for the same lesson depending on which page they open.

**Fix**: Regenerate lessons/NN/* from the master deck/textbook (after the master is corrected).

### [MAJOR] Lesson 2.8 textbook uses pandas

**File**: lessons/08/textbook.html:252

**Evidence**: `corr = df.select(pl.exclude("quality")).to_pandas().corr()`

**Fix**: Use polars (`df.select(pl.exclude("quality")).corr()` or `pl.corr` selects as on line 254). (Same violation in master textbook.md:2621, 3496, 3670, 3734 — see textbook API finding.)

### [MINOR] Lesson 2.4 report example: CI and p-value inconsistent

**File**: lessons/04/slides.html:234; lessons/04/textbook.html:367-368

**Evidence**: "1.3 pp (95% CI 0.6 to 2.0, p = 0.003)" — half-width 0.7 → SE ≈ 0.357, z ≈ 3.64, p ≈ 0.0003; p = 0.003 implies CI ≈ 0.44–2.16.

**Fix**: e.g. "CI 0.4 to 2.2, p = 0.003".

### [MINOR] Incorrect statements in lesson derivations/explanations

**File / Evidence / Fix**:
- lessons/02/slides.html:167 — "The cross-term vanishes because each X_i−μ is independent of X̄−μ" — false (Σ2(X_i−μ)(X̄−μ) = 2n(X̄−μ)², combining to −n(X̄−μ)²). Use the identity in lessons/02/textbook.html:249.
- lessons/01/textbook.html:632-633 — Bessel's correction "derived from the Normal MLE"; the MLE gives the biased /n estimator. Reword.
- lessons/01/notes.html:190-191 — E[X²]−μ² "is what numpy uses under the hood for one-pass variance"; numpy `_var` is two-pass (mean first). Remove.
- lessons/06/textbook.html:150-152 — BFGS described as Newton-Raphson/IRLS "under the hood"; BFGS is quasi-Newton. Reword.
- lessons/06/textbook.html:476, 510 — 6 pairwise t-tests give FWER ≈26%; simulation of the exact exercise (4×30 N(0,1), 4,000 reps) gives **0.202** (pairwise tests are correlated). Say ≈20% (≤26%).

### [MINOR] 4800/5200 called "borderline" though the lesson's own test flags clear SRM

**File**: lessons/07/textbook.html:391

**Evidence**: Same lesson's SRM table: χ² = 16, p = 0.00006, "SRM"; textbook's own code flags it at 1%.

**Fix**: Call it clear SRM; use e.g. 4900/5100 for borderline.

### [MINOR] Lesson notes' slide numbering does not match the slide pages

**File**: lessons/0{1..8}/notes.html headers

**Evidence**: `<section>` counts vs notes claim/entries: 01 15 vs 14/14; 02 20 vs 17/16; 03 16 vs 14/14; 04 14 vs 13/13; 05 17 vs 15/15; 06 18 vs 14/14; 07 15 vs 14/13; 08 14 vs 16/14.

**Fix**: Regenerate notes against current slides.

### [MINOR] index.html dataset table and README exercise table wrong

**File**: index.html:76-79; README.md:22-29

**Evidence**: index lists `ab_test.parquet` (2.3/2.4/2.7) and `did_policy.parquet` (2.7) — neither exists (ex_3/4/7 use experiment_data.parquet; DiD is simulated by `simulate_hdb_cooling_measures`); lists icu_patients.parquet for 2.6 but ex_6 uses hdb_resale. README says ex_3 "HDB / A/B test" and ex_8 "Mixed datasets"; each uses one dataset (experiment_data, hdb_resale).

**Fix**: Correct both tables.

### [MINOR] "Course Home" link broken

**File**: index.html:36

**Evidence**: `href="../index.html"` → `modules/index.html` does not exist (same pattern in mlfp01–06).

**Fix**: Point to an existing course home or create it.

## Assessment (assessment/README.md, task_1..4)

All four graders pass against their own solutions (13/13, 14/14, 14/14, 12/12; run with PYTHONDONTWRITEBYTECODE=1, writing only to the existing git-ignored .data_cache).

### [MAJOR] Task 4 calls feature_timestamp a point-in-time anchor but aggregates events after it

**File**: assessment/task_4/problem.md:19-38, 75-76; task_4/solution.py; task_4/grader.py (`_reference`)

**Evidence**: Joining event timestamps to admit_time: vitals 30,326 / 69,421 after admit (29,548 after discharge); labs 15,044 / 30,000 (14,780); meds 10,004 / 20,000 (9,851). Problem says "with a point-in-time anchor" and "No leakage" but no `timestamp <= feature_timestamp` filter exists in problem, solution or grader reference.

**Problem**: Contradicts the 2.8 point-in-time lesson being assessed; a student who correctly filters to the anchor fails the vitals/labs/meds checks. Synthetic event timestamps are unrelated to stay windows (events years before admit).

**Fix**: Add an explicit "events with timestamp ≤ admit_time" rule to problem/solution/grader (and regenerate events within stays), or stop calling it point-in-time.

### [MAJOR] Task 2 rewards "significant uplift" on a cohort Task 1 shows is SRM-contaminated

**File**: assessment/task_2/problem.md:7-11, 62; task_2/grader.py:178 (`bootstrap_ci_excludes_zero`)

**Evidence**: Task 1 reference on the same cohort: `SRM chi2 = 1535.46, flag = True`. Task 2 asks for "a confidence interval that survives a sceptic" and awards "CI excludes zero (significant lift)". deck.html:3122 "Do NOT proceed with analysis on SRM-contaminated data"; lessons/07/textbook.html:193 "If SRM is detected, do not trust the A/B test results".

**Fix**: Frame Task 2 as illustrative and require students to state why it cannot support a ship decision, or use an SRM-free cohort.

### [MAJOR] Assessment starters/problems dictate every formula and the exact code

**File**: assessment/task_2/starter.py:42-60; task_1/starter.py:38-54; task_3/starter.py (TODO block); task_*/problem.md ("Required computation")

**Evidence**: Task 2 starter contains the bootstrap loop verbatim plus `theta = np.cov(metric, pre, ddof=1)[0,1] / np.var(pre, ddof=1)` and the CUPED/BH steps; Task 1 TODOs spell out each Bayes/χ²/Beta formula. README claims "the difficulty is in the statistics (which population, which covariate, which correction)" but every choice is fixed by the problem.

**Problem**: Violates specs/redlines.md R6 ("BLOCKED: … Exercises that can be solved by copy-pasting") and domain-integrity.md (AI-resilient assessment).

**Fix**: Give the business question and deliverable keys only; remove formula/code dictation; let the grader verify outcomes.

### [MAJOR] Assessment omits several module outcomes and never uses a Kailash engine

**File**: assessment/README.md:14-19; all task_*/solution.py

**Evidence**: No MLE/MAP fit, no ANOVA/Tukey, no power analysis/MDE, no permutation test, no DiD (all module outcomes in specs/module-2.md). `grep -ri kailash assessment/` → nothing; Task 4 is titled "Feature Store" but uses no FeatureStore/FeatureEngineer/ExperimentTracker.

**Problem**: Conflicts with R6 ("covers the full module") and domain-integrity.md ("MUST NOT test generic ML theory instead of Kailash SDK patterns").

**Fix**: Add MLE/MAP, ANOVA+Tukey, MDE, DiD+parallel-trends items; make Task 4 build and query a real FeatureStore.

### [MINOR] "12 automated checks" does not match the graders

**File**: assessment/task_{1,2,3}/problem.md ("Grading (12 automated checks…)"); task_1/grader.py:12

**Evidence**: Graders report max = 13 (task 1), 14 (task 2), 14 (task 3), 12 (task 4).

**Fix**: Correct the counts.

### [MINOR] Task 3 grader never checks the required f_p_value

**File**: assessment/task_3/grader.py (`grade()`)

**Evidence**: `f_p_value` is in REQUIRED_KEYS; no `c[...]` check compares it.

**Fix**: Add `_close(r["f_p_value"], ref["f_p_value"], rtol=1e-3, atol=1e-9)`.

### [MINOR] Task 3 grades a "strongest predictor" that is a statistical tie

**File**: assessment/task_3/grader.py (`strongest_predictor_correct`)

**Evidence**: Reference IRLS: β(debt_to_income) = 0.42976 vs β(credit_utilization) = 0.42684; difference −0.0029, SE 0.0133 (z = −0.22).

**Problem**: Exact-string grading on noise, in a task whose lesson is "significance ≠ importance".

**Fix**: Accept either, or ask whether the top two differ significantly.

## Cross-cutting

### [MAJOR] Real companies named with fabricated internal activities/statistics; prior-curriculum reference

**File**: Exercises — DBS/OCBC/UOB (solutions/ex_2/02, ex_2/05, ex_4/03, ex_6/03:293-308, ex_8/02:365-391, ex_8/04:341), GIC (ex_2/03), Grab/GrabPay (ex_4/02, ex_7/02:67-70, ex_7/03:71-74, 225-239), Shopee (ex_3/01 "Shopee-scale", ex_4/01, ex_7/01:76-78, 283-296), Knight Frank/Edmund Tie (ex_6/01:290), PropertyGuru (ex_8/01:279-306), ERA (ex_8/03:308-362) — in solutions and local copies. Deck — deck.html:3313-3316 (Google Analytics, Salesforce, Mixpanel, SAP), notes 1171/3130 (Microsoft ExP), 3242 (Uber, Airbnb, Stripe). Textbook — textbook.md:3192-3193 (Microsoft, Netflix, Airbnb). Lessons — lessons/04/notes.html:52 ("the workflow Booking.com, Grab and Shopee all run"); lessons/04/textbook.html:201 ("framework from the original MLFP curriculum").

**Evidence**: e.g. "At Shopee (Singapore), CUPED reduced experiment duration from 14 days to 7 days"; "At Grab… ~30% of 'significant' results were false positives"; "DBS tests a new cashback tier"; and the DBS figures contradict each other (`annual_apps = 25_000  # DBS HDB mortgage applications per year`, ex_6/03:308, vs "DBS processes ~8,000 HDB mortgages per month", ex_8/02:371).

**Problem**: None is an institutional partner or funder, but these are unverified claims attributed to named commercial entities, presented as fact; `rules/independence.md` says "No proprietary product names … No commercial entities". The "original MLFP curriculum" phrase references a prior course iteration (CLAUDE.md directive 6). Scholarly citations (Deng et al. 2013, Kohavi et al.) are acceptable; MAS/URA/data.gov.sg as public-sector context is acceptable where the claims are accurate (see ex_2/02 and ex_8/04 findings for inaccurate ones).

**Fix**: Replace with generic actors ("a Singapore bank", "a ride-hailing platform", "an e-commerce marketplace", "web analytics tool / CRM / ERP"), label figures as illustrative, and drop the prior-curriculum reference.

## Coverage summary

**What was checked**

- **Master deck**: deck.html (3,514 lines, 99 slides). Every slide's text and speaker-notes aside was read. All 11 `<pre>` code blocks were introspected against the installed kailash-ml with `inspect.signature`/`hasattr`. KaTeX currency `$` collisions were scanned; none remain.
- **Master textbook**: textbook.md (4,358 lines) read in full. Every worked numeric example was recomputed (Bayes, Normal-Normal, CI, power, BH, SRM, OLS via `np.linalg.lstsq`, ANOVA, CUPED, DiD). Every code block was checked against the installed API and the dataset schema.
- **Speaker notes**: speaker-notes.md (1,031 lines) was compared with the deck structure.
- **Exercises**: 33 solution technique files, 33 local scaffolds, 8 `__init__` files, and 9 shared/mlfp02 helpers. Each solution/local pair was diffed and checkpoint asserts compared. Key numbers were recomputed in `.venv`. No full solution scripts were run.
- **Assessment**: 17 files. All four graders were run against their solutions (13/13, 14/14, 14/14, 12/12).
- **Lessons and module pages**: 24 lesson pages plus index.html and README.md were converted to text and read. Every relative href/src was checked (one broken link), and dataset paths/columns and API signatures were verified.
- **Rules**: independence/naming checked across all of the above. No hardcoded LLM model names and no `import pandas` were found; pandas usage appears only via `.to_pandas()` in textbook.md and lessons/08.

**Spec coverage (specs/module-2.md)**

- **Deck and textbook** teach essentially every listed topic. Gaps:
  - BCa appears only in deck notes, though it is in the textbook.
  - The capstone wine-quality option has no data.
  - The textbook defers inverse propensity weighting and leakage via broken cross-references.
- **Exercises are thinner than the spec**:
  - Not exercised: Poisson; Exponential fit to economic data; LLN; parametric bootstrap; one-sample/one-tailed tests; likelihood plot; ExperimentTracker in the 2.4 A/B exercise; log-linear model; k-fold CV; lat/lon features; multiclass logit; DiD placebo test; FeatureEngineer.
  - Replaced rather than exercised: the employee-attrition data (spec 2.6) is replaced by an HDB above-median target; the cooling-measures DiD runs on simulated means.
  - Broken or never-executed: the capstone has no project options and no logistic model; Tukey HSD is mis-implemented; the FeatureStore steps never execute.
  - TrainingPipeline (listed among the M2 engines) is used in no M2 solution.

**Not run / excluded**: full solution execution (handled separately), the colab-selfcontained* notebooks, and the items listed as already known.
