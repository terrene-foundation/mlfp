# MLFP04 Audit — Unsupervised ML and Advanced Techniques

Repo: /Users/esperie/repos/lyceum/courses/mlfp · module `modules/mlfp04/` · installed stack: kailash-ml 2.2.2, polars 1.41.2 (introspected via `.venv/bin/python`). Read-only audit; no repo files changed. Colab notebooks ignored; ALREADY-KNOWN items excluded.

**Totals: BLOCKING 15 · MAJOR 38 · MINOR 44**

Headline themes:
1. Almost every "Kailash Engine" code sample in deck, textbook, lesson pages and notes uses fabricated or stale APIs (AutoMLEngine modes, EnsembleEngine score blending, OnnxBridge.load, ModelVisualizer.scree_plot/topic_heatmap/line/heatmap, DriftMonitor(reference=...)), while the engines the exercises and assessment actually use (ClusteringEngine, DimReductionEngine, AnomalyDetectionEngine, SklearnTrainable) are never taught.
2. textbook.md and lesson textbook pages load datasets that do not exist (8 + 4 files), so no worked example runs and quoted results are fabricated.
3. Several exercises produce meaningless evidence: circular labels (ex_4 is_fraud from num_returns; ex_5.4 target = sum of features; ex_6.5 lexicon labels), random-noise data (ex_8.3-8.5), hash-vector "Word2Vec", degenerate recommender evaluation (ex_7), NaN intrinsic dimension (ex_3.5), silently-failing EnsembleEngine call (ex_4.4).
4. Spec 4.8's central exercise (from-scratch NN on HDB) does not exist, yet deck, notes and lesson pages describe it.
5. Assessment covers 4 of 8 lessons, is fill-in-the-blank, and the deck describes a quiz + project that does not exist. All 4 reference solutions pass their graders (10/10 each, run locally).

## BLOCKING and MAJOR index
- [BLOCKING] textbook.md worked examples and drills load datasets that do not exist anywhere in the repo or data manifest
- [BLOCKING] lessons/04–07 textbook pages load non-existent datasets
- [BLOCKING] textbook.md "Kailash Engine" sections call APIs that do not exist (AutoMLEngine modes, EnsembleEngine.blend(scores=), ModelVisualizer.line/heatmap, OnnxBridge.export(input_shape=)/load)
- [BLOCKING] lessons/01 and lessons/04 textbook code calls non-existent module / wrong kwargs / wrong EnsembleEngine signature
- [MAJOR] lesson textbook pages claim engine behaviour that does not exist (AutoMLEngine clustering sweep, search_space=, ModelVisualizer.scree_plot/topic_heatmap/embedding projection, AutoMLEngine recommender backend)
- [MAJOR] textbook.md Lesson 4.1 Drill 3 answer key is wrong (DBSCAN eps=2.0 does not merge the blobs)
- [MAJOR] textbook.md Isolation Forest intuition inverts the exponent
- [MAJOR] textbook.md claims kernel PCA has a "well-defined reconstruction pre-image"
- [MAJOR] textbook.md PCA reconstruction-error formula is off by a factor of (n−1)
- [MAJOR] lessons/07 textbook says content-based filtering works for brand-new users
- [MAJOR] textbook.md teaches pandas and a removed Polars API in Lesson 4.5
- [MAJOR] textbook.md Lesson 4.6 drills import gensim, which is not installed; LDA is fit on TF-IDF
- [MAJOR] textbook.md drill "solutions" that are stubs
- [MAJOR] Lesson textbook "Exercise N" briefs describe tasks the actual exercises do not contain (notably Exercise 8)
- [MAJOR] Assessment covers only 4 of 8 lessons and omits the spec's integrated project
- [MAJOR] Assessment tasks are fill-in-the-blank: the problem/starter dictate the exact calls
- [BLOCKING] AutoMLEngine slide code does not match installed API (master deck + lesson 4.1)
- [BLOCKING] EnsembleEngine "anomaly score blending" slide code and concept are wrong
- [BLOCKING] OnnxBridge/TrainingPipeline slide code is not the installed API; notes overstate OnnxBridge
- [BLOCKING] Slides call non-existent ModelVisualizer and DriftMonitor methods
- [MAJOR] Assessment slide describes a quiz + end-to-end project that does not exist
- [MAJOR] "Looking Ahead: Module 5" previews Module 6 content
- [MAJOR] Exercise slides describe work the exercises do not contain (worst: 4.8)
- [MAJOR] Universal approximation theorem misstated as requiring "2+ hidden layers"
- [MAJOR] EM wrongly called the backbone of matrix factorisation and neural-network training
- [MAJOR] Lesson 4.3 slide claims UMAP inter-cluster distances are meaningful
- [MAJOR] Lesson 4.7 item-based CF worked example is logically wrong
- [MAJOR] Per-lesson slide decks omit spec topics; divisive clustering missing everywhere
- [BLOCKING] EnsembleEngine.blend() called with wrong kwargs; TypeError silently swallowed
- [BLOCKING] Levina–Bickel intrinsic-dimension estimator always returns NaN, and the taught formula is wrong
- [BLOCKING] LOF application says LOF flags a tight, dense fraud cluster — LOF does the opposite
- [MAJOR] ex_4 anomaly label is a threshold on an input feature (circular evaluation); data is not financial transactions
- [MAJOR] Clustering / GMM / dim-reduction features include the outcome label `churned`
- [MAJOR] ex_4.4 theory: "ensemble always wins" because blend errors are the intersection of detector errors
- [MAJOR] ex_3 ranks reducers by K-means silhouette in the embedding, contradicting its own t-SNE caveats
- [MAJOR] No t-SNE/UMAP embedding plot or loadings heatmap although spec and headers promise them
- [MAJOR] From-scratch EM is never compared with sklearn GMM, though the header claims it
- [MAJOR] Gap statistic never exercised
- [MAJOR] AutoMLEngine only configured, never run, while the file claims "automated clustering comparison"
- [MAJOR] Spectral-vs-K-means silhouette interpretation wrong; silhouette computed in a different space than stated
- [MAJOR] Invented statistics attributed to named real organisations as "disclosed/published" figures
- [BLOCKING] local ex_5/02 crashes with NameError — speed sweep stripped but its variables still used
- [BLOCKING] Ex 5.4 target is a deterministic function of the baseline features — rule features cannot help
- [BLOCKING] Ex 7 recommender evaluation is degenerate — all methods tie random ranking; ALS is far worse than the global mean
- [BLOCKING] Ex 6.5 "Word2Vec" features are random hash vectors, labels are circular, test set fully leaked
- [MAJOR] Ex 6 corpus is 62 duplicated paragraphs, mostly ML textbook text — not Singapore news or reviews
- [MAJOR] Ex 6.3 fits LDA on TF-IDF and calls training perplexity "held-out"
- [MAJOR] Spec 4.6 gaps: TF-IDF not implemented from scratch; UMass coherence missing; wrong printed count
- [MAJOR] Spec 4.8 gaps: no from-scratch network, no HDB regression, thin loss/optimiser coverage
- [MAJOR] Ex 8.3-8.5 train on random-noise images with random labels — comparisons are meaningless
- [MAJOR] R9A: several files lack behaviour visualisation; local scaffolds strip required phases
- [MAJOR] Ex 8.5 says OnnxBridge cannot export torch models — installed kailash-ml can
- [MAJOR] Invented statistics attributed to named real organisations, some labelled "real production"

---

## Part A — Textbook (textbook.md + lessons/NN/textbook.html), README, index, assessment

### [BLOCKING] textbook.md worked examples and drills load datasets that do not exist anywhere in the repo or data manifest

**File**: modules/mlfp04/textbook.md:252, 629, 860, 1090, 1294, 1541, 1811, 2169

**Evidence**:
```
252:  df = loader.load("mlfp04", "sg_retail_customers.csv")
629:  df = loader.load("mlfp04", "sg_ecommerce_customers.csv")
860:  df = loader.load("mlfp04", "sg_ecommerce_features.csv")
1090: df = loader.load("mlfp04", "sg_transactions.csv")
1294: df = loader.load("mlfp04", "sg_retail_baskets.csv")
1541: df = loader.load("mlfp04", "sg_policy_submissions.csv")
1811: df = loader.load("mlfp04", "sg_retail_interactions.csv")
2169: df = loader.load("mlfp04", "sg_hdb_prices.csv")
```
`ls data/mlfp04/` → `preference_pairs.parquet  sg_domain_qa.parquet` only; `grep -rl` for these names across the repo (excluding .venv) matches only textbook.md / lessons/05/textbook.html / mlfp06/textbook.md. Every worked example in all 8 lessons therefore raises `FileNotFoundError`, and the "results" quoted after them (e.g. 4.1 "The elbow is at K=5 … Silhouette score peaks at K=5 with s=0.38", "HDBSCAN finds four clusters and labels approximately 800 points (5.3%) as noise", 4.3 "4 components explain ~80% of variance", 4.2 "Typically 15–25% of customers fall on boundaries") are outputs of code that cannot run. Column names used (`recency/frequency/monetary`, `amount/hour_of_day/merchant_risk_score/distance_km`, `floor_area_sqm/storey_range_mid/…/town_encoded`) also do not exist in any shipped dataset.

**Problem**: The chapter's "every derivation leads to running code" promise is false; all eight worked examples and all drills that build on them are non-runnable, and quoted numbers are fabricated.

**Fix**: Rewrite each worked example against the datasets the exercises actually use (`mlfp03/ecommerce_customers.parquet` for 4.1–4.4, `shared.mlfp04.ex_5.generate_transactions` for 4.5, `mlfp03/documents.parquet` for 4.6, `shared.mlfp04.ex_7.build_rating_dataset` for 4.7, `mlfp01/hdbprices.csv` for 4.8) and replace quoted numbers with real outputs.

### [BLOCKING] lessons/04–07 textbook pages load non-existent datasets

**File**: lessons/04/textbook.html:329 (`financial_transactions.parquet`), lessons/05/textbook.html:322 (`sg_retail_baskets.parquet`), lessons/06/textbook.html:335 (`sg_news.parquet`), lessons/07/textbook.html:328 (`sg_retail_purchases.csv`)

**Evidence**: `tx = loader.load("mlfp04", "financial_transactions.parquet")`, `baskets = loader.load("mlfp04", "sg_retail_baskets.parquet")`, `news = loader.load("mlfp04", "sg_news.parquet")`, `purchases = loader.load("mlfp04", "sg_retail_purchases.csv")`. None of these files exist (`ls data/mlfp04/` → only `preference_pairs.parquet`, `sg_domain_qa.parquet`). The "Try It Yourself → Exercise N" boxes then tell students to "Load the financial transactions dataset" (L04), "Load the Singapore retail basket dataset" (L05), "Load the Singapore news corpus" (L06) — the real exercises use `mlfp03/ecommerce_customers.parquet` with a proxy top-1%-returns label (shared/mlfp04/ex_4.py:68-80), synthetic `generate_transactions` (shared/mlfp04/ex_5.py:125), and `mlfp03/documents.parquet` (shared/mlfp04/ex_6.py:68).

**Problem**: Students following the lesson pages hit `FileNotFoundError` and are told the exercise uses data it does not.

**Fix**: Point each lesson page's code and Exercise brief at the dataset loaded by the corresponding `shared/mlfp04/ex_N.py`.

### [BLOCKING] textbook.md "Kailash Engine" sections call APIs that do not exist (AutoMLEngine modes, EnsembleEngine.blend(scores=), ModelVisualizer.line/heatmap, OnnxBridge.export(input_shape=)/load)

**File**: modules/mlfp04/textbook.md:219-227, 550-555, 838-850, 1077-1084, 1282-1286, 1527-1531, 1796-1800, 2151-2158

**Evidence** (installed kailash-ml 2.2.2, `.venv/bin/python` introspection):
- `AutoMLEngine.__init__(self, *, config: AutoMLConfig, tenant_id: str, actor_id: str, connection=None, …)`; public members `['actor_id','cost_tracker','run','tenant_id','trials']` — no `task=`/`method=` kwargs, no `.fit`, `.mine_rules`, `.extract_topics`, `.recommend`. So `AutoMLEngine(task="clustering")` (l.222), `AutoMLEngine(task="clustering", method="gmm")` (l.552), `AutoMLEngine(task="association").mine_rules(...)` (l.1284-5), `AutoMLEngine(task="topic_modelling").extract_topics(...)` (l.1529-30), `AutoMLEngine(task="recommendation").fit(...).recommend(...)` (l.1798-1800) all raise `TypeError`. `AutoMLConfig.task_type` only accepts `classification|regression|ranking|clustering` (automl/engine.py:162-170).
- `EnsembleEngine.blend(self, models: list, data: pl.DataFrame, target: str, *, weights=None, method='soft', …)` — `ensemble.blend(scores=[...], weights=[...])` (l.1080-1083) → `TypeError: unexpected keyword 'scores'`.
- `ModelVisualizer` methods: `box_plot, calibration_curve, confusion_matrix, feature_importance, histogram, learning_curve, metric_comparison, precision_recall_curve, residuals, roc_curve, scatter, training_history` — no `line` (l.842) or `heatmap` (l.847).
- `OnnxBridge.export(self, model, framework: str, schema=None, *, output_path=None, n_features=None, sample_input=None)`; members `['check_compatibility','export','validate']` — `bridge.export(model, input_shape=(1,4), output_path=...)` (l.2154) misses `framework` and passes unknown `input_shape`; `bridge.load(...)` (l.2156) does not exist.

**Problem**: Every "The Kailash Engine" section of the chapter teaches a fabricated API; students copying them get TypeError/AttributeError, and the real M4 engines are never named (see next finding).

**Fix**: Replace with the real surfaces: `kailash_ml.engines.clustering.ClusteringEngine().fit(df, algorithm=..., n_clusters=...)` / `.sweep_k(...)` (4.1, 4.2 with `algorithm="gmm"`), `kailash_ml.engines.dim_reduction.DimReductionEngine().reduce(df, algorithm="pca"|"nmf", n_components=...)` (4.3, 4.6), `kailash_ml.engines.anomaly_detection.AnomalyDetectionEngine().detect(...)` (4.4), `EnsembleEngine().blend(models, data, target, weights=...)` with a correct supervised framing, `ModelVisualizer.scatter/training_history/metric_comparison`, `OnnxBridge().export(model, framework="pytorch", output_path=..., sample_input=...)`. For 4.5 and 4.7, state plainly that no kailash engine exists (as lessons/05 does) instead of inventing one.

### [BLOCKING] lessons/01 and lessons/04 textbook code calls non-existent module / wrong kwargs / wrong EnsembleEngine signature

**File**: lessons/01/textbook.html:523-532 (code block), lessons/04/textbook.html:256-264

**Evidence**:
- `from kailash_ml.engines.automl_engine import AutoMLEngine, AutoMLConfig` → `ModuleNotFoundError: No module named 'kailash_ml.engines.automl_engine'` (engines package contains `anomaly_detection, clustering, data_explorer, dim_reduction, drift_monitor, ensemble, …`; AutoMLEngine lives in `kailash_ml.automl.engine`). `AutoMLConfig(task_type, metric_name, direction, search_strategy, max_trials, time_budget_seconds, …, agent, auto_approve, max_llm_cost_usd, min_confidence, seed)` — the page's `metric_to_optimize=` and `search_n_trials=` kwargs raise `TypeError`.
- lessons/04: `blended = ensemble.blend(scores, normalise=True, agg="mean")` vs `blend(self, models, data, target, *, weights, method, test_size, seed)` → `TypeError` (no `normalise`/`agg`, missing `data`, `target`). The same block passes `iforest.decision_function(X)` raw, which is *higher = more normal*, into a blend whose other inputs are *higher = more anomalous* (`-lof.negative_outlier_factor_`), so even a working blend would invert the IF signal; the page's own later block correctly negates it (`iforest_scores = -iforest.decision_function(X)`).

**Fix**: `from kailash_ml.automl.engine import AutoMLConfig` (or `from kailash_ml import AutoMLEngine`) with `metric_name=`/`max_trials=`; replace the blend snippet with the real signature or with plain normalised averaging, and negate `decision_function`.

### [MAJOR] lesson textbook pages claim engine behaviour that does not exist (AutoMLEngine clustering sweep, search_space=, ModelVisualizer.scree_plot/topic_heatmap/embedding projection, AutoMLEngine recommender backend)

**File**: lessons/01/textbook.html:536-544 (rendered text l.341-344), lessons/02/textbook.html:340-345, lessons/03/textbook.html:338-345, lessons/06/textbook.html:315-322, lessons/07/textbook.html:~300-310 ("Kailash Integration")

**Evidence**:
- L01: "Even with `agent=False`, the engine still performs a random search over K-means, GMM, HDBSCAN, and spectral clustering across a sensible hyperparameter grid, reporting silhouette, DB, and CH for each trial." — `AutoMLEngine.run(self, *, space, trial_fn, …)` docstring: "The caller owns the trainer — AutoMLEngine provides governance + audit only." No clustering algorithms are referenced anywhere in `kailash_ml/automl/*.py` (grep for hdbscan/spectral/gmm/KMeans returns nothing).
- L02: "Pass `search_space=["kmeans", "gmm", "hdbscan"]` and the engine will fit each, compute silhouette and BIC" — no such parameter.
- L03: "ModelVisualizer includes a `scree_plot()` helper"; L06: "ModelVisualizer has a `topic_heatmap()` helper … wraps BERTopic's own visualisation API"; L07: "AutoMLEngine handles recommendation tasks when configured with a collaborative filtering backend. The ModelVisualizer can project learned user and item embeddings into 2D" — none of these methods/backends exist (ModelVisualizer member list above; AutoMLConfig task_type list excludes recommendation).
- Meanwhile the real M4 engines (`ClusteringEngine`, `DimReductionEngine`, `AnomalyDetectionEngine`) — which the assessment *requires* — appear 0 times in textbook.md, all 8 lesson textbook pages, deck.html and speaker-notes.md (`grep -c` = 0 for each).

**Problem**: Students are taught a fictional engine surface and never shown the real one they are examined on.

Additionally the per-lesson slides/notes contain no bridge slide for these engines (deck-side evidence: `grep -c` = 0 in deck.html, speaker-notes.md and all 16 lessons/*/slides.html + notes.html).

**Fix**: Add a per-lesson "Kailash bridge" slide (4.1 ClusteringEngine, 4.3/4.6 DimReductionEngine, 4.4 AnomalyDetectionEngine, 4.8 SklearnTrainable + OnnxBridge) and replace each "Kailash Engine" box with the real engine and a call that matches `inspect.signature` (ClusteringEngine.fit/sweep_k, DimReductionEngine.reduce, AnomalyDetectionEngine.detect, ModelVisualizer.scatter); delete the scree_plot/topic_heatmap/recommender claims.

### [MAJOR] textbook.md Lesson 4.1 Drill 3 answer key is wrong (DBSCAN eps=2.0 does not merge the blobs)

**File**: modules/mlfp04/textbook.md:401

**Evidence**: Answer: "At ε = 2.0, everything merges into a single cluster because the neighbourhood radius is large enough to connect all three blobs." Running the drill's exact data (`make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)`, `DBSCAN(eps, min_samples=5)`): `eps=0.5 → 3 clusters, 10 noise`; `eps=2.0 → 3 clusters, 0 noise`; even `eps=5.0 → 3 clusters`.

**Problem**: Students who run the drill get a result contradicting the key and learn a wrong intuition about ε scale.

**Fix**: State that at ε=2.0 the three well-separated blobs remain 3 clusters with all noise absorbed; to show merging, use overlapping blobs (e.g. `cluster_std=2.5`) or a much larger ε (e.g. ≥ the inter-blob gap) and verify by running.

### [MAJOR] textbook.md Isolation Forest intuition inverts the exponent

**File**: modules/mlfp04/textbook.md:1041

**Evidence**: "if E[h(x)] is much smaller than c(n), the exponent is a large negative number, so s → 1 (anomalous)."

**Problem**: The exponent is −E[h]/c(n). If E[h] ≪ c(n) the exponent → 0⁻ (small magnitude), giving s = 2⁰ → 1. A "large negative" exponent gives s → 0 (very normal). The stated reasoning contradicts the formula one line above.

**Fix**: "…the ratio E[h]/c(n) is close to 0, so the exponent approaches 0 and s → 1 (anomalous). If E[h] ≈ c(n), the exponent ≈ −1 and s ≈ 0.5. If E[h] ≫ c(n), the exponent is large and negative and s → 0."

### [MAJOR] textbook.md claims kernel PCA has a "well-defined reconstruction pre-image"

**File**: modules/mlfp04/textbook.md:811

**Evidence**: "Kernel PCA … has the advantage of having a well-defined reconstruction pre-image."

**Problem**: The opposite is true: the kernel-PCA pre-image problem is ill-posed — a point in the implicit feature space generally has no exact pre-image in input space, so reconstruction requires approximate methods (fixed-point iteration, or sklearn's `fit_inverse_transform=True`, which learns an approximate inverse via kernel ridge regression).

**Fix**: "Unlike PCA, kernel PCA has no exact inverse: reconstructing input-space points (the pre-image problem) needs an approximation such as sklearn's `fit_inverse_transform=True`."

### [MAJOR] textbook.md PCA reconstruction-error formula is off by a factor of (n−1)

**File**: modules/mlfp04/textbook.md:791-793

**Evidence**: `‖X̃ − X̂̃‖_F² = Σ_{i=k+1}^{p} λ_i` where λ_i are eigenvalues of C = X̃ᵀX̃/(n−1) (l.757) and the same section states λ_i = σ_i²/(n−1) (l.781).

**Problem**: By Eckart–Young, ‖X̃ − X̂̃‖_F² = Σ_{i>k} σ_i² = (n−1) Σ_{i>k} λ_i. The textbook's equation is dimensionally inconsistent with its own definitions (lessons/03/textbook.html states it correctly in σ²).

**Fix**: `‖X̃ − X̂̃‖_F² = Σ_{i>k} σ_i² = (n−1) Σ_{i>k} λ_i`, i.e. the per-sample squared error is (n−1)/n · Σ_{i>k} λ_i.

### [MAJOR] lessons/07 textbook says content-based filtering works for brand-new users

**File**: lessons/07/textbook.html:136

**Evidence**: "Content-based filtering does not need other users' data. It works even for brand-new users (no cold-start problem on the item side) as long as you have item features."

**Problem**: Content-based filtering solves *item* cold start (new items have features), but not *user* cold start: a brand-new user has no liked items from which to build the profile (the page's own snippet builds `user_profile = item_features[purchased_indices].mean(axis=0)`). textbook.md:1715 and the "Hybrid systems" paragraph on the same page state the correct direction.

**Fix**: "It works for brand-new *items* (no item cold-start) as long as they have features; it still needs some history for a new *user*."

### [MAJOR] textbook.md teaches pandas and a removed Polars API in Lesson 4.5

**File**: modules/mlfp04/textbook.md:1297-1300, 1305-1309, 1338-1404; lessons/05/textbook.html:316 block

**Evidence**: `basket = df.pivot(index="transaction_id", columns="product", values="quantity")`; `basket_pd = basket.to_pandas().set_index("transaction_id")`; `rules.sort_values(...)`, `rules.iloc[0]`, `freq_low["itemsets"].apply(len)`; lessons/05: `apriori(baskets.to_pandas(), …)`. Installed polars 1.41.2: `DataFrame.pivot(self, on, on_columns=None, *, index=None, values=None, …)` — there is no `columns=` parameter, so the pivot raises `TypeError` before pandas is even reached. CLAUDE.md Directive 2: "No pandas in any exercise … Every data operation uses polars or kailash_ml APIs."

**Problem**: The lesson's core code is both broken and in violation of the course's polars-only directive (the actual ex_5 solutions implement Apriori in polars/Python).

**Fix**: Use `df.pivot(on="product", index="transaction_id", values="quantity")`; build the one-hot basket in polars and either reuse the from-scratch Apriori in `solutions/ex_5/01_apriori_from_scratch.py` or isolate the mlxtend call behind a helper that converts at the boundary (and say why), keeping rule post-processing in polars.

### [MAJOR] textbook.md Lesson 4.6 drills import gensim, which is not installed; LDA is fit on TF-IDF

**File**: modules/mlfp04/textbook.md:1610-1611 (Drill 2), 1626-1640 (Drill 3), 1559, lessons/06/textbook.html:328 block (`LatentDirichletAllocation(...).fit(X_tfidf)`)

**Evidence**: `from gensim.models.coherencemodel import CoherenceModel` — `.venv/bin/python -c "import gensim"` → `ModuleNotFoundError`. `lda.fit_transform(tfidf_matrix)` / `.fit(X_tfidf)`.

**Problem**: The NPMI-coherence drills (the spec's required topic-quality metric) cannot run. LDA is a generative model of integer word counts; feeding TF-IDF weights violates its likelihood (sklearn's LDA docs specify a term-count matrix), so the NMF-vs-LDA comparison in the drills and worked example is methodologically wrong.

**Fix**: Compute NPMI with the course's own implementation (the one in `solutions/ex_6`) or add gensim to dependencies; fit LDA on `CountVectorizer` counts while keeping TF-IDF for NMF.

### [MAJOR] textbook.md drill "solutions" that are stubs

**File**: modules/mlfp04/textbook.md:1393-1409 (4.5 Drill 5), 1909-1918 (4.7 Drill 3), 686-697 (4.2 Drill 4)

**Evidence**: 4.5 Drill 5: `# Add binary feature based on co-occurrence` / `# (implementation depends on data structure)` then uses undefined `X_base`, `X_with_rules`, `y`. 4.7 Drill 3: body is only comments plus `print(f"k={k}: train_rmse=..., val_rmse=...")`. 4.2 Drill 4: calls undefined `compute_log_likelihood`.

**Problem**: Answer keys presented as "Solution" are placeholders (no-stubs rule) and cannot be run or checked.

**Fix**: Write complete solutions (rule-feature construction in polars + LogisticRegression comparison; an ALS train/validation split loop over k; a defined log-likelihood helper).

### [MAJOR] Lesson textbook "Exercise N" briefs describe tasks the actual exercises do not contain (notably Exercise 8)

**File**: lessons/08/textbook.html:360-370 (also l.208-214 Worked Example goal), lessons/05/textbook.html:360-368, lessons/04/textbook.html:382-390

**Evidence**: L08: "Exercise 8. Build a 3-layer neural network from scratch for HDB price prediction … Add dropout (p=0.2) … Add batch normalisation … Replace SGD with Adam … Add cosine learning rate scheduling and early stopping." Actual `solutions/ex_8/`: `01_xor_proof.py` (XOR), `02_activations_init.py`, `03_cnn_residual.py`, `04_optimisers_schedulers.py`, `05_regularisation_training.py`; `grep -il hdb solutions/ex_8/*.py` → no matches; the image data is random noise (`shared/mlfp04/ex_8.py:96` `X = rng.standard_normal(...)`). L05 asks for "a logistic regression to predict high-value baskets"; L04 for "the financial transactions dataset".

**Problem**: The lesson pages promise an exercise (and the spec 4.8 exercise "3-layer network from scratch for HDB price prediction") that does not exist; students cannot find it.

**Fix**: Either rewrite the Exercise boxes to describe ex_8.1–8.5 / ex_4 / ex_5 as built, or add the from-scratch HDB network the spec requires and keep the boxes.

### [MAJOR] Assessment covers only 4 of 8 lessons and omits the spec's integrated project

**File**: modules/mlfp04/assessment/README.md:18-26; specs/module-4.md:346 ("End of Module Assessment: Quiz + project (unsupervised analysis → DL bridge: cluster data, reduce dimensions, build neural network on discovered features)"); specs/redlines.md:105-116 (Redline 6: "Comprehensive — covers the full module")

**Evidence**: Tasks = clustering (4.1), PCA + Isolation Forest (4.3/4.4), NMF topics (4.6), MLPClassifier on circles (4.8). Nothing assesses EM/GMM (4.2), t-SNE/UMAP or kernel PCA (4.3), LOF/score blending/EnsembleEngine (4.4), association rules (4.5), LDA/BERTopic/NPMI (4.6), recommenders/ALS (4.7), or backprop/training toolkit (4.8). No quiz exists (`ls modules/mlfp04` has no `quiz/`). No task chains clustering → reduction → NN on discovered features.

**Problem**: Redline 6 and the spec's assessment design are not met; half the module's outcomes are unassessed.

**Fix**: Add tasks for 4.2, 4.5, 4.7 and an integrated cluster→reduce→NN capstone task (or extend existing tasks), and either add the quiz or record the spec deviation.

### [MAJOR] Assessment tasks are fill-in-the-blank: the problem/starter dictate the exact calls

**File**: assessment/task_1/problem.md:30-35 + starter.py:57-65; task_2/problem.md:31-41 + starter.py:49-58; task_3/starter.py:106-115; task_4/starter.py:56-67

**Evidence**: task_3 TODO 1 gives the literal code `pl.from_numpy(M, schema=[f"t{i}" for i in range(M.shape[1])])` and TODO 2 the literal call `DimReductionEngine().reduce(matrix_df, algorithm="nmf", n_components=N_TOPICS, seed=42)`; task_4 TODO 1 gives the full `MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=2000, random_state=SEED)` with `target=TARGET, metric="accuracy"`; task_1 problem gives `sweep_k(zdf, range(2, 9), algorithm="kmeans", criterion="silhouette")` and `fit(zdf, algorithm="kmeans", n_clusters=optimal_k)`. The reference `solve()` bodies are 6–15 lines (`diff starter.py solution.py`). All four reference solutions pass their graders (ran `grader.py solution.py` for tasks 1–4: 10/10 each).

**Problem**: Redline 6 BLOCKS "Fill-in-the-blank with one-line answers" and "Exercises that can be solved by copy-pasting"; a strong engineer completes all four in well under the 90-minute floor. The `ClusteringEngine.sweep_k`, `DimReductionEngine(nmf)` and `SklearnTrainable` APIs relied on are not taught in any module material (0 hits in deck/textbook/lessons/notes; `sweep_k` and NMF-via-engine are used in no exercise).

**Fix**: Specify outcomes, not calls (e.g. "recover the personas and justify K"), remove literal code from TODOs, add design decisions (scaling choice, K selection, contamination choice, architecture) and multi-step integration; teach the engines in the lessons the tasks draw on.

### [MINOR] Assessment starters reference grader.py that is stripped from the student repo

**File**: assessment/task_{1,2,3,4}/starter.py:9-10 ("python grader.py starter.py     # grade your attempt"); scripts/build-student-repo.sh:74-97

**Evidence**: build-student-repo.sh excludes `grader.py`, `solution.py` and `README.md` from the student assessment folder ("NEVER ship answer keys … grader.py re-derives the answer"). assessment/README.md also says graders are withheld.

**Problem**: Students are told to run a file they do not have; README (the only overview with the "10 checks"/marking information) is also stripped, so the "Grading (10 automated checks)" sections in problem.md are the only remaining description.

**Fix**: Remove the `python grader.py` line from starter docstrings (say "submitted to the portal for grading").

### [MINOR] task_3 problem says domain labels exist only in the grader, but the starter tells students to read them

**File**: assessment/task_3/problem.md:13-16; starter.py:112-113

**Evidence**: problem.md: "The portal does not know which document belongs to which domain — the domain labels exist only in the grader". starter TODO 4: "Compute topic_purity vs the true domains … true ids come from frame["category"]" and `load_documents()` filters on `category`.

**Fix**: Reword the scenario ("labels are available for evaluation only; do not use them to fit"), or drop `topic_purity` from the return contract and let the grader compute it.

### [MINOR] task_1 grader does not enforce the promised label range and crashes on negative labels

**File**: assessment/task_1/problem.md:47 ("Each label is an integer in [0, n_clusters)"); grader.py:113, 117-123

**Evidence**: `sizes = np.bincount(lab.astype(int), …)` runs before and outside any try — a submission containing −1 (e.g. HDBSCAN noise) raises `ValueError` from bincount and the grader exits with a traceback rather than a score; `labels_valid_ints` checks only `lab.min() >= 0`, not `< n_clusters`.

**Fix**: Wrap the bincount in the try block and add `int(lab.max()) < int(r["n_clusters"])` to `labels_valid_ints`.

### [MINOR] README.md and index.html dataset tables do not match what the exercises load

**File**: modules/mlfp04/README.md:22-31; modules/mlfp04/index.html (Datasets You Will Use table)

**Evidence**: index.html: "E-commerce customer features ~10K rows" (actual `mlfp03/ecommerce_customers.parquet` = 50,000 rows); "Synthetic 2D Gaussian mixture 1K points" (`N_SYNTH = 600`, shared/mlfp04/ex_2.py:56); "Financial transactions ~50K rows 4.4" (ex_4 uses ecommerce customers with a top-1% `num_returns` proxy label, shared/mlfp04/ex_4.py:68-80); "Singapore retail transactions ~20K baskets" (`generate_transactions(n=2500)` in every ex_5 file); "Singapore news articles ~5K documents" (`mlfp03/documents.parquet` = 500 mixed-topic docs incl. `ml_fundamentals`). README lists ex_4 "Financial transactions", ex_6 "Singapore news", ex_8 "HDB / synthetic" (ex_8 has no HDB data).

**Fix**: Update both tables to the real sources and sizes.

### [MINOR] textbook.md dropout description contradicts its own drill and PyTorch

**File**: modules/mlfp04/textbook.md:2044 vs 2268-2280

**Evidence**: l.2044: "During inference, dropout is turned off and activations are scaled by (1−p) to compensate." Drill 1 solution (l.2274) uses inverted dropout: `return a * mask / (1 - p)` in training, no scaling at eval — which is also what `torch.nn.Dropout` (used in ex_8) does.

**Fix**: Describe inverted dropout (scale by 1/(1−p) during training, identity at inference), optionally noting the original formulation.

### [MINOR] Small factual slips in textbook.md / lesson pages

**File / Evidence / Fix**:
- textbook.md:1503 "variational inference or collapsed Gibbs sampling — both are variants of the EM framework" — collapsed Gibbs sampling is MCMC, not EM. Fix: "variational inference (a variational-EM procedure) or collapsed Gibbs sampling (MCMC)".
- textbook.md:1009 "Distance-based approach … Isolation Forest and LOF use this idea" — Isolation Forest uses random partitioning, not distances. Fix: list IF as isolation/partition-based.
- textbook.md:461 "Mixture of Experts … the backbone of models like GPT-4"; lessons/02/textbook.html:318-320 "GPT-4 … use MoE layers … typically 2 out of 8" — GPT-4's architecture has not been published; present as reported/speculative or cite an openly documented MoE model.
- lessons/01/textbook.html:324 "Ward's linkage is mathematically equivalent to greedy K-means" — Ward greedily minimises the WCSS increase per merge; it is not equivalent to K-means. Fix: "Ward greedily minimises the same within-cluster sum of squares that K-means minimises."
- lessons/04/textbook.html:308 "DriftMonitor uses the same anomaly-detection machinery … the same algorithms" — kailash_ml/engines/drift_monitor.py implements PSI, KS-test, Jensen–Shannon, chi² (docstring l.3-5), not IF/LOF. Fix: describe it as distribution-shift testing.
- lessons/05/textbook.html:240-242 FP-Growth "is the default in modern libraries (mlxtend, SparkML, polars' own extensions)" — polars has no FP-Growth implementation. Remove "polars' own extensions".
- lessons/03/textbook.html:393 "Not centring the data … If you skip standardisation, the first PC will align with the direction of the mean" — conflates centring with standardisation. Fix: "If you skip centring…".
- textbook.md:1755 "In Lessons 4.1–4.6 … no loss function guided the discovery … PCA found directions by variance" while l.1751 says PCA and CF share "the same mechanism: minimise a reconstruction error" and K-means/NMF (taught in 4.1/4.6) also minimise explicit losses. Reword the pivot as "first time a model is fit iteratively to a *partially observed* target and its learned factors are reused as embeddings", or acknowledge that K-means/PCA/NMF are also optimisation problems.
- textbook.md:1488 promises "That derivation comes in Lesson 4.8, where you will see that Word2Vec is a shallow neural network" — Lesson 4.8 in textbook.md never mentions Word2Vec (`grep -n -i word2vec textbook.md` has no hit after l.1700). Add the derivation or remove the promise.
- textbook.md:1319-1322: `df_features = df.with_columns((pl.col("noodles") & pl.col("eggs")) …)` references columns that do not exist in the long-format `df` loaded two blocks earlier; and textbook.md:315 uses `labels_km`, never defined (the loop variable is `labels`).
- textbook.md:368-382 Drill 1 asks to "Verify that your implementation produces the same cluster assignments as sklearn.cluster.KMeans" and 4.6 Drill 1 (l.1581) to "Verify your result matches TfidfVectorizer"; neither solution performs the check, and the TF-IDF one would not match (sklearn default uses raw counts, smooth idf `ln((1+N)/(1+df))+1`, and L2 normalisation, not the textbook's length-normalised tf and `log(N/df)`).

### [MINOR] textbook.md is thinner than the spec on several required items (deck covers most)

**File**: modules/mlfp04/textbook.md vs specs/module-4.md

**Evidence** (`grep -c`): Isomap/LLE/MDS manifold-learning reference table (spec 4.3) — 0 in textbook.md and lesson textbook pages (deck.html:899-901 only); intrinsic dimension — 0 in textbook.md; SVD++ and hybrid systems (spec 4.7) — 0 in textbook.md; PReLU, Swish (activation list) and triplet/contrastive/reconstruction losses, step decay / one-cycle (spec 4.8 toolkit) — absent from textbook.md (PReLU and triplet loss appear only in speaker-notes.md:1028,1086, not on any slide); content-based recommender has no worked code in textbook.md 4.7.

**Fix**: Add the missing reference tables/paragraphs to textbook.md 4.3, 4.7, 4.8.

## Part B — Master deck, speaker notes, lesson slides & notes

### [BLOCKING] AutoMLEngine slide code does not match installed API (master deck + lesson 4.1)
**File**: modules/mlfp04/deck.html:502-517; lessons/01/slides.html:332-341; speaker-notes.md:205
**Evidence**: Deck `from kailash_ml import AutoMLEngine, AutoMLConfig` → `ImportError: cannot import name 'AutoMLConfig' from 'kailash_ml'`; deck `AutoMLConfig(task=..., algorithms=..., metric=..., n_trials=50)` → `TypeError: unexpected keyword argument 'task'`. Lesson 01 `from kailash_ml.engines.automl_engine import ...` → `ModuleNotFoundError`; `metric_to_optimize=`, `search_n_trials=` → TypeError. Installed: `AutoMLConfig(task_type, metric_name, direction, search_strategy, max_trials, …, agent, max_llm_cost_usd, …)`; `AutoMLEngine.__init__(self, *, config, tenant_id, actor_id, …)`; public methods `run(space=, trial_fn=)`, `trials`, … — no `fit`, `results.best_model`, `.labels`, `.probas`. The exercise (`solutions/ex_1/05_evaluation_profiling.py:43-51`) already uses `kailash_ml.automl.engine.AutoMLConfig(task_type=, metric_name=, max_trials=)`.
**Problem**: Code shown to students raises on line 1; notes l.205 tell instructors to pass `task="clustering"`.
**Fix**: Use the exercise's pattern (`from kailash_ml.automl.engine import AutoMLConfig`; `AutoMLConfig(task_type="clustering", metric_name="silhouette", direction="maximize", max_trials=20, agent=False, max_llm_cost_usd=1.0)`), show `AutoMLEngine(config=..., tenant_id=..., actor_id=...).run(space=..., trial_fn=...)` or drop fit/best_model lines; update notes l.205.

### [BLOCKING] EnsembleEngine "anomaly score blending" slide code and concept are wrong
**File**: deck.html:1066-1075; lessons/04/slides.html:181-191; speaker-notes.md:548-553; lessons/04/notes.html:203-205
**Evidence**: Installed `EnsembleEngine.blend(self, models: list, data: pl.DataFrame, target: str, *, weights=None, method='soft', test_size=0.2, seed=42)`; source raises ValueError unless method ∈ {"soft","hard"} and wraps sklearn VotingClassifier/VotingRegressor. `hasattr(EnsembleEngine(), 'add')` → False. Deck calls `ensemble.add(...)` and `blend(df, method="weighted_average")`; `ZScoreDetector` does not exist; IsolationForest/LocalOutlierFactor never imported. Lesson 04 calls `blend(scores={...}, normalise=True, agg="mean")`. Notes claim "score normalisation is automatic", "isotonic calibration … otherwise rank normalisation", "agg='mean' or agg='max'".
**Problem**: Every call raises; `blend()` is a supervised voting ensemble requiring a target — it does not normalise/blend unsupervised anomaly scores.
**Fix**: Show manual min-max normalisation + averaging (lesson 04 already does at slides l.165-173); if EnsembleEngine stays, present `blend(models=[...], data=df, target="is_fraud", method="soft")` over fitted estimators as `solutions/ex_4/04_ensemble_blending.py` does; remove calibration/auto-normalisation claims from notes.

### [BLOCKING] OnnxBridge/TrainingPipeline slide code is not the installed API; notes overstate OnnxBridge
**File**: deck.html:2015-2034; speaker-notes.md:1128-1130
**Evidence**: `TrainingPipeline.__init__(self, feature_store, registry)`, methods train/evaluate/retrain/calibrate (no `fit`); deck calls `TrainingPipeline(model="neural_network", hidden_layers=[128,64,32], activation="relu", optimizer="adamw", epochs=100, early_stopping=True)` and `pipeline.fit(...)`. OnnxBridge methods: check_compatibility, export, validate; `export(self, model, framework, schema=None, *, output_path=None, …)` requires `framework`; deck's `bridge.load(...)`/`loaded.predict(...)` don't exist. Notes: "OnnxBridge … handles the training loop, checkpointing, ONNX export, and inference serving."
**Fix**: Train in torch as `ex_8/05_regularisation_training.py` does, then `OnnxBridge().export(model, framework="pytorch", output_path=..., sample_input=...)` + `validate(...)`; rewrite notes l.1128 (export/validate only).

### [BLOCKING] Slides call non-existent ModelVisualizer and DriftMonitor methods
**File**: lessons/03/slides.html:332-338; lessons/03/notes.html:266-268; lessons/06/slides.html:503; lessons/06/notes.html:240; lessons/04/slides.html:234-247; lessons/04/notes.html:249-256
**Evidence**: ModelVisualizer methods: box_plot, calibration_curve, confusion_matrix, feature_importance, histogram, learning_curve, metric_comparison, precision_recall_curve, residuals, roc_curve, scatter, training_history — no `scree_plot`, no `topic_heatmap`. `DriftMonitor.__init__(self, conn: ConnectionManager, *, tenant_id, psi_threshold=0.2, ks_threshold=0.05, …)`, methods `check_drift`, `set_reference_data` — no `check`, no `method=`; slide shows `DriftMonitor(reference=X_train, method="iforest")` / `monitor.check(X_today)`.
**Problem**: Calls raise; claims that ModelVisualizer annotates loadings / wraps BERTopic heatmaps and that "DriftMonitor uses this (Isolation Forest)" are false (PSI/KS); "Full DriftMonitor coverage in M5" is wrong (spec: M3.8).
**Fix**: Build scree with plotly or ModelVisualizer.metric_comparison/scatter; drop topic_heatmap claim; real DriftMonitor constructor + set_reference_data/check_drift, described as PSI/KS; "M5" → "M3.8".

### [MAJOR] Assessment slide describes a quiz + end-to-end project that does not exist
**File**: deck.html:2109-2143; speaker-notes.md:1165-1172, 1200
**Evidence**: Deck lists a "Quiz" on 8 topics and "Project: End-to-end pipeline: cluster → reduce → detect anomalies → build NN → compare"; notes l.1200 "Remind them of the end-of-module project deadline". assessment/README.md: four independent auto-graded 25-mark tasks, no quiz, no project.
**Fix**: Rewrite slide and notes to the four-task format (or build the quiz/project — see Part A assessment-coverage finding).

### [MAJOR] "Looking Ahead: Module 5" previews Module 6 content
**File**: deck.html:2148-2167; speaker-notes.md:1181-1191
**Evidence**: "M5: LLMs, AI Agents & RAG Systems … Large Language Models: GPT architecture, fine-tuning … AI Agents … RAG". specs/_index.md: M5 = Deep Learning & Vision/Transfer (autoencoders, CNN, RNN, transformers, GAN, diffusion, GNN, transfer, RL); LLM/agents/RAG = M6. lessons/08/slides.html:601 has the correct M5 list.
**Fix**: Replace bullets with M5 spec topics; move LLM/agents/RAG to "Module 6".

### [MAJOR] Exercise slides describe work the exercises do not contain (worst: 4.8)
**File**: deck.html:2047-2053 (Ex 4.8), speaker-notes.md:1143-1146; deck.html:535 (Ex 4.1), 1119 (Ex 4.4), 1444 (Ex 4.6)
**Evidence**: Deck "Build a 3-layer neural network for HDB price prediction. Implement forward pass, loss (SSE), backpropagation, gradient descent"; notes "from scratch in polars + numpy … No framework". Actual solutions/ex_8: 01_xor_proof.py (torch XOR), 02_activations_init.py, 03_cnn_residual.py (CNN), 04_optimisers_schedulers.py, 05_regularisation_training.py — all torch, `grep -il hdb` none. Ex 4.1 "Use AutoMLEngine to compare all algorithms" — exercise only constructs an AutoMLConfig, never runs it. Ex 4.4 "financial transaction data" — exercise uses ecommerce_customers with is_fraud = top-1% num_returns. Ex 4.6 "Classify customer review sentiment using topic features" — 05_sentiment_word2vec.py uses Word2Vec doc vectors + lexicon on the news corpus.
**Fix**: Update exercise slides/notes to match real files; separately the spec's from-scratch HDB NN exercise is missing (see Part C).

### [MAJOR] Universal approximation theorem misstated as requiring "2+ hidden layers"
**File**: deck.html:1765; speaker-notes.md:967, 1012; lessons/08/textbook.html:184 ("Two or more hidden layers can approximate any continuous function (universal approximation theorem)")
**Evidence**: "Universal approximation theorem: A network with 2+ hidden layers and nonlinear activations can represent ANY continuous function"; notes l.967/1012 repeat it; notes l.972 correctly cite Cybenko/Hornik two-layer (one-hidden-layer) networks.
**Problem**: UAT needs ONE hidden layer with enough units, and guarantees approximation on a compact set, not exact representation; material self-contradicts (textbook.md:1979 is correct).
**Fix**: "A network with ONE hidden layer, a non-polynomial activation and enough units can approximate any continuous function on a compact domain to arbitrary accuracy; depth buys parameter efficiency."

### [MAJOR] EM wrongly called the backbone of matrix factorisation and neural-network training
**File**: deck.html:667 (aside 673)
**Evidence**: "EM works for ANY latent variable model … It is the backbone of LDA (4.6), matrix factorisation (4.7), and the training of neural networks (4.8)."; aside "…implicitly in backpropagation."
**Fix**: "EM applies to latent-variable models such as GMMs and LDA (variational EM). ALS in 4.7 shares the alternating structure but is not EM; neural networks are trained by gradient descent."

### [MAJOR] Lesson 4.3 slide claims UMAP inter-cluster distances are meaningful
**File**: lessons/03/slides.html:280-284
**Evidence**: "Local structure preserved AND inter-cluster distances meaningful" (vs t-SNE "inter-cluster distances are meaningless").
**Fix**: "Better global layout than t-SNE, but cluster sizes and inter-cluster distances are still not quantitatively meaningful."

### [MAJOR] Lesson 4.7 item-based CF worked example is logically wrong
**File**: lessons/07/slides.html:157-165
**Evidence**: Matrix shows Carol rated Film B = 2 and Film E = 5, Film E rated only by Carol; slide: "Carol likes Film E. Film E ~ Film B? → Recommend Film B to Carol."
**Problem**: Recommends an item Carol already rated (low); E–B similarity is uncomputable (no co-raters); the computed D≈A similarity is unused.
**Fix**: Use D≈A: a user who rated D highly and has "?" for A → recommend A.

### [MAJOR] Per-lesson slide decks omit spec topics; divisive clustering missing everywhere
**File**: lessons/0{1,3,6,8}/slides.html; deck.html:316-339
**Evidence**: 0 matches across all lessons/*/slides.html for: ARI/NMI external metrics (4.1); Isomap/LLE/MDS (4.3); UMass, sentiment analysis (4.6); softmax, gradient clipping, focal/contrastive/KL loss taxonomy, ELU, PReLU (4.8). Divisive (top-down) clustering (spec 4.1 "Agglomerative vs divisive") — 0 in deck, lesson slides and notes. PReLU, ELU, triplet loss only in speaker-notes.md:1028, 1086; deck activation/loss tables (deck.html:1840-1846, 1935-1943) lack them although notes l.1086 say "Walk the table … Triplet loss".
**Fix**: Add divisive bullet to both decks; add missing rows to master tables; add listed topics to per-lesson decks or link to master slides.

### [MINOR] Notes "Slide N" numbering out of step with lesson slides in 4 lessons
**File**: lessons/02/notes.html, lessons/03/notes.html, lessons/06/notes.html, lessons/08/notes.html
**Evidence**: slide/notes counts L02 19/18, L03 19/18, L06 18/17, L08 19/18. L03 slide 12 "Same data, different preservation guarantees" has no note (all later notes shifted); L06 slide 9 has no note ("comparison table on Slide 14" is slide 15); L08 slides 2, 4, 8 no notes, "From-scratch code" note duplicated, activation note names Softmax while slide shows Swish.
**Fix**: Regenerate these notes against current slides.

### [MINOR] Exercise paths in lesson slides/notes point to single files that do not exist
**File**: lessons/01/slides.html:370; lessons/01/notes.html:287; lessons/04/notes.html:216; lessons/05/notes.html:168
**Evidence**: `modules/mlfp04/solutions/ex_1.py`, `ex_4.py`, `ex_5.py` — exercises are directories (`solutions/ex_1/01_kmeans.py` …).
**Fix**: Point to `solutions/ex_N/` or the specific technique file.

### [MINOR] Master deck PCA/DR formula and table slips
**File**: deck.html:803, 827, 873-877
**Evidence/Fix**: l.803 `‖X − X̂‖² = Σ_{i>k} λ_i` with λ = covariance eigenvalues → should be `Σ σ_i² = (n−1) Σ λ_i` (same error as textbook.md:791). l.827 kernel PCA "No inverse transform" → "only an approximate learned inverse (`fit_inverse_transform=True`)". l.873 t-SNE "O(n²)" → Barnes-Hut (sklearn default) is O(n log n).

### [MINOR] ALS update formula has transposed shapes and ignores the observed-only mask
**File**: deck.html:1547-1548
**Evidence**: `U = (VᵀV + λI)⁻¹ VᵀRᵀ` with R users×items, V items×k → result is k×users (= Uᵀ); dense form treats missing ratings as zeros while the slide's objective sums only over observed entries.
**Fix**: Per-user form `u_u = (V_{Ω_u}ᵀ V_{Ω_u} + λI)⁻¹ V_{Ω_u}ᵀ r_{u,Ω_u}` (as lesson 07 code does).

### [MINOR] FP-Growth "single pass" contradicts master deck
**File**: lessons/05/slides.html:152; lessons/05/notes.html:154
**Evidence**: "Builds a compressed FP-tree in a single pass" vs deck.html:1173 / speaker-notes.md:608 "Two passes". FP-tree construction needs two scans (item counts, then ordered insertion).
**Fix**: "Two passes to build the FP-tree, then recursive mining."

### [MINOR] BERTopic UMAP dimensionality inconsistent
**File**: lessons/06/slides.html:418-419 vs lessons/06/notes.html:219-220
**Evidence**: Slide "UMAP — 2 dimensions"; notes "5 dimensions" (BERTopic default; solutions/ex_6/04_bertopic.py:67 "~5D").
**Fix**: "~5 dimensions (2D only for plotting)".

### [MINOR] Lesson 4.8 says a linear layer on [u;v] is "Same as MF"
**File**: lessons/08/slides.html:83-93
**Evidence**: "+ Hidden layer: W [u;v] + b — linear transform — Still linear. Same as MF."
**Problem**: MF score uᵀv is bilinear; a linear map of the concatenation is additive and cannot represent it.
**Fix**: "Still linear — and weaker than MF: cannot express uᵀv without a non-linearity."

### [MINOR] Lesson 4.1/4.2 slide wording slips
**File**: lessons/01/slides.html:152, 366; lessons/02/slides.html:299, 319
**Evidence/Fix**: l.152 "Both steps strictly decrease the inertia" → "never increase". l.366 "Run GMM and HDBSCAN with the chosen K" → HDBSCAN takes no K (min_cluster_size). L02 l.299 "GPT-4, Mixtral, DeepSeek — all use MoE" → GPT-4 architecture undisclosed; hedge (master notes l.313 say "believed to"). L02 l.319 "agree to 3 decimal places" → sklearn uses k-means init, reg_covar=1e-6, tol=1e-3; say "agree closely".

### [MINOR] Lesson 4.3 numbers contradict each other and the master deck
**File**: lessons/03/slides.html:37, 133, 324
**Evidence**: Outcome "compress 50 features into 6" vs worked example "15 features → 6 components"; rule of thumb "≥ 90%" vs deck.html:807/920/937 and speaker-notes.md:543 "95%".
**Fix**: One feature count; one threshold or "90–95% is the usual range".

### [MINOR] Factual errors in speaker/lesson notes asides
**File/Fix**:
- speaker-notes.md:273 "log-likelihood is concave in (mu, Sigma)" → concave in (μ, Σ⁻¹), not Σ.
- speaker-notes.md:980 "MSE = (1/2)·(y − ŷ)²" → that is SSE/2, not MSE.
- lessons/02/notes.html:172 "variational inference, which we use in M6 for LLM fine-tuning" → M6 fine-tuning (LoRA/DPO/GRPO) does not use VI; remove.
- lessons/06/notes.html:188-190 "Both [collapsed Gibbs and VI] integrate out theta and phi to sample z" → only collapsed Gibbs does; VI fits factorised approximate posteriors.
- lessons/04/notes.html:226-228 "Set it to 10% … your top-10 flagged points are all normal" → contamination only moves the threshold; ranking/top-10 unchanged; say "you flag 100× too many points".

### [MINOR] Unsourced claims about named companies' internal systems
**File**: lessons/07/slides.html:319; lessons/07/notes.html:92-94, 221; speaker-notes.md:800, 843, 845, 856
**Evidence**: "Most production recommenders at Shopee, Grab, and Netflix are hybrids"; "FairPrice Online's 'similar products' section is content-based"; "Shopee's recommendation system blends both"; "powers YouTube's recommender"; "Powered Spotify's early Discover Weekly".
**Problem**: Unverified assertions about specific companies (not Kailash comparisons, so not an independence violation; no institutional partners, universities, funders or course codes found in this slice).
**Fix**: Make generic or cite a source.

Not reported (verified OK): no hardcoded LLM model names in slide code (only `all-MiniLM-L6-v2` sentence-embedding model at lessons/06/slides.html:496, matching the exercise); lesson 05 support/confidence/lift arithmetic, lesson 08 forward-pass (0.46) and chain-rule (−0.84 → w = 0.584) numbers, lesson 01 CH-inertia equivalence all correct.

## Part C — Exercises ex_1 to ex_4 (solutions, local scaffolds, shared/mlfp04/ex_1..4.py)

### [BLOCKING] EnsembleEngine.blend() called with wrong kwargs; TypeError silently swallowed
**File**: modules/mlfp04/solutions/ex_4/04_ensemble_blending.py:173-183; modules/mlfp04/local/ex_4/04_ensemble_blending.py (TODO hint "Call engine.blend(estimators=estimators, X=X, weights=[...])")
**Evidence**: `engine.blend(estimators=estimators, X=X, weights=[...])` inside `except (TypeError, AttributeError): engine_blend = weighted_blend`. Installed `blend(self, models, data: pl.DataFrame, target: str, *, weights=None, method='soft', test_size=0.2, seed=42)`; running it → `TypeError EnsembleEngine.blend() got an unexpected keyword argument 'estimators'`.
**Problem**: The engine never runs; the "EnsembleEngine blend" row is a copy of the AUC-weighted blend while the reflection claims "Called kailash-ml EnsembleEngine.blend() with a custom adapter". Spec 4.4's `blend()/stack()/bag()/boost()` API is not exercised at all; the silent fallback violates zero-tolerance Rule 3; students following the local hint hit the same hidden failure.
**Fix**: Call the real API (`engine.blend(models=[...], data=df_with_label, target="is_fraud", weights=...)`, framed as a supervised classifier ensemble) or use `AnomalyDetectionEngine` for unsupervised score blending; remove the try/except fallback; fix the local hint; add at least one `stack`/`bag`/`boost` call.

### [BLOCKING] Levina–Bickel intrinsic-dimension estimator always returns NaN, and the taught formula is wrong
**File**: modules/mlfp04/solutions/ex_3/05_comparison.py:75, 206-225, 228, 355-357 (same in local)
**Evidence**: Formula taught `d_hat = 1 / mean(log(d_k/d_1))`. Code: `nn = NearestNeighbors(n_neighbors=max(k_values)).fit(X_s); dists,_ = nn.kneighbors(X_s); d_1 = dists[:, 0]; valid = (d_k > 0) & (d_1 > 0)`. Querying with training points returns each point as its own neighbour — on random 8-d data `d1 zero frac 1.0` — so `valid` is empty and the function returns NaN (tracker logs 0.0). With self-neighbour removed, this formula gives 2.69 on true d=8 data; the correct MLE gives 8.56.
**Problem**: The spec 4.3 intrinsic-dimension task prints `nan`, and the taught formula underestimates d by ~H_{k−1}.
**Fix**: Drop the self-neighbour (`kneighbors(n_neighbors=k+1)[:, 1:]`); use m_k(x) = [ (1/(k−1)) Σ_{j<k} log(T_k(x)/T_j(x)) ]⁻¹ averaged over points; correct the theory line.

### [BLOCKING] LOF application says LOF flags a tight, dense fraud cluster — LOF does the opposite
**File**: modules/mlfp04/solutions/ex_4/03_local_outlier_factor.py:257-264, 358; local/ex_4/03_local_outlier_factor.py APPLY block
**Evidence**: "The fraud cluster is TIGHT — it has high LOCAL density… Surrounding legitimate buyers have LOWER local density… LOF catches it because the ratio … is extreme." Same file l.63-65: "LOF >> 1.0 means p sits in a sparser pocket than its neighbours".
**Problem**: LOF(p) = mean lrd(neighbours)/lrd(p). A point inside a dense cluster of ≥ k members has LOF ≈ 1; denser than surroundings → LOF < 1 (most normal). The scenario teaches the inverse of the algorithm and contradicts its own theory.
**Fix**: Rewrite around points in sparse pockets next to dense clusters; state that anomaly groups larger than n_neighbors are masked from LOF.

### [MAJOR] ex_4 anomaly label is a threshold on an input feature (circular evaluation); data is not financial transactions
**File**: shared/mlfp04/ex_4.py:46-57, 68-79, 88 (affects every ex_4 file)
**Evidence**: `is_fraud = num_returns >= quantile(0.99)`; `FEATURE_BLOCKLIST` does not include `num_returns`, so it stays a feature. Threshold = 3.0, positive rate 1.382%, `num_returns` ∈ {0..6}.
**Problem**: Every AUC/AP comparison measures how much a detector weights `num_returns`; "ensemble beats single", "IF finds interactions", "true anomalies vs false positives" are artefacts. "fraud" is misleading; spec 4.4 asks for financial transaction data.
**Fix**: Use a dataset with an independent anomaly label, or at minimum drop `num_returns` from features and rename the label `high_returns`.

### [MAJOR] Clustering / GMM / dim-reduction features include the outcome label `churned`
**File**: shared/mlfp04/ex_1.py:63-69, shared/mlfp04/ex_2.py:105-110, shared/mlfp04/ex_3.py:58-63
**Evidence**: Feature filter keeps every Int/Float column except `customer_id` → `['total_revenue','order_count','avg_order_value','days_since_last_order','customer_tenure_days','satisfaction_score','num_returns','churned']`; `churned` is binary (37,706 vs 12,294).
**Problem**: "Behavioural segmentation" partly splits on a target label; the standardised binary column dominates Euclidean structure, distorts silhouette and yields degenerate full-covariance GMM components.
**Fix**: Exclude `churned` (use it only for post-hoc profiling); treat `satisfaction_score`/`num_returns` as discrete deliberately.

### [MAJOR] ex_4.4 theory: "ensemble always wins" because blend errors are the intersection of detector errors
**File**: solutions/ex_4/04_ensemble_blending.py:59, 69-70; local same block ("the blend's error set is strictly smaller than any single detector's error set")
**Evidence**: "The blend's errors are the INTERSECTION of the individual errors — strictly smaller than any single detector." The file's own checkpoint (l.205-207) only requires `best_ensemble_auc >= best_single_auc - 0.05`.
**Fix**: Blending reduces variance when detector errors are weakly correlated and scores comparably calibrated; it can underperform the best single detector. Rename the "Always Wins" heading.

### [MAJOR] ex_3 ranks reducers by K-means silhouette in the embedding, contradicting its own t-SNE caveats
**File**: shared/mlfp04/ex_3.py:78-93; solutions/ex_3/03_tsne.py:205; 04_umap.py:225; 05_comparison.py:9-10, 181-183, 285, 433-436
**Evidence**: 03_tsne.py:66-69 "Cluster SIZES… meaningless", "Distances BETWEEN clusters are meaningless"; yet best perplexity = `max(... silhouette)`; 05 "Always compare on the same silhouette ruler" across 2-D t-SNE and 6-D PCA.
**Problem**: Silhouette in an embedding measures blob-likeness, not structure preservation; t-SNE/UMAP inflate it by construction.
**Fix**: Use `sklearn.manifold.trustworthiness` / kNN-overlap (plus reconstruction error for PCA); keep silhouette only as "clusterability" with a caveat.

### [MAJOR] No t-SNE/UMAP embedding plot or loadings heatmap although spec and headers promise them
**File**: solutions/ex_3/03_tsne.py:22, 04_umap.py:22, 01_pca.py:23
**Evidence**: Headers promise "2D embedding scatter", "+ 2D scatter", "loadings heatmap"; `grep -n "scatter|go\.|heatmap" solutions/ex_3/*.py` finds no plot calls (only metric_comparison bars and training_history lines; loadings only printed).
**Problem**: Spec 4.3 "Compare t-SNE vs UMAP visualisations (vary hyperparameters)" and R9A visual proof unmet.
**Fix**: Add 2-D scatters per perplexity / UMAP config (coloured by K-means label or a profile feature) and a PC1–3 loadings heatmap.

### [MAJOR] From-scratch EM is never compared with sklearn GMM, though the header claims it
**File**: solutions/ex_2/02_sklearn_gmm.py:10, 287 (local identical)
**Evidence**: "Verify the library result matches the from-scratch EM from 2.1"; `grep fit_gmm_em` hits only 01_em_from_scratch.py.
**Problem**: Spec 4.2 assessment criterion "Comparison with GMM shows similar results" unmet.
**Fix**: Fit `GaussianMixture(3)` on `X_synth`, compare log-likelihood/weights/means (after label matching) with `fit_gmm_em`, assert closeness.

### [MAJOR] Gap statistic never exercised
**File**: solutions/ex_1/*.py
**Evidence**: `grep -i "gap stat"` over solutions/ex_1 → nothing; deck.html:311 "Silhouette and gap statistic are better. Show all three in the exercise."
**Fix**: Add a gap-statistic sweep (B uniform references in the bounding box, Gap(k)=E*[log W_k]−log W_k, 1-SE rule) to 01_kmeans.py.

### [MAJOR] AutoMLEngine only configured, never run, while the file claims "automated clustering comparison"
**File**: solutions/ex_1/05_evaluation_profiling.py:12-13, 197-214, 428
**Evidence**: WHAT YOU'LL LEARN "Use the kailash-ml AutoMLEngine to run automated clustering comparison"; code builds `AutoMLConfig(...)` then `_ = AutoMLEngine  # silence the import-not-used checker; engine used below` — it is not used below; `.run` never called.
**Fix**: Construct and `.run(...)` with a trial function (agent=False), or reword to "configure".

### [MAJOR] Spectral-vs-K-means silhouette interpretation wrong; silhouette computed in a different space than stated
**File**: solutions/ex_1/04_spectral.py:126, 138-141
**Evidence**: Comment "We pick K by silhouette in the SPECTRAL embedding space" but code `silhouette_score(X_spec, labels)` uses original features; printed "(positive = spectral wins; expected on non-convex structure)".
**Problem**: Euclidean silhouette is convex-biased and penalises correct non-convex partitions; students learn to read it as evidence for spectral.
**Fix**: Correct the comment; state the bias; demonstrate spectral's advantage on `make_moons` with ARI.

### [MAJOR] Invented statistics attributed to named real organisations as "disclosed/published" figures
**File**: e.g. solutions/ex_1/05_evaluation_profiling.py:318-319; ex_2/02_sklearn_gmm.py:190; ex_2/03_covariance_types.py:188, 192; ex_2/04_mixture_of_experts.py:240, 244; ex_3/01_pca.py:222-223; ex_3/03_tsne.py:193-194; ex_3/04_umap.py:190-191, 210-212; ex_3/05_comparison.py:278; ex_4/02_isolation_forest.py:180; ex_4/03_local_outlier_factor.py:266; locals repeat ("from Grab risk disclosures", "from Carousell public disclosures", "from a Lazada 2024 study…")
**Evidence**: "Lazada 2024 disclosure"; "published Grab-scale studies"; "Changi Q4 2024 retail experiment report (internal, cited in CAG's 2025 annual report)"; "An MAS 2024 financial stability review noted … raised the positive predictive value of STR review from ~11% to ~23%". Verifiably wrong: "Singapore police reported S$651M in 2024 scam losses" (S$651.8M is the 2023 figure; 2024 was ~S$1.1B); "banks submit STRs to the MAS COSMIC platform" (STRs go to STRO, Singapore Police Force; COSMIC is MAS's bank-to-bank sharing platform).
**Problem**: Made-up figures presented as statements by real regulators/companies; reputational risk. (Company names as scenario subjects are not an independence.md violation.)
**Fix**: Label figures as illustrative assumptions; remove "disclosed/published/report" attributions; correct scam-loss year and STR route.

### [MINOR] Dendrogram colour threshold does not match the stated cut at K
**File**: solutions/ex_1/02_hierarchical.py:200, 202
**Evidence**: `color_threshold=0.7 * Z[-CUT_K, 2]`, title "cut at K=5"; a threshold below Z[-K,2] colours ≥ K+1 groups.
**Fix**: `(Z[-CUT_K,2] + Z[-(CUT_K-1),2]) / 2`.

### [MINOR] ex_1.5 DBSCAN eps uses the 2nd-derivative argmax that ex_1.3 says overshoots
**File**: solutions/ex_1/05_evaluation_profiling.py:149-150 vs 03_density_based.py:104-108
**Evidence**: 05 `eps_suggested = float(k_dist[int(np.argmax(diffs2)) + 2])`; 03 "argmax of the 2nd derivative latches onto the steepest tail jump … and over-shoots".
**Fix**: Reuse 03's Kneedle chord-distance elbow.

### [MINOR] k-means++ checkpoint/printout asserts something not guaranteed
**File**: solutions/ex_1/01_kmeans.py:158, 164-166
**Evidence**: `assert km_plus.inertia_ <= km_random.inertia_ + 1` and unconditional "k-means++ … faster and lower inertia"; with n_init=10 random init can win by > 1.
**Fix**: Conditional print; drop the assert or compare single-init runs averaged over seeds.

### [MINOR] False claims that ClusteringEngine supports hierarchical clustering
**File**: solutions/ex_1/01_kmeans.py:299-300; 05_evaluation_profiling.py:405-406, 426-427
**Evidence**: "sweep_k runs … for any supported algorithm (kmeans, hierarchical, dbscan, spectral, gmm)"; installed `_SUPPORTED_ALGORITHMS = {"kmeans","dbscan","gmm","spectral"}`, and `sweep_k` rejects dbscan.
**Fix**: List kmeans/gmm/spectral for `sweep_k`; kmeans/dbscan/gmm/spectral for `fit`.

### [MINOR] Stale "kailash-ml 1.5.1" references
**File**: solutions/ex_1/02_hierarchical.py:302, 312; 03_density_based.py:306, 311, 332; 04_spectral.py:255; ex_2 files (l.255/382/249); ex_3/02_kernel_pca.py:247, 262; ex_4 files (l.235/322/328); shared/mlfp04/ex_1.py:170, ex_3.py:123, ex_4.py:277
**Evidence**: installed `kailash_ml.__version__` = 2.2.2.
**Fix**: Drop or update version numbers.

### [MINOR] 04_spectral: "top-k eigenvectors" wording and an overclaiming reflection
**File**: solutions/ex_1/04_spectral.py:10, 195, 288-289
**Evidence**: Header "top-k eigenvectors of the graph Laplacian" (theory l.63 correctly says smallest); "corridor identified by the top eigenvector"; reflection "[x] Build an RBF affinity matrix and the graph Laplacian" but students only call `SpectralClustering`.
**Fix**: "smallest" / "Fiedler vector"; add a hand-built affinity→Laplacian→eigenvector step or reword.

### [MINOR] Wrong GMM complexity in selection table
**File**: solutions/ex_1/05_evaluation_profiling.py:341
**Evidence**: "GMM … O(nK^2d)".
**Fix**: Full-covariance EM is O(n·K·d²) per iteration + O(K·d³) for inversions.

### [MINOR] Covariance-type ordering and "spherical = soft K-means"
**File**: solutions/ex_2/03_covariance_types.py:60-61, 63, 282, 285; local "(equivalent to K-means)"
**Evidence**: "full > tied > diag > spherical"; with shared `count_gmm_params`, d=8, K=5: tied = 80, diag = 84 (not nested). "spherical … Mathematically equivalent to soft K-means."
**Fix**: full ⊃ {tied, diag} ⊃ spherical; soft K-means is spherical with shared equal variance and equal weights; hard K-means the σ→0 limit.

### [MINOR] Wrong Mixtral figures; inconsistent run counts
**File**: solutions/ex_2/04_mixture_of_experts.py:69-71, 74-75, 318, 337; local theory block
**Evidence**: "Mixtral 8x7B has 8 experts of 7B params each … ~14B active params, not 56B"; "gating network … is a tiny MLP". Mixtral: 46.7B total, ~12.9B active (attention shared), router is a single linear layer. l.318 "nine clustering runs" vs l.337 "eight runs".
**Fix**: Correct figures/router description; consistent count.

### [MINOR] Contradictory "only method" claims about inverse / out-of-sample transform
**File**: solutions/ex_3/01_pca.py:331-334; 02_kernel_pca.py:149, 183; 04_umap.py:339-341; 05_comparison.py:314; local 02
**Evidence**: "Kernel PCA has no inverse_transform" — installed `KernelPCA.inverse_transform` exists with `fit_inverse_transform=True`; `umap.UMAP.inverse_transform` exists; 04 "UMAP is the only method in this exercise that can [embed new data]" but `PCA.transform`/`KernelPCA.transform` do too.
**Fix**: PCA exact linear reconstruction; KernelPCA/UMAP approximate inverses; PCA, KernelPCA, UMAP all transform out-of-sample; t-SNE does not.

### [MINOR] UMAP "out-of-sample" slice contains all fit rows; config count wrong
**File**: solutions/ex_3/04_umap.py:20, 93-95, 334 (same 05_comparison.py:98-103)
**Evidence**: `fit_idx = subsample_indices(n, 3000)` and `transform_idx = subsample_indices(n, 10000)` both seed 42 → 3000/3000 fit rows inside the transform slice; header/reflection say "6 hyperparameter configurations", `umap_configs` has 4.
**Fix**: Transform rows from `np.setdiff1d(all, fit_idx)`; "4 configurations".

### [MINOR] "Beats random" checkpoint uses 0.4
**File**: solutions/ex_4/01_statistical_methods.py:149-150 (local same)
**Evidence**: `assert z_metrics["auc_roc"] > 0.4, "Z-score AUC should beat random floor"`.
**Fix**: `> 0.5`.

### [MINOR] Inverted sklearn convention in LOF comment
**File**: solutions/ex_4/03_local_outlier_factor.py:122-123
**Evidence**: "negative_outlier_factor_ is negated so that 'more negative = more normal'" — in sklearn more negative = more anomalous (code correct, comment wrong).
**Fix**: "more negative = more anomalous".

### [MINOR] Wrong dataset size/dimensionality in comments
**File**: shared/mlfp04/ex_1.py:57, ex_2.py:88 ("~6K rows"); solutions/ex_1/01_kmeans.py:219 ("6K customers"); 02_hierarchical.py:92 ("6-dim"); ex_4/02_isolation_forest.py:271 + local ("40+ feature tabular data")
**Evidence**: `ecommerce_customers.parquet` is (50000, 16); 8 numeric clustering features (7 in ex_4).
**Fix**: Correct numbers.

### [MINOR] Local scaffold drops a checkpoint assert
**File**: local/ex_4/02_isolation_forest.py (~l.120-124) vs solution l.142
**Evidence**: solution `assert iso_scores.std() > 0, "Scores should vary across rows"` missing in local (4 vs 3 asserts); exercise-standards.md: checkpoints never stripped.
**Fix**: Restore the assert.

### [MINOR] NaN silhouette can be selected as "best" and crash the tracker
**File**: solutions/ex_1/03_density_based.py:274-276, 290
**Evidence**: `max(dbscan_results.items(), key=lambda x: x[1]["sil"])` — if the first entry is NaN, all later comparisons are False so NaN wins; `log_metric` raises MetricValueError on non-finite.
**Fix**: Filter NaN before max; wrap logged values in `_finite`.

Other checks (no finding): no hardcoded LLM model names, no pandas, no institutional partners/course codes; no undefined names in local scaffolds beyond `____` blanks (AST scan); WHAT YOU'LL LEARN and REFLECTION blocks present in every file.

## Part D — Exercises ex_5 to ex_8 (solutions, local scaffolds, shared/mlfp04/ex_5..8.py)

### [BLOCKING] local ex_5/02 crashes with NameError — speed sweep stripped but its variables still used
**File**: modules/mlfp04/local/ex_5/02_fp_growth.py:233, 247, 256-257
**Evidence**: ruff F821: undefined `apriori_times`, `fp_times`, `sizes`; l.233 `speedup = apriori_times[-1] / fp_times[-1] ...`. Defined in solutions/ex_5/02_fp_growth.py:264-278; that block is absent from local.
**Problem**: A student who fills every blank still hits NameError at the TRACK step; the R9A plots (itemset-frequency bar, speed comparison) are also gone.
**Fix**: Restore the sweep + plots in local (blanks allowed) or remove the references.

### [BLOCKING] Ex 5.4 target is a deterministic function of the baseline features — rule features cannot help
**File**: modules/mlfp04/solutions/ex_5/04_rule_features.py:225-234, 432-438 (same in local)
**Evidence**: `y = (basket_sizes >= 6)` with `X_baseline = onehot`; row-sum of the one-hot equals basket size (`np.all(X.sum(1)==len)` True); rerun baseline: LR AUC = 1.0, RF AUC = 0.969; interpretation "rule features typically add 2-5 points of AUC".
**Problem**: Label is a linear threshold on the sum of baseline features; spec 4.5 criterion "Rules used as features improve supervised model" can never be shown; printed interpretation and the "3-5% improvement" story are false for this setup.
**Fix**: Use a target not derived from the features (held-out future category purchase, or price-weighted spend label where bundles change price).

### [BLOCKING] Ex 7 recommender evaluation is degenerate — all methods tie random ranking; ALS is far worse than the global mean
**File**: shared/mlfp04/ex_7.py:41-51, 156-186; solutions/ex_7/04_matrix_factorisation.py:246-250, 328-330; solutions/ex_7/05_hybrid_evaluation.py:326-328, 414-416 (same claims in local)
**Evidence**: Rerun of 05's functions on the shared data (holdout):

| Method | RMSE | MAP |
|---|---|---|
| Content-based | 1.428 | 0.657 |
| User-CF | 0.549 | 0.672 |
| Item-CF | 0.500 | 0.741 |
| ALS | 0.956 (train 0.043) | 0.724 |
| Hybrid | 0.620 | 0.693 |
| Global mean | 0.499 | 0.692 |
| Random scores | – | 0.721 (P@5 0.522, same as every method) |

~3 holdout items per user (50×0.30×0.20) so P@5 ignores ranking; user subspace "recovery" 0.26 vs 0.22 for a random 5-dim subspace (and it compares the arbitrary first 5 of 10 learned columns). Reflection prints "[x] Recovered the true latent subspace"; KEY INSIGHT "Hybrids lift ranking quality" — measured lift is negative.
**Problem**: Students are told ALS discovers latent structure and hybrids win while numbers show chance-level or worse-than-mean performance; ALS (k=10, λ=0.1, ~12 ratings/user) massively overfits.
**Fix**: Larger/denser data (e.g. 2000×300, more holdout per user); add global/item-mean baselines; k = N_LATENT_TRUE with larger λ and bias terms; measure subspace recovery with all learned columns; tune blend weights on a validation split; make printed claims conditional on measured numbers.

### [BLOCKING] Ex 6.5 "Word2Vec" features are random hash vectors, labels are circular, test set fully leaked
**File**: modules/mlfp04/solutions/ex_6/05_sentiment_word2vec.py:101-138, 194-200, 226-235; shared/mlfp04/ex_6.py:235-243 (local same)
**Evidence**: `pseudo_word2vec` draws a random vector seeded by `abs(hash(token))` (str hashes are salted per process, so "deterministic" is also false). `labels = (lex_scores > 0)` and baseline prediction `(lex_scores > 0)` → lexicon accuracy 1.0 by identity. Corpus 500 rows / 62 unique `content`; 100/100 test docs have their content in train. APPLY text: classifier "learns 'not good' drifts toward the negative region", embeddings "pretrained on enormous corpora"; shared scenario says Word2Vec transfers "via shared subword tokens" (that is FastText).
**Problem**: Students believe they built Word2Vec sentiment features and measured a real test accuracy — neither is true; no real word embeddings are used anywhere in Ex 6 although spec 4.6 requires them.
**Fix**: Real pretrained vectors (or train Word2Vec on a real corpus), a human-labelled review dataset, dedupe and split by unique content, fix the subword claim.

### [MAJOR] Ex 6 corpus is 62 duplicated paragraphs, mostly ML textbook text — not Singapore news or reviews
**File**: shared/mlfp04/ex_6.py:58-71 (feeds 6.2-6.5)
**Evidence**: `data/mlfp03/documents.parquet`: 500 rows, 62 unique content, 342 unique titles; source `textbook` 312; largest category `ml_fundamentals` 75; docs ~290 chars.
**Problem**: Spec 4.6 calls for Singapore news + customer reviews; duplicates distort topic sizes, LDA perplexity, NPMI and every split; scenarios (SPH newsroom, MAS complaints) don't match the data.
**Fix**: Provide a real deduplicated news + review corpus via MLFPDataLoader; at minimum `unique(subset="content")` in `load_corpus`.

### [MAJOR] Ex 6.3 fits LDA on TF-IDF and calls training perplexity "held-out"
**File**: solutions/ex_6/03_lda_topics.py:81-87, 98, 112
**Evidence**: `X = TfidfVectorizer(...).fit_transform(documents)` → `lda.fit(X)`; `perp = lda.perplexity(X)`; prints "Lower perplexity = better held-out fit".
**Fix**: CountVectorizer for LDA; perplexity on a held-out split; fix wording. (Same misuse in textbook.md — Part A.)

### [MAJOR] Spec 4.6 gaps: TF-IDF not implemented from scratch; UMass coherence missing; wrong printed count
**File**: solutions/ex_6/01_tfidf_bm25.py:46-48, 92-95, 108
**Evidence**: Theory `TF = count/length`, `IDF = log(N/df)` but code uses sklearn `TfidfVectorizer` (raw counts, `ln((1+N)/(1+df))+1`, L2) so printed IDFs don't match the stated formula; no "umass" anywhere in ex_6; l.108 prints "'singapore' appears in 4/8 docs" but TOY_CORPUS docs 0,1,4,6,7 contain it (5/8).
**Fix**: numpy TF-IDF from counts asserted against the stated formula; add UMass; "5/8".

### [MAJOR] Spec 4.8 gaps: no from-scratch network, no HDB regression, thin loss/optimiser coverage
**File**: solutions/ex_8/01-05; shared/mlfp04/ex_8.py
**Evidence**: No manual forward pass/backprop/chain-rule code — every model is `nn.Linear`/`nn.Sequential` with torch optimisers; data is XOR or random images; no HDB data, no regression output; no MSE/MAE/focal/contrastive/triplet/KL losses, no RMSProp, no StepLR/OneCycle/ReduceLROnPlateau.
**Problem**: Spec core outcome ("Build a neural network from scratch … for HDB price prediction", linear regression as NN, regression vs classification output, loss taxonomy) not exercised. (Deck/textbook promise this exercise — Parts A/B.)
**Fix**: Add a numpy 3-layer HDB regressor with manual backprop; compare dropout/BN/Adam/LR schedule on it; add a loss-taxonomy section.

### [MAJOR] Ex 8.3-8.5 train on random-noise images with random labels — comparisons are meaningless
**File**: shared/mlfp04/ex_8.py:87-100; solutions/ex_8/03_cnn_residual.py:124-127; 04_optimisers_schedulers.py:259-260; 05_regularisation_training.py:125-148, 309-312
**Evidence**: `X = rng.standard_normal(...)`, `y = (rng.random((n, 5)) > 0.85)` drawn independently.
**Problem**: Labels carry no signal; interpretations ("Adam converges within the first two epochs", dropout/BN effect, early stopping, "val tracks train") present noise as signal; spec criterion "each training technique improves convergence (demonstrated with plots)" cannot be met.
**Fix**: Real or structured data (HDB, or synthetic images whose labels depend on drawn shapes), or state that only mechanics are shown.

### [MAJOR] R9A: several files lack behaviour visualisation; local scaffolds strip required phases
**File**: solutions/ex_5/01 (CSV only, l.198-215), ex_5/03 (CSV only, l.279-282); ex_8/01, 02, 03 (loss curves only); local/ex_5/02 and 04 (plots stripped); local/ex_6/01-05 (THEORY stripped)
**Evidence**: `grep write_html|fig` = 0 for solutions ex_5/01, 03; ex_8/01-03 only `viz.training_history`; local ex_5/02, 04 = 0; `grep -c THEORY` = 0 for local/ex_6/01-05 vs 1 in each solution.
**Fix**: Itemset lattice / rule scatter (ex_5), XOR decision-boundary contour (ex_8/01), feature maps (ex_8/03); restore stripped phases in local.

### [MAJOR] Ex 8.5 says OnnxBridge cannot export torch models — installed kailash-ml can
**File**: solutions/ex_8/05_regularisation_training.py:337-352, 402-404 (local same comment)
**Evidence**: "OnnxBridge's native export is scoped to sklearn / lightgbm, so for torch.nn.Module we call torch.onnx.export directly"; installed `kailash_ml/bridge/onnx_bridge.py:213-239` dispatches `framework in ("torch","lightning","catboost")`, `_COMPAT_MATRIX` has "torch"; reflection claims "Exported … via kailash-ml OnnxBridge".
**Fix**: `OnnxBridge().export(model, framework="torch", sample_input=dummy_input, output_path=onnx_output)`; drop the false comment.

### [MAJOR] Invented statistics attributed to named real organisations, some labelled "real production"
**File/Evidence**: ex_5/01:243-248 (local "Internal A/B tests at tier-1 SG grocers show"); ex_5/02:319, 333-338 (GrabFood, "A 2024 internal experiment"); ex_5/03:311 ("Watsons SG's reported online GMV"); shared/mlfp04/ex_6.py:200-243 (ST Engineering, SPH, MAS "18 days to 11", Grab, DBS); ex_6/01:246-251 ("~14% more accurate … on the Lemur/TREC test collection", "S$450K (industry benchmark)"); ex_6/03:218-219 ("(MAS 2024 operational review)"); ex_6/04:235-237 ("Grab's internal benchmark", "(Grab CX 2025)"); ex_7/04:295-296 ("Spotify's 2019 paper reported … 30% lift"); ex_8/01:144-147, 170 ("93% recall on a 2024 DBS sample"); ex_8/02:168-170 (Grab "2023 re-architecture"); ex_8/03:132-148, 168 (NUH "trialled a 24-layer plain CNN in 2022", "AUC 0.93", "SingHealth HPC cluster"; reflection "NUH's real production AUC lift" — NUH is in NUHS, not SingHealth); ex_8/04:268-283, 303 (Sea/Shopee Pay "real production rollout"); ex_8/05:317-323, 404 (DSO "real edge-inference rollout").
**Problem**: Fabricated metrics/citations presented as fact about real companies, a regulator, a public hospital and a defence lab.
**Fix**: Reword as clearly hypothetical; remove fake citations and "real production" wording.

### [MINOR] Local ex_6 drops checkpoint asserts
**File**: local/ex_6/01_tfidf_bm25.py ~l.153; local/ex_6/02_nmf_topics.py:80-83
**Evidence**: solution `assert saturation_scores[-1] < 10 * saturation_scores[0]` (ex_6/01:222-224) and `assert recon_error < 1.0` (ex_6/02:122) absent from local.
**Fix**: Restore both.

### [MINOR] XOR theory has the classes swapped
**File**: solutions/ex_8/01_xor_proof.py:51-52 (and local)
**Evidence**: "two positive points at (+, +) and (-, -)"; code `(X0>0) ^ (X1>0)` → label at (+,+) is 0.
**Fix**: Positives at (+,−) and (−,+).

### [MINOR] Ex 8.2 checkpoint tests the opposite of its message; two theory slips
**File**: solutions/ex_8/02_activations_init.py:47-49, 104-106, 140-142
**Evidence**: `Kaiming[-1] < Zeros[-1] + 0.1` lets Kaiming be up to 0.1 worse while the message says "Kaiming must beat zero init"; "roughly half the neurons are dead at epoch 0" (ReLU zeros ~half the inputs per neuron, not dead neurons); Tanh trained with Kaiming(relu) init, contradicting the file's pairing rule.
**Fix**: `< Zeros[-1] - 0.1`; reword; pair Tanh with Xavier.

### [MINOR] Recommender theory misstatements
**File/Evidence**: ex_7/01:46-47 "low-rated items push it away" (weights are raw 1-5 ratings, all positive); ex_7/02:51-53 says without centring generous and tough raters "look dissimilar" (raw cosine makes them look similar); ex_7/03:204, 209-211 12M items vs 1.8M users then "item-CF scales with the smaller dimension"; ex_7/03:248 "Netflix … converged on item-CF"; ex_7/04:67-69 "RMSE is monotone non-increasing by construction" (only the regularised objective is); ex_7/05:222 blend weights fit on holdout MAP (test leakage).
**Fix**: Correct each; weight the blend on a validation split.

### [MINOR] NMF and NPMI misstatements; NPMI implementation inflates coherence
**File/Evidence**: ex_6/02:60-62 "just convex optimisation" (NMF is non-convex); ex_6/02:133 "NPMI > 0.1 = topics cohere above chance" (chance is 0); ex_6/03:205-206 "NMF would hard-assign" contradicts l.68-69; shared ex_6.py:104-129 `compute_npmi` tokenises with `.split()` (keeps punctuation) while topic words are vectoriser tokens, and skips never-co-occurring pairs instead of scoring −1.
**Fix**: Correct wording; use the vectoriser analyser; score unseen pairs −1.

### [MINOR] Wrong numbers and a non-existent install extra
**File/Evidence**: ex_6/04:240 35K×0.85×S$3.20 = S$95,200, minus S$180 = S$95,020 (not "S$94,820"); ex_8/04:270-281 1-in-10 aborts × S$1,200 ≈ S$3.6K/month, not "~S$18,000/month"; ex_6/04:49, 130-134 message says `uv sync --extra topic-models` but no such extra exists (`bertopic>=0.16` is a core dependency, pyproject.toml:59).
**Fix**: Correct arithmetic and message.

### [MINOR] "Next" pointers send students to the wrong place
**File/Evidence**: ex_6/05:75, 270 and ex_7/05:420 (local too) point to "MLFP05" for neural networks — next is Ex 8 (spec 4.6 says M4.8); ex_6/02:187 calls BERTopic the "next exercise" (it is 6.4).
**Fix**: Point to Exercise 8 / 04_bertopic.

### [MINOR] Ex 5.4 printed "destination contract" signatures don't match the installed API
**File**: solutions/ex_5/04_rule_features.py:546-547
**Evidence**: prints `FeatureEngineer().generate(df)` and `TrainingPipeline().train(schema, model, eval)`; installed `generate(self, data, schema, *, strategies=None)`, `TrainingPipeline.__init__(self, feature_store, registry)`, `train(self, data, schema, model_spec, eval_spec, experiment_name, ...)`.
**Fix**: Show correct signatures.

### [MINOR] Small consistency slips in ex_5
**File/Evidence**: ex_5/02:64 FP-tree "two passes" vs l.313, 329, 426 "single-pass"; ex_5/01:308 mlxtend "returns a polars-friendly table" (returns pandas); `generate_transactions(n=2500)` yields 2,374 baskets (empties dropped) while text says "2,500 baskets".
**Fix**: Align wording.

### [MINOR] Hardcoded embedding model name
**File**: solutions/ex_6/04_bertopic.py:102 (and local hint)
**Evidence**: `embedding_model="all-MiniLM-L6-v2"` hardcoded; English-only while the text sells it as multilingual. (Not an LLM model; noted under check 8 for env-model hygiene.)
**Fix**: Read from env (e.g. `TOPIC_EMBED_MODEL` in .env.example); name a multilingual model if making the multilingual claim.

Other checks (no finding): no hardcoded LLM model names; no partner/university/course-code references.

---

## Coverage summary

What was checked (all 8 CHECK items):
- **Specs/rules read**: CLAUDE.md, specs/_index.md, specs/module-4.md, specs/redlines.md (R5–R10), specs/exercise-mapping.md (M4), .claude/rules/exercise-standards.md, independence.md, domain-integrity.md.
- **Teaching material (21 files)**: deck.html (2,225 lines), speaker-notes.md (1,203), textbook.md (2,663 — read in full), README.md, index.html, lessons/01–08/{slides,textbook,notes}.html (24 files). All code blocks extracted and checked against installed signatures (AutoMLEngine/AutoMLConfig, ClusteringEngine, DimReductionEngine, AnomalyDetectionEngine, EnsembleEngine, ModelVisualizer, OnnxBridge, TrainingPipeline, DriftMonitor, FeatureEngineer, to_sklearn_input, polars DataFrame.pivot); dataset paths checked against data/ and the repo; relative links in lesson pages checked (none broken); drill answers spot-run (DBSCAN Drill 3).
- **Exercises (83 files)**: 39 solution technique files + 39 local scaffolds across ex_1–ex_8 (incl. __init__), 8 shared/mlfp04 helpers; solution-vs-local diff, AST undefined-name scan, checkpoint/assert parity, header/WHAT YOU'LL LEARN/REFLECTION presence, API introspection, small reruns of suspect computations (intrinsic dimension, ex_5.4 baseline AUC, ex_7 metrics, label construction).
- **Assessment (17 files)**: README + 4×{problem.md, starter.py, solution.py, grader.py}; all four graders run against their solutions (each passed 10/10, ~25 s); build-student-repo.sh exclusion rules checked.
- **Independence/naming**: no institutional partners, universities, funding bodies or prior course codes found; no Kailash-vs-commercial product comparisons. Real company/regulator names appear as scenario subjects (acceptable) but many carry invented "disclosed"/"published" statistics (reported as MAJOR in Parts C/D, MINOR in Part B).
- **Hardcoded model names**: no LLM model names in M4 code; one hardcoded sentence-embedding model (`all-MiniLM-L6-v2`, ex_6/04 and lessons/06 slides) noted as MINOR.

Total files examined: ~125 module files plus 8 spec/rule files and the installed kailash-ml / polars sources used for verification.
