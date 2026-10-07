# MLFP03 Audit — Supervised ML for Building and Deploying Models

Audit date: 2026-10-02/03. Read-only. Stack verified in `.venv`: kailash 2.44.1, kailash-ml 2.2.2, kailash-dataflow 2.12.0, kailash-nexus 2.11.0, polars 1.41.2, scikit-learn 1.9.0, xgboost 3.3.0, lightgbm 4.6.0, catboost 1.2.10, shap 0.52.0, lime 0.2.0.1.
Excluded per brief: Colab notebooks, redline-check "0 engines", code-fit font sizing, solution runtime pass/fail.

**Totals: 21 BLOCKING · 40 MAJOR · 57 MINOR**

Sections: A. Cross-cutting data leakage · B. Deck / speaker notes / lesson slides+notes · C. Textbook / lesson textbook pages / README / index · D. Exercises ex_1–ex_4 · E. Exercises ex_5–ex_8 · F. Assessment · G. Coverage summary

---

## A. Cross-cutting data leakage (exercises)

### [BLOCKING] Planted leak column `future_default_indicator` (and `customer_id`) used as features in every credit-based exercise (ex_2, ex_4, ex_5, ex_6, ex_7, ex_8)
- **File:** `shared/mlfp03/ex_2.py:51-98`, `shared/mlfp03/ex_4.py:59-104` (`prepare_credit_split`), `shared/mlfp03/ex_5.py:100-111`, `ex_6.py:53,91`, `ex_7.py:49,116,170`, `ex_8.py:66-86` (drops only ID columns at :73-75).
- **Evidence:** `scripts/generate_datasets.py:923-929`: "TEMPORAL LEAKAGE TRAP: future_default_indicator is perfectly correlated with default … Students must catch this in EDA". Verified in `.venv`: `default` mean is 0.934 when indicator=1 (n=13,665) vs 0.0016 when 0 (n=86,335). Every loader builds `[c for c in cols if c != "default"]`; `prepare_credit_split()["feature_names"]` has 35 features incl. `customer_id` and `future_default_indicator`. `grep -r future_default modules shared` (non-Colab) → no hits. Fork measurements: LightGBM AUC 0.993 / AP 0.971 with leak vs AUC 0.777 / AP 0.369 without; XGBoost gives the column 95.7% of importance; in ex_6 mean|SHAP| 3.67 for the leak vs 0.71 next.
- **Problem:** Every ex_4 boosting comparison, ex_5 imbalance/calibration result, ex_6 SHAP/LIME/permutation explanation, ex_7 tuning, and ex_8 production pipeline is driven by a leak — in the module that teaches leakage detection (3.1). The imbalance lesson effectively has no imbalance problem left. In ex_2 the feature set also includes `default` itself when regressing `credit_utilization`.
- **Fix:** Drop `customer_id`, `application_id`, `future_default_indicator` (and `default` in ex_2) in every shared loader (e.g. `exclude_columns=` in `PreprocessingPipeline.setup`), ideally via an explicit EDA step that discovers the leak. Then re-derive all thresholds and gates that only pass because of it (e.g. `auc_pr > 0.5` in `solutions/ex_7/05_orchestrated_pipeline.py:285,366` would fail at ~0.37) and all hard-coded interpretation text.

### [BLOCKING] ex_3 model-zoo churn label is a deterministic function of one feature
- **File:** `shared/mlfp03/ex_3.py:57-75` (`DROP_COLS = ["customer_id","review_text","product_categories"]`, comment says these "leak the target").
- **Evidence:** Verified: `churned=0` ⇔ `days_since_last_order` ∈ [1,180]; `churned=1` ⇔ [181,729]. Fork measured single-feature AUC 1.000; held-out accuracy depth-1 tree 1.000, RF 1.000, SVC 0.977, GaussianNB 0.971.
- **Problem:** The entire Lesson 3.3 comparison ("model selection justified with data evidence") runs on a leaked label; any tree trivially wins; SVM/KNN/NB tuning sweeps teach nothing. (The assessment explicitly avoids this column for the same reason — README: "the native `churned` column is a near-deterministic function of recency".)
- **Fix:** Add `days_since_last_order` to `DROP_COLS` or use the assessment's derived `premium_response` target; re-verify checkpoints and printed claims.

---

## B. Deck, speaker notes, lesson slides and notes

### [BLOCKING] Every Kailash code snippet in the master deck uses a non-existent API
- **File:** `modules/mlfp03/deck.html:443-448, 453-456, 692-701, 967-976, 1196-1208, 1707-1719, 1734-1744, 1759-1775, 1792-1804, 1910-1926, 1973-1982, 2062-2078`; callout 1859.
- **Evidence (each exec'd in .venv):** `FeatureEngineer().fit` → AttributeError (real: `generate`, `select`); `FeatureStore()` → TypeError missing `dataflow`; `TrainingPipeline(model=...)` → TypeError (real `(feature_store, registry)` + `.train(data, schema, model_spec, eval_spec, experiment_name)`); `AutoMLEngine(task=...)` → TypeError; `from kailash import Runtime` / `register_node` → ImportError; `from kailash_ml import SearchSpace` / `DriftSpec` → ImportError (live in `kailash_ml.engines.hyperparameter_search` / `.drift_monitor`); `ModelRegistry()` → TypeError missing `conn`, no `.register` (real `register_model`, `promote_model`); `from kailash_dataflow import ...` → ModuleNotFoundError (package is `dataflow`, no `field`); `ConnectionManager` has no `__aenter__`/`create_tables`; `DriftMonitor(reference_data=...)` → TypeError (real `(conn, *, tenant_id, ...)` + `set_reference_data`/`check_drift`); `WorkflowBuilder.connect` signature is `(from_node, to_node, mapping)`; slide 67 claims `TrainingPipeline(cv_strategy="nested")` which doesn't exist.
- **Problem:** Students copy-paste "Kailash Bridge" snippets; every one fails at import/construction, and they contradict the module's own working solutions (e.g. ex_7/01 uses `from kailash.runtime import LocalRuntime`, `add_node("PythonCodeNode", ...)`).
- **Fix:** Rewrite all snippets from working patterns in `solutions/ex_7/02-04`, `shared/mlfp03/ex_1.py:410`, and assessment `task_2..4/solution.py`; remove `cv_strategy="nested"`.

### [BLOCKING] Lesson slides 01, 07, 08 show Kailash code that fails
- **File:** `lessons/01/slides.html:276-282`; `lessons/07/slides.html:169-187, 213-229, 354-367`; `lessons/08/slides.html:154-184, 226-237`.
- **Evidence:** `FeatureEngineer().aggregate` / `.then_derive` → AttributeError; `from kailash import WorkflowBuilder, PythonCodeNode` fails (no top-level `PythonCodeNode`); `ConditionalNode`, `Runtime` → ImportError (conditional node is `kailash.nodes.logic.SwitchNode`); `SearchConfig(method=, metric=, cv_folds=)` — real fields `strategy`, `metric_to_optimize`, no `cv_folds`; `HyperparameterSearch(model_class=)` (real `(pipeline)`, async `.search`); `MetricSpec(auc=)`, `ModelSignature(output_schema=)` → TypeError; `registry.promote` absent; `from kailash_dataflow` → ModuleNotFoundError; `DriftSpec(features=, ks_alpha=, check_interval=)` (real: `feature_columns, psi_threshold, ks_threshold, on_drift_detected`), `monitor.check` absent.
- **Problem:** Broken code in student-facing lesson pages.
- **Fix:** Replace with the API used in solutions ex_1/05, ex_7/02-04 and assessment task_4.

### [BLOCKING] Lesson 3.6 slides: SHAP waterfall and LIME snippets both raise
- **File:** `lessons/06/slides.html:92-99, 106-116`.
- **Evidence:** `shap.plots.waterfall(shap_values[0])` on `explainer.shap_values(...)` output → `TypeError: The waterfall plot requires an Explanation object` (run with shap 0.52). `explain_instance(X_test[0].to_numpy(), ...)` — polars `X_test[0]` is a 1×d DataFrame → `ValueError: could not broadcast input array from shape (3,) into shape (1,)`.
- **Fix:** `sv = explainer(X_test); shap.plots.waterfall(sv[0])` (binary RF: `sv[0, :, 1]`); LIME: pass `X_test.to_numpy()[0]`.

### [BLOCKING] Lesson 3.6 states the fairness impossibility theorem wrongly ("Pick at most two")
- **File:** `lessons/06/slides.html:161-169` (SVG text at 168: "Pick at most two (Chouldechova 2017, Kleinberg et al. 2016)").
- **Problem:** With unequal base rates the three criteria are pairwise incompatible except for trivial/perfect predictors (independence+separation forces equal base rates or an uninformative classifier; independence+sufficiency forces Y⊥G; separation+sufficiency is Kleinberg/Chouldechova). "Pick two" teaches that achievable pairs exist. Also contradicts deck slide 58 / speaker-notes:840 ("not all three").
- **Fix:** "With unequal base rates these criteria are pairwise incompatible (except for perfect prediction) — you must choose which one to prioritise."

### [MAJOR] Deck and speaker notes describe exercise datasets and engine usage the exercises do not have
- **File:** `deck.html:342, 477, 725, 985, 1213, 2365-2366`; `speaker-notes.md:184, 195, 307, 453, 581, 1235`.
- **Evidence:** Deck: "Exercise 3.1: Engineer features for HDB price prediction (geocode...)", "polynomial degree on HDB data", slide 85 lists 3.1/3.2 as "HDB resale prices". Actual: `shared/mlfp03/ex_1.py:75-79` loads ICU tables; ex_2 uses `make_sine_dataset` + `sg_credit_scoring`; `grep -li hdb solutions/*/*.py` → only ex_8/05. Notes: "students use FeatureEngineer ... save to a FeatureStore" (184), "Students use AutoMLEngine in the exercise" (453), "The exercise uses this [TrainingPipeline]" (581), "Every model in this module is trained through TrainingPipeline" (307); grep: AutoMLEngine/FeatureEngineer/FeatureStore in no solution, TrainingPipeline only in ex_7.
- **Fix:** Update deck slides 14/23/32/41/85 and notes to ICU (3.1) and synthetic sine + credit (3.2); remove/correct the "used in the exercise" engine claims.

### [MAJOR] Assessment slide describes a quiz + project structure that does not exist
- **File:** `deck.html:2329-2349`; `speaker-notes.md:1220-1224`.
- **Evidence:** "Quiz (40%) ... ML Pipeline Project (60%)" vs `assessment/README.md`: "Total: 100 marks across 4 tasks" (20/25/25/30), auto-graded, e-commerce dataset; no M3 quiz exists.
- **Fix:** Replace slide 84 and notes with the 4-task, 100-mark structure.

### [MAJOR] "AML opening case" is never introduced, its numbers conflict, and it is attributed to a real bank
- **File:** `deck.html:1249, 1254-1260, 1266`; `speaker-notes.md:440, 595, 605, 615-616`.
- **Evidence:** Headline "99.9% accuracy", body "99.99% accurate", notes "99.999% accurate"; 100 positives in 10M ⇒ all-negative accuracy is 99.999%. Attributed to "The Credit Suisse AML case from the opening" (deck 1249), "M1's opening" (notes 605), "A Singapore bank" (615). No AML case exists in M3's opening or in M1/M2 decks (grep "money laundering"/"99.9%" in mlfp01/mlfp02 deck.html → nothing).
- **Fix:** Introduce the case in M3 (or drop "remember"), use 99.999% consistently, remove the Credit Suisse attribution.

### [MAJOR] Brier score described as a pure calibration measure with "0 = perfect calibration"
- **File:** `deck.html:1443-1444`; `speaker-notes.md:721`.
- **Evidence:** "Measures calibration quality. Lower is better. 0 = perfect calibration."
- **Problem:** Brier = reliability − resolution + uncertainty; a perfectly calibrated base-rate predictor (p=0.3) scores 0.21; 0 means perfect deterministic prediction.
- **Fix:** "Lower is better; 0 = perfect prediction. Brier mixes calibration and discrimination; use the reliability diagram/ECE for calibration specifically."

### [MAJOR] Lesson 3.5 slides apply the cost threshold to a class-weighted, uncalibrated model
- **File:** `lessons/05/slides.html:268-277` (formula at 206-211); same pattern in `lessons/05/textbook.html:423-435`.
- **Evidence:** `LGBMClassifier(class_weight={0:1, 1:100})`, then `t_opt = 100/(100+10_000)` on its `predict_proba`, calibration only afterwards.
- **Problem:** t* = c_FP/(c_FP+c_FN) assumes calibrated, unweighted probabilities; the 100× weight already shifts the boundary to ≈1/101, so the cost ratio is applied twice and almost everyone is flagged; calibrating after thresholding is the wrong order.
- **Fix:** Either unweighted → calibrate → t*, or weighted model with threshold 0.5 / validation-tuned — not both; explain the equivalence.

### [MAJOR] Lesson 3.4 early stopping on the test set
- **File:** `lessons/04/slides.html:428-437`; `lessons/04/notes.html:252-256`.
- **Evidence:** `xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])` presented as "Standard way to pick n_estimators"; trap 2: "Always use eval_set".
- **Fix:** Carve `X_val` from train; `eval_set=[(X_val, y_val)]`.

### [MAJOR] Stacking and blending (spec 3.5) not taught in deck/slides
- **File:** `deck.html:2271, 2507`; `specs/module-3.md` Lesson 3.5.
- **Evidence:** grep stacking/blending/EnsembleEngine across deck, notes, lessons → only the engine map row "EnsembleEngine | 3.5 | Stacking and blending (preview for M4)"; slide 89 then lists EnsembleEngine as "New engine" in M4.
- **Fix:** Add a stacking/blending slide (with EnsembleEngine) to 3.5 in deck and lesson 05; make the M4 preview consistent.

### [MAJOR] Certificate / module structure misstated
- **File:** `lessons/08/slides.html:300`; `lessons/08/notes.html:40-43, 50, 123-126, 131`; `speaker-notes.md:71, 1280`; also `lessons/08/textbook.html:321, 329`.
- **Evidence:** "Foundation Certificate complete. The Advanced Certificate (M4-M6)", "building toward this for 24 lessons", "M5 brings LLMs and RAG; M6 is alignment and governance". CLAUDE.md: Foundation = M1-4 (32 lessons), Advanced = M5-6; MLFP05 = DL/vision, MLFP06 = LLMs/agents.
- **Fix:** "Module 3 complete; Foundation Certificate continues with M4"; correct M5/M6 descriptions.

### [MAJOR] Speaker notes make API/capability claims false for the installed SDK
- **File:** `speaker-notes.md:896, 911-914, 929, 932, 944, 1015, 1031, 1034, 1046, 1113`; `lessons/02/notes.html:280-282`; `lessons/04/notes.html:278-280`.
- **Evidence:** `db.predictions.create(...)` (real: `db.express.create("Model", {...})`); `async with ConnectionManager() as conn` (no `__aenter__`; needs url); SearchSpace tuple `('float', 0.01, 0.3, 'log')` (real `ParamDistribution(name, type, low, high)`); `ConditionalNode` doesn't exist; "Gaussian process or TPE... HyperparameterSearch supports both" (hyperparameter_search.py:543-602 uses only optuna defaults/TPE; strategies grid/random/bayesian/successive_halving); "Kailash supports all three [model card] export formats" (no model-card code in kailash_ml); "MetricSpec documents ... thresholds that trigger alerts" (fields: name, value, split, higher_is_better); "HyperparameterSearch ... runs [nested CV] automatically" (no CV field); "AutoMLEngine ... ensembles the three boosters automatically".
- **Fix:** Correct each claim to the installed SDK or delete.

### [MINOR] Boosting comparison table says LightGBM needs manual categorical encoding
- **File:** `deck.html:1176` ("Categoricals | Manual encoding | Manual encoding | Native").
- **Fix:** LightGBM "Native (`categorical_feature`)"; XGBoost "Native (`enable_categorical=True`) or encoding".

### [MINOR] Nested-CV fit count wrong
- **File:** `deck.html:1857` ("25 total fits for 5x5"); notes:976 correctly says k_outer × k_inner × trials.
- **Fix:** "5 × (5 × K + 1) fits for K candidates".

### [MINOR] Lesson 3.3 tree example: mislabelled gain and wrong diagram numbers
- **File:** `lessons/03/slides.html:280, 225, 237-238`; `lessons/03/notes.html:225`.
- **Evidence:** "IG = 0.469 - 0.290 = 0.179" is a Gini decrease while IG is defined via entropy (line 142); root shows "Gini = 0.46" with 11 vs 12 points (= 0.499); second split "age < 55?" on a node marked Gini 0.00 with both children class 1 (zero gain).
- **Fix:** Label "Gini decrease", root 0.50, redraw split 2.

### [MINOR] "SVM needs probability=True to get AUC"
- **File:** `lessons/03/notes.html:285`. sklearn's roc_auc scorer uses `decision_function`.
- **Fix:** Remove or say probability=True only for calibrated probabilities.

### [MINOR] Incorrect XGBoost history claims
- **File:** `lessons/04/notes.html:46-48`; `speaker-notes.md:516`; `deck.html:1104`.
- **Evidence:** "won XGBoost the KDD 2016 best paper" (KDD 2016 best research paper was FRAUDAR); "Most gradient boosting implementations only use the gradient" (LightGBM and CatBoost also use the Hessian).
- **Fix:** Drop award claim; "classic Friedman/sklearn GBM is first-order".

### [MINOR] Reliability-diagram over/under-confidence rule wrong
- **File:** `speaker-notes.md:720` ("Above the line means underconfident; below means overconfident").
- **Fix:** "Above = predicted probabilities too low; below = too high"; over-confidence = above in low bins and below in high bins.

### [MINOR] No Free Lunch citation wrong/inconsistent
- **File:** `deck.html:755` ("Wolpert & Macready (1997)" — optimisation NFL) vs `speaker-notes.md:340` ("Wolpert 1996").
- **Fix:** Cite Wolpert (1996) "The Lack of A Priori Distinctions Between Learning Algorithms" in both.

### [MINOR] Lesson 3.2 Bayesian notation clashes with deck; Stein remark wrong
- **File:** `lessons/02/slides.html:173`; `deck.html:643`; `lessons/02/notes.html:112-115`.
- **Evidence:** Lesson uses τ² as noise variance; deck defines τ² as prior variance (τ² = σ²/λ). Notes: ridge "works even when you have infinite data".
- **Fix:** σ² noise, τ² prior, λ = σ²/τ²; drop the "infinite data" phrase (shrinkage gain → 0 as n→∞; Stein needs d ≥ 3).

### [MINOR] Per-lesson notes and timing out of sync with slides
- **File:** `lessons/0{2..8}/notes.html` headers; `speaker-notes.md:3, 1312`.
- **Evidence:** Declared vs actual slide counts: 02 14/15, 03 15/17, 04 14/16 (notes skip two diagram slides, so "Slide 9" = on-screen 11), 05 14/15, 06 13/12, 07 13/12, 08 13/11. speaker-notes.md says module is "~180 minutes"; lesson pages sum to ~560.
- **Fix:** Regenerate per-lesson notes against current slides; reconcile timing.

### [MINOR] Speaker notes promise three exercise formats
- **File:** `speaker-notes.md:1237, 1304` ("local .py, Jupyter notebook, and Colab") vs two-format rule.
- **Fix:** Two formats.

### [MINOR] Slide-87 notes don't match slide
- **File:** `speaker-notes.md:1264-1266` vs `deck.html:2407-2419` (notes: "small data → KNN or linear ... start with XGBoost"; slide: "Random Forest or SVM", "LightGBM for speed").
- **Fix:** Align.

### [MINOR] M4 preview lists an M5 engine
- **File:** `deck.html:2507` ("New engines: EnsembleEngine, InferenceServer"); InferenceServer is MLFP05.
- **Fix:** Remove InferenceServer.

### [MINOR] Unsupported regulatory and named-bank claims in notes
- **File:** `speaker-notes.md:722, 828, 947, 1080, 1110`; `lessons/06/notes.html:57`.
- **Evidence:** "MAS FEAT explicitly requires demographic fairness testing", "MAS FEAT guidelines recommend model cards", "credit risk teams at DBS, UOB, and OCBC all use PSI as the primary drift metric", "MAS expects calibrated probabilities ... Platt scaling is the standard".
- **Problem:** FEAT is non-binding principles and mentions neither model cards nor Platt; named-bank practice claims are unsourced (independence/naming).
- **Fix:** Soften to "FEAT principles call for fairness and transparency assessment"; remove named-bank claims.

### [MINOR] Conflicting PR-AUC rule of thumb
- **File:** `speaker-notes.md:640` ("under 5%") vs `lessons/05/slides.html:176`, `lessons/05/notes.html:126` ("< 20%").
- **Fix:** Pick one.

### [MINOR] Conformal guarantee stated without its conditions (deck/slides)
- **File:** `deck.html:2133`; `lessons/08/slides.html:245-246`.
- **Evidence:** "regardless of the underlying distribution" with no exchangeability; code `np.quantile(residuals, 0.90)`.
- **Fix:** "assuming exchangeable data"; `np.quantile(r, np.ceil((n+1)*0.9)/n, method="higher")`.

### [MINOR] Master-deck XGBoost derivation incomplete
- **File:** `deck.html:1088-1114` — Ω(f_t) undefined; jumps from Taylor expansion to Gain without w* = −G/(H+λ) (lesson 04 slides 210-225 have it). Spec LO "Derive the XGBoost split gain formula".
- **Fix:** Add Ω = γT + ½λΣw² and the leaf-weight step to slide 37.

---

## C. Textbook, lesson textbook pages, README, index

### [BLOCKING] Nearly every Kailash snippet in textbook.md fails against the installed API
- **File:** `textbook.md:28, 276-292, 319-322, 449-452, 763-777, 1172-1189, 1631-1664, 2097-2103, 2743-2763, 2784-2825, 2832-2845, 2939, 2951-2966, 3080-3125, 3193-3207, 3274-3276, 3375-3395`.
- **Evidence (inspect/hasattr):** `FeatureSchema` has no `fields=`, `FeatureField` no `min`/`max`; `FeatureEngineer(feature_store=None, *, max_features=50)` has only `generate`/`select` (no `schema=`, `.transform`, `.validate`); `DataExplorer.profile` is a coroutine, `DataProfile` has no `missing_summary`; `ExperimentTracker` requires `await ExperimentTracker.create(store_url=…)`, `start_run` has no `run_name`; `PreprocessingPipeline()` has `setup`/`transform` (no `fit_transform`); `kailash_ml.CrossValidator` absent; `TrainingPipeline.fit`, `AutoMLEngine.fit` absent (`train`, `run`); `from kailash_ml import SearchSpace, ParamDistribution, SearchConfig` → ImportError; `ParamDistribution` no `log_uniform`; `SearchConfig` no `cv_folds`/`scoring`; `HyperparameterSearch.fit` absent (`search`); `ModelRegistry(path=)` invalid; `ModelSignature`/`MetricSpec(values=)` invalid; `kailash.db.models` → ModuleNotFoundError; `express.create(conn, Class, data)` invalid (real `(model: str, data: dict)`); `DriftSpec` not exported, `DriftMonitor(spec=).check` invalid. Line 28 says "kailash-ml 0.4+"; 2.2.2 installed.
- **Fix:** Rewrite every snippet from working patterns (`shared/mlfp03/ex_1.py:410`, `solutions/ex_7/02_dataflow_persistence.py:70`, `solutions/ex_7/03_hyperparameter_search.py:94-154`, assessment solutions); update version line.

### [BLOCKING] Lesson textbook pages show Kailash/helper code that fails
- **File:** `lessons/01/textbook.html:371-385`; `03/textbook.html:489`; `04/textbook.html:329`; `07/textbook.html:104-126, 169-217`; `08/textbook.html:58-76, 144-150, 250`.
- **Evidence:** `fe.aggregate(...).then_join(...).then_derive(...)`, `FeatureField(min=, max=)` absent; `to_sklearn_input(df, target=..., standardise=...)` — real `(df, feature_columns=None, target_column=None)` returning a 3-tuple; `from kailash import PythonCodeNode, ConditionalNode, Runtime` — all `hasattr(kailash, …)` False; `SearchSpace({...})`, `HyperparameterSearch(model_class=)`, `ModelRegistry()`, `MetricSpec(auc=)`, `registry.promote`; `from kailash_dataflow import DataFlow, field` → ModuleNotFoundError; top-level `await`; `DriftMonitor(spec=DriftSpec(features=, ks_alpha=, check_interval=)).check`. Lifecycle "experiment→staging→production→retired" vs registry stages staging/shadow/production/archived.
- **Fix:** Same rewrite, sourced from ex_N solutions.

### [BLOCKING] Worked examples load files and columns that do not exist
- **File:** `textbook.md:311, 330-336, 805-809, 1209-1214, 1685, 2126, 2448, 2875, 3280`; `lessons/01/textbook.html:465-481`; `lessons/02/textbook.html:503-506`; `lessons/04/textbook.html:328`.
- **Evidence:** `data/mlfp02` contains only `sg_credit_scoring.parquet` (examples load `credit_scoring.parquet`, `hdbprices.csv`). HDB parquet has `month, town, flat_type, block, street_name, …, resale_price` — no `resale_date`, `address`, `flat_age`, `distance_to_mrt_km`, `resale_price_sgd`. E-commerce file has no `age`, `total_purchases`, `sessions_per_month`, `support_tickets`. Credit has `income_sgd`, `debt_to_income`, `credit_utilization` (not `income`, `debt_ratio`, `credit_utilisation`). ICU tables have no `died`/mortality column (lesson 01 `y = features["died"]`); `vitals.timestamp` is String but lesson 01 does datetime subtraction; lesson 01 also keeps leaky `los_days` and string columns into `mutual_info_classif`.
- **Fix:** Use real files/columns; align lesson 01 to ex_1's actual target; drop `los_days` and non-numeric columns.

### [BLOCKING] Textbook credit examples train on `customer_id` and the planted leak column
- **File:** `textbook.md:1687-1689, 2128, 2453-2454, 2884, 3282-3283`.
- **Evidence:** `feature_cols = [c for c in credit.columns if c != "default"]` includes `customer_id` (String) and `future_default_indicator` (see Section A).
- **Problem:** String columns make `.to_numpy()` object arrays (XGBoost/LightGBM fail); even when fixed, the model trains on the leak Lesson 3.1 teaches students to catch; the quoted "AP 0.55–0.62" (line 1765) cannot be honestly reproduced.
- **Fix:** Explicitly drop `customer_id` and `future_default_indicator`; encode categoricals as `shared/mlfp03/ex_4.py` does; reference Lesson 3.1's leakage rule.

### [BLOCKING] "Point-in-time correct" rolling median actually looks into the future
- **File:** `textbook.md:346-366`; repeated at 538 (reflection Q3) and 3741 (Appendix A).
- **Evidence:** `group_by_dynamic("resale_date", every="1mo", period="12mo", by="block", closed="left")` is claimed to give "the window [t − 12 months, t)". Executed on dates 2020-01…2020-06 with values 1…6: the window labelled 2020-01-01 contains [1.0, …, 6.0] — it starts at t and runs forward.
- **Problem:** After the join each row gets the median of its own price plus the next 12 months — exactly the leak the lesson claims to prevent.
- **Fix:** `hdb.rolling(index_column="resale_date", period="12mo", group_by="block", closed="left")` (window [t−12mo, t)) or `offset="-12mo"`; correct all three explanations.

### [BLOCKING] `scale_pos_weight` inverted
- **File:** `textbook.md:2169`.
- **Evidence:** `scale_pos_weight=y_train.sum() / (len(y_train) - y_train.sum())` = n_pos/n_neg ≈ 0.148; line 1964 says weight = n_majority/n_minority; Appendix A line 3779 uses 11.5.
- **Problem:** Down-weights the minority class — the opposite of cost-sensitive learning.
- **Fix:** `(len(y_train) - y_train.sum()) / y_train.sum()`.

### [BLOCKING] Calibration-plot reading reversed
- **File:** `textbook.md:2085`; `lessons/05/textbook.html:308, 325-332` (SVG).
- **Evidence:** "An S-shape (low bins below the diagonal, high bins above) indicates over-confidence … typical of neural networks and boosted trees. The fix is Platt scaling, which is an inverse sigmoid." The lesson 05 SVG "over-confident" curve lies entirely below the diagonal (e.g. x=160 → y=260 vs diagonal 237).
- **Problem:** Over-confidence = low bins *above*, high bins *below* the diagonal (with predicted on x, observed on y); low-below/high-above is under-confidence (classic boosted-tree sigmoid distortion, Niculescu-Mizil & Caruana 2005). A curve entirely below is uniform over-prediction. Platt is a sigmoid fit. Contradicts line 2033. Breaks the LO "read and interpret calibration plots".
- **Fix:** Swap the description; redraw SVG as transposed S; call Platt a sigmoid fit; make boosted-tree claims consistent.

### [BLOCKING] Lesson 3.7 workflow examples do not run
- **File:** `textbook.md:2643-2654, 2704-2709, 2912-2922`.
- **Evidence:** Node types `DataLoaderNode`, `PreprocessNode`, `TrainNode`, `EvalNode`, `ConditionalNode` don't exist (verified `NodeRegistry.get("ConditionalNode")` → NodeConfigurationError; `SwitchNode` exists). Executed `add_connection("load","data","use","data")` between PythonCodeNodes → `NameError: name 'data' is not defined` (outputs are under `result`); with `"result.data"` → `{'out': 6}`.
- **Fix:** Wire `"result.<key>"`; use `SwitchNode`; delete fictitious node types.

### [MAJOR] Gradient-boosting update rule has the wrong sign
- **File:** `textbook.md:1441`; `lessons/04/textbook.html:102`.
- **Evidence:** "F_m(x) = F_{m-1}(x) - eta * h_m(x) where h_m is a decision tree fit to the negative gradients"; lesson 04 says fit f_m "to the gradient" then add. Contradicts textbook:1819 (`F_m = F_{m-1} + eta * f_m`).
- **Fix:** `F_m = F_{m-1} + η h_m`, h_m fit to −g; "negative gradient" in lesson 04.

### [MAJOR] False Singapore regulatory claims
- **File:** `textbook.md:2289, 3730`.
- **Evidence:** "MAS requires that credit decisions … be explainable under the Code of Consumer Banking Practice. The PDPA gives individuals the right to request an explanation of automated decisions"; "MAS Notice 635 requires explainable credit decisions."
- **Problem:** The Code of Consumer Banking Practice is issued by the Association of Banks in Singapore, not MAS, and does not mandate explainability; the PDPA has no right to explanation of automated decisions (that is a GDPR concept); MAS Notice 635 concerns unsecured credit limits, not explainability.
- **Fix:** Cite MAS FEAT Principles (2018)/Veritas as non-binding guidance; remove PDPA/Notice 635 claims.

### [MAJOR] Brier score of a "perfectly calibrated" model misstated
- **File:** `textbook.md:2041` ("A perfectly calibrated model has Brier score equal to p(1 − p)").
- **Problem:** p(1−p) is the uncertainty term — the Brier of the constant base-rate predictor; a calibrated model with resolution scores lower, down to 0.
- **Fix:** "The constant base-rate predictor scores p(1−p); a calibrated model with discriminating power scores lower; a perfect one scores 0."

### [MAJOR] "Baseline accuracy" prints the predicted-positive rate
- **File:** `textbook.md:2149` — `print(f"Baseline accuracy: {(p_base > 0.5).astype(int).mean():.3f}")`.
- **Fix:** `((p_base > 0.5).astype(int) == y_test).mean()`.

### [MAJOR] CV example raises and leaks across folds
- **File:** `textbook.md:772-786`.
- **Evidence:** `StratifiedKFold` on a continuous regression target with `scoring="r2"` (sklearn: "Supported target types are: ('binary', 'multiclass')"); `pipe.fit_transform(X)` on all X before `cross_val_score` — the leak line 765 says the pipeline avoids.
- **Fix:** `KFold`; fit preprocessing inside each fold.

### [MAJOR] Feature/target rows misaligned after `drop_nulls()`
- **File:** `textbook.md:387-388, 808-809, 1213-1214, 1688-1689`.
- **Evidence:** `X = df.select(cols).drop_nulls()…; y = df.select(target)…[:len(X)]`.
- **Problem:** Pairs features with wrong labels.
- **Fix:** `df.drop_nulls(subset=cols+[target])` first, then select X and y from the same frame.

### [MAJOR] Spec topics missing or thin in textbook/lesson pages
- **File:** `textbook.md`; `lessons/05-08/textbook.html` vs `specs/module-3.md` 3.5–3.8.
- **Evidence:** Stacking/blending absent (only engine table row at textbook:3972); ALE absent from textbook.md, one paragraph with no formula/example in lesson 06; async/await primer for DB operations absent although DataFlow code uses async; logic nodes for branching/merging only via fictitious `ConditionalNode` (no MergeNode/SwitchNode).
- **Fix:** Add these sections in both textbook and lesson pages.

### [MAJOR] textbook.md and lesson textbook pages teach contradictory material
- **File:** `textbook.md` vs `lessons/0N/textbook.html`.
- **Evidence:** 3.1 textbook uses HDB, lesson 01 uses ICU mortality, ex_1 uses ICU LOS; 3.2 textbook uses credit, lesson 02 uses HDB; ML pipeline 8 stages (textbook:102) vs 7 (lesson 01:122); ridge-as-MAP λ = 1/(2τ²) (textbook:721, wrong for unscaled SSE loss) vs λ = σ²/τ² (lesson 02, correct); model card 9 Mitchell sections (textbook:3231) vs 7 different sections (lesson 08); optimal threshold "0.15–0.25" (textbook:2195) vs 0.0099 (textbook:2266, lesson 05).
- **Fix:** One dataset per lesson matching the exercises; reconcile numbers; λ = σ²/τ²; delete/correct textbook:2195.

### [MINOR] README and index.html misdescribe datasets, sizes and engines
- **File:** `README.md:24-31`; `index.html:80-83, 91-100`.
- **Evidence:** README: ex_2 "HDB resale", ex_3 "E-commerce / HDB" — solutions use sine + `sg_credit_scoring` (`shared/mlfp03/ex_2.py:69`) and e-commerce only. index.html: ~6K ICU admissions / ~5K credit / ~5K e-commerce; actual 8,000 / 100,000 / 50,000 (lessons 03/04 also "~5000"). FeatureEngineer/FeatureStore/AutoMLEngine listed but used by no M3 solution.
- **Fix:** Correct the tables.

### [MINOR] Broken "Course Home" link
- **File:** `index.html:36` — `href="../index.html"` → `modules/index.html` does not exist.
- **Fix:** Point to the real course home or remove.

### [MINOR] AUC-PR baseline misdescribed
- **File:** `textbook.md:1880` ("AUC-PR was 0.09 — worse than random" at 0.4% base rate). Random AP = base rate (0.004); 0.09 is ~22× random.
- **Fix:** "far below a useful level", not "worse than random".

### [MINOR] Polynomial feature count wrong
- **File:** `textbook.md:179` ("C(50+3, 3) = 22,100"). C(53,3) = 23,426; 22,100 = C(52,3).
- **Fix:** Correct the number.

### [MINOR] ElasticNet α notation conflicts with sklearn
- **File:** `textbook.md:673-677` vs `848`. Formula uses α as mixing ("α = 1 is pure Lasso") while code uses `ElasticNet(alpha=0.01, l1_ratio=0.5)` (sklearn alpha = strength, l1_ratio = mix; loss scaled 1/(2n), L2 term ½).
- **Fix:** Add mapping: textbook λ ↔ sklearn `alpha`, textbook α ↔ `l1_ratio`.

### [MINOR] Focal-loss gradient only valid for y = 1
- **File:** `textbook.md:2013` (`dFL/dz = (1 - p)^gamma * (gamma*p*log(p) + p - y)`). For y=0 the gradient is p^γ(p − γ(1−p)log(1−p)).
- **Fix:** State for y=1 or write in p_t form.

### [MINOR] Hessian interpretation backwards
- **File:** `textbook.md:1547` ("small hessian (model is uncertain)"). For log-loss h = p(1−p) is maximal at p=0.5; small h = confident.
- **Fix:** Reverse.

### [MINOR] GOSS sampling fraction inconsistent
- **File:** `textbook.md:1585-1590`; `lessons/04/textbook.html` GOSS list. "Randomly sample b% of the remaining samples" with weight (1−a)/b; in the LightGBM paper b is a fraction of the full dataset.
- **Fix:** "Sample b × N rows from the remaining (1 − a) × N."

### [MINOR] Wrong cross-references to prior lessons
- **File:** `textbook.md:438, 439, 440, 697, 903, 904, 1297, 2979, 3509`.
- **Evidence:** Bayesian thinking cited as M2 "2.3" (is 2.1); linear regression "2.4" (is 2.5); hypothesis testing "2.2" (is 2.3); Polars dates "1.6" (Data Visualisation); "1.8 Python classes and inheritance" (M1 spec 1.7: classes as users, not authors).
- **Fix:** Correct numbers; remove the classes reference.

### [MINOR] Bayes-optimal threshold condition misstated
- **File:** `lessons/05/textbook.html:481` ("0.5 is Bayes-optimal only if the two error costs are equal and the classes are balanced"). With calibrated posteriors, equal costs suffice.
- **Fix:** Drop the class-balance condition.

### [MINOR] Impossibility theorem "two of three" framing (textbook page)
- **File:** `lessons/06/textbook.html:220` ("Any model that satisfies two of the three necessarily violates the third"). Same error as the lesson 06 slide finding.
- **Fix:** "No two can generally hold together when base rates differ, except trivial/perfect predictors."

### [MINOR] Lesson 06 textbook fairness snippets broken
- **File:** `lessons/06/textbook.html:125, 272`.
- **Evidence:** `shap.plots.waterfall(shap_values[0])` on ndarray → TypeError (run); `fpr = (pred[mask] & ~y_test[mask]).sum() / (~y_test[mask]).sum()` uses bitwise NOT on ints (~0 = −1) → wrong FPR.
- **Fix:** `explainer(X_test)[0]`; `(y_test[mask] == 0)`.

### [MINOR] Conformal quantile drops finite-sample correction (lesson page)
- **File:** `lessons/08/textbook.html:225, 261` — `np.quantile(residuals, 1 - alpha)` with "guaranteed coverage"; α called "coverage level" (it is miscoverage). `textbook.md:3220` uses the correct ⌈(n+1)(1−α)⌉/n.
- **Fix:** Use the corrected level; fix α wording.

### [MINOR] Lesson 03 text contradicts its own table
- **File:** `lessons/03/textbook.html:540` ("Naive Bayes is eight times slower") — table shows GaussianNB 0.1 s, fastest.
- **Fix:** "faster".

### [MINOR] MI vs Pearson claim wrong
- **File:** `lessons/01/textbook.html:546` (MI "is robust to the unit of measurement in a way that a correlation-based filter is not"). Pearson is invariant to affine unit changes; MI's advantage is invariance to monotone non-linear transforms and capturing non-linear dependence.
- **Fix:** Reword.

### [MINOR] Independence/naming slips in textbook
- **File:** `textbook.md:96, 3726, 3805`.
- **Evidence:** "Remember the deck 4A ordering" (prior-course deck reference in student text); "DBS Bank (hypothetically)…" and model name `dbs_credit_default`.
- **Fix:** Remove "deck 4A"; use "a Singapore retail bank" and a generic model name.

### [MINOR] Credit-card story presents an unverifiable quote as fact
- **File:** `textbook.md:2287` — attributes "The algorithm is proprietary, and none of our engineers can explain individual decisions" to the company; the 2021 regulator report found no unlawful discrimination.
- **Fix:** Paraphrase as a reported customer-service response and note the outcome.

---

## D. Exercises ex_1 – ex_4 (solutions, local scaffolds, shared helpers)

(The planted-leak finding for ex_2/ex_4 and the ex_3 churn-label finding are in Section A.)

### [BLOCKING] ex_2 regularisation demos are degenerate; printed interpretations are false
- **File:** `shared/mlfp03/ex_2.py:51-98`; `solutions/ex_2/02_ridge_regression.py:69-70, 134-136`; `solutions/ex_2/03_lasso_elasticnet.py:196-198`; `solutions/ex_2/05_learning_curves.py`.
- **Evidence:** Target `credit_utilization` correlates 0.973 with feature `avg_balance_utilization`; OLS test R² 0.953 on 80,000×35. Ridge test MSE 0.001217 at every α 0.001–100; coefficient norm 0.14041 → 0.13694 at α=1000. Lasso keeps 0 features for α ≥ 1 (4 of 7 grid points constant); at α=0.1 only `avg_balance_utilization` survives; ElasticNet keeps 1–2 features for every l1_ratio. Feature list includes `customer_id`, `default`, `future_default_indicator`.
- **Problem:** Code claims "~45 features and modest sample size, OLS will overfit and Ridge should help" (02:69-70), "At α=1000 the coefficients are all squished near zero" (02:134-136), ElasticNet "keeps correlated groups together" (03:196-198) — all contradicted by output. Students see no regularisation effect.
- **Fix:** A target not near-duplicated by a feature; fewer rows / more features (or polynomial terms); exclude ID/outcome/leak columns; separate α grid for Lasso (1/(2n) loss scaling).

### [BLOCKING] Learning-curve diagnosis taught backwards
- **File:** `solutions/ex_2/05_learning_curves.py:49-51, 60-61`; REFLECTION Step 3 at 237-238.
- **Evidence:** "CONVERGED-FAR APART — train much higher than test, both flat. Diagnosis: HIGH BIAS. More data WON'T help; you need a richer model"; "The gap between train and test curves is the VARIANCE component" (line 60 — contradicts line 50); "If the curve is flat but the gap is large, switch to a richer model class".
- **Problem:** A large train/validation gap is high variance (regularise, simplify, more data); high bias is both curves converged close together at a poor score (add capacity). The file prescribes the opposite remedy.
- **Fix:** Use the standard reading and correct REFLECTION Step 3.

### [MAJOR] ex_1 teaches point-in-time correctness on target-derived whole-stay features
- **File:** `shared/mlfp03/ex_1.py:123-169, 270-279, 368-391`; `solutions/ex_1/01_feature_engineering.py:44-52`; `solutions/ex_1/05_validation_and_tracking.py:265-275`.
- **Evidence:** Target is `los_days > median` (lines 368, 381-391); `medication_intensity`/`lab_intensity` divide by `los_days` (274-279); vitals/meds/labs aggregated over `[admit_time, discharge_time]` while theory says "If prediction time is the moment of ICU admission … discharge-day vitals do not exist yet". ex_1/05 prints a hard-coded `"[ok] vitals / medications / labs filtered to [admit_time, discharge_time]"` and only WARNs on `discharge_time`. Best single-feature AUC on the target 0.512.
- **Problem:** The leakage lesson rests on leaky features with a fake audit; selection methods rank noise.
- **Fix:** Prediction cutoff (e.g. `admit_time + 24h`); drop LOS-normalised features; audit fails on target-source or `discharge*` columns in `feature_cols`; revisit target/cohort.

### [MAJOR] Nested-CV "optimism bias" compares against a held-out test score
- **File:** `solutions/ex_2/04_cross_validation.py:102-109, 146-148`; hint `local/ex_2/04:74-75`.
- **Evidence:** `biased_score = float(ridge_cv.score(X_test, y_test))`; `print(f"Optimism bias (standard - nested): {biased_score - nested_mean:+.4f}")`.
- **Problem:** A test-set score is an honest estimate, not the optimistic same-fold CV score; the printed "optimism bias" measures nothing — defeats the spec LO "Implement nested CV for unbiased model selection".
- **Fix:** Use the best mean inner-CV score (`GridSearchCV.best_score_` or RidgeCV CV results) as the biased number.

### [MAJOR] TimeSeriesSplit / GroupKFold demonstrated on non-temporal data with random fake groups; no stratified k-fold
- **File:** `solutions/ex_2/04_cross_validation.py:171-195, 222-224`.
- **Evidence:** `X_train` from a shuffled 80/20 split of a cross-sectional table; `groups = np.repeat(np.arange(n // 5 + 1), 5)[:n]; rng.shuffle(groups)`; interpretation "If TimeSeriesSplit gives a substantially LOWER R² … your data has temporal leakage". StratifiedKFold appears only inside an ex_3 helper.
- **Fix:** Use data with real timestamps and repeated entities (e.g. ICU grouped by `patient_id`: 3,927 patients, 7,818 admissions); add StratifiedKFold.

### [MAJOR] ex_4 tuning selects hyperparameters and early-stopping round on the test set
- **File:** `solutions/ex_4/04_boosting_tuning.py:107-124, 129-135, 146-153`; `ex_4/02:109`; `ex_4/03:118-124`.
- **Evidence:** Heatmap scored with `average_precision_score(y_test, y_p)` then `heatmap.argmax()`; `eval_set=[(X_test, y_test)]`; reported on same `X_test`.
- **Fix:** Validation split / CV from `X_train` for sweep and early stopping; report on `X_test` once.

### [MAJOR] Spec 3.1 / 3.4 techniques not exercised
- **File:** `solutions/ex_1/*`, `solutions/ex_4/*`.
- **Evidence:** grep in `solutions/ex_1` for SequentialFeatureSelector/forward/backward/lag/rolling/day-of-week/PolynomialFeatures/FeatureEngineer/correlation-threshold → nothing; grep AdaBoost in `solutions/ex_4` → nothing; `make_catboost` receives ordinal-encoded data with no `cat_features`.
- **Problem:** Missing: temporal features, interaction/polynomial generation, forward/backward wrapper selection (only RFE), correlation-threshold filter, kailash `FeatureEngineer` (spec "generate + select"), AdaBoost warm-up, CatBoost native categoricals (the advantage the text describes is never exercised).
- **Fix:** Add to ex_1 and ex_4 respectively.

### [MAJOR] Spec 3.3 items missing: NB variants, post-pruning, same-CV comparison
- **File:** `solutions/ex_3/03_naive_bayes.py`, `04_decision_tree.py`, `06_model_zoo.py:84-90`.
- **Evidence:** Only `GaussianNB`; no `ccp_alpha`/cost-complexity pruning; `06_model_zoo` scores each model once on a 1,000-row holdout with hard-coded hyperparameters (k=11, depth=7, C=1), not the tuned values; REFLECTION claims "Trained 5 classical models on identical data and folds".
- **Problem:** Spec assessment: "Comparison uses consistent evaluation (same CV splits)".
- **Fix:** Add MultinomialNB/BernoulliNB and post-pruning; in 06 use `cross_validate` with shared `cv`, tuned hyperparameters, CV mean±std and timing columns.

### [MAJOR] "Visualise" phases produce no visual; scaffolds strip figures (ex_1–ex_6)
- **File:** `solutions/ex_3/01-06` (e.g. `01_svm.py:139-161` computes mesh `Z` and only prints "Decision mesh shape"); `solutions/ex_2/01, 02, 04` (zero figures); `local/ex_1/*` (solutions 7 `write_html`, locals 0); `local/ex_4/01`; `local/ex_5/02-05` (4/0, 4/0, 6/0, 4/0) and `local/ex_6/02-04` (2/0 each).
- **Problem:** Spec 3.2 "Bias-variance demonstrated visually (train/test error curves)", 3.5 "Generate calibration plot" (local/ex_5/05 has none), R9A visual proof — students never see a decision boundary, bias-variance curve, ridge path or reliability diagram.
- **Fix:** Render `Z` as contour+scatter; add bias²/variance/test-MSE vs degree and coefficient-path plots; regenerate locals keeping plotting code with only arguments blanked.

### [MAJOR] Regularisation-path, learning-curve and sweep plots have wrong x-axes
- **File:** `solutions/ex_2/03_lasso_elasticnet.py:215-236`; `solutions/ex_2/05_learning_curves.py:131-137`; `solutions/ex_3/01_svm.py:174`, `02_knn.py:163`, `05_random_forest.py:177`.
- **Evidence:** `ModelVisualizer.training_history` (kailash-ml 2.2.2 source) plots `x=list(range(1, len(values)+1))`; axes labelled "Regularisation Strength (α)" (log) and "Training set size (samples)" actually show 1…7 / 1…6.
- **Fix:** Build `go.Scatter(x=ALPHAS …)` / `go.Scatter(x=train_sizes …)` directly.

### [MAJOR] Bias-variance interpretation contradicts the printed output
- **File:** `solutions/ex_2/01_bias_variance.py:124-127, 196-203`.
- **Evidence:** Re-running the exact RNG: degree 1 bias² 0.1836 / var 0.0054; degree 10 0.0156 / 0.1821; degree 15 0.0090 / 0.0060 → "Dominant" prints Bias at degree 15; Checkpoint 2 passes by 0.0054 < 0.0060; test MSE 0.0383 (deg 6) → 0.0446 (deg 20), no "blow up".
- **Problem:** "At degree=15, Variance dominates" and "test MSE … then blows up" are false for the shipped seed.
- **Fix:** More bootstrap replicates and a fixed test grid (e.g. 200 points on [0,1]); compare degree 1 vs 10; reword to match output.

### [MAJOR] FeatureSchema "validation" checks names only; wrong dtypes; target declared as a feature
- **File:** `solutions/ex_1/05_validation_and_tracking.py:95-160`.
- **Evidence:** `age` Int64 (declared float64); `n_unique_medications`, `n_abnormal_labs` UInt32 (declared int64); `los_days` (target source) declared a feature; `entity_id_column="patient_id"` although rows are admissions; loop only asserts `field_def.name in features.columns` yet prints "FeatureSchema validated against the matrix".
- **Fix:** Compare dtypes/nullability and fail on mismatch; remove `los_days`; entity key `admission_id`.

### [MAJOR] Fabricated statistics, incidents and regulatory requirements attributed to real organisations (ex_1–ex_8)
- **File:** ex_1–4: `solutions/ex_1/04_embedded_selection.py:259`, `solutions/ex_2/04_cross_validation.py:267-270`, `solutions/ex_2/02_ridge_regression.py:231-234`, `solutions/ex_2/01_bias_variance.py:209`, `shared/mlfp03/ex_3.py:268-269`, `solutions/ex_1/05:383` (10 solution + 9 local files). ex_5–8: `ex_5/01:24,158-171`, `ex_5/02:75-76,252-283,306`, `ex_5/03:203-233`, `ex_5/04:276-281`, `ex_5/05:248-277`, `ex_6/03:221-225`, `ex_6/04:178-208`, `ex_6/05:280-303`, `ex_7/02:214-218`, `shared/mlfp03/ex_7.py:234-257`, `ex_8/01:253-290`, `ex_8/02:251-267`, `ex_8/03:139,273-287`, `ex_8/04:280-303`, `ex_8/05:253` (+ locals).
- **Evidence:** "DBS's internal studies show each percentage point of fraud recall … S$4.2M/year"; "A previous GrabPay model used shuffled k-fold and reported 93% fraud recall. When deployed, true recall was 71%"; "Ridge-based scorecards earn a 15 bp reduction in the PD floor"; "Sources in reading notes (SGX retail analyst reports, Shopee/Lazada 2024 ops reviews)"; UK Care.data / Dutch SyRI described as leakage incidents (they were consent / human-rights failures); "real UOB card-fraud story … rolled back the model within 10 days"; "UOB's 2023 SHAP interaction audit … reduced the DIR gap from 0.71 to 0.89"; "SMOTE is cited in 92% … <10% of production deployments (Fernandez et al. 2018)"; "PDPA and the MAS Notice on Credit Decisions require … a SPECIFIC reason" (adverse-action notices are US ECOA/Reg B); "MAS 2024 Fair Dealing Guidelines require … an ANNUAL fairness disclosure"; "~12% credit default rate (MAS FSR 2024)"; brand names Chanel, Marina Bay Sands; DBS, OCBC, UOB, Maybank, StanChart, Citi, AIA, StarHub, SingHealth, SGH, NHCS, MOH, Grab Financial throughout.
- **Problem:** Invented facts presented as true about named commercial entities, papers and regulators — misleading and violates independence rules (commercial references).
- **Fix:** Anonymised archetypes ("a Singapore retail bank"); label figures "illustrative"; remove invented regulatory requirements/citations.

### [MINOR] `.to_pandas()` in a solution
- **File:** `solutions/ex_1/01_feature_engineering.py:183-184` (`features.select(...).to_pandas()` then `.corr()`).
- **Fix:** `features.select(corr_cols).corr()`; `.to_numpy()` into `px.imshow`.

### [MINOR] Deprecated `penalty="l1"` in LogisticRegression (sklearn 1.9)
- **File:** `solutions/ex_1/04_embedded_selection.py:103-109`, `solutions/ex_1/05:199-201`, `local/ex_1/05:124`, hint `local/ex_1/04:74`.
- **Evidence:** "FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10 … Use l1_ratio=1"; "UserWarning: Inconsistent values: penalty=l1 with l1_ratio=0.0".
- **Fix:** `LogisticRegression(l1_ratio=1.0, C=…, solver="saga")`.

### [MINOR] RFE elimination curve selects features outside CV
- **File:** `solutions/ex_1/03_wrapper_selection.py:138-157` (`rfe_curve.fit` on all rows, then `cross_val_score` on reduced features; plain 3-fold lets patients span folds).
- **Fix:** `cross_val_score(Pipeline([("rfe", RFE(...)), ("rf", ...)]), …, cv=GroupKFold, groups=patient_id)`.

### [MINOR] XGBoost described as "exact split finding"
- **File:** `solutions/ex_4/03_lightgbm_catboost.py:56-57`. XGBoost ≥2.0 `tree_method='auto'` → `hist`.
- **Fix:** Histogram by default; `exact` optional.

### [MINOR] Wrong dataset size in interpretations
- **File:** `solutions/ex_4/03_lightgbm_catboost.py:137-141, 168-170` ("On 5K rows…"); `X_train.shape` = (80000, 35).
- **Fix:** Correct and re-derive timing claims.

### [MINOR] XGBoost importance axis mislabelled
- **File:** `solutions/ex_4/02_xgboost.py:134-137, 160` ("Gain (total split-loss reduction …)"); default resolves to `gain` = average gain per split.
- **Fix:** Relabel or set `importance_type="total_gain"`.

### [MINOR] Early-stopping chart plots the same LightGBM score twice; dead code
- **File:** `solutions/ex_4/04_boosting_tuning.py:201-205` ("Fixed 500 rounds" bar uses `lgb_es_metrics["auc_pr"]`), line 115 `auc_roc = float(average_precision_score(...))  # fallback`.
- **Fix:** Train a fixed-500 LightGBM for the bar; remove line 115.

### [MINOR] Business arithmetic inconsistent
- **File:** `solutions/ex_1/02_filter_selection.py:282-287` ("S$45 per study" vs calculation at S$0.75/min → ≈S$246K; S$45/min would be ≈S$14.8M).
- **Fix:** Make the rate consistent.

### [MINOR] SVM margin called "calibrated"; ridge shrinkage called "uniform"
- **File:** `solutions/ex_3/01_svm.py:204-205`; `solutions/ex_2/02_ridge_regression.py:49-51`.
- **Problem:** SVM decision values are uncalibrated (hence Platt); ridge shrinks low-variance principal directions more.
- **Fix:** Correct both.

### [MINOR] Local scaffolds ex_1–ex_4 don't preserve header/REFLECTION blocks; one checkpoint stripped
- **File:** `local/ex_1..ex_4`.
- **Evidence:** WHAT YOU'LL LEARN differs in 17/20 pairs, REFLECTION in 20/20 (rule: verbatim); PREREQUISITES missing in `local/ex_3/02-06`; TASK list missing in `local/ex_1/02-05`, `local/ex_3/02-06`; `local/ex_1/05` drops `assert len(final_features) <= len(feature_cols)` (solutions/ex_1/05:225).
- **Fix:** Regenerate scaffolds preserving these blocks and all asserts.

---

## E. Exercises ex_5 – ex_8

(The planted-leak finding is in Section A; fabricated-claims finding merged into Section D.)

### [BLOCKING] ex_7 workflows use node types that do not exist; failure is swallowed and the checkpoint passes
- **File:** `solutions/ex_7/01_workflow_builder.py:74-144, 192`; `solutions/ex_7/05_orchestrated_pipeline.py:107-176`; matching locals.
- **Evidence:** Verified `NodeRegistry.get("DataPreprocessNode")`/`"ConditionalNode"` → NodeConfigurationError (`PythonCodeNode`, `SwitchNode` OK); fork also confirmed `ModelTrainNode`, `ModelEvalNode`, `PersistNode`. `.build()` fails "Node 'DataPreprocessNode' not found in registry"; `except Exception as exc:` (line 136) → `results, run_id = {}, "fallback-manual-run"` (143), so `assert run_id is not None` always passes. `WorkflowBuilder("credit_scoring_pipeline")` passes a string where `edge_config: dict` is expected.
- **Problem:** Students never see a workflow execute; no custom node (`@register_node`, `Node` subclass, `PythonCodeNode`) and no conditional/Switch node runs — spec 3.7 requires both; error hiding violates zero-tolerance Rule 3.
- **Fix:** Build the DAG from `PythonCodeNode` + `SwitchNode` and/or a `@register_node` `Node` subclass wired with `add_connection("…","result.<key>",…)`; remove the try/except; assert on real node outputs.

### [BLOCKING] Fairness audit splits groups on `customer_id`; protected attribute never audited
- **File:** `solutions/ex_6/05_fairness_audit.py:170, 183-221`; `shared/mlfp03/ex_6.py:58, 104`.
- **Evidence:** Equalized odds and impossibility base rates use `synthetic_group_split(X_test, feature_idx=0)`; feature 0 is `customer_id`. `PROTECTED_CANDIDATES = ["age","gender","ethnicity","marital_status"]` — dataset column is `race`; `age` skipped for >10 unique values. Calibration parity not computed.
- **Problem:** Equalized-odds and base-rate "groups" are arbitrary ID ranges; spec 3.6 "Fairness measured quantitatively … Impossibility theorem explained with model-specific example" is not met.
- **Fix:** Use `race`, `gender`, binned `age` for DI, equalized odds and per-group calibration; remove the `customer_id` split.

### [BLOCKING] SHAP additivity check compares log-odds to probabilities
- **File:** `solutions/ex_6/01_shap_global.py:96-116`; `local/ex_6/01_shap_global.py:92-94` (hint repeats the bug).
- **Evidence:** `shap_sum = shap_vals[i].sum() + expected_value` (log-odds; expected_value −7.83) compared with `predict_proba(...)[:,1]`; gaps 2.4–14.1 on first 10 rows vs 5e-14 against `predict(raw_score=True)`; checkpoint never tests the gap but prints "additivity verified".
- **Fix:** Compare to `model.predict(X, raw_score=True)` (or `model_output="probability"`); assert mean gap < 1e-6.

### [BLOCKING] "Focal loss" exercise does not implement focal loss
- **File:** `solutions/ex_5/03_loss_functions.py:47-70, 85-102, 222`.
- **Evidence:** Loop only sweeps `scale_pos_weight = alpha * base_weight`; gamma appears only in the plotting formula (:178); :67-70 claims this captures "the focus on hard examples effect"; :47-50 says p=0.99 contributes "almost as much" loss as p=0.51 (actual CE 0.010 vs 0.673, 67×).
- **Problem:** Class reweighting is constant per class and cannot down-weight easy examples; spec 3.5 requires Focal Loss with γ.
- **Fix:** Implement FL as a LightGBM custom objective (grad/hess) and sweep γ; correct the CE claim (many easy examples collectively dominate).

### [BLOCKING] Cost-sensitive learning claimed to improve calibration; conclusion printed unconditionally; contradictory/incorrect calibration theory
- **File:** `solutions/ex_5/02_sampling_strategies.py:244-248, 271-274, 289-290, 305`; `solutions/ex_5/05_calibration.py:59-61, 82-85, 143-146`.
- **Evidence:** Leak-free Brier: baseline 0.098, SMOTE 0.098, `scale_pos_weight` 0.169, cost-matrix 0.432 (mean predicted p 0.127 → 0.353 → 0.641). 02:289 prints "Cost-sensitive delivered better-calibrated probabilities" regardless of results; 05:60 says reweighting compresses scores "towards 0 and 1", 05:146 says "towards 0.5"; 05:84 says isotonic "can correct non-monotonic miscalibrations".
- **Problem:** Reweighting inflates probabilities and damages calibration (that's why Platt/isotonic follow); isotonic is monotone by construction.
- **Fix:** State that weighting over-predicts positives and worsens Brier; compute the verdict from the table; isotonic corrects any monotone non-sigmoid distortion.

### [MAJOR] Lesson 3.8 engines missing from ex_8 (DriftMonitor not used; no DataFlow; partial CRUD in ex_7)
- **File:** `solutions/ex_8/02_drift_monitoring.py:44, 105-113`; all ex_8; `solutions/ex_7/02_dataflow_persistence.py`.
- **Evidence:** `grep DriftMonitor` in ex_7/ex_8 code → comments only; 8.2 builds a `DriftSpec`, prints it, then uses hand-rolled `compute_psi`/`compute_ks`; no DataFlow in ex_8; ex_7 only `express.create`/`express.list`; ex_8/05:339 claims "8.2 DriftMonitor: PSI + KS".
- **Problem:** Spec 3.8: "Set up DriftMonitor with alerting thresholds … Drift detected when injected. DataFlow CRUD operations work."
- **Fix:** `DriftMonitor(conn, tenant_id=…, psi_threshold=, ks_threshold=)` + `set_reference_data`/`check_drift` (as assessment task_4 does); add read/update/delete.

### [MAJOR] Production-readiness gates hard-coded `True`; stub model card written so the gate passes
- **File:** `solutions/ex_8/05_production_readiness.py:116-131`; `local/ex_8/05_production_readiness.py:97-113`.
- **Evidence:** Reproducibility, Fairness, SHAP, Rollback gates are literal `True`; missing card → stub written; local says "Each predicate must be an expression, not a constant — the autograder flips your model" while shipping those constants (no exercise autograder exists).
- **Fix:** Derive each gate from real artefacts; fail when the card is missing.

### [MAJOR] Model card asserts fairness results never measured
- **File:** `solutions/ex_8/03_model_card.py:176, 178, 182` ("disparate impact within 0.8-1.25 band", "within MAS FEAT recommended band"). Nothing in ex_8 computes DI; ex_6 writes no `ex6_fairness.html`; FEAT defines no numeric band.
- **Fix:** Compute or load DI / equalized odds and print actual values.

### [MAJOR] Threshold optimised and savings reported on the same test set
- **File:** `solutions/ex_5/04_threshold_optimisation.py:105-131, 286-297` (:74-75 says to sweep "on a validation set").
- **Fix:** Tune on validation / out-of-fold; report on test.

### [MAJOR] Bayesian-vs-grid "lift" compares different metrics on different data; ROI figure mismatch
- **File:** `solutions/ex_7/03_hyperparameter_search.py:224-296, 310`; `shared/mlfp03/ex_7.py`.
- **Evidence:** Bayesian scored on last-20% test rows; grid via 5-fold CV on train; the search's own holdout is a random 20% of the full frame (trials trained on the reporting rows). :310 says `headline_roi_text()` "uses the conservative S$4M figure"; helper computes 560 × 18,000 × 0.65 = S$6.55M.
- **Fix:** Evaluate both on identical unseen folds/holdout; correct the S$ figure.

### [MAJOR] SHAP-interaction theory wrong
- **File:** `solutions/ex_6/04_shap_interactions.py:49, 59-62, 126-132`.
- **Evidence:** `phi_i = phi_ii + (1/2)·Σ phi_ij`; numerically (shap 0.52) the row-sum of the interaction matrix equals φ_i (error 6e-15), the ½ version is off by 0.51. Off-diagonal mass called "EXACTLY the non-linearity" — it is non-additivity (additive models with non-linear main effects have zero interactions).
- **Fix:** `phi_i = phi_ii + Σ_{j≠i} phi_ij`; rename to "interaction share".

### [MAJOR] Chart titled "SHAP Feature Importance" shows LightGBM split counts
- **File:** `solutions/ex_6/01_shap_global.py:132-134` (`ModelVisualizer.feature_importance` uses `model.feature_importances_`).
- **Fix:** Plot mean|SHAP| (or `ModelExplainer`), or retitle as split importance.

### [MAJOR] Conformal coverage guarantee misstated
- **File:** `solutions/ex_8/01_conformal_prediction.py:159-161, 265-270, 322-324`.
- **Evidence:** Singleton {0} described as "model 90% confident: no default" for that applicant (guarantee is marginal); "holds regardless of which macroeconomic regime" (requires exchangeability); "If coverage drifts … retrain, don't re-calibrate" (recalibrating q̂ is the standard remedy).
- **Fix:** Marginal coverage under exchangeability; recommend recalibration under shift.

### [MAJOR] Spec 3.5/3.6 topics not exercised
- **File:** `solutions/ex_5/*`, `solutions/ex_6/*`.
- **Evidence:** grep → no stacking/blending, no ALE, no calibration parity, no regression metrics (R², MAE, RMSE, MAPE), no log loss in ex_5; KernelSHAP prose only; `ModelExplainer` unused in ex_6.
- **Fix:** Add brief stacking demo, ALE on a correlated feature, per-group calibration, regression-metrics panel.

### [MAJOR] ex_5/ex_6/ex_8 bypass Kailash engines (Framework-First directive)
- **File:** `ex_5/01:106-107`, `02:100-125`, `03:90-96`, `05:98-114`; `shared/mlfp03/ex_6.py:127-135, 159`; `shared/mlfp03/ex_8.py:116-144`; `ex_7/02:114-122`, `ex_7/03:224-288`, `ex_7/05:278-330`.
- **Evidence:** Direct `lgb.LGBMClassifier(...).fit`, `CalibratedClassifierCV`, `shap.TreeExplainer`; `ex_7/05:192` says "no raw .fit() in user code" then re-fits raw LightGBM at :278.
- **Fix:** Train via `TrainingPipeline`, explain via `ModelExplainer`, engine metrics (assessment solutions show the pattern).

### [MAJOR] ex_5 interpretation text hard-codes leak-era / wrong outcomes
- **File:** `solutions/ex_5/01_metrics_and_baseline.py:169-171` ("Recall ~20-30%", "~S$9M"); `solutions/ex_5/02_sampling_strategies.py:271-273` ("Brier 2-3x better" for cost-sensitive).
- **Problem:** Hard-coded claims about outputs that the code does not produce (with or without the leak).
- **Fix:** Recompute after the leak fix and quote from the actual run (f-strings from computed values).

### [MINOR] "your F1 textbook would congratulate you" for an all-negative model
- **File:** `solutions/ex_5/01:52`. All-negative F1 = 0. **Fix:** "accuracy".

### [MINOR] "Best recall" label uses best-AUC-PR row
- **File:** `solutions/ex_5/03:237` (`best_by_pr`). **Fix:** Relabel or select by recall.

### [MINOR] Dead, wrong `default_cost` computation
- **File:** `solutions/ex_5/04:132-136` (overwritten at :141). **Fix:** Delete.

### [MINOR] Four-fifths rule attributed to ECOA
- **File:** `solutions/ex_6/05:10, 211`. It originates in the EEOC Uniform Guidelines (employment). **Fix:** Correct attribution.

### [MINOR] SHAP "handles correlated features correctly" overclaim; checkpoint message asserts nothing
- **File:** `solutions/ex_6/02:68-70, 197` ("ranking matches SHAP top-10" with no overlap assert). **Fix:** Soften claim; assert overlap or change message.

### [MINOR] "Borderline" LIME applicant is the median-ranked one (p≈0.01)
- **File:** `solutions/ex_6/03:107`. **Fix:** Pick the row nearest the decision threshold.

### [MINOR] Hand-built `ModelSignature` never attached
- **File:** `solutions/ex_7/04:90-95`; reflection "Registered … with signature" inaccurate (TrainingPipeline auto-signature). **Fix:** Pass the signature to registration or correct the text.

### [MINOR] Hint lists wrong metric keys
- **File:** `local/ex_7/03:258` ("accuracy, f1, auc, auc_pr, log_loss, brier"); actual keys accuracy, f1, auc_roc, auc_pr, log_loss. **Fix:** Correct the hint.

### [MINOR] `registry.promote(...)` does not exist
- **File:** `solutions/ex_8/04:77, 297` (`hasattr` False; real `promote_model(name, version, target_stage, reason=)`). **Fix:** Use `promote_model`.

### [MINOR] `compute_psi` drops out-of-range current values
- **File:** `shared/mlfp03/ex_8.py:176-182` (reference histogram edges; values outside range silently dropped → PSI understates). **Fix:** Outer edges ±inf.

### [MINOR] Conformal quantile uses default linear interpolation
- **File:** `solutions/ex_8/01:122-123`. **Fix:** `method="higher"` with the ⌈(n+1)(1−α)⌉/n level.

### [MINOR] Checkpoint asserts stripped in ex_5/ex_8 locals
- **File:** `local/ex_5/02` (`sample_weights … == DEFAULT_COSTS.fn`), `local/ex_5/03` (Brier in [0,1]), `local/ex_8/01` (α-monotonic coverage), `local/ex_8/03` (`"AUC-ROC" in model_card`, `"Coverage" in model_card`). Rule: checkpoints never stripped. **Fix:** Restore verbatim.

---

## F. Assessment (task_1 … task_4)

All four reference solutions pass their own graders (run in .venv): task_1 12/12 (~30 s), task_2 12/12 (~63 s), task_3 13/13 (~15 s), task_4 11/11 (~18 s). Solution outputs match the "Visible sanity checks" in each problem.md (e.g. t1 positive rate 0.2537, top feature `loyal_and_satisfied`; t2 best LR AUC 0.911; t3 recall 0.622→0.720, acc 0.856→0.840, AUC 0.896; t4 version 1/production, AUC 0.902, clean 0 feats, shift 4 feats "severe"). Check counts in each problem.md match the graders.

### [MAJOR] Task 1: grader cannot detect leaky (all-rows) feature selection, contrary to problem.md
- **File:** `assessment/task_1/problem.md:79-80, 97-99`; `assessment/task_1/grader.py:208-241`.
- **Evidence:** problem.md: "Selection MUST be fit on the train split only — fitting on all rows leaks the test distribution and fails the leakage checks." Ran the grader against the solution module with `TRAIN_FRACTION = 1.0` (selection fit on all 10,000 rows): `passed=True 12/12`, selected features identical in membership. The grader's "leakage" checks only test forbidden columns (check 2) and overlap ≥6 with a train-split RF ranking (check 10) — neither distinguishes train-only from all-rows fitting.
- **Problem:** The assessed learning outcome (leakage-free selection, 20 marks) is not actually graded; the statement to students is false.
- **Fix:** Either add a check that detects all-rows fitting (e.g. require the submission to also return the fitted train frame/`n_train` and verify `selected.data` height == 7,500, or construct the target so train-only vs all-rows rankings provably differ), or remove the claim from problem.md.

### [MINOR] Task 2 "F1 ≥ 0.80" is support-weighted F1, unstated
- **File:** `assessment/task_2/problem.md:45-49, 64, 70`.
- **Evidence:** kailash-ml `metrics/_registry.py:235-238`: `f1_score(y_true, y_pred, average="weighted")`. Solution table shows f1 ≈ accuracy (LR 0.863 vs 0.867) — dominated by the 75% majority class.
- **Problem:** In a module teaching that minority-class metrics matter, an unqualified "F1 ≥ 0.80" on a 3:1 imbalanced target reads as positive-class F1 (which would be far lower).
- **Fix:** State "support-weighted F1 (TrainingPipeline's `f1`)" in the contract and target.

### [MINOR] Starters instruct students to run a grader that is withheld
- **File:** `assessment/task_1..4/starter.py:12` ("python grader.py starter.py     # grade your attempt") vs `assessment/README.md:65` ("The automated graders are withheld") and commit 2860a5a6 ("build-student-repo.sh ships only problem.md + starter.py").
- **Fix:** Remove the grader instruction from starters (e.g. "run `python starter.py` and compare against the visible sanity checks in problem.md").

---

## G. Coverage summary

- **Specs/rules read:** CLAUDE.md, specs/_index.md, specs/module-3.md, specs/redlines.md, specs/exercise-mapping.md (M3), .claude/rules/exercise-standards.md, independence.md, domain-integrity.md, env-models.md.
- **Teaching materials (29 files):** deck.html (90 slides, 2,578 lines), speaker-notes.md (1,312), textbook.md (4,011), README.md, index.html, lessons/01–08 × {slides, notes, textbook}.html (24 files). Every Kailash snippet was exec'd or signature-checked in .venv; shap/lime/polars/PythonCodeNode behaviours reproduced directly; relative links scanned (one broken).
- **Exercises (84 files):** 40 solution .py, 40 local .py, shared/mlfp03/ex_1..ex_8.py + __init__.py. Every solution/local pair diffed; asserts compared by AST; all locals pass `py_compile`; no `____` in solutions; no `import pandas` (one `.to_pandas()`); no hardcoded LLM model names; data facts (leak columns, label determinism, target correlations, Brier/SHAP numbers) verified with short .venv probes (no full exercise runs).
- **Assessment (17 files):** README + 4 × {problem.md, starter.py, solution.py, grader.py}; all four graders run against solutions; task_1 leakage claim tested by monkeypatched re-grade; starter vs solution diffs reviewed.
- **Not reported (per brief / by design):** Colab notebooks; redline-check; code-fit sizing; solution runtime pass/fail. No module quiz exists — the deliberate design per commit 2860a5a6 ("The exam IS the assessment") — reported only where the deck contradicts it.
- **Total files examined:** ~130.
