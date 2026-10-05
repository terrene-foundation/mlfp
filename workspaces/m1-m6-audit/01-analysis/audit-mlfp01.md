# MLFP Module 1 (mlfp01) — Correctness and Completeness Audit

Date: 2026-10-02. Read-only audit of `modules/mlfp01/` (excluding `colab-selfcontained*/`) plus `shared/run_profile.py`, `shared/kailash_helpers.py`, `shared/data_loader.py`.
Installed stack: kailash-ml 2.2.2, polars 1.41.2, plotly 6.8.0 (introspected via `.venv/bin/python`).
Data facts used throughout (from `data/`, which `MLFPDataLoader` reads first):

- `mlfp01/hdb_resale.parquet`: (50150, 11), 2015-01..2024-12, 27 towns in UPPERCASE, 6 flat types, `remaining_lease` is a string. The data is synthetic, with no price trend by year.
- `mlfp01/sg_weather.csv`: (12, 3) with columns `month` (full names), `mean_temperature_c`, and `total_rainfall_mm` (i64). Mean temperature is 27.47, max is 28.3 (May), rain mean is 171.75.
- `mlfp01/sg_taxi_trips.parquet`: (50000, 12) with columns `trip_id, pickup_datetime, dropoff_datetime, pickup_zone, dropoff_zone, distance_km, fare_sgd, tip_sgd, payment_type, passengers, pickup_latitude, pickup_longitude`.
- `mlfp01/sg_cpi.csv`, `sg_employment.csv` (key column `quarter`, no `date`), `sg_fx_rates.csv`, `economic_indicators.csv` (401 x 8, key `period`).
- `mlfp_assessment/mrt_stations.parquet`: 150 rows over 32 Title-Case towns (several stations per town). `schools.parquet`: 242 rows.

Installed API facts used throughout:

- `DataExplorer`:
  - Methods are `profile`, `compare`, `to_html`, `visualize`. All four are `async`. There is no `check_alerts`.
  - `profile(data, *, columns=None)` takes no `name=` argument.
  - `compare(data_a, data_b, *, columns=None)` returns a dict with keys `profile_a, profile_b, column_deltas, shape_comparison, shared_columns, missing_in_a, missing_in_b`.
- `DataProfile` fields: `n_rows, n_columns, columns, correlation_matrix, …, duplicate_count, duplicate_pct, alerts`.
- `alerts` is a list of dicts with keys `type/column(s)/value/severity`. There is no `message` key.
- Alert types emitted, exactly eight: `high_nulls, constant, high_skewness, high_zeros, high_cardinality, high_correlation, duplicates, imbalanced`.
- `AlertConfig` fields: `high_correlation_threshold=0.9, high_null_pct_threshold=0.05, constant_threshold=1, high_cardinality_ratio=0.9, skewness_threshold=2.0, zero_pct_threshold=0.5, imbalance_ratio_threshold=0.1, duplicate_pct_threshold=0.0`.
- `PreprocessingPipeline()`:
  - The constructor takes no arguments.
  - Methods are `setup(data, target, *, …)`, `transform`, `inverse_transform`, `get_config`.
  - There is no `fit_transform` and no `steps_applied`.
- `ModelVisualizer` methods: `box_plot, calibration_curve, confusion_matrix(y_true, y_pred, labels), feature_importance, histogram, learning_curve, metric_comparison, precision_recall_curve, residuals, roc_curve, scatter, training_history`.

---

## BLOCKING

### [BLOCKING] `shared.run_profile` / `run_compare` / `run_alerts` are broken against the installed API, and every call site uses yet another signature

**File:** `shared/run_profile.py:32, 51, 73`. Call sites:
- `modules/mlfp01/deck.html:2506-2509, 2552, 2588, 2596, 2635, 3039-3055`
- `lessons/07/slides.html:144-150, 210, 273, 332, 351, 374`
- `lessons/08/slides.html:338-341, 451-452`
- `lessons/08/textbook.html:337-348, 398-399`

**Evidence:**
```
run_profile(df)                          -> TypeError DataExplorer.profile() got an unexpected keyword argument 'name'
run_profile(DataExplorer(), df)          -> TypeError DataExplorer.profile() got an unexpected keyword argument 'name'
run_profile(explorer, df, df_compare=df) -> TypeError run_profile() got an unexpected keyword argument 'df_compare'
```
- `run_compare` calls `explorer.compare(profiles, names=names)`. The real signature is `compare(data_a, data_b, *, columns)`.
- `run_alerts` calls `explorer.check_alerts(...)`, which does not exist (`hasattr(DataExplorer,'check_alerts')` is False).
- Teaching material calls `run_profile(explorer, df)`, `run_profile(explorer, df, config=config)` and `run_profile(explorer, df_a, df_compare=df_b)`. None of these matches the helper's own `(df, name)` signature.

**Problem:** Spec 1.7 says "Async hidden behind `shared.run_profile()` sync wrapper". The wrapper raises on every call, and the deck and lesson code that uses it cannot run. The exercises work around it with `asyncio.run` directly.

**Fix:**
- Rewrite `run_profile(df, alert_config=None)` as `asyncio.run(DataExplorer(alert_config=alert_config).profile(df))`.
- Rewrite `run_compare(df_a, df_b)` as `asyncio.run(DataExplorer().compare(df_a, df_b))`.
- Delete `run_alerts`, or reimplement it via `profile(...).alerts`.
- Update every call site listed above to that single signature.

### [BLOCKING] DataExplorer / AlertConfig code in the deck and Lesson 1.7/1.8 pages uses attributes and kwargs that do not exist

**File:**
- `deck.html:2511-2517` (`profile.row_count`, `profile.column_count`, `profile.missing_summary`, `alert.severity/.column/.message`)
- `deck.html:2541-2549` (`AlertConfig(missing_threshold=…, outlier_std=…, duplicate_threshold=…, skew_threshold=…, correlation_threshold=…, cardinality_threshold=…, constant_threshold=0.99)`)
- `deck.html:2598-2601, 3056, 3064` (`explorer.compare(profile_before, profile_after)` then `.summary`)
- `lessons/07/slides.html:150-151, 203-208, 275-277, 377-379` (`alert.type/.message`, `AlertConfig(null_threshold=…, skew_threshold=…, correlation_threshold=…, outlier_iqr_factor=…)`, `comparison.differences`, `diff.severity`)
- `lessons/07/notes.html:177-186` (`profile.column_stats["cpi"].mean`)
- `lessons/08/slides.html:343, 452`
- `lessons/08/textbook.html:351-352`

**Evidence:**
```
AlertConfig(missing_threshold=0.05)   -> TypeError ... unexpected keyword argument 'missing_threshold'
AlertConfig(null_threshold=0.10)      -> TypeError ... unexpected keyword argument 'null_threshold'
AlertConfig(outlier_iqr_factor=2.5)   -> TypeError ... unexpected keyword argument 'outlier_iqr_factor'
DataProfile fields: n_rows, n_columns, columns, ..., alerts   (no row_count / column_count / missing_summary / column_stats)
alerts are dicts: {"type","column"|"columns","value","severity"}  (no attribute access, no "message")
DataExplorer.compare is async, takes two DataFrames, returns a dict (no .summary / .differences)
```
- `deck.html:2598` calls the async `compare` without awaiting it, so it returns a coroutine.

**Problem:** Every profiling code sample students see in the deck and in the 1.7/1.8 slides raises on the first line. `constant_threshold=0.99` also misteaches the field: it is an int unique-count threshold (default 1), not a fraction. `lessons/07/textbook.html:238-265, 421-459` uses the correct API, so the master deck and slides contradict the lesson textbook.

**Fix:** Replace these samples with the pattern already used in `lessons/07/textbook.html:421-459`:
- `AlertConfig(high_null_pct_threshold=…, skewness_threshold=…, high_correlation_threshold=…, …)`
- `profile.n_rows` / `profile.n_columns`
- `alert["type"]`, `alert.get("column", alert.get("columns"))`
- `asyncio.run(explorer.compare(df_raw, df_clean))["column_deltas"]`

The fixed `run_profile` wrapper from the previous finding can stand in for `asyncio.run`.

### [BLOCKING] "8 alert types" are taught wrong: Outlier and Type-inference alerts do not exist; high_zeros and imbalanced are omitted

**File:**
- `deck.html:2520, 2555-2567`
- `lessons/07/slides.html:109-122` (table rows `outliers > 3 IQR`, `low_cardinality < 3 unique`, `type_inference mismatch`; `constant` described as `std = 0`)
- `lessons/07/slides.html:246-251` and `lessons/08/slides.html:346-350` (sample outputs printing `type_inference:` and `outliers:` alerts)
- `lessons/07/notes.html:127-134, 220`
- `speaker-notes.md:702`
- The spec repeats the error in `specs/module-1.md` (Lesson 1.7 topics)

**Evidence:**
- `kailash_ml/engines/data_explorer.py::_generate_alerts` (lines ~930-1045) emits only `high_nulls, constant, high_skewness, high_zeros, high_cardinality, high_correlation, duplicates, imbalanced`.
- `constant` fires on `unique_count <= constant_threshold`, not on std = 0.
- There is no low-cardinality alert; `high_cardinality` fires when `cardinality_ratio > 0.9`, i.e. near-unique columns.
- `lessons/07/textbook.html:148-158` lists the correct eight, so the Lesson 1.7 textbook and slides contradict each other.

**Problem:** Students learn alert categories the engine never produces. They will look for "outliers" and "type_inference" alerts that cannot fire, and never learn `high_zeros` or `imbalanced`. Outlier statistics do exist per column (`ColumnProfile.outlier_count/outlier_pct`), but not as an alert.

**Fix:**
- Replace every alert table and sample output with the real eight types and thresholds (default `high_null_pct_threshold=0.05`, `skewness_threshold=2.0`, `high_correlation_threshold=0.9`, `high_cardinality_ratio=0.9`, `zero_pct_threshold=0.5`, `imbalance_ratio_threshold=0.1`, `duplicate_pct_threshold=0.0`, `constant_threshold=1`).
- Say that outliers are a per-column statistic, not an alert.
- Correct `specs/module-1.md` 1.7 to match.

### [BLOCKING] PreprocessingPipeline is taught with a non-existent API (`PreprocessingPipeline(numeric_strategy=…)`, `.fit_transform()`, `.steps_applied`)

**File:**
- `deck.html:2873-2883, 2904-2905, 2922-2923, 3053-3054`
- `lessons/08/slides.html:247-256, 380-387` (and the "fit_transform vs transform" slide around 221-249)
- `lessons/08/textbook.html:264-280, 376-381`
- `lessons/08/notes.html:161-189`

**Evidence:**
```
PreprocessingPipeline(numeric_strategy="median") -> TypeError ... unexpected keyword argument 'numeric_strategy'
hasattr(PreprocessingPipeline, "fit_transform") -> False ; hasattr(PreprocessingPipeline(), "steps_applied") -> False
Installed: PreprocessingPipeline() ; .setup(data, target, *, train_size=0.8, seed=42, normalize=True, normalize_method='zscore',
           categorical_encoding='onehot', imputation_strategy='mean', ...) -> SetupResult ; .transform ; .inverse_transform
```

**Problem:**
- This is the required Lesson 1.8 engine, and every sample raises.
- The "fit on train / transform on test" lesson is built around a method that doesn't exist. The real `setup()` requires a `target` and does the train/test split itself.
- `textbook.md:3372-3383, 3600-3611` and `solutions/ex_8.py` use `setup(...)` correctly, so the deck and lesson pages contradict them.

**Fix:**
- Rewrite these samples as `pipeline = PreprocessingPipeline(); result = pipeline.setup(data=df, target="fare_sgd", categorical_encoding="onehot", imputation_strategy="median", normalize=True)`.
- Show `result`'s train/test frames.
- Reframe "fit vs transform" around `setup()` (fits on the train split) and `transform()` (applies to new data).

### [BLOCKING] ModelVisualizer calls in the deck and Lesson 1.6 pages do not exist or have the wrong signature

**File:**
- `deck.html:2319-2324` (`viz.plot_correlation(df, title=…)`, `viz.plot_distribution(df, column=…, title=…)`)
- `deck.html:3060-3061` (same, in the capstone code)
- `lessons/06/slides.html:333-339` and `lessons/06/textbook.html:390-397` (`viz.confusion_matrix(matrix=corr.to_numpy(), labels=…, title=…)`)

**Evidence:**
```
hasattr(ModelVisualizer, "plot_correlation") / "plot_distribution" -> False
ModelVisualizer().confusion_matrix(matrix=[[1,0],[0,1]], labels=["a","b"], title="t")
  -> TypeError ModelVisualizer.confusion_matrix() got an unexpected keyword argument 'matrix'
Installed: confusion_matrix(self, y_true, y_pred, labels=None)
```

**Problem:**
- The deck's ModelVisualizer slide and the capstone call methods that don't exist.
- The Lesson 1.6 heatmap (one of the six required chart types) passes a correlation matrix to a function that computes a confusion matrix from label vectors. It raises TypeError. Even with positional args, it would plot the wrong thing.

**Fix:**
- Replace with installed methods: `viz.histogram(df, "resale_price", bins=40, title=…)` and `viz.scatter(...)`.
- For the heatmap, use `plotly.graph_objects.Heatmap` on `df.corr()` (which is what `solutions/ex_6.py` does), or `DataExplorer.visualize()`.
- Remove `matrix=`/`title=` usage of `confusion_matrix`.

### [BLOCKING] Deck and lesson code reference dataset files and columns that do not exist

**File:**
- `deck.html:456-459` (`pl.read_csv("data/singapore_weather.csv")`, "(1095, 8)", columns `date, station, rainfall_mm, temp_max, temp_min, humidity` at 486-544)
- `deck.html:585` (`loader.load("mlfp01", "singapore_weather.csv")`)
- `deck.html:758, 937, 1580-1582, 1943, 1951, 1987, 2634, 2918-2919` (`hdb_resale.csv`, `mrt_stations.csv`, `primary_schools.csv`, `economic.csv` via hard-coded `data/` paths)
- `deck.html:848` (`remaining_lease_months`)
- `deck.html:3046, 3060` (`"taxi_trips.csv"`, column `trip_distance`)
- `lessons/07/slides.html:147` (`"sg_economic_indicators.csv"`, with columns `date, cpi, employment, fx_rate, country` at 195-199, 264-269)
- `lessons/08/slides.html:303-308, 319-324` and `lessons/08/textbook.html:341-382` (`"sg_taxi_trips.csv"`, "(10000, 8)", columns `fare_amount, trip_distance_km, passenger_count`)

**Evidence:**
- `ls data/mlfp01` lists `economic_indicators.csv hdb_resale.parquet sg_cpi.csv sg_employment.csv sg_fx_rates.csv sg_taxi_trips.parquet sg_weather.csv`.
- `sg_weather.csv` is (12, 3).
- HDB has `remaining_lease` (String, e.g. "71 years 11 months"), not `remaining_lease_months`.
- Taxi is (50000, 12) with `fare_sgd`, `distance_km`, `passengers`.
- `MLFPDataLoader.load` raises `FileNotFoundError` for an unknown name after trying the Drive download.

**Problem:**
- Students copying any of these slides get FileNotFoundError or ColumnNotFoundError.
- The Exercise 1.1 slide's three questions ("How many days…", "hottest temperature ever recorded", "average daily rainfall") cannot be answered from the 12-row monthly file.
- Hard-coded `data/...` paths also contradict the course's own "never hardcode file paths" rule (`lessons/08/slides.html:385`).

**Fix:**
- Use `MLFPDataLoader().load("mlfp01", "<real filename>")` with the real column names everywhere.
- Rewrite sample outputs from actual runs.
- Rewrite the Exercise 1.1 questions for monthly data (hottest month, wettest month, mean monthly rainfall), or ship the ~1K-row daily dataset the spec calls for.

### [BLOCKING] Lesson 1.4 join teaching is wrong for the course data: keys never match, and MRT is not one-row-per-town

**File:**
- `lessons/04/textbook.html:289-298` ("MRT stations has one row per town — the key is unique on the right side… the output has the same row count as HDB")
- `lessons/04/slides.html:138-145` (`# same number of rows — left join preserves all HDB rows`)
- `lessons/04/notes.html:131-133, 168-170` ("HDB has 26 towns. MRT has 24 towns. Matched 24.")
- `lessons/04/textbook.html:216-221, 401-428`
- `deck.html:1554-1559, 1580-1594`
- `textbook.md:1849` ("The MRT table has one row per town"), `1861`, `1879`, `1884-1893` (row count "equals hdb"), `1900-1904`, `1928` (expects 0 nulls), `1975-1982` (shows correlations −0.58 / 0.42 that come out null/NaN on the real data)

**Evidence:**
```
hdb.join(mrt.select("town","nearest_mrt","distance_to_mrt_km"), on="town", how="left") -> 50150 rows, 50150 nulls in distance_to_mrt_km
(HDB towns are 'BISHAN'… ; MRT towns are 'Bishan'… -> 0 matches)
after mrt.town.str.to_uppercase(): left join -> 186997 rows (MRT has 150 rows over 32 towns)
HDB has 27 towns; MRT has 32 distinct towns
```
The schools join has the same casing problem, so in the textbook version `school_count` becomes 0 everywhere after `fill_null(0)`.

**Problem:**
- The worked example silently produces 100% nulls. Every town then becomes "Unknown" in `desirability_tier`.
- The text claims row count is preserved "because the key is unique". It only looks preserved because nothing matched.
- Once casing is fixed, the join explodes 3.7x, which is the exact many-to-many failure the lesson warns about, while the lesson asserts the right side is unique.
- `solutions/ex_4.py` normalises case and aggregates per town, so the lesson pages contradict the exercise.

**Fix:**
- In the lesson pages and deck, normalise case first (`mrt.with_columns(pl.col("town").str.to_uppercase())`).
- Aggregate MRT to one row per town before joining, as the lesson already does for schools.
- Correct the cardinality paragraph and the "26/24/24" notes using the actual key-check output.

### [BLOCKING] Lesson 1.1 pages show fabricated outputs that contradict `sg_weather.csv`, and misread what `describe()` reports for the string column

**File:**
- `lessons/01/slides.html:314-325, 429-437` (schema shows `total_rainfall_mm: Float64`; describe shows mean 26.68, std 0.43, rain 152.85/56.71; "Hottest: May at 27.2°C")
- `lessons/01/slides.html:210-250` and `lessons/01/textbook.html:256-298` (a `humidity_pct` Int64 column, plus "Jan 26.0 194.7", "Feb 26.5 63.4")
- `lessons/01/textbook.html:386-414, 390-395, 499-502` (CV ≈ 37%)
- `lessons/01/notes.html:325-336, 412, 422-424`
- `textbook.md:279, 349-359, 366, 401-404, 417-433, 438-440, 469, 493-497, 538-545, 563`. This is a third, different set of fake values: January rain 242.4, hottest month 28.7, wettest December 258.8, driest 112.5, temperature std "about 1°C", and the conclusion "December is both the coldest and the wettest month" at line 497.

**Evidence:** Actual `df.describe()` on the file:
- temperature: mean 27.466667, std 0.624257, min 26.5, max 28.3
- rainfall (i64): mean 171.75, std 40.10, min 108, max 254
- month min/max: April/September

The rows are `January 26.5 167`, `February 27.1 108`, … The hottest month is May at 28.3. There is no humidity column. The textbook's own head output shows "Apr 27.5", which exceeds its claimed max of 27.2.

The notes say: "Min: 26.0 in April. Max: 27.2 in May" and "describe() output where max was Sep 27.2 — different tie-breaking; both are correct."

**Problem:**
- Every number a student checks in the first lesson disagrees with their screen.
- The notes teach that `describe()`'s min/max row for `month` identifies the month of extreme temperature. It is the lexicographic min/max of the month strings, unrelated to temperature. "Tie-breaking" is a false explanation.

**Fix:**
- Regenerate all Lesson 1.1 outputs from the real file: mean 27.47, std 0.62, CV ≈ 23%, hottest May 28.3 °C, wettest November 254 mm.
- Remove the humidity column from the diagrams.
- Replace the tie-breaking note with: "for a string column, describe() min/max are alphabetical, not related to other columns".

### [BLOCKING] The "HDB Flash Crash" opening case is fabricated, presented as real, self-contradictory, and absent from the course dataset

**File:**
- `deck.html:192-264` (notes at 219: "This is a real Singapore case")
- `deck.html:2131`
- `speaker-notes.md:73-108, 176, 943`
- `textbook.md:92-100, 2464, 3803`

**Evidence:**
- The deck version: Oct 2023, prices up "10 consecutive quarters", medians in three estates "dropped 8-12% in a single month", "bimodal", only 3-room and 4-room affected, root cause "subsidised transfers entering the public dataset without classification".
- The textbook version: "In 2023", a dip "roughly thirty percent below", "Nobody noticed for three weeks", "a batch of subsidised intra-family transfers… accidentally classified as open-market resales".
- The deck claims "you will be able to load this exact dataset, filter for those estates, and identify the anomaly yourself" (211).
- The course dataset is synthetic and flat. Queenstown monthly medians for 2023 range 0.999M to 1.106M with no dip (e.g. 2023-09 998,711; 2023-10 1,060,499; 2023-11 1,019,113). Annual medians are ~840–853k for every year.
- There is no public record of such an incident in HDB's published resale data.

**Problem:** The module's motivating story is presented as a real event with a specific cause and impact on "YOUR property value" (`speaker-notes.md:99-105`). The two tellings contradict each other, and the promised hands-on discovery is impossible with the shipped data. This is false knowledge about a real public dataset and agency.

**Fix:** Either reframe it explicitly as a hypothetical/illustrative scenario (and drop "real case", "this exact dataset", and the causal claims), or replace it with a documented real case. If the exercise is to be kept, inject the anomaly into the synthetic dataset and say that it is synthetic.

### [BLOCKING] ex_8 capstone uses column names that don't exist in the taxi data, so cleaning, PreprocessingPipeline and most charts are silently skipped

**File:** `solutions/ex_8.py:154, 163, 332, 353, 404, 463, 509, 535, 564, 632, 684, 697, 757`. The same guards are in `local/ex_8.py`.

**Evidence:**
- The real schema has `fare_sgd`, `distance_km`, one lat/lng pair, and no `trip_duration_sec`.
- The guards `if "fare" in cols`, `if "trip_duration_sec" in cols` and `len(lat_cols) >= 2` are all False.
- The script prints "'fare' column not found — skipping PreprocessingPipeline". Checkpoint 8 still passes because its asserts sit under `if result is not None`.
- Only `ex8_hourly_volume.html` is produced.
- What happens to the planted dirt:
  - left in place: 1,000 negative `fare_sgd`, 500 `passengers <= 0`, 500 future-dated trips, 15 `payment_type` spellings, 2,500 null `pickup_zone`
  - never detected: 250 duplicate `trip_id`s (not full-row duplicates)

**Problem:**
- The capstone never exercises PreprocessingPipeline, which spec 1.8 requires.
- It produces one chart where the spec requires at least 3.
- Most `____` blanks in local/ex_8.py sit in dead branches, so a student who leaves them blank still passes every checkpoint.

**Fix:**
- Rewrite against the real schema (`fare_sgd`; duration from `dropoff_datetime - pickup_datetime`; `distance_km`).
- Add cleaning for passengers, future dates, payment_type and duplicate trip_id.
- Remove the silent `if col in cols` guards.
- Make Checkpoints 5, 8 and 9 assert unconditionally (`result is not None`, `(fare_sgd > 0).all()`, `len(viz_files) >= 3`).

### [BLOCKING] Weekday encoding is taught wrong: `dt.weekday()` is ISO 1=Mon … 7=Sun, but ex_8 and the textbook treat it as 0-based

**File:**
- `solutions/ex_8.py:437, 455-457`; same in `local/ex_8.py`
- `textbook.md:3314` ("0–6 in Polars (0 is Monday, 6 is Sunday)") and `textbook.md:3526-3528`

**Evidence:**
- ex_8 uses `(dt.weekday() >= 5).alias("is_weekend")` and `.when(pl.col("day_of_week") == 4).then(pl.lit("friday"))`.
- Polars 1.41.2 on Mon..Sun returns `[1,2,3,4,5,6,7]`, and `pl.Series([date(2024,1,1), date(2024,1,7)]).dt.weekday()` returns `[1, 7]`.

**Problem:** "Weekend" includes Friday, and "friday" is actually Thursday. The weekday-vs-Friday-vs-weekend chart is mislabelled, and the textbook states the wrong encoding as fact.

**Fix:** Use `>= 6` for the weekend and `== 5` for Friday. Correct textbook.md:3314 to "1–7 (Monday = 1, Sunday = 7)". Add a comment on ISO numbering.

### [BLOCKING] textbook.md Lesson 1.8 worked example is written for taxi columns that don't exist, and calls a missing ModelVisualizer method

**File:** `textbook.md:3434, 3487-3495, 3540-3570, 3580-3609, 3624-3627, 3721-3722`

**Evidence:**
- The real columns are `fare_sgd`, `distance_km` and a single `pickup_latitude/longitude` pair. There is no `fare`, no `trip_duration_sec`, and no dropoff coordinates.
- `taxi.select([..., "fare"])` raises ColumnNotFoundError. Step 5 `pipeline.setup(data=pipeline_df, target="fare")` therefore crashes.
- `hasattr(ModelVisualizer, "feature_distribution")` returns False, so Step 6 raises AttributeError.
- The fare and duration filters (3487, 3493), the haversine/speed block (3540, which needs two lat columns) and `fare_per_km` (3567) are silently skipped by their guards.
- Drill 2 (3721) uses `pickup_lat`/`pickup_lng` and an un-imported `math`.
- Line 3434 tells students to look for negative fares and bad durations under names that do not exist. `fare_sgd` min is −49.97.

**Problem:** The capstone walkthrough in the module textbook cannot be followed end to end.

**Fix:**
- Use `fare_sgd` as the filter and the target.
- Derive duration from `dropoff_datetime - pickup_datetime`.
- Use `distance_km` instead of the two-point haversine.
- Replace `feature_distribution` with `viz.histogram(data=taxi_clean, column="fare_sgd", …)`.
- Fix the Drill 2 column names and import `math`.

### [BLOCKING] Assessment Task 2 grader fails correct answers because it compares rows by position under a non-unique sort key

**File:** `assessment/task_2/grader.py:145-152` (with `solution.py:87` `.sort(["sale_year","town"])`)

**Evidence:**
- The reference was wrapped to only reverse input order before the required sort (`solve().reverse().sort(["sale_year","town"], maintain_order=True)`). Every value was identical and the output was correctly sorted. The grader gave 5/10, failing `storey_midpoint_correct`, `remaining_lease_correct`, `flat_type_rooms_correct`, `flat_age_correct` and `price_per_sqm_correct`.
- Each `(sale_year, town)` pair has hundreds of rows, so row order within ties is arbitrary.

**Problem:** Any valid pipeline that changes row order before the final sort (concat of lease formats, lazy execution) loses half the marks.

**Fix:** Sort both frames by all output columns (or join on a row id) before comparing values, or make the required sort key unique in both `problem.md` and the grader.

### [BLOCKING] Assessment Task 4 reference keeps a flagged duplicate row, so the correct cleaning step fails

**File:**
- `assessment/task_4/grader.py:130` (`row_count_101`)
- `assessment/task_4/solution.py:26-81`
- `assessment/task_4/problem.md:25, 47`

**Evidence:**
- The quarterly slice contains the exact duplicate row `2019-2 | 0.65 | 2.32 | 1.04 | 19.9 | 175.0 | 9,657,346` twice.
- DataExplorer flags it (`{'type': 'duplicates', 'count': 2}`) on both the raw and the reference-cleaned frame.
- The reference plus `.unique()` gives `row_count_101: false` and stops at 3/4.

**Problem:** The task says to find issues with DataExplorer and fix them. A student who removes the flagged duplicate is penalised, and the answer key teaches that a flagged duplicate belongs in "clean" data.

**Fix:** Add deduplication to `problem.md`, `solution._clean` and `grader._reference_clean`, expect 100 rows, and rename the check.

---

## MAJOR

### [MAJOR] `speaker-notes.md` (the "Master Speaker Notes" linked from index.html) belongs to a different, older deck

**File:** `modules/mlfp01/speaker-notes.md` (entire file); linked from `index.html:178` as "Master Speaker Notes (markdown, all 8 lessons)"

**Evidence:**
- The notes describe 82 slides; `deck.html` has 78 `<section>`s with different titles.
- The notes include about 38 slides of M2-level statistics that are not in the deck, e.g. "Slide 26: The Exponential Family", "Slide 31: Fisher Information", "Slide 33: Information Geometry", "Slide 39: Neyman-Pearson", "Slide 43: BCa Bootstrap".
- They cover slides that don't exist ("Slide 8: This Affected YOUR Property Value", "Slide 56: ConnectionManager: Where Results Live").
- They reference `ex_1_1.py` (line 905), Jupyter as a third format (66, 916), and a fourth M1 engine, ConnectionManager (52, 744-760).
- They title the course "ML Engineering from Foundations to Mastery" (line 12), which is the ASCENT program's title.
- Technical slips inside:
  - line 450: "KL divergence is the local quadratic approximation of this Riemannian distance". This is inverted: the Fisher metric is the second-order approximation of KL.
  - line 530: "The likelihood ratio test is the most powerful test at any significance level". This holds only for simple-vs-simple hypotheses (Neyman–Pearson lemma).

**Problem:** An instructor following the master notes will be out of step with every slide, and will promise theory and engines the module doesn't contain.

**Fix:** Regenerate `speaker-notes.md` from the `<aside class="notes">` blocks in `deck.html` (78 slides), or delete it and point index.html at `lessons/NN/notes.html`.

### [MAJOR] The deck setup slide sends students to exercise-format directories that don't exist (and are banned)

**File:** `deck.html:163-174, 182-184`

**Evidence:**
- The slide's table lists `Jupyter | modules/mlfp01/notebooks/` and `Google Colab | modules/mlfp01/colab/`, plus "Three Exercise Formats… All three formats produce identical results".
- The actual directories are `local/`, `colab-selfcontained/` and `colab-selfcontained-solutions/`.
- `.claude/rules/two-format.md` blocks `notebooks/` and `colab/`.

**Problem:** Students are pointed at non-existent paths, and the slide teaches a three-format model the course abolished.

**Fix:** Change the slide to two formats: `modules/mlfp01/local/` (VS Code) and `modules/mlfp01/colab-selfcontained/` (Colab). Update the notes the same way.

### [MAJOR] Deck and Lesson 1.6 claim Plotly requires pandas and teach `.to_pandas()`, contradicting the Polars-only mandate

**File:**
- `deck.html:2262, 2269, 2278, 2285, 2295, 2359, 2376`
- `lessons/06/slides.html:357-361, 366-368`
- `lessons/06/textbook.html:406-414`

**Evidence:**
- `deck.html:2295`: "Plotly requires pandas. Use `.to_pandas()`…".
- `lessons/06/slides.html:366`: "Plotly Express needs pandas for its color= grouping".
- With the installed plotly 6.8.0, `px.bar(pl.DataFrame(...), x="town", y="count", color="flat_type")` works on the Polars frame directly (2 traces). `px.scatter(pl.DataFrame)` also works.

**Problem:** The claim is false, and it introduces pandas into material for a course whose CLAUDE.md directive 2 is "No pandas in any exercise". `solutions/ex_6.py` does not use pandas, so the slides also contradict the exercise.

**Fix:** Pass Polars frames straight to `px.*`. Compute the correlation with `df.select(...).corr()` and plot it with `px.imshow(corr.to_numpy(), x=cols, y=cols)`. Delete the "requires pandas" notes.

### [MAJOR] Lesson 1.5 lazy-frame examples fail

**File:**
- `lessons/05/slides.html:306-313` and `lessons/05/textbook.html:272-279` (`pl.scan_parquet("hdb_resale.parquet").filter(pl.col("year") >= 2020)`)
- `lessons/05/textbook.html:446-447` (`pl.scan_parquet(loader.path("mlfp01", "hdb_resale.parquet"))`)

**Evidence:**
- The raw parquet has no `year` column. The relative path `hdb_resale.parquet` does not exist from the repo root (the file is under `data/mlfp01/`).
- `MLFPDataLoader` has no `path` attribute. Its public methods are `load, load_raw, load_hf, list_files, assessment, mlfp01…mlfp06`.

**Problem:** The only lazy-frame code in the lesson raises FileNotFoundError, ColumnNotFoundError or AttributeError.

**Fix:** Use `pl.scan_parquet(MLFPDataLoader().load_raw("mlfp01", "hdb_resale.parquet"))`. Add `.with_columns(pl.col("month").str.slice(0,4).cast(pl.Int32).alias("year"))` before the filter.

### [MAJOR] Lesson 1.7 textbook worked example cannot run, and its AlertConfig comments misstate the defaults

**File:**
- `lessons/07/textbook.html:389-402` (`monthly_spine.join(employment, on="date", …)` and `for c in employment.columns if c != "date"`)
- `lessons/07/textbook.html:305-312` (`await` at module level)
- `lessons/07/textbook.html:238-244`

**Evidence:**
- `sg_employment.csv` has the schema `quarter, employment_rate, unemployment_rate, median_income, labour_force`. There is no `date` column, so the join raises ColumnNotFoundError.
- The try/except sample uses `await explorer.profile(df)` outside an `async def`, which is a SyntaxError in a script.
- The comments read `high_correlation_threshold=0.95  # raise from 0.80 default` and `high_null_pct_threshold=0.10  # tighten from 0.15 default`. The installed defaults are 0.9 and 0.05, and moving 0.05 to 0.10 relaxes the null alert; it does not tighten it.

**Problem:** The lesson's merge step fails, and the threshold guidance is wrong in both value and direction.

**Fix:** Parse `quarter` into a quarter-start date before the spine join, as `solutions/ex_7.py` does. Wrap the try/except in an `async def`, or use `asyncio.run`. Correct the comments ("raise from 0.90 default", "relax from 0.05 default").

### [MAJOR] textbook.md's DataExplorer "full API" reference lists attributes and keys that do not exist

**File:** `textbook.md:299, 2824, 2905-2913, 2946, 2951, 2954, 2961`

**Evidence:**
- `DataProfile` has `correlation_matrix` and `spearman_matrix`; there is no `pearson_matrix`.
- `ColumnProfile` has `min_val`/`max_val`, not `min`/`max`.
- `inferred_type` values in the source are `numeric|categorical|boolean|constant|id|text`. There is no "temporal".
- `compare()` returns `shape_comparison={"rows_a","rows_b","cols_a","cols_b"}`, and `column_deltas` items use `null_pct_delta`, not `null_delta`.
- `_generate_alerts` emits only "info" and "warning" severities, with no recommendation string.

**Problem:** `profile.pearson_matrix` and `col.min`/`col.max` raise AttributeError, and the documented return shapes are wrong.

**Fix:**
- Rename to `correlation_matrix`, `min_val`/`max_val` and `null_pct_delta`.
- Correct the `shape_comparison` keys and the inferred-type list.
- Remove the "error" severity and the recommendation field.

### [MAJOR] textbook.md Lesson 1.2–1.3 expected outputs describe a ~487K-row HDB file; the course file has 50,150 rows

**File:** `textbook.md:17, 212, 312, 633, 696, 726, 862-871, 897-899, 920, 1022-1032, 1141, 1321, 1453, 1502-1511, 1546-1561, 1629, 1986, 2017, 2618`

**Evidence:**
- The textbook prints `Shape: (487_293, 11)` with float prices. The actual shape is (50150, 11) with i64 `resale_price`.
- Counts printed vs actual:
  - AMK: 28,847 printed vs 2,486 actual
  - 4 ROOM: 199,254 vs 20,299
  - S$300–500k: 188,912 vs 2,885
  - AMK 4-room ≤500k: 7,214 vs 0
- Price tiers: the actual split is luxury 35,157 / premium 11,664 / mid_range 2,160 / budget 1,169. The textbook says "mid-range dominates… luxury… about 10%" (1032).
- 27.5% of sales are ≥ S$1M, which the textbook calls "the tail of the tail".
- The actual top towns by median are TOA PAYOH, KALLANG/WHAMPOA and BUKIT TIMAH, with BOON LAY last. Line 1453 says Bukit Timah/Central Area top and Sembawang/Choa Chu Kang last.

**Problem:** Every Lesson 1.2–1.3 "Expected output", and the interpretation built on it, contradicts what students will see.

**Fix:** Regenerate all expected outputs from `data/mlfp01/hdb_resale.parquet`. Use ~50,000 rows and 27 towns.

### [MAJOR] textbook.md calls the HDB file "real data… published openly by the HDB", then promises trends the (synthetic) data does not have

**File:** `textbook.md:840` (contradicted by its own line 3985, which says the datasets "are synthetic extensions"), `1650, 2281, 2612, 2634, 2696, 2771, 2775, 2783`

**Evidence:**
- The median price by year is flat (2015 850,015; 2018 853,241; 2020 839,409; 2024 849,174), and corr(year, price) = −0.0008.
- Prices range from a min of 10 to a max of 9,000,000.
- corr(year, lease_commence_date) = 0.002. The textbook says "strongly positively correlated… 0.3–0.5".
- corr(area, price) = 0.469, so r² ≈ 0.22. The textbook says area "explains 40–50%".
- `price_per_sqm` skew is 22.1 vs 11.4 for price. The textbook says it is "less skewed".
- The median price is $849k. The textbook says "clustered in $350k–$600k".
- Lines 1650 and 2281 promise a "clear upward trend ~3–5% annually" and YoY of "+10% to +20%" for 2022–23.

**Problem:** Students are told the data is real and shows trends that their output flatly contradicts.

**Fix:** Pick one:
- State that the dataset is synthetic and rewrite the interpretations from actual output.
- Regenerate the dataset with realistic trends.

### [MAJOR] textbook.md describes the `RdBu_r` correlation colours backwards

**File:** `textbook.md:2510` ("blue for positive, red for negative"), `2696` (strong positive = "dark blue"). `lessons/06/notes.html:226-227` has the same reversal ("red for negative, blue for positive").

**Evidence:** `px.colors.diverging.RdBu_r` runs from `rgb(5,48,97)` (blue, at −1) to `rgb(103,0,31)` (red, at +1). The deck's notes (`deck.html:2389`) state it correctly: red = positive.

**Problem:** Students will read the sign of every correlation heatmap backwards, and the module's materials disagree with each other.

**Fix:** Change these lines to "red = positive, blue = negative" and "dark red".

### [MAJOR] textbook.md omits spec-required Lesson 1.6 and 1.8 topics

**File:** `textbook.md` Lesson 1.6 (2441-2734) and Lesson 1.8 (3245-3739)

**Evidence:**
- `grep -iE "requests|httpx|REST|GET|POST"` finds no API content; OneMap appears only in the reading list (3987).
- There is no project-structure or multi-file section (`grep "project structure|main.py"` finds nothing).
- "Stacked bar" appears only in a table row (2453). The "Six HDB Charts" worked example builds five charts, with Step 6 just saving files.
- There is no 100%-stacked / Likert example.
- The Gestalt list (2518-2524) omits enclosure.

**Problem:**
- Spec 1.8 requires "REST APIs: GET, POST, JSON responses, query parameters" and "Project structure: modules, imports".
- Spec 1.6 requires stacked and 100%-stacked bars and the six Gestalt principles.
- The deck and lesson pages cover these, but the master textbook does not.

**Fix:**
- Add a REST/JSON extraction section and a project-structure section to Lesson 1.8.
- Add the stacked-bar sixth chart and a 100%-stacked example to Lesson 1.6.
- Add enclosure to the Gestalt list.

### [MAJOR] ex_4 treats the distance between MRT stations as each flat's distance to the MRT

**File:** `solutions/ex_4.py:34, 211-222, 397-399, 476-493, 808-815`; identical lines in `local/ex_4.py`

**Evidence:**
- `mrt_stations.parquet` is one row per station. Row `Bukit Batok (1.3485,103.7496) nearest_mrt=Bukit Gombak distance_to_mrt_km=1.145`. The haversine distance from Bukit Batok to Bukit Gombak station (1.3586,103.7516) is 1.145 km, an exact match.
- The exercise takes the per-town minimum of this station-to-station gap. It joins that onto every sale as "distance to MRT", grades flats "walkable/near/far" on 5/10-minute-walk thresholds, and interprets "closer MRT → higher price".

**Problem:** This is the core "distance-to-amenity feature" required by spec 1.4. It is mislabelled, and the walkability interpretation is wrong. `nearest_mrt` names a station in a different town.

**Fix:** Compute a real feature with the exercise's own `haversine_km` (town centroid or block to nearest station), or rename the column `station_spacing_km` and remove the walkability framing.

### [MAJOR] ex_4's case-sensitivity demo prints the opposite of what it teaches

**File:** `solutions/ex_4.py:180-188` (`local/ex_4.py:180-188`)

**Evidence:** The demo compares `hdb["town"][0]` = `'BUKIT PANJANG'` with `mrt_stations["town"][0]` = `'Jurong East'`. So `hdb_sample_town == mrt_sample_town.upper()` is False, and it prints "Match after .upper()? False".

**Problem:** The demo is meant to prove that the towns differ only in case.

**Fix:** Compare the same town on both sides, e.g. `t="BISHAN"` and the MRT row filtered by `str.to_uppercase()==t`.

### [MAJOR] ex_5 (and the Lesson 1.5 pages) compute YoY and rolling windows by row, not by calendar month

**File:**
- `solutions/ex_5.py:121-135, 263-264, 706`; `local/ex_5.py:265-270`
- The same teaching is in `lessons/05/slides.html:221-229` and `lessons/05/textbook.html:225, 410`

**Evidence:**
- The HDB town × month grid has 3,236 of 3,240 cells. BUKIT TIMAH is missing 2015-10, 2017-02 and 2022-07; CENTRAL AREA is missing 2015-08.
- `shift(12).over("town")` therefore misaligns 40 rows.
- Two misaligned rows appear in the exercise's printed "Top 10 YoY Gains": CENTRAL AREA 2016-03 vs 2015-02, and BUKIT TIMAH 2022-09 vs 2021-08.

**Problem:** Spec 1.5's assessment criterion is "YoY computed with proper time alignment". The material teaches `shift(12)` as "same month last year", which only holds with no gaps.

**Fix:** Join each town to a complete town × month grid before windowing, or compute the 12-month-ago value via a join on `transaction_date.dt.offset_by("-12mo")`. Add a sentence on why gaps break row-based shifts.

### [MAJOR] ex_6 and ex_5 interpretations state data facts the bundled HDB data contradicts

**File:**
- `solutions/ex_6.py:34, 130, 182, 208, 407, 654`; `solutions/ex_5.py:34, 539`; and the local copies (e.g. `local/ex_6.py:411, 659`)
- Same class: `solutions/ex_2.py:559-561, 718-721`; `solutions/ex_3.py:350, 541-543`; `solutions/ex_1.py:497-499, 568-569, 688-692`

**Evidence:**
- HDB: the median is S$849k and 58% of sales are above S$800k (P10–P90 is 538k–1.22M). Annual medians are 839–853k, and corr(price, year) is −0.001. There are no `1 ROOM` rows, and `MULTI-GENERATION` (1.49M) is above `EXECUTIVE`.
- The comments say:
  - "most transactions cluster in S$350k-600k"
  - "At 100 sqm, prices range from S$400k to S$900k+" (actual P5–P95 at ~100 sqm is 695k–1.03M)
  - "resale_price vs year: positive (prices have risen over time)"
  - "Rising medians confirm price appreciation"
  - "clear price ladder from 1-room to Executive"
- Weather (ex_1): the comment says "Rainfall CV ≈ 30%" (actual 23.4%) and "wettest are Nov-Jan" (January's 167 mm is below the mean). It also calls r = −0.499 "weak", although the script's own threshold labels it moderate.
- The same wrong price narrative ("350k–600k", "mean > median", "area explains 40-50%") appears in `lessons/06/slides.html:244-254`, `lessons/06/textbook.html:296-298` and `lessons/06/notes.html:155-178`.

**Problem:** The printed output contradicts the interpretation students are told to draw.

**Fix:** Rewrite the interpretations from actual output, or replace the synthetic HDB file with real HDB resale data that matches the stated provenance (and then re-verify every claim).

### [MAJOR] ex_6 chart cheat-sheet names a non-existent ModelVisualizer method and misdescribes two others; repurposed methods produce mislabelled axes

**File:**
- `solutions/ex_6.py:102-105` and `local/ex_6.py:103-106` (cheat sheet)
- `solutions/ex_6.py:476-483, 506-541` (`training_history` as a year line)
- `solutions/ex_6.py:290, 302, 315, 348, 685` (`metric_comparison` as bars)
- `solutions/ex_8.py:732-736`
- `lessons/06/slides.html:297-305` ("feature_importance = horizontal bar" comment on a `metric_comparison` call)
- `lessons/06/textbook.html:322-324`

**Evidence:**
- `dir(ModelVisualizer)` has no `feature_distribution`.
- `confusion_matrix(y_true, y_pred, labels)` cannot draw an arbitrary 2-D grid, and `feature_importance` needs a fitted model.
- The installed `training_history` plots `x=range(1, n+1)` with the x-axis title "Epoch" (checked: `fig.layout.xaxis.title.text == "Epoch"`).
- `metric_comparison` draws vertical grouped bars titled "Model"/"Score" (orientation None, barmode "group").

**Problem:**
- The guide tells students to call a method that raises AttributeError.
- The "Year" line shows 1…10 instead of 2015…2024 and labels it "Epoch".
- The hourly chart shows 1…24 for hours 0…23.
- The bar charts are labelled Model/Score.
- Lesson 1.6's criterion is "No misleading axes".

**Fix:** Remove `feature_distribution`. Map the heatmap to `go.Heatmap` and categories to `metric_comparison`. After each repurposed call, set real x values and axis titles (`fig.update_traces(x=years)`, `fig.update_layout(xaxis_title=…, yaxis_title=…)`), or use `go.Scatter`/`go.Bar` directly.

### [MAJOR] Spec-required techniques are missing from the exercises: seasonality (1.5), REST extraction (1.8), original-vs-cleaned compare() (1.7/1.8), and the async wrapper (1.7)

**File:** `solutions/ex_5.py:12`; `solutions/ex_7.py:441-457`; `solutions/ex_8.py` (whole file); `solutions/ex_1.py` (Tasks 5–6)

**Evidence:**
- `grep -i season solutions/ex_5.py` matches only the learning-objective line.
- ex_8 has no `requests`/`httpx`/`urllib`; its data comes from `loader.load`.
- ex_7 Task 7 compares pre-COVID with COVID-era data, not original vs cleaned. ex_8 compares alert counts only and never calls `compare()`.
- ex_7/ex_8 call `asyncio.run`/`await` directly.
- ex_1 has no "answer 3 questions from describe()" task (`grep -i question` finds none).
- ex_4 never runs an outer join (`how="full"` appears only in printed text).

**Problem:**
- Spec 1.5 "Identify trends and seasonality", spec 1.8 "Extract data from REST APIs" / "load from API", spec 1.7 "Compare original vs cleaned version", spec 1.1's three describe() questions, and spec 1.4's "left, inner, outer" are taught in slides but not exercised.
- `index.html`/README advertise them.

**Fix:**
- Add a month-of-year seasonality task to ex_5.
- Add a REST GET step to ex_8 (OneMap or data.gov.sg, with an offline fallback).
- Add `compare(raw, cleaned)` to ex_7 or ex_8.
- Add three explicit describe() questions to ex_1.
- Add a small `how="full"` join to ex_4.
- Once `shared.run_profile` is fixed, use it in ex_7/ex_8.

### [MAJOR] Assessment Task 1 grader gives full marks for a wrong payment mapping and the wrong dedup rule

**File:** `assessment/task_1/grader.py:72, 111-115, 142`

**Evidence:**
- A copy of the solution with Grab/NETS labels swapped and dedup keeping the lowest fare scored `"passed": true` (10/10).
- The grader checks only the set of payment labels and the uniqueness/count of trip_id. All 250 duplicate trip_ids have differing fares.

**Problem:** `problem.md` steps 3 and 6 (exact mapping; "keep highest fare_sgd") are never checked row-wise.

**Fix:** Build the reference frame in the grader. Join on `trip_id` and compare `payment_type`, `fare_sgd`, `dropoff_datetime`, `trip_duration_min` and `implied_speed_kmh`.

### [MAJOR] Assessment Task 4 grader trusts the student's self-reported alert count and never checks `trade_balance_sgd_bn` imputation values

**File:** `assessment/task_4/grader.py:139-149, 165`

**Evidence:**
- `c["cleaning_reduced_alerts"] = out["clean_alert_count"] < out["raw_alert_count"]` uses the student's number. The grader never re-profiles `out["cleaned"]`.
- For `trade_balance_sgd_bn`, only "no nulls" is checked.

**Problem:** Hard-coding `"clean_alert_count": 0` passes the central "prove the fix worked" requirement. Mean or zero imputation passes where `problem.md:31-32` requires the median.

**Fix:** Recompute alerts on `out["cleaned"]` in the grader and require equality with the reported count. Add a `_close` value check on `trade_balance_sgd_bn`.

### [MAJOR] The M1 assessment does not cover visualisation or ETL/PreprocessingPipeline, and every task is fully dictated

**File:** `assessment/README.md:13-18`; `specs/module-1.md` (End of Module Assessment); `.claude/rules/domain-integrity.md`

**Evidence:**
- The spec requires the end-of-module assessment to cover "data types, Polars operations, viz principles, ETL concepts".
- All four tasks are Polars cleaning/window tasks plus DataExplorer. None touches chart selection or ModelVisualizer (1.6), REST/API extraction or PreprocessingPipeline (1.8).
- Each `problem.md` spells out every threshold, mapping and defect (e.g. `task_1/problem.md:29-34`, `task_2/problem.md:18-21`), and the graders score output values only.

**Problem:**
- Two of five module objectives are never assessed.
- domain-integrity.md forbids answers "a language model could generate without running the code" and grading "without assessing the decision behind the code".

**Fix:**
- Add a task (or a part of Task 4) on chart-type choice with ModelVisualizer and on PreprocessingPipeline/ETL.
- Stop listing the defects. Require students to discover them with DataExplorer and justify each fix in a short rationale graded against their own profile output.

---

## MINOR

### [MINOR] HDB row count stated as ~500,000 or "15 million" throughout (actual 50,150)

**File:**
- `deck.html:32, 859, 1391`
- `speaker-notes.md:15, 643`
- `lessons/02/slides.html:32, 39-50, 254-266`
- `lessons/02/textbook.html:36, 75-77, 278-293, 361, 371` ("Shape: (500123, 11)")
- `lessons/02/notes.html:35, 46, 138, 244-246, 296, 344`
- `lessons/03/slides.html:126`; `lessons/03/textbook.html:250`
- `lessons/04/notes.html:56`; `lessons/04/slides.html:61`; `lessons/04/textbook.html:71, 94, 427-428`
- `lessons/05/notes.html:205-206`; `lessons/05/slides.html:238`; `lessons/05/textbook.html:284, 308`
- `lessons/06/notes.html:172`; `lessons/06/slides.html:301`; `lessons/06/textbook.html:305, 402`
- `solutions/ex_1.py:730`, `ex_2.py:35`, `ex_3.py:34`, `ex_4.py:371`, `ex_6.py:34`, `ex_5.py:34`, and the local copies

**Evidence:** `hdb_resale.parquet` is (50150, 11). The notes for `lessons/01/notes.html:473` and `lessons/08/notes.html:256` correctly say 50,000.

**Fix:** Use "~50,000 rows" (and "50k × 150 = 7.5 million" in ex_4). Remove the "15 million rows" claims.

### [MINOR] Town and flat-type counts are stated as 26 towns / 7 flat types (data has 27 / 6)

**File:**
- `deck.html:1767` (`# 26 rows`) and `2236`
- `lessons/03/notes.html:45, 144-145, 165, 182-186` ("26 towns × 7 flat types ≈ 180")
- `lessons/03/textbook.html:49-53, 225, 281`

**Evidence:** `hdb["town"].n_unique()` is 27; the flat types are EXECUTIVE, 2 ROOM, 3 ROOM, 4 ROOM, MULTI-GENERATION, 5 ROOM.

**Fix:** Use 27 towns and 6 flat types, or say "~27".

### [MINOR] Lesson 1.3 says Polars auto-names duplicate aggregations `resale_price_2`; it actually raises DuplicateError

**File:** `lessons/03/notes.html:169-172`; `lessons/03/textbook.html:476-479`

**Evidence:** `df.group_by('t').agg(pl.col('p').mean(), pl.col('p').median())` raises `DuplicateError column with name 'p' has more than one occurrence`.

**Fix:** Say "without `.alias`, two aggregations on the same column collide and Polars raises a DuplicateError".

### [MINOR] The deck aggregation table lists `pl.mean()` … `pl.count()` as no-argument functions and uses deprecated `pl.count()`

**File:** `deck.html:1168-1177, 1189, 1340-1342, 2372`; `lessons/08/slides.html:404`; `lessons/08/textbook.html:391`

**Evidence:**
- `pl.mean()` gives `TypeError Col.__call__() missing 1 required positional argument: 'name'`.
- `pl.count()` gives `DeprecationWarning: pl.count() is deprecated. Please use pl.len() instead.`

**Fix:** List the expression forms (`pl.col("x").mean()`, …) and use `pl.len()` for row counts.

### [MINOR] Deck "Looping Over Districts" mislabels its output

**File:** `deck.html:1253-1268`

**Evidence:**
- `towns = …unique()…; for town in towns[:5]:  # top 5` takes the first 5 in arbitrary order, not the top 5.
- `avg = summary["avg_price"][0]` is the average of the most expensive flat type in the town (`district_summary` groups by `flat_type` and sorts descending), but it is printed as the town's average.

**Fix:** Use `towns = df["town"].unique().sort().to_list()[:5]` and compute the town mean with `df.filter(...)["resale_price"].mean()`, or relabel the output.

### [MINOR] Lesson 1.2 slides select a column the slides never create

**File:** `lessons/02/slides.html:339-347` (`.select("transaction_date", …)`)

**Evidence:** The slides' `with_columns` (268-272) adds only `price_per_sqm` and `year`. The textbook adds `transaction_date` at `lessons/02/textbook.html:345`, but the slides do not, so the chain raises ColumnNotFoundError.

**Fix:** Add `pl.col("month").str.to_date("%Y-%m").alias("transaction_date")` to the slide's `with_columns`.

### [MINOR] The deck's join guidance contradicts itself

**File:** `deck.html:1570` ("Inner join is the safest. Left join is the most common.") vs `deck.html:1714` ("left join is the safe default") and `lessons/04/notes.html:157-159` ("Inner is a gotcha — you lose rows silently")

**Fix:** Change 1570 to "Left join is the safe default for enrichment; inner join silently drops unmatched rows."

### [MINOR] index.html dataset table is inaccurate, and its Course Home link is broken

**File:** `modules/mlfp01/index.html:36, 74-76`

**Evidence:**
- `economic_indicators.csv` is listed as "Used in 1.7", but `solutions/ex_7.py:56-58` loads `sg_cpi.csv`, `sg_employment.csv` and `sg_fx_rates.csv`. `economic_indicators.csv` is used only by `assessment/task_4`.
- "Primary schools ~350 rows" vs 242 actual.
- `href="../index.html"` points to `modules/index.html`, which does not exist (link check over the index, deck and 24 lesson pages found this as the only broken relative link).

**Fix:** List the three files actually used in 1.7 and note economic_indicators as the assessment dataset. Change the schools row to ~240. Point Course Home at an existing page or add `modules/index.html`.

### [MINOR] README exercise 8 title differs from the spec and every other page

**File:** `modules/mlfp01/README.md:31` ("Data Cleaning and End-to-End Project")

**Evidence:** `specs/module-1.md` Lesson 1.8, `README.md:18`, `index.html:165` and `deck.html:2736` all say "Data Pipelines and End-to-End Project".

**Fix:** Rename the row.

### [MINOR] ex_2 / ex_3 local scaffolds diverge from their solutions (checkpoints and reflection not preserved verbatim)

**File:**
- `local/ex_2.py:711-716, 727-735` vs `solutions/ex_2.py:756-764, 775-788`
- `local/ex_3.py` around 422 and 759-763 vs `solutions/ex_3.py:477-488, 854-882`

**Evidence:**
- ex_2 local drops `recent_premium.height < hdb.height`, the descending-sort and `cv_pct` asserts, and substitutes other checks. Its reflection is rewritten and claims `.dt.truncate()`, which is never used.
- ex_3 local drops the `position_summary`/`position_counts` blocks, the `assert position_counts.height > 0`, and one reflection line.

**Problem:** exercise-standards.md says checkpoints are never stripped and reflection is kept verbatim.

**Fix:** Copy the solution's checkpoint 10 and reflection verbatim into local, and restore the dropped blocks.

### [MINOR] Small comment errors in ex_2 / ex_3 / ex_5 / ex_7 / ex_8

**File and evidence:**
- `solutions/ex_2.py:532-536`:
  - It says "We don't have build year directly", but `lease_commence_date` exists.
  - The code buckets by transaction year, not "first transaction per block".
- `ex_2.py:611`: says "first 3 towns", but the loop breaks at 5.
- `ex_2.py:627`: says the 50th-percentile row is "not the same as the median", but it is the median (849,126 vs 849,124).
- `solutions/ex_3.py:666-669`: "Average Quarterly Volume" is a pooled `pl.len()` total.
- `solutions/ex_5.py:790` (local 799): lists `cum_mean`, but `hasattr(pl.Expr,'cum_mean')` is False.
- `solutions/ex_7.py:37`: mentions "trade-flow columns", but no such columns exist.
- `solutions/ex_8.py:250`: `alert.get("column","N/A")` prints N/A for high_correlation alerts, which use the key `"columns"`.
- `solutions/ex_8.py:772-783`: an "Equal Weight Placeholder" chart with synthetic weights (zero-tolerance Rule 2).
- `ex_8.py:915, 922`: wrong M2 title, and a promise that the taxi data becomes an M2 feature-store entry (no M2 file uses taxi data).

**Fix:** Correct each comment as stated. Drop the placeholder chart. Use `alert.get("column", alert.get("columns"))`.

### [MINOR] ex_1–ex_3 use Python `if/else` before Lesson 1.4 and omit the spec's required forward reference

**File:** `solutions/ex_1.py:648-651, 681-686`; `ex_2.py:258-261, 612-618, 700-717`; `ex_3.py:101-114`

**Evidence:** Spec 1.2 says: "Python if/else is deferred to 1.4. Add forward-reference: 'You are writing expressions… Python also has if/else… Lesson 1.4.'" `grep "Lesson 1.4" solutions/ex_2.py` returns nothing. ex_4 Task 7 then presents if/elif/else as new.

**Fix:** Add the forward reference to ex_2, and either remove the early if/else or re-sequence the spec.

### [MINOR] Assessment Task 3 instructs a deprecated Polars argument

**File:** `assessment/task_3/problem.md:24`; `task_3/starter.py:33` (`min_periods=1`)

**Evidence:** Polars 1.41.2 warns that "`min_periods` for `Expr.rolling_mean` is deprecated… renamed to `min_samples` in version 1.21.0". The solution and grader use `min_samples=1`.

**Fix:** Change both to `min_samples=1`.

### [MINOR] Assessment Task 4 "quarterly median" is ambiguous, and its sync-`asyncio.run` pattern breaks in Colab

**File:**
- `assessment/task_4/problem.md:31-32`; `task_4/starter.py:14, 35-39`; `solution.py:96`
- `assessment/README.md:38`

**Evidence:**
- The reference uses one median over all quarterly rows (inflation 2.11). The per-quarter reading (Q1 1.74, Q2 2.02, Q3 2.11, Q4 2.24) fails `inflation_correct`.
- `asyncio.run` inside `solve()` raises `RuntimeError: asyncio.run() cannot be called from a running event loop` in Colab/Jupyter, which the README says is supported.

**Fix:** Reword to "the single median of the column over all quarterly rows". Route profiling through the fixed sync wrapper, or state that Task 4 must be run as a script.

### [MINOR] textbook.md drill and walkthrough code slips

**File and evidence:**
- `textbook.md:3225`: the Drill 1 answer gives the default `high_correlation_threshold` as 0.80. The installed default is 0.9.
- `3287`: `import kailash_ml.imputation` raises ModuleNotFoundError.
- `3133-3183`: `compare_periods()` has no `return`, so `comparison` in `main()` is None.
- `1992/2041`: Drill 4 uses `pl.col("level")`, but `schools` has `type` (primary/secondary/JC).
- `2569`: says `confusion_matrix` draws a "correlation" heatmap, which contradicts line 2694.

**Fix:**
- Change 0.80 to 0.90.
- Remove the `kailash_ml.imputation` import.
- Add `return comparison`.
- Use `pl.col("type") == "primary"`.
- Drop "correlation" from the `confusion_matrix` row.

### [MINOR] textbook.md factual and internal slips

**File and evidence:**
- `textbook.md:1258`: "Dictionaries are unordered". They are insertion-ordered since Python 3.7.
- `1337`: "7 flat types". The data has 6.
- `1982`: refers to "Slide 77". The deck's slide 77 is the Lesson 1.8 Recap.
- `2412-2413`: `cagr_3y` applies `**(1/3)` to first/last values spanning 2015–2024, about 10 years.
- `3573`: "Six new features" is followed by a list of eight.
- `3778`: gives the Module 2 title as "Feature Engineering and Experiment Design". The spec title is "Statistical Mastery for Machine Learning and AI Success".

**Fix:** Correct each item as stated.

### [MINOR] textbook.md further-reading entries are misattributed or unverifiable

**File:** `textbook.md:3957, 3981`

**Evidence:**
- *Python Polars: The Definitive Guide* (O'Reilly) is by Jeroen Janssens and Thijs Nieuwdorp, not Ritchie Vink.
- "Sadowski, Caitlin, and Yarden Katz. *Data Quality for the Numerate*. O'Reilly (forthcoming)" is not a known publication.

**Fix:** Correct the Polars book authors and remove the unverifiable entry.

---

## Checks with no findings

- **Independence/naming (check 7):** no institutional partners, universities, funding bodies, prior course codes, or commercial-product comparisons of Kailash were found in any M1 file.
  - "Grab" and "NETS" appear only as payment-method values in the Task 1 taxi dataset, which is acceptable as dataset content.
  - Mentions of Excel, Airflow and Dagster are generic tool references in teaching notes, not Kailash positioning.
- **Hardcoded LLM model names (check 8):** none found in M1 code, decks or lessons.
- **Solutions vs local scaffolds:** no `____` left in solutions; no solution code leaked into local; all checkpoints present apart from the ex_2/ex_3 divergences above.
- **Assessment:** each reference solution passes its own grader (Task 1 10/10, Task 2 10/10, Task 3 11/11, Task 4 12/12; 13–31 s each), and the README mark totals (100) match. Unmodified Starter 1 scores 1/10, as it should.

## Coverage summary

**What was checked:**
- **Specs and rules:** `specs/_index.md`, `specs/module-1.md`, `specs/redlines.md`, the mlfp01 section of `specs/exercise-mapping.md`, and the rules `exercise-standards.md`, `independence.md`, `domain-integrity.md`, `two-format.md` and `env-models.md`.
- **Master materials:**
  - `deck.html` (3,179 lines, 78 slides): all code blocks and prose read.
  - `speaker-notes.md` (1,001 lines): read and mapped against the deck.
  - `textbook.md` (4,001 lines): every line read, code verified against the installed API, and every claimed number checked against the data.
  - `README.md` and `index.html`, including the dataset table and links.
- **Lesson pages:** all 24 files `lessons/01-08/{slides,textbook,notes}.html`. All 138 code blocks were extracted and checked against the installed API and real data, and all prose was read. A relative-link check was run over the index, the deck and the 24 lesson pages.
- **Exercises:** 8 solutions and 8 local scaffolds (`solutions/ex_1-8.py`, `local/ex_1-8.py`), diffed pairwise. Every kailash_ml call was checked against kailash-ml 2.2.2. Data claims were checked against the bundled data files.
- **Assessment:** `assessment/README.md` plus 16 task files (4 tasks × problem/starter/solution/grader). All four graders were run against their references, and against deliberately altered submissions to test false passes and false failures.
- **Shared helpers:** `shared/run_profile.py`, `shared/kailash_helpers.py`, `shared/data_loader.py` and `shared/__init__.py`.
- **Data:** all 7 `data/mlfp01` files and the 2 `data/mlfp_assessment` files used by M1 were loaded and their shapes and schemas inspected.

**Totals:** about 66 files reviewed (excluding generated Colab notebooks, PDFs and `__pycache__`).

**Excluded per instructions:** `colab-selfcontained*/` notebooks, solution runtime pass/fail, code-fit font sizing, and redline-check.py false "0 engines" results.
