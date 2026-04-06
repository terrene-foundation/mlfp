# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 1 — AI-Resilient Assessment Questions

Statistics, Probability & Data Fluency
Covers: Python basics, Polars, DataExplorer, PreprocessingPipeline, ModelVisualizer
"""

QUIZ = {
    "module": "ASCENT01",
    "title": "Statistics, Probability & Data Fluency",
    "questions": [
        # ── Lesson 1: Python basics, variables, f-strings ─────────────────
        {
            "id": "1.1.1",
            "lesson": "1.1",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student wrote this code to report the average HDB resale price. "
                "The output is '28.8°C' instead of 'S$485,000'. What are the two bugs?"
            ),
            "code": (
                "mean_price = 485000\n"
                "celsius_avg = 28.8\n"
                "print(f'Average HDB price: {celsius_avg:.1f}°C')"
            ),
            "options": [
                "A) Wrong variable referenced in the f-string (celsius_avg instead of mean_price); missing S$ currency format",
                "B) Missing import polars; wrong variable name in f-string",
                "C) mean_price should be a float; f-strings cannot format currency",
                "D) Missing load_dotenv(); wrong decimal places",
            ],
            "answer": "A",
            "explanation": (
                "The f-string uses celsius_avg (28.8) instead of mean_price (485000), "
                "and there is no currency prefix or thousands separator. "
                "The correct print would be: print(f'Average HDB price: S${mean_price:,.0f}')"
            ),
            "learning_outcome": "Use variables and f-string formatting correctly",
        },
        {
            "id": "1.1.2",
            "lesson": "1.1",
            "type": "context_apply",
            "difficulty": "foundation",
            "question": (
                "In Exercise 1, the sg_weather.csv dataset is loaded. "
                "After calling df.describe(), the std dev of mean_temperature_c is very small "
                "(less than 2.0). What does this tell you about Singapore's temperature data, "
                "and which Polars expression extracts just the mean and std columns from describe()?"
            ),
            "code": (
                "stats = df.describe()\n"
                "# Extract only the mean and std rows — fill in the filter"
            ),
            "options": [
                "A) High variance; stats.filter(pl.col('statistic').is_in(['mean', 'std']))",
                "B) Low variance; stats.select(['mean', 'std'])",
                "C) Bimodal distribution; stats.filter(pl.col('statistic') == 'mean')",
                "D) Low variance (temperatures cluster tightly around the mean); stats.filter(pl.col('statistic').is_in(['mean', 'std']))",
            ],
            "answer": "D",
            "explanation": (
                "Singapore is near the equator so temperature varies little year-round — "
                "a small std dev confirms this. "
                "describe() returns a column called 'statistic' with rows like 'mean', 'std', etc., "
                "so you filter on that column, not on column names."
            ),
            "learning_outcome": "Interpret describe() output and understand equatorial climate patterns",
        },
        {
            "id": "1.1.3",
            "lesson": "1.1",
            "type": "output_interpret",
            "difficulty": "foundation",
            "question": (
                "When you run Exercise 1 on the sg_weather.csv data and print df.dtypes, "
                "you see 'month: Utf8'. A student then writes df['month'].mean(). "
                "What error will they get, and what is the correct approach to find the month "
                "with the most rainfall?"
            ),
            "options": [
                "A) No error; mean() works on any column type",
                "B) AttributeError; use df.month.max()",
                "C) InvalidOperationError because Utf8 columns cannot be averaged; use df.filter(pl.col('total_rainfall_mm') == df['total_rainfall_mm'].max())['month'][0]",
                "D) TypeError; use df['month'].cast(pl.Float64).mean()",
            ],
            "answer": "C",
            "explanation": (
                "Polars raises InvalidOperationError when you call a numeric aggregation on a text column. "
                "To find the wettest month you filter where rainfall equals the max rainfall value, "
                "then extract the month string from the result."
            ),
            "learning_outcome": "Distinguish Polars dtype constraints and use filter-with-max pattern",
        },
        # ── Lesson 2: Filtering, with_columns, method chaining ────────────
        {
            "id": "1.2.1",
            "lesson": "1.2",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "This code tries to filter HDB transactions between S$300k and S$500k, "
                "but always returns 0 rows. What is wrong?"
            ),
            "code": (
                "affordable = hdb.filter(\n"
                "    pl.col('resale_price') >= 300_000 & pl.col('resale_price') <= 500_000\n"
                ")"
            ),
            "options": [
                "A) Missing parentheses around each condition — & has higher precedence than >=, so the expression is parsed as >= (300_000 & pl.col(...)) which is a type error or always False",
                "B) & should be and; Python uses 'and' not '&' for DataFrames",
                "C) filter() does not accept two conditions at once; use filter().filter()",
                "D) pl.col() cannot be used with numeric literals",
            ],
            "answer": "A",
            "explanation": (
                "Python's bitwise & has higher operator precedence than comparison operators. "
                "Without parentheses, Python evaluates 300_000 & pl.col('resale_price') first (nonsense), "
                "then >= on the result. "
                "The fix: (pl.col('resale_price') >= 300_000) & (pl.col('resale_price') <= 500_000)"
            ),
            "learning_outcome": "Apply correct parenthesisation when combining Polars Boolean conditions",
        },
        {
            "id": "1.2.2",
            "lesson": "1.2",
            "type": "architecture_decision",
            "difficulty": "foundation",
            "question": (
                "In Exercise 2, you add a price_per_sqm column to the HDB DataFrame. "
                "A student writes two lines: "
                "hdb = hdb.with_columns((pl.col('resale_price') / pl.col('floor_area_sqm')).alias('price_per_sqm')) "
                "then hdb = hdb.with_columns(pl.col('price_per_sqm').round(2).alias('price_per_sqm')). "
                "How can this be written more efficiently as a single with_columns() call, "
                "and why does this matter in production pipelines?"
            ),
            "options": [
                "A) It cannot be combined; Polars requires separate calls for derived columns",
                "B) Use hdb.assign() instead; with_columns() does not support chaining",
                "C) hdb = hdb.with_columns((pl.col('resale_price') / pl.col('floor_area_sqm')).round(2).alias('price_per_sqm')); single call avoids materialising an intermediate DataFrame",
                "D) Use pl.concat() to merge the results of two separate with_columns() calls",
            ],
            "answer": "C",
            "explanation": (
                "You can chain Polars expressions before .alias(), so compute and round in one expression. "
                "A single with_columns() call is more efficient because it avoids allocating an intermediate "
                "DataFrame with the unrounded column — important when the DataFrame is large."
            ),
            "learning_outcome": "Compose Polars expressions efficiently inside with_columns()",
        },
        {
            "id": "1.2.3",
            "lesson": "1.2",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 2 creates a price_tier column using pl.when().then().otherwise(). "
                "If you run the tier_counts group_by in Exercise 2 and see that 'budget' has far "
                "more transactions than 'luxury', which Polars expression correctly computes "
                "the fraction of all transactions that fall in the 'premium' tier?"
            ),
            "options": [
                "A) tier_counts.filter(pl.col('price_tier') == 'premium')['count'].sum() / tier_counts['count'].mean()",
                "B) tier_counts['price_tier'].value_counts()['premium']",
                "C) tier_counts.filter(pl.col('price_tier') == 'premium')['count'][0] / tier_counts['count'].sum()",
                "D) hdb.filter(pl.col('price_tier') == 'premium').height / hdb.height",
            ],
            "answer": "D",
            "explanation": (
                "After group_by/agg you have one row per tier with a count column. "
                "Filter to the premium row, take its count, divide by the total of all counts. "
                "Option D also works on the original DataFrame but requires the price_tier column to exist. "
                "Option B works directly on the tier_counts result shown in the exercise output."
            ),
            "learning_outcome": "Derive proportions from group_by/agg output",
        },
        # ── Lesson 3: Functions, group_by, agg ───────────────────────────
        {
            "id": "1.3.1",
            "lesson": "1.3",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student tries to compute district IQR using map_elements() "
                "but gets an error. What is wrong with this code?"
            ),
            "code": (
                "import pandas as pd\n"
                "district_stats = (\n"
                "    hdb.group_by('town')\n"
                "    .agg(\n"
                "        pd.Series(pl.col('resale_price')).quantile(0.75)\n"
                "        - pd.Series(pl.col('resale_price')).quantile(0.25)\n"
                "    )\n"
                ")"
            ),
            "options": [
                "A) pandas is imported and used inside a Polars expression — this project requires polars only, and pd.Series cannot wrap a Polars expression; use pl.col().quantile(0.75) - pl.col().quantile(0.25)",
                "B) group_by() does not accept string arguments",
                "C) agg() must receive a named alias via .alias()",
                "D) quantile() is not available in Polars group_by context",
            ],
            "answer": "A",
            "explanation": (
                "The course rule is polars-only — no pandas imports. "
                "Even if pandas were allowed, you cannot pass a Polars lazy expression to pd.Series(). "
                "The correct approach uses Polars native quantile within agg(): "
                "(pl.col('resale_price').quantile(0.75) - pl.col('resale_price').quantile(0.25)).alias('iqr_price')"
            ),
            "learning_outcome": "Use Polars native quantile inside group_by/agg without pandas",
        },
        {
            "id": "1.3.2",
            "lesson": "1.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "After running the district_stats aggregation in Exercise 3, "
                "the cv_price_pct column for QUEENSTOWN is 32.1 and for WOODLANDS is 18.4. "
                "What does this tell you about HDB price dispersion in these two towns, "
                "and which town should a buyer target if they want predictable pricing?"
            ),
            "options": [
                "A) QUEENSTOWN has more transactions; WOODLANDS has fewer — neither is about pricing predictability",
                "B) CV measures the number of outliers; QUEENSTOWN has more outliers",
                "C) QUEENSTOWN has wider price spread relative to its mean (CV=32%) indicating mixed flat types or renovation premiums; WOODLANDS is more homogeneous — buyers wanting predictable pricing should target WOODLANDS",
                "D) Both CVs are acceptable; the difference is not meaningful below 50%",
            ],
            "answer": "C",
            "explanation": (
                "Coefficient of variation = std / mean * 100. "
                "A higher CV means prices are more spread out relative to the average — "
                "buying in QUEENSTOWN is riskier because the same flat type can vary widely. "
                "WOODLANDS' lower CV means prices are more consistent and predictable."
            ),
            "learning_outcome": "Interpret coefficient of variation as a relative dispersion metric",
        },
        {
            "id": "1.3.3",
            "lesson": "1.3",
            "type": "process_doc",
            "difficulty": "intermediate",
            "question": (
                "Using the district_report_line() function pattern from Exercise 3, "
                "write the Polars chain that produces a DataFrame with columns "
                "[town, transaction_count, median_price, cv_price_pct] "
                "for the 5 towns with the LOWEST transaction count. "
                "Show each step and explain why you use .tail(5) instead of .head(5)."
            ),
            "options": [
                "A) district_stats.sort('transaction_count').head(5) — head gives lowest when sorted ascending",
                "B) district_stats.sort('transaction_count', descending=True).tail(5) — tail gives lowest when sorted descending",
                "C) district_stats.filter(pl.col('transaction_count') < 100) — filter is cleaner than sort+slice",
                "D) Either A or B work; Polars head/tail are interchangeable with sort direction",
            ],
            "answer": "A",
            "explanation": (
                "Sort ascending (default) puts lowest first, so .head(5) gives the 5 towns with fewest transactions. "
                "Sorting descending and using .tail(5) also works but is less readable. "
                "Option A is the idiomatic pattern: sort ascending, take head."
            ),
            "learning_outcome": "Combine sort() and head/tail correctly for extremes",
        },
        # ── Lesson 4: Joins, multi-table data ────────────────────────────
        {
            "id": "1.4.1",
            "lesson": "1.4",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student runs the join from Exercise 4 but every row has null for "
                "distance_to_mrt_km. They inspect both tables and find the join key "
                "in hdb is 'ANG MO KIO' but in mrt_stations it is 'Ang Mo Kio'. "
                "Which of the following fixes this without modifying the source data?"
            ),
            "code": (
                "hdb_enriched = hdb.join(\n"
                "    mrt_stations.select('town', 'nearest_mrt', 'distance_to_mrt_km'),\n"
                "    on='town',\n"
                "    how='left',\n"
                ")"
            ),
            "options": [
                "A) Change how='left' to how='inner' to force matching",
                "B) Reverse the join: mrt_stations.join(hdb, on='town', how='left')",
                "C) Use .join_asof() instead of .join() to handle fuzzy string matching",
                "D) Normalise the join key before joining: add a lowercase town column to both DataFrames and join on that column",
            ],
            "answer": "D",
            "explanation": (
                "Case mismatch is the most common join failure in real datasets. "
                "The fix is to normalise both keys to the same case before joining: "
                "hdb.with_columns(pl.col('town').str.to_lowercase().alias('town_key')) "
                "and similarly for mrt_stations, then join on 'town_key'. "
                "Switching to inner join just drops unmatched rows — it doesn't fix the data."
            ),
            "learning_outcome": "Diagnose and fix join failures caused by case mismatch",
        },
        {
            "id": "1.4.2",
            "lesson": "1.4",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 4, when aggregating school_counts to town level and then joining "
                "to hdb_enriched, you use .first() for the distance_to_mrt_km column in the "
                "final group_by. Why is .first() correct here while .mean() would be wrong?"
            ),
            "options": [
                "A) distance_to_mrt_km is constant within each town (it came from a town-level lookup table), so every row in the group has the same value — .first() extracts one copy without recomputing a spurious average",
                "B) .mean() is slower than .first() for numeric columns",
                "C) .first() works only on string columns; .mean() would produce a type error",
                "D) group_by() requires you to use .first() for all non-aggregated columns",
            ],
            "answer": "A",
            "explanation": (
                "The MRT proximity data was joined from a town-level table, so every HDB transaction "
                "in a given town carries the same distance_to_mrt_km value. "
                "Taking the mean of identical values is harmless but misleading — "
                ".first() is semantically correct and signals to the reader that the value "
                "does not vary within the group."
            ),
            "learning_outcome": "Choose the correct aggregation function based on data grain and join structure",
        },
        # ── Lesson 5: Window functions, lazy frames ───────────────────────
        {
            "id": "1.5.1",
            "lesson": "1.5",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student tries to compute a 12-month rolling average per town "
                "but gets the same value in every row. What is the bug?"
            ),
            "code": (
                "monthly_prices = monthly_prices.with_columns(\n"
                "    pl.col('median_price_sqm')\n"
                "    .rolling_mean(window_size=12)\n"
                "    .alias('rolling_12m_price_sqm'),  # missing .over()\n"
                ")"
            ),
            "options": [
                "A) rolling_mean() should be rolling_average()",
                "B) Missing .over('town') — without it, the rolling window runs across all rows in the DataFrame regardless of town boundaries, mixing towns together",
                "C) window_size=12 must be specified as a string: '12mo'",
                "D) alias() must come before rolling_mean()",
            ],
            "answer": "B",
            "explanation": (
                "Without .over('town'), Polars applies the rolling window globally across all rows "
                "in the sorted DataFrame, crossing town boundaries. "
                "Rows from different towns get averaged together, producing meaningless values. "
                ".over('town') partitions the window function so each town is processed independently."
            ),
            "learning_outcome": "Apply .over() for per-group window functions",
        },
        {
            "id": "1.5.2",
            "lesson": "1.5",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "After running Exercise 5, the recent_yoy LazyFrame shows BISHAN with "
                "mean_yoy_pct = 8.2% and std_yoy_pct = 6.1%, while TAMPINES shows "
                "mean_yoy_pct = 7.8% and std_yoy_pct = 1.3%. "
                "An investment analyst asks: which town has more predictable price growth? "
                "Explain using the coefficient that quantifies this relationship."
            ),
            "options": [
                "A) BISHAN — higher mean growth always means more predictable appreciation",
                "B) Both are equally predictable because their mean growth rates are similar",
                "C) TAMPINES — lower std_yoy_pct relative to mean_yoy_pct indicates lower growth volatility; compute CV = std/mean: BISHAN CV = 74%, TAMPINES CV = 17%, so TAMPINES growth is far more predictable",
                "D) BISHAN — the absolute std of 6.1% is not large enough to matter",
            ],
            "answer": "C",
            "explanation": (
                "Predictability of growth is measured by the coefficient of variation of the growth rate "
                "(std_yoy / mean_yoy). BISHAN CV = 6.1/8.2 = 74% — highly volatile. "
                "TAMPINES CV = 1.3/7.8 = 17% — very consistent. "
                "An investor who needs predictable annual appreciation should prefer TAMPINES "
                "despite its slightly lower mean growth."
            ),
            "learning_outcome": "Apply CV to assess volatility of time-series growth rates from exercise output",
        },
        {
            "id": "1.5.3",
            "lesson": "1.5",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student writes a lazy frame pipeline but never sees any output. "
                "The script runs without error but prints nothing. What is the bug?"
            ),
            "code": (
                "result = (\n"
                "    monthly_prices.lazy()\n"
                "    .filter(pl.col('transaction_date') >= pl.date(2021, 1, 1))\n"
                "    .drop_nulls('yoy_price_change_pct')\n"
                "    .group_by('town').agg(\n"
                "        pl.col('yoy_price_change_pct').mean().alias('mean_yoy_pct')\n"
                "    )\n"
                "    .sort('mean_yoy_pct', descending=True)\n"
                "    # collect() is missing\n"
                ")\n"
                "print(result.head(10))"
            ),
            "options": [
                "A) lazy() should not be called on a DataFrame; use pl.scan_parquet()",
                "B) .collect() is missing before print() — LazyFrame.head() returns another LazyFrame, not data; you must call .collect() to execute the plan",
                "C) group_by() is not supported on LazyFrames",
                "D) sort() must come before group_by() in a lazy plan",
            ],
            "answer": "B",
            "explanation": (
                "Without .collect(), the variable holds a LazyFrame (a query plan), not a DataFrame. "
                "Calling .head(10) on a LazyFrame returns another LazyFrame. "
                "print() will show the plan representation, not actual data. "
                "Add .collect() after .sort() to materialise the result."
            ),
            "learning_outcome": "Understand that LazyFrame requires .collect() to materialise results",
        },
        # ── Lesson 6: ModelVisualizer, EDA charts ─────────────────────────
        {
            "id": "1.6.1",
            "lesson": "1.6",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student tries to create a correlation heatmap in Exercise 6 but gets a "
                "TypeError on the viz.confusion_matrix() call. What is wrong?"
            ),
            "code": (
                "from kailash_ml import ModelVisualizer\n"
                "viz = ModelVisualizer()\n"
                "# hdb_numeric is a Polars DataFrame with numeric columns\n"
                "fig = viz.confusion_matrix(\n"
                "    matrix=hdb_numeric,  # passing the whole DataFrame\n"
                "    labels=numeric_cols,\n"
                ")"
            ),
            "options": [
                "A) confusion_matrix() should be called confusion_heatmap()",
                "B) ModelVisualizer must be imported from kailash_ml.engines.model_visualizer",
                "C) confusion_matrix() expects a 2D list of floats (list[list[float]]), not a Polars DataFrame — you must build the correlation matrix as a nested list first",
                "D) labels must be a Polars Series, not a Python list",
            ],
            "answer": "C",
            "explanation": (
                "ModelVisualizer.confusion_matrix() accepts a plain Python 2D list as the matrix argument. "
                "In Exercise 6, the correlation values are computed with a nested loop using .pearson_corr() "
                "and assembled into a list[list[float]] before being passed to the visualiser. "
                "Passing the raw DataFrame causes a TypeError because the method cannot iterate the Polars structure."
            ),
            "learning_outcome": "Pass correctly typed arguments to ModelVisualizer methods",
        },
        {
            "id": "1.6.2",
            "lesson": "1.6",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "Exercise 6 uses viz.feature_importance() to create a bar chart of median HDB "
                "price by town even though no ML model is involved. A student asks: "
                "why use feature_importance() for a bar chart instead of a generic chart library? "
                "And what is the correct argument type for the importance_dict parameter?"
            ),
            "options": [
                "A) There is no good reason; you should use plotly.express.bar() directly for non-ML charts",
                "B) importance_dict must be a Polars Series with a label index",
                "C) feature_importance() is the only bar chart method; it requires a trained sklearn model as input",
                "D) ModelVisualizer provides a consistent API across all EDA chart types — feature_importance() accepts any dict[str, float] mapping labels to values, not just ML feature importances",
            ],
            "answer": "D",
            "explanation": (
                "The Kailash course uses ModelVisualizer for all visualisation to keep the API surface consistent. "
                "feature_importance() is a horizontal bar chart that accepts any dict[str, float]. "
                "In Exercise 6 you pass {town_name: median_price} — the semantics are 'rank by value', "
                "which works equally well for feature importances, prices, volumes, or any ranked metric."
            ),
            "learning_outcome": "Reuse ModelVisualizer methods beyond their original ML context",
        },
        {
            "id": "1.6.3",
            "lesson": "1.6",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "After running Exercise 6 and examining ex6_correlation_heatmap.html, "
                "you find that resale_price and floor_area_sqm have Pearson r = 0.57, "
                "while resale_price and year have r = 0.31. "
                "Write the ModelVisualizer call that produces a scatter plot of "
                "price_per_sqm vs year (not resale_price), "
                "and explain what a positive r between price_per_sqm and year means for buyers."
            ),
            "options": [
                "A) viz.scatter_plot(x_values=hdb['year'].to_list(), y_values=hdb['price_per_sqm'].drop_nulls().to_list(), x_label='Year', y_label='Price per sqm (S$)'); positive r means per-sqm prices rose over time independent of flat size",
                "B) viz.scatter_plot(x_values=hdb['year'].to_list(), y_values=hdb['resale_price'].to_list()); a positive r means prices rose over time",
                "C) viz.feature_distribution(values=hdb['price_per_sqm'].to_list(), feature_name='Price per sqm'); scatter_plot() only accepts pre-sampled arrays",
                "D) viz.training_history(history={'price_per_sqm': hdb['price_per_sqm'].to_list()}); year must be the x_label not a value list",
            ],
            "answer": "A",
            "explanation": (
                "scatter_plot() requires x_values and y_values as plain Python lists plus optional axis labels. "
                "drop_nulls() removes any rows where price_per_sqm is null to avoid list alignment issues. "
                "A positive Pearson r between year and price_per_sqm shows that even after normalising by flat size, "
                "prices have risen — buyers pay more per square metre each year, independent of flat size inflation."
            ),
            "learning_outcome": "Construct correct ModelVisualizer scatter_plot() call and interpret correlation direction",
        },
        # ── Lesson 7 & 8: DataExplorer, PreprocessingPipeline ─────────────
        {
            "id": "1.7.1",
            "lesson": "1.7",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student configures DataExplorer with AlertConfig but gets an ImportError. "
                "What is wrong with this import?"
            ),
            "code": (
                "from kailash_ml import AlertConfig\n"
                "explorer = DataExplorer(\n"
                "    AlertConfig(correlation_threshold=0.85)\n"
                ")"
            ),
            "options": [
                "A) DataExplorer is not in kailash_ml; it must be imported from kailash.engines",
                "B) AlertConfig is not exported from the top-level kailash_ml package — it lives in kailash_ml.engines.data_explorer; correct import: from kailash_ml.engines.data_explorer import AlertConfig",
                "C) AlertConfig is a class method of DataExplorer; it should not be imported",
                "D) The constructor argument should be alert_config=AlertConfig(...), not a positional argument",
            ],
            "answer": "B",
            "explanation": (
                "kailash_ml's top-level __init__ exports the major engine classes but not all config dataclasses. "
                "AlertConfig must be imported from its module: "
                "from kailash_ml.engines.data_explorer import AlertConfig. "
                "DataExplorer itself is available from the top-level import."
            ),
            "learning_outcome": "Use correct import paths for kailash_ml engine config classes",
        },
        {
            "id": "1.7.2",
            "lesson": "1.7",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Running DataExplorer on the HDB dataset produces this alert:\n\n"
                "  ALERT [HIGH_CARDINALITY] column='storey_range' unique_ratio=0.94\n\n"
                "What does this mean for feature engineering, and should you encode this "
                "column as ordinal or drop it before training a tree model?"
            ),
            "options": [
                "A) 94% of values in storey_range are unique — it behaves like an ID column, not a category. For a tree model, consider extracting numeric floor number (e.g., lower floor of range) rather than encoding the raw string as ordinal, which would create a near-unique feature with no generalisation power",
                "B) HIGH_CARDINALITY means the column has many nulls; drop it",
                "C) Ordinal encoding is always correct for string columns regardless of cardinality",
                "D) HIGH_CARDINALITY means the column is already numeric; no encoding needed",
            ],
            "answer": "A",
            "explanation": (
                "A unique_ratio near 1.0 means almost every row has a different value. "
                "Encoding 'storey_range' as ordinal would give each category a unique integer, "
                "which the tree splits on without learning a generalizable pattern. "
                "The correct engineering step is to parse the numeric floors from the string "
                "(e.g., '10 TO 12' → 11 as mid-storey) to get a compact, ordered numeric feature."
            ),
            "learning_outcome": "Interpret DataExplorer HIGH_CARDINALITY alert and decide on feature engineering action",
        },
        {
            "id": "1.8.1",
            "lesson": "1.8",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student calls PreprocessingPipeline but the resulting train_data still "
                "contains null values. What is missing from this setup call?"
            ),
            "code": (
                "from kailash_ml import PreprocessingPipeline\n"
                "pipeline = PreprocessingPipeline()\n"
                "result = pipeline.setup(\n"
                "    data=credit,\n"
                "    target='default',\n"
                "    train_size=0.8,\n"
                "    seed=42,\n"
                "    normalize=True,\n"
                "    categorical_encoding='ordinal',\n"
                "    # imputation_strategy is missing\n"
                ")"
            ),
            "options": [
                "A) normalize=True automatically handles nulls",
                "B) imputation_strategy is not a valid parameter; use fillna() after setup()",
                "C) Without imputation_strategy, PreprocessingPipeline leaves nulls untouched — add imputation_strategy='median' or 'mean' to fill numeric nulls before training",
                "D) PreprocessingPipeline cannot handle nulls at all; drop them before calling setup()",
            ],
            "answer": "C",
            "explanation": (
                "PreprocessingPipeline only applies imputation when imputation_strategy is explicitly set. "
                "Without it, null values propagate through to train_data and test_data. "
                "Most tree-based models in sklearn handle nulls, but linear models (Ridge, Lasso) do not, "
                "so always specify imputation_strategy='median' as a safe default."
            ),
            "learning_outcome": "Configure PreprocessingPipeline with imputation to handle null values",
        },
        {
            "id": "1.8.2",
            "lesson": "1.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You need to prepare the sg_credit_scoring dataset for both a Lasso regression "
                "and a LightGBM classifier. PreprocessingPipeline has normalize and "
                "categorical_encoding parameters. Which combination is correct for Lasso, "
                "and which is correct for LightGBM, and why do they differ?"
            ),
            "options": [
                "A) Both require normalize=True and categorical_encoding='one_hot'",
                "B) Both require normalize=False; PreprocessingPipeline normalisation causes data leakage",
                "C) Lasso: normalize=False; LightGBM: normalize=True — tree models are more sensitive to scale",
                "D) Lasso: normalize=True, categorical_encoding='ordinal'; LightGBM: normalize=False, categorical_encoding='ordinal'. Lasso requires normalisation because L1 penalty treats all coefficients equally regardless of scale; LightGBM builds splits so scale is irrelevant, and ordinal is sufficient for categorical columns",
            ],
            "answer": "D",
            "explanation": (
                "Lasso's L1 penalty applies the same regularisation strength to all coefficients. "
                "If features have vastly different scales (income in thousands vs. age in decades), "
                "the penalty shrinks large-scale features disproportionately. "
                "Normalisation removes this distortion. "
                "LightGBM builds decision trees using threshold splits — scale does not affect "
                "which threshold is optimal, so normalisation adds no benefit."
            ),
            "learning_outcome": "Select correct normalisation and encoding for different model families",
        },
        # ── Additional questions covering lessons 1–8 breadth ─────────────
        {
            "id": "1.1.4",
            "lesson": "1.1",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student loads data in Exercise 1 using a hardcoded path. "
                "What is wrong with this, and what is the correct pattern used throughout the course?"
            ),
            "code": (
                "import polars as pl\n"
                "df = pl.read_csv('/Users/student/Downloads/sg_weather.csv')  # Bug"
            ),
            "options": [
                "A) pl.read_csv() should be pl.scan_csv() for large files",
                "B) Hardcoded paths break on any other machine — use ASCENTDataLoader: from shared import ASCENTDataLoader; loader = ASCENTDataLoader(); df = loader.load('ascent01', 'sg_weather.csv')",
                "C) The file extension must be .parquet; CSVs are not supported",
                "D) import polars must come after from shared import ASCENTDataLoader",
            ],
            "answer": "B",
            "explanation": (
                "ASCENTDataLoader abstracts the file location — it resolves paths correctly on any machine, "
                "handles Colab vs local differences, and caches downloads. "
                "Hardcoded paths (/Users/student/...) fail immediately on any other system. "
                "This loader pattern is used in every exercise in the course."
            ),
            "learning_outcome": "Use ASCENTDataLoader for portable data loading across all exercise formats",
        },
        {
            "id": "1.4.3",
            "lesson": "1.4",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "After the join in Exercise 4, you call .null_count() on the enriched DataFrame. "
                "nearest_mrt has 340 nulls out of 15,000 rows. "
                "Which Polars expression correctly filters the enriched DataFrame to keep ONLY "
                "rows that have a valid MRT match, and what does this operation risk?"
            ),
            "options": [
                "A) hdb_enriched.drop_nulls() — drops all rows with any null in any column",
                "B) hdb_enriched.filter(pl.col('nearest_mrt').is_not_null()) — keeps only rows with an MRT match. The risk: the 340 unmatched rows are dropped, potentially introducing selection bias if the unmatched towns have systematically different price characteristics",
                "C) hdb_enriched.fill_null('Unknown') — replaces nulls with a placeholder string",
                "D) hdb_enriched.filter(pl.col('nearest_mrt') != None) — Python None comparison works in Polars",
            ],
            "answer": "B",
            "explanation": (
                "is_not_null() is the correct Polars predicate for non-null filtering. "
                "Using != None in a Polars expression does not work as expected — "
                "you must use .is_not_null(). "
                "The selection bias risk is real: if the 340 unmatched rows are all in remote towns "
                "with lower prices, dropping them would upwardly bias subsequent price analysis."
            ),
            "learning_outcome": "Apply is_not_null() for null filtering and identify selection bias risk from dropping rows",
        },
        {
            "id": "1.6.4",
            "lesson": "1.6",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student saves a Plotly figure in Exercise 6 but the HTML file cannot be "
                "opened by the teaching assistant. What is wrong?"
            ),
            "code": (
                "fig = viz.feature_distribution(values=prices, feature_name='Resale Price')\n"
                "fig.show()  # Shows in Jupyter — but this is the local format exercise\n"
                "# Missing: fig.write_html(...)"
            ),
            "options": [
                "A) fig.show() is the correct output method for all three exercise formats",
                "B) In local (.py) format exercises, figures must be saved with fig.write_html('filename.html') — fig.show() opens a browser window that closes when the script ends and leaves no artefact for the TA to review",
                "C) write_html() is only available in Jupyter; local scripts must use fig.save()",
                "D) Both fig.show() and fig.write_html() are required; using only one causes an error",
            ],
            "answer": "B",
            "explanation": (
                "The three-format rule: in local .py scripts, fig.write_html() is the output method. "
                "fig.show() opens a browser that blocks the script and produces no saved file. "
                "In Jupyter and Colab notebooks, fig.show() displays inline — no write_html() needed. "
                "The course rules/three-format.md specifies this distinction explicitly."
            ),
            "learning_outcome": "Use fig.write_html() in local script format vs fig.show() in notebooks",
        },
        {
            "id": "1.3.4",
            "lesson": "1.3",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 3, you iterate over district_stats with iter_rows(named=True). "
                "A student suggests using a list comprehension on .to_dicts() instead. "
                "For a 26-row district summary, which is more Pythonic and why would "
                "the choice matter for a DataFrame with 500,000 rows?"
            ),
            "options": [
                "A) to_dicts() is always faster; iter_rows() is deprecated",
                "B) iter_rows() does not support named=True for DataFrames with more than 100 columns",
                "C) For 26 rows both are equivalent in practice. For 500,000 rows, iter_rows() is preferable: it yields one dict at a time without materialising the entire list in memory, while to_dicts() allocates all 500,000 dicts simultaneously. iter_rows() also signals intent — you are processing row-by-row, not needing all rows at once",
                "D) to_dicts() is the only option; iter_rows() requires a LazyFrame",
            ],
            "answer": "C",
            "explanation": (
                "Both produce the same output for small DataFrames. "
                "iter_rows() is a generator — it yields rows lazily, keeping memory O(1) at any point. "
                "to_dicts() materialises all rows as a list of dicts upfront — O(n) memory. "
                "For the 26-row district summary in Exercise 3, the difference is negligible. "
                "For large DataFrames, prefer iter_rows() to avoid memory spikes."
            ),
            "learning_outcome": "Distinguish iter_rows() lazy generation from to_dicts() full materialisation",
        },
    ],
}

if __name__ == "__main__":
    for q in QUIZ["questions"]:
        print(f"\n{'=' * 60}")
        print(f"[{q['id']}] ({q['type']}) — Lesson {q['lesson']}  [{q['difficulty']}]")
        print(f"{'=' * 60}")
        print(q["question"])
        if q.get("code"):
            print(f"\n```python\n{q['code']}\n```")
        for opt in q["options"]:
            print(f"  {opt}")
        print(f"\nAnswer: {q['answer']}")
