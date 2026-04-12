# MODULE 1: Machine Learning Data Pipelines and Visualisation Mastery with Python

**Description**: Zero to productive. Learn Python by exploring real Singapore data. Every Python concept grounded in a data task.

**Module Learning Objectives**: By the end of M1, students can:
- Write Python programs with variables, functions, loops, conditionals, and collections
- Load, filter, join, aggregate, and transform datasets using Polars
- Create interactive visualisations using Plotly via ModelVisualizer
- Profile datasets automatically and detect data quality issues
- Build end-to-end data pipelines: load, profile, clean, visualise, report

**Kailash Engines**: DataExplorer, PreprocessingPipeline, ModelVisualizer

---

## Lesson 1.1: Your First Data Exploration

**Prerequisites**: None (first lesson)
**Spectrum Position**: Data acquisition — before features exist

**Topics**:
- Python basics: variables, data types (int, float, str, bool), `print()`, f-strings
- Polars: `pl.read_csv()`, `df.shape`, `df.columns`, `df.head()`, `df.describe()`
- First 90 min: Python REPL basics. Second 90 min: load and explore real data.

**Key Concepts**: Variable assignment, data types, string formatting, DataFrame as the fundamental data structure

**Learning Objectives**: Students can:
- Assign variables, perform arithmetic, format output strings
- Load a CSV file into a Polars DataFrame
- Inspect shape, columns, head, and summary statistics

**Exercise**: Load Singapore weather CSV (~1K rows). Print shape, column names, first 5 rows, summary statistics. Answer 3 data questions using only `describe()` output.

**Assessment Criteria**: Code runs without error. All 3 questions answered correctly with evidence from data.

**R5 Source**: Deck 1B (data types, operators, variables) + PCML1-1

---

## Lesson 1.2: Filtering and Transforming Data

**Prerequisites**: 1.1 (variables, DataFrame basics)
**Spectrum Position**: Data selection — choosing relevant observations

**Topics**:
- Booleans and comparison operators (`>`, `<`, `==`, `!=`)
- Polars expressions: `pl.col()`, `filter()`, `select()`, `sort()`, `with_columns()`
- Method chaining (fluent API style)

**Note**: Students use boolean expressions within Polars (declarative). Python `if/else` is deferred to 1.4. Add forward-reference: "You are writing expressions that evaluate to True/False. Python also has `if/else` for code-level decisions — Lesson 1.4."

**Key Concepts**: Boolean logic, expression-based filtering, method chaining, column transformation

**Learning Objectives**: Students can:
- Filter rows by one or more conditions
- Select specific columns
- Sort data by any column
- Create new computed columns

**Exercise**: Filter HDB resale data by town, price range, and date. Create a new column (price per sqm). Sort by price descending.

**Assessment Criteria**: Correct filters applied. New column computed correctly. Output sorted.

**R5 Source**: Deck 1B (comparison operators) + PCML1-2

---

## Lesson 1.3: Functions and Aggregation

**Prerequisites**: 1.2 (filtering, expressions)
**Spectrum Position**: Data summarisation — compressing information

**Topics**:
- `def` functions, parameters, `return` statements
- `for` loops, lists, dictionaries
- Polars: `group_by()`, `agg()`, `pl.mean()`, `pl.sum()`, `pl.count()`
- Writing helper functions for reusable analysis

**Key Concepts**: Function abstraction, iteration, collection types, grouped aggregation

**Learning Objectives**: Students can:
- Define functions that accept parameters and return values
- Use loops to process collections
- Aggregate data by groups (mean, sum, count per category)
- Write reusable helper functions for common data tasks

**Exercise**: Write functions to compute district-level statistics (mean price, transaction count, price range) for HDB data. Use `group_by` + `agg` to produce summary table.

**Assessment Criteria**: Functions are reusable (not hardcoded). Aggregation produces correct grouped results.

**R5 Source**: Deck 1B (functions, collections) + PCML1-3

---

## Lesson 1.4: Joins and Multi-Table Data

**Prerequisites**: 1.3 (functions, collections, aggregation)
**Spectrum Position**: Data integration — combining information sources

**Topics**:
- `if/else/elif` conditional statements
- `import` and packages
- Join concepts: left, inner, outer. When to use each.
- Polars: `join()`, multi-table operations on HDB 15M rows
- Dictionary lookups and mapping

**Key Concepts**: Conditional logic, package imports, relational joins, multi-source data integration

**Learning Objectives**: Students can:
- Write conditional logic for branching decisions
- Import and use external packages
- Join multiple DataFrames on shared keys
- Reason about which join type to use for a given task

**Exercise**: Join HDB resale data with MRT station data and school data. Compute distance-to-amenity features. Handle missing joins with appropriate join type.

**Assessment Criteria**: Correct join type selected. Missing data handled (not silently dropped). Combined dataset has expected row count.

**R5 Source**: Deck 2B (merging: join, merge, concat) + ASCENT M1 ex_1 (joins portion)

---

## Lesson 1.5: Window Functions and Trends

**Prerequisites**: 1.4 (joins, conditionals)
**Spectrum Position**: Temporal feature creation — extracting time-based patterns

**Topics**:
- Polars window functions: `over()`, `rolling_mean()`, `shift()`
- Rolling aggregations, YoY calculations, moving averages
- Lazy frames (introduced as performance optimisation, not core concept): `scan_csv()`, `collect()`

**Note**: Lazy frames are a "make it faster" add-on, not a prerequisite. Students can complete all exercises with eager evaluation.

**Key Concepts**: Window functions, rolling statistics, temporal trends, lazy evaluation

**Learning Objectives**: Students can:
- Compute rolling averages and YoY changes
- Use window functions for within-group calculations
- Identify trends and seasonality in time-series data
- Understand when lazy evaluation helps performance

**Exercise**: Compute 3-month rolling average of HDB prices per district. Calculate YoY price change. Identify districts with highest/lowest growth trends.

**Assessment Criteria**: Rolling calculations correct. YoY computed with proper time alignment.

**R5 Source**: ASCENT M1 ex_1 (windows portion)

---

## Lesson 1.6: Data Visualisation

**Prerequisites**: 1.5 (aggregation, trends)
**Spectrum Position**: Data communication — making patterns visible

**Topics**:
- Visualisation principles from Deck 1C:
  - Why visualise (tables vs charts for decision-making)
  - Attributes of good charts: simple, clean, subtle attention, truthful
  - Gestalt principles: proximity, similarity, closure, enclosure, continuity, connection
  - Visual order: Z-pattern reading (left-to-right, top-to-bottom)
  - Charts to avoid: 3D charts, misleading pie charts
- Chart selection by data type:
  - Heatmaps (correlation), line charts (time-series), vertical bars (comparison), horizontal bars (categorical)
  - Stacked bars (composition), 100% stacked (Likert/survey data)
- Plotly Express / ModelVisualizer for interactive charts

**Key Formulas**: None (visual design principles, not mathematical)

**Learning Objectives**: Students can:
- Select the appropriate chart type for a given data question
- Create interactive visualisations with Plotly via ModelVisualizer
- Apply Gestalt principles to improve chart readability
- Identify and avoid misleading chart designs

**Exercise**: Create 6 different chart types from HDB data: heatmap (price correlation), line (price trend), bar (district comparison), scatter (size vs price), histogram (price distribution), stacked bar (flat type composition).

**Assessment Criteria**: Chart type appropriate for each question. No misleading axes. Interactive features (hover, zoom) functional.

**R5 Source**: Deck 1C (30 slides on viz principles, chart types, Plotly)

---

## Lesson 1.7: Automated Data Profiling

**Prerequisites**: 1.6 (visualisation)
**Spectrum Position**: Automated data assessment — machine-detected quality issues

**Topics**:
- DataExplorer: automated profiling with 8 alert types
- AlertConfig: configure thresholds for missing values, outliers, duplicates, skew, correlation, cardinality, constants, type inference
- DataProfile object: access profiling results programmatically
- `compare()`: compare two datasets (before/after cleaning, train/test distributions)
- Classes as users (not authors): students use DataExplorer, not build it
- `try/except` basics for error handling
- Async hidden behind `shared.run_profile()` sync wrapper

**Key Concepts**: Automated data quality assessment, alert configuration, dataset comparison

**Learning Objectives**: Students can:
- Run automated data profiling on any dataset
- Configure alert thresholds for data quality rules
- Compare two datasets and identify distribution differences
- Handle errors gracefully with try/except

**Exercise**: Profile dirty economic indicators dataset. Identify issues (missing values, outliers, skew). Configure alerts. Compare original vs cleaned version.

**Assessment Criteria**: All data quality issues identified. Alerts configured with sensible thresholds. Comparison shows improvement.

**R5 Source**: ASCENT M1 ex_3

---

## Lesson 1.8: Data Pipelines and End-to-End Project

**Prerequisites**: All of M1 (1.1-1.7)
**Spectrum Position**: Complete data pipeline — acquisition to report

**Topics**:
- None/null handling: `is_null()`, `fill_null()`, `drop_nulls()`
- ETL concepts: Extract (APIs, files), Transform (clean, encode, scale), Load (output)
- REST APIs: GET, POST, JSON responses, query parameters (OneMap Singapore example from Deck 1C)
- PreprocessingPipeline: auto-detect data types, encode categoricals, scale numerics, impute missing values
- Full pipeline: load -> profile -> clean -> visualise -> report
- Project structure: modules, imports, putting it all together

**Key Concepts**: ETL pipeline, data cleaning automation, API data extraction, preprocessing pipeline

**Learning Objectives**: Students can:
- Build a complete data pipeline from raw data to clean output
- Extract data from REST APIs
- Use PreprocessingPipeline for automated cleaning
- Structure a multi-file Python project

**Exercise**: Build full EDA pipeline for messy taxi trip data: load from API -> profile with DataExplorer -> clean with PreprocessingPipeline -> visualise key patterns -> generate HTML report.

**Assessment Criteria**: Pipeline runs end-to-end. Data quality improved (fewer missing values, outliers handled). Report contains at least 3 visualisations with insights.

**R5 Source**: Deck 1C (APIs, REST) + PCML1-5 (ETL dashboard) + ASCENT M1 ex_5

**End of Module Assessment**: Quiz (AI-resilient, context-specific questions on data types, Polars operations, viz principles, ETL concepts).
