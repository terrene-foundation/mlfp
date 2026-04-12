# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 1: Your First Data Exploration
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Assign variables (strings, integers, floats) and format output with f-strings
#   - Load a CSV file into a Polars DataFrame and inspect its shape and columns
#   - Compute summary statistics with describe() and column-level aggregations
#   - Filter rows to find extreme values (max, min) in a dataset
#   - Build a formatted summary report combining all of the above
#
# PREREQUISITES: None — this is the very first exercise in the course.
#
# ESTIMATED TIME: 30-45 minutes
#
# TASKS:
#   1. Store data in variables (strings, numbers, f-strings)
#   2. Load the dataset and inspect its shape, columns, and types
#   3. Compute summary statistics with describe()
#   4. Find the hottest, coldest, and wettest months
#   5. Build a formatted summary report
#
# DATASET: Singapore monthly weather data (temperature, rainfall, humidity)
#   Source: Meteorological Service Singapore (data.gov.sg)
#   Rows: ~12 monthly records | Columns: month, mean_temperature_c, total_rainfall_mm
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
# MLFPDataLoader handles finding data files for you — whether you're
# running locally (VS Code) or on Google Colab. Just specify the module
# and filename, and it will locate the data automatically.
loader = MLFPDataLoader()
df = loader.load("mlfp01", "sg_weather.csv")

print("=" * 60)
print("  MLFP01 Exercise 1: Your First Data Exploration")
print("=" * 60)
print(f"\n  Data loaded: sg_weather.csv ({df.height} rows, {df.width} columns)")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Python basics — variables, strings, numbers, f-strings
# ══════════════════════════════════════════════════════════════════════
# In Python, a variable stores a value. The name goes on the left,
# the value on the right. Python figures out the type automatically —
# no need to declare "this is a string" or "this is a number".

city = "Singapore"  # str: a piece of text, always in quotes
country = "Singapore"  # another string
years_of_data = 30  # int: a whole number
latitude = 1.35  # float: a decimal number

# f-strings let you embed variables directly in text.
# Put the variable name inside curly braces {} within an f"..." string.
print(f"Dataset: Weather data for {city}, {country}")
print(f"Coverage: {years_of_data} years of records")
print(f"Latitude: {latitude} degrees north")

# You can do arithmetic directly in f-strings
celsius_avg = 27.5
fahrenheit_avg = celsius_avg * 9 / 5 + 32
print(f"Average temperature: {celsius_avg}°C / {fahrenheit_avg:.1f}°F")
# The :.1f means "show 1 decimal place" — we'll use format specifiers often.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert isinstance(city, str), "city should be a string"
assert isinstance(years_of_data, int), "years_of_data should be an integer"
assert isinstance(latitude, float), "latitude should be a float"
assert fahrenheit_avg > 80, "Fahrenheit conversion looks wrong"
print("\n✓ Checkpoint 1 passed — variables and f-strings working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Load data and inspect its shape and structure
# ══════════════════════════════════════════════════════════════════════
# A DataFrame is a table of data — rows are observations, columns are
# measurements. Think of it like a spreadsheet in your code.
# Polars is the data library we use throughout this course.

# .shape returns a tuple: (number_of_rows, number_of_columns)
# A tuple is like a list but cannot be changed — (rows, cols)
rows, cols = df.shape
print(f"=== Dataset Overview ===")
print(f"Rows: {rows:,}")  # :, adds comma thousands separator
print(f"Columns: {cols}")

# .columns returns a list of the column names
print(f"\nColumn names:")
for col_name in df.columns:
    # This is a for loop — it runs the indented block once per item
    print(f"  - {col_name}")

# .dtypes returns the data type of each column
# Polars uses its own type system: String=text, Float64=decimal, Int64=integer
print(f"\nData types:")
for col_name, dtype in zip(df.columns, df.dtypes):
    # zip() pairs two lists together: [(col1, dtype1), (col2, dtype2), ...]
    print(f"  {col_name}: {dtype}")

# .head(n) shows the first n rows — always look at raw data before analysis
print(f"\nFirst 5 rows:")
print(df.head(5))

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert rows > 0, "DataFrame has no rows — check data loading"
assert cols >= 2, "DataFrame should have at least 2 columns"
assert "month" in df.columns, "Expected a 'month' column"
print("\n✓ Checkpoint 2 passed — data loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Summary statistics with describe()
# ══════════════════════════════════════════════════════════════════════
# .describe() computes count, mean, std, min, max for every numeric column.
# This single call replaces writing 5 separate calculations — that's
# the point of a data library: common operations should take one line.

print(f"=== Summary Statistics ===")
print(df.describe())

# You can access a single column with df["column_name"]
# Then call aggregation methods directly on that column
mean_temp = df["mean_temperature_c"].mean()
min_temp = df["mean_temperature_c"].min()
max_temp = df["mean_temperature_c"].max()
std_temp = df["mean_temperature_c"].std()

print(f"\nTemperature details:")
print(f"  Average:  {mean_temp:.2f}°C")
print(f"  Minimum:  {min_temp:.2f}°C")
print(f"  Maximum:  {max_temp:.2f}°C")
print(f"  Std dev:  {std_temp:.2f}°C")
# INTERPRETATION: std dev measures how spread out the values are.
# A small std dev means temperatures cluster tightly around the mean.
# Singapore's tropical climate means low temperature variation (~1°C std).

mean_rain = df["total_rainfall_mm"].mean()
max_rain = df["total_rainfall_mm"].max()
print(f"\nRainfall details:")
print(f"  Average:  {mean_rain:.1f} mm/month")
print(f"  Maximum:  {max_rain:.1f} mm/month")
# INTERPRETATION: Singapore receives ~170mm of rain per month on average.
# The monsoon season (Nov-Jan) can bring 250+ mm — nearly 50% above average.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert mean_temp is not None, "mean_temp should not be None"
assert 20 < mean_temp < 35, f"mean_temp={mean_temp} seems wrong for Singapore"
assert mean_rain > 0, "mean_rain should be positive"
print("\n✓ Checkpoint 3 passed — summary statistics computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Find extremes — hottest, coldest, and wettest months
# ══════════════════════════════════════════════════════════════════════
# .filter() keeps rows where a condition is True
# pl.col("column_name") refers to a column inside a Polars expression
# .max() returns the single highest value in a column

# The hottest month: filter where temperature equals the maximum temperature
hottest_row = df.filter(pl.col("mean_temperature_c") == df["mean_temperature_c"].max())
coldest_row = df.filter(pl.col("mean_temperature_c") == df["mean_temperature_c"].min())
wettest_row = df.filter(pl.col("total_rainfall_mm") == df["total_rainfall_mm"].max())

# [0] gets the first (only) element from the filtered result
hottest_month = hottest_row["month"][0]
hottest_temp = hottest_row["mean_temperature_c"][0]

coldest_month = coldest_row["month"][0]
coldest_temp = coldest_row["mean_temperature_c"][0]

wettest_month = wettest_row["month"][0]
wettest_rain = wettest_row["total_rainfall_mm"][0]

print(f"=== Extreme Months ===")
print(f"Hottest:  {hottest_month} at {hottest_temp:.1f}°C")
print(f"Coldest:  {coldest_month} at {coldest_temp:.1f}°C")
print(f"Wettest:  {wettest_month} with {wettest_rain:.1f} mm")
# INTERPRETATION: Singapore's hottest months are May-Jun (pre-monsoon),
# coldest are Dec-Jan (NE monsoon), and wettest are Nov-Jan (monsoon peak).

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert hottest_temp >= coldest_temp, "Hottest should be >= coldest"
assert wettest_rain >= mean_rain, "Wettest month should be above average"
assert isinstance(hottest_month, str), "Month should be a string"
print("\n✓ Checkpoint 4 passed — extreme values found correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Formatted summary report
# ══════════════════════════════════════════════════════════════════════
# Let's combine everything into a clean, readable report.
# Good formatting makes data accessible to non-technical readers.
# "=" * 50 creates a string of 50 "=" characters — a simple separator.

separator = "═" * 58

print(f"\n{separator}")
print(f"  SINGAPORE WEATHER SUMMARY")
print(f"{separator}")
print(f"  Total records:   {rows:>6,}")
print(f"  Date columns:    {cols:>6}")
print(f"")
print(f"  Temperature (°C)")
print(f"    Mean: {mean_temp:>8.2f}")
print(f"    Min:  {min_temp:>8.2f}  ({coldest_month})")
print(f"    Max:  {max_temp:>8.2f}  ({hottest_month})")
print(f"")
print(f"  Rainfall (mm/month)")
print(f"    Mean: {mean_rain:>8.1f}")
print(f"    Max:  {max_rain:>8.1f}  ({wettest_month})")
print(f"{separator}")
# The :>8 in format strings means "right-align in a field 8 chars wide"
# This keeps your numbers lined up in neat columns

# ── Checkpoint 5 ─────────────────────────────────────────────────────
# If you've reached here without errors, all five tasks are complete!
print("\n✓ Checkpoint 5 passed — summary report generated\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ Variables: str, int, float — Python's core building blocks
  ✓ f-strings: embedding variables and formatting numbers in text
  ✓ DataFrames: loading CSV data, inspecting shape/columns/types
  ✓ Aggregations: mean, min, max, std via column methods
  ✓ Filtering: finding rows that match a condition with .filter()
  ✓ Formatted output: building readable reports with alignment

  NEXT: In Exercise 2, you'll learn to filter and transform data
  using Polars expressions — selecting rows by condition, creating
  new columns, and chaining operations together. The HDB resale
  dataset (500K+ transactions) will be your playground.
""")
