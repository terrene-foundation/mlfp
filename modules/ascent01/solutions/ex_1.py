# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT01 — Exercise 1: Your First Data Exploration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Learn Python's core building blocks — variables, strings,
#   numbers, and print() — then apply them immediately to real Singapore
#   weather data using Polars, the data library we use throughout the course.
#
# TASKS:
#   1. Store data in variables (strings, numbers, f-strings)
#   2. Load the dataset and read its shape, columns, and first rows
#   3. Use describe() to get summary statistics
#   4. Find the hottest, coldest, and wettest months
#   5. Print a formatted summary report
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()
df = loader.load("ascent01", "sg_weather.csv")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Python basics — variables, strings, numbers, f-strings
# ══════════════════════════════════════════════════════════════════════

# A variable stores a value. The name on the left, value on the right.
# Python figures out the type automatically — no need to declare it.

city = "Singapore"  # str: a piece of text, always in quotes
country = "Singapore"  # another string
years_of_data = 30  # int: a whole number
latitude = 1.35  # float: a decimal number

# f-strings let you embed variables directly in text.
# Put the variable name inside curly braces {}  within an f"..." string.
print(f"Dataset: Weather data for {city}, {country}")
print(f"Coverage: {years_of_data} years of records")
print(f"Latitude: {latitude} degrees north")

# You can do arithmetic directly in f-strings
celsius_avg = 27.5
fahrenheit_avg = celsius_avg * 9 / 5 + 32
print(f"Average temperature: {celsius_avg}°C / {fahrenheit_avg:.1f}°F")
# The :.1f means "show 1 decimal place" — we'll see more formatting later


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Load data and inspect its shape and structure
# ══════════════════════════════════════════════════════════════════════

# .shape returns a tuple: (number_of_rows, number_of_columns)
# A tuple is like a list but cannot be changed — (rows, cols)
rows, cols = df.shape
print(f"\n=== Dataset Overview ===")
print(f"Rows: {rows:,}")  # :, adds comma thousands separator
print(f"Columns: {cols}")

# .columns returns a list of the column names
print(f"\nColumn names:")
for col_name in df.columns:
    # This is a for loop — it runs the indented block once per item
    print(f"  - {col_name}")

# .dtypes returns the data type of each column
# Polars uses its own type system: Utf8=text, Float64=decimal, Int64=integer
print(f"\nData types:")
for col_name, dtype in zip(df.columns, df.dtypes):
    # zip() pairs two lists together: [(col1, dtype1), (col2, dtype2), ...]
    print(f"  {col_name}: {dtype}")

# .head(n) shows the first n rows — always look at raw data before analysis
print(f"\nFirst 5 rows:")
print(df.head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Summary statistics with describe()
# ══════════════════════════════════════════════════════════════════════

# .describe() computes count, mean, std, min, max for every numeric column.
# This single call replaces writing 5 separate functions — that's the
# point of a data library: common operations should take one line.
print(f"\n=== Summary Statistics ===")
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
# std dev measures spread — a small std dev means values cluster around the mean

mean_rain = df["total_rainfall_mm"].mean()
max_rain = df["total_rainfall_mm"].max()
print(f"\nRainfall details:")
print(f"  Average:  {mean_rain:.1f} mm/month")
print(f"  Maximum:  {max_rain:.1f} mm/month")


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

# .item() extracts a single value from a one-cell DataFrame
hottest_month = hottest_row["month"][0]  # [0] gets the first (only) element
hottest_temp = hottest_row["mean_temperature_c"][0]

coldest_month = coldest_row["month"][0]
coldest_temp = coldest_row["mean_temperature_c"][0]

wettest_month = wettest_row["month"][0]
wettest_rain = wettest_row["total_rainfall_mm"][0]

print(f"\n=== Extreme Months ===")
print(f"Hottest:  {hottest_month} at {hottest_temp:.1f}°C")
print(f"Coldest:  {coldest_month} at {coldest_temp:.1f}°C")
print(f"Wettest:  {wettest_month} with {wettest_rain:.1f} mm")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Formatted summary report
# ══════════════════════════════════════════════════════════════════════

# Let's put it all together in a readable report
# "=" * 50 creates a string of 50 "=" characters — a simple separator
separator = "=" * 50

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

print("\n✓ Exercise 1 complete — Python basics + first data exploration")
