---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.1: Your First Data Exploration

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Create and manipulate Python variables (strings, integers, floats)
- Use `print()` to inspect values and debug code
- Load a CSV file into a Polars DataFrame
- Explore data using `shape`, `head()`, and `describe()`

---

## Recap

Welcome to ASCENT! This is your very first lesson.
No prior programming experience is assumed.

We will learn Python **through** data exploration using the Kailash SDK.

---

## What is a Variable?

A variable is a **name** that points to a value.

```python
city = "Singapore"
temperature = 31.5
population = 5_917_600
```

- `city` is a **string** (text inside quotes)
- `temperature` is a **float** (decimal number)
- `population` is an **integer** (whole number)

---

## The `print()` Function

`print()` displays values in your terminal or notebook.

```python
city = "Singapore"
print(city)           # Singapore
print(type(city))     # <class 'str'>
print(len(city))      # 9
```

Use `print()` liberally when exploring data -- it is your primary debugging tool.

---

## String Basics

```python
first = "Machine"
second = "Learning"

# Concatenation
full = first + " " + second    # "Machine Learning"

# f-strings (formatted strings)
module = 1
print(f"Welcome to ASCENT Module {module}")
```

f-strings are the modern Python way to embed values in text.

---

## Numbers and Arithmetic

```python
price = 450_000
rooms = 4

price_per_room = price / rooms       # 112500.0
price_rounded = price // rooms       # 112500 (integer division)
remainder = price % rooms            # 0

squared = rooms ** 2                 # 16
```

Python supports `+`, `-`, `*`, `/`, `//`, `%`, `**`.

---

## Why Polars, Not Pandas?

| Aspect  | Polars                       | Pandas                        |
| ------- | ---------------------------- | ----------------------------- |
| Speed   | Rust-powered, multi-threaded | Single-threaded               |
| Memory  | Efficient Arrow format       | High memory usage             |
| API     | Expression-based, composable | Method-chaining, inconsistent |
| Used by | Kailash SDK (native)         | Legacy codebases              |

ASCENT uses **Polars exclusively** -- the industry is moving this direction.

---

## Loading Your First Dataset

```python
import polars as pl

# Using the ASCENT data loader
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdbprices.csv")
```

The `ASCENTDataLoader` handles file paths across local, Jupyter, and Colab environments.

---

## Inspecting the DataFrame: `shape`

```python
print(df.shape)
# (12345, 8)
```

Returns a tuple: **(rows, columns)**

- 12,345 rows = 12,345 HDB transactions
- 8 columns = 8 attributes per transaction

---

## Inspecting the DataFrame: `head()`

```python
print(df.head())
```

```
shape: (5, 8)
┌───────┬──────────┬───────────┬──────────┐
│ town  ┆ flat_type┆ floor_area┆ price    │
│ ---   ┆ ---      ┆ ---       ┆ ---      │
│ str   ┆ str      ┆ f64       ┆ i64      │
╞═══════╪══════════╪═══════════╪══════════╡
│ ANG.. ┆ 3 ROOM  ┆ 67.0      ┆ 310000   │
│ ...   ┆ ...      ┆ ...       ┆ ...      │
└───────┴──────────┴───────────┴──────────┘
```

`head(n)` shows the first `n` rows (default 5).

---

## Inspecting the DataFrame: `describe()`

```python
print(df.describe())
```

Shows summary statistics for every column:

- **count** -- number of non-null values
- **mean** -- average (numeric columns)
- **std** -- standard deviation
- **min / max** -- range
- **25% / 50% / 75%** -- quartiles

---

## Column Names and Data Types

```python
print(df.columns)
# ['town', 'flat_type', 'floor_area', 'price', ...]

print(df.dtypes)
# [Utf8, Utf8, Float64, Int64, ...]
```

Polars infers types automatically:

- `Utf8` = text/string
- `Float64` = decimal number
- `Int64` = whole number

---

## Selecting a Single Column

```python
prices = df["price"]
print(prices.mean())
print(prices.max())
print(prices.min())
```

Bracket notation extracts one column as a **Series** (a single column of data).

---

## Combining What We Know

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdbprices.csv")

rows, cols = df.shape
print(f"Dataset has {rows:,} transactions and {cols} features")
print(f"Average price: ${df['price'].mean():,.0f}")
print(f"Price range: ${df['price'].min():,} to ${df['price'].max():,}")
```

---

## Exercise Preview

**Exercise 1.1: HDB Price Explorer**

You will:

1. Load the HDB resale prices dataset
2. Inspect its shape, columns, and types
3. Calculate basic statistics (mean, median, range)
4. Use f-strings to print a formatted summary report

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                      | Fix                                        |
| ---------------------------- | ------------------------------------------ |
| `import pandas as pd`        | Use `import polars as pl`                  |
| Forgetting quotes on strings | `city = Singapore` vs `city = "Singapore"` |
| Using `=` instead of `==`    | `=` assigns, `==` compares                 |
| Case sensitivity             | `Price` and `price` are different          |
| Missing `f` in f-strings     | `f"Value: {x}"` not `"Value: {x}"`         |

---

## Summary

- Variables store values: strings, integers, floats
- `print()` and f-strings display and format output
- Polars loads CSV data into DataFrames
- `shape`, `head()`, `describe()` are your first exploration tools
- Always use the `ASCENTDataLoader` for consistent data access

---

## Next Lesson

**Lesson 1.2: Filtering and Transforming**

We will learn:

- Boolean logic (True/False)
- Polars expressions with `pl.col()`
- Filtering rows with `filter()`
- Selecting and creating columns with `select()` and `with_columns()`
