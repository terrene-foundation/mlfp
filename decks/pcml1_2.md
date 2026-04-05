---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.2: Filtering and Transforming

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Evaluate boolean expressions (True/False)
- Use `pl.col()` to reference columns in Polars expressions
- Filter rows with `filter()` based on conditions
- Select, rename, and create columns with `select()`, `sort()`, and `with_columns()`

---

## Recap: Lesson 1.1

- Variables hold strings, integers, and floats
- `print()` and f-strings display formatted output
- Polars loads CSVs into DataFrames
- `shape`, `head()`, `describe()` explore data structure

---

## Booleans: True and False

```python
is_expensive = True
is_small = False

price = 500_000
print(price > 400_000)   # True
print(price == 300_000)   # False
print(price != 300_000)   # True
```

Comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`

---

## Combining Conditions

```python
price = 500_000
rooms = 4

# AND — both must be true
print(price > 400_000 and rooms >= 4)   # True

# OR — at least one must be true
print(price < 300_000 or rooms >= 4)    # True

# NOT — inverts the result
print(not is_small)                      # True
```

---

## Polars Expressions: `pl.col()`

`pl.col()` is the foundation of all Polars data manipulation.

```python
import polars as pl

# Reference a column by name
pl.col("price")

# Apply operations to it
pl.col("price") * 1.07          # add 7% GST
pl.col("price").mean()           # average price
pl.col("floor_area").max()       # largest flat
```

Expressions are **lazy** -- they describe what to compute, not when.

---

## Filtering Rows: `filter()`

```python
# Flats in Tampines
tampines = df.filter(pl.col("town") == "TAMPINES")

# Expensive flats (over $500k)
expensive = df.filter(pl.col("price") > 500_000)

# Combine conditions with & (and) and | (or)
big_tampines = df.filter(
    (pl.col("town") == "TAMPINES") & (pl.col("floor_area") > 100)
)
```

Note: Use `&` and `|` (not `and`/`or`) inside Polars expressions.

---

## Why Parentheses Matter

```python
# WRONG — Python operator precedence breaks this
df.filter(pl.col("town") == "TAMPINES" & pl.col("floor_area") > 100)

# CORRECT — parentheses around each condition
df.filter(
    (pl.col("town") == "TAMPINES") & (pl.col("floor_area") > 100)
)
```

Always wrap each condition in parentheses when combining with `&` or `|`.

---

## Selecting Columns: `select()`

```python
# Select specific columns
subset = df.select("town", "price", "floor_area")

# Select with expressions
subset = df.select(
    pl.col("town"),
    pl.col("price") / 1000,       # price in thousands
)
```

`select()` returns a **new** DataFrame with only the specified columns.

---

## Renaming with `alias()`

```python
result = df.select(
    pl.col("town"),
    (pl.col("price") / 1000).alias("price_k"),
    (pl.col("price") / pl.col("floor_area")).alias("price_per_sqm"),
)
```

`alias()` gives a computed expression a column name.

---

## Sorting: `sort()`

```python
# Sort by price ascending (default)
df_sorted = df.sort("price")

# Sort by price descending
df_sorted = df.sort("price", descending=True)

# Sort by multiple columns
df_sorted = df.sort("town", "price", descending=[False, True])
```

---

## Adding Columns: `with_columns()`

```python
df_enriched = df.with_columns(
    (pl.col("price") / pl.col("floor_area")).alias("price_per_sqm"),
    (pl.col("price") * 1.07).alias("price_with_gst"),
)
```

`with_columns()` keeps **all existing columns** and adds new ones.

Compare:

- `select()` -- returns only the columns you specify
- `with_columns()` -- returns all columns plus new ones

---

## Chaining Operations

```python
result = (
    df
    .filter(pl.col("town") == "TAMPINES")
    .with_columns(
        (pl.col("price") / pl.col("floor_area")).alias("psm")
    )
    .sort("psm", descending=True)
    .head(10)
)
```

Polars operations chain naturally. Each step returns a new DataFrame.

---

## The `select()` vs `with_columns()` Decision

```
┌─────────────────────────────────────────────┐
│  Need only specific columns?                │
│  YES → select()                             │
│  NO  → with_columns() (keep everything)     │
└─────────────────────────────────────────────┘
```

Rule of thumb:

- **Reporting/display** -- use `select()` to keep it clean
- **Adding features for analysis** -- use `with_columns()` to preserve data

---

## Full Example

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdbprices.csv")

top_value = (
    df
    .filter(pl.col("flat_type") == "4 ROOM")
    .with_columns(
        (pl.col("price") / pl.col("floor_area")).alias("price_psm")
    )
    .sort("price_psm")
    .head(5)
    .select("town", "floor_area", "price", "price_psm")
)
print(top_value)
```

---

## Exercise Preview

**Exercise 1.2: HDB Price Filter Dashboard**

You will:

1. Filter flats by town, type, and price range
2. Create computed columns (price per sqm, price bands)
3. Sort and display the top 10 best-value flats
4. Chain multiple operations into a single pipeline

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                                   | Fix                                              |
| ----------------------------------------- | ------------------------------------------------ | ------------------ |
| `and`/`or` in Polars expressions          | Use `&` / `                                      | ` with parentheses |
| Forgetting `pl.col()`                     | `df.filter("price" > 500_000)` will not work     |
| Confusing `select()` and `with_columns()` | `select` drops columns; `with_columns` keeps all |
| Missing `alias()` on computed columns     | Unnamed expressions produce generic column names |

---

## Summary

- Booleans and comparisons control data filtering
- `pl.col()` references columns in Polars expressions
- `filter()` selects rows, `select()` selects columns
- `with_columns()` adds new columns while keeping existing ones
- `sort()` orders data; chain operations for complex pipelines

---

## Next Lesson

**Lesson 1.3: Functions and Aggregation**

We will learn:

- Writing reusable functions with `def` and `return`
- `for` loops for repetitive tasks
- `group_by()` and `agg()` for summarising data by category
