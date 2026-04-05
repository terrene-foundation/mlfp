---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.4: Joins and Multi-Table Data

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Create and access Python dictionaries
- Use `if`/`elif`/`else` for conditional logic
- Join two DataFrames using `join()` with left and inner strategies
- Understand when to use each join type

---

## Recap: Lesson 1.3

- Functions (`def`, `return`) make code reusable
- `for` loops iterate over collections
- `group_by().agg()` summarises data by category
- Multiple aggregation expressions produce rich summaries

---

## Python Dictionaries

Dictionaries store **key-value pairs** -- like a lookup table.

```python
flat_info = {
    "town": "TAMPINES",
    "type": "4 ROOM",
    "price": 480_000,
    "floor_area": 92.0,
}

print(flat_info["town"])       # TAMPINES
print(flat_info["price"])      # 480000
```

Keys are unique. Values can be any type.

---

## Dictionary Operations

```python
# Add or update a key
flat_info["year"] = 2024

# Check if a key exists
if "price" in flat_info:
    print("Price is available")

# Get with a default (avoids errors)
lease = flat_info.get("lease_years", "Unknown")

# Loop through keys and values
for key, value in flat_info.items():
    print(f"{key}: {value}")
```

---

## Conditional Logic: `if` / `elif` / `else`

```python
price = 480_000

if price > 1_000_000:
    band = "Million-dollar"
elif price > 500_000:
    band = "Premium"
elif price > 300_000:
    band = "Mid-range"
else:
    band = "Affordable"

print(f"${price:,} is in the '{band}' band")
```

Python evaluates conditions top-to-bottom; first match wins.

---

## Conditionals in Functions

```python
def classify_flat(floor_area):
    if floor_area >= 110:
        return "Large"
    elif floor_area >= 80:
        return "Medium"
    else:
        return "Small"

# Apply to a DataFrame column using map_elements
df_classified = df.with_columns(
    pl.col("floor_area").map_elements(
        classify_flat, return_dtype=pl.Utf8
    ).alias("size_class")
)
```

---

## Polars `when().then().otherwise()`

The Polars-native way to do conditional logic (preferred over `map_elements`):

```python
df_classified = df.with_columns(
    pl.when(pl.col("floor_area") >= 110).then(pl.lit("Large"))
      .when(pl.col("floor_area") >= 80).then(pl.lit("Medium"))
      .otherwise(pl.lit("Small"))
      .alias("size_class")
)
```

This runs in Rust (fast) instead of Python (slow).

---

## Why Joins Matter

Real data lives in **multiple tables**.

```
┌──────────────┐    ┌───────────────────┐
│ transactions │    │ town_demographics  │
│              │    │                    │
│ town         │───>│ town              │
│ price        │    │ population        │
│ floor_area   │    │ median_income     │
│ flat_type    │    │ mrt_stations      │
└──────────────┘    └───────────────────┘
```

Joins combine these tables on a shared column (the **key**).

---

## Inner Join

Returns only rows where the key exists in **both** tables.

```python
result = transactions.join(
    demographics,
    on="town",
    how="inner"
)
```

```
Transactions: 10 towns     Demographics: 8 towns
Result: only the 8 towns that appear in BOTH
```

Use when: you only want complete records.

---

## Left Join

Returns **all rows** from the left table, with matches from the right.

```python
result = transactions.join(
    demographics,
    on="town",
    how="left"
)
```

```
Transactions: 10 towns     Demographics: 8 towns
Result: all 10 towns; 2 have null demographics
```

Use when: you want to keep all your primary data, even without a match.

---

## Join Diagram (Text)

```
INNER JOIN:          LEFT JOIN:
  A   B                A   B
 ┌─┬─┐              ┌───┬─┐
 │ │█│              │███│█│
 │ │█│              │███│█│
 └─┴─┘              │███│ │
  overlap only       └───┴─┘
                     all of A + matches from B
```

---

## Join with Different Column Names

```python
# When the key columns have different names
result = sales.join(
    regions,
    left_on="town_name",
    right_on="town",
    how="left"
)
```

Use `left_on` and `right_on` when the join key has different names in each table.

---

## Multi-Key Joins

```python
# Join on multiple columns
result = transactions.join(
    price_index,
    on=["town", "flat_type", "year"],
    how="left"
)
```

Joining on multiple columns ensures precise matching -- "TAMPINES + 4 ROOM + 2024" matches exactly one row.

---

## Handling Duplicate Columns

```python
# Polars adds a suffix to duplicate column names
result = df1.join(df2, on="town", how="left", suffix="_right")

# Columns: town, price, population, price_right
# (if both tables had a "price" column)
```

The `suffix` parameter controls naming of duplicated columns.

---

## Full Example: Enriched Analysis

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
transactions = loader.load("ascent01", "hdbprices.csv")
demographics = loader.load("ascent01", "town_demographics.csv")

enriched = (
    transactions
    .join(demographics, on="town", how="left")
    .with_columns(
        (pl.col("price") / pl.col("median_income")).alias("affordability")
    )
    .group_by("town")
    .agg(pl.col("affordability").mean().alias("avg_affordability"))
    .sort("avg_affordability")
)
print(enriched)
```

---

## Exercise Preview

**Exercise 1.4: Multi-Table HDB Analysis**

You will:

1. Load transaction and demographic datasets
2. Join them using inner and left joins
3. Create affordability metrics from combined data
4. Handle missing data from unmatched joins
5. Build a dictionary-based town profile report

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                    | Fix                                                  |
| -------------------------- | ---------------------------------------------------- |
| Wrong join type            | Inner drops unmatched rows; left keeps them          |
| Duplicate column confusion | Use `suffix` parameter to clarify                    |
| Joining on wrong column    | Verify with `df.columns` and `head()` before joining |
| Dictionary `KeyError`      | Use `.get(key, default)` instead of `[key]`          |
| Forgetting `elif`          | Using multiple `if` statements checks all conditions |

---

## Summary

- Dictionaries store key-value pairs for structured lookups
- `if`/`elif`/`else` handles conditional logic
- `when().then().otherwise()` is the Polars-native conditional
- `join()` combines tables: `inner` for matches only, `left` for all primary data
- Multi-key joins ensure precise matching across tables

---

## Next Lesson

**Lesson 1.5: Window Functions and Trends**

We will learn:

- Window functions with `over()`
- Rolling averages with `rolling_mean()`
- Temporal shifts with `shift()`
- Lazy frames for optimised computation
