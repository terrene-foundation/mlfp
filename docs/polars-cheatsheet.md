# Pandas to Polars Cheatsheet

This course uses polars exclusively. Here are the key differences from pandas.

---

## Import

```python
# pandas
import pandas as pd

# polars
import polars as pl
```

## Read Data

```python
# pandas
df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")

# polars
df = pl.read_csv("data.csv")
df = pl.read_parquet("data.parquet")
```

## Basic Operations

```python
# pandas: df.head(), df.info(), df.describe()
# polars:
df.head()
df.describe()
df.schema        # column types
df.shape         # (rows, cols)
df.null_count()  # missing values per column
```

## Column Selection

```python
# pandas
df["col"]           # Series
df[["col1", "col2"]]  # DataFrame

# polars
df["col"]           # Series
df.select("col1", "col2")  # DataFrame
df.select(pl.col("col1"), pl.col("col2"))  # Explicit
```

## Filtering

```python
# pandas
df[df["age"] > 30]
df[(df["age"] > 30) & (df["city"] == "Singapore")]

# polars
df.filter(pl.col("age") > 30)
df.filter((pl.col("age") > 30) & (pl.col("city") == "Singapore"))
```

## Aggregation

```python
# pandas
df.groupby("city")["price"].mean()
df.groupby("city").agg({"price": "mean", "qty": "sum"})

# polars
df.group_by("city").agg(pl.col("price").mean())
df.group_by("city").agg(
    pl.col("price").mean().alias("avg_price"),
    pl.col("qty").sum().alias("total_qty"),
)
```

## Sorting

```python
# pandas
df.sort_values("price", ascending=False)

# polars
df.sort("price", descending=True)
```

## New Columns

```python
# pandas
df["total"] = df["price"] * df["qty"]

# polars
df = df.with_columns(
    (pl.col("price") * pl.col("qty")).alias("total")
)
```

## Missing Values

```python
# pandas
df.isna().sum()
df.fillna(0)
df.dropna()

# polars
df.null_count()
df.fill_null(0)
df.drop_nulls()
```

## Join / Merge

```python
# pandas
pd.merge(df1, df2, on="id", how="left")

# polars
df1.join(df2, on="id", how="left")
```

## Type Casting

```python
# pandas
df["col"].astype(float)

# polars
df.with_columns(pl.col("col").cast(pl.Float64))
```

## Lazy Evaluation (polars-only superpower)

```python
# Polars can defer execution for optimization
result = (
    pl.scan_csv("large_file.csv")    # lazy — no data loaded yet
    .filter(pl.col("country") == "SG")
    .group_by("city")
    .agg(pl.col("price").mean())
    .collect()                         # execute optimized plan
)
```

## Key Differences

| Feature | pandas | polars |
|---------|--------|--------|
| Mutability | Mutable (in-place ops) | Immutable (returns new df) |
| Index | Has index | No index |
| Missing | NaN | null |
| Speed | Single-threaded | Multi-threaded by default |
| Memory | Higher (copies) | Lower (Arrow-backed) |
| Lazy eval | No | Yes (scan_*) |
