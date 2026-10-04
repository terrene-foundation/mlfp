# MLFP02 — Task 5: Point-in-Time Features with the Kailash FeatureStore

**Weight**: 15 marks · **Outcomes**: 2.8 (feature engineering with a domain rationale, FeatureStore lifecycle, point-in-time correctness)
**Data**: `mlfp01/hdb_resale.parquet` (raw HDB resale transactions).

## Scenario

The valuation model from Task 3 should know what the local market looked like
when each flat was sold. The data team wants a **town-level market feature**
published to the team's Kailash FeatureStore, so that any model — training
or live pricing — can ask "what did we know about this town at time _t_?" and
get the same answer.

Feature definition (agreed with the valuers):

- For every town and every calendar month _M_ between the earliest and latest
  sale month present in the input, the features **as of the start of month
  _M_** describe that town's sales in the **six calendar months before _M_**
  (months _M−6_ … _M−1_):
  - `median_psm_6m` — median price per square metre (`resale_price / floor_area_sqm`);
  - `volume_6m` — number of sales.
- Only genuine sales count: resale prices outside S$100,000–S$2,000,000 and
  sales dated before the lease commenced are recording errors.
- A town-month with no sales in its window gets **no row** (no nulls).

## What to submit

`starter.py` with three functions (signatures fixed):

| Function                                                   | Returns / does                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `town_month_features(transactions)`                        | polars DataFrame with columns `town` (str), `as_of` (Datetime, first instant of month _M_), `median_psm_6m` (Float64), `volume_6m` (Int64) — one row per town-month as defined above                                                                                                                                                                     |
| `publish_features(features, store_url, town_codes)`        | materialises the features into a Kailash `FeatureStore` backed by the DataFlow database at `store_url`, keyed by the integer `town_id = town_codes[town]` with event time `as_of`, and returns the `kailash_ml.features.FeatureSchema` used (entity column `town_id`, timestamp column `as_of`, fields `median_psm_6m` and `volume_6m`)                  |
| `features_for_sales(sales, store_url, schema, town_codes)` | for each row of `sales` (columns include `txn_id`, `town`, `month`), the features **read from the FeatureStore at `store_url`** as known at the start of the sale month (the most recent features published at or before that instant). Returns `txn_id`, `median_psm_6m`, `volume_6m`; features are null when the store knows nothing yet for that town |

The store is single-tenant: use the tenant id `"_single"`.

## Acceptance criteria

- The grader runs your build on a **secret** set of towns and months and
  compares every row with an independent implementation.
- It rebuilds with the most recent months **removed** — every feature row
  that both builds produce must be identical. A feature that uses the sale
  month itself or anything later fails.
- It opens your store with its own `FeatureStore` and checks as-of snapshots
  at secret timestamps.
- It serves held-out sales from a store **it** has written (with values you
  cannot recompute), so serving must really read the store.

## Rules

- Polars only (no pandas). Use `kailash_ml.features` (`FeatureSchema`,
  `FeatureField`, `FeatureGroup`, `FeatureStore`) with `dataflow.DataFlow`;
  no raw SQL. The FeatureStore API is async.
- Materialising is slow (roughly a dozen rows a second on a laptop) —
  develop on two or three towns and a two-year window. Give DataFlow an
  absolute `sqlite:///` path.
