# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.7: Geo Features — Putting the Model on the Map
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Engineer location features when the dataset has no coordinates:
#     a town-centroid lookup + haversine distance to the CBD
#   - Be honest about proxy granularity (all flats in a town share one
#     point — the feature is a TOWN effect measured in km)
#   - Read a distance gradient as $/km and as % per km
#   - Test an interaction: does distance matter MORE for larger flats?
#   - Validate the geo lift with 5-fold CV (not just in-sample R²)
#
# PREREQUISITES: 01_ols_from_scratch.py, 06_kfold_cv.py (CV discipline)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — location as a feature, and the proxy's limits
#   2. Build — town centroids, haversine distance to Raffles Place
#   3. Train — base vs +distance vs +interaction, CV-validated
#   4. Visualise — the price-distance gradient on real transactions
#   5. Apply — a valuer's $/km rule and where it breaks
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from shared.mlfp02.ex_5 import (
    CBD_RAFFLES_PLACE,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET,
    TOWN_CENTROIDS,
    add_geo_features,
    build_design_matrix,
    fit_ols,
    kfold_indices,
    load_hdb_clean,
    ols_r2_on,
    print_coef_table,
    track_train_run,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Location as a Feature, and the Proxy's Limits
# ════════════════════════════════════════════════════════════════════════
# "Location, location, location" is a regression problem: how much of the
# price is WHERE the flat is? The resale file names the town but carries
# no coordinates. The teaching proxy here is a TOWN-CENTROID lookup:
# every flat inherits its town centre's (lat, lon), and the feature is
# the haversine distance from that centre to Raffles Place (CBD):
#
#   haversine: d = 2R·arcsin(√(sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2)))
#
# Honesty about the proxy: all flats in a town share one point, so
# "dist_to_cbd" is really a TOWN fixed effect expressed in kilometres.
# It cannot see a premium block beside an MRT inside the town. That is
# the granularity trade you make explicit when you report the model —
# a production pipeline would geocode block + street (OneMap) instead.
#
# Why distance rather than 27 town dummies? Distance is ONE parameter
# with a shape (a gradient); dummies are 26 parameters with no shape.
# When the gradient is real, one parameter generalises better and reads
# directly: "each km from the CBD is associated with $β less".


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Town centroids and the distance feature
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.7: Geo Features")
print("=" * 70)

hdb_all = load_hdb_clean()
# TODO: sentinel-price hygiene — keep resale prices in [100_000, 5_000_000]
hdb = hdb_all.filter(____)
# TODO: add town centroid lat/lon and haversine distance-to-CBD, then
# drop rows whose town has no centroid match
# Hint: add_geo_features(hdb).drop_nulls(subset=["dist_to_cbd_km"])
geo = ____

print(f"\n  Rows with geo features: {geo.height:,} "
      f"(of {hdb.height:,} cleaned; towns matched: {len(TOWN_CENTROIDS)})")
print(f"  CBD reference: Raffles Place {CBD_RAFFLES_PLACE}")
dist = geo["dist_to_cbd_km"].to_numpy()
print(f"  Distance to CBD: min {dist.min():.1f} km, median "
      f"{np.median(dist):.1f} km, max {dist.max():.1f} km")

# The raw gradient, no model: mean price by town vs distance
# TODO: per-town summary — mean resale_price, the town's distance, and
# row count, sorted by distance
# Hint: geo.group_by("town").agg(pl.col(TARGET).mean().alias("mean_price"),
#   pl.col("dist_to_cbd_km").first().alias("dist"),
#   pl.len().alias("n")).sort("dist")
town_stats = ____
print(f"\n{'Town':<18} {'km to CBD':>9} {'Mean price':>12} {'n':>7}")
print("-" * 50)
for row in town_stats.head(6).iter_rows(named=True):
    print(f"{row['town']:<18} {row['dist']:>9.1f} {row['mean_price']:>12,.0f} {row['n']:>7,}")
print("  ...")
for row in town_stats.tail(3).iter_rows(named=True):
    print(f"{row['town']:<18} {row['dist']:>9.1f} {row['mean_price']:>12,.0f} {row['n']:>7,}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert geo.height > 20_000, "Most cleaned rows should match a town centroid"
assert (geo["dist_to_cbd_km"] >= 0).all(), "Distances must be non-negative"
assert geo["dist_to_cbd_km"].max() < 30, "Singapore spans under 30 km"
print("\n--- Checkpoint 1 passed --- geo features built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Base vs +distance vs +interaction, CV-validated
# ════════════════════════════════════════════════════════════════════════

X_base, y, names_base = build_design_matrix(geo)
dist_col = geo["dist_to_cbd_km"].to_numpy().astype(np.float64)
area_col = geo["floor_area_sqm"].to_numpy().astype(np.float64)

# TODO: design matrix with the distance column appended
# Hint: np.column_stack([X_base, dist_col])
X_geo = ____
# TODO: design matrix with distance AND the distance×area interaction
# (divide the product by 100 to keep its coefficient readable)
X_full = ____
names_geo = [*names_base, "dist_to_cbd_km"]
names_full = [*names_base, "dist_to_cbd_km", "dist×area/100"]


def cv5_r2(Xm: np.ndarray) -> np.ndarray:
    return np.array(
        [ols_r2_on(Xm, y, tr, te) for tr, te in kfold_indices(len(y), 5, 42)]
    )


# TODO: fit all three models and compute 5-fold CV R² for each
# Hint: fit_ols(X_base, y) … cv5_r2(X_base) …
fit_base = ____
fit_geo = ____
fit_full = ____
cv_base = ____
cv_geo = ____
cv_full = ____

print("=== In-sample ===")
print(f"  Base R²:          {fit_base['R2']:.4f}")
print(f"  +distance R²:     {fit_geo['R2']:.4f}")
print(f"  +interaction R²:  {fit_full['R2']:.4f}")
print("\n=== 5-fold CV (out-of-sample) ===")
print(f"  Base:          {cv_base.mean():.4f} ± {cv_base.std(ddof=1):.4f}")
print(f"  +distance:     {cv_geo.mean():.4f} ± {cv_geo.std(ddof=1):.4f}")
print(f"  +interaction:  {cv_full.mean():.4f} ± {cv_full.std(ddof=1):.4f}")

print("\n=== +distance model coefficients ===")
print_coef_table(names_geo, fit_geo)
# TODO: pull the distance coefficient out of the fitted model
# Hint: fit_geo["beta"][names_geo.index("dist_to_cbd_km")]
beta_km = ____
print(f"\nDistance gradient: {beta_km:,.0f} $/km "
      f"(ceteris paribus — holding size, storey, lease constant)")

# ── Log the fits to ExperimentTracker ────────────────────────────────
# TODO: Log the three CV means, the $/km gradient, and the full
# in-sample R²
# Hint: track_train_run(experiment=..., run_name=..., params={...}, metrics={...})
run_id = track_train_run(
    experiment="mlfp02_ex5_07_geo_features",
    run_name="base_vs_distance_vs_interaction",
    params={
        "geo_proxy": "town_centroid_haversine",
        "cbd": "raffles_place",
        "cv": "5-fold seed=42",
    },
    metrics={
        "cv_base": ____,
        "cv_distance": ____,
        "cv_interaction": ____,
        "beta_per_km": ____,
        "insample_r2_full": ____,
    },
)
print(f"\nLogged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert beta_km < 0, "Price should FALL with distance from the CBD"
assert cv_geo.mean() > cv_base.mean(), "Distance must add out-of-sample signal"
print("\n--- Checkpoint 2 passed --- geo models fitted and validated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: The price-distance gradient
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
# TODO: scatter of town mean price vs distance, marker size √n, town
# names as hover text
# Hint: go.Scatter(x=town_stats["dist"].to_numpy(),
#   y=town_stats["mean_price"].to_numpy(), mode="markers",
#   marker={"size": np.sqrt(town_stats["n"].to_numpy()) / 2, ...},
#   text=town_stats["town"].to_numpy(), name=...)
fig.add_trace(____)
# Model-implied gradient at the median feature values
median_area = float(np.median(area_col))
median_storey = float(np.median(geo["storey_midpoint"].to_numpy()))
median_lease = float(np.median(geo["remaining_lease_years"].to_numpy()))
d_grid = np.linspace(0, 25, 100)
# TODO: the ceteris-paribus gradient line — intercept + medians × their
# betas + beta_km × d_grid
gradient_line = ____
fig.add_trace(
    go.Scatter(
        x=d_grid,
        y=gradient_line,
        mode="lines",
        name=f"Model gradient at medians ({beta_km:,.0f} $/km)",
        line={"color": "red", "dash": "dash"},
    )
)
fig.update_layout(
    title="The Price-Distance Gradient: Town Means vs the Model's Ceteris-Paribus Line",
    xaxis_title="Town-centroid distance to Raffles Place (km)",
    yaxis_title="Mean resale price ($)",
    height=480,
)
fig_path = OUTPUT_DIR / "geo_gradient.html"
fig.write_html(str(fig_path))
print(f"Saved: {fig_path}")
# INTERPRETATION: the dashed line is ceteris paribus — distance at FIXED
# flat characteristics. Towns above the line (e.g. mature estates with
# larger average flats) are expensive for reasons the distance feature
# alone cannot see; towns below it are cheaper than their location
# predicts. The scatter AROUND the line is why town dummies still win
# in-sample — and why they cost 26 parameters.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert fig_path.exists(), "Figure must be written"
print("\n--- Checkpoint 3 passed --- visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A Valuer's $/km Rule — and Where It Breaks
# ════════════════════════════════════════════════════════════════════════
# A valuation firm (anonymised) wants a one-line desk rule: "prices drop
# $X per km from the CBD." The +distance model gives the number; the
# town residuals tell the desk where the rule is unsafe.

print("=== APPLICATION: The $/km Desk Rule ===")
print(f"\n  Desk rule: {beta_km:,.0f} $/km, ceteris paribus")
# TODO: the dollar effect of moving 10 km outward, and as a share of the
# mean price
ten_km = ____
print(f"  A 10 km move outward: {ten_km:,.0f} $ "
      f"({ten_km / y.mean():+.1%} of the mean price)")

# Where the rule breaks: largest town-level residuals from the +distance
# model's town means
town_pred = (
    fit_geo["beta"][0]
    + fit_geo["beta"][1] * median_area
    + fit_geo["beta"][2] * median_storey
    + fit_geo["beta"][3] * median_lease
    + beta_km * town_stats["dist"].to_numpy()
)
# TODO: town residuals (actual mean minus gradient prediction) and the
# ranking by absolute size
# Hint: resid = town_stats["mean_price"].to_numpy() - town_pred;
#       order = np.argsort(np.abs(resid))[::-1]
resid = ____
order = ____
town_names = town_stats["town"].to_list()
town_dists = town_stats["dist"].to_list()
town_means = town_stats["mean_price"].to_list()
print(f"\n  Towns the gradient misprices most (at median characteristics):")
for i in order[:5]:
    i = int(i)
    print(
        f"    {town_names[i]:<18} {town_dists[i]:>5.1f} km  "
        f"actual {town_means[i]:>10,.0f}  gradient {town_pred[i]:>10,.0f}  "
        f"residual {resid[i]:>+10,.0f}"
    )
print(
    "\n  Positive residuals are premiums the distance proxy cannot see:\n"
    "  mature-estate amenities, school zones, MRT adjacency. The desk\n"
    "  rule is a first pass — the residuals are the watchlist for where\n"
    "  a town-level adjustment (or a real geocode) must override it."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(order) == town_stats.height, "Residual ranking must cover all towns"
assert np.isfinite(resid).all(), "Residuals must be finite"
print("\n--- Checkpoint 4 passed --- application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (5.7 — Geo Features)")
print("═" * 70)
print(
    """
  ✓ No coordinates in the data? A town-centroid lookup + haversine
    builds a real distance feature — with the granularity caveat stated
  ✓ One distance parameter generalises where 26 town dummies memorise:
    the CV comparison quantifies exactly what the gradient buys
  ✓ The coefficient IS the desk rule: $/km, ceteris paribus
  ✓ Town residuals from the gradient are the honest failure map —
    premiums the proxy cannot see get listed, not hidden
  ✓ Interactions (dist × area) test whether the gradient itself tilts
    with flat size — and CV decides if the tilt generalises

  NEXT: Exercise 6 moves from continuous prices to BINARY outcomes —
  logistic regression, odds ratios, and classification metrics.
"""
)

print("\n✓ Exercise 5.7 complete — Geo Features")
