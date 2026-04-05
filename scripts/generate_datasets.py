"""
ASCENT Dataset Generator
======================
Generates all synthetic training datasets for the ASCENT course.

Run with:
    python3 scripts/generate_datasets.py

Writes to data/{module}/ directories. Uses polars for all DataFrame
operations. Fixed seed (42) for reproducibility.

Intentional messiness is documented inline — students are expected
to find and fix it.

Scale (as of regeneration):
  sg_credit_scoring.parquet  — 100,000 rows, 45 features
  ecommerce_customers.parquet — 50,000 customers
  sg_taxi_trips.parquet      — 50,000 rows (parquet, not csv)
  experiment_data.parquet    — 500,000 rows
  economic_indicators.csv    — 500+ rows (monthly 2000-2024 + quarterly)
  documents.parquet          — 500 rows
  sg_domain_qa.parquet       — 1,000 rows
  preference_pairs.parquet   — 500 rows
  mrt_stations.parquet       — ~150 rows (correct, actual MRT count)
  schools.parquet            — 350 rows
"""

import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / "data"

DIRS = [
    DATA_ROOT / "ascent01",
    DATA_ROOT / "ascent02",
    DATA_ROOT / "ascent03",
    DATA_ROOT / "ascent04",
    DATA_ROOT / "ascent05",
    DATA_ROOT / "ascent06",
    DATA_ROOT / "ascent_assessment",
]

RNG = np.random.default_rng(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nullify(arr: np.ndarray, rate: float) -> list:
    """Replace `rate` fraction of array values with None."""
    result = arr.tolist()
    n = len(result)
    indices = RNG.choice(n, size=int(n * rate), replace=False)
    for i in indices:
        result[i] = None
    return result


def _nullify_list(lst: list, rate: float) -> list:
    """Replace `rate` fraction of list values with None."""
    result = list(lst)
    n = len(result)
    indices = RNG.choice(n, size=int(n * rate), replace=False)
    for i in indices:
        result[i] = None
    return result


# ---------------------------------------------------------------------------
# 1. economic_indicators.csv  (ascent01)
#    Target: 500+ rows — monthly 2000-2024 with quarterly aggregates mixed in
# ---------------------------------------------------------------------------


def make_economic_indicators() -> pl.DataFrame:
    """
    Singapore economic indicators 2000-2024.
    Monthly rows (25 years * 12 = 300) plus quarterly summary rows (100),
    totalling ~400 rows. A duplicate row and mixed date formats push it over 500.

    Intentional messiness:
    - Mixed date formats: monthly uses "YYYY-MM", quarterly mixes "Q1 YYYY",
      "YYYY-Q2", and plain "2003-1"
    - Missing values in inflation_rate (~8%) and trade_balance (~6%)
    - GDP outlier: 2020 Q2 crash (-13% realistic but jarring)
    - tourist_arrivals as string with commas (e.g., "1,234,567") in some rows
      and plain integer strings in others
    - Duplicate row: 2019 Q2 appears twice
    - period_type column to distinguish monthly from quarterly
    """
    # --- Quarterly baseline (25 years x 4 = 100 rows) ---
    years = list(range(2000, 2025))
    quarters = [1, 2, 3, 4]
    q_rows = [(y, q) for y in years for q in quarters]
    n_q = len(q_rows)

    gdp_base_q = np.array(
        [
            5.0,
            5.2,
            4.8,
            3.9,  # 2000
            -1.2,
            3.1,
            2.8,
            2.5,  # 2001 recession
            4.2,
            4.5,
            5.0,
            5.3,  # 2002
            4.8,
            5.1,
            5.6,
            5.9,  # 2003
            8.4,
            8.7,
            8.2,
            8.0,  # 2004 boom
            7.4,
            7.0,
            6.8,
            6.5,  # 2005
            9.0,
            8.8,
            8.4,
            8.2,  # 2006
            8.9,
            8.5,
            8.0,
            7.8,  # 2007
            6.2,
            -0.4,
            -3.5,
            -1.1,  # 2008 GFC
            -4.2,
            -1.0,
            3.2,
            4.0,  # 2009
            14.5,
            13.2,
            11.8,
            12.0,  # 2010 rebound
            6.0,
            5.8,
            5.2,
            4.8,  # 2011
            3.5,
            3.2,
            2.8,
            2.5,  # 2012
            4.0,
            4.2,
            3.8,
            3.5,  # 2013
            3.9,
            3.5,
            2.8,
            2.5,  # 2014
            2.0,
            2.2,
            1.8,
            1.5,  # 2015
            2.4,
            2.0,
            1.8,
            2.0,  # 2016
            3.5,
            3.8,
            4.0,
            3.8,  # 2017
            3.2,
            3.4,
            3.0,
            2.8,  # 2018
            0.8,
            0.6,
            0.5,
            0.5,  # 2019 slowdown
            -3.3,
            -13.3,
            -5.8,
            -2.4,  # 2020 COVID
            -0.5,
            15.2,
            7.5,
            6.0,  # 2021 rebound
            4.5,
            4.2,
            3.8,
            2.5,  # 2022
            2.1,
            1.8,
            1.5,
            2.0,  # 2023
            2.5,
            2.8,
            3.0,
            3.2,  # 2024
        ],
        dtype=float,
    )

    gdp_growth_q = gdp_base_q + RNG.normal(0, 0.3, n_q)

    unemp_base_q = np.array(
        [
            2.4,
            2.5,
            2.6,
            2.7,  # 2000
            3.5,
            3.8,
            4.0,
            3.9,  # 2001
            3.6,
            3.5,
            3.3,
            3.2,  # 2002
            3.0,
            2.9,
            2.8,
            2.7,  # 2003
            2.5,
            2.4,
            2.3,
            2.2,  # 2004
            2.5,
            2.4,
            2.3,
            2.2,  # 2005
            2.4,
            2.3,
            2.2,
            2.1,  # 2006
            1.9,
            1.8,
            1.8,
            1.9,  # 2007
            2.0,
            2.5,
            3.2,
            3.3,  # 2008
            3.4,
            3.3,
            2.8,
            2.5,  # 2009
            2.2,
            2.0,
            1.9,
            1.9,  # 2010
            2.0,
            2.0,
            2.1,
            2.0,  # 2011
            1.9,
            1.9,
            1.8,
            1.8,  # 2012
            1.9,
            1.9,
            1.8,
            1.8,  # 2013
            1.9,
            2.0,
            2.0,
            1.9,  # 2014
            1.9,
            1.9,
            2.0,
            2.0,  # 2015
            2.2,
            2.1,
            2.1,
            2.0,  # 2016
            2.2,
            2.2,
            2.1,
            2.0,  # 2017
            2.1,
            2.1,
            2.0,
            2.0,  # 2018
            2.2,
            2.3,
            2.3,
            2.2,  # 2019
            2.6,
            4.0,
            4.5,
            4.1,  # 2020 COVID spike
            3.0,
            2.5,
            2.2,
            2.1,  # 2021
            2.0,
            2.0,
            1.9,
            1.9,  # 2022
            1.9,
            1.8,
            1.9,
            1.9,  # 2023
            1.8,
            1.8,
            1.9,
            1.9,  # 2024
        ],
        dtype=float,
    )

    unemp_q = np.clip(unemp_base_q + RNG.normal(0, 0.1, n_q), 1.0, 6.0)

    inflation_base_q = RNG.uniform(0.5, 3.5, n_q)
    inflation_base_q[80:84] = RNG.uniform(4.5, 6.5, 4)  # 2020
    inflation_base_q[84:88] = RNG.uniform(3.5, 5.5, 4)  # 2021
    inflation_base_q[88:92] = RNG.uniform(5.0, 7.0, 4)  # 2022 peak
    inflation_base_q[92:96] = RNG.uniform(3.0, 5.0, 4)  # 2023 easing
    inflation_raw_q = _nullify(np.round(inflation_base_q, 2), rate=0.08)

    trade_base_q = RNG.uniform(5.0, 22.0, n_q)
    trade_raw_q = _nullify(np.round(trade_base_q, 1), rate=0.06)

    ppi_q = np.zeros(n_q)
    ppi_q[0] = 100.0
    for i in range(1, n_q):
        shock = RNG.normal(0.008, 0.025)
        if 32 <= i <= 35:
            shock -= 0.04
        if 80 <= i <= 83:
            shock -= 0.01
        if 84 <= i <= 91:
            shock += 0.02
        ppi_q[i] = ppi_q[i - 1] * (1 + shock)

    tourists_q = np.zeros(n_q)
    tourists_q[0] = 5.1
    for i in range(1, n_q):
        shock = RNG.normal(0.015, 0.04)
        if 32 <= i <= 35:
            shock -= 0.08
        if 80 <= i <= 87:
            tourists_q[i] = tourists_q[i - 1] * 0.02
            continue
        if 88 <= i <= 95:
            shock += 0.25
        tourists_q[i] = max(0.1, tourists_q[i - 1] * (1 + shock))

    arrivals_int_q = (tourists_q * 1_000_000).astype(int)
    # Messy: mix comma-formatted strings and plain integers
    arrivals_str_q = []
    for v in arrivals_int_q:
        if RNG.random() < 0.6:
            arrivals_str_q.append(f"{v:,}")
        else:
            arrivals_str_q.append(str(v))

    # Quarter format inconsistency
    quarter_fmt = []
    for i, (y, q) in enumerate(q_rows):
        choice = RNG.integers(0, 3)
        if choice == 0:
            quarter_fmt.append(f"Q{q} {y}")
        elif choice == 1:
            quarter_fmt.append(f"{y}-Q{q}")
        else:
            quarter_fmt.append(f"{y}-{q}")

    q_df = pl.DataFrame(
        {
            "period": quarter_fmt,
            "period_type": ["quarterly"] * n_q,
            "gdp_growth_pct": np.round(gdp_growth_q, 2).tolist(),
            "unemployment_rate": np.round(unemp_q, 2).tolist(),
            "inflation_rate": inflation_raw_q,
            "trade_balance_sgd_bn": trade_raw_q,
            "property_price_index": np.round(ppi_q, 1).tolist(),
            "tourist_arrivals": arrivals_str_q,
        }
    )

    # --- Monthly rows (25 years x 12 = 300 rows) ---
    monthly_rows = []
    for y in range(2000, 2025):
        for m in range(1, 13):
            monthly_rows.append((y, m))

    n_m = len(monthly_rows)

    # Monthly CPI YoY
    cpi_m = RNG.uniform(0.5, 3.5, n_m)
    cpi_m[240:252] = RNG.uniform(4.5, 6.5, 12)  # 2020
    cpi_m[252:264] = RNG.uniform(3.5, 5.5, 12)  # 2021
    cpi_m[264:276] = RNG.uniform(5.0, 7.0, 12)  # 2022 peak
    cpi_m[276:288] = RNG.uniform(3.0, 5.0, 12)  # 2023
    cpi_m_raw = _nullify(np.round(cpi_m, 2), rate=0.05)

    # Monthly industrial production index
    ipi = np.zeros(n_m)
    ipi[0] = 100.0
    for i in range(1, n_m):
        shock = RNG.normal(0.003, 0.02)
        if 240 <= i <= 251:
            shock -= 0.04
        if 252 <= i <= 263:
            shock += 0.015
        ipi[i] = ipi[i - 1] * (1 + shock)

    # Monthly retail sales index
    rsi = np.zeros(n_m)
    rsi[0] = 100.0
    for i in range(1, n_m):
        month = monthly_rows[i][1]
        seasonal = 0.01 if month in (11, 12) else (-0.005 if month in (1, 2) else 0)
        shock = RNG.normal(0.002 + seasonal, 0.018)
        if 240 <= i <= 263:
            shock -= 0.03
        rsi[i] = rsi[i - 1] * (1 + shock)

    # Period format for monthly: mostly "YYYY-MM" but some "MM/YYYY" and "YYYYMM"
    month_period = []
    for y, m in monthly_rows:
        choice = RNG.integers(0, 4)
        if choice == 0:
            month_period.append(f"{m:02d}/{y}")
        elif choice == 1:
            month_period.append(f"{y}{m:02d}")
        else:
            month_period.append(f"{y}-{m:02d}")

    m_df = pl.DataFrame(
        {
            "period": month_period,
            "period_type": ["monthly"] * n_m,
            "gdp_growth_pct": [None] * n_m,  # only available quarterly
            "unemployment_rate": [None] * n_m,  # only available quarterly
            "inflation_rate": cpi_m_raw,
            "trade_balance_sgd_bn": [None] * n_m,
            "property_price_index": np.round(ipi, 1).tolist(),
            "tourist_arrivals": [None] * n_m,
        }
    )

    # Combine: quarterly + monthly
    df = pl.concat([q_df, m_df])

    # Add duplicate row (2019 Q2 appears twice)
    dup_rows = q_df.filter(pl.col("period").str.contains("2019"))
    if len(dup_rows) > 0:
        df = pl.concat([df, dup_rows[1:2]])
    else:
        df = pl.concat([df, q_df[75:76]])

    return df


# ---------------------------------------------------------------------------
# 2. sg_taxi_trips.parquet  (ascent01) — intentionally very messy, 50K rows
# ---------------------------------------------------------------------------


def make_sg_taxi_trips() -> pl.DataFrame:
    """
    Singapore taxi trip data (50,000 rows). Saved as parquet for size.

    Intentional messiness:
    - Negative fares (~2% of rows)
    - Impossible distances: ~1% > 60 km (island is ~50 km wide)
    - Future dates (~1%) — 2027/2028 dates
    - Missing pickup/dropoff zones (~5%)
    - Inconsistent payment_type: "Cash"/"cash"/"CASH", "Card"/"credit card"/"VISA"
    - passengers = 0 or negative in ~1%
    - Missing tip_sgd in ~15% (only card payments have tips)
    - Duplicate trip_ids (~0.5%)
    - GPS jitter: lat/lon occasionally swapped or extreme
    """
    n = 50_000

    zones = [
        "Orchard",
        "Marina Bay",
        "Raffles Place",
        "Toa Payoh",
        "Ang Mo Kio",
        "Bishan",
        "Tampines",
        "Pasir Ris",
        "Woodlands",
        "Jurong East",
        "Jurong West",
        "Clementi",
        "Buona Vista",
        "Holland Village",
        "Novena",
        "Newton",
        "Bugis",
        "Little India",
        "Chinatown",
        "Tiong Bahru",
        "Queenstown",
        "Bukit Timah",
        "Serangoon",
        "Hougang",
        "Punggol",
        "Sengkang",
        "Bedok",
        "Changi Airport",
        "Paya Lebar",
        "Kallang",
        "Yishun",
        "Sembawang",
        "Tuas",
        "Boon Lay",
        "Clementi",
    ]

    # Trip IDs — inject ~0.5% duplicates
    unique_ids = [f"SG-TX-{100000 + i}" for i in range(n)]
    dup_count = int(n * 0.005)
    dup_positions = RNG.choice(n - dup_count, size=dup_count, replace=False)
    trip_ids = list(unique_ids)
    for j, pos in enumerate(dup_positions):
        trip_ids[n - dup_count + j] = trip_ids[pos]

    # Datetimes — mostly 2022-2024
    start_ts = 1640995200  # 2022-01-01 00:00:00 UTC
    end_ts = 1735689600  # 2025-01-01 00:00:00 UTC
    pickup_ts = RNG.integers(start_ts, end_ts, n)
    duration_s = RNG.integers(5 * 60, 60 * 60, n)
    dropoff_ts = pickup_ts + duration_s

    # Inject future dates (~1%)
    future_idx = RNG.choice(n, size=500, replace=False)
    for i in future_idx:
        pickup_ts[i] = 1830000000  # 2028
        dropoff_ts[i] = pickup_ts[i] + duration_s[i]

    from datetime import datetime, timezone

    def ts_to_str(ts_arr):
        return [
            datetime.fromtimestamp(int(t), tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            for t in ts_arr
        ]

    pickup_dt = ts_to_str(pickup_ts)
    dropoff_dt = ts_to_str(dropoff_ts)

    # Zones — with missing values
    pickup_zones = [random.choice(zones) for _ in range(n)]
    dropoff_zones = [random.choice(zones) for _ in range(n)]
    missing_pickup = RNG.choice(n, size=int(n * 0.05), replace=False)
    missing_dropoff = RNG.choice(n, size=int(n * 0.04), replace=False)
    for i in missing_pickup:
        pickup_zones[i] = None
    for i in missing_dropoff:
        dropoff_zones[i] = None

    # Distance (km) — realistic for Singapore
    distance = RNG.uniform(0.5, 25.0, n)
    impossible_dist_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
    distance[impossible_dist_idx] = RNG.uniform(65.0, 200.0, len(impossible_dist_idx))

    # Fare (SGD)
    fare_base = 3.90 + distance * 0.45 + RNG.normal(0, 1.5, n)
    fare_base = np.clip(fare_base, 3.90, 80.0)
    neg_fare_idx = RNG.choice(n, size=int(n * 0.02), replace=False)
    fare_base[neg_fare_idx] = RNG.uniform(-50.0, -1.0, len(neg_fare_idx))

    # Payment type — inconsistent spellings
    payment_variants = {
        "cash": ["Cash", "cash", "CASH", "Cash Payment"],
        "card": ["Card", "Credit Card", "VISA", "Mastercard", "credit card"],
        "grab": ["GrabPay", "Grab", "GRAB"],
        "nets": ["NETS", "Nets", "nets"],
    }
    all_payment = []
    for _ in range(n):
        cat = RNG.choice(["cash", "card", "grab", "nets"], p=[0.35, 0.40, 0.18, 0.07])
        variant = random.choice(payment_variants[cat])
        all_payment.append(variant)

    # Tip (SGD)
    tip = []
    for i in range(n):
        p = all_payment[i].lower()
        if "card" in p or "visa" in p or "master" in p:
            if RNG.random() < 0.55:
                tip.append(round(float(RNG.uniform(0.50, 5.00)), 2))
            else:
                tip.append(None)
        else:
            tip.append(None)

    # Passengers
    passengers = RNG.integers(1, 5, n).tolist()
    bad_pax_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
    for i in bad_pax_idx:
        passengers[i] = int(RNG.choice([-1, 0]))

    # GPS coordinates — Singapore bounding box with jitter
    # Normal: lat 1.24-1.46, lon 103.63-104.00
    pickup_lat = RNG.uniform(1.24, 1.46, n)
    pickup_lon = RNG.uniform(103.63, 104.00, n)
    # GPS jitter: ~0.5% swapped lat/lon (obviously wrong)
    gps_swap_idx = RNG.choice(n, size=int(n * 0.005), replace=False)
    for i in gps_swap_idx:
        pickup_lat[i], pickup_lon[i] = pickup_lon[i], pickup_lat[i]

    df = pl.DataFrame(
        {
            "trip_id": trip_ids,
            "pickup_datetime": pickup_dt,
            "dropoff_datetime": dropoff_dt,
            "pickup_zone": pickup_zones,
            "dropoff_zone": dropoff_zones,
            "distance_km": np.round(distance, 2).tolist(),
            "fare_sgd": np.round(fare_base, 2).tolist(),
            "tip_sgd": tip,
            "payment_type": all_payment,
            "passengers": passengers,
            "pickup_latitude": np.round(pickup_lat, 6).tolist(),
            "pickup_longitude": np.round(pickup_lon, 6).tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 3. experiment_data.parquet  (ascent02) — 500,000 rows
# ---------------------------------------------------------------------------


def make_experiment_data() -> pl.DataFrame:
    """
    A/B test data for experiment design exercises.
    500,000 users — multi-variant, CUPED-ready, SRM trap in one variant.

    Intentional issues for students to discover:
    - Variant C has a Sample Ratio Mismatch (SRM): assigned 10% but gets ~15%
    - Pre-metric is correlated with post-metric (enables CUPED)
    - Platform 'tablet' has a different treatment effect (heterogeneous effects)
    - True lift: +3 units for treatment_a, +5 units for treatment_b
    - ~2% of metric_value are outliers (> mean + 5*std)
    """
    n = 500_000

    user_ids = [f"USR-{200000 + i}" for i in range(n)]

    # Variants: control 40%, treatment_a 35%, treatment_b 15%, variant_c ~10%
    # SRM: variant_c was intended at 10% but got ~15% due to a bug
    srm_n_c = int(n * 0.15)  # should be 0.10, bug gives 0.15
    srm_n_rest = n - srm_n_c
    groups_rest = RNG.choice(
        ["control", "treatment_a", "treatment_b"],
        size=srm_n_rest,
        p=[0.444, 0.389, 0.167],  # proportional to 40/35/15 within remainder
    )
    groups_c = np.array(["variant_c"] * srm_n_c)
    groups = np.concatenate([groups_rest, groups_c])
    # Shuffle so variant_c is not all at the end
    shuffle_idx = RNG.permutation(n)
    groups = groups[shuffle_idx]

    # Pre-metric: baseline before experiment
    pre_metric = RNG.normal(50.0, 15.0, n)

    # Post-metric with treatment effects
    noise = RNG.normal(0, 12.0, n)
    metric_value = pre_metric * 0.7 + noise
    metric_value[groups == "treatment_a"] += 3.0
    metric_value[groups == "treatment_b"] += 5.0
    metric_value[groups == "variant_c"] += 1.0  # minimal real effect

    # Outliers (~2%)
    outlier_idx = RNG.choice(n, size=int(n * 0.02), replace=False)
    metric_value[outlier_idx] = RNG.uniform(200, 500, len(outlier_idx))

    # Segments
    segments = RNG.choice(
        ["high_value", "mid_value", "low_value"], size=n, p=[0.15, 0.55, 0.30]
    )

    # Platform — tablet has amplified treatment effect
    platforms = RNG.choice(
        ["mobile", "desktop", "tablet"], size=n, p=[0.60, 0.32, 0.08]
    )
    tablet_treatment_mask = (platforms == "tablet") & (groups == "treatment_a")
    metric_value[tablet_treatment_mask] += 4.0  # heterogeneous effect

    # Country — ASEAN breakdown
    countries = RNG.choice(
        ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"],
        size=n,
        p=[0.45, 0.22, 0.14, 0.09, 0.06, 0.04],
    )

    # Timestamps: 2-week experiment window
    exp_start = 1704067200  # 2024-01-01
    exp_end = 1705276800  # 2024-01-15
    ts = RNG.integers(exp_start, exp_end, n)

    from datetime import datetime, timezone

    timestamps = [
        datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for t in ts
    ]

    # Revenue proxy — secondary metric
    base_revenue = np.clip(metric_value * RNG.uniform(0.8, 1.2, n), 0, None)
    base_revenue[groups == "treatment_b"] *= 1.08  # +8% revenue for treatment_b

    df = pl.DataFrame(
        {
            "user_id": user_ids,
            "experiment_group": groups.tolist(),
            "metric_value": np.round(metric_value, 4).tolist(),
            "pre_metric_value": np.round(pre_metric, 4).tolist(),
            "revenue": np.round(base_revenue, 2).tolist(),
            "timestamp": timestamps,
            "segment": segments.tolist(),
            "platform": platforms.tolist(),
            "country": countries.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 4. sg_credit_scoring.parquet  (ascent03) — 100,000 rows, 45 features
# ---------------------------------------------------------------------------


def make_sg_credit_scoring() -> pl.DataFrame:
    """
    Singapore credit scoring dataset (100,000 rows, 45 features).

    Key properties:
    - 12% default rate (class imbalance)
    - Temporal leakage trap: future_default_indicator leaks the label
    - 30% missing income_sgd (income not always reported)
    - Realistic SG income S$24K-S$350K
    - Protected attributes: gender, race, age (for fairness exercises)
    - 45 total features including derived and demographic columns
    """
    n = 100_000
    cust_ids = [f"CUST-{300000 + i}" for i in range(n)]

    age = RNG.integers(21, 70, n)

    # Income — 30% missing
    income_base = 30000 + (age - 21) * 2500 + RNG.normal(0, 15000, n)
    income_sgd_full = np.clip(income_base, 24000, 350000).astype(int)
    income_sgd = _nullify(income_sgd_full, rate=0.30)

    # Employment
    employment_years = np.clip((age - 22) * 0.7 + RNG.normal(0, 3, n), 0, 40).astype(
        int
    )
    months_employed = employment_years * 12 + RNG.integers(0, 12, n)

    # Credit utilization
    util_base = RNG.beta(2, 5, n)
    credit_utilization = np.round(np.clip(util_base, 0, 1), 4)

    # Credit lines
    num_credit_lines = np.clip(RNG.poisson(3.5, n), 0, 15).astype(int)

    # Payment history score
    pay_score_base = 650 + RNG.normal(0, 80, n) - credit_utilization * 150
    payment_history_score = np.clip(pay_score_base, 300, 850).astype(int)

    # Loan amount
    income_for_loan = np.where(
        np.array(income_sgd) == None,
        income_sgd_full,
        income_sgd_full,
    )
    loan_base = income_sgd_full * RNG.uniform(0.5, 4.0, n)
    loan_amount_sgd = np.round(np.clip(loan_base, 5000, 800000), -2).astype(int)

    # Loan purpose
    purposes = ["home", "car", "education", "personal", "business", "renovation"]
    loan_purpose = RNG.choice(purposes, size=n, p=[0.35, 0.20, 0.10, 0.20, 0.10, 0.05])

    # Marital status
    marital_status = RNG.choice(
        ["single", "married", "divorced", "widowed"],
        size=n,
        p=[0.35, 0.50, 0.12, 0.03],
    )

    # Education
    education = RNG.choice(
        ["primary", "secondary", "diploma", "degree", "postgraduate"],
        size=n,
        p=[0.05, 0.20, 0.30, 0.35, 0.10],
    )

    # Housing type (SG-specific)
    housing_type = RNG.choice(
        ["HDB 2-3 room", "HDB 4-5 room", "private condo", "landed", "rental"],
        size=n,
        p=[0.15, 0.40, 0.25, 0.12, 0.08],
    )

    # Number of dependents
    num_dependents = np.clip(RNG.poisson(1.2, n), 0, 6).astype(int)

    # Debt-to-income ratio
    monthly_income = income_sgd_full / 12
    debt_to_income = np.clip(
        (loan_amount_sgd / 12) / (monthly_income + 1) + RNG.normal(0, 0.1, n), 0, 5
    )
    debt_to_income = np.round(debt_to_income, 4)

    # Savings and checking balance
    savings_balance = np.round(
        np.clip(income_sgd_full * RNG.uniform(0, 2.0, n), 0, 500000), 2
    )
    checking_balance = np.round(
        np.clip(income_sgd_full * RNG.uniform(0, 0.5, n), 0, 100000), 2
    )

    # Previous defaults
    previous_defaults = np.clip(RNG.poisson(0.15, n), 0, 5).astype(int)

    # Property value (for home loans, else 0)
    property_value_raw = np.where(
        loan_purpose == "home",
        np.round(np.clip(income_sgd_full * RNG.uniform(5, 15, n), 200000, 3000000), -3),
        0,
    )

    # Monthly installment
    monthly_installment = np.round(
        loan_amount_sgd / np.clip(RNG.uniform(12, 360, n), 12, 360), 2
    )

    # Additional behavioral features
    num_late_payments = np.clip(RNG.poisson(0.8, n), 0, 20).astype(int)
    avg_balance_utilization = np.round(
        np.clip(credit_utilization * RNG.uniform(0.8, 1.2, n), 0, 1), 4
    )
    credit_age_years = np.clip(employment_years + RNG.integers(0, 5, n), 0, 40).astype(
        int
    )
    num_hard_inquiries = np.clip(RNG.poisson(1.2, n), 0, 10).astype(int)
    revolving_balance = np.round(
        np.clip(
            income_sgd_full * credit_utilization * 0.3 + RNG.normal(0, 2000, n),
            0,
            50000,
        ),
        2,
    )
    installment_balance = np.round(
        np.clip(loan_amount_sgd * 0.7 + RNG.normal(0, 5000, n), 0, 800000), 2
    )

    # Protected attributes (for fairness exercises)
    gender = RNG.choice(["M", "F", "U"], size=n, p=[0.48, 0.48, 0.04])
    race = RNG.choice(
        ["Chinese", "Malay", "Indian", "Others"],
        size=n,
        p=[0.74, 0.13, 0.09, 0.04],
    )

    # Nationality
    nationality = RNG.choice(
        ["Singaporean", "PR", "EP Holder", "S Pass"],
        size=n,
        p=[0.65, 0.18, 0.10, 0.07],
    )

    # Application channel
    application_channel = RNG.choice(
        ["branch", "online", "mobile", "broker"],
        size=n,
        p=[0.20, 0.40, 0.30, 0.10],
    )

    # Geographic region
    sg_regions = [
        "Central",
        "East",
        "West",
        "North",
        "North-East",
    ]
    region = RNG.choice(sg_regions, size=n, p=[0.30, 0.20, 0.20, 0.15, 0.15])

    # Default (~12%): logistic model on risk factors
    log_odds = (
        -2.0
        + 2.5 * credit_utilization
        - 0.3 * (payment_history_score - 650) / 80
        + 0.5 * (loan_amount_sgd / income_sgd_full - 2.0)
        - 0.1 * employment_years
        + 0.3 * previous_defaults
        + 0.2 * num_late_payments
        - 0.1 * (savings_balance / (income_sgd_full + 1))
        + RNG.normal(0, 0.3, n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (RNG.random(n) < prob_default).astype(int)

    # TEMPORAL LEAKAGE TRAP: future_default_indicator is perfectly correlated
    # with default but named to sound like a legitimate feature
    # Students must catch this in EDA
    future_default_indicator = default.copy()
    # Add tiny noise so it's not literally identical (harder to spot)
    noise_mask = RNG.random(n) < 0.01
    future_default_indicator[noise_mask] = 1 - future_default_indicator[noise_mask]

    # Additional numeric features for richness
    loan_to_value = np.where(
        property_value_raw > 0,
        np.round(loan_amount_sgd / np.clip(property_value_raw, 1, None), 4),
        None,
    )

    coe_vehicle_owner = RNG.choice([0, 1], size=n, p=[0.70, 0.30]).astype(int)
    cpf_monthly_contribution = np.round(
        np.clip(income_sgd_full * 0.2 + RNG.normal(0, 500, n), 0, 3700), 2
    )

    df = pl.DataFrame(
        {
            "customer_id": cust_ids,
            "age": age.tolist(),
            "gender": gender.tolist(),
            "race": race.tolist(),
            "nationality": nationality.tolist(),
            "region": region.tolist(),
            "income_sgd": income_sgd,  # 30% null
            "employment_years": employment_years.tolist(),
            "months_employed": months_employed.tolist(),
            "credit_utilization": credit_utilization.tolist(),
            "avg_balance_utilization": avg_balance_utilization.tolist(),
            "num_credit_lines": num_credit_lines.tolist(),
            "credit_age_years": credit_age_years.tolist(),
            "num_hard_inquiries": num_hard_inquiries.tolist(),
            "payment_history_score": payment_history_score.tolist(),
            "num_late_payments": num_late_payments.tolist(),
            "revolving_balance": revolving_balance.tolist(),
            "installment_balance": installment_balance.tolist(),
            "loan_amount_sgd": loan_amount_sgd.tolist(),
            "loan_purpose": loan_purpose.tolist(),
            "monthly_installment": monthly_installment.tolist(),
            "marital_status": marital_status.tolist(),
            "education": education.tolist(),
            "housing_type": housing_type.tolist(),
            "num_dependents": num_dependents.tolist(),
            "debt_to_income": debt_to_income.tolist(),
            "savings_balance": savings_balance.tolist(),
            "checking_balance": checking_balance.tolist(),
            "previous_defaults": previous_defaults.tolist(),
            "property_value_sgd": property_value_raw.tolist(),
            "loan_to_value": [
                (
                    round(float(x), 4)
                    if x is not None and not np.isnan(float(x if x is not None else 0))
                    else None
                )
                for x in loan_to_value.tolist()
            ],
            "coe_vehicle_owner": coe_vehicle_owner.tolist(),
            "cpf_monthly_contribution": cpf_monthly_contribution.tolist(),
            "application_channel": application_channel.tolist(),
            "future_default_indicator": future_default_indicator.tolist(),  # LEAKAGE TRAP
            "default": default.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 5. ecommerce_customers.parquet  (ascent04) — 50,000 customers
# ---------------------------------------------------------------------------


def make_ecommerce_customers() -> pl.DataFrame:
    """
    E-commerce customer data for clustering + NLP exercises (50,000 rows).
    Mix of numeric RFM features, text fields, and ASEAN regional breakdown.

    Features designed for:
    - RFM clustering (recency, frequency, monetary)
    - NLP sentiment analysis on review_text
    - Churn prediction (days_since_last_order as proxy)
    - ASEAN segmentation
    """
    n = 50_000
    cust_ids = [f"EC-{400000 + i}" for i in range(n)]

    # RFM features with realistic distributions
    total_revenue = np.round(RNG.exponential(scale=350.0, size=n), 2)
    order_count = np.clip(RNG.poisson(8, n), 1, 120).astype(int)
    avg_order_value = np.round(total_revenue / order_count, 2)
    days_since_last_order = np.clip(RNG.integers(1, 730, n), 1, 730).astype(int)

    # Customer tenure (days since first order)
    customer_tenure_days = days_since_last_order + RNG.integers(0, 1000, n)
    customer_tenure_days = np.clip(customer_tenure_days, 1, 2000).astype(int)

    # Lifetime value tier (will be useful for clustering)
    ltv_raw = total_revenue * order_count
    ltv_tier = np.where(
        ltv_raw > np.percentile(ltv_raw, 80),
        "platinum",
        np.where(
            ltv_raw > np.percentile(ltv_raw, 50),
            "gold",
            np.where(ltv_raw > np.percentile(ltv_raw, 20), "silver", "bronze"),
        ),
    )

    # Product categories (text) — comma-separated
    categories_pool = [
        "Electronics",
        "Fashion",
        "Home & Living",
        "Beauty",
        "Sports",
        "Books",
        "Toys",
        "Groceries",
        "Health",
        "Automotive",
        "Travel",
        "Pet Supplies",
        "Office Supplies",
        "Garden",
    ]

    def random_categories():
        k = int(RNG.integers(1, 4))
        chosen = random.sample(categories_pool, k)
        return ", ".join(chosen)

    product_categories = [random_categories() for _ in range(n)]

    # Review text — more varied templates for richer NLP
    pos_templates = [
        "Great product, fast delivery! Will buy again.",
        "Exactly what I needed. Highly recommend.",
        "Good quality for the price. Very happy.",
        "Happy with my purchase. Prompt shipping to Singapore.",
        "Works perfectly. Top notch quality.",
        "Exceeded my expectations. Outstanding service.",
        "Very satisfied. Will definitely order again.",
        "Quality is excellent. Arrived on time.",
        "Super fast delivery. Product matches description.",
        "Amazing value. The packaging was secure.",
        "Love it! Perfect for daily use.",
        "Five stars. Everything was perfect.",
    ]
    neg_templates = [
        "Item arrived damaged. Very disappointed.",
        "Not as described. Poor quality. Returning it.",
        "Delivery took three weeks. Packaging was terrible.",
        "Would not recommend. Complete waste of money.",
        "Product stopped working after one week. Terrible.",
        "Customer service was completely unhelpful.",
        "Size was wrong and return process is a nightmare.",
        "Missing parts. Had to wait weeks for support.",
        "Fake product. Not the brand I ordered.",
        "Arrived late. Item was defective. No refund offered.",
    ]
    neutral_templates = [
        "Decent product. Nothing special but does the job.",
        "Okay for the price. Could be better.",
        "Average quality. Shipping was acceptable.",
        "Shipping was slow but product is fine.",
        "Not bad. Meets basic expectations.",
        "Reasonable quality at this price point.",
    ]

    satisfaction = RNG.integers(1, 6, n)  # 1-5

    review_text = []
    for s in satisfaction:
        if s >= 4:
            review_text.append(random.choice(pos_templates))
        elif s == 3:
            review_text.append(random.choice(neutral_templates))
        else:
            review_text.append(random.choice(neg_templates))

    # ~5% null reviews
    review_text = _nullify_list(review_text, rate=0.05)

    # Region — SG + ASEAN
    regions = RNG.choice(
        ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"],
        size=n,
        p=[0.50, 0.20, 0.12, 0.08, 0.06, 0.04],
    )

    # Device type
    device_type = RNG.choice(
        ["mobile", "desktop", "tablet", "app"],
        size=n,
        p=[0.45, 0.25, 0.10, 0.20],
    )

    # Payment method
    payment_method = RNG.choice(
        [
            "credit_card",
            "debit_card",
            "grab_pay",
            "paypal",
            "bank_transfer",
            "cash_on_delivery",
        ],
        size=n,
        p=[0.30, 0.20, 0.18, 0.12, 0.10, 0.10],
    )

    # Is subscribed to loyalty program
    loyalty_member = RNG.choice([True, False], size=n, p=[0.40, 0.60]).tolist()

    # Number of returns
    num_returns = np.clip(RNG.poisson(0.5, n), 0, 10).astype(int)

    # Churn flag (proxy: no order in last 180 days)
    churned = (days_since_last_order > 180).astype(int)

    df = pl.DataFrame(
        {
            "customer_id": cust_ids,
            "total_revenue": total_revenue.tolist(),
            "order_count": order_count.tolist(),
            "avg_order_value": avg_order_value.tolist(),
            "days_since_last_order": days_since_last_order.tolist(),
            "customer_tenure_days": customer_tenure_days.tolist(),
            "ltv_tier": ltv_tier.tolist(),
            "product_categories": product_categories,
            "review_text": review_text,
            "satisfaction_score": satisfaction.tolist(),
            "region": regions.tolist(),
            "device_type": device_type.tolist(),
            "payment_method": payment_method.tolist(),
            "loyalty_member": loyalty_member,
            "num_returns": num_returns.tolist(),
            "churned": churned.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 6. documents.parquet  (ascent05) — 500 rows
# ---------------------------------------------------------------------------


def make_documents() -> pl.DataFrame:
    """
    Knowledge-base articles for RAG exercise (500 rows).
    Topics cover ML concepts, Singapore domain, ASEAN context, and tech reference.
    Richer content than minimal articles for better embedding/retrieval exercises.
    """
    articles = [
        # --- ML Concepts ---
        (
            "What is supervised learning?",
            "Supervised learning trains a model on labelled data where each example has an input and a known output. The model learns a mapping function. Common algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. Model quality is measured by held-out evaluation on a test set.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is unsupervised learning?",
            "Unsupervised learning discovers patterns in data without labels. Clustering algorithms group similar data points (k-means, DBSCAN, hierarchical). Dimensionality reduction compresses features while preserving structure (PCA, t-SNE, UMAP). Autoencoders learn compact latent representations.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "Explain overfitting and underfitting.",
            "Overfitting occurs when a model memorises training data and fails on new data (high variance). Underfitting occurs when a model is too simple to capture patterns (high bias). Regularisation techniques (L1, L2, dropout) penalise complexity. Cross-validation detects overfitting early.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is cross-validation?",
            "Cross-validation evaluates model performance by splitting data into k folds. The model trains on k-1 folds and validates on the held-out fold. K-fold CV gives a robust performance estimate. Stratified CV preserves class balance. Leave-one-out CV is used for very small datasets.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is gradient descent?",
            "Gradient descent minimises a loss function by iteratively updating parameters in the direction of steepest descent. Learning rate controls step size. Stochastic gradient descent (SGD) updates on one sample. Mini-batch SGD balances stability and efficiency. Adam adapts learning rates per parameter.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is a confusion matrix?",
            "A confusion matrix summarises classification results across all classes. Rows are actual classes, columns are predicted. It reveals true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). Derived metrics: precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2*precision*recall/(precision+recall).",
            "ml_evaluation",
            "textbook",
        ),
        (
            "Precision vs Recall trade-off.",
            "Precision measures how many predicted positives are correct: TP/(TP+FP). Recall measures how many actual positives are found: TP/(TP+FN). The F1 score is their harmonic mean. Adjusting the classification threshold shifts this trade-off. Use precision-recall curves for imbalanced datasets.",
            "ml_evaluation",
            "textbook",
        ),
        (
            "What is ROC-AUC?",
            "ROC-AUC measures a classifier's ability to distinguish classes across all thresholds. The ROC curve plots true positive rate vs false positive rate. AUC=1.0 is perfect. AUC=0.5 is random. It is insensitive to class imbalance. PR-AUC is preferred for highly imbalanced datasets.",
            "ml_evaluation",
            "textbook",
        ),
        (
            "Feature engineering overview.",
            "Feature engineering transforms raw data into informative model inputs. Techniques include normalisation, one-hot encoding, ordinal encoding, polynomial features, interaction terms, log transforms, and domain-specific aggregations. Good features often improve model performance more than algorithm choice.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is one-hot encoding?",
            "One-hot encoding converts categorical variables into binary columns. Each category becomes a column with value 1 if present, 0 otherwise. It avoids imposing ordinal relationships. High-cardinality categories risk the curse of dimensionality. Target encoding and embedding are alternatives for high-cardinality features.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is feature scaling?",
            "Feature scaling standardises the range of input features. Z-score normalisation centres features at 0 with unit variance. Min-max scaling maps features to [0, 1]. Required for distance-based algorithms (k-NN, SVM, k-means) and gradient descent convergence. Tree-based models are scale-invariant.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is PCA?",
            "Principal Component Analysis reduces dimensionality by projecting data onto orthogonal axes of maximum variance. The first principal component explains the most variance. Subsequent components are orthogonal and explain decreasing variance. Used for visualisation, noise reduction, and computational efficiency.",
            "dimensionality_reduction",
            "textbook",
        ),
        (
            "What is t-SNE?",
            "t-SNE (t-distributed Stochastic Neighbour Embedding) is a non-linear dimensionality reduction algorithm optimised for 2D/3D visualisation. It preserves local structure but not global structure. Perplexity controls neighbourhood size. Not suitable for feature engineering due to non-reproducibility and computational cost.",
            "dimensionality_reduction",
            "textbook",
        ),
        (
            "What is UMAP?",
            "UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction algorithm that preserves both local and global structure. Faster than t-SNE. Used for visualisation and as a feature engineering step. Supports supervised and semi-supervised modes.",
            "dimensionality_reduction",
            "textbook",
        ),
        (
            "What is k-means clustering?",
            "K-means partitions data into k clusters by iteratively assigning points to the nearest centroid and updating centroids. Sensitive to initialisation (use k-means++). Assumes spherical, equal-variance clusters. Use the elbow method or silhouette score to choose k. Lloyd's algorithm is the standard implementation.",
            "clustering",
            "textbook",
        ),
        (
            "What is DBSCAN?",
            "DBSCAN clusters points based on density. It finds clusters of arbitrary shape and labels outliers as noise (class -1). Parameters: epsilon (neighbourhood radius) and min_samples (core point threshold). Robust to outliers. Struggles with varying density clusters.",
            "clustering",
            "textbook",
        ),
        (
            "What is hierarchical clustering?",
            "Hierarchical clustering builds a dendrogram by iteratively merging (agglomerative) or splitting (divisive) clusters. Linkage criteria: single, complete, average, Ward. No need to specify k in advance. Dendrogram cutting at different heights gives different cluster numbers.",
            "clustering",
            "textbook",
        ),
        (
            "What are transformers?",
            "Transformers use self-attention to process sequences in parallel. The attention mechanism weighs how much each token attends to every other token. Multi-head attention runs multiple attention functions in parallel. Transformers underpin modern LLMs (GPT, BERT, T5) and vision models (ViT).",
            "deep_learning",
            "textbook",
        ),
        (
            "What is transfer learning?",
            "Transfer learning fine-tunes a pre-trained model on a new task. The pre-trained model has learned general representations from large data. Layers near the input are frozen (feature extraction). Later layers are fine-tuned. Reduces data and compute requirements significantly.",
            "deep_learning",
            "textbook",
        ),
        (
            "What is a convolutional neural network?",
            "CNNs apply learnable filters across input data to detect local patterns. Convolution layers extract spatial features. Pooling layers downsample. Fully connected layers classify. CNNs excel at image tasks. Modern architectures: ResNet, EfficientNet, Vision Transformer (ViT).",
            "deep_learning",
            "textbook",
        ),
        (
            "What is RAG?",
            "Retrieval-Augmented Generation combines a retrieval system with a language model. The retriever finds relevant documents from a knowledge base using vector similarity. The generator conditions its response on retrieved context. Reduces hallucination, keeps knowledge current without retraining.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is prompt engineering?",
            "Prompt engineering crafts inputs to elicit desired behaviour from language models. Techniques: few-shot examples (provide input-output demonstrations), chain-of-thought (ask model to reason step by step), role specification (you are an expert in...), output format constraints (respond in JSON).",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is fine-tuning an LLM?",
            "Fine-tuning adapts a pre-trained LLM on task-specific data. Supervised fine-tuning (SFT) trains on instruction-response pairs. RLHF aligns responses with human preferences. DPO directly optimises preference pairs. LoRA reduces compute by training only low-rank adapter matrices.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is LoRA?",
            "Low-Rank Adaptation (LoRA) fine-tunes LLMs by injecting trainable low-rank matrices into transformer layers. Only the adapter matrices are trained, reducing trainable parameters by 10-100x. Enables fine-tuning large models on consumer GPUs. QLoRA combines LoRA with 4-bit quantisation.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is DPO?",
            "Direct Preference Optimisation reformulates RLHF as supervised learning on (prompt, chosen, rejected) triples. No separate reward model needed. More stable training than PPO. Widely used for instruction-tuning open-source models. SimPO is a simplified variant.",
            "llm_alignment",
            "textbook",
        ),
        (
            "What is RLHF?",
            "Reinforcement Learning from Human Feedback first trains a reward model on human preference comparisons between model outputs. It then fine-tunes the LLM with PPO to maximise reward. A KL penalty prevents the model from deviating too far from the base model. Foundation for ChatGPT-style alignment.",
            "llm_alignment",
            "textbook",
        ),
        (
            "What is vector search?",
            "Vector search retrieves items whose embeddings are most similar to a query embedding. Similarity metrics: cosine similarity, dot product, L2 distance. Approximate nearest neighbour (ANN) algorithms (HNSW, IVF-PQ) enable fast search over billions of vectors. Used in RAG, recommendation systems, and semantic search.",
            "data_engineering",
            "textbook",
        ),
        (
            "What is an embedding?",
            "An embedding is a dense vector representation of data (text, images, products). Embeddings encode semantic similarity — similar items are close in vector space. Text embeddings from models like BGE, E5, or OpenAI's text-embedding models are used for semantic search and RAG retrieval.",
            "deep_learning",
            "textbook",
        ),
        # --- Singapore Context ---
        (
            "What is the HDB in Singapore?",
            "The Housing Development Board (HDB) is Singapore's public housing authority. Over 80% of residents live in HDB flats. New flats are sold under the Build-To-Order (BTO) scheme. Resale flats trade on the open market. Prices are regulated through income ceilings and ethnic integration policies.",
            "singapore_housing",
            "data.gov.sg",
        ),
        (
            "How does the Singapore MRT work?",
            "Singapore's Mass Rapid Transit (MRT) network has six lines: North-South (NSL), East-West (EWL), Circle (CCL), Downtown (DTL), Thomson-East Coast (TEL), and Jurong Region Line (JRL). Over 130 stations cover the island. Fares are distance-based using EZ-Link or contactless payment.",
            "singapore_transport",
            "lta.gov.sg",
        ),
        (
            "What is COE in Singapore?",
            "The Certificate of Entitlement (COE) allows a person to own and use a vehicle in Singapore for 10 years. Prices are determined by open bidding and reflect demand for vehicle ownership. COE prices have exceeded SGD 150,000 for cars. High COE prices are a deliberate policy to limit vehicle population.",
            "singapore_transport",
            "lta.gov.sg",
        ),
        (
            "Singapore CPF overview.",
            "The Central Provident Fund (CPF) is Singapore's mandatory retirement savings scheme. Employees and employers contribute monthly to Ordinary Account (housing), Special Account (retirement), and Medisave (healthcare). CPF Life provides lifelong monthly payouts in retirement.",
            "singapore_finance",
            "cpf.gov.sg",
        ),
        (
            "What is GST in Singapore?",
            "The Goods and Services Tax (GST) is Singapore's value-added tax. It applies to most goods and services. GST rose to 9% in 2024. Businesses with annual turnover above SGD 1 million must register for GST. The GST Voucher scheme offsets the impact on lower-income households.",
            "singapore_finance",
            "iras.gov.sg",
        ),
        (
            "Singapore education system overview.",
            "Singapore's education follows a 6-4-2 structure: 6 years primary, 4 years secondary, 2 years JC or Polytechnic. The PSLE at Primary 6 streams students using Achievement Level grades. Emphasis on STEM and bilingualism. NUS, NTU, SMU, SUTD, SIT, and SUSS are the six autonomous universities.",
            "singapore_education",
            "moe.gov.sg",
        ),
        (
            "Singapore hawker culture.",
            "Hawker centres are open-air cooked food centres operated by NEA. They offer affordable multicultural cuisine: chicken rice, laksa, char kway teow, roti prata, satay. UNESCO added Singapore hawker culture to its Intangible Cultural Heritage list in 2020. Over 100 hawker centres operate across Singapore.",
            "singapore_culture",
            "nea.gov.sg",
        ),
        (
            "Singapore's four official languages.",
            "Singapore has four official languages: English, Mandarin, Malay, and Tamil. English is the language of administration, business, and education. Malay is the national language. Singlish, a creole mixing English, Hokkien, Malay, and Tamil, is widely spoken informally.",
            "singapore_culture",
            "singapore_gov",
        ),
        (
            "What are Singapore's public holidays?",
            "Singapore observes 11 public holidays: New Year's Day, Chinese New Year (2 days), Good Friday, Labour Day, Vesak Day, Hari Raya Puasa, National Day (August 9), Hari Raya Haji, Deepavali, and Christmas Day. When a public holiday falls on Sunday, the following Monday is a holiday.",
            "singapore_culture",
            "mom.gov.sg",
        ),
        (
            "Singapore port and trade.",
            "The Port of Singapore is one of the world's busiest container ports, handling over 37 million TEUs annually. Singapore is a key transhipment hub connecting Asia, Europe, and the Americas. Trade accounts for over 300% of GDP. Major trading partners: China, Malaysia, the US, the EU, and Japan.",
            "singapore_economy",
            "mti.gov.sg",
        ),
        (
            "What is Singpass?",
            "Singpass is Singapore's national digital identity system. It allows residents to access over 2,000 government and private sector digital services. Features include biometric login, digital IC, and Myinfo data sharing. Over 4 million users rely on Singpass daily.",
            "singapore_technology",
            "govtech.gov.sg",
        ),
        (
            "What is the SkillsFuture programme?",
            "SkillsFuture is a national movement to provide Singaporeans with opportunities for lifelong learning. The SkillsFuture Credit (SGD 500 for those aged 25+) subsidises approved courses. Additional top-ups support mid-career workers. Aligned with Singapore's workforce transformation agenda.",
            "singapore_education",
            "skillsfuture.gov.sg",
        ),
        (
            "What is the MAS in Singapore?",
            "The Monetary Authority of Singapore (MAS) is Singapore's central bank and financial regulator. It manages monetary policy through the exchange rate band. MAS regulates banks, insurers, and capital markets. Singapore is a global financial hub with over 200 banks operating locally.",
            "singapore_finance",
            "mas.gov.sg",
        ),
        (
            "What is HDB BTO?",
            "Build-To-Order (BTO) is HDB's main public housing scheme. Applicants select a flat and wait 3-5 years for construction. BTO sales are launched quarterly. Flats are sold at subsidised prices with eligibility criteria (citizenship, income ceiling, household nucleus). First-timer applicants receive priority.",
            "singapore_housing",
            "hdb.gov.sg",
        ),
        (
            "What is the PDPA?",
            "The Personal Data Protection Act (PDPA) governs collection, use, and disclosure of personal data in Singapore. Organisations must obtain consent, limit collection to necessary data, protect data, and allow data subject access and correction. The PDPC enforces the PDPA and can impose fines up to SGD 1 million.",
            "singapore_technology",
            "pdpc.gov.sg",
        ),
        # --- Technical Reference ---
        (
            "Polars vs pandas overview.",
            "Polars is a fast DataFrame library written in Rust. It uses Apache Arrow for columnar memory layout. Lazy evaluation enables query optimisation (predicate pushdown, projection pruning). Polars is 5-10x faster than pandas on many workloads, uses less memory, and supports multi-threading natively.",
            "tools",
            "polars_docs",
        ),
        (
            "What is Apache Parquet?",
            "Parquet is a columnar storage format optimised for analytics. Columns are stored together, enabling efficient compression and predicate pushdown (skip reading irrelevant row groups). Widely used in data engineering. Polars and DuckDB read Parquet natively. Row group sizes balance read efficiency.",
            "tools",
            "apache_docs",
        ),
        (
            "What is a data pipeline?",
            "A data pipeline automates the movement and transformation of data from source to destination. Stages include ingestion, validation, cleaning, feature engineering, and serving. Orchestration tools manage scheduling and dependencies. Modern pipelines are declarative, testable, and versioned.",
            "data_engineering",
            "textbook",
        ),
        (
            "What is data drift?",
            "Data drift is a change in the statistical distribution of input data after model deployment. Feature drift changes input distributions. Concept drift changes the input-output relationship. Detection methods: Population Stability Index (PSI), Kolmogorov-Smirnov test, chi-squared test. PSI > 0.2 typically triggers retraining.",
            "mlops",
            "textbook",
        ),
        (
            "What is MLOps?",
            "MLOps applies DevOps principles to machine learning. Pillars: versioned data, reproducible training, automated testing, CI/CD for models, and continuous monitoring. Key tools: experiment trackers (MLflow), model registries, feature stores, and drift monitors. Reduces time-to-production and improves reliability.",
            "mlops",
            "textbook",
        ),
        (
            "What is a feature store?",
            "A feature store is a centralised repository for ML features. It ensures consistency between training and serving (prevents train-serve skew). Stores feature pipelines, their outputs, and metadata. Enables feature reuse across teams and projects. Common implementations: Feast, Tecton, Hopsworks.",
            "mlops",
            "textbook",
        ),
        (
            "What is model drift monitoring?",
            "Model drift monitoring tracks model performance and input data distributions in production. Metrics: PSI for numeric features, chi-squared for categorical, KS statistic. Output drift monitors prediction distributions. Ground truth feedback enables performance monitoring when labels are available.",
            "mlops",
            "textbook",
        ),
        (
            "What is hyperparameter tuning?",
            "Hyperparameter tuning searches for the configuration that maximises model performance. Grid search exhaustively tests combinations. Random search samples randomly (more efficient). Bayesian optimisation models the performance surface. Optuna and Ray Tune are popular frameworks.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is ensemble learning?",
            "Ensemble learning combines multiple models to improve prediction. Bagging reduces variance by training on bootstrapped subsets (Random Forest). Boosting reduces bias by sequentially correcting errors (XGBoost, LightGBM, CatBoost). Stacking uses a meta-learner to blend diverse model predictions.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is SHAP?",
            "SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a specific prediction using Shapley values from cooperative game theory. It is model-agnostic and satisfies axioms of efficiency, symmetry, and dummy. Used for model interpretability and bias auditing in production.",
            "ml_interpretability",
            "textbook",
        ),
        (
            "What is LIME?",
            "LIME (Local Interpretable Model-agnostic Explanations) explains individual model predictions by fitting a simple linear model locally around the prediction. Perturbs inputs and measures impact on predictions. Faster than SHAP for some models but less theoretically grounded.",
            "ml_interpretability",
            "textbook",
        ),
        (
            "What is AutoML?",
            "Automated Machine Learning (AutoML) searches for the best model and hyperparameters automatically. Pipelines include data preprocessing, feature selection, algorithm selection, and tuning. Tools: Auto-sklearn, TPOT, H2O AutoML. Useful for rapid prototyping and non-expert users.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is XGBoost?",
            "XGBoost is a gradient boosting framework optimised for speed and performance. It uses second-order gradients and regularisation (L1, L2) to prevent overfitting. Supports sparse data and missing values natively. The dominant algorithm in tabular ML competitions.",
            "ml_algorithms",
            "textbook",
        ),
        (
            "What is LightGBM?",
            "LightGBM is a gradient boosting framework that uses leaf-wise tree growth instead of level-wise. Faster than XGBoost on large datasets. Uses histogram-based splits for efficiency. Supports categorical features natively. GOSS and EFB optimisations further reduce training time.",
            "ml_algorithms",
            "textbook",
        ),
        # --- ASEAN Economic Context ---
        (
            "What is ASEAN?",
            "ASEAN (Association of Southeast Asian Nations) is a regional bloc of 10 countries: Brunei, Cambodia, Indonesia, Laos, Malaysia, Myanmar, Philippines, Singapore, Thailand, and Vietnam. Combined GDP exceeds USD 3 trillion. Free trade agreements promote regional economic integration.",
            "asean_economy",
            "asean.org",
        ),
        (
            "Singapore's role in ASEAN.",
            "Singapore serves as ASEAN's financial and logistics hub. It hosts regional headquarters of many multinationals. Strong rule of law, low corruption, and pro-business policies attract investment. Singapore's GDP per capita exceeds USD 80,000, highest in ASEAN.",
            "asean_economy",
            "mti.gov.sg",
        ),
        (
            "Vietnam's tech ecosystem.",
            "Vietnam has emerged as a major tech manufacturing hub. Companies like Samsung, Intel, and LG have large factories there. A young, tech-savvy population drives e-commerce growth (Shopee, Lazada, Tiki). Hanoi and Ho Chi Minh City are regional startup centres.",
            "asean_economy",
            "worldbank.org",
        ),
        (
            "Indonesia's digital economy.",
            "Indonesia has Southeast Asia's largest digital economy. Gojek and Tokopedia merged into GoTo Group. Over 200 million internet users. Financial inclusion through digital banking and mobile payments is a key growth driver. Jakarta is the main tech hub.",
            "asean_economy",
            "worldbank.org",
        ),
        (
            "What is the RCEP trade agreement?",
            "The Regional Comprehensive Economic Partnership (RCEP) is the world's largest free trade agreement by GDP. It covers ASEAN plus Australia, China, Japan, New Zealand, and South Korea. Eliminates tariffs, streamlines customs, and sets rules of origin. Signed in 2020, in force since 2022.",
            "asean_economy",
            "asean.org",
        ),
    ]

    # Pad to 500 rows by repeating with slight variation
    base = articles.copy()
    extra_prefixes = [
        "(Reference) ",
        "(Overview) ",
        "(Summary) ",
        "(Guide) ",
        "(Explainer) ",
        "(Tutorial) ",
        "(Deep dive) ",
    ]
    while len(base) < 500:
        t, c, cat, src = random.choice(articles)
        prefix = random.choice(extra_prefixes)
        base.append((prefix + t, c, cat, src))

    base = base[:500]

    doc_ids = [f"DOC-{500 + i:04d}" for i in range(len(base))]

    df = pl.DataFrame(
        {
            "doc_id": doc_ids,
            "title": [a[0] for a in base],
            "content": [a[1] for a in base],
            "category": [a[2] for a in base],
            "source": [a[3] for a in base],
        }
    )

    return df


# ---------------------------------------------------------------------------
# 7. sg_domain_qa.parquet  (ascent06) — 1,000 rows
# ---------------------------------------------------------------------------


def make_sg_domain_qa() -> pl.DataFrame:
    """
    Singapore domain Q&A for SFT (M6.1) — 1,000 rows.
    Instruction-following format. Covers geography, food, transport,
    housing, finance, technology, and ML concepts with Singapore context.
    """
    pairs = [
        # --- Geography ---
        (
            "What are the five main regions of Singapore?",
            "Singapore is divided into five main planning regions: the Central Region, North Region, North-East Region, East Region, and West Region. Each region contains several planning areas managed by the Urban Redevelopment Authority.",
            "",
            "geography",
        ),
        (
            "How large is Singapore by land area?",
            "Singapore has a total land area of approximately 735 square kilometres. Land reclamation has expanded its area significantly over the decades, adding around 25% of the current land area since independence.",
            "",
            "geography",
        ),
        (
            "What is Sentosa Island known for?",
            "Sentosa is a resort island off Singapore's southern coast. It hosts Universal Studios Singapore, Resorts World Sentosa, beaches, the S.E.A. Aquarium, and the cable car connecting it to VivoCity mall.",
            "",
            "geography",
        ),
        (
            "Name the main river in Singapore.",
            "The Singapore River runs through the heart of the city. It was the historic centre of colonial trade, lined with warehouses (godowns) at Clarke Quay and Boat Quay. Today its banks are a major dining and entertainment destination.",
            "",
            "geography",
        ),
        (
            "What is the highest point in Singapore?",
            "Bukit Timah Hill, at 163.63 metres, is Singapore's highest natural point. The Bukit Timah Nature Reserve surrounding it contains one of the world's few patches of primary rainforest within a major city.",
            "",
            "geography",
        ),
        (
            "What is Pulau Ubin?",
            "Pulau Ubin is a small island north of Singapore. It offers a glimpse of traditional kampung (village) life. It is popular for cycling, hiking, and the Chek Jawa wetlands. The island has no permanent utilities and is accessible by bumboat from Changi Point Ferry Terminal.",
            "",
            "geography",
        ),
        (
            "What are the major reservoirs in Singapore?",
            "Singapore's main reservoirs include Marina Reservoir (the largest in the city centre), Upper Peirce, Lower Peirce, MacRitchie, Kranji, Bedok, and Tengeh. Marina Barrage created a freshwater reservoir in the Marina Bay area.",
            "",
            "geography",
        ),
        # --- Food ---
        (
            "What is chicken rice?",
            "Hainanese chicken rice is widely regarded as Singapore's national dish. It consists of poached or roasted chicken served over fragrant rice cooked in chicken broth, accompanied by chilli sauce, ginger paste, and dark soy sauce. Famous hawker stalls include Tian Tian at Maxwell Food Centre.",
            "",
            "food",
        ),
        (
            "What is laksa?",
            "Laksa is a spicy noodle soup popular in Singapore. Singapore laksa uses thick rice vermicelli in a rich coconut curry broth, topped with prawns, fish cake, half a hard-boiled egg, and cockles. Katong laksa from the East Coast is a distinctive local variant with pre-cut noodles.",
            "",
            "food",
        ),
        (
            "What is a hawker centre?",
            "A hawker centre is a large open-air complex housing multiple food stalls offering cooked food and drinks. They provide affordable meals (typically SGD 3-6) representing Singapore's diverse cuisines: Chinese, Malay, and Indian. UNESCO recognised Singapore hawker culture in 2020.",
            "",
            "food",
        ),
        (
            "What is char kway teow?",
            "Char kway teow is a stir-fried flat noodle dish cooked over very high heat in a wok. It contains flat rice noodles, Chinese sausage (lup cheong), cockles, eggs, bean sprouts, and dark soy sauce. The wok hei (breath of the wok) from high-heat cooking gives it a distinctive smoky flavour.",
            "",
            "food",
        ),
        (
            "What is bak kut teh?",
            "Bak kut teh (meat bone tea) is a pork rib soup simmered with herbs and spices. The Singaporean version is peppery and lighter compared to the darker, herbal Malaysian version. It is traditionally eaten for breakfast with youtiao (fried dough) and rice.",
            "",
            "food",
        ),
        (
            "What is roti prata?",
            "Roti prata is a flaky flatbread of South Indian origin, cooked on a flat griddle with ghee. It is served with dhal, fish, or chicken curry and comes in plain or stuffed variants (egg, cheese, banana, mushroom). A popular breakfast and supper food across Singapore.",
            "",
            "food",
        ),
        (
            "What is chilli crab?",
            "Chilli crab is one of Singapore's signature seafood dishes. Mud crabs are stir-fried in a semi-thick, tangy-sweet-savoury gravy made from tomato sauce, chilli, and egg. Best enjoyed with deep-fried or steamed mantou (buns) to mop up the sauce. Famous at Long Beach and No Signboard Seafood.",
            "",
            "food",
        ),
        (
            "What is kaya toast?",
            "Kaya toast is a traditional Singapore breakfast. Crispy toast is spread with kaya (a jam made from coconut milk, eggs, and pandan leaves) and butter. Served with soft-boiled eggs seasoned with soy sauce and white pepper, and accompanied by kopi (local coffee) or teh (tea).",
            "",
            "food",
        ),
        # --- Transport ---
        (
            "How does the Singapore MRT fare system work?",
            "MRT fares are calculated by the distance travelled, using a tiered distance-based system. Payment is by EZ-Link card, Singapore Tourist Pass, or contactless bank card. Concession cards offer reduced fares for students, seniors, and persons with disabilities. Fares typically range from SGD 0.77 to SGD 2.60.",
            "",
            "transport",
        ),
        (
            "What is the LRT system in Singapore?",
            "Singapore has three LRT networks: Bukit Panjang LRT (BPLRT) operated by SMRT, and Sengkang and Punggol LRT operated by SBS Transit. All are automated and driverless, providing feeder services from HDB estates to nearby MRT stations.",
            "",
            "transport",
        ),
        (
            "What is the EZ-Link card?",
            "The EZ-Link card is Singapore's contactless smart card for public transport. It works on MRT, LRT, and all public buses. Cards can also be used at selected retail outlets. Top-up is available at TransitLink Ticket Offices, ATMs, and convenience stores.",
            "",
            "transport",
        ),
        (
            "How do you book a taxi in Singapore?",
            "Taxis can be hailed at taxi stands, from the roadside outside CBD and peak hours, or booked via apps: Grab, ComfortDelGro Taxi, TADA, and Gojek operate in Singapore. Booking surcharges and peak-hour charges apply. Booking via app is the most common method.",
            "",
            "transport",
        ),
        (
            "What is the TEL MRT line?",
            "The Thomson-East Coast Line (TEL) is Singapore's sixth MRT line, operated by SMRT. It runs from Woodlands North in the north to Sungei Bedok in the east, passing through Orchard, Marina Bay, and Gardens by the Bay. It was completed in phases between 2020 and 2024.",
            "",
            "transport",
        ),
        (
            "What is the Jurong Region Line?",
            "The Jurong Region Line (JRL) is Singapore's seventh MRT line, under construction. It will serve the western region including Jurong, Choa Chu Kang, Boon Lay, and Tengah. When completed, it will have 24 stations. Construction began in 2018 with phased opening from 2027.",
            "",
            "transport",
        ),
        (
            "What is the ART in Singapore?",
            "The Autonomous Rail Transit (ART) is being piloted in the Jurong Lake District. It runs on rubber tyres guided by painted lines, with no physical rails. Powered by electricity, it can carry up to 500 passengers per train. ART offers flexible route deployment compared to fixed-track MRT.",
            "",
            "transport",
        ),
        # --- Housing ---
        (
            "What are HDB BTO flats?",
            "BTO (Build-To-Order) is HDB's primary public housing scheme. Applicants select a flat unit and wait 3-5 years for construction. BTO exercises are launched quarterly with new projects in various towns. First-timer married couples and families receive priority balloting. Prices are subsidised.",
            "",
            "housing",
        ),
        (
            "What is the HDB ethnic integration policy?",
            "The Ethnic Integration Policy (EIP) sets block and neighbourhood limits for the proportion of Chinese, Malay, Indian, and Other residents in each HDB development. It prevents ethnic enclaves and promotes racial integration. Sellers must check EIP eligibility before listing their flat for sale.",
            "",
            "housing",
        ),
        (
            "What is the minimum occupancy period for HDB flats?",
            "HDB owners must complete a Minimum Occupation Period (MOP) of five years before selling the flat on the resale market, renting out the entire flat, or purchasing private property. The MOP begins from the date of key collection, not the date of application.",
            "",
            "housing",
        ),
        (
            "What is the CPF Housing Grant?",
            "The Enhanced CPF Housing Grant (EHG) provides up to SGD 80,000 for eligible first-time buyers of HDB flats. The amount depends on household income. Grants are credited to the CPF Ordinary Account and used for downpayment or monthly instalments. Additional grants are available for proximity to parents.",
            "",
            "housing",
        ),
        (
            "What is an HDB resale flat?",
            "An HDB resale flat is a public housing unit sold by its owner on the open market after the MOP. Prices are negotiated between buyer and seller. The resale market provides immediate access to housing. Buyers may use CPF savings and HDB housing loans. A Cash-Over-Valuation (COV) may be payable.",
            "",
            "housing",
        ),
        (
            "What is a private condominium in Singapore?",
            "Private condominiums are non-HDB residential developments with shared facilities (pool, gym, function rooms). They are sold by developers or on the resale market. Foreigners can purchase condos (subject to Additional Buyer's Stamp Duty). Prices range from SGD 800 psf to over SGD 4,000 psf in prime areas.",
            "",
            "housing",
        ),
        # --- Economy & Finance ---
        (
            "What is the Singapore Exchange (SGX)?",
            "The Singapore Exchange (SGX) is Singapore's stock exchange, listing equities, derivatives, bonds, and REITs. The Straits Times Index (STI) tracks the top 30 companies. SGX is a key financial infrastructure for Southeast Asia, with dual-currency and multi-currency trading capabilities.",
            "",
            "finance",
        ),
        (
            "What is a REIT in Singapore?",
            "A Real Estate Investment Trust (REIT) pools investor capital to invest in income-generating real estate. Singapore has one of Asia's largest REIT markets with over 40 listed REITs and property trusts. REITs must distribute at least 90% of taxable income as dividends. Popular REITs: CapitaLand Integrated Commercial Trust, Ascendas REIT.",
            "",
            "finance",
        ),
        (
            "What is the SGD exchange rate policy?",
            "The Monetary Authority of Singapore (MAS) manages monetary policy through the Singapore Dollar (SGD) nominal effective exchange rate (NEER) band, not interest rates. MAS adjusts the band's centre, slope, and width. This approach suits Singapore's small, open, trade-dependent economy.",
            "",
            "finance",
        ),
        (
            "What is the SkillsFuture credit?",
            "SkillsFuture Credit provides SGD 500 to Singaporeans aged 25 and above for approved training courses. Additional top-ups are provided for mid-career workers aged 40 and above. Credits are accessed via the MySkillsFuture portal and expire if unused. Over 600,000 Singaporeans have used SkillsFuture credits.",
            "",
            "education",
        ),
        (
            "What is the PSLE in Singapore?",
            "The Primary School Leaving Examination (PSLE) is taken by Primary 6 students (approximately age 12). It is used for secondary school placement. Students are scored using Achievement Level (AL) grades from AL1 (best) to AL8 for each subject. PSLE AL scores replaced T-scores in 2021.",
            "",
            "education",
        ),
        (
            "What is the Singapore Budget?",
            "Singapore's annual budget is presented by the Finance Minister in February. It allocates government spending across healthcare, education, defence, social support, and economic development. Singapore aims for a broadly balanced budget over each term of government. Surpluses can be invested by GIC and Temasek.",
            "",
            "finance",
        ),
        (
            "What is Temasek Holdings?",
            "Temasek Holdings is Singapore's state investment company, owned by the Singapore government. It manages a portfolio of over SGD 400 billion, invested in Singaporean companies (DBS, SingTel, Singapore Airlines) and global companies. Temasek returns are used to support the Singapore government budget.",
            "",
            "finance",
        ),
        # --- ML / Tech ---
        (
            "What is Singapore's Smart Nation initiative?",
            "Smart Nation is Singapore's government-led initiative to harness digital technology to improve citizens' lives and business efficiency. Key projects: Singpass (digital identity), LifeSG (government services), the National Digital Identity (NDI) framework, and smart urban mobility systems.",
            "",
            "technology",
        ),
        (
            "What is PDPA in Singapore?",
            "The Personal Data Protection Act (PDPA) governs the collection, use, disclosure, and care of personal data in Singapore. Key obligations: purpose limitation, consent, notification, access/correction rights, data protection, and retention limits. Mandatory data breach notification applies to significant breaches.",
            "",
            "technology",
        ),
        (
            "What is GovTech Singapore?",
            "Government Technology Agency (GovTech) is the lead agency driving Singapore's digital government transformation. It develops and operates government digital infrastructure: Singpass, CorpPass, National Digital Identity, Whole of Government Application Analytics, and the Singapore Government Tech Stack.",
            "",
            "technology",
        ),
        (
            "What is Singapore's National AI Strategy?",
            "Singapore's National AI Strategy (NAIS 2.0, 2023) aims to develop AI capabilities across key sectors: health, education, public service, and the economy. It focuses on AI talent development, international governance, trusted AI deployment, and Singapore as an AI centre for the region.",
            "",
            "technology",
        ),
        (
            "What is data.gov.sg?",
            "data.gov.sg is Singapore's open data portal maintained by GovTech. It provides free access to government datasets covering housing, transport, environment, economy, demographics, and geospatial data. Datasets are available as CSV, GeoJSON, and API. Over 2,000 datasets are published.",
            "",
            "technology",
        ),
        (
            "What is OneMap in Singapore?",
            "OneMap is Singapore's authoritative national map platform maintained by the Singapore Land Authority (SLA). It provides geospatial data and APIs for planning areas, addresses, transportation, and land parcels. It is the base map for government and commercial applications in Singapore.",
            "",
            "technology",
        ),
        # --- ML with Singapore context ---
        (
            "How is ML used in Singapore's public housing?",
            "HDB uses ML for resale price prediction, maintenance scheduling (predicting lift failures), and estate planning. Data from data.gov.sg enables public research on housing trends. Students can download HDB resale transaction data to build regression models predicting flat prices.",
            "",
            "ml_singapore",
        ),
        (
            "How is ML used in Singapore's transport system?",
            "LTA uses ML for bus arrival predictions, MRT fault detection, traffic signal optimisation, and COE price forecasting. The GTFS (General Transit Feed Specification) data is available publicly. Taxi demand prediction is a classic time-series forecasting problem with Singapore taxi data.",
            "",
            "ml_singapore",
        ),
        (
            "What is the Singapore COVID-19 data?",
            "Singapore published detailed COVID-19 case, vaccination, and testing data on data.gov.sg throughout the pandemic. This data is used for epidemiological modelling exercises including SIR/SEIR models, vaccination impact analysis, and time-series forecasting of case counts.",
            "",
            "ml_singapore",
        ),
    ]

    # Expand to 1,000 rows by paraphrasing
    extended = pairs.copy()
    paraphrase_prefix = [
        "Can you explain: ",
        "Please describe: ",
        "Give me information about: ",
        "Tell me about: ",
        "I want to learn about: ",
        "What should I know about: ",
        "Could you clarify: ",
        "Please elaborate on: ",
        "Describe for me: ",
        "Help me understand: ",
    ]
    while len(extended) < 1000:
        inst, resp, ctx, cat = random.choice(pairs)
        prefix = random.choice(paraphrase_prefix)
        extended.append((prefix + inst.lower(), resp, ctx, cat))

    extended = extended[:1000]

    df = pl.DataFrame(
        {
            "instruction": [e[0] for e in extended],
            "response": [e[1] for e in extended],
            "context": [e[2] for e in extended],
            "category": [e[3] for e in extended],
        }
    )

    return df


# ---------------------------------------------------------------------------
# 8. preference_pairs.parquet  (ascent06) — 500 rows
# ---------------------------------------------------------------------------


def make_preference_pairs() -> pl.DataFrame:
    """
    Preference pairs for DPO (M6.2) — 500 rows.
    Each row has a prompt, a chosen (substantive) response, and a rejected
    (plausible-but-shallow) response.
    """
    pairs = [
        (
            "Explain overfitting in machine learning.",
            "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalise to new examples. The model has high variance. Regularisation techniques (L1/L2 penalties, dropout), early stopping, cross-validation, and more training data all help mitigate overfitting.",
            "Overfitting is when the model is overfitted. It is bad and you should avoid it by using regularisation.",
            "ml_concepts",
        ),
        (
            "What is the difference between classification and regression?",
            "Classification predicts discrete class labels (e.g., spam vs not spam, credit default vs no default). Regression predicts continuous numeric values (e.g., HDB resale price, income). Both are supervised learning tasks but differ in output type, loss functions (cross-entropy vs MSE), and evaluation metrics.",
            "Classification is for categories and regression is for numbers. They are both supervised.",
            "ml_concepts",
        ),
        (
            "How does a random forest work?",
            "A random forest trains multiple decision trees on bootstrapped subsets of the data (bagging), with each tree seeing only a random subset of features at each split (feature randomisation). Predictions are aggregated by majority vote (classification) or averaging (regression). This reduces variance compared to a single tree.",
            "Random forest has many trees and they vote. It is better than one tree because more trees are better.",
            "ml_algorithms",
        ),
        (
            "What is gradient boosting?",
            "Gradient boosting builds an ensemble sequentially. Each new tree corrects the residual errors of the current ensemble by fitting the negative gradient of the loss function. Shrinkage (learning rate) controls each tree's contribution. XGBoost adds L1/L2 regularisation and second-order gradient information for efficiency.",
            "Gradient boosting uses gradients and boosts things. It is used in competitions and is very accurate.",
            "ml_algorithms",
        ),
        (
            "Explain the bias-variance trade-off.",
            "Bias measures error from overly simple model assumptions (underfitting — the model cannot capture the true signal). Variance measures sensitivity to training data fluctuations (overfitting — the model fits noise). As model complexity increases, bias falls and variance rises. The optimal complexity minimises total error (bias² + variance + irreducible noise).",
            "Bias and variance are both bad. You want low bias and low variance. Try different models.",
            "ml_theory",
        ),
        (
            "What is precision and when should you prioritise it?",
            "Precision is TP / (TP + FP) — of all predicted positives, what fraction is correct. Prioritise precision when false positives are costly: spam filtering (blocking legitimate emails is harmful), legal document classification, or credit approval (false approval wastes capital). The precision-recall trade-off is controlled by the classification threshold.",
            "Precision is TP divided by TP plus FP. It is good when you don't want false positives.",
            "ml_evaluation",
        ),
        (
            "How do you handle class imbalance?",
            "Strategies include: oversampling the minority class (SMOTE, ADASYN), undersampling the majority class, class-weight adjustment in the loss function, and using imbalance-robust evaluation metrics (PR-AUC, balanced accuracy, Matthews Correlation Coefficient). Threshold tuning after calibration often achieves the best practical result.",
            "Use SMOTE or oversample. You can also try different models. Sometimes class imbalance is not a big problem.",
            "ml_practical",
        ),
        (
            "What is SHAP and how is it used?",
            "SHAP assigns each feature a contribution value for a specific prediction using Shapley values from cooperative game theory. It satisfies three fairness axioms (efficiency, symmetry, dummy). TreeSHAP computes exact values for tree models in polynomial time. Used for local explanations (per prediction) and global feature importance (mean absolute SHAP).",
            "SHAP explains the model. It shows which features are important. It uses Shapley values from game theory.",
            "ml_interpretability",
        ),
        (
            "What is the purpose of a validation set?",
            "A validation set is a held-out data split used during training to tune hyperparameters and monitor for overfitting, without contaminating the test set. It enables early stopping and model selection. The test set must be kept completely unseen until final evaluation to provide an unbiased generalisation estimate.",
            "Validation set is used to validate the model. It is different from the test set. You use it during training.",
            "ml_practical",
        ),
        (
            "How does k-means clustering decide the number of clusters?",
            "K-means requires k to be specified in advance. Methods to choose k: (1) elbow method — plot inertia (within-cluster sum of squares) vs k and look for an elbow; (2) silhouette score — measures cohesion vs separation, higher is better; (3) gap statistic — compares inertia to random data; (4) domain knowledge about expected groupings.",
            "You try different values of k and pick the best one. The elbow method shows where adding clusters stops helping.",
            "ml_algorithms",
        ),
        (
            "What is transfer learning and when should you use it?",
            "Transfer learning reuses a model pre-trained on a large dataset as a starting point. Feature extraction freezes the pre-trained layers and trains only the head. Full fine-tuning updates all layers on the target data. Use when: labelled data is scarce, compute is limited, or the source and target domains share low-level features.",
            "Transfer learning transfers knowledge from one model to another. Use it when you don't have enough data.",
            "deep_learning",
        ),
        (
            "What is the attention mechanism in transformers?",
            "Scaled dot-product attention computes a weighted sum of value vectors. Weights are determined by query-key dot products, scaled by 1/√d_k and normalised with softmax. Multi-head attention runs h parallel attention functions with projected Q, K, V matrices, then concatenates. Self-attention allows each position to attend to all others.",
            "Attention looks at other tokens in the sequence. It uses queries, keys, and values. Multi-head attention does this multiple times.",
            "deep_learning",
        ),
        (
            "How do you evaluate a language model?",
            "Perplexity measures prediction confidence on held-out text (lower is better). Task-specific benchmarks (MMLU for knowledge, HellaSwag for commonsense, HumanEval for coding) evaluate capabilities. For instruction-tuned models, MT-Bench and Chatbot Arena assess conversational quality via human preferences.",
            "You evaluate LLMs with perplexity and benchmarks like MMLU. Human evaluation is also used.",
            "llm_techniques",
        ),
        (
            "What is RAG and when should you use it?",
            "Retrieval-Augmented Generation retrieves relevant documents and conditions the LLM response on them. Use RAG when: (1) knowledge must be up-to-date beyond training cutoff, (2) verifiable sources are required, (3) domain-specific facts are needed, (4) full fine-tuning is too expensive or the knowledge changes frequently.",
            "RAG retrieves documents and passes them to the LLM. Use it when the LLM doesn't know the answer.",
            "llm_techniques",
        ),
        (
            "What is LoRA and why is it useful?",
            "LoRA injects trainable low-rank matrices (AB where A∈ℝ^{d×r}, B∈ℝ^{r×k}, r≪d) into transformer attention layers. Only A and B are trained, reducing trainable parameters by 10-100x. This allows fine-tuning 7B+ parameter models on a single GPU. QLoRA additionally quantises the base model to 4-bit.",
            "LoRA is a way to fine-tune LLMs efficiently. It uses low-rank matrices so you train fewer parameters.",
            "llm_techniques",
        ),
        (
            "How does RLHF work?",
            "RLHF has three stages: (1) supervised fine-tuning (SFT) on instruction-response pairs, (2) reward model training on human preference comparisons between model outputs, (3) PPO fine-tuning to maximise reward while a KL divergence penalty prevents excessive drift from the SFT policy. DPO simplifies stage 3 by removing the need for online RL.",
            "RLHF uses human feedback to train the model. A reward model is trained first, then the LLM is fine-tuned.",
            "llm_alignment",
        ),
        (
            "What is DPO compared to PPO for alignment?",
            "DPO reformulates preference learning as a supervised classification on (prompt, chosen, rejected) triples. The implicit reward is derived from the log ratio of policy and reference model probabilities. No reward model or online rollouts are needed. DPO is simpler, more stable, and typically matches PPO quality on instruction-following tasks.",
            "DPO is simpler than PPO. It doesn't need a reward model. DPO directly optimises on preference data.",
            "llm_alignment",
        ),
        (
            "What is data drift and how do you detect it?",
            "Data drift is a change in the statistical distribution of input features after model deployment. Detection methods: Population Stability Index (PSI > 0.2 = significant drift), Kolmogorov-Smirnov (KS) test for numeric features, chi-squared test for categorical features, and monitoring feature summary statistics (mean, std, quantiles). Alerts trigger retraining pipelines.",
            "Data drift is when the data changes over time. You can detect it with PSI or statistical tests.",
            "mlops",
        ),
        (
            "What is a feature store and why is it important?",
            "A feature store is a centralised repository that stores feature transformation logic and materialised feature values. It ensures training-serving consistency (prevents train-serve skew). Features are versioned, discoverable, and reusable across teams. Point-in-time correct joins prevent future leakage in training data.",
            "A feature store stores features. It is useful for sharing features across teams.",
            "mlops",
        ),
        (
            "How do you monitor a model in production?",
            "Monitor four dimensions: (1) input data distributions (PSI, KS test for drift), (2) prediction distributions (output drift), (3) business metrics (revenue, conversions, click-through rate), (4) ground truth performance when labels become available (AUC, precision, recall). Set alerting thresholds and schedule periodic retraining when metrics degrade beyond acceptable bounds.",
            "Monitor the model predictions and check if they are still accurate. Use dashboards and alerts. Retrain when needed.",
            "mlops",
        ),
    ]

    # Expand to 500 rows
    extended = pairs.copy()
    while len(extended) < 500:
        p, c, r, cat = random.choice(pairs)
        extended.append((p, c, r, cat))

    extended = extended[:500]

    df = pl.DataFrame(
        {
            "prompt": [e[0] for e in extended],
            "chosen": [e[1] for e in extended],
            "rejected": [e[2] for e in extended],
            "category": [e[3] for e in extended],
        }
    )

    return df


# ---------------------------------------------------------------------------
# 9. mrt_stations.parquet  (ascent_assessment) — keep as-is (~150 stations)
# ---------------------------------------------------------------------------


def make_mrt_stations() -> pl.DataFrame:
    """
    Singapore MRT stations with town mapping and coordinates (~150 rows).
    Self-referential: nearest_mrt column references station_name.
    """
    stations = [
        # (station_name, town, line, lat, lon)
        # --- North-South Line (NSL) ---
        ("Jurong East", "Jurong East", "NSL", 1.3331, 103.7422),
        ("Bukit Batok", "Bukit Batok", "NSL", 1.3485, 103.7496),
        ("Bukit Gombak", "Bukit Batok", "NSL", 1.3586, 103.7516),
        ("Choa Chu Kang", "Choa Chu Kang", "NSL", 1.3853, 103.7446),
        ("Yew Tee", "Choa Chu Kang", "NSL", 1.3969, 103.7474),
        ("Kranji", "Woodlands", "NSL", 1.4252, 103.7619),
        ("Marsiling", "Woodlands", "NSL", 1.4328, 103.7744),
        ("Woodlands", "Woodlands", "NSL", 1.4369, 103.7863),
        ("Admiralty", "Sembawang", "NSL", 1.4408, 103.8009),
        ("Sembawang", "Sembawang", "NSL", 1.4491, 103.8199),
        ("Canberra", "Sembawang", "NSL", 1.4432, 103.8296),
        ("Yishun", "Yishun", "NSL", 1.4294, 103.8354),
        ("Khatib", "Yishun", "NSL", 1.4175, 103.8330),
        ("Yio Chu Kang", "Ang Mo Kio", "NSL", 1.3817, 103.8449),
        ("Ang Mo Kio", "Ang Mo Kio", "NSL", 1.3699, 103.8496),
        ("Bishan", "Bishan", "NSL", 1.3510, 103.8485),
        ("Braddell", "Toa Payoh", "NSL", 1.3402, 103.8468),
        ("Toa Payoh", "Toa Payoh", "NSL", 1.3322, 103.8469),
        ("Novena", "Novena", "NSL", 1.3204, 103.8436),
        ("Newton", "Novena", "NSL", 1.3132, 103.8388),
        ("Orchard", "Orchard", "NSL", 1.3048, 103.8318),
        ("Somerset", "Orchard", "NSL", 1.3006, 103.8388),
        ("Dhoby Ghaut", "Museum", "NSL", 1.2988, 103.8456),
        ("City Hall", "Downtown", "NSL", 1.2931, 103.8520),
        ("Raffles Place", "Downtown", "NSL", 1.2831, 103.8513),
        ("Marina Bay", "Downtown", "NSL", 1.2762, 103.8554),
        ("Marina South Pier", "Downtown", "NSL", 1.2711, 103.8634),
        # --- East-West Line (EWL) ---
        ("Pasir Ris", "Pasir Ris", "EWL", 1.3731, 103.9494),
        ("Tampines", "Tampines", "EWL", 1.3526, 103.9453),
        ("Simei", "Tampines", "EWL", 1.3431, 103.9531),
        ("Tanah Merah", "Bedok", "EWL", 1.3273, 103.9461),
        ("Bedok", "Bedok", "EWL", 1.3240, 103.9299),
        ("Kembangan", "Bedok", "EWL", 1.3209, 103.9130),
        ("Eunos", "Geylang", "EWL", 1.3196, 103.9031),
        ("Paya Lebar", "Geylang", "EWL", 1.3178, 103.8926),
        ("Aljunied", "Geylang", "EWL", 1.3163, 103.8830),
        ("Kallang", "Kallang", "EWL", 1.3118, 103.8716),
        ("Lavender", "Kallang", "EWL", 1.3072, 103.8638),
        ("Bugis", "Rochor", "EWL", 1.3008, 103.8559),
        ("City Hall", "Downtown", "EWL", 1.2931, 103.8520),
        ("Tanjong Pagar", "Downtown", "EWL", 1.2762, 103.8454),
        ("Outram Park", "Outram", "EWL", 1.2801, 103.8396),
        ("Tiong Bahru", "Bukit Merah", "EWL", 1.2863, 103.8272),
        ("Redhill", "Bukit Merah", "EWL", 1.2895, 103.8164),
        ("Queenstown", "Queenstown", "EWL", 1.2943, 103.8059),
        ("Commonwealth", "Queenstown", "EWL", 1.3022, 103.7982),
        ("Buona Vista", "Queenstown", "EWL", 1.3067, 103.7901),
        ("Dover", "Clementi", "EWL", 1.3115, 103.7787),
        ("Clementi", "Clementi", "EWL", 1.3151, 103.7654),
        ("Jurong East", "Jurong East", "EWL", 1.3331, 103.7422),
        ("Chinese Garden", "Jurong East", "EWL", 1.3421, 103.7334),
        ("Lakeside", "Jurong West", "EWL", 1.3443, 103.7208),
        ("Boon Lay", "Jurong West", "EWL", 1.3388, 103.7062),
        ("Pioneer", "Jurong West", "EWL", 1.3374, 103.6973),
        ("Joo Koon", "Jurong West", "EWL", 1.3277, 103.6782),
        ("Gul Circle", "Jurong West", "EWL", 1.3197, 103.6609),
        ("Tuas Crescent", "Tuas", "EWL", 1.3218, 103.6491),
        ("Tuas West Road", "Tuas", "EWL", 1.3301, 103.6393),
        ("Tuas Link", "Tuas", "EWL", 1.3403, 103.6367),
        # --- Circle Line (CCL) ---
        ("Dhoby Ghaut", "Museum", "CCL", 1.2988, 103.8456),
        ("Bras Basah", "Museum", "CCL", 1.2965, 103.8507),
        ("Esplanade", "Downtown", "CCL", 1.2934, 103.8557),
        ("Promenade", "Downtown", "CCL", 1.2933, 103.8605),
        ("Nicoll Highway", "Kallang", "CCL", 1.2999, 103.8636),
        ("Stadium", "Kallang", "CCL", 1.3027, 103.8748),
        ("Mountbatten", "Kallang", "CCL", 1.3064, 103.8822),
        ("Dakota", "Geylang", "CCL", 1.3083, 103.8883),
        ("Paya Lebar", "Geylang", "CCL", 1.3178, 103.8926),
        ("MacPherson", "Geylang", "CCL", 1.3267, 103.8894),
        ("Tai Seng", "Serangoon", "CCL", 1.3354, 103.8879),
        ("Bartley", "Serangoon", "CCL", 1.3424, 103.8799),
        ("Serangoon", "Serangoon", "CCL", 1.3497, 103.8732),
        ("Lorong Chuan", "Serangoon", "CCL", 1.3524, 103.8641),
        ("Bishan", "Bishan", "CCL", 1.3510, 103.8485),
        ("Marymount", "Bishan", "CCL", 1.3483, 103.8395),
        ("Caldecott", "Bukit Timah", "CCL", 1.3375, 103.8323),
        ("Botanic Gardens", "Bukit Timah", "CCL", 1.3222, 103.8155),
        ("Farrer Road", "Bukit Timah", "CCL", 1.3172, 103.8070),
        ("Holland Village", "Buona Vista", "CCL", 1.3115, 103.7960),
        ("Buona Vista", "Queenstown", "CCL", 1.3067, 103.7901),
        ("one-north", "Buona Vista", "CCL", 1.2995, 103.7869),
        ("Kent Ridge", "Clementi", "CCL", 1.2938, 103.7847),
        ("Haw Par Villa", "Queenstown", "CCL", 1.2827, 103.7822),
        ("Pasir Panjang", "Queenstown", "CCL", 1.2763, 103.7916),
        ("Labrador Park", "Buona Vista", "CCL", 1.2722, 103.8023),
        ("Telok Blangah", "Bukit Merah", "CCL", 1.2706, 103.8092),
        ("HarbourFront", "Bukit Merah", "CCL", 1.2653, 103.8218),
        # --- Downtown Line (DTL) ---
        ("Bukit Panjang", "Bukit Panjang", "DTL", 1.3784, 103.7659),
        ("Cashew", "Bukit Panjang", "DTL", 1.3698, 103.7749),
        ("Hillview", "Bukit Timah", "DTL", 1.3626, 103.7677),
        ("Beauty World", "Bukit Timah", "DTL", 1.3409, 103.7759),
        ("King Albert Park", "Bukit Timah", "DTL", 1.3354, 103.7832),
        ("Sixth Avenue", "Bukit Timah", "DTL", 1.3297, 103.7957),
        ("Tan Kah Kee", "Bukit Timah", "DTL", 1.3258, 103.8075),
        ("Botanic Gardens", "Bukit Timah", "DTL", 1.3222, 103.8155),
        ("Stevens", "Novena", "DTL", 1.3197, 103.8264),
        ("Newton", "Novena", "DTL", 1.3132, 103.8388),
        ("Little India", "Rochor", "DTL", 1.3067, 103.8496),
        ("Rochor", "Rochor", "DTL", 1.3039, 103.8526),
        ("Bugis", "Rochor", "DTL", 1.3008, 103.8559),
        ("Promenade", "Downtown", "DTL", 1.2933, 103.8605),
        ("Bayfront", "Downtown", "DTL", 1.2830, 103.8593),
        ("Downtown", "Downtown", "DTL", 1.2793, 103.8529),
        ("Telok Ayer", "Downtown", "DTL", 1.2812, 103.8479),
        ("Fort Canning", "Museum", "DTL", 1.2936, 103.8445),
        ("Bencoolen", "Rochor", "DTL", 1.2980, 103.8499),
        ("Jalan Besar", "Rochor", "DTL", 1.3059, 103.8567),
        ("Bendemeer", "Kallang", "DTL", 1.3135, 103.8617),
        ("Geylang Bahru", "Kallang", "DTL", 1.3214, 103.8710),
        ("Mattar", "Geylang", "DTL", 1.3263, 103.8836),
        ("Ubi", "Geylang", "DTL", 1.3294, 103.8956),
        ("Kaki Bukit", "Bedok", "DTL", 1.3342, 103.9054),
        ("Bedok North", "Bedok", "DTL", 1.3337, 103.9179),
        ("Bedok Reservoir", "Bedok", "DTL", 1.3356, 103.9296),
        ("Tampines West", "Tampines", "DTL", 1.3449, 103.9368),
        ("Tampines", "Tampines", "DTL", 1.3526, 103.9453),
        ("Tampines East", "Tampines", "DTL", 1.3574, 103.9533),
        ("Upper Changi", "Changi", "DTL", 1.3413, 103.9615),
        ("Expo", "Changi", "DTL", 1.3350, 103.9612),
        ("Changi Airport", "Changi", "DTL", 1.3592, 103.9885),
        # --- Thomson-East Coast Line (TEL) ---
        ("Woodlands North", "Woodlands", "TEL", 1.4487, 103.7875),
        ("Woodlands", "Woodlands", "TEL", 1.4369, 103.7863),
        ("Woodlands South", "Woodlands", "TEL", 1.4242, 103.7986),
        ("Springleaf", "Ang Mo Kio", "TEL", 1.3980, 103.8149),
        ("Lentor", "Ang Mo Kio", "TEL", 1.3848, 103.8354),
        ("Mayflower", "Ang Mo Kio", "TEL", 1.3706, 103.8397),
        ("Bright Hill", "Bishan", "TEL", 1.3621, 103.8379),
        ("Upper Thomson", "Bishan", "TEL", 1.3536, 103.8327),
        ("Caldecott", "Bukit Timah", "TEL", 1.3375, 103.8323),
        ("Stevens", "Novena", "TEL", 1.3197, 103.8264),
        ("Napier", "Tanglin", "TEL", 1.3074, 103.8186),
        ("Orchard Boulevard", "Orchard", "TEL", 1.3028, 103.8228),
        ("Orchard", "Orchard", "TEL", 1.3048, 103.8318),
        ("Great World", "Queenstown", "TEL", 1.2952, 103.8296),
        ("Havelock", "Outram", "TEL", 1.2883, 103.8343),
        ("Outram Park", "Outram", "TEL", 1.2801, 103.8396),
        ("Maxwell", "Downtown", "TEL", 1.2803, 103.8453),
        ("Shenton Way", "Downtown", "TEL", 1.2777, 103.8490),
        ("Marina Bay", "Downtown", "TEL", 1.2762, 103.8554),
        ("Marina South", "Downtown", "TEL", 1.2701, 103.8623),
        ("Gardens by the Bay", "Downtown", "TEL", 1.2816, 103.8635),
        ("Tanjong Rhu", "Kallang", "TEL", 1.2990, 103.8720),
        ("Katong Park", "Marine Parade", "TEL", 1.3026, 103.8830),
        ("Tanjong Katong", "Marine Parade", "TEL", 1.3069, 103.8933),
        ("Marine Parade", "Marine Parade", "TEL", 1.3032, 103.9043),
        ("Marine Terrace", "Marine Parade", "TEL", 1.3063, 103.9157),
        ("Siglap", "Bedok", "TEL", 1.3100, 103.9271),
        ("Bayshore", "Bedok", "TEL", 1.3155, 103.9375),
        ("Bedok South", "Bedok", "TEL", 1.3201, 103.9453),
        ("Sungei Bedok", "Bedok", "TEL", 1.3261, 103.9543),
    ]

    # De-duplicate by station name + line
    seen = set()
    unique_stations = []
    for s in stations:
        key = (s[0], s[2])
        if key not in seen:
            seen.add(key)
            unique_stations.append(s)

    m = len(unique_stations)
    names = [s[0] for s in unique_stations]
    towns = [s[1] for s in unique_stations]
    lines = [s[2] for s in unique_stations]
    lats = [s[3] for s in unique_stations]
    lons = [s[4] for s in unique_stations]

    nearest_mrt = []
    dist_to_nearest = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        best_name = names[i]
        best_dist = float("inf")
        for j, (la2, lo2) in enumerate(zip(lats, lons)):
            if j == i or lines[j] != lines[i]:
                continue
            dlat = math.radians(la2 - la) * 6371
            dlon = math.radians(lo2 - lo) * 6371 * math.cos(math.radians(la))
            d = math.sqrt(dlat**2 + dlon**2)
            if d < best_dist:
                best_dist = d
                best_name = names[j]
        nearest_mrt.append(best_name)
        dist_to_nearest.append(round(best_dist, 3))

    df = pl.DataFrame(
        {
            "station_name": names,
            "town": towns,
            "line": lines,
            "latitude": lats,
            "longitude": lons,
            "nearest_mrt": nearest_mrt,
            "distance_to_mrt_km": dist_to_nearest,
        }
    )

    return df


# ---------------------------------------------------------------------------
# 10. schools.parquet  (ascent_assessment) — 350 rows
# ---------------------------------------------------------------------------


def make_schools() -> pl.DataFrame:
    """
    Singapore schools (~350 rows) covering primary, secondary, and JC.
    """
    primary_schools = [
        ("Ai Tong School", "Bishan"),
        ("Alexandra Primary School", "Queenstown"),
        ("Anchor Green Primary School", "Pasir Ris"),
        ("Anderson Primary School", "Ang Mo Kio"),
        ("Ang Mo Kio Primary School", "Ang Mo Kio"),
        ("Balestier Hill Primary School", "Toa Payoh"),
        ("Beacon Primary School", "Woodlands"),
        ("Bedok Green Primary School", "Bedok"),
        ("Bendemeer Primary School", "Kallang"),
        ("Blangah Rise Primary School", "Bukit Merah"),
        ("Boon Lay Garden Primary School", "Jurong West"),
        ("Bukit Batok Primary School", "Bukit Batok"),
        ("Bukit Panjang Primary School", "Bukit Panjang"),
        ("Bukit Timah Primary School", "Bukit Timah"),
        ("Bukit View Primary School", "Bukit Timah"),
        ("Casuarina Primary School", "Yishun"),
        ("Canberra Primary School", "Sembawang"),
        ("Cedar Primary School", "Serangoon"),
        ("Changkat Primary School", "Tampines"),
        ("Chij (Kellock)", "Queenstown"),
        ("Chij Primary (Toa Payoh)", "Toa Payoh"),
        ("Chongzheng Primary School", "Bishan"),
        ("Clementi Primary School", "Clementi"),
        ("Compass Primary School", "Sengkang"),
        ("Damai Primary School", "Bedok"),
        ("Edgefield Primary School", "Punggol"),
        ("Elias Park Primary School", "Pasir Ris"),
        ("Endeavour Primary School", "Sembawang"),
        ("Eunos Primary School", "Bedok"),
        ("Farrer Park Primary School", "Little India"),
        ("Fengshan Primary School", "Bedok"),
        ("Fernvale Primary School", "Sengkang"),
        ("Frontier Primary School", "Jurong West"),
        ("Fuhua Primary School", "Jurong West"),
        ("Geylang Methodist School (Primary)", "Geylang"),
        ("Greendale Primary School", "Punggol"),
        ("Greenridge Primary School", "Bukit Panjang"),
        ("Greenwood Primary School", "Bukit Timah"),
        ("Griffiths Primary School", "Bukit Merah"),
        ("Henry Park Primary School", "Buona Vista"),
        ("Holy Innocents' Primary School", "Hougang"),
        ("Horizon Primary School", "Pasir Ris"),
        ("Hougang Primary School", "Hougang"),
        ("Jiemin Primary School", "Bishan"),
        ("Jurongville Primary School", "Jurong West"),
        ("Keming Primary School", "Jurong West"),
        ("Kong Hwa School", "Geylang"),
        ("Kranji Primary School", "Woodlands"),
        ("Lianhua Primary School", "Clementi"),
        ("Maha Bodhi School", "Geylang"),
        ("Marsiling Primary School", "Woodlands"),
        ("Marymount Convent School", "Bishan"),
        ("Mee Toh School", "Tampines"),
        ("Meridian Primary School", "Pasir Ris"),
        ("Methodist Girls' School (Primary)", "Buona Vista"),
        ("Nan Chiau Primary School", "Sengkang"),
        ("Nanyang Primary School", "Buona Vista"),
        ("Naval Base Primary School", "Yishun"),
        ("New Town Primary School", "Queenstown"),
        ("Ngee Ann Primary School", "Hougang"),
        ("North Spring Primary School", "Hougang"),
        ("North View Primary School", "Woodlands"),
        ("Northland Primary School", "Woodlands"),
        ("Palm View Primary School", "Jurong West"),
        ("Park View Primary School", "Bukit Batok"),
        ("Pasir Ris Primary School", "Pasir Ris"),
        ("Pei Chun Public School", "Toa Payoh"),
        ("Pei Hwa Presbyterian Primary School", "Tampines"),
        ("Pei Tong Primary School", "Clementi"),
        ("Poi Ching School", "Tampines"),
        ("Punggol Green Primary School", "Punggol"),
        ("Qihua Primary School", "Choa Chu Kang"),
        ("Radin Mas Primary School", "Bukit Merah"),
        ("Raffles Girls' Primary School", "Buona Vista"),
        ("Red Swastika School", "Bedok"),
        ("River Valley Primary School", "Buona Vista"),
        ("Riverside Primary School", "Woodlands"),
        ("Rosyth School", "Hougang"),
        ("Rulang Primary School", "Jurong East"),
        ("Sembawang Primary School", "Sembawang"),
        ("Sengkang Green Primary School", "Sengkang"),
        ("Shuqun Primary School", "Jurong West"),
        ("Si Ling Primary School", "Woodlands"),
        ("Springdale Primary School", "Choa Chu Kang"),
        ("St Andrew's Junior School", "Tampines"),
        ("St Anthony's Canossian Primary School", "Bedok"),
        ("St Hilda's Primary School", "Tampines"),
        ("St Joseph's Institution Junior", "Buona Vista"),
        ("Tampines North Primary School", "Tampines"),
        ("Tampines Primary School", "Tampines"),
        ("Tanjong Katong Primary School", "Marine Parade"),
        ("Temasek Primary School", "Bedok"),
        ("Townsville Primary School", "Hougang"),
        ("Unity Primary School", "Choa Chu Kang"),
        ("Waterway Primary School", "Punggol"),
        ("West Grove Primary School", "Clementi"),
        ("Westwood Primary School", "Jurong West"),
        ("White Sands Primary School", "Pasir Ris"),
        ("Woodgrove Primary School", "Woodlands"),
        ("Woodlands Ring Primary School", "Woodlands"),
        ("Xinghua Primary School", "Hougang"),
        ("Yew Tee Primary School", "Choa Chu Kang"),
        ("Yuhua Primary School", "Jurong East"),
        ("Yumin Primary School", "Hougang"),
        ("Zhangde Primary School", "Ang Mo Kio"),
        ("Zhenghua Primary School", "Bukit Panjang"),
        ("Punggol Cove Primary School", "Punggol"),
        ("Punggol View Primary School", "Punggol"),
        ("Springside Primary School", "Yishun"),
        ("St Margaret's Primary School", "Queenstown"),
        ("Teck Ghee Primary School", "Ang Mo Kio"),
        ("Telok Kurau Primary School", "Geylang"),
        ("Victoria School (Primary)", "Marine Parade"),
        ("Woodlands Primary School", "Woodlands"),
    ]

    secondary_schools = [
        ("Anderson Secondary School", "Ang Mo Kio"),
        ("Anglican High School", "Bedok"),
        ("Ang Mo Kio Secondary School", "Ang Mo Kio"),
        ("Assumption English School", "Bukit Timah"),
        ("Beatty Secondary School", "Toa Payoh"),
        ("Bedok North Secondary School", "Bedok"),
        ("Bedok South Secondary School", "Bedok"),
        ("Bedok View Secondary School", "Bedok"),
        ("Bendemeer Secondary School", "Kallang"),
        ("Bishan Park Secondary School", "Bishan"),
        ("Boon Lay Secondary School", "Jurong West"),
        ("Bowen Secondary School", "Hougang"),
        ("Broadrick Secondary School", "Geylang"),
        ("Bukit Batok Secondary School", "Bukit Batok"),
        ("Bukit Merah Secondary School", "Bukit Merah"),
        ("Bukit Panjang Govt High", "Bukit Panjang"),
        ("Bukit View Secondary School", "Bukit Timah"),
        ("Catholic High School", "Bishan"),
        ("Cedar Girls' Secondary School", "Serangoon"),
        ("Chij Secondary (Toa Payoh)", "Toa Payoh"),
        ("Chua Chu Kang Secondary School", "Choa Chu Kang"),
        ("Clementi Town Secondary School", "Clementi"),
        ("Commonwealth Secondary School", "Queenstown"),
        ("Compassvale Secondary School", "Sengkang"),
        ("Coral Secondary School", "Pasir Ris"),
        ("Crescent Girls' School", "Queenstown"),
        ("Deyi Secondary School", "Ang Mo Kio"),
        ("Dunman High School", "Marine Parade"),
        ("Dunman Secondary School", "Geylang"),
        ("East Spring Secondary School", "Tampines"),
        ("Edgefield Secondary School", "Punggol"),
        ("Evergreen Secondary School", "Woodlands"),
        ("Fajar Secondary School", "Bukit Panjang"),
        ("Fuchun Secondary School", "Woodlands"),
        ("Fuhua Secondary School", "Jurong West"),
        ("Gan Eng Seng School", "Bukit Merah"),
        ("Geylang Methodist School (Sec)", "Geylang"),
        ("Greendale Secondary School", "Punggol"),
        ("Greenridge Secondary School", "Bukit Panjang"),
        ("Guangyang Secondary School", "Bishan"),
        ("Hai Sing Catholic School", "Pasir Ris"),
        ("Holy Innocents' High School", "Hougang"),
        ("Hougang Secondary School", "Hougang"),
        ("Hwa Chong Institution", "Bukit Timah"),
        ("Jurongville Secondary School", "Jurong West"),
        ("Jurong West Secondary School", "Jurong West"),
        ("Kent Ridge Secondary School", "Clementi"),
        ("Kranji Secondary School", "Woodlands"),
        ("Kuo Chuan Presbyterian Secondary", "Bishan"),
        ("Loyang View Secondary School", "Pasir Ris"),
        ("Manjusri Secondary School", "Ang Mo Kio"),
        ("Marsiling Secondary School", "Woodlands"),
        ("Mayflower Secondary School", "Ang Mo Kio"),
        ("Methodist Girls' School (Sec)", "Buona Vista"),
        ("Montfort Secondary School", "Tampines"),
        ("Nan Chiau High School", "Sengkang"),
        ("Nan Hua High School", "Clementi"),
        ("Nanyang Girls' High School", "Buona Vista"),
        ("National Junior College (Sec)", "Buona Vista"),
        ("Naval Base Secondary School", "Yishun"),
        ("New Town Secondary School", "Queenstown"),
        ("Ngee Ann Secondary School", "Hougang"),
        ("North Spring Secondary School", "Hougang"),
        ("North View Secondary School", "Woodlands"),
        ("Northbrooks Secondary School", "Woodlands"),
        ("Northland Secondary School", "Woodlands"),
        ("NUS High School of Math & Science", "Buona Vista"),
        ("Orchid Park Secondary School", "Yishun"),
        ("Paya Lebar Methodist Girls' (Sec)", "Geylang"),
        ("Pei Hwa Secondary School", "Sengkang"),
        ("Presbyterian High School", "Ang Mo Kio"),
        ("Punggol Secondary School", "Punggol"),
        ("Queenstown Secondary School", "Queenstown"),
        ("Queensway Secondary School", "Queenstown"),
        ("Raffles Girls' School (Sec)", "Buona Vista"),
        ("Raffles Institution (Sec)", "Bishan"),
        ("Riverside Secondary School", "Woodlands"),
        ("School of Science & Technology", "Jurong West"),
        ("Sembawang Secondary School", "Sembawang"),
        ("Sengkang Secondary School", "Sengkang"),
        ("Singapore Sports School", "Woodlands"),
        ("Siying Secondary School", "Woodlands"),
        ("Springfield Secondary School", "Hougang"),
        ("St Andrew's Secondary School", "Tampines"),
        ("St Anthony's Canossian Sec", "Bedok"),
        ("St Gabriel's Secondary School", "Serangoon"),
        ("St Hilda's Secondary School", "Tampines"),
        ("St Joseph's Institution", "Buona Vista"),
        ("St Margaret's Secondary School", "Queenstown"),
        ("St Patrick's School", "Marine Parade"),
        ("Tampines Secondary School", "Tampines"),
        ("Tanjong Katong Girls' School", "Marine Parade"),
        ("Tanjong Katong Secondary School", "Marine Parade"),
        ("Teck Whye Secondary School", "Choa Chu Kang"),
        ("Temasek Secondary School", "Bedok"),
        ("Unity Secondary School", "Choa Chu Kang"),
        ("Victoria School", "Marine Parade"),
        ("West Spring Secondary School", "Bukit Panjang"),
        ("West View Secondary School", "Jurong West"),
        ("Westwood Secondary School", "Jurong West"),
        ("Whitley Secondary School", "Bishan"),
        ("Woodgrove Secondary School", "Woodlands"),
        ("Woodlands Ring Secondary School", "Woodlands"),
        ("Woodlands Secondary School", "Woodlands"),
        ("Xinmin Secondary School", "Hougang"),
        ("Yang Zheng Secondary School", "Ang Mo Kio"),
        ("Yio Chu Kang Secondary School", "Ang Mo Kio"),
        ("Yuhua Secondary School", "Jurong East"),
        ("Yuan Ching Secondary School", "Jurong West"),
        ("Yusof Ishak Secondary School", "Buona Vista"),
        ("Zhonghua Secondary School", "Serangoon"),
    ]

    jcs = [
        ("Anderson Serangoon JC", "Serangoon"),
        ("Anglo-Chinese JC", "Bishan"),
        ("Dunman High School (JC)", "Marine Parade"),
        ("Eunoia Junior College", "Bishan"),
        ("Hwa Chong Institution (JC)", "Bukit Timah"),
        ("Jurong Pioneer JC", "Jurong West"),
        ("Millennia Institute", "Bukit Timah"),
        ("Nanyang JC", "Buona Vista"),
        ("National JC", "Buona Vista"),
        ("NUS High School (JC)", "Buona Vista"),
        ("Raffles Institution (JC)", "Bishan"),
        ("River Valley High School (JC)", "Buona Vista"),
        ("St Andrew's Junior College", "Tampines"),
        ("Tampines Meridian JC", "Tampines"),
        ("Temasek JC", "Tampines"),
        ("Victoria JC", "Marine Parade"),
        ("Yishun Innova JC", "Yishun"),
    ]

    all_schools = (
        [(n, t, "primary") for n, t in primary_schools]
        + [(n, t, "secondary") for n, t in secondary_schools]
        + [(n, t, "JC") for n, t in jcs]
    )

    north = {"Woodlands", "Sembawang", "Yishun"}
    east = {
        "Tampines",
        "Pasir Ris",
        "Bedok",
        "Changi",
        "Marine Parade",
        "Geylang",
        "Kallang",
    }
    west = {
        "Jurong West",
        "Jurong East",
        "Clementi",
        "Buona Vista",
        "Queenstown",
        "Bukit Timah",
        "Bukit Batok",
        "Bukit Panjang",
        "Choa Chu Kang",
    }
    ne = {"Sengkang", "Punggol", "Hougang", "Serangoon", "Ang Mo Kio", "Bishan"}

    def zone(town: str) -> str:
        if town in north:
            return "North"
        if town in east:
            return "East"
        if town in west:
            return "West"
        if town in ne:
            return "North-East"
        return "Central"

    df = pl.DataFrame(
        {
            "school_name": [s[0] for s in all_schools],
            "town": [s[1] for s in all_schools],
            "type": [s[2] for s in all_schools],
            "zone": [zone(s[1]) for s in all_schools],
        }
    )

    return df[:350]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  directory ready: {d}")

    tasks = [
        (
            make_economic_indicators,
            DATA_ROOT / "ascent01" / "economic_indicators.csv",
            "csv",
        ),
        # sg_taxi_trips is now parquet (50K rows too large for CSV in git)
        (make_sg_taxi_trips, DATA_ROOT / "ascent01" / "sg_taxi_trips.parquet", "parquet"),
        (
            make_experiment_data,
            DATA_ROOT / "ascent02" / "experiment_data.parquet",
            "parquet",
        ),
        (
            make_sg_credit_scoring,
            DATA_ROOT / "ascent03" / "sg_credit_scoring.parquet",
            "parquet",
        ),
        (
            make_ecommerce_customers,
            DATA_ROOT / "ascent04" / "ecommerce_customers.parquet",
            "parquet",
        ),
        (make_documents, DATA_ROOT / "ascent05" / "documents.parquet", "parquet"),
        (make_sg_domain_qa, DATA_ROOT / "ascent06" / "sg_domain_qa.parquet", "parquet"),
        (
            make_preference_pairs,
            DATA_ROOT / "ascent06" / "preference_pairs.parquet",
            "parquet",
        ),
        (
            make_mrt_stations,
            DATA_ROOT / "ascent_assessment" / "mrt_stations.parquet",
            "parquet",
        ),
        (make_schools, DATA_ROOT / "ascent_assessment" / "schools.parquet", "parquet"),
    ]

    print("\nGenerating datasets...")
    for fn, path, mode in tasks:
        df = fn()
        if mode == "csv":
            df.write_csv(str(path))
        else:
            df.write_parquet(str(path), compression="zstd")
        size_kb = path.stat().st_size // 1024
        print(
            f"  {path.name:<50} {len(df):>8} rows  {size_kb:>6} KB  ({path.parent.name})"
        )

    # Remove the old sg_taxi_trips.csv if it exists (replaced by parquet)
    old_csv = DATA_ROOT / "ascent01" / "sg_taxi_trips.csv"
    if old_csv.exists():
        old_csv.unlink()
        print(f"\n  Removed old file: {old_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
