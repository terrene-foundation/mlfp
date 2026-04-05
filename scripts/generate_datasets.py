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


# ---------------------------------------------------------------------------
# 1. economic_indicators.csv  (ascent01)
# ---------------------------------------------------------------------------


def make_economic_indicators() -> pl.DataFrame:
    """
    Singapore economic indicators 2000-2024 (quarterly).
    ~100 rows (25 years × 4 quarters).

    Intentional messiness:
    - Missing values in inflation_rate (~8%) and trade_balance (~6%)
    - Mixed quarter formats: some "Q1", some "1Q", some plain "1"
    - GDP outlier: 2020 Q1-Q2 crash (-13% realistic but jarring)
    - tourist_arrivals as string with commas (e.g., "1,234,567")
    - Duplicate row: 2019 Q2 appears twice
    """
    years = list(range(2000, 2025))  # 25 years
    quarters = [1, 2, 3, 4]
    rows = [(y, q) for y in years for q in quarters]  # 100 rows

    n = len(rows)

    # --- GDP growth (%) — realistic SG trajectory ---
    gdp_base = np.array(
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

    gdp_noise = RNG.normal(0, 0.3, n)
    gdp_growth = gdp_base + gdp_noise

    # --- Unemployment rate (%) ---
    unemp_base = np.array(
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

    unemp = np.clip(unemp_base + RNG.normal(0, 0.1, n), 1.0, 6.0)

    # --- Inflation (CPI YoY %) ---
    inflation_base = RNG.uniform(0.5, 3.5, n)
    # COVID supply shock
    inflation_base[80:84] = RNG.uniform(4.5, 6.5, 4)  # 2020
    inflation_base[84:88] = RNG.uniform(3.5, 5.5, 4)  # 2021
    inflation_base[88:92] = RNG.uniform(5.0, 7.0, 4)  # 2022 peak
    inflation_base[92:96] = RNG.uniform(3.0, 5.0, 4)  # 2023 easing
    inflation_raw = _nullify(np.round(inflation_base, 2), rate=0.08)

    # --- Trade balance (SGD bn) ---
    trade_base = RNG.uniform(5.0, 22.0, n)
    trade_raw = _nullify(np.round(trade_base, 1), rate=0.06)

    # --- Property price index (base 100 = 2000 Q1) ---
    ppi = np.zeros(n)
    ppi[0] = 100.0
    quarterly_drift = 0.008  # ~3.2% per year long-run
    for i in range(1, n):
        shock = RNG.normal(quarterly_drift, 0.025)
        # 2009 dip, 2020 flat, then surge
        if 32 <= i <= 35:
            shock -= 0.04
        if 80 <= i <= 83:
            shock -= 0.01
        if 84 <= i <= 91:
            shock += 0.02
        ppi[i] = ppi[i - 1] * (1 + shock)

    # --- Tourist arrivals (millions) — stored as messy string ---
    tourists_m = np.zeros(n)
    tourists_m[0] = 5.1
    for i in range(1, n):
        base_growth = 0.015
        shock = RNG.normal(base_growth, 0.04)
        if 32 <= i <= 35:
            shock -= 0.08  # SARS 2003
        if 80 <= i <= 87:
            tourists_m[i] = tourists_m[i - 1] * 0.02  # COVID collapse
            continue
        if 88 <= i <= 95:
            shock += 0.25  # rapid rebound
        tourists_m[i] = max(0.1, tourists_m[i - 1] * (1 + shock))

    # Convert to integer arrivals (thousands) as string with commas — messy
    arrivals_int = (tourists_m * 1_000_000).astype(int)
    arrivals_str = [f"{v:,}" for v in arrivals_int]

    # --- Quarter format: intentional inconsistency ---
    fmt_choices = ["Q{q}", "{q}Q", "{q}"]
    quarter_fmt = []
    for i, (y, q) in enumerate(rows):
        choice = RNG.integers(0, 3)
        if choice == 0:
            quarter_fmt.append(f"Q{q}")
        elif choice == 1:
            quarter_fmt.append(f"{q}Q")
        else:
            quarter_fmt.append(str(q))

    df = pl.DataFrame(
        {
            "year": [r[0] for r in rows],
            "quarter": quarter_fmt,
            "gdp_growth_pct": np.round(gdp_growth, 2).tolist(),
            "unemployment_rate": np.round(unemp, 2).tolist(),
            "inflation_rate": inflation_raw,
            "trade_balance_sgd_bn": trade_raw,
            "property_price_index": np.round(ppi, 1).tolist(),
            "tourist_arrivals": arrivals_str,
        }
    )

    # --- Duplicate row (2019 Q2 appears twice) ---
    dup_row = df.filter(
        (pl.col("year") == 2019) & (pl.col("quarter").str.contains("2"))
    )
    if len(dup_row) == 0:
        # quarter col might be "2Q" or "2"
        dup_row = df[75:76]  # 2019 Q4 fallback
    df = pl.concat([df, dup_row])

    return df


# ---------------------------------------------------------------------------
# 2. sg_taxi_trips.csv  (ascent01) — intentionally very messy
# ---------------------------------------------------------------------------


def make_sg_taxi_trips() -> pl.DataFrame:
    """
    Singapore taxi trip data (~2,000 rows).

    Intentional messiness:
    - Negative fares (~2% of rows)
    - Impossible distances: ~1% > 60 km (island is ~50 km wide)
    - Future dates (~1%) — 2027/2028 dates
    - Missing pickup/dropoff zones (~5%)
    - Inconsistent payment_type: "Cash"/"cash"/"CASH", "Card"/"credit card"/"VISA"
    - passengers = 0 or negative in ~1%
    - Missing tip_sgd in ~15% (only card payments have tips)
    """
    n = 2000

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
    ]

    # Trip IDs
    trip_ids = [f"SG-TX-{100000 + i}" for i in range(n)]

    # Datetimes — mostly 2022-2024
    start_ts = 1640995200  # 2022-01-01 00:00:00 UTC
    end_ts = 1735689600  # 2025-01-01 00:00:00 UTC
    pickup_ts = RNG.integers(start_ts, end_ts, n)
    # trip duration: 5-60 minutes
    duration_s = RNG.integers(5 * 60, 60 * 60, n)
    dropoff_ts = pickup_ts + duration_s

    # Inject future dates (~1%)
    future_idx = RNG.choice(n, size=20, replace=False)
    for i in future_idx:
        pickup_ts[i] = 1830000000  # 2028
        dropoff_ts[i] = pickup_ts[i] + duration_s[i]

    def ts_to_str(ts_arr):
        from datetime import datetime, timezone

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
    # Inject impossible distances
    impossible_dist_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
    distance[impossible_dist_idx] = RNG.uniform(65.0, 200.0, len(impossible_dist_idx))

    # Fare (SGD) — roughly 0.50 flag-fall + 0.22/100m
    fare_base = 3.90 + distance * 0.45 + RNG.normal(0, 1.5, n)
    fare_base = np.clip(fare_base, 3.90, 80.0)
    # Inject negative fares
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

    # Tip (SGD) — only card payments typically tip; 15% overall null
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
        }
    )

    return df


# ---------------------------------------------------------------------------
# 3. experiment_data.parquet  (ascent02)
# ---------------------------------------------------------------------------


def make_experiment_data() -> pl.DataFrame:
    """
    A/B test data for experiment design exercises.
    ~1,000 rows — single experiment with control/treatment arms.

    Useful for:
    - t-tests, Mann-Whitney
    - CUPED variance reduction
    - Segmented analysis (platform, segment)
    """
    n = 1000
    user_ids = [f"USR-{200000 + i}" for i in range(n)]

    groups = RNG.choice(["control", "treatment"], size=n, p=[0.50, 0.50])

    # pre_metric_value — baseline before experiment
    pre_metric = RNG.normal(50.0, 15.0, n)

    # metric_value — post experiment
    # treatment has true lift of +3 units
    noise = RNG.normal(0, 12.0, n)
    metric_value = pre_metric * 0.7 + noise
    treatment_mask = groups == "treatment"
    metric_value[treatment_mask] += 3.0

    # segment: high_value / mid_value / low_value
    segments = RNG.choice(
        ["high_value", "mid_value", "low_value"], size=n, p=[0.15, 0.55, 0.30]
    )

    # platform: mobile / desktop / tablet
    platforms = RNG.choice(
        ["mobile", "desktop", "tablet"], size=n, p=[0.60, 0.32, 0.08]
    )

    # timestamps spread over 2-week experiment window
    exp_start = 1704067200  # 2024-01-01
    exp_end = 1705276800  # 2024-01-15
    ts = RNG.integers(exp_start, exp_end, n)

    from datetime import datetime, timezone

    timestamps = [
        datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for t in ts
    ]

    df = pl.DataFrame(
        {
            "user_id": user_ids,
            "experiment_group": groups.tolist(),
            "metric_value": np.round(metric_value, 4).tolist(),
            "pre_metric_value": np.round(pre_metric, 4).tolist(),
            "timestamp": timestamps,
            "segment": segments.tolist(),
            "platform": platforms.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 4. sg_credit_scoring.parquet  (ascent03)
# ---------------------------------------------------------------------------


def make_sg_credit_scoring() -> pl.DataFrame:
    """
    Singapore credit scoring dataset (~5,000 rows).

    Realistic features:
    - SG income ranges (S$24k–S$300k/year)
    - ~8% default rate (class imbalance)
    - Correlated features (income ↔ employment_years, credit utilization ↔ default)
    """
    n = 5000
    cust_ids = [f"CUST-{300000 + i}" for i in range(n)]

    age = RNG.integers(21, 70, n)

    # Income correlated with age (experience)
    income_base = 30000 + (age - 21) * 2500 + RNG.normal(0, 15000, n)
    income_sgd = np.clip(income_base, 24000, 350000).astype(int)

    # Employment years — 0 if student/unemployed
    employment_years = np.clip((age - 22) * 0.7 + RNG.normal(0, 3, n), 0, 40).astype(
        int
    )

    # Credit utilization: 0-1, higher → more risk
    util_base = RNG.beta(2, 5, n)  # right-skewed, mostly low
    credit_utilization = np.round(np.clip(util_base, 0, 1), 4)

    # Number of credit lines
    num_credit_lines = np.clip(RNG.poisson(3.5, n), 0, 15).astype(int)

    # Payment history score: 300-850
    pay_score_base = 650 + RNG.normal(0, 80, n) - credit_utilization * 150
    payment_history_score = np.clip(pay_score_base, 300, 850).astype(int)

    # Loan amount
    loan_base = income_sgd * RNG.uniform(0.5, 4.0, n)
    loan_amount_sgd = np.round(np.clip(loan_base, 5000, 800000), -2).astype(int)

    # Loan purpose
    purposes = ["home", "car", "education", "personal", "business", "renovation"]
    loan_purpose = RNG.choice(purposes, size=n, p=[0.35, 0.20, 0.10, 0.20, 0.10, 0.05])

    # Default (~8%): logistic model on risk factors
    # Intercept calibrated so base rate ≈ 8% given typical feature values
    log_odds = (
        -2.5
        + 2.5 * credit_utilization
        - 0.3 * (payment_history_score - 650) / 80
        + 0.5 * (loan_amount_sgd / income_sgd - 2.0)
        - 0.1 * employment_years
        + RNG.normal(0, 0.3, n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (RNG.random(n) < prob_default).astype(int)

    df = pl.DataFrame(
        {
            "customer_id": cust_ids,
            "age": age.tolist(),
            "income_sgd": income_sgd.tolist(),
            "employment_years": employment_years.tolist(),
            "credit_utilization": credit_utilization.tolist(),
            "num_credit_lines": num_credit_lines.tolist(),
            "payment_history_score": payment_history_score.tolist(),
            "loan_amount_sgd": loan_amount_sgd.tolist(),
            "loan_purpose": loan_purpose.tolist(),
            "default": default.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 5. ecommerce_customers.parquet  (ascent04)
# ---------------------------------------------------------------------------


def make_ecommerce_customers() -> pl.DataFrame:
    """
    E-commerce customer data for clustering + NLP exercises (~3,000 rows).
    Mix of numeric RFM features and text fields.
    """
    n = 3000
    cust_ids = [f"EC-{400000 + i}" for i in range(n)]

    # RFM features
    total_revenue = np.round(RNG.exponential(scale=250.0, size=n), 2)
    order_count = np.clip(RNG.poisson(7, n), 1, 80).astype(int)
    avg_order_value = np.round(total_revenue / order_count, 2)
    days_since_last_order = np.clip(RNG.integers(1, 730, n), 1, 730).astype(int)

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
    ]

    def random_categories():
        k = RNG.integers(1, 4)
        chosen = random.sample(categories_pool, int(k))
        return ", ".join(chosen)

    product_categories = [random_categories() for _ in range(n)]

    # Review text — template-based, realistic but short
    pos_templates = [
        "Great product, fast delivery!",
        "Exactly what I needed. Will buy again.",
        "Good quality for the price.",
        "Happy with my purchase. Prompt shipping.",
        "Works perfectly. Highly recommend.",
        "Exceeded my expectations. Top notch.",
        "Very satisfied. Would order again.",
        "Quality is excellent. Arrived on time.",
    ]
    neg_templates = [
        "Item arrived damaged. Very disappointed.",
        "Not as described. Poor quality.",
        "Delivery took too long. Packaging was bad.",
        "Would not recommend. Waste of money.",
        "Product stopped working after one week.",
        "Customer service was unhelpful.",
        "Size was wrong. Had to return.",
        "Missing parts. Had to contact support.",
    ]
    neutral_templates = [
        "Decent product. Nothing special.",
        "Okay for the price.",
        "Average quality. Does the job.",
        "Shipping was slow but product is fine.",
        "Not bad. Could be better.",
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

    # Region — SG + ASEAN
    regions = RNG.choice(
        ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"],
        size=n,
        p=[0.50, 0.20, 0.12, 0.08, 0.06, 0.04],
    )

    df = pl.DataFrame(
        {
            "customer_id": cust_ids,
            "total_revenue": total_revenue.tolist(),
            "order_count": order_count.tolist(),
            "avg_order_value": avg_order_value.tolist(),
            "days_since_last_order": days_since_last_order.tolist(),
            "product_categories": product_categories,
            "review_text": review_text,
            "satisfaction_score": satisfaction.tolist(),
            "region": regions.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 6. documents.parquet  (ascent05)
# ---------------------------------------------------------------------------


def make_documents() -> pl.DataFrame:
    """
    Short knowledge-base articles for RAG exercise (~200 rows).
    Topics cover ML concepts, Singapore context, and general tech.
    """
    articles = [
        # --- ML Concepts ---
        (
            "What is supervised learning?",
            "Supervised learning trains a model on labelled data where each example has an input and a known output. The model learns a mapping function. Common algorithms include linear regression, decision trees, and neural networks.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is unsupervised learning?",
            "Unsupervised learning discovers patterns in data without labels. Clustering groups similar data points. Dimensionality reduction compresses features. Autoencoders learn compact representations.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "Explain overfitting and underfitting.",
            "Overfitting occurs when a model memorises training data and fails on new data (high variance). Underfitting occurs when a model is too simple to capture patterns (high bias). Regularisation, cross-validation, and more data help combat overfitting.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is cross-validation?",
            "Cross-validation evaluates model performance by splitting data into k folds. The model trains on k-1 folds and evaluates on the held-out fold. Repeating across all folds gives a robust performance estimate.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is gradient descent?",
            "Gradient descent minimises a loss function by iteratively moving parameters in the direction of steepest descent. Learning rate controls step size. Stochastic gradient descent uses mini-batches for efficiency.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is a confusion matrix?",
            "A confusion matrix summarises classification results. Rows are actual classes, columns are predicted classes. It reveals true positives, false positives, true negatives, and false negatives for each class.",
            "ml_evaluation",
            "textbook",
        ),
        (
            "Precision vs Recall trade-off.",
            "Precision measures how many predicted positives are correct. Recall measures how many actual positives are found. The F1 score is their harmonic mean. Adjusting the classification threshold shifts this trade-off.",
            "ml_evaluation",
            "textbook",
        ),
        (
            "What is ROC-AUC?",
            "ROC-AUC measures a classifier's ability to distinguish classes across all thresholds. AUC = 1.0 is perfect. AUC = 0.5 is random. It is insensitive to class imbalance compared to accuracy.",
            "ml_evaluation",
            "textbook",
        ),
        (
            "Feature engineering overview.",
            "Feature engineering transforms raw data into informative inputs. Techniques include normalisation, one-hot encoding, polynomial features, interaction terms, and domain-specific transformations. Good features often matter more than model choice.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is one-hot encoding?",
            "One-hot encoding converts categorical variables into binary columns. Each category becomes a column with value 1 if present, 0 otherwise. Avoids imposing ordinal relationships.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is feature scaling?",
            "Feature scaling standardises the range of input features. Z-score normalisation centres features at 0 with unit variance. Min-max scaling maps features to [0, 1]. Required for distance-based algorithms like k-NN and SVM.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is PCA?",
            "Principal Component Analysis reduces dimensionality by projecting data onto orthogonal axes of maximum variance. The first principal component explains the most variance. Used for visualisation, noise reduction, and feature compression.",
            "dimensionality_reduction",
            "textbook",
        ),
        (
            "What is k-means clustering?",
            "K-means partitions data into k clusters by iteratively assigning points to the nearest centroid and updating centroids. Sensitive to initialisation and assumes spherical clusters. Use the elbow method to choose k.",
            "clustering",
            "textbook",
        ),
        (
            "What is DBSCAN?",
            "DBSCAN clusters points based on density. It can find arbitrarily shaped clusters and identifies outliers as noise. Parameters: epsilon (neighbourhood radius) and min_samples (density threshold).",
            "clustering",
            "textbook",
        ),
        (
            "What are transformers?",
            "Transformers use self-attention to process sequences in parallel. They underpin modern LLMs. The attention mechanism weighs how much each token attends to every other token in context.",
            "deep_learning",
            "textbook",
        ),
        (
            "What is transfer learning?",
            "Transfer learning fine-tunes a pre-trained model on a new task. The pre-trained model has learned general representations from large data. Fine-tuning adapts these to the specific domain with less data.",
            "deep_learning",
            "textbook",
        ),
        (
            "What is RAG?",
            "Retrieval-Augmented Generation combines a retrieval system with a language model. The retriever finds relevant documents. The generator conditions its response on those documents. Reduces hallucination and keeps knowledge current.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is prompt engineering?",
            "Prompt engineering crafts inputs to elicit desired behaviour from language models. Techniques include few-shot examples, chain-of-thought reasoning, role specification, and output format constraints.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is fine-tuning an LLM?",
            "Fine-tuning adapts a pre-trained LLM on task-specific data. Supervised fine-tuning (SFT) uses labelled instruction-response pairs. RLHF aligns responses using human preference signals.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is LoRA?",
            "Low-Rank Adaptation (LoRA) fine-tunes LLMs by injecting trainable low-rank matrices into transformer layers. It reduces trainable parameters by 10-100x while matching full fine-tuning quality.",
            "llm_techniques",
            "textbook",
        ),
        # --- Singapore Context ---
        (
            "What is the HDB in Singapore?",
            "The Housing Development Board (HDB) is Singapore's public housing authority. Over 80% of residents live in HDB flats. Flats are sold on 99-year leases. The resale market is active and regulated by income ceilings and ethnic quotas.",
            "singapore_housing",
            "data.gov.sg",
        ),
        (
            "How does the Singapore MRT work?",
            "Singapore's Mass Rapid Transit (MRT) network has six lines covering the island. The EWL, NSL, CCL, DTL, TEL, and JRL serve over 130 stations. Train frequency is 2-5 minutes during peak hours. Fares are distance-based.",
            "singapore_transport",
            "lta.gov.sg",
        ),
        (
            "What is COE in Singapore?",
            "The Certificate of Entitlement (COE) allows a person to own and use a vehicle in Singapore for 10 years. COE prices are determined by a bidding system and reflect vehicle demand. Prices have exceeded SGD 100,000.",
            "singapore_transport",
            "lta.gov.sg",
        ),
        (
            "Singapore CPF overview.",
            "The Central Provident Fund (CPF) is Singapore's mandatory savings scheme. Employees and employers contribute monthly. CPF savings fund retirement (CPF Life), healthcare (Medisave), and housing. Contribution rates vary by age.",
            "singapore_finance",
            "cpf.gov.sg",
        ),
        (
            "What is GST in Singapore?",
            "The Goods and Services Tax (GST) is Singapore's value-added tax. It applies to most goods and services. GST rose to 9% in 2024. Businesses with annual turnover above SGD 1 million must register for GST.",
            "singapore_finance",
            "iras.gov.sg",
        ),
        (
            "Singapore education system overview.",
            "Singapore's education follows a 6-4-2 structure: 6 years primary, 4 years secondary, 2 years junior college. The PSLE at Primary 6 streams students. Strong emphasis on STEM and bilingualism. NUS, NTU, and SMU are top universities.",
            "singapore_education",
            "moe.gov.sg",
        ),
        (
            "Singapore hawker culture.",
            "Hawker centres are open-air cooked food centres in Singapore. They offer affordable multicultural cuisine: chicken rice, laksa, char kway teow, roti prata. UNESCO added hawker culture to its Intangible Cultural Heritage list in 2020.",
            "singapore_culture",
            "nea.gov.sg",
        ),
        (
            "Singapore's four official languages.",
            "Singapore has four official languages: English, Mandarin, Malay, and Tamil. English is the language of administration and business. Singlish, a creole variety, is widely spoken informally.",
            "singapore_culture",
            "singapore_gov",
        ),
        (
            "What are Singapore's public holidays?",
            "Singapore observes 11 public holidays: New Year's Day, Chinese New Year (2 days), Good Friday, Labour Day, Vesak Day, Hari Raya Puasa, National Day, Hari Raya Haji, Deepavali, and Christmas Day.",
            "singapore_culture",
            "mom.gov.sg",
        ),
        (
            "Singapore port and trade.",
            "The Port of Singapore is one of the world's busiest container ports. Singapore is a key transhipment hub. Trade accounts for over 300% of GDP. Major trading partners include China, Malaysia, the US, and the EU.",
            "singapore_economy",
            "mti.gov.sg",
        ),
        # --- Technical Reference ---
        (
            "Polars vs pandas overview.",
            "Polars is a fast DataFrame library written in Rust. It uses Apache Arrow for columnar memory. Lazy evaluation enables query optimisation. It is 5-10x faster than pandas on many workloads and uses less memory.",
            "tools",
            "polars_docs",
        ),
        (
            "What is Apache Parquet?",
            "Parquet is a columnar storage format optimised for analytics. It supports efficient compression and encoding. Columns are stored together, enabling predicate pushdown and projection pruning. Widely used in data engineering.",
            "tools",
            "apache_docs",
        ),
        (
            "What is a data pipeline?",
            "A data pipeline automates the movement and transformation of data from source to destination. Stages include ingestion, validation, cleaning, feature engineering, and serving. Orchestration tools manage scheduling and dependencies.",
            "data_engineering",
            "textbook",
        ),
        (
            "What is data drift?",
            "Data drift occurs when the statistical properties of input data change after model deployment. Feature drift changes the distribution of inputs. Concept drift changes the input-output relationship. Monitoring metrics include PSI and KS statistic.",
            "mlops",
            "textbook",
        ),
        (
            "What is MLOps?",
            "MLOps applies DevOps principles to machine learning. It covers versioned data, reproducible training, automated deployment, and continuous monitoring. Key tools include experiment trackers, model registries, and feature stores.",
            "mlops",
            "textbook",
        ),
        (
            "What is a feature store?",
            "A feature store is a centralised repository for ML features. It ensures consistency between training and serving by storing feature pipelines and their outputs. It reduces duplication and accelerates model development.",
            "mlops",
            "textbook",
        ),
        (
            "What is model drift monitoring?",
            "Model drift monitoring tracks model performance and input data distributions in production. Alerts fire when metrics degrade beyond thresholds. Population Stability Index (PSI) measures input distribution shift.",
            "mlops",
            "textbook",
        ),
        (
            "What is hyperparameter tuning?",
            "Hyperparameter tuning searches for the configuration that maximises model performance. Methods include grid search, random search, and Bayesian optimisation. Cross-validation estimates generalisation during search.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is ensemble learning?",
            "Ensemble learning combines multiple models to improve prediction. Bagging reduces variance (Random Forest). Boosting reduces bias (XGBoost, LightGBM). Stacking uses a meta-learner to blend predictions.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is SHAP?",
            "SHAP (SHapley Additive exPlanations) explains individual model predictions by attributing a contribution to each feature. It satisfies game-theoretic fairness axioms. Used for model interpretability and bias auditing.",
            "ml_interpretability",
            "textbook",
        ),
    ]

    # Pad to ~200 rows by repeating with slight variation
    base = articles.copy()
    while len(base) < 200:
        t, c, cat, src = random.choice(articles)
        base.append((t + " (variant)", c, cat, src))

    base = base[:200]

    doc_ids = [f"DOC-{500 + i:04d}" for i in range(len(base))]
    titles = [a[0] for a in base]
    contents = [a[1] for a in base]
    cats = [a[2] for a in base]
    sources = [a[3] for a in base]

    df = pl.DataFrame(
        {
            "doc_id": doc_ids,
            "title": titles,
            "content": contents,
            "category": cats,
            "source": sources,
        }
    )

    return df


# ---------------------------------------------------------------------------
# 7. sg_domain_qa.parquet  (ascent06)
# ---------------------------------------------------------------------------


def make_sg_domain_qa() -> pl.DataFrame:
    """
    Singapore domain Q&A for SFT (M6.1) — ~500 rows.
    Instruction-following format.
    """
    pairs = [
        # --- Geography ---
        (
            "What are the five main regions of Singapore?",
            "Singapore is divided into five main planning regions: the Central Region, North Region, North-East Region, East Region, and West Region. Each region contains several planning areas.",
            "",
            "geography",
        ),
        (
            "How large is Singapore by land area?",
            "Singapore has a total land area of approximately 735 square kilometres, making it one of the world's smallest countries. Land reclamation has expanded its area over the decades.",
            "",
            "geography",
        ),
        (
            "What is Sentosa Island known for?",
            "Sentosa is a resort island off Singapore's southern coast. It hosts Universal Studios Singapore, Resorts World Sentosa, beaches, and the cable car. It is a major tourism and entertainment destination.",
            "",
            "geography",
        ),
        (
            "Name the main river in Singapore.",
            "The Singapore River is the main river and runs through the heart of the city. It was historically the centre of trade and commerce. Today its banks are lined with restaurants and bars.",
            "",
            "geography",
        ),
        (
            "What is the highest point in Singapore?",
            "Bukit Timah Hill, at 163.63 metres, is the highest natural point in Singapore. It is located within the Bukit Timah Nature Reserve, which contains one of the world's few patches of primary rainforest within a city.",
            "",
            "geography",
        ),
        # --- Food ---
        (
            "What is chicken rice?",
            "Hainanese chicken rice is Singapore's national dish. It consists of poached or roasted chicken served over fragrant rice cooked in chicken broth, accompanied by chilli sauce, ginger paste, and dark soy sauce.",
            "",
            "food",
        ),
        (
            "What is laksa?",
            "Laksa is a spicy noodle soup popular in Singapore. Singapore laksa uses thick bee hoon in a rich coconut curry broth topped with prawns, fish cake, and cockles. Katong laksa is a famous local variant.",
            "",
            "food",
        ),
        (
            "What is a hawker centre?",
            "A hawker centre is a large open-air complex housing multiple food stalls. They offer affordable local food including Chinese, Malay, and Indian cuisine. UNESCO recognised Singapore hawker culture in 2020.",
            "",
            "food",
        ),
        (
            "What is char kway teow?",
            "Char kway teow is a stir-fried flat noodle dish cooked over high heat in a wok. It contains rice noodles, Chinese sausage, cockles, eggs, bean sprouts, and dark soy sauce. The wok hei (breath of the wok) is essential.",
            "",
            "food",
        ),
        (
            "What is bak kut teh?",
            "Bak kut teh is a pork rib soup simmered with herbs and spices. The Singaporean version is peppery and light compared to the darker Malaysian version. It is typically eaten for breakfast with youtiao (fried dough).",
            "",
            "food",
        ),
        (
            "What is roti prata?",
            "Roti prata is a flaky flatbread of South Indian origin cooked on a griddle. It is served with curry and comes in plain or stuffed variants (egg, cheese, banana). A popular breakfast and supper food.",
            "",
            "food",
        ),
        (
            "What is chilli crab?",
            "Chilli crab is one of Singapore's signature seafood dishes. Mud crabs are stir-fried in a semi-thick gravy made from tomato and chilli sauce. It is tangy, sweet, and savoury. Best enjoyed with mantou (steamed buns).",
            "",
            "food",
        ),
        # --- Transport ---
        (
            "How does the Singapore MRT fare system work?",
            "MRT fares in Singapore are distance-based and calculated by the distance travelled. Fares are paid using the EZ-Link card or Singapore Tourist Pass. Contactless payment is also accepted. Concession cards offer discounts for students and seniors.",
            "",
            "transport",
        ),
        (
            "What is the LRT system in Singapore?",
            "The Light Rapid Transit (LRT) system provides feeder services in Bukit Panjang, Sengkang, and Punggol. It connects HDB estates to MRT stations. The trains are automated and driverless.",
            "",
            "transport",
        ),
        (
            "What is the EZ-Link card?",
            "The EZ-Link card is a contactless smart card used for public transport in Singapore. It works on MRT, LRT, and buses. It can also be used at selected retail outlets. Cards can be topped up at ticketing machines or ATMs.",
            "",
            "transport",
        ),
        (
            "How do you book a taxi in Singapore?",
            "Taxis can be hailed at taxi stands or booked via ride-hailing apps such as Grab, ComfortDelGro, or TADA. Street hailing is also possible outside CBD and during non-peak hours. Ride-sharing services like GrabShare offer lower fares.",
            "",
            "transport",
        ),
        (
            "What is the TEL MRT line?",
            "The Thomson-East Coast Line (TEL) is Singapore's sixth MRT line. It connects Woodlands North in the north to Sungei Bedok in the east. It passes through major hubs including Orchard, Marina Bay, and Gardens by the Bay.",
            "",
            "transport",
        ),
        (
            "What is the SBS Transit bus network?",
            "SBS Transit operates one of Singapore's two major bus networks along with SMRT. It runs hundreds of routes across the island. The Singapore Bus Service (SBS) network has been in operation since the 1970s.",
            "",
            "transport",
        ),
        # --- Housing ---
        (
            "What are HDB BTO flats?",
            "BTO stands for Build-To-Order. It is the main scheme for Singaporeans to buy new HDB flats directly from HDB. Applicants ballot for flats and wait 3-5 years for construction. BTO flats are sold at subsidised prices with eligibility criteria.",
            "",
            "housing",
        ),
        (
            "What is the HDB ethnic integration policy?",
            "The Ethnic Integration Policy (EIP) sets quotas for the proportion of Chinese, Malay, and Indian residents in each HDB block and neighbourhood. It ensures racial integration and prevents ethnic enclaves.",
            "",
            "housing",
        ),
        (
            "What is the minimum occupancy period for HDB flats?",
            "HDB flat owners must fulfil a Minimum Occupation Period (MOP) of 5 years before they can sell the flat on the resale market or buy private property. The MOP starts from the date of key collection.",
            "",
            "housing",
        ),
        (
            "What is the CPF Housing Grant?",
            "The CPF Housing Grant provides financial assistance for Singaporeans buying HDB flats. The Enhanced CPF Housing Grant (EHG) offers up to SGD 80,000 for eligible first-time buyers. Grants are credited to the CPF Ordinary Account.",
            "",
            "housing",
        ),
        (
            "What is a HDB en bloc?",
            "An HDB estate renewal programme involves the government redeveloping ageing HDB estates. Residents are relocated with compensation. The Selective En Bloc Redevelopment Scheme (SERS) is the formal programme for this.",
            "",
            "housing",
        ),
        # --- Economy & Finance ---
        (
            "What is the Singapore Exchange (SGX)?",
            "The Singapore Exchange (SGX) is Singapore's stock exchange. It lists equities, derivatives, and REITs. Major indices include the Straits Times Index (STI). SGX is a key financial hub for Southeast Asia.",
            "",
            "finance",
        ),
        (
            "What is a REIT in Singapore?",
            "A Real Estate Investment Trust (REIT) pools investor funds to invest in income-generating real estate. Singapore has one of Asia's largest REIT markets. REITs are listed on SGX and required to distribute at least 90% of income as dividends.",
            "",
            "finance",
        ),
        (
            "What is the Singapore dollar exchange rate policy?",
            "The Monetary Authority of Singapore (MAS) manages monetary policy through the exchange rate rather than interest rates. MAS manages the SGD nominal effective exchange rate within a policy band. This approach suits Singapore's open economy.",
            "",
            "finance",
        ),
        (
            "What is the SkillsFuture credit?",
            "SkillsFuture credit is a government initiative that provides Singaporeans aged 25 and above with SGD 500 in credits for approved training courses. It is topped up periodically. The aim is lifelong learning and skills upgrading.",
            "",
            "education",
        ),
        (
            "What is the PSLE in Singapore?",
            "The Primary School Leaving Examination (PSLE) is taken by Primary 6 students. It determines secondary school placement. Students are scored using Achievement Level (AL) grades. The PSLE is a major milestone in Singapore's education system.",
            "",
            "education",
        ),
        # --- ML / Tech ---
        (
            "What is Singapore's Smart Nation initiative?",
            "Smart Nation is Singapore's whole-of-government drive to harness technology for a better quality of life. Key pillars include digital identity (Singpass), e-government, smart urban mobility, and a digital economy.",
            "",
            "technology",
        ),
        (
            "What is PDPA in Singapore?",
            "The Personal Data Protection Act (PDPA) governs the collection, use, and disclosure of personal data in Singapore. Organisations must obtain consent, protect data, and allow individuals to access or correct their data. The PDPC enforces the PDPA.",
            "",
            "technology",
        ),
        (
            "What is GovTech Singapore?",
            "GovTech is Singapore's government technology agency. It builds digital infrastructure including Singpass, CorpPass, and government APIs. GovTech leads data analytics and AI adoption across government agencies.",
            "",
            "technology",
        ),
        (
            "What is the Singapore AI Strategy?",
            "Singapore's National AI Strategy (NAIS) aims to deploy AI across key sectors: health, education, transport, safety, and the public sector. It invests in AI talent, research, and governance frameworks including the Model AI Governance Framework.",
            "",
            "technology",
        ),
        (
            "What is data.gov.sg?",
            "data.gov.sg is Singapore's open data portal maintained by GovTech. It provides free access to government datasets covering housing, transport, environment, economy, and demographics. Datasets are available in CSV, GeoJSON, and API formats.",
            "",
            "technology",
        ),
    ]

    # Expand to ~500 rows by paraphrasing templates
    extended = pairs.copy()
    paraphrase_prefix = [
        "Can you explain: ",
        "Please describe: ",
        "Give me information about: ",
        "Tell me about: ",
        "I want to learn about: ",
        "What should I know about: ",
    ]
    while len(extended) < 500:
        inst, resp, ctx, cat = random.choice(pairs)
        prefix = random.choice(paraphrase_prefix)
        extended.append((prefix + inst.lower(), resp, ctx, cat))

    extended = extended[:500]

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
# 8. preference_pairs.parquet  (ascent06)
# ---------------------------------------------------------------------------


def make_preference_pairs() -> pl.DataFrame:
    """
    Preference pairs for DPO (M6.2) — ~300 rows.
    Each row has a prompt, a chosen (good) response, and a rejected (bad) one.
    """
    pairs = [
        # (prompt, chosen, rejected, category)
        (
            "Explain overfitting in machine learning.",
            "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalise to new examples. The model has high variance. Regularisation, early stopping, and more training data mitigate it.",
            "Overfitting is when the model is overfitted. It is bad and you should avoid it by using regularisation.",
            "ml_concepts",
        ),
        (
            "What is the difference between classification and regression?",
            "Classification predicts discrete class labels (e.g., spam/not spam). Regression predicts continuous numeric values (e.g., house prices). Both are supervised learning tasks but differ in output type and loss functions used.",
            "Classification is for categories and regression is for numbers. They are both supervised. Classification uses accuracy.",
            "ml_concepts",
        ),
        (
            "How does a random forest work?",
            "A random forest trains multiple decision trees on random subsets of the data and features (bagging). Each tree votes and the majority class (or mean for regression) is the final prediction. It reduces variance compared to a single tree.",
            "Random forest has many trees and they vote. It is better than one tree because more trees are better.",
            "ml_algorithms",
        ),
        (
            "What is gradient boosting?",
            "Gradient boosting builds an ensemble sequentially. Each new tree corrects the residual errors of the previous ensemble by fitting the gradient of the loss function. XGBoost and LightGBM are efficient implementations.",
            "Gradient boosting uses gradients and boosts things. It is used in competitions and is very accurate.",
            "ml_algorithms",
        ),
        (
            "Explain the bias-variance trade-off.",
            "Bias measures error from incorrect model assumptions (underfitting). Variance measures sensitivity to training data fluctuations (overfitting). As model complexity increases, bias falls and variance rises. Optimal complexity minimises total error.",
            "Bias and variance are both bad. You want low bias and low variance. Try different models.",
            "ml_theory",
        ),
        (
            "What is precision and when should you prioritise it?",
            "Precision is the fraction of predicted positives that are correct: TP / (TP + FP). Prioritise precision when false positives are costly, such as in spam filtering (you do not want to block legitimate emails) or medical screening.",
            "Precision is TP divided by TP plus FP. It is good when you don't want false positives. It is part of the classification report.",
            "ml_evaluation",
        ),
        (
            "How do you handle class imbalance?",
            "Strategies include oversampling the minority class (SMOTE), undersampling the majority class, adjusting class weights in the loss function, and using evaluation metrics robust to imbalance such as PR-AUC or balanced accuracy.",
            "Use SMOTE or oversample. You can also try different models. Sometimes class imbalance is not a big problem.",
            "ml_practical",
        ),
        (
            "What is SHAP and how is it used?",
            "SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a specific prediction using Shapley values from game theory. It is model-agnostic and provides both local (per-prediction) and global (overall) explanations.",
            "SHAP explains the model. It shows which features are important. It uses Shapley values from game theory.",
            "ml_interpretability",
        ),
        (
            "What is the purpose of a validation set?",
            "A validation set is a held-out split used during training to tune hyperparameters and monitor generalisation without contaminating the test set. It allows early stopping and model selection. The test set is used only once for final evaluation.",
            "Validation set is used to validate the model. It is different from the test set. You use it during training.",
            "ml_practical",
        ),
        (
            "How does k-means clustering decide the number of clusters?",
            "K-means requires k to be specified in advance. Common methods to choose k include the elbow method (plotting inertia vs k), silhouette score (measures cluster cohesion vs separation), and domain knowledge about expected groupings.",
            "You try different values of k and pick the best one. The elbow method shows where adding clusters stops helping much.",
            "ml_algorithms",
        ),
        (
            "What is transfer learning and when should you use it?",
            "Transfer learning uses a model pre-trained on a large dataset as a starting point for a new task. It is useful when labelled data is scarce or computation is limited. Fine-tuning adapts the pre-trained weights to the new domain.",
            "Transfer learning transfers knowledge from one model to another. Use it when you don't have enough data. Fine-tuning is the main technique.",
            "deep_learning",
        ),
        (
            "What is the attention mechanism in transformers?",
            "Attention allows each position in a sequence to attend to all other positions. For each query, it computes a weighted sum of values, where weights are determined by query-key dot products scaled and normalised with softmax. Multi-head attention runs this in parallel.",
            "Attention looks at other tokens in the sequence. It uses queries, keys, and values. Multi-head attention does this multiple times.",
            "deep_learning",
        ),
        (
            "How do you evaluate a language model?",
            "LLMs are evaluated with perplexity (measures prediction confidence on held-out text), task-specific benchmarks (MMLU, HellaSwag), and human preference evaluations. For instruction-tuned models, MT-Bench and Chatbot Arena assess conversational quality.",
            "You evaluate LLMs with perplexity and benchmarks like MMLU. Human evaluation is also used. RLHF helps improve the model.",
            "llm_techniques",
        ),
        (
            "What is RAG and when should you use it?",
            "Retrieval-Augmented Generation retrieves relevant documents and conditions the LLM response on them. Use RAG when: (1) knowledge must be up-to-date, (2) you need verifiable sources, (3) fine-tuning is too expensive, or (4) domain-specific facts are needed.",
            "RAG retrieves documents and passes them to the LLM. Use it when the LLM doesn't know the answer or when you need up-to-date information.",
            "llm_techniques",
        ),
        (
            "What is LoRA and why is it useful?",
            "LoRA (Low-Rank Adaptation) fine-tunes LLMs by adding pairs of low-rank matrices to transformer layers. Only these matrices are trained, reducing trainable parameters by 10-100x. This lowers GPU memory requirements and training time while matching full fine-tuning quality.",
            "LoRA is a way to fine-tune LLMs efficiently. It uses low-rank matrices so you train fewer parameters. It saves GPU memory.",
            "llm_techniques",
        ),
        (
            "How does RLHF work?",
            "RLHF (Reinforcement Learning from Human Feedback) first trains a reward model on human preference comparisons between model outputs. It then fine-tunes the LLM with PPO to maximise reward. A KL penalty prevents the model from deviating too far from the base model.",
            "RLHF uses human feedback to train the model. A reward model is trained first, then the LLM is fine-tuned with reinforcement learning.",
            "llm_alignment",
        ),
        (
            "What is DPO compared to PPO for alignment?",
            "DPO (Direct Preference Optimisation) reformulates RLHF as a supervised learning problem on preference pairs, avoiding the need to train a separate reward model or run online RL. It is simpler, more stable, and often matches PPO quality.",
            "DPO is simpler than PPO. It doesn't need a reward model. DPO directly optimises on preference data.",
            "llm_alignment",
        ),
        (
            "What is data drift and how do you detect it?",
            "Data drift is a change in the statistical distribution of input data after model deployment. Detection methods include Population Stability Index (PSI), Kolmogorov-Smirnov test, and monitoring feature summary statistics. A PSI > 0.2 typically triggers retraining.",
            "Data drift is when the data changes over time. You can detect it with PSI or statistical tests. It means you should retrain your model.",
            "mlops",
        ),
        (
            "What is a feature store and why is it important?",
            "A feature store is a centralised repository for ML features. It decouples feature engineering from model training, ensures training-serving consistency, enables feature reuse across teams, and stores feature metadata and statistics.",
            "A feature store stores features. It is useful for sharing features across teams and making sure training and serving use the same features.",
            "mlops",
        ),
        (
            "How do you monitor a model in production?",
            "Monitor input data distributions (PSI, KS test), model predictions (output drift), business metrics (revenue, conversions), and ground-truth feedback when available. Set up alerts for significant deviations. Schedule periodic retraining when performance degrades.",
            "Monitor the model predictions and check if they are still accurate. Use dashboards and alerts. Retrain when needed.",
            "mlops",
        ),
    ]

    # Expand to ~300 rows
    extended = pairs.copy()
    while len(extended) < 300:
        p, c, r, cat = random.choice(pairs)
        extended.append((p, c, r, cat))

    extended = extended[:300]

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
# 9. mrt_stations.parquet  (ascent_assessment)
# ---------------------------------------------------------------------------


def make_mrt_stations() -> pl.DataFrame:
    """
    Singapore MRT stations with town mapping and coordinates (~200 rows).
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
        ("City Hall", "Downtown", "EWL", 1.2931, 103.8520),  # interchange
        ("Tanjong Pagar", "Downtown", "EWL", 1.2762, 103.8454),
        ("Outram Park", "Outram", "EWL", 1.2801, 103.8396),
        ("Tiong Bahru", "Bukit Merah", "EWL", 1.2863, 103.8272),
        ("Redhill", "Bukit Merah", "EWL", 1.2895, 103.8164),
        ("Queenstown", "Queenstown", "EWL", 1.2943, 103.8059),
        ("Commonwealth", "Queenstown", "EWL", 1.3022, 103.7982),
        ("Buona Vista", "Queenstown", "EWL", 1.3067, 103.7901),
        ("Dover", "Clementi", "EWL", 1.3115, 103.7787),
        ("Clementi", "Clementi", "EWL", 1.3151, 103.7654),
        ("Jurong East", "Jurong East", "EWL", 1.3331, 103.7422),  # interchange
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

    # De-duplicate by station name (keep first occurrence for each name+line)
    seen = set()
    unique_stations = []
    for s in stations:
        key = (s[0], s[2])  # name + line
        if key not in seen:
            seen.add(key)
            unique_stations.append(s)

    unique_stations = unique_stations[:200]
    m = len(unique_stations)

    names = [s[0] for s in unique_stations]
    towns = [s[1] for s in unique_stations]
    lines = [s[2] for s in unique_stations]
    lats = [s[3] for s in unique_stations]
    lons = [s[4] for s in unique_stations]

    # Nearest MRT (other station on same line that is closest)
    nearest_mrt = []
    dist_to_nearest = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        best_name = names[i]
        best_dist = float("inf")
        same_line = lines[i]
        for j, (la2, lo2) in enumerate(zip(lats, lons)):
            if j == i:
                continue
            if lines[j] != same_line:
                continue
            # Approximate haversine (small angle)
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
# 10. schools.parquet  (ascent_assessment)
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

    # Determine planning zone from town
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
    # Create directories
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  directory ready: {d}")

    tasks = [
        # (generator_fn, output_path, write_mode)
        (
            make_economic_indicators,
            DATA_ROOT / "ascent01" / "economic_indicators.csv",
            "csv",
        ),
        (make_sg_taxi_trips, DATA_ROOT / "ascent01" / "sg_taxi_trips.csv", "csv"),
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
            df.write_parquet(str(path))
        size_kb = path.stat().st_size // 1024
        print(
            f"  {path.name:<45} {len(df):>6} rows  {size_kb:>5} KB  ({path.parent.name})"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
