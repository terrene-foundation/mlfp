#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP03 Assessment Task 1 — Feature Engineering & Selection.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader re-derives the target, the engineered features, and a leakage-free
importance ranking independently, then checks the submission against strict
invariants. All checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
TRAIN_FRACTION = 0.75
TOP_K = 8
BASE_FEATURES = [
    "total_revenue",
    "order_count",
    "avg_order_value",
    "days_since_last_order",
    "customer_tenure_days",
    "satisfaction_score",
    "num_returns",
    "loyalty_int",
]
ENGINEERED_FEATURES = [
    "revenue_per_order",
    "returns_per_order",
    "is_satisfied",
    "loyal_and_satisfied",
    "tenure_years",
    "spend_per_tenure_day",
]
FORBIDDEN = {"customer_id", "review_text", "churned", "ltv_tier", "product_categories"}


def _reference() -> pl.DataFrame:
    """Independent reference: target + 14-column candidate matrix."""
    df = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
    df = df.sort("customer_id").head(N_ROWS)
    rng = np.random.default_rng(SEED)

    def z(col: str) -> np.ndarray:
        a = df[col].to_numpy().astype(float)
        return (a - a.mean()) / (a.std() + 1e-9)

    loyal = df["loyalty_member"].cast(pl.Int64).to_numpy().astype(float)
    sat_high = (df["satisfaction_score"] >= 4).cast(pl.Int64).to_numpy().astype(float)
    logit = (
        1.0 * z("satisfaction_score")
        + 0.9 * loyal
        + 0.8 * z("avg_order_value")
        - 0.7 * z("num_returns")
        + 0.5 * z("order_count")
        + 1.4 * (loyal * sat_high)
        + rng.normal(0.0, 1.3, size=df.height)
    )
    df = df.with_columns(
        [
            pl.col("loyalty_member").cast(pl.Int64).alias("loyalty_int"),
            pl.Series(TARGET, (logit > 2.0).astype(np.int64)),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("total_revenue") / pl.col("order_count")).alias("revenue_per_order"),
            (pl.col("num_returns") / pl.col("order_count")).alias("returns_per_order"),
            (pl.col("satisfaction_score") >= 4).cast(pl.Int64).alias("is_satisfied"),
            (
                pl.col("loyalty_int")
                * (pl.col("satisfaction_score") >= 4).cast(pl.Int64)
            ).alias("loyal_and_satisfied"),
            (pl.col("customer_tenure_days") / 365.0).alias("tenure_years"),
            (pl.col("total_revenue") / pl.col("customer_tenure_days")).alias(
                "spend_per_tenure_day"
            ),
        ]
    )
    return df.select(BASE_FEATURES + ENGINEERED_FEATURES + [TARGET])


def _reference_importance_top_k(ref: pl.DataFrame, k: int) -> list[str]:
    """RandomForest importance ranking on the TRAIN split (leakage-free)."""
    n_train = int(TRAIN_FRACTION * ref.height)
    train = ref.head(n_train)
    cand = BASE_FEATURES + ENGINEERED_FEATURES
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(train.select(cand).to_numpy(), train[TARGET].to_numpy())
    order = sorted(zip(cand, rf.feature_importances_), key=lambda t: -t[1])
    return [c for c, _ in order[:k]]


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a: pl.Series, b: pl.Series, tol: float = 1e-6) -> bool:
    try:
        return bool((a.cast(pl.Float64) - b.cast(pl.Float64)).abs().max() < tol)
    except Exception:
        return False


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "total": 0, "max": 0}
    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score
    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve() function"
        return score
    try:
        r = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    c = score["checks"]
    ref = _reference()

    # 1. shape of the return value
    c["returns_dict"] = isinstance(r, dict) and {
        "feature_matrix",
        "engineered_columns",
        "selected_features",
        "target_column",
    }.issubset(r.keys())
    if not c["returns_dict"]:
        return _finalize(score)

    fm = r.get("feature_matrix")
    eng = r.get("engineered_columns")
    sel = r.get("selected_features")

    c["types_valid"] = (
        isinstance(fm, pl.DataFrame)
        and isinstance(eng, list)
        and isinstance(sel, list)
        and r.get("target_column") == TARGET
    )
    if not c["types_valid"]:
        return _finalize(score)

    # 2. no id / text / native-label leakage in the candidate matrix
    c["no_leakage_columns"] = (
        len(FORBIDDEN.intersection(set(fm.columns))) == 0 and TARGET in fm.columns
    )

    # 3. exact engineered feature names
    c["engineered_names_correct"] = set(eng) == set(ENGINEERED_FEATURES)

    # 4. candidate matrix carries all 14 features + target
    expected_cols = set(BASE_FEATURES + ENGINEERED_FEATURES + [TARGET])
    c["candidate_columns_present"] = expected_cols.issubset(set(fm.columns))

    # 5. derived target re-derivation (element-wise, exact) — strong anti-stub
    if TARGET in fm.columns and fm.height == ref.height:
        c["target_correct"] = _close(fm[TARGET], ref[TARGET], tol=1e-9)
    else:
        c["target_correct"] = False

    # 6. engineered features computed correctly (element-wise) — interaction + ratio
    eng_ok = True
    for col in ("loyal_and_satisfied", "returns_per_order"):
        if col in fm.columns and fm.height == ref.height:
            eng_ok = eng_ok and _close(fm[col], ref[col], tol=1e-6)
        else:
            eng_ok = False
    c["engineered_interaction_ratio_correct"] = eng_ok

    # 7. engineered features computed correctly (element-wise) — spend + revenue
    eng_ok2 = True
    for col in ("revenue_per_order", "spend_per_tenure_day"):
        if col in fm.columns and fm.height == ref.height:
            eng_ok2 = eng_ok2 and _close(fm[col], ref[col], tol=1e-4)
        else:
            eng_ok2 = False
    c["engineered_spend_revenue_correct"] = eng_ok2

    # 8. selection shape: exactly TOP_K, all valid candidates, target excluded
    cand_set = set(BASE_FEATURES + ENGINEERED_FEATURES)
    c["selection_shape_valid"] = (
        len(sel) == TOP_K
        and len(set(sel)) == TOP_K
        and set(sel).issubset(cand_set)
        and TARGET not in sel
    )

    # 9. selection surfaced the true strongest driver (interaction or satisfaction)
    c["selection_top_driver"] = (
        len(sel) > 0
        and sel[0] in {"loyal_and_satisfied", "satisfaction_score"}
    )

    # 10. leakage-free signal: overlap with independently re-derived importance top-K
    try:
        ref_top = set(_reference_importance_top_k(ref, TOP_K))
        c["selection_overlaps_truth"] = len(ref_top.intersection(set(sel))) >= 6
    except Exception:
        c["selection_overlaps_truth"] = False

    # 11. selected features carry real signal: held-out AUC above an honest floor
    try:
        n_train = int(TRAIN_FRACTION * ref.height)
        train, test = ref.head(n_train), ref.tail(ref.height - n_train)
        rf = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)
        rf.fit(train.select(sel).to_numpy(), train[TARGET].to_numpy())
        auc = roc_auc_score(
            test[TARGET].to_numpy(), rf.predict_proba(test.select(sel).to_numpy())[:, 1]
        )
        c["selected_features_predict"] = bool(auc >= 0.84)
    except Exception:
        c["selected_features_predict"] = False

    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["max"] > 0 and score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
