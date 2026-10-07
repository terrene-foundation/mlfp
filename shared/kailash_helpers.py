# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Common Kailash SDK setup patterns for MLFP exercises."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """Load .env and validate common configuration.

    Call this at the top of every exercise that needs API keys or DB connections.
    """
    # Find .env by walking up from the exercise file
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try repo root
        for parent in Path.cwd().parents:
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break

    load_dotenv(env_path)


def get_connection_manager(db_url: str | None = None):
    """Create a ConnectionManager for kailash-ml engines.

    Args:
        db_url: Database URL. Defaults to SQLite at ./mlfp.db
    """
    from kailash.db import ConnectionManager

    url = db_url or os.environ.get("DATABASE_URL", "sqlite:///mlfp.db")
    return ConnectionManager(url)


def get_device() -> "torch.device":
    """Return the best available compute device as a ``torch.device``.

    Routes through ``kailash_ml.device()`` (the canonical kailash-ml 0.12+
    accelerator detector) so MPS / CUDA / ROCm / Intel XPU / CPU are all
    selected by the same logic the rest of the platform uses. The previous
    hand-rolled ``mps→cuda→cpu`` cascade is replaced because:

      * kailash-ml's detector knows about ROCm, Intel XPU, and fp16/bf16
        capability flags — the cascade in this helper did not.
      * Apple-Silicon students get the Metal Performance Shaders backend
        with mixed-precision (fp16) without any opt-in.
      * One detection point means lessons that print "Using device: …"
        agree with what kailash-ml's MLEngine() actually picks.
    """
    import kailash_ml as km
    import torch

    backend = km.device()  # BackendInfo (auto MPS on Mac, CUDA on Linux+NVIDIA, …)
    return torch.device(backend.device_string)


def get_llm_model() -> str:
    """Get the configured LLM model name from environment."""
    setup_environment()
    model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    if not model:
        raise EnvironmentError(
            "No LLM model configured. Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env"
        )
    return model


# ════════════════════════════════════════════════════════════════════════
# SPLIT FIRST, THEN PREPROCESS
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's ``PreprocessingPipeline.setup()`` fits its imputer, its
# categorical encoder (including TARGET encoding) and its scaler on EVERY
# row it is given, and only afterwards splits train/test. Handing it the
# whole dataset therefore lets test-set statistics leak into training.
#
# The helpers below hold the test rows out FIRST, fit the pipeline on the
# training rows only, and push the held-out rows through the fitted
# ``transform()``. ``setup()`` still splits internally, but that split now
# happens inside the training rows, so it cannot leak anything.

_SETUP_OPTIONS: frozenset[str] = frozenset(
    {
        "normalize",
        "normalize_method",
        "categorical_encoding",
        "imputation_strategy",
        "impute_n_neighbors",
        "remove_outliers",
        "outlier_threshold",
        "max_cardinality",
        "exclude_columns",
        "remove_multicollinearity",
        "multicollinearity_threshold",
        "pca",
        "pca_components",
        "transform_target",
        "fix_imbalance",
        "imbalance_method",
    }
)

# Arguments the helpers own themselves; passing them through would
# re-introduce the leak (``data`` = the full frame) or fight the split.
_OWNED_BY_HELPER: frozenset[str] = frozenset({"data", "target", "train_size", "seed"})


@dataclass(frozen=True)
class PreprocessResult:
    """Outcome of :func:`split_then_preprocess` / :func:`preprocess_train_test`.

    Mirrors the fields of kailash-ml's ``SetupResult`` that exercises use,
    plus the fitted pipeline and a record of what was done to which split.

    ``train_data`` / ``test_data``
        Both produced by the SAME pipeline, fitted on the training rows only.
    ``transformers``
        The fitted imputer / encoder / scaler state (e.g.
        ``transformers["ordinal_mappings"]``), learned from training rows.
    ``train_only_steps``
        Steps applied to the training split only (outlier removal,
        class-imbalance resampling). The test split is never filtered or
        resampled — it stays a faithful sample of the data.
    ``test_null_fills``
        Columns that were complete in the training rows but have gaps in
        the test rows (the fitted pipeline has no learned fill for them),
        mapped to the TRAINING statistic used to fill them.
    """

    train_data: pl.DataFrame
    test_data: pl.DataFrame
    target_column: str
    task_type: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    transformers: dict[str, Any]
    pipeline: Any
    original_shape: tuple[int, int]
    train_raw_rows: int
    test_raw_rows: int
    stratified: bool
    train_only_steps: tuple[str, ...] = ()
    test_null_fills: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


def _looks_like_classification(series: pl.Series) -> bool:
    """Same rule kailash-ml uses to call a target "classification"."""
    if series.dtype in (pl.Boolean, pl.Categorical, pl.Utf8, pl.String):
        return True
    return series.drop_nulls().n_unique() <= 20


def split_raw_train_test(
    df: pl.DataFrame,
    target: str,
    *,
    test_size: float | int = 0.2,
    seed: int = 42,
    stratify: bool | Literal["auto"] = "auto",
) -> tuple[pl.DataFrame, pl.DataFrame, bool]:
    """Hold out test rows from the RAW frame before anything is fitted.

    Rows are shuffled with ``seed``. ``test_size`` is a fraction in (0, 1)
    or an exact row count. With ``stratify="auto"`` a classification target
    (same rule as kailash-ml) is stratified so every class keeps its share
    in both splits; ``True`` / ``False`` force the choice.

    Returns ``(train_raw, test_raw, stratified)``; both splits keep the
    shuffled order.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    n = df.height
    if n < 2:
        raise ValueError(f"Need at least 2 rows to split, got {n}.")
    if isinstance(test_size, np.integer):
        test_size = int(test_size)
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError(
            f"test_size must be a fraction in (0, 1) or a row count, got {test_size!r}."
        )
    if isinstance(test_size, int):
        if not 1 <= test_size < n:
            raise ValueError(f"test_size as a row count must be in [1, {n - 1}], got {test_size}.")
        test_fraction = test_size / n
    else:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size as a fraction must be in (0, 1), got {test_size}.")
        test_fraction = float(test_size)

    is_classification = _looks_like_classification(df[target])
    if stratify == "auto":
        stratified = is_classification
    elif stratify is True:
        if not is_classification:
            raise ValueError(
                f"stratify=True needs a classification target; '{target}' has "
                f"{df[target].drop_nulls().n_unique()} distinct values."
            )
        stratified = True
    elif stratify is False:
        stratified = False
    else:
        raise TypeError(f"stratify must be True, False or 'auto', got {stratify!r}.")

    perm = np.random.RandomState(seed).permutation(n)
    shuffled = df[perm.tolist()]

    if stratified:
        # Within each class (in shuffled order), the first round(n_c * f)
        # rows go to test — the class share is the same in both splits.
        is_test = pl.int_range(pl.len()).over(target) < (
            pl.len().over(target) * test_fraction
        ).round()
        flagged = shuffled.with_columns(is_test.alias("__split_is_test"))
        test_raw = flagged.filter(pl.col("__split_is_test")).drop("__split_is_test")
        train_raw = flagged.filter(~pl.col("__split_is_test")).drop("__split_is_test")
    else:
        n_test = test_size if isinstance(test_size, int) else int(round(n * test_fraction))
        train_raw = shuffled.head(n - n_test)
        test_raw = shuffled.tail(n_test)

    if train_raw.height == 0 or test_raw.height == 0:
        raise ValueError(
            f"Split produced an empty side (train={train_raw.height}, "
            f"test={test_raw.height}); adjust test_size."
        )
    return train_raw, test_raw, stratified


def _fill_unfitted_test_nulls(
    test_raw: pl.DataFrame,
    train_raw: pl.DataFrame,
    fitted: Any,
    strategy: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fill test gaps the fitted pipeline has no learned value for.

    kailash-ml only stores a fill value for columns that had gaps in the
    rows it was fitted on. A column that is complete in the training rows
    but has gaps in the test rows would otherwise reach the model as null.
    The fill is the TRAINING statistic for the same strategy (never a test
    statistic), and every such column is logged and returned.
    """
    if strategy == "drop":
        return test_raw, {}  # transform() drops incomplete rows itself
    learned = set(fitted.transformers.get("imputer_stats", {}))
    if strategy in ("knn", "iterative"):
        learned |= set(fitted.transformers.get("sklearn_imputer_cols", []))

    fills: dict[str, Any] = {}
    for col in fitted.numeric_columns + fitted.categorical_columns:
        if col in learned or col not in test_raw.columns:
            continue
        if test_raw[col].null_count() == 0:
            continue
        train_col = train_raw[col].drop_nulls()
        if col in fitted.categorical_columns or strategy == "mode":
            modes = train_col.mode().sort()
            if len(modes) == 0:
                raise ValueError(f"Cannot fill '{col}': no observed training values.")
            value = modes[0]
        elif strategy == "median":
            value = train_col.median()
        else:  # mean (and the numeric fallback for any other strategy)
            value = train_col.mean()
        if value is None:
            raise ValueError(f"Cannot fill '{col}': no observed training values.")
        fills[col] = value
    if fills:
        logger.warning(
            "Test rows have gaps in columns that were complete in training; "
            "filled with TRAINING statistics: %s",
            fills,
        )
        test_raw = test_raw.with_columns(
            [pl.col(c).fill_null(v) for c, v in fills.items()]
        )
    return test_raw, fills


def preprocess_train_test(
    train_raw: pl.DataFrame,
    test_raw: pl.DataFrame,
    target: str,
    *,
    seed: int = 42,
    **setup_kwargs: Any,
) -> PreprocessResult:
    """Fit ``PreprocessingPipeline`` on ``train_raw`` only; transform both.

    ``setup_kwargs`` are ``PreprocessingPipeline.setup()`` options
    (``normalize``, ``categorical_encoding``, ``imputation_strategy`` …).
    Steps that change the ROWS of a split are applied to training rows only:

    * ``remove_outliers`` — outlier rows are dropped from training only.
    * ``fix_imbalance`` — SMOTE / ADASYN (or the class-weight flag) is
      applied to the whole transformed training split, never to test.

    Use this directly inside cross-validation folds; use
    :func:`split_then_preprocess` for a single train/test split.
    """
    from kailash_ml import PreprocessingPipeline

    owned = sorted(_OWNED_BY_HELPER & set(setup_kwargs))
    if owned:
        raise TypeError(
            f"{owned} are set by the split-first helper itself and cannot be "
            "passed through to setup()."
        )
    unknown = sorted(set(setup_kwargs) - _SETUP_OPTIONS)
    if unknown:
        raise TypeError(f"Unknown PreprocessingPipeline.setup() options: {unknown}.")
    for name, frame in (("train_raw", train_raw), ("test_raw", test_raw)):
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' not found in {name}.")
    if train_raw.height == 0 or test_raw.height == 0:
        raise ValueError("train_raw and test_raw must both have rows.")
    if set(train_raw.columns) != set(test_raw.columns):
        raise ValueError("train_raw and test_raw must have the same columns.")

    fix_imbalance = bool(setup_kwargs.pop("fix_imbalance", False))
    imbalance_method = setup_kwargs.get("imbalance_method", "smote")
    remove_outliers = bool(setup_kwargs.get("remove_outliers", False))

    pipeline = PreprocessingPipeline()
    # The pipeline only ever sees training rows. Its internal split stays
    # inside them; resampling is applied below to the full training split.
    fitted = pipeline.setup(
        data=train_raw, target=target, seed=seed, fix_imbalance=False, **setup_kwargs
    )

    train_only: list[str] = []
    if remove_outliers:
        # setup() removed the outlier rows (training rows only); its two
        # internal pieces together are every surviving training row.
        train_data = pl.concat([fitted.train_data, fitted.test_data], how="vertical")
        train_only.append(
            f"remove_outliers: {train_raw.height - train_data.height} training rows dropped"
        )
    else:
        train_data = pipeline.transform(train_raw)

    strategy = setup_kwargs.get("imputation_strategy", "mean")
    test_filled, test_fills = _fill_unfitted_test_nulls(test_raw, train_raw, fitted, strategy)
    test_data = pipeline.transform(test_filled)

    if fix_imbalance and fitted.task_type == "classification":
        resample = getattr(pipeline, "_apply_imbalance_correction", None)
        if resample is None:
            raise RuntimeError(
                "This kailash-ml version has no imbalance-correction step on "
                "PreprocessingPipeline; cannot apply fix_imbalance to the training split."
            )
        before = train_data.height
        train_data = resample(train_data, target, imbalance_method, seed)
        train_only.append(
            f"fix_imbalance ({imbalance_method}): training rows {before} -> {train_data.height}"
        )

    summary = "\n".join(
        [
            f"Task type: {fitted.task_type}",
            f"Split first: {train_raw.height} train rows / {test_raw.height} test rows "
            "(pipeline fitted on the train rows only)",
            f"Numeric features: {len(fitted.numeric_columns)}",
            f"Categorical features: {len(fitted.categorical_columns)}",
            f"Encoding: {setup_kwargs.get('categorical_encoding', 'onehot')}",
            f"Normalization: {'yes' if setup_kwargs.get('normalize', True) else 'no'}",
            f"Imputation: {strategy}",
            f"Train-only steps: {'; '.join(train_only) if train_only else 'none'}",
            f"Transformed train: {train_data.height} rows x {train_data.width} cols",
            f"Transformed test: {test_data.height} rows x {test_data.width} cols",
        ]
    )

    return PreprocessResult(
        train_data=train_data,
        test_data=test_data,
        target_column=target,
        task_type=fitted.task_type,
        numeric_columns=list(fitted.numeric_columns),
        categorical_columns=list(fitted.categorical_columns),
        transformers=dict(fitted.transformers),
        pipeline=pipeline,
        original_shape=(train_raw.height + test_raw.height, train_raw.width),
        train_raw_rows=train_raw.height,
        test_raw_rows=test_raw.height,
        stratified=False,
        train_only_steps=tuple(train_only),
        test_null_fills=test_fills,
        summary=summary,
    )


def split_then_preprocess(
    df: pl.DataFrame,
    target: str,
    *,
    test_size: float | int = 0.2,
    seed: int = 42,
    stratify: bool | Literal["auto"] = "auto",
    **setup_kwargs: Any,
) -> PreprocessResult:
    """Leak-free replacement for ``PreprocessingPipeline().setup(df, ...)``.

    1. Split the RAW frame first (seeded shuffle; stratified on the target
       for classification when ``stratify="auto"``).
    2. Fit ``PreprocessingPipeline`` on the training rows only.
    3. ``transform()`` the test rows with the fitted pipeline.

    ``setup_kwargs`` are passed to ``PreprocessingPipeline.setup()``; see
    :func:`preprocess_train_test` for how row-changing options are handled.
    """
    train_raw, test_raw, stratified = split_raw_train_test(
        df, target, test_size=test_size, seed=seed, stratify=stratify
    )
    result = preprocess_train_test(train_raw, test_raw, target, seed=seed, **setup_kwargs)
    split_line = "stratified on the target" if stratified else "random"
    return replace(
        result,
        stratified=stratified,
        summary=result.summary.replace("Split first:", f"Split first ({split_line}):", 1),
    )


def create_visualizer():
    """Return a ModelVisualizer with the P2 experimental notice acknowledged.

    kailash-ml 2.2.2 emits ExperimentalWarning (a UserWarning) at
    ModelVisualizer construction; exercises run under warnings-as-errors,
    so the notice is acknowledged narrowly here — the ONE construction site
    every course file should use.
    """
    import warnings as _warnings

    from kailash_ml import ModelVisualizer
    from kailash_ml._decorators import ExperimentalWarning as _EW

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", _EW)
        return ModelVisualizer()

# ── SDK-internal deprecation nag filter (strict-gate hygiene) ────────────
# torch 2.12's ONNX exporter warns that dynamic_axes is "not recommended"
# under dynamo=True — a deprecation notice inside OnnxBridge's internals,
# not a correctness signal, and fatal under warnings-as-errors. Acknowledged
# here (kailash_helpers is imported by every shared.mlfpNN package, so every
# technique file gets it); message-matched, everything else still errors.
import warnings as _warnings

_warnings.filterwarnings(
    "ignore", message=r".*dynamic_axes.*not recommended.*", category=UserWarning
)
# matplotlib's Agg backend (forced in headless/script runs) nags
# "FigureCanvasAgg is non-interactive, and thus cannot be shown" on every
# plt.show() — a backend notice, not a correctness signal.
_warnings.filterwarnings(
    "ignore", message=r".*non-interactive.*cannot be shown.*", category=UserWarning
)

def hdb_storey_range_expr(column: str = "storey_range"):
    """Polars expression normalising HDB storey_range letter-O typos.

    The raw HDB resale file has letter-O typos ("O4 TO 06", "1O TO 12",
    "28 TO 3O"); read a letter O next to a digit as zero, leaving "TO"
    alone. Single home for the rule (P2): M2 ex_8 and M3 ex_5/07 both
    normalise at the use-site.
    """
    import polars as pl

    return (
        pl.col(column)
        .str.replace_all(r"\bO(\d)", "0${1}")
        .str.replace_all(r"(\d)O\b", "${1}0")
    )

