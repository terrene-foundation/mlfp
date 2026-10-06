# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 1 — Prompt Engineering.

Contains: SST-2 dataset loading, Ollama-backed Kaizen Delegate wrappers,
accuracy/tokens/latency metrics, label normalisation.

Technique-specific code (prompt templates, reasoning extraction, majority
vote, structured output Signatures) does NOT belong here — it lives in
the per-technique files.

Provider note: every LLM call routes through ``shared.mlfp06._ollama_bootstrap``
to a locally-running Ollama daemon. There is NO silent OpenAI fallback —
``OllamaUnreachableError`` propagates so a missing daemon surfaces loudly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from shared.kailash_helpers import setup_environment
from shared.mlfp06._ollama_bootstrap import (
    DEFAULT_CHAT_MODEL,
    OLLAMA_BASE_URL,
    make_delegate,
    preflight_ollama,
    run_delegate_text,
)

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

MODEL = DEFAULT_CHAT_MODEL

CATEGORIES: list[str] = ["positive", "negative"]
DEFAULT_EVAL_N: int = 20

# Output directory for any artifacts (comparison tables, plots)
OUTPUT_DIR = Path("outputs") / "ex1_prompting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Each technique file writes its MEASURED metrics here so later files can
# compare against real numbers from your own run (never invented baselines).
METRICS_FILE = OUTPUT_DIR / "technique_metrics.json"

# ILLUSTRATIVE reference price, USD per 1M tokens (input + output blended).
# Local Ollama is free — this constant only translates the token counts you
# measured into "what this workload WOULD cost on a hosted small model at an
# assumed price". It is not a bill and not any provider's quoted price;
# change it to your own provider's rate when you size a real deployment.
REFERENCE_PRICE_USD_PER_MTOK: float = 0.50

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — SST-2 sentiment (Stanford Sentiment Treebank)
# ════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data/mlfp06/sst2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "sst2_200.parquet"


def load_sst2(n_rows: int = 200) -> pl.DataFrame:
    """Load SST-2 from HuggingFace, cache as parquet. Returns polars DataFrame.

    Columns: text (str), label (positive|negative), label_id (0|1)
    """
    if CACHE_FILE.exists():
        return pl.read_parquet(CACHE_FILE)

    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_rows, len(ds))))

    label_names = {0: "negative", 1: "positive"}
    rows = [
        {
            "text": row["sentence"],
            "label": label_names[row["label"]],
            "label_id": row["label"],
        }
        for row in ds
    ]
    df = pl.DataFrame(rows)
    df.write_parquet(CACHE_FILE)
    return df


def get_eval_docs(n: int = DEFAULT_EVAL_N) -> pl.DataFrame:
    """Return first N rows of SST-2 for prompting evaluation."""
    return load_sst2().head(n)


# ════════════════════════════════════════════════════════════════════════
# LLM CALL — Kaizen Delegate wrapper
# ════════════════════════════════════════════════════════════════════════


_OLLAMA_CHECKED = False


def ensure_ollama() -> None:
    """Fail loudly (once per process) if the Ollama daemon or model is missing.

    Raises ``OllamaUnreachableError`` with the fix ("start Ollama: ollama
    serve" / "ollama pull <model>") instead of letting the first LLM call
    die with a low-level connection error.
    """
    global _OLLAMA_CHECKED
    if not _OLLAMA_CHECKED:
        preflight_ollama(required_models=[MODEL])
        _OLLAMA_CHECKED = True


async def run_delegate(prompt: str) -> tuple[str, int, float]:
    """Run an Ollama-backed Delegate and return ``(text, total_tokens, elapsed_s)``.

    Per ``rules/zero-tolerance.md`` Rule 2 there is NO silent stub fallback:
    if the Ollama daemon is unreachable, the underlying ``OllamaUnreachableError``
    propagates so the student fixes the environment instead of seeing a
    fake-zero accuracy result.

    Note: the previous OpenAI-backed signature returned ``cost_usd`` instead of
    ``total_tokens``. Ollama is free, so cost is meaningless; tokens are the
    honest "load each technique put on the model" signal.
    """
    ensure_ollama()
    delegate = make_delegate()
    text, usage, elapsed = await run_delegate_text(delegate, prompt)
    return text, usage["total_tokens"], elapsed


# ════════════════════════════════════════════════════════════════════════
# LABEL NORMALISATION
# ════════════════════════════════════════════════════════════════════════


def normalise_label(raw: str) -> str:
    """Map free-form LLM output onto {positive, negative}. Fallbacks to 'unknown'."""
    lower = raw.strip().lower()
    # prefer the last line — CoT/ZS-CoT puts the final answer there
    last_line = lower.split("\n")[-1]
    if "negative" in last_line:
        return "negative"
    if "positive" in last_line:
        return "positive"
    if "negative" in lower and "positive" not in lower:
        return "negative"
    if "positive" in lower and "negative" not in lower:
        return "positive"
    if lower.count("negative") > lower.count("positive"):
        return "negative"
    if lower.count("positive") > lower.count("negative"):
        return "positive"
    return "unknown"


# ════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════


def reference_cost_usd(total_tokens: int) -> float:
    """Illustrative cost of ``total_tokens`` at :data:`REFERENCE_PRICE_USD_PER_MTOK`."""
    return total_tokens / 1_000_000 * REFERENCE_PRICE_USD_PER_MTOK


def compute_metrics(results: list[dict], name: str) -> dict[str, Any]:
    """Compute accuracy, total_tokens, avg_latency from a results list.

    Each result dict must have keys: correct (bool), tokens (int), elapsed (float).
    The Ollama migration replaced the previous ``cost`` field with ``tokens`` —
    a free local provider has no cost to report, but token throughput is the
    honest comparison signal across techniques.
    """
    n = len(results)
    if n == 0:
        return {
            "strategy": name,
            "n": 0,
            "accuracy": 0.0,
            "total_tokens": 0,
            "avg_latency_s": 0.0,
            "ref_cost_usd": 0.0,
        }
    acc = sum(r["correct"] for r in results) / n
    total_tokens = sum(int(r.get("tokens", 0)) for r in results)
    avg_latency = sum(r.get("elapsed", 0.0) for r in results) / n
    return {
        "strategy": name,
        "n": n,
        "accuracy": acc,
        "total_tokens": total_tokens,
        "avg_latency_s": avg_latency,
        "ref_cost_usd": reference_cost_usd(total_tokens),
        # Per-document record so later files can compare rungs on the SAME
        # documents (e.g. self-consistency evaluates only the first 10).
        "per_doc": [
            {
                "correct": bool(r["correct"]),
                "tokens": int(r.get("tokens", 0)),
                "elapsed": float(r.get("elapsed", 0.0)),
            }
            for r in results
        ],
    }


def print_summary(results: list[dict], name: str) -> None:
    """Print a one-line accuracy/tokens/latency summary for a technique."""
    m = compute_metrics(results, name)
    print(
        f"  {name}: accuracy={m['accuracy']:.0%} | "
        f"tokens={m['total_tokens']} | "
        f"avg_latency={m['avg_latency_s']:.2f}s | n={m['n']}"
    )
    print(
        f"    illustrative hosted cost at ${REFERENCE_PRICE_USD_PER_MTOK:.2f}/Mtok: "
        f"${m['ref_cost_usd']:.5f} for {m['n']} docs "
        f"(local Ollama itself costs $0)"
    )


# ════════════════════════════════════════════════════════════════════════
# MEASURED-METRICS STORE — compare techniques on YOUR numbers only
# ════════════════════════════════════════════════════════════════════════


def save_technique_metrics(metrics: dict[str, Any]) -> Path:
    """Persist one technique's measured metrics (keyed by ``strategy``)."""
    store: dict[str, Any] = {}
    if METRICS_FILE.exists():
        store = json.loads(METRICS_FILE.read_text())
    store[metrics["strategy"]] = metrics
    METRICS_FILE.write_text(json.dumps(store, indent=2))
    return METRICS_FILE


def load_technique_metrics(
    strategies: list[str], n_docs: int | None = None
) -> list[dict[str, Any]]:
    """Return measured metrics for ``strategies`` saved by earlier files.

    Strategies whose file has not been run yet are reported and skipped —
    no placeholder numbers are substituted. With ``n_docs``, each saved
    run is re-scored on its first ``n_docs`` documents so that a technique
    evaluated on a smaller subset is compared on the same documents.
    """
    store: dict[str, Any] = {}
    if METRICS_FILE.exists():
        store = json.loads(METRICS_FILE.read_text())
    found = []
    for name in strategies:
        if name not in store:
            continue
        saved = store[name]
        if n_docs is not None:
            subset = [
                {"correct": d["correct"], "tokens": d["tokens"], "elapsed": d["elapsed"]}
                for d in saved["per_doc"][:n_docs]
            ]
            saved = compute_metrics(subset, name)
        found.append(saved)
    missing = [s for s in strategies if s not in store]
    if missing:
        print(
            f"  Not yet measured (run their files first to compare): {missing}"
        )
    return found


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — strategy comparison table
# ════════════════════════════════════════════════════════════════════════


def build_comparison_df(all_metrics: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn a list of compute_metrics() dicts into a polars DataFrame."""
    return pl.DataFrame(
        [{k: v for k, v in m.items() if k != "per_doc"} for m in all_metrics]
    )


# ════════════════════════════════════════════════════════════════════════
# PLOT HELPERS — matplotlib visualisations for prompt-engineering results
# ════════════════════════════════════════════════════════════════════════


def plot_accuracy_bars(
    results: list[dict],
    categories: list[str],
    title: str,
    filename: str,
) -> None:
    """Per-category accuracy bar chart from a results list."""
    cat_correct: dict[str, int] = {c: 0 for c in categories}
    cat_total: dict[str, int] = {c: 0 for c in categories}
    for r in results:
        true = r["true"]
        if true in cat_total:
            cat_total[true] += 1
            if r["correct"]:
                cat_correct[true] += 1
    names = list(cat_total.keys())
    accs = [cat_correct[c] / max(cat_total[c], 1) for c in names]

    _, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=["steelblue", "darkorange"], edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.02,
            f"{acc:.0%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_comparison_bars(
    metrics_list: list[dict[str, Any]],
    title: str,
    filename: str,
) -> None:
    """Grouped bar chart comparing accuracy, tokens, and latency across strategies."""
    names = [m["strategy"] for m in metrics_list]
    accs = [m["accuracy"] for m in metrics_list]
    tokens = [m["total_tokens"] for m in metrics_list]
    latencies = [m["avg_latency_s"] for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    colors = [
        "steelblue",
        "darkorange",
        "seagreen",
        "mediumpurple",
        "crimson",
        "goldenrod",
    ][: len(names)]

    axes[0].bar(names, accs, color=colors, edgecolor="white")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    axes[1].bar(names, tokens, color=colors, edgecolor="white")
    axes[1].set_ylabel("Total tokens")
    axes[1].set_title("Tokens")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, latencies, color=colors, edgecolor="white")
    axes[2].set_ylabel("Avg latency (s)")
    axes[2].set_title("Latency")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_tokens_vs_accuracy(
    metrics_list: list[dict[str, Any]],
    title: str,
    filename: str,
) -> None:
    """Scatter plot of total tokens consumed vs accuracy across strategies."""
    _, ax = plt.subplots(figsize=(7, 5))
    colors = [
        "steelblue",
        "darkorange",
        "seagreen",
        "mediumpurple",
        "crimson",
        "goldenrod",
    ]
    for i, m in enumerate(metrics_list):
        ax.scatter(
            m["total_tokens"],
            m["accuracy"],
            s=120,
            color=colors[i % len(colors)],
            zorder=3,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.annotate(
            m["strategy"],
            (m["total_tokens"], m["accuracy"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    ax.set_xlabel("Total tokens")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


# Backwards-compat alias for any caller still importing the old name.
plot_cost_vs_accuracy = plot_tokens_vs_accuracy


def plot_vote_agreement(
    results: list[dict],
    n_samples: int,
    title: str,
    filename: str,
) -> None:
    """Histogram of vote-agreement counts for self-consistency results."""
    agreement_counts = [len(set(r["votes"])) for r in results]
    _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        agreement_counts,
        bins=range(1, n_samples + 2),
        color="steelblue",
        edgecolor="white",
        align="left",
        rwidth=0.7,
    )
    ax.set_xlabel("Distinct labels in votes")
    ax.set_ylabel("Number of samples")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, n_samples + 1))
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_extraction_accuracy(
    results: list,
    field_names: list[str],
    title: str,
    filename: str,
) -> None:
    """Bar chart of extraction completeness per output field."""
    field_fill = {}
    n = len(results)
    for field in field_names:

        def _get(r: object, f: str) -> object:
            """Get field from dict or object — supports both 0.9 dicts and legacy attrs."""
            if isinstance(r, dict):
                return r.get(f)
            return getattr(r, f, None)

        filled = sum(
            1
            for r in results
            if _get(r, field) is not None
            and (
                not isinstance(_get(r, field), (list, str))
                or len(_get(r, field)) > 0  # type: ignore[arg-type]
            )
        )
        field_fill[field] = filled / max(n, 1)
    names = list(field_fill.keys())
    rates = list(field_fill.values())

    _, ax = plt.subplots(figsize=(7, 4))
    colors = [
        "seagreen" if r >= 0.9 else "darkorange" if r >= 0.5 else "crimson"
        for r in rates
    ]
    bars = ax.barh(names, rates, color=colors, edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Extraction rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, rate in zip(bars, rates):
        ax.text(
            rate + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0%}",
            va="center",
            fontsize=10,
        )
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")
