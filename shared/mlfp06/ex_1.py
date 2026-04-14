# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 1 — Prompt Engineering.

Contains: LLM model resolution, SST-2 dataset loading, Kaizen Delegate
wrappers, accuracy/cost/latency metrics, label normalisation.

Technique-specific code (prompt templates, reasoning extraction, majority
vote, structured output Signatures) does NOT belong here — it lives in
the per-technique files.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv

from kaizen_agents import Delegate

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

MODEL = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not MODEL:
    raise EnvironmentError(
        "Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env before running"
    )

CATEGORIES: list[str] = ["positive", "negative"]
DEFAULT_EVAL_N: int = 20

# Output directory for any artifacts (comparison tables, plots)
OUTPUT_DIR = Path("outputs") / "ex1_prompting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


async def run_delegate(prompt: str, max_cost: float = 0.5) -> tuple[str, float, float]:
    """Run a Kaizen Delegate call and return (text, cost_usd, elapsed_s).

    Uses the MODEL resolved from the environment. Never hardcode models.
    """
    delegate = Delegate(model=MODEL, max_llm_cost_usd=max_cost)
    response, cost = "", 0.0
    t0 = time.perf_counter()
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
        if hasattr(event, "cost"):
            cost = event.cost
    elapsed = time.perf_counter() - t0
    return response, cost, elapsed


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


def compute_metrics(results: list[dict], name: str) -> dict[str, Any]:
    """Compute accuracy, total_cost, avg_latency from a results list.

    Each result dict must have keys: correct (bool), cost (float), elapsed (float).
    """
    n = len(results)
    if n == 0:
        return {
            "strategy": name,
            "n": 0,
            "accuracy": 0.0,
            "total_cost": 0.0,
            "avg_latency_s": 0.0,
        }
    acc = sum(r["correct"] for r in results) / n
    total_cost = sum(r.get("cost", 0.0) for r in results)
    avg_latency = sum(r.get("elapsed", 0.0) for r in results) / n
    return {
        "strategy": name,
        "n": n,
        "accuracy": acc,
        "total_cost": total_cost,
        "avg_latency_s": avg_latency,
    }


def print_summary(results: list[dict], name: str) -> None:
    """Print a one-line accuracy/cost/latency summary for a technique."""
    m = compute_metrics(results, name)
    print(
        f"  {name}: accuracy={m['accuracy']:.0%} | "
        f"cost=${m['total_cost']:.4f} | "
        f"avg_latency={m['avg_latency_s']:.2f}s | n={m['n']}"
    )


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — strategy comparison table
# ════════════════════════════════════════════════════════════════════════


def build_comparison_df(all_metrics: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn a list of compute_metrics() dicts into a polars DataFrame."""
    return pl.DataFrame(all_metrics)


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

    fig, ax = plt.subplots(figsize=(6, 4))
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
    """Grouped bar chart comparing accuracy, cost, and latency across strategies."""
    names = [m["strategy"] for m in metrics_list]
    accs = [m["accuracy"] for m in metrics_list]
    costs = [m["total_cost"] for m in metrics_list]
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

    axes[1].bar(names, costs, color=colors, edgecolor="white")
    axes[1].set_ylabel("Total cost (USD)")
    axes[1].set_title("Cost")
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


def plot_cost_vs_accuracy(
    metrics_list: list[dict[str, Any]],
    title: str,
    filename: str,
) -> None:
    """Scatter plot of cost vs accuracy across strategies."""
    fig, ax = plt.subplots(figsize=(7, 5))
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
            m["total_cost"],
            m["accuracy"],
            s=120,
            color=colors[i % len(colors)],
            zorder=3,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.annotate(
            m["strategy"],
            (m["total_cost"], m["accuracy"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    ax.set_xlabel("Total cost (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_vote_agreement(
    results: list[dict],
    n_samples: int,
    title: str,
    filename: str,
) -> None:
    """Histogram of vote-agreement counts for self-consistency results."""
    agreement_counts = [len(set(r["votes"])) for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
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
        filled = sum(
            1
            for r in results
            if hasattr(r, field)
            and getattr(r, field) is not None
            and (
                not isinstance(getattr(r, field), (list, str))
                or len(getattr(r, field)) > 0
            )
        )
        field_fill[field] = filled / max(n, 1)
    names = list(field_fill.keys())
    rates = list(field_fill.values())

    fig, ax = plt.subplots(figsize=(7, 4))
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
