# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / Evaluation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Evaluate aligned models against benchmarks
# LEVEL: Advanced
# PARITY: Python-only
# VALIDATES: AlignmentEvaluator, EvalResult, TaskResult, EvalConfig,
#            QUICK_TASKS, STANDARD_TASKS
#
# Run: uv run python textbook/python/07-align/08_evaluation.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash_align import AlignmentEvaluator, EvalConfig, EvalResult, TaskResult
from kailash_align.config import QUICK_TASKS, STANDARD_TASKS

# ── 1. EvalConfig defaults ─────────────────────────────────────────

default_eval = EvalConfig()

assert default_eval.tasks == ("arc_easy", "hellaswag", "truthfulqa_mc1")
assert default_eval.limit == 100, "Default limit=100 for interactive use"
assert default_eval.batch_size == "auto"
assert default_eval.num_fewshot is None, "Uses task default"
assert default_eval.device is None, "Auto-detect device"
assert default_eval.local_files_only is False
assert default_eval.use_adapter is True

# ── 2. EvalConfig is frozen ────────────────────────────────────────

try:
    default_eval.limit = 50  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass

# ── 3. EvalConfig validation — limit must be >= 1 ──────────────────

try:
    EvalConfig(limit=0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

try:
    EvalConfig(limit=-1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

# ── 4. Custom EvalConfig ───────────────────────────────────────────

custom_eval = EvalConfig(
    tasks=("arc_easy", "arc_challenge", "mmlu"),
    limit=50,
    batch_size="16",
    num_fewshot=5,
    device="cpu",
    use_adapter=False,
)

assert custom_eval.tasks == ("arc_easy", "arc_challenge", "mmlu")
assert custom_eval.limit == 50
assert custom_eval.batch_size == "16"
assert custom_eval.num_fewshot == 5
assert custom_eval.device == "cpu"
assert custom_eval.use_adapter is False

# ── 5. Task presets ─────────────────────────────────────────────────
# QUICK_TASKS: ~5 minutes on single GPU with limit=100
# STANDARD_TASKS: ~30-60 minutes on A100 with limit=100

assert isinstance(QUICK_TASKS, list)
assert len(QUICK_TASKS) == 3
assert "arc_easy" in QUICK_TASKS
assert "hellaswag" in QUICK_TASKS
assert "truthfulqa_mc1" in QUICK_TASKS

assert isinstance(STANDARD_TASKS, list)
assert len(STANDARD_TASKS) == 6
assert "mmlu" in STANDARD_TASKS
assert "winogrande" in STANDARD_TASKS
assert "arc_challenge" in STANDARD_TASKS

# All quick tasks are in standard tasks
for task in QUICK_TASKS:
    assert task in STANDARD_TASKS, f"Quick task {task} should also be in standard"

# ── 6. TaskResult dataclass ─────────────────────────────────────────

task_result = TaskResult(
    task_name="arc_easy",
    metrics={"acc": 0.75, "acc_norm": 0.72, "acc_stderr": 0.02},
    num_samples=100,
    task_version="1.0",
)

assert task_result.task_name == "arc_easy"
assert task_result.metrics["acc"] == 0.75
assert task_result.num_samples == 100
assert task_result.task_version == "1.0"

# ── 7. TaskResult serialization ─────────────────────────────────────

task_dict = task_result.to_dict()
assert task_dict["task_name"] == "arc_easy"
assert task_dict["metrics"]["acc"] == 0.75
assert task_dict["num_samples"] == 100

# Round-trip: to_dict -> from_dict
restored = TaskResult.from_dict(task_dict)
assert restored.task_name == task_result.task_name
assert restored.metrics == task_result.metrics
assert restored.num_samples == task_result.num_samples
assert restored.task_version == task_result.task_version

# ── 8. EvalResult dataclass ────────────────────────────────────────

eval_result = EvalResult(
    adapter_name="my-adapter",
    adapter_version="3",
    task_results=[
        TaskResult(
            task_name="arc_easy",
            metrics={"acc": 0.75, "acc_norm": 0.72},
            num_samples=100,
        ),
        TaskResult(
            task_name="hellaswag",
            metrics={"acc": 0.62, "acc_norm": 0.58},
            num_samples=100,
        ),
        TaskResult(
            task_name="truthfulqa_mc1",
            metrics={"acc": 0.41, "mc2": 0.55},
            num_samples=100,
        ),
    ],
    eval_config={"tasks": ["arc_easy", "hellaswag", "truthfulqa_mc1"], "limit": 100},
    total_duration_seconds=245.7,
)

assert eval_result.adapter_name == "my-adapter"
assert eval_result.adapter_version == "3"
assert len(eval_result.task_results) == 3
assert eval_result.total_duration_seconds == 245.7

# ── 9. EvalResult.summary — quick accuracy lookup ──────────────────
# summary uses the first metric containing 'acc' as the primary metric.

summary = eval_result.summary
assert "arc_easy" in summary
assert summary["arc_easy"] == 0.75
assert "hellaswag" in summary
assert summary["hellaswag"] == 0.62
assert "truthfulqa_mc1" in summary
assert summary["truthfulqa_mc1"] == 0.41

# ── 10. EvalResult serialization ───────────────────────────────────

eval_dict = eval_result.to_dict()
assert eval_dict["adapter_name"] == "my-adapter"
assert eval_dict["adapter_version"] == "3"
assert len(eval_dict["task_results"]) == 3
assert eval_dict["total_duration_seconds"] == 245.7

# Round-trip
restored_eval = EvalResult.from_dict(eval_dict)
assert restored_eval.adapter_name == eval_result.adapter_name
assert restored_eval.adapter_version == eval_result.adapter_version
assert len(restored_eval.task_results) == len(eval_result.task_results)
assert restored_eval.summary == eval_result.summary

# ── 11. AlignmentEvaluator creation ────────────────────────────────
# AlignmentEvaluator wraps lm-eval-harness for standard benchmarks
# and supports custom evaluation. Does not require GPU to instantiate.

evaluator = AlignmentEvaluator()
assert evaluator._registry is None

from kailash_align import AdapterRegistry

registry = AdapterRegistry()
evaluator_with_registry = AlignmentEvaluator(adapter_registry=registry)
assert evaluator_with_registry._registry is registry

# ── 12. EvalResult with no accuracy metrics ─────────────────────────
# summary gracefully handles tasks with no 'acc' key.

no_acc_result = EvalResult(
    adapter_name="test",
    adapter_version="1",
    task_results=[
        TaskResult(
            task_name="custom_task",
            metrics={"f1": 0.85, "precision": 0.9},
            num_samples=50,
        ),
    ],
    eval_config={"tasks": ["custom_task"]},
    total_duration_seconds=10.0,
)
assert no_acc_result.summary == {}, "No 'acc' metrics -> empty summary"

print("PASS: 07-align/08_evaluation")
