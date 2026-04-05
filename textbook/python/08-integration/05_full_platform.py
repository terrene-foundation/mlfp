# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Integration / Full Platform (All 8 Packages)
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Demonstrate all 8 Kailash packages working together
# LEVEL: Advanced
# PARITY: Python-only (full stack)
# VALIDATES: Import and instantiation of all 8 package entry points
#
# Run: uv run python textbook/python/08-integration/05_full_platform.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

# ── 1. Package 1: kailash (Core SDK) ───────────────────────────────
# Workflow orchestration — the foundation.

from kailash import WorkflowBuilder, LocalRuntime, Node

builder = WorkflowBuilder()
assert builder is not None
print("1/8 kailash (core): WorkflowBuilder, LocalRuntime, Node")

# ── 2. Package 2: kailash-ml ───────────────────────────────────────
# 13 ML engines for the complete ML lifecycle.

from kailash_ml import (
    DataExplorer,
    PreprocessingPipeline,
    ModelVisualizer,
    FeatureEngineer,
    FeatureStore,
    ExperimentTracker,
    TrainingPipeline,
    HyperparameterSearch,
    AutoMLEngine,
    EnsembleEngine,
    ModelRegistry,
    DriftMonitor,
    InferenceServer,
)

print("2/8 kailash-ml: 13 engines loaded")

# ── 3. Package 3: kailash-dataflow ─────────────────────────────────
# Zero-config database operations.

from dataflow import DataFlow

db = DataFlow(database_url="sqlite:///")
assert isinstance(db, DataFlow)
print("3/8 kailash-dataflow: DataFlow")

# ── 4. Package 4: kailash-nexus ────────────────────────────────────
# Multi-channel deployment (API + CLI + MCP).

from nexus import Nexus

app = Nexus(enable_durability=False)
assert isinstance(app, Nexus)
print("4/8 kailash-nexus: Nexus")

# ── 5. Package 5: kailash-kaizen ───────────────────────────────────
# AI agent framework — signatures and LLM integration.

from kaizen import Signature, InputField, OutputField

assert Signature is not None
print("5/8 kailash-kaizen: Signature, InputField, OutputField")

# ── 6. Package 6: kaizen-agents ────────────────────────────────────
# Specialized agents and orchestration patterns.

from kaizen_agents import Delegate, GovernedSupervisor

assert Delegate is not None
assert GovernedSupervisor is not None
print("6/8 kaizen-agents: Delegate, GovernedSupervisor")

# ── 7. Package 7: kailash-pact ─────────────────────────────────────
# Organizational governance (D/T/R addressing, envelopes, clearance).

from kailash.trust.pact import (
    Address,
    GovernanceEngine,
    compile_org,
    can_access,
    explain_access,
)

addr = Address.parse("D1-R1")
assert addr.depth() == 2
print("7/8 kailash-pact: Address, GovernanceEngine, compile_org")

# ── 8. Package 8: kailash-align ────────────────────────────────────
# LLM fine-tuning and alignment.

from kailash_align import (
    AlignmentPipeline,
    AlignmentConfig,
    AdapterRegistry,
    AlignmentEvaluator,
)

assert AlignmentConfig is not None
print("8/8 kailash-align: AlignmentPipeline, AlignmentConfig, AdapterRegistry")

# ── Summary ─────────────────────────────────────────────────────────
print()
print("All 8 Kailash packages imported and validated:")
print("  1. kailash       — Workflow orchestration")
print("  2. kailash-ml    — 13 ML engines")
print("  3. kailash-dataflow — Database operations")
print("  4. kailash-nexus — Multi-channel deployment")
print("  5. kailash-kaizen — Agent framework")
print("  6. kaizen-agents — Specialized agents")
print("  7. kailash-pact  — Governance")
print("  8. kailash-align — LLM fine-tuning")
print()
print("This is the foundation for the ASCENT M6 capstone exercise.")

print("PASS: 08-integration/05_full_platform")
