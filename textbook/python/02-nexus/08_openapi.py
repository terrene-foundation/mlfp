# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / OpenAPI
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Auto-generate OpenAPI documentation from registered handlers
# LEVEL: Advanced
# PARITY: Python-only
# VALIDATES: OpenApiGenerator, OpenApiInfo, add_workflow, add_handler,
#            generate, generate_json
#
# Run: uv run python textbook/python/02-nexus/08_openapi.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json

from nexus.openapi import OpenApiGenerator, OpenApiInfo

# ── 1. OpenApiInfo ─────────────────────────────────────────────────
# OpenApiInfo configures the info section of the OpenAPI spec.
# It has sensible defaults for Kailash Nexus projects.

info = OpenApiInfo(
    title="My ML Platform API",
    version="2.0.0",
    description="Production ML inference endpoints",
    contact_name="Platform Team",
    contact_email="platform@example.com",
    license_name="Apache-2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0",
    terms_of_service="https://example.com/terms",
)

assert info.title == "My ML Platform API"
assert info.version == "2.0.0"
assert info.license_name == "Apache-2.0"

# ── 2. OpenApiInfo defaults ───────────────────────────────────────
# The defaults are Kailash-branded and Apache-2.0 licensed.

defaults = OpenApiInfo()

assert defaults.title == "Kailash Nexus API"
assert defaults.version == "1.0.0"
assert defaults.license_name == "Apache-2.0"
assert defaults.contact_name is None  # Optional
assert defaults.terms_of_service is None  # Optional

# ── 3. Create an OpenApiGenerator ─────────────────────────────────
# The generator accepts an OpenApiInfo object or individual
# title/version parameters. It also supports a servers list.

gen = OpenApiGenerator(
    info=info,
    servers=[
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "http://localhost:8000", "description": "Development"},
    ],
)

assert isinstance(gen, OpenApiGenerator)

# Shorthand construction with title/version
gen_simple = OpenApiGenerator(title="Simple API", version="0.1.0")
assert isinstance(gen_simple, OpenApiGenerator)

# ── 4. Register a handler function ────────────────────────────────
# add_handler() derives the request schema from the function's type
# annotations. Required parameters (no default) and optional ones
# (with defaults) are correctly distinguished.


async def predict(model_name: str, features: list, threshold: float = 0.5) -> dict:
    """Run inference on a trained model."""
    return {"prediction": 1, "confidence": 0.95}


gen.add_handler(
    name="predict",
    handler_func=predict,
    description="Run model inference",
    tags=["inference"],
)

# ── 5. Register a handler with minimal types ──────────────────────


async def health_check(service: str = "all") -> dict:
    """Check service health."""
    return {"status": "healthy"}


gen.add_handler(name="health", handler_func=health_check, tags=["system"])

# ── 6. Register a workflow ─────────────────────────────────────────
# add_workflow() derives schema from the workflow's metadata or nodes.
# For this tutorial, we use a simple workflow.

from kailash import WorkflowBuilder

wb = WorkflowBuilder()
wb.add_node(
    "PythonCodeNode",
    "greet",
    {
        "code": "output = f'Hello, {name}!'",
        "inputs": {"name": "str"},
        "output_type": "str",
    },
)
workflow = wb.build(name="greeter")

gen.add_workflow(
    name="greet",
    workflow=workflow,
    description="Greet a user by name",
    tags=["greetings"],
)

# ── 7. Generate the spec ──────────────────────────────────────────
# generate() returns the complete OpenAPI 3.0.3 specification as a
# Python dictionary. All registered handlers and workflows are
# included as paths.

spec = gen.generate()

assert spec["openapi"] == "3.0.3"
assert spec["info"]["title"] == "My ML Platform API"
assert spec["info"]["version"] == "2.0.0"
assert spec["info"]["license"]["name"] == "Apache-2.0"
assert spec["info"]["contact"]["name"] == "Platform Team"
assert spec["info"]["contact"]["email"] == "platform@example.com"
assert spec["info"]["termsOfService"] == "https://example.com/terms"

# Servers are included
assert len(spec["servers"]) == 2
assert spec["servers"][0]["url"] == "https://api.example.com"

# ── 8. Verify handler paths ───────────────────────────────────────
# Each handler creates a POST endpoint at /workflows/{name}/execute.

paths = spec["paths"]

assert "/workflows/predict/execute" in paths
predict_op = paths["/workflows/predict/execute"]["post"]
assert predict_op["operationId"] == "execute_predict"
assert predict_op["tags"] == ["inference"]
assert predict_op["description"] == "Run model inference"
assert "requestBody" in predict_op
assert "200" in predict_op["responses"]
assert "400" in predict_op["responses"]
assert "500" in predict_op["responses"]

assert "/workflows/health/execute" in paths
health_op = paths["/workflows/health/execute"]["post"]
assert health_op["tags"] == ["system"]

# ── 9. Verify workflow paths ──────────────────────────────────────
# Workflows get both an execute endpoint and an info endpoint.

assert "/workflows/greet/execute" in paths
assert "/workflows/greet/workflow/info" in paths

greet_info = paths["/workflows/greet/workflow/info"]["get"]
assert greet_info["operationId"] == "info_greet"
assert greet_info["tags"] == ["greetings"]

# ── 10. Verify schemas ────────────────────────────────────────────
# Handler schemas are derived from function signatures.
# The schema captures types, required fields, and defaults.

schemas = spec["components"]["schemas"]

assert "predict_input" in schemas
predict_schema = schemas["predict_input"]
assert predict_schema["type"] == "object"

props = predict_schema["properties"]
assert props["model_name"]["type"] == "string"
assert props["features"]["type"] == "array"
assert props["threshold"]["type"] == "number"
assert props["threshold"]["default"] == 0.5

# model_name and features are required (no default)
assert "model_name" in predict_schema["required"]
assert "features" in predict_schema["required"]
# threshold has a default, so it's optional
assert "threshold" not in predict_schema["required"]

# ── 11. generate_json() ───────────────────────────────────────────
# Convenience method that returns the spec as a formatted JSON string.

json_str = gen.generate_json(indent=2)
parsed = json.loads(json_str)

assert parsed["openapi"] == "3.0.3"
assert parsed["info"]["title"] == "My ML Platform API"

# ── 12. Type mapping ──────────────────────────────────────────────
# Python types are mapped to OpenAPI types:
#   str -> string, int -> integer, float -> number,
#   bool -> boolean, list -> array, dict -> object,
#   bytes -> string (format: binary)
# Optional[X] unwraps to the inner type.
# Unknown types fall back to "string".

from nexus.openapi import _python_type_to_openapi

assert _python_type_to_openapi(str) == {"type": "string"}
assert _python_type_to_openapi(int) == {"type": "integer"}
assert _python_type_to_openapi(float) == {"type": "number"}
assert _python_type_to_openapi(bool) == {"type": "boolean"}
assert _python_type_to_openapi(dict) == {"type": "object"}
assert _python_type_to_openapi(list) == {"type": "array", "items": {"type": "string"}}
assert _python_type_to_openapi(bytes) == {"type": "string", "format": "binary"}

# ── 13. Key concepts ──────────────────────────────────────────────
# - OpenApiInfo: title, version, description, contact, license
# - OpenApiGenerator(info=, servers=): builds OpenAPI 3.0.3 specs
# - add_handler(): derives schema from function signatures
# - add_workflow(): derives schema from workflow metadata
# - generate() -> dict, generate_json() -> str
# - Schemas capture types, required fields, and defaults
# - Handlers create POST /workflows/{name}/execute paths
# - Workflows also get GET /workflows/{name}/workflow/info paths
# - install(app) mounts GET /openapi.json on a FastAPI/Starlette app
# - NOTE: We don't call install() or start() because they need a running server

print("PASS: 02-nexus/08_openapi")
