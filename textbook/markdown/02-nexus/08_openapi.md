# Chapter 8: OpenAPI

## Overview

Nexus can auto-generate OpenAPI 3.0.3 specifications from registered handlers and workflows. The `OpenApiGenerator` derives request schemas from function type annotations, maps Python types to OpenAPI types, and produces a complete spec with paths, schemas, and server information. This chapter covers `OpenApiInfo`, schema generation from handler signatures, workflow documentation, and the type mapping system.

## Prerequisites

- Completed Chapters 1-7
- Familiarity with the OpenAPI / Swagger specification format
- Understanding of Python type annotations

## Concepts

### Automatic Schema Derivation

When you register a handler function like `async def predict(model_name: str, features: list, threshold: float = 0.5)`, the generator inspects the type annotations to produce:

- **Required fields**: parameters without defaults (`model_name`, `features`)
- **Optional fields**: parameters with defaults (`threshold`)
- **Type mappings**: `str` -> `string`, `int` -> `integer`, `float` -> `number`, etc.

### Handler vs Workflow Paths

Handlers create a single endpoint: `POST /workflows/{name}/execute`. Workflows create two: an execute endpoint and an info endpoint at `GET /workflows/{name}/workflow/info`.

### Python-to-OpenAPI Type Mapping

| Python Type   | OpenAPI Type              |
| ------------- | ------------------------- |
| `str`         | `string`                  |
| `int`         | `integer`                 |
| `float`       | `number`                  |
| `bool`        | `boolean`                 |
| `list`        | `array` (items: string)   |
| `dict`        | `object`                  |
| `bytes`       | `string` (format: binary) |
| `Optional[X]` | unwraps to inner type     |

## Key API

| Method / Property                                        | Parameters                                                                                                            | Returns            | Description                           |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------- |
| `OpenApiInfo()`                                          | `title`, `version`, `description`, `contact_name`, `contact_email`, `license_name`, `license_url`, `terms_of_service` | `OpenApiInfo`      | Spec info section                     |
| `OpenApiGenerator(info, servers)`                        | `info: OpenApiInfo`, `servers: list[dict]`                                                                            | `OpenApiGenerator` | Create generator                      |
| `gen.add_handler(name, handler_func, description, tags)` | handler function + metadata                                                                                           | `None`             | Register handler for docs             |
| `gen.add_workflow(name, workflow, description, tags)`    | workflow + metadata                                                                                                   | `None`             | Register workflow for docs            |
| `gen.generate()`                                         | --                                                                                                                    | `dict`             | Generate OpenAPI 3.0.3 spec           |
| `gen.generate_json(indent)`                              | `indent: int`                                                                                                         | `str`              | Generate spec as JSON string          |
| `_python_type_to_openapi(type)`                          | Python type                                                                                                           | `dict`             | Convert Python type to OpenAPI schema |

## Code Walkthrough

### Step 1: Configure OpenApiInfo

```python
from nexus.openapi import OpenApiGenerator, OpenApiInfo

info = OpenApiInfo(
    title="My ML Platform API",
    version="2.0.0",
    description="Production ML inference endpoints",
    contact_name="Platform Team",
    contact_email="platform@example.com",
    license_name="Apache-2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0",
)
```

Defaults: `title="Kailash Nexus API"`, `version="1.0.0"`, `license_name="Apache-2.0"`.

### Step 2: Create Generator with Servers

```python
gen = OpenApiGenerator(
    info=info,
    servers=[
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "http://localhost:8000", "description": "Development"},
    ],
)
```

### Step 3: Register a Handler

```python
async def predict(model_name: str, features: list, threshold: float = 0.5) -> dict:
    """Run inference on a trained model."""
    return {"prediction": 1, "confidence": 0.95}

gen.add_handler(
    name="predict",
    handler_func=predict,
    description="Run model inference",
    tags=["inference"],
)
```

The generator inspects `predict`'s annotations: `model_name` and `features` become required fields; `threshold` becomes optional with default `0.5`.

### Step 4: Register a Workflow

```python
from kailash import WorkflowBuilder

wb = WorkflowBuilder()
wb.add_node("PythonCodeNode", "greet", {
    "code": "output = f'Hello, {name}!'",
    "inputs": {"name": "str"},
    "output_type": "str",
})

gen.add_workflow(
    name="greet",
    workflow=wb.build(name="greeter"),
    description="Greet a user by name",
    tags=["greetings"],
)
```

### Step 5: Generate the Spec

```python
spec = gen.generate()

assert spec["openapi"] == "3.0.3"
assert spec["info"]["title"] == "My ML Platform API"
assert "/workflows/predict/execute" in spec["paths"]
assert "/workflows/greet/execute" in spec["paths"]
assert "/workflows/greet/workflow/info" in spec["paths"]
```

### Step 6: Verify Schemas

```python
schemas = spec["components"]["schemas"]
predict_schema = schemas["predict_input"]

assert predict_schema["properties"]["model_name"]["type"] == "string"
assert predict_schema["properties"]["features"]["type"] == "array"
assert predict_schema["properties"]["threshold"]["type"] == "number"
assert predict_schema["properties"]["threshold"]["default"] == 0.5
assert "model_name" in predict_schema["required"]
assert "threshold" not in predict_schema["required"]
```

### Step 7: JSON Output

```python
import json

json_str = gen.generate_json(indent=2)
parsed = json.loads(json_str)
assert parsed["openapi"] == "3.0.3"
```

## Common Mistakes

| Mistake                                        | Problem                                 | Fix                                                   |
| ---------------------------------------------- | --------------------------------------- | ----------------------------------------------------- |
| Missing type annotations on handler params     | Parameters typed as `string` by default | Always annotate function parameters                   |
| Forgetting to add handlers before `generate()` | Empty paths in spec                     | Register all handlers first                           |
| Using complex nested types                     | Mapped to `string` fallback             | Use simple types or document complex schemas manually |
| Not setting `description` on handlers          | Empty operation descriptions            | Always provide descriptions                           |

## Exercises

1. **Type Mapping**: Write a handler function with parameters of every supported type (`str`, `int`, `float`, `bool`, `list`, `dict`, `bytes`). Generate the spec and verify each parameter's OpenAPI type mapping.

2. **Multi-Handler Spec**: Register five handlers across three tag groups. Generate the spec and verify the tag structure in the output. Count the total paths.

3. **JSON Export**: Generate a complete spec, export it as JSON, and validate that it parses back to an identical dictionary. Write the JSON to a file for use with Swagger UI.

## Key Takeaways

- `OpenApiInfo` configures the info section with sensible Kailash-branded defaults.
- `OpenApiGenerator` builds OpenAPI 3.0.3 specs from handler and workflow registrations.
- `add_handler()` derives request schemas from Python function signatures automatically.
- Handlers create `POST /workflows/{name}/execute`; workflows also get `GET .../workflow/info`.
- `generate()` returns a dict; `generate_json()` returns a formatted JSON string.
- Python types map directly to OpenAPI types; unknown types fall back to `string`.

## Next Chapter

[Chapter 9: Probes](09_probes.md) -- Configure health, readiness, and startup probes for Kubernetes deployments.
