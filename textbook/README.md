# Kailash SDK Textbook

Comprehensive tutorials for the Kailash Python SDK and Kailash Rust SDK, from basic to advanced. Each tutorial is runnable code that validates the API works as documented.

**Supplementary to**: [ASCENT — ML Engineering from Foundations to Mastery](../README.md) (ASCENT) at Terrene Open Academy
**Maintained by**: Terrene Foundation

## Purpose

1. **Developer reference** — learn every engine, primitive, and pattern across all 8 Kailash packages
2. **API validation** — each tutorial includes assertions that catch spec mismatches
3. **Parity testing** — Python and Rust tutorials side by side, documenting cross-language equivalence

## Structure

| Chapter     | Package                                   | Python       | Rust        | Parity                 |
| ----------- | ----------------------------------------- | ------------ | ----------- | ---------------------- |
| 00-core     | `kailash` / `kailash-core`                | 10 tutorials | 7 tutorials | High (20 known DIVs)   |
| 01-dataflow | `kailash-dataflow`                        | 9 tutorials  | 5 tutorials | High                   |
| 02-nexus    | `kailash-nexus`                           | 9 tutorials  | 6 tutorials | High                   |
| 03-kaizen   | `kailash-kaizen`                          | 6 tutorials  | 5 tutorials | Moderate               |
| 04-agents   | `kaizen-agents`                           | 10 tutorials | 4 tutorials | Low (Py richer)        |
| 05-ml       | `kailash-ml` / `kailash-ml-*`             | 16 tutorials | 8 tutorials | Different architecture |
| 06-pact     | `kailash-pact` / `kailash-governance`     | 9 tutorials  | 5 tutorials | High                   |
| 07-align    | `kailash-align` / `kailash-align-serving` | 9 tutorials  | 2 tutorials | Low (Py richer)        |
| 08          | Integration (Py) / RL (Rs)                | 5 tutorials  | 2 tutorials | —                      |
| **Total**   |                                           | **73**       | **44**      | **117 tutorials**      |

## Setup

### Python

```bash
cd /path/to/ascent
uv sync

# Run a single tutorial
uv run python textbook/python/00-core/01_workflow_builder.py

# Run all tutorials in a chapter
for f in textbook/python/00-core/*.py; do uv run python "$f"; done
```

### Rust

```bash
# Requires kailash-rs cloned at ~/repos/loom/kailash-rs
cd textbook/rust
cargo build --workspace

# Run a single tutorial
cargo run -p tutorial-core --bin 01_workflow_builder
```

## Parity Matrix

See [PARITY.md](PARITY.md) for the complete cross-language parity matrix, referencing the 20 known divergences from `kailash-rs/tests/parity/divergences.json`.

## Tutorial Format

Every tutorial follows a consistent pattern:

1. **Header** — objective, level, parity status, what API is validated
2. **Imports** — exact imports from the SDK
3. **Demonstration** — focused example of one concept
4. **Validation** — assertions that verify the API contract
5. **Edge cases** — common mistakes and error handling
6. **PASS output** — prints `PASS: [name]` on success

## Progression

Within each chapter, tutorials progress from basic to advanced:

- **Basic** (01-03): Constructor, primary method, basic usage
- **Intermediate** (04-06): Configuration, composition, common patterns
- **Advanced** (07+): Edge cases, performance, production patterns

## License

Apache 2.0 — Terrene Foundation
