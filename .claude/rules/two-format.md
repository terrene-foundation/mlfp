---
paths:
  - "modules/**"
---

# Two-Format Consistency

Every exercise MUST exist in two formats with identical learning content.

## Formats

| Format   | Location                                                     | Data Loading         | Best For                      |
| -------- | ------------------------------------------------------------ | -------------------- | ----------------------------- |
| VS Code  | `modules/mlfpNN/local/ex_N/NN_technique.py`                  | `shared.data_loader` | Full async, local development |
| Colab    | `modules/mlfpNN/colab-selfcontained/ex_N/NN_technique.ipynb` | Inlined helpers, pip | Zero-install, GPU access      |
| Solution | `modules/mlfpNN/solutions/ex_N/NN_technique.py`              | `shared.data_loader` | Source of truth               |

Exercises with multiple techniques are directories (Redline 10). Each technique file follows the 5-phase structure independently.

### Solution `_shared.py` vs Student `helpers.py`

Solutions contain `_shared.py` with reusable infrastructure. Students receive `helpers.py` (renamed, no underscore prefix). Both MUST include `sys.path` setup to find the project root. `_shared.py` is NOT distributed to students.

## Format Differences (ONLY these may differ)

| Aspect        | Local (.py)           | Colab (.ipynb)                                 |
| ------------- | --------------------- | ---------------------------------------------- |
| Setup         | None (uv sync)        | Cell 0 `!pip install` + Cell 1 inlined helpers |
| Data loading  | `shared.data_loader`  | Same helpers, inlined                          |
| Async         | `asyncio.run(main())` | Top-level `await`                              |
| Visualization | `fig.write_html()`    | Inline display                                 |

## Exercise Code MUST Be Identical

The `# TODO:` markers, hints, variable names, Kailash engine calls, and expected outputs MUST be the same across both formats. Only the preamble (setup/data-loading) differs.

## Data Loading Pattern

```python
# Both formats use MLFPDataLoader
from shared.data_loader import MLFPDataLoader
loader = MLFPDataLoader()
df = loader.load("mlfp01", "hdbprices.csv")
```

The loader auto-detects Colab vs local and uses the appropriate backend.

## Canonical Conversion Tool

Local `.py` is the source of truth. Colab notebooks are generated, never hand-written.

```bash
# Single module (solutions + students)
python scripts/generate_selfcontained_notebook.py \
  --solutions modules/mlfp01/solutions \
  --local modules/mlfp01/local \
  --out-solutions modules/mlfp01/colab-selfcontained-solutions \
  --out-students modules/mlfp01/colab-selfcontained

# Single file
python scripts/generate_selfcontained_notebook.py \
  modules/mlfp05/solutions/ex_1/01_standard_ae.py \
  --out modules/mlfp05/colab-selfcontained-solutions/ex_1/01_standard_ae.ipynb
```

The generator: strips all `from shared.*` imports (single-line + multi-line paren forms), dedupes `from __future__`, inlines per-module helpers into Cell 1, AST-validates every code cell before writing.

## MUST NOT

- Write an exercise in only one format
- Change exercise logic between formats
- Use `pd.read_csv()` in any format (polars only)
- Hardcode Google Drive paths in local format
- Skip the setup cell in Colab notebooks
