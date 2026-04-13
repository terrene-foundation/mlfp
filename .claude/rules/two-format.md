---
paths:
  - "modules/**"
---

# Two-Format Consistency

Every exercise MUST exist in two formats with identical learning content.

## Formats

| Format   | Location                                        | Data Loading         | Best For                      |
| -------- | ----------------------------------------------- | -------------------- | ----------------------------- |
| VS Code  | `modules/mlfpNN/local/ex_N/NN_technique.py`     | `shared.data_loader` | Full async, local development |
| Colab    | `modules/mlfpNN/colab/ex_N/NN_technique.ipynb`  | Drive mount + gdown  | Zero-install, GPU access      |
| Solution | `modules/mlfpNN/solutions/ex_N/NN_technique.py` | `shared.data_loader` | Source of truth               |

Exercises with multiple techniques are directories (Redline 10). Each technique file follows the 5-phase structure independently.

### Solution `_shared.py` vs Student `helpers.py`

Solutions contain `_shared.py` with reusable infrastructure. Students receive `helpers.py` (renamed, no underscore prefix). Both MUST include `sys.path` setup to find the project root. `_shared.py` is NOT distributed to students.

## Format Differences (ONLY these may differ)

| Aspect        | Local (.py)            | Colab (.ipynb)                    |
| ------------- | ---------------------- | --------------------------------- |
| Setup         | None (uv sync)         | `!pip install` + Drive mount cell |
| Data loading  | gdown + `.data_cache/` | Drive mount path                  |
| Async         | `asyncio.run(main())`  | Top-level `await`                 |
| Visualization | `fig.write_html()`     | Inline display                    |

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
# Convert a single exercise directory
python scripts/py_to_notebook.py modules/mlfp01/local/ex_1/

# Convert an entire module
python scripts/py_to_notebook.py --module mlfp01

# Convert all modules
python scripts/py_to_notebook.py --all
```

The converter handles: `asyncio.run()` -> `await`, setup cell injection, copyright stripping, TASK headers -> markdown cells.

## MUST NOT

- Write an exercise in only one format
- Change exercise logic between formats
- Use `pd.read_csv()` in any format (polars only)
- Hardcode Google Drive paths in local format
- Skip the setup cell in Colab notebooks
