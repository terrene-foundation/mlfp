# Three-Format Consistency

Every exercise MUST exist in three formats with identical learning content.

## Format Differences (ONLY these may differ)

| Aspect | Local (.py) | Jupyter (.ipynb) | Colab (.ipynb) |
|--------|------------|------------------|----------------|
| Setup | None (uv sync) | `%pip install` cell | `!pip install` + Drive mount cell |
| Data loading | gdown + `.data_cache/` | gdown + `.data_cache/` | Drive mount path |
| Async | `asyncio.run(main())` | Top-level `await` | Top-level `await` |
| Visualization | `fig.write_html()` | Inline display | Inline display |

## Exercise Code MUST Be Identical

The `# TODO:` markers, hints, variable names, Kailash engine calls, and expected outputs MUST be the same across all three formats. Only the preamble (setup/data-loading) differs.

## Data Loading Pattern

```python
# All three formats use ASCENTDataLoader
from shared.data_loader import ASCENTDataLoader
loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdbprices.csv")
```

The loader auto-detects Colab vs local and uses the appropriate backend.

## MUST NOT

- Write an exercise in only one or two formats
- Change exercise logic between formats
- Use `pd.read_csv()` in any format (polars only)
- Hardcode Google Drive paths in local/Jupyter formats
- Skip the setup cell in Colab notebooks
