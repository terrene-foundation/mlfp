# ASCENT — Practical Course in Machine Learning

Professional-grade machine learning engineering training powered by the [Kailash Python SDK](https://github.com/terrene-foundation/kailash-py). Stanford CS229 depth meets production reality.

**Institution**: Terrene Open Academy | **License**: Apache 2.0 | **Python**: 3.10+

---

## What You'll Learn

Theory deep enough to debug production failures. Practice on messy, real-world data at scale. Every concept bridges mathematical foundations to production Kailash engines.

| Module | Topic | Theory | Practice (Kailash) |
|--------|-------|--------|-------------------|
| 1 | Statistics & Data Fluency | Bayesian estimation, MLE, hypothesis testing | DataExplorer, PreprocessingPipeline |
| 2 | Feature Engineering & Experiments | Causal inference, A/B testing, CUPED | FeatureStore, FeatureEngineer |
| 3 | Supervised ML — Theory to Production | Bias-variance, SHAP/LIME, calibration | TrainingPipeline, ModelRegistry, WorkflowBuilder |
| 4 | Unsupervised ML, NLP & Deep Learning | EM/GMM, UMAP, attention, BERTopic | DriftMonitor, InferenceServer, Nexus |
| 5 | LLMs, AI Agents & RAG | Transformers, RAG evaluation, agent safety | Kaizen Delegate, ReActAgent, ML agents |
| 6 | Alignment, Governance & Deployment | LoRA/DPO, EU AI Act, PACT D/T/R, RL | AlignmentPipeline, GovernanceEngine, Nexus |

## Quick Start

```bash
git clone https://github.com/terrene-foundation/ascent.git
cd ascent
uv venv && uv sync
uv run python modules/ascent01/local/01_polars_fundamentals.py
```

Three formats available:
- `modules/ascent*/local/` — Python scripts (recommended)
- `modules/ascent*/notebooks/` — Jupyter notebooks
- `modules/ascent*/colab/` — Google Colab notebooks

See [docs/setup-guide.md](docs/setup-guide.md) for detailed instructions.

## Data

Datasets load automatically from a shared Google Drive. First run downloads and caches locally.

```python
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdbprices.csv")
```

## Kailash Platform

| Package | Purpose | Install |
|---------|---------|---------|
| kailash | Core workflow orchestration | `pip install kailash` |
| kailash-ml | ML lifecycle (13 engines) | `pip install kailash-ml` |
| kailash-dataflow | Database operations | `pip install kailash-dataflow` |
| kailash-nexus | API + CLI + MCP deployment | `pip install kailash-nexus` |
| kailash-kaizen | AI agent framework | `pip install kailash-kaizen` |
| kailash-pact | Governance (D/T/R) | `pip install kailash-pact` |
| kailash-align | LLM fine-tuning | `pip install kailash-align` |

## Contributing

This is a Terrene Foundation project. Contributions welcome via pull requests.

## License

Apache 2.0 — see [LICENSE](LICENSE).
