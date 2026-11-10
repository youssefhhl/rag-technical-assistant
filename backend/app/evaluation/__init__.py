"""RAGAS evaluation package (Phase 9.1).

Evaluate the production RAG pipeline (Qdrant dense + BM25 sparse -> RRF ->
Cross-Encoder -> Groq generation) with four RAGAS metrics using a golden
dataset grounded in the indexed FastAPI documentation:

- ``dataset``: golden dataset loading/validation (answerable + unanswerable);
- ``samples``: mapping pipeline outputs to the RAGAS ``SingleTurnSample``;
- ``pipeline``: read-only invocation of the real production pipeline;
- ``evaluator``: judge LLM (Groq via existing settings) + the four metrics;
- ``report``/``evaluation_runner``: aggregation, JSON reports and the CLI.

Run it with::

    uv run python -m app.evaluation.evaluation_runner

Only imports dataset/samples here so that importing the package stays light;
the RAGAS-dependent modules import on first use via ``evaluation_runner``.
"""

from app.evaluation.dataset import GoldenDataset, GoldenQuestion, load_golden_dataset
from app.evaluation.samples import PipelineRunResult, build_single_turn_sample

__all__ = [
    "GoldenDataset",
    "GoldenQuestion",
    "PipelineRunResult",
    "build_single_turn_sample",
    "load_golden_dataset",
]
