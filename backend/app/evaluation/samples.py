"""Mapping from pipeline outputs + golden questions to RAGAS samples (Phase 9.1).

The separation below is the core integrity rule of the evaluation:

- ``retrieved_contexts`` = the chunks actually fetched by the production
  hybrid retrieval, after reranking, exactly as handed to the generator;
- ``reference`` / ``reference_contexts`` = the manually verified golden
  answer / excerpts from the official docs.

Golden reference material is never used to fake ``retrieved_contexts``, and
retrieved chunks are never copied into the reference fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.evaluation import ragas_compat  # noqa: F401  # shim must come before ragas
from ragas.dataset_schema import SingleTurnSample


def build_single_turn_sample(
    question: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
    reference_contexts: list[str],
) -> SingleTurnSample:
    """Map one evaluation sample to the RAGAS ``SingleTurnSample`` schema.

    ``retrieved_contexts`` carries the real pipeline output; ``reference`` and
    ``reference_contexts`` carry the golden material used by
    ContextPrecision / ContextRecall only.
    """
    return SingleTurnSample(
        user_input=question,
        response=response,
        retrieved_contexts=list(retrieved_contexts),
        reference=reference,
        reference_contexts=list(reference_contexts),
    )


@dataclass
class PipelineRunResult:
    """What one production pipeline invocation produced (plus metrics)."""

    question: str
    response: str
    retrieved_contexts: list[str] = field(default_factory=list)
    latency_seconds: float | None = None
    error: str | None = None
    # Only for unanswerable questions: did the pipeline abstain?
    abstained: bool | None = None


def responses_from_errors(
    question: str, error: str, latency_seconds: float | None = None
) -> PipelineRunResult:
    """A recorded pipeline failure (no fabricated answer text)."""
    return PipelineRunResult(
        question=question,
        response="",
        latency_seconds=latency_seconds,
        error=error,
    )
