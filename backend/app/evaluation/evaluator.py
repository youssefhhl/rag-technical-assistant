"""RAGAS evaluator wiring (Phase 9.1): judge models + the four metrics.

Judge LLM: RAGAS talks to LLMs through LangChain chat models. The production
stack routes through LiteLLM/Gralla Groq, so the evaluator reuses the same
environment configuration: ``GROQ_API_KEY`` + ``LLM_MODEL`` (or
``EVALUATOR_MODEL`` override) from ``app.core.config.Settings``, pointed at
Groq's OpenAI-compatible endpoint via ``ChatOpenAI``. Temperature defaults to
0.0 for consistent judgments. No API key is ever hardcoded: with a missing
key the evaluator raises a clear error instead of silently returning fake
scores.

Embeddings: AnswerRelevancy needs an embedder. We reuse the production E5
service via the existing ``E5LangChainEmbeddings`` LangChain adapter.

Metrics (exactly four, RAGAS 0.2.15 API - per-sample ``single_turn_ascore``):

- Faithfulness        - response vs retrieved contexts (hallucination check)
- AnswerRelevancy     - response vs the user question
- ContextPrecision    - ranking quality, judged against the golden reference
- ContextRecall       - golden-reference coverage of the retrieved contexts

Scores are collected one sample at a time (sequential, not concurrent, to keep
Groq rate limits comfortable) and any metric failure is recorded explicitly;
it is never converted to a fabricated score.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dataclasses import dataclass, field

from app.evaluation import ragas_compat  # noqa: F401  # shim must load before ragas
from ragas.dataset_schema import SingleTurnSample  # noqa: F401
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.metrics.base import Metric

from app.core.config import Settings, get_settings
from app.embeddings.langchain_adapter import E5LangChainEmbeddings
from app.embeddings.service import get_embedding_service

logger = logging.getLogger(__name__)

#: The four selected metrics, in stable order.
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class ScoreOutcome:
    """Per-metric scores for one sample plus the exact failure causes."""

    scores: dict[str, float | None] = field(default_factory=dict)
    """Metric name -> real score, or None when the metric failed."""
    errors: dict[str, str] = field(default_factory=dict)
    """Metric name -> error message (metrics that completed have no entry)."""


def install_do_not_track() -> None:
    """Disable RAGAS anonymous telemetry (evaluation runs stay private)."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def groq_model_id(settings: Settings, override: str | None = None) -> str:
    """The model id Groq's OpenAI-compatible API expects (LiteLLM-prefix-free).

    Settings use the LiteLLM convention ``groq/<provider-model>``; the prefix
    is stripped for the direct API. A non-Groq model is a clear configuration
    error rather than a silent mis-routing.
    """
    model = (override or settings.evaluator_model or settings.llm_model).strip()
    if model.startswith("groq/"):
        return model[len("groq/") :]
    raise ValueError(
        f"Evaluator model {model!r} is not routed through Groq. The evaluation "
        "LLM reuses the production Groq configuration; set LLM_MODEL or "
        "EVALUATOR_MODEL to a 'groq/<model>' value."
    )


def build_judge_llm(settings: Settings | None = None, model: str | None = None):
    """ragas LLM wrapper for RAGAS, wired to Groq via existing settings.

    Raises a clear ``ValueError`` (not a silent failure) when no API key is
    configured.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    settings = settings or get_settings()
    if not settings.groq_api_key:
        raise ValueError(
            "RAGAS evaluation requires an LLM provider key. Set GROQ_API_KEY "
            "in the environment (the judge reuses the production Groq config)."
        )
    chat = ChatOpenAI(
        model=groq_model_id(settings, model),
        base_url=GROQ_BASE_URL,
        api_key=settings.groq_api_key,
        temperature=settings.evaluator_temperature,
    )
    # ragas metrics speak to BaseRagasLLM wrappers around LangChain models.
    return LangchainLLMWrapper(chat)


def build_judge_embeddings():
    """RAGAS embeddings reusing the production E5 service (LangChain adapter)."""
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(E5LangChainEmbeddings(get_embedding_service()))


def build_metrics(
    judge_llm: Any,
    judge_embeddings: Any,
    run_config: Any = None,
) -> list[Metric]:
    """Instantiate the four metrics with the judge models injected.

    ``run_config`` mirrors what ``ragas.evaluate`` does internally (LLM retries
    / timeouts before any scoring happens).
    """
    from ragas.run_config import RunConfig

    pairs = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
    }
    for metric in pairs.values():
        metric.init(run_config or RunConfig(max_workers=1))
    return [pairs[name] for name in METRIC_NAMES]


class RagasEvaluator:
    """Scores one sample at a time with explicit per-metric error capture."""

    def __init__(
        self,
        metrics: list[Metric] | None = None,
        evaluator_model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        install_do_not_track()
        self._settings = settings or get_settings()
        self.evaluator_model = evaluator_model
        self.metrics = metrics or build_metrics(
            judge_llm=build_judge_llm(self._settings, evaluator_model),
            judge_embeddings=build_judge_embeddings(),
        )

    async def score_sample(self, sample: SingleTurnSample) -> ScoreOutcome:
        """Score the four metrics for one sample.

        A failed metric is recorded as ``None`` with its exact error message
        captured - never faked with a number. A NaN result (the judge produced
        output ragas could not parse) gets one retry before being recorded as
        a failure: a first unparseable judgment is not evidence, and ragas
        cannot retry non-exception failures itself.
        """
        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for metric in self.metrics:
            key = getattr(metric, "name", "")
            scores[key] = None
            for attempt in range(self._nan_retries + 1):
                try:
                    value = await metric.single_turn_ascore(
                        sample, timeout=self._timeout
                    )
                except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                    message = f"{type(exc).__name__}: {exc}"
                    logger.warning("Metric %s failed: %s", key, message)
                    errors[key] = message
                    break  # exceptions are recorded immediately, no retry loop
                if value == value:  # not NaN: a real judgment
                    scores[key] = float(value)
                    errors.pop(key, None)  # clear the stale NaN note from attempt 1
                    break
                errors[key] = (
                    "NaN score - the judge output could not be parsed into a "
                    "judgment (warning from ragas)"
                )
                logger.warning(
                    "Metric %s returned NaN (attempt %d/%d)",
                    key, attempt + 1, self._nan_retries + 1,
                )
        return ScoreOutcome(scores=scores, errors=errors)

    @property
    def _timeout(self) -> float:
        return 120.0

    @property
    def _nan_retries(self) -> int:
        return 1
