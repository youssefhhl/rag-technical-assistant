"""Tests for the RAGAS evaluator wiring (Phase 9.1).

External LLM calls are mocked/faked: no Groq API key is required. The real
metrics are only checked for construction; scoring paths use fake metric
objects to exercise aggregation and error capture.
"""

from typing import Any

import asyncio

import pytest

from app.core.config import Settings
from app.evaluation import ragas_compat  # noqa: F401  # shim before ragas
from app.evaluation import evaluator as evaluator_module
from app.evaluation.evaluator import RagasEvaluator, groq_model_id


def make_settings(**overrides) -> Settings:
    values = {"_env_file": None, "llm_model": "groq/openai/gpt-oss-120b"}
    values.update(overrides)
    return Settings(**values)


class FakeMetric:
    """Duck-typed ragas metric for scoring-path tests."""

    def __init__(self, name: str, value: Any, raises: Exception | None = None):
        self.name = name
        self._value = value
        self._raises = raises

    async def single_turn_ascore(self, sample: Any, timeout: float | None = None) -> float:
        if self._raises:
            raise self._raises
        return self._value


def make_evaluator(metrics, settings=None) -> RagasEvaluator:
    evaluator = RagasEvaluator(
        metrics=metrics,
        settings=settings or make_settings(),
    )
    return evaluator


def any_sample():
    from app.evaluation.samples import build_single_turn_sample

    return build_single_turn_sample("q", "r", ["c"], "ref", ["ref-ctx"])


# -- score collection -----------------------------------------------------------


def test_scores_are_collected_per_metric():
    metrics = [
        FakeMetric("faithfulness", 0.9),
        FakeMetric("answer_relevancy", 0.8),
    ]
    evaluator = make_evaluator(metrics)
    outcome = asyncio.run(evaluator.score_sample(any_sample()))
    assert outcome.scores == {"faithfulness": 0.9, "answer_relevancy": 0.8}
    assert outcome.errors == {}


def test_metric_failure_is_none_not_a_fabricated_score():
    metrics = [
        FakeMetric("faithfulness", 0.9),
        FakeMetric("context_precision", None, raises=RuntimeError("LLM down")),
    ]
    evaluator = make_evaluator(metrics)
    outcome = asyncio.run(evaluator.score_sample(any_sample()))
    assert outcome.scores["faithfulness"] == 0.9
    assert outcome.scores["context_precision"] is None
    assert "LLM down" in outcome.errors["context_precision"]


def test_nan_score_is_reported_as_failure():
    metrics = [FakeMetric("answer_relevancy", float("nan"))]
    evaluator = make_evaluator(metrics)
    outcome = asyncio.run(evaluator.score_sample(any_sample()))
    assert outcome.scores["answer_relevancy"] is None
    assert "NaN" in outcome.errors["answer_relevancy"]


# -- judge model configuration --------------------------------------------------


def test_groq_model_id_strips_litellm_prefix():
    settings = make_settings(llm_model="groq/openai/gpt-oss-120b")
    assert groq_model_id(settings) == "openai/gpt-oss-120b"


def test_groq_model_id_accepts_override():
    settings = make_settings()
    assert groq_model_id(settings, "groq/llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"


def test_non_groq_model_is_a_clear_configuration_error():
    settings = make_settings(llm_model="openai/gpt-4o")
    with pytest.raises(ValueError, match="routed through Groq"):
        groq_model_id(settings)


def test_missing_api_key_raises_clear_error(monkeypatch):
    settings = make_settings(groq_api_key=None)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        evaluator_module.build_judge_llm(settings)


def test_evaluator_model_falls_back_to_llm_model():
    settings = make_settings(evaluator_model=None)
    assert groq_model_id(settings) == "openai/gpt-oss-120b"


def test_metrics_module_exports_exactly_the_four_selected_metrics():
    assert evaluator_module.METRIC_NAMES == (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )


class FlakyMetric:
    """Returns NaN first, a real score on the second attempt."""

    def __init__(self, name: str, final: float):
        self.name = name
        self.calls = 0
        self._final = final

    async def single_turn_ascore(self, sample: Any, timeout: float | None = None) -> float:
        self.calls += 1
        return float("nan") if self.calls == 1 else self._final


def test_nan_metric_gets_one_retry_then_scored():
    import asyncio

    metric = FlakyMetric("faithfulness", 0.75)
    evaluator = make_evaluator([metric])
    outcome = asyncio.run(evaluator.score_sample(any_sample()))
    assert metric.calls == 2
    assert outcome.scores["faithfulness"] == 0.75
    assert outcome.errors == {}


def test_nan_metric_fails_after_retry_exhausted():
    import asyncio

    evaluator = make_evaluator([FakeMetric("faithfulness", float("nan"))])
    outcome = asyncio.run(evaluator.score_sample(any_sample()))
    assert outcome.scores["faithfulness"] is None
    assert "NaN" in outcome.errors["faithfulness"]
