"""Tests for the RAGAS sample mapping (Phase 9.1).

Core integrity rule verified here: retrieved contexts (real pipeline output)
and reference material (golden dataset) travel in strictly separate fields.
"""

from app.evaluation.samples import (
    PipelineRunResult,
    build_single_turn_sample,
    responses_from_errors,
)


def test_sample_has_all_ragas_fields():
    sample = build_single_turn_sample(
        question="What is FastAPI?",
        response="A modern web framework.",
        retrieved_contexts=["chunk-retrieved-1"],
        reference="expected answer",
        reference_contexts=["golden-excerpt"],
    )
    data = sample.model_dump()
    assert data["user_input"] == "What is FastAPI?"
    assert data["response"] == "A modern web framework."
    assert data["retrieved_contexts"] == ["chunk-retrieved-1"]
    assert data["reference"] == "expected answer"
    assert data["reference_contexts"] == ["golden-excerpt"]


def test_retrieved_and_reference_contexts_do_not_share_content():
    """Retrieved chunks are never the reference; reference never fake-retrieved."""
    sample = build_single_turn_sample(
        question="q",
        response="r",
        retrieved_contexts=["REAL-CHUNK", "REAL-CHUNK-2"],
        reference="ref",
        reference_contexts=["GOLDEN-EXCERPT"],
    )
    data = sample.model_dump()
    assert set(data["retrieved_contexts"]).isdisjoint(data["reference_contexts"])


def test_response_is_used_for_faithfulness_relevancy_metrics():
    # Faithfulness/AnswerRelevancy required columns per ragas 0.2.15.
    sample = build_single_turn_sample(
        question="q", response="r", retrieved_contexts=["c"],
        reference="ref", reference_contexts=[],
    )
    assert sample.response == "r"


def test_reference_fields_used_by_precision_recall():
    # ContextPrecision/ContextRecall read 'reference'; golden material never
    # leaks into retrieved_contexts.
    sample = build_single_turn_sample(
        question="q", response="r", retrieved_contexts=["c"],
        reference="ref", reference_contexts=["golden"],
    )
    assert sample.reference == "ref"
    assert all("golden" not in c.lower() for c in sample.retrieved_contexts)


def test_error_result_has_no_fabricated_response():
    result = responses_from_errors("q?", "ConnectionError: timeout")
    assert result.response == ""
    assert result.error == "ConnectionError: timeout"
    assert result.latency_seconds is None
    assert result.retrieved_contexts == []
    assert result.abstained is None


def test_pipeline_result_defaults_empty_contexts():
    result = PipelineRunResult(question="q", response="ok")
    assert result.retrieved_contexts == []
    assert result.error is None
    assert result.latency_seconds is None
    assert result.abstained is None
