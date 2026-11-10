"""Tests for the corrective retrieval (Phase 10, lightweight CRAG).

The query rewriter and the hybrid search are mocked: no LLM, no service.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.retrieval.corrective import CorrectiveRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import RerankHit
from app.retrieval.validation import RetrievalLabel, RetrievalValidator


def make_settings(**overrides) -> Settings:
    values = {"_env_file": None, "scrape_request_delay_seconds": 0}
    values.update(overrides)
    return Settings(**values)


def scored_hit(score: float, index: int = 0) -> RerankHit:
    return RerankHit(
        content=f"chunk {index}",
        metadata={"source": f"https://docs.test/{index}", "chunk_index": index},
        rerank_score=score,
    )


class FakeHybrid:
    """Records the queries it receives and returns canned hits."""

    def __init__(self, hits=None) -> None:
        self.hits = hits if hits is not None else []
        self.queries: list[str] = []
        self.top_ks: list = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        self.top_ks.append(top_k)
        return list(self.hits)


class FailingRewriter:
    def reformulate(self, question):
        raise RuntimeError("rewriter exploded")


# -- the reformulation itself -----------------------------------------------------


def test_reformulate_uses_the_injected_completion():
    seen: list = []
    rewriter = QueryRewriter(
        completion=lambda messages: seen.append(messages) or "FastAPI overview"
    )

    assert rewriter.reformulate("What is FastAPI?") == "FastAPI overview"
    assert "What is FastAPI?" in seen[0][1]["content"]


def test_reformulate_falls_back_to_the_question_on_llm_error():
    def boom(messages):
        raise RuntimeError("provider down")

    rewriter = QueryRewriter(completion=boom)

    assert rewriter.reformulate("What is FastAPI?") == "What is FastAPI?"


def test_reformulate_falls_back_to_the_question_on_empty_answer():
    rewriter = QueryRewriter(completion=lambda messages: "   ")

    assert rewriter.reformulate("What is FastAPI?") == "What is FastAPI?"


def test_reformulate_of_an_empty_question_is_empty():
    rewriter = QueryRewriter(completion=lambda messages: "should not be called")

    assert rewriter.reformulate("  ") == ""


# -- the corrective attempt ---------------------------------------------------------


def test_corrective_attempt_searches_once_with_the_reformulated_query():
    hybrid = FakeHybrid(hits=[scored_hit(7.0)])
    rewriter = QueryRewriter(completion=lambda messages: "FastAPI introduction")
    corrective = CorrectiveRetriever(hybrid, rewriter)

    attempt = corrective.attempt("What is FastAPI?", top_k=3)

    assert attempt is not None
    assert attempt.query == "FastAPI introduction"
    assert hybrid.queries == ["FastAPI introduction"]
    assert hybrid.top_ks == [3]
    assert attempt.assessment.label is RetrievalLabel.RELEVANT
    assert attempt.sufficient is True


def test_corrective_attempt_is_skipped_when_the_reformulation_is_identical():
    hybrid = FakeHybrid(hits=[scored_hit(7.0)])
    rewriter = QueryRewriter(completion=lambda messages: "What is FastAPI?")
    corrective = CorrectiveRetriever(hybrid, rewriter)

    attempt = corrective.attempt("What is FastAPI?")

    assert attempt is None
    assert hybrid.queries == []  # no useless second search


def test_corrective_attempt_is_skipped_when_the_reformulation_fails():
    hybrid = FakeHybrid()
    rewriter = QueryRewriter(completion=lambda messages: "")
    corrective = CorrectiveRetriever(hybrid, rewriter)

    assert corrective.attempt("What is FastAPI?") is None
    assert hybrid.queries == []


def test_corrective_attempt_without_hybrid_service_is_none():
    corrective = CorrectiveRetriever(
        None, QueryRewriter(completion=lambda messages: "other wording")
    )

    assert corrective.attempt("What is FastAPI?") is None


def test_corrective_attempt_without_rewriter_is_none():
    corrective = CorrectiveRetriever(FakeHybrid(), None)

    assert corrective.attempt("What is FastAPI?") is None


def test_corrective_attempt_keeps_an_irrelevant_result_irrelevant():
    hybrid = FakeHybrid(hits=[scored_hit(-9.0)])
    rewriter = QueryRewriter(completion=lambda messages: "another phrasing")
    corrective = CorrectiveRetriever(hybrid, rewriter)

    attempt = corrective.attempt("Unrelated question?")

    assert attempt is not None
    assert attempt.assessment.label is RetrievalLabel.IRRELEVANT
    assert attempt.sufficient is False


def test_corrective_attempt_uses_the_configured_thresholds():
    hybrid = FakeHybrid(hits=[scored_hit(3.0)])
    rewriter = QueryRewriter(completion=lambda messages: "another phrasing")
    validator = RetrievalValidator(settings=make_settings(retrieval_relevant_threshold=9.0))
    corrective = CorrectiveRetriever(hybrid, rewriter, validator=validator)

    attempt = corrective.attempt("Question?")

    assert attempt.assessment.label is RetrievalLabel.PARTIALLY_RELEVANT


def test_corrective_retriever_survives_a_raising_rewriter():
    """A broken rewriter must not break the pipeline."""
    hybrid = FakeHybrid()
    corrective = CorrectiveRetriever(hybrid, FailingRewriter())

    with pytest.raises(RuntimeError):
        corrective.attempt("Question?")
    assert hybrid.queries == []
