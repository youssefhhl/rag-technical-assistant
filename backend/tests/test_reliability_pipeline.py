"""Tests for the reliability pipeline (Phase 10) - everything is mocked."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.retrieval.hybrid_search import HybridHit
from app.retrieval.rag_service import RAGAnswer
from app.retrieval.reranker import RerankHit
from app.retrieval.reliability import ABSTENTION_MESSAGE, ReliabilityPipeline
from app.retrieval.validation import RetrievalLabel, RetrievalValidator


def make_settings(**overrides) -> Settings:
    values = {"_env_file": None, "scrape_request_delay_seconds": 0}
    values.update(overrides)
    return Settings(**values)


def scored_hit(score: float, index: int = 0) -> RerankHit:
    return RerankHit(
        content=f"chunk {index}",
        metadata={
            "source": "https://fastapi.tiangolo.com/",
            "title": "FastAPI",
            "chunk_index": index,
        },
        rerank_score=score,
    )


class FakeRAGService:
    """Records the generation calls; never calls a provider."""

    def __init__(self, answer: str = "FastAPI is a web framework.", citations=None) -> None:
        self.answer = answer
        self.citations = citations
        self.calls: list[tuple[str, list]] = []

    def generate(self, question, hits, top_k=None):
        self.calls.append((question, list(hits)))
        if self.citations is not None:
            return RAGAnswer(answer=self.answer, citations=list(self.citations))
        citations = [
            {
                "title": hit.metadata.get("title", ""),
                "source_url": hit.metadata.get("source", ""),
                "chunk_index": str(hit.metadata.get("chunk_index", 0)),
            }
            for hit in hits
        ]
        return RAGAnswer(answer=self.answer, citations=citations)


class FakeHybrid:
    def __init__(self, hits=None) -> None:
        self.hits = hits if hits is not None else []
        self.queries: list[str] = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        return list(self.hits)


class FakeRewriter:
    def __init__(self, query: str = "reformulated query") -> None:
        self.query = query
        self.calls: list[str] = []

    def reformulate(self, question):
        self.calls.append(question)
        return self.query


# -- the happy path ---------------------------------------------------------------


def test_relevant_context_is_answered_and_cited():
    rag = FakeRAGService()
    pipeline = ReliabilityPipeline(rag_service=rag)

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(8.0)])

    assert outcome.answer == "FastAPI is a web framework."
    assert outcome.label is RetrievalLabel.RELEVANT
    assert outcome.abstained is False
    assert outcome.corrected is False
    assert len(rag.calls) == 1
    assert len(outcome.citations) == 1


def test_partially_relevant_context_is_still_answered():
    rag = FakeRAGService()
    pipeline = ReliabilityPipeline(rag_service=rag)

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(-1.0)])

    assert outcome.abstained is False
    assert outcome.label is RetrievalLabel.PARTIALLY_RELEVANT
    assert len(rag.calls) == 1


# -- abstention -------------------------------------------------------------------


def test_irrelevant_context_abstains_without_generating():
    rag = FakeRAGService()
    pipeline = ReliabilityPipeline(rag_service=rag)

    outcome = pipeline.answer("Unrelated question?", [scored_hit(-9.0)])

    assert outcome.abstained is True
    assert outcome.answer == ABSTENTION_MESSAGE
    assert outcome.citations == []
    assert rag.calls == []  # the generator was never called


def test_empty_retrieval_abstains_without_generating():
    rag = FakeRAGService()
    pipeline = ReliabilityPipeline(rag_service=rag)

    outcome = pipeline.answer("What is FastAPI?", [])

    assert outcome.abstained is True
    assert outcome.answer == ABSTENTION_MESSAGE
    assert rag.calls == []


def test_abstention_message_matches_the_generator_phrase():
    """The API keeps a single stable abstention wording."""
    from app.evaluation.dataset import ABSTENTION_PLACEHOLDER

    assert ABSTENTION_MESSAGE == ABSTENTION_PLACEHOLDER


# -- corrective retrieval (CRAG) ----------------------------------------------------


def test_corrective_retrieval_answers_with_the_second_result():
    rag = FakeRAGService()
    hybrid = FakeHybrid(hits=[scored_hit(7.0)])
    rewriter = FakeRewriter("FastAPI introduction")
    pipeline = ReliabilityPipeline(
        hybrid_service=hybrid, rewriter=rewriter, rag_service=rag
    )

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(-9.0)])

    assert outcome.abstained is False
    assert outcome.corrected is True
    assert outcome.corrective_query == "FastAPI introduction"
    assert outcome.label is RetrievalLabel.RELEVANT
    assert hybrid.queries == ["FastAPI introduction"]
    assert rag.calls[0][1][0].rerank_score == pytest.approx(7.0)


def test_only_one_corrective_attempt_is_allowed():
    rag = FakeRAGService()
    hybrid = FakeHybrid(hits=[scored_hit(-9.0)])  # the correction is useless too
    rewriter = FakeRewriter("another phrasing")
    pipeline = ReliabilityPipeline(
        hybrid_service=hybrid, rewriter=rewriter, rag_service=rag
    )

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(-9.0)])

    assert outcome.abstained is True
    assert hybrid.queries == ["another phrasing"]  # exactly one corrective search
    assert rag.calls == []


def test_abstention_when_the_corrective_retrieval_is_also_irrelevant():
    rag = FakeRAGService()
    hybrid = FakeHybrid(hits=[scored_hit(-8.0)])
    rewriter = FakeRewriter("another phrasing")
    pipeline = ReliabilityPipeline(
        hybrid_service=hybrid, rewriter=rewriter, rag_service=rag
    )

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(-9.0)])

    assert outcome.abstained is True
    assert outcome.answer == ABSTENTION_MESSAGE
    assert outcome.corrective_query == "another phrasing"
    assert outcome.corrected is False  # the correction did not help


def test_crag_can_be_disabled():
    rag = FakeRAGService()
    hybrid = FakeHybrid(hits=[scored_hit(7.0)])
    rewriter = FakeRewriter("FastAPI introduction")
    pipeline = ReliabilityPipeline(
        hybrid_service=hybrid,
        rewriter=rewriter,
        rag_service=rag,
        crag_enabled=False,
    )

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(-9.0)])

    assert outcome.abstained is True
    assert hybrid.queries == []
    assert rewriter.calls == []


def test_the_original_question_is_never_modified():
    rag = FakeRAGService()
    hybrid = FakeHybrid(hits=[scored_hit(7.0)])
    rewriter = FakeRewriter("FastAPI introduction")
    pipeline = ReliabilityPipeline(
        hybrid_service=hybrid, rewriter=rewriter, rag_service=rag
    )

    pipeline.answer("What is FastAPI?", [scored_hit(-9.0)])

    # the generation always receives the original question
    assert rag.calls[0][0] == "What is FastAPI?"


# -- citation verification ----------------------------------------------------------


def test_invalid_citations_are_dropped_from_the_response():
    rag = FakeRAGService(
        answer="Answer.",
        citations=[
            {
                "title": "FastAPI",
                "source_url": "https://fastapi.tiangolo.com/",
                "chunk_index": "0",
            },
            {
                "title": "Invented",
                "source_url": "https://invented.test/doc",
                "chunk_index": "1",
            },
        ],
    )
    pipeline = ReliabilityPipeline(rag_service=rag)

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(8.0)])

    assert len(outcome.citations) == 1
    assert outcome.citations[0]["source_url"] == "https://fastapi.tiangolo.com/"
    assert len(outcome.rejected_citations) == 1
    assert "does not match" in outcome.rejected_citations[0]["reason"]


def test_citations_keep_their_format():
    rag = FakeRAGService(answer="Answer.")
    pipeline = ReliabilityPipeline(rag_service=rag)

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(8.0)])

    citation = outcome.citations[0]
    assert set(citation) == {"title", "source_url", "chunk_index"}
    assert isinstance(citation["chunk_index"], str)


def test_the_answer_is_kept_even_when_a_citation_is_invalid():
    rag = FakeRAGService(
        answer="FastAPI is a web framework.",
        citations=[{"title": "", "source_url": "", "chunk_index": "0"}],
    )
    pipeline = ReliabilityPipeline(rag_service=rag)

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(8.0)])

    assert outcome.answer == "FastAPI is a web framework."
    assert outcome.citations == []
    assert len(outcome.rejected_citations) == 1


# -- configuration -------------------------------------------------------------------


def test_the_validator_thresholds_come_from_the_settings():
    rag = FakeRAGService()
    pipeline = ReliabilityPipeline(
        rag_service=rag,
        validator=RetrievalValidator(settings=make_settings(retrieval_relevant_threshold=9.0)),
    )

    outcome = pipeline.answer("What is FastAPI?", [scored_hit(3.0)])

    assert outcome.label is RetrievalLabel.PARTIALLY_RELEVANT
    assert outcome.abstained is False


def test_pipeline_without_a_generator_fails_loudly():
    pipeline = ReliabilityPipeline(rag_service=None)

    try:
        pipeline.answer("What is FastAPI?", [scored_hit(8.0)])
    except RuntimeError as exc:
        assert "RAGService" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("a missing generator must be reported")


def test_unscored_hits_never_abstain():
    """A pure-RRF configuration stays permissive (no false abstention)."""
    rag = FakeRAGService()
    pipeline = ReliabilityPipeline(rag_service=rag)
    hits = [HybridHit(content="chunk", metadata={"source": "https://x.test/"}, rrf_score=0.03)]

    outcome = pipeline.answer("What is FastAPI?", hits)

    assert outcome.abstained is False
    assert len(rag.calls) == 1
