"""Tests for the citation verification (Phase 10) - no service, no LLM."""

from __future__ import annotations

from app.retrieval.citations import (
    REASON_INVALID_CHUNK_INDEX,
    REASON_MALFORMED,
    REASON_MISSING_SOURCE,
    REASON_MISSING_TITLE,
    REASON_UNKNOWN_DOCUMENT,
    CitationVerifier,
)
from app.retrieval.hybrid_search import HybridHit


def make_hit(index: int = 3, source: str = "https://fastapi.tiangolo.com/") -> HybridHit:
    return HybridHit(
        content="FastAPI is a web framework.",
        metadata={"source": source, "title": "FastAPI", "chunk_index": index},
        rrf_score=0.03,
    )


def citation(index: int = 3, **overrides) -> dict:
    data = {
        "title": "FastAPI > Performance",
        "source_url": "https://fastapi.tiangolo.com/",
        "chunk_index": str(index),
    }
    data.update(overrides)
    return data


# -- valid citations -------------------------------------------------------------


def test_citation_matching_a_retrieved_document_is_kept():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(3)])

    assert verdict.valid == [citation(3)]
    assert verdict.rejected == []


def test_several_valid_citations_are_kept_in_order():
    hits = [make_hit(3), make_hit(7, source="https://fastapi.tiangolo.com/async")]
    verifier = CitationVerifier(hits)

    verdict = verifier.verify(
        [citation(3), citation(7, source_url="https://fastapi.tiangolo.com/async")]
    )

    assert len(verdict.valid) == 2
    assert verdict.rejected_count == 0


def test_integer_chunk_index_is_accepted():
    """RAGService stores the chunk index as a string."""
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(3, chunk_index=3)])

    assert len(verdict.valid) == 1


def test_no_citation_gives_an_empty_verdict():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([])

    assert verdict.valid == []
    assert verdict.rejected == []


# -- invalid citations ------------------------------------------------------------


def test_citation_to_a_document_that_was_not_retrieved_is_rejected():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(99)])

    assert verdict.valid == []
    assert verdict.rejected[0].reason == REASON_UNKNOWN_DOCUMENT


def test_citation_to_an_unknown_source_is_rejected():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(3, source_url="https://elsewhere.test/x")])

    assert verdict.rejected[0].reason == REASON_UNKNOWN_DOCUMENT


def test_citation_without_source_metadata_is_rejected():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(3, source_url="")])

    assert verdict.rejected[0].reason == REASON_MISSING_SOURCE


def test_citation_without_title_metadata_is_rejected():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(3, title="")])

    assert verdict.rejected[0].reason == REASON_MISSING_TITLE


def test_citation_with_an_unusable_chunk_index_is_rejected():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(3, chunk_index="not-a-number")])

    assert verdict.rejected[0].reason == REASON_INVALID_CHUNK_INDEX


def test_malformed_citation_is_rejected():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify(["not a citation"])

    assert verdict.rejected[0].reason == REASON_MALFORMED


def test_valid_and_invalid_citations_are_separated():
    verifier = CitationVerifier([make_hit(3)])

    verdict = verifier.verify([citation(3), citation(42), citation(3, title="")])

    assert len(verdict.valid) == 1
    assert verdict.rejected_count == 2
    assert verdict.rejected[0].reason == REASON_UNKNOWN_DOCUMENT
    assert verdict.rejected[1].reason == REASON_MISSING_TITLE


def test_a_hit_without_source_metadata_is_never_cited():
    """A chunk whose metadata has no source cannot back a citation."""
    hit = HybridHit(content="c", metadata={"title": "FastAPI"}, rrf_score=0.03)
    verifier = CitationVerifier([hit])

    verdict = verifier.verify([citation(0, source_url="")])

    assert verdict.rejected[0].reason == REASON_MISSING_SOURCE


def test_rejected_citation_keeps_its_original_content():
    verifier = CitationVerifier([make_hit(3)])
    bad = citation(99)

    verdict = verifier.verify([bad])

    assert verdict.rejected[0].citation == bad
