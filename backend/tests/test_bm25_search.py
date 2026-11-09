"""Tests for BM25 lexical search (real rank_bm25 in-memory, no persistence)."""

import json

import pytest
from rank_bm25 import BM25Okapi

from app.retrieval.bm25_indexing import Bm25Document, Bm25IndexBundle, tokenize
from app.retrieval.bm25_search import Bm25Hit, Bm25SearchService


def make_bundle() -> Bm25IndexBundle:
    documents = [
        Bm25Document(
            content="FastAPI routes are defined with the router decorator.",
            metadata={"source": "https://docs.test/api", "title": "API", "chunk_index": 0},
        ),
        Bm25Document(
            content="Docker compose exposes ports for the Qdrant service.",
            metadata={"source": "https://docs.test/docker", "title": "Docker", "chunk_index": 1},
        ),
        Bm25Document(
            content="Label configuration lives under the settings key.",
            metadata={"source": "https://docs.test/label", "title": "Label", "chunk_index": 2},
        ),
    ]
    tokenized = [tokenize(document.content) for document in documents]
    return Bm25IndexBundle(
        documents=documents,
        bm25=BM25Okapi(tokenized),
        tokenized_corpus=tokenized,
        source_names=["test"],
        built_at="now",
    )


# -- ranking ------------------------------------------------------------------


def test_matching_query_ranks_relevant_chunk_first():
    service = Bm25SearchService(make_bundle())

    # Only the Docker chunk contains both query terms -> unambiguous top hit.
    hits = service.search("qdrant ports", k=3)

    assert len(hits) == 1
    assert hits[0].metadata["source"] == "https://docs.test/docker"
    assert hits[0].bm25_score > 0.0


def test_scores_are_descending():
    service = Bm25SearchService(make_bundle())

    hits = service.search("routes configuration", k=3)

    assert [hit.bm25_score for hit in hits] == sorted(
        (hit.bm25_score for hit in hits), reverse=True
    )


def test_top_k_limits_results():
    service = Bm25SearchService(make_bundle())

    hits = service.search("routes configuration docker", k=2)

    assert len(hits) <= 2


def test_no_matching_terms_returns_empty_list():
    service = Bm25SearchService(make_bundle())

    assert service.search("zzzqqq") == []


def test_terms_absent_from_corpus_return_empty():
    service = Bm25SearchService(make_bundle())

    assert service.search("of") == []  # token does not occur in any chunk


def test_empty_query_returns_empty_list():
    service = Bm25SearchService(make_bundle())

    assert service.search("") == []
    assert service.search("   ") == []


def test_k_below_one_returns_empty_list():
    service = Bm25SearchService(make_bundle())

    assert service.search("docker", k=0) == []


# -- metadata / hit shape ------------------------------------------------------


def test_hit_preserves_original_content_and_metadata():
    service = Bm25SearchService(make_bundle())

    hit = service.search("qdrant service ports", k=1)[0]

    assert isinstance(hit, Bm25Hit)
    assert hit.content == "Docker compose exposes ports for the Qdrant service."
    assert hit.metadata == {
        "source": "https://docs.test/docker",
        "title": "Docker",
        "chunk_index": 1,
    }
    assert hit.source_url == "https://docs.test/docker"
    assert hit.title == "Docker"
    assert hit.chunk_index == 1
    assert hit.heading_hierarchy == []


def test_format_contains_score_title_slug_and_content():
    service = Bm25SearchService(make_bundle())

    formatted = service.search("qdrant ports", k=1)[0].format()

    assert "bm25" in formatted
    assert "Docker" in formatted
    assert "chunk 1" in formatted
    assert "Qdrant service" in formatted
