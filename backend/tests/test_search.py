"""Tests for the semantic search result formatting (Phase 3 vector search only)."""

import pytest
from langchain_core.documents import Document

from app.vector_store.search import SearchHit, SemanticSearchService


def make_hit(**overrides) -> SearchHit:
    values = {
        "content": "Ports allow services to be exposed on the host machine.",
        "source_url": "https://docs.test/compose",
        "title": "Docker Compose",
        "heading_hierarchy": ["Docker Compose", "Services", "Ports"],
        "similarity_score": 0.9153,
        "chunk_index": 4,
    }
    values.update(overrides)
    return SearchHit(**values)


def test_hit_format_includes_score_source_and_hierarchy():
    text = make_hit().format()
    assert "[0.9153] Docker Compose — Docker Compose > Services > Ports" in text
    assert "source: https://docs.test/compose (chunk 4)" in text
    assert "exposed on the host machine" in text


def test_hit_format_falls_back_to_title_without_hierarchy():
    text = make_hit(heading_hierarchy=[]).format()
    assert "[0.9153] Docker Compose — Docker Compose" in text


class FakeStore:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.calls = []

    def similarity_search_with_score(self, query: str, k: int = 5):
        if not query.strip():
            raise AssertionError("empty query reached the vector store")
        self.calls.append({"query": query, "k": k})
        return [(document, 0.5 + 0.1 * i) for i, document in enumerate(self._documents)]


def test_service_search_maps_fields():
    store = FakeStore(
        [
            Document(
                page_content="chunk text",
                metadata={
                    "source": "https://docs.test/a",
                    "title": "Doc title",
                    "heading_hierarchy": ["Doc title", "Section"],
                    "chunk_index": 2,
                },
            )
        ]
    )
    service = SemanticSearchService(store)  # type: ignore[arg-type]
    hits = service.search("how do ports work?", k=3)

    assert store.calls == [{"query": "how do ports work?", "k": 3}]
    assert len(hits) == 1
    assert hits[0].content == "chunk text"
    assert hits[0].source_url == "https://docs.test/a"
    assert hits[0].title == "Doc title"
    assert hits[0].heading_hierarchy == ["Doc title", "Section"]
    assert hits[0].chunk_index == 2
    assert hits[0].similarity_score == pytest.approx(0.5)


def test_service_empty_query_returns_no_hits():
    service = SemanticSearchService(FakeStore([]))  # type: ignore[arg-type]
    assert service.search("   ") == []
