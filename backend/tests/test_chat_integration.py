"""Real end-to-end integration test for ``POST /api/v1/chat``.

Unlike ``tests/test_chat_api.py`` (which mocks every external service) this
module talks to the **live** infrastructure:

* PostgreSQL  (conversation storage)
* Qdrant      (dense retrieval, collection ``docs_chunks``)
* BM25 index  (sparse retrieval, ``data/indexes/bm25/bm25_index.pkl``)
* Groq        (answer generation, ``settings.llm_model``)

The tests are skipped when a required service is unreachable so the unit-test
suite stays runnable offline.  Run them explicitly with::

    pytest tests/test_chat_integration.py -v
"""

from __future__ import annotations

import socket

import pytest

from app.core.config import get_settings

QUESTION = "What is FastAPI and what are its main advantages?"


def _service_available(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def qdrant_available() -> bool:
    settings = get_settings()
    url = settings.qdrant_url.replace("http://", "").replace("https://", "")
    host, _, port = url.partition(":")
    return _service_available(host, int(port or 6333))


@pytest.fixture(scope="module")
def bm25_index_available() -> bool:
    from app.retrieval.bm25_indexing import load_bm25_index

    try:
        return load_bm25_index() is not None
    except (FileNotFoundError, ValueError):
        # The index was never built (or is incompatible) -> skip, do not error.
        return False


@pytest.fixture(scope="module")
def groq_available() -> bool:
    return bool(get_settings().groq_api_key)


@pytest.fixture(scope="module")
def live_client():
    """A TestClient wired to the real services (no dependency overrides)."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


@pytest.mark.integration
def test_chat_returns_a_real_grounded_answer(
    live_client, qdrant_available, bm25_index_available, groq_available
):
    """The endpoint must answer with grounded text and real citations."""
    if not (qdrant_available and bm25_index_available and groq_available):
        pytest.skip("qdrant / bm25 / groq not available")

    response = live_client.post("/api/v1/chat", json={"message": QUESTION, "conversation_id": None})

    assert response.status_code == 200, response.text

    data = response.json()
    assert data["conversation_id"], "a conversation id must be returned"
    answer = data["answer"]
    assert answer.strip(), "the answer must not be empty"
    assert "I don't have enough information" not in answer, (
        f"the retrieved corpus should answer this question, got: {answer!r}"
    )
    assert "FastAPI" in answer

    citations = data["citations"]
    assert citations, "citations must be present when documents are retrieved"
    for citation in citations:
        assert citation["title"], "citation title must not be empty"
        assert citation["title"] != "Doc", "placeholder title leaked into a citation"
        assert citation["source_url"].startswith("http"), "citation must carry a real URL"
        assert citation["source_url"] != "https://example.com", "placeholder URL leaked"
        assert citation["chunk_index"].isdigit()


@pytest.mark.integration
def test_qdrant_returns_relevant_fastapi_chunks(qdrant_available):
    """The dense engine must return real FastAPI documentation chunks."""
    if not qdrant_available:
        pytest.skip("qdrant not available")

    from app.api.chat_router import _make_qdrant_engine

    hits = _make_qdrant_engine().search(QUESTION, k=10)
    assert hits, "qdrant returned no hits"

    joined = " ".join(h.content.lower() for h in hits)
    assert "fastapi" in joined, "qdrant hits do not mention FastAPI"

    for hit in hits:
        assert hit.content.strip(), "a qdrant hit has empty content"
        assert hit.source_url.startswith("http"), "qdrant hit has no source url"
        assert hit.metadata.get("title"), "qdrant hit has no title"
        assert isinstance(hit.metadata.get("chunk_index"), int)


@pytest.mark.integration
def test_bm25_returns_relevant_fastapi_chunks(bm25_index_available):
    """The sparse engine must return real FastAPI documentation chunks."""
    if not bm25_index_available:
        pytest.skip("bm25 index not available")

    from app.api.chat_router import _make_bm25_engine

    hits = _make_bm25_engine().search(QUESTION, k=10)
    assert hits, "bm25 returned no hits"

    joined = " ".join(h.content.lower() for h in hits)
    assert "fastapi" in joined, "bm25 hits do not mention FastAPI"

    for hit in hits:
        assert hit.content.strip(), "a bm25 hit has empty content"
        assert hit.metadata.get("source", "").startswith("http")
        assert hit.metadata.get("title")


@pytest.mark.integration
def test_hybrid_pipeline_preserves_real_metadata(
    qdrant_available, bm25_index_available
):
    """RRF + reranking must keep the real content and metadata of every chunk."""
    if not (qdrant_available and bm25_index_available):
        pytest.skip("qdrant / bm25 not available")

    from app.api.chat_router import (
        _make_bm25_engine,
        _make_qdrant_engine,
        _make_reranker,
    )
    from app.retrieval.hybrid_search import HybridSearchService

    hybrid = HybridSearchService(
        vector_search=_make_qdrant_engine(),
        bm25_search=_make_bm25_engine(),
        reranker=_make_reranker(),
    )
    hits = hybrid.search(QUESTION, top_k=5)
    assert len(hits) == 5

    for hit in hits:
        assert hit.content.strip(), "a fused hit has empty content"
        assert "HybridHit(" not in hit.content, "the object repr leaked into the content"
        assert hit.metadata.get("source", "").startswith("http")
        assert hit.metadata.get("title")
        assert isinstance(hit.metadata.get("chunk_index"), int)


@pytest.mark.integration
def test_context_block_contains_real_chunk_text(
    qdrant_available, bm25_index_available
):
    """The prompt handed to Groq must contain the retrieved chunk text."""
    if not (qdrant_available and bm25_index_available):
        pytest.skip("qdrant / bm25 not available")

    from app.api.chat_router import (
        _make_bm25_engine,
        _make_qdrant_engine,
        _make_reranker,
    )
    from app.retrieval.context_builder import ContextBuilder
    from app.retrieval.hybrid_search import HybridSearchService

    hybrid = HybridSearchService(
        vector_search=_make_qdrant_engine(),
        bm25_search=_make_bm25_engine(),
        reranker=_make_reranker(),
    )
    hits = hybrid.search(QUESTION, top_k=5)
    block, entries = ContextBuilder(list(hits), max_chunks=5).build()

    assert block.strip(), "the context block must not be empty"
    assert "example.com" not in block, "placeholder URL leaked into the context"
    for entry in entries:
        assert entry.content.strip(), "a context entry has empty content"
        assert entry.content[:80] in block, "the chunk text is missing from the context"
        assert entry.source_url.startswith("http")
