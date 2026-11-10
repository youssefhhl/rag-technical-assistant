"""Tests for the Chat API: metadata mapping, citation extraction, endpoint.

All external services (Qdrant, BM25, Groq) are mocked here - these are unit
tests.  The real end-to-end run against the live services lives in
``tests/test_chat_integration.py``.
"""

from __future__ import annotations

import pytest

from app.retrieval.context_builder import (
    ContextBuilder,
    resolve_citation_title,
    resolve_chunk_index,
    resolve_document_title,
    resolve_section,
    resolve_source_url,
)
from app.retrieval.hybrid_search import HybridHit
from app.retrieval.rag_service import RAGService


# -- metadata mapping ---------------------------------------------------------

#: The real payload shape stored in the Qdrant collection ``docs_chunks``.
QDRANT_METADATA = {
    "source": "https://fastapi.tiangolo.com/",
    "title": "FastAPI",
    "source_name": "fastapi",
    "document_type": "website",
    "heading_hierarchy": ["FastAPI", "Performance"],
    "chunk_index": 3,
}


def test_document_title_is_read_from_metadata():
    assert resolve_document_title(QDRANT_METADATA) == "FastAPI"


def test_source_url_is_read_from_metadata():
    assert resolve_source_url(QDRANT_METADATA) == "https://fastapi.tiangolo.com/"


def test_chunk_index_is_read_from_metadata():
    assert resolve_chunk_index(QDRANT_METADATA) == 3


def test_section_is_the_heading_path():
    assert resolve_section(QDRANT_METADATA) == "FastAPI > Performance"


def test_citation_title_combines_document_title_and_section():
    assert resolve_citation_title(QDRANT_METADATA) == "FastAPI > Performance"


def test_citation_title_is_not_duplicated_when_section_starts_with_title():
    md = {"title": "FastAPI", "heading_hierarchy": ["FastAPI", "Requirements"]}
    assert resolve_citation_title(md) == "FastAPI > Requirements"


def test_citation_title_falls_back_to_document_title_without_headings():
    assert resolve_citation_title({"title": "FastAPI"}) == "FastAPI"


def test_citation_title_falls_back_to_section_without_title():
    md = {"heading_hierarchy": ["FastAPI", "Requirements"]}
    assert resolve_citation_title(md) == "FastAPI > Requirements"


def test_missing_metadata_never_produces_placeholder_title():
    """No "Doc" / "https://example.com" style defaults may leak through."""
    assert resolve_citation_title({}) == ""
    assert resolve_citation_title(None) == ""
    assert resolve_source_url({}) == ""
    assert resolve_source_url(None) == ""
    assert resolve_document_title({}) == ""


def test_chunk_index_tolerates_non_numeric_values():
    assert resolve_chunk_index({"chunk_index": "7"}) == 7
    assert resolve_chunk_index({"chunk_index": "not-a-number"}) == 0
    assert resolve_chunk_index({}) == 0


def test_section_accepts_a_plain_string_heading():
    assert resolve_section({"heading_hierarchy": "FastAPI"}) == "FastAPI"


# -- citation extraction -------------------------------------------------------


def make_hit(content: str, metadata: dict) -> HybridHit:
    return HybridHit(
        content=content,
        metadata=metadata,
        rrf_score=0.03,
        vector_rank=1,
        bm25_rank=2,
    )


def test_context_builder_maps_real_metadata():
    hits = [make_hit("FastAPI > Performance\n\nBenchmarks...", QDRANT_METADATA)]
    block, entries = ContextBuilder(hits, max_chunks=5).build()

    assert len(entries) == 1
    entry = entries[0]
    assert entry.title == "FastAPI > Performance"
    assert entry.source_url == "https://fastapi.tiangolo.com/"
    assert entry.chunk_index == 3
    assert entry.content.startswith("FastAPI > Performance")


def test_context_block_contains_real_chunk_text_and_url():
    hits = [make_hit("Benchmarks show FastAPI is fast.", QDRANT_METADATA)]
    block, _ = ContextBuilder(hits, max_chunks=5).build()

    assert "Benchmarks show FastAPI is fast." in block
    assert "https://fastapi.tiangolo.com/" in block
    assert "FastAPI > Performance" in block
    assert "Doc" not in block
    assert "example.com" not in block


def test_rag_service_citations_use_real_metadata(monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "FastAPI is fast.")

    hits = [
        make_hit("chunk one", dict(QDRANT_METADATA, chunk_index=0)),
        make_hit("chunk two", dict(QDRANT_METADATA, chunk_index=5)),
    ]
    answer = rag.generate("What is FastAPI?", hits)

    assert len(answer.citations) == 2
    assert answer.citations[0]["title"] == "FastAPI > Performance"
    assert answer.citations[0]["source_url"] == "https://fastapi.tiangolo.com/"
    assert answer.citations[0]["chunk_index"] == "0"
    assert answer.citations[1]["chunk_index"] == "5"


def test_rag_service_citations_contain_no_placeholders(monkeypatch):
    """Regression: citations used to be hard-coded to Doc / example.com."""
    rag = RAGService()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "FastAPI is fast.")

    hits = [make_hit("chunk", QDRANT_METADATA)]
    answer = rag.generate("What is FastAPI?", hits)

    for citation in answer.citations:
        assert citation["title"] != "Doc"
        assert citation["source_url"] != "https://example.com"
        assert citation["source_url"].startswith("http")


def test_rag_service_citations_are_empty_without_hits(monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "no info")
    answer = rag.generate("What is FastAPI?", [])
    assert answer.citations == []


# -- query rewriting (production classes) --------------------------------------


def test_query_rewriter_reads_object_style_response():
    """LiteLLM returns a pydantic Message - it must not be subscripted."""
    from app.retrieval.llm_content import extract_message_content

    class Message:
        content = "how do I deploy a FastAPI app?"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    assert extract_message_content(Response()) == "how do I deploy a FastAPI app?"


def test_query_rewriter_reads_dict_style_response():
    from app.retrieval.llm_content import extract_message_content

    response = type("R", (), {"choices": [{"message": {"content": "rewritten"}}]})()
    assert extract_message_content(response) == "rewritten"


def test_query_rewriter_reads_legacy_text_response():
    from app.retrieval.llm_content import extract_message_content

    response = type("R", (), {"choices": [{"text": "legacy"}]})()
    assert extract_message_content(response) == "legacy"


def test_query_rewriter_handles_empty_choices():
    from app.retrieval.llm_content import extract_message_content

    response = type("R", (), {"choices": []})()
    assert extract_message_content(response) == ""


def test_production_rewriter_sees_the_full_windowed_history(tmp_path):
    """The real ContextualSearchService + QueryRewriter must pass the whole window."""
    from sqlalchemy import create_engine

    from app.database import init_database
    from app.database.repository import ConversationRepository
    from app.retrieval.contextual_search import ContextualSearchService
    from app.retrieval.query_rewriter import QueryRewriter

    class FakeHybrid:
        def search(self, query, top_k=None):
            return ["hit"]

    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    init_database(engine)
    repo = ConversationRepository(engine)
    for i in range(15):
        repo.append_message("conv-1", "user", f"m{i}")

    seen: list = []
    rewriter = QueryRewriter(completion=lambda messages: seen.append(messages) or "rewritten")
    service = ContextualSearchService(repo, rewriter, FakeHybrid(), window=10)

    service.search("conv-1", "latest question?")

    prompt = seen[0][1]["content"]
    assert "m14" in prompt
    assert "m5" in prompt
    assert "m4" not in prompt


# -- chat endpoint (mocked services) ------------------------------------------


class FakeSearchService:
    """Stands in for ContextualSearchService with deterministic hits."""

    def __init__(self, hits: list) -> None:
        self._hits = hits
        self.questions: list[str] = []

    def search(self, conversation_id: str, question: str, top_k: int | None = None):
        self.questions.append(question)
        from app.retrieval.contextual_search import ContextualOutcome

        return ContextualOutcome(
            question=question,
            rewritten_query=question,
            history_used=0,
            hits=self._hits,
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient whose repository, retrieval and LLM are replaced by fakes."""
    import importlib

    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine

    from app.database import init_database
    from app.main import create_app

    chat_router = importlib.import_module("app.api.chat_router")

    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    init_database(engine)
    from app.database.repository import ConversationRepository

    hits = [make_hit("FastAPI is a modern web framework.", QDRANT_METADATA)]

    def fake_search_service():
        return FakeSearchService(hits)

    class FakeRAGService:
        def generate(self, question, hits, top_k=None):
            from app.retrieval.rag_service import RAGAnswer

            builder = ContextBuilder(list(hits), max_chunks=5)
            _, entries = builder.build()
            citations = [
                {
                    "title": e.title,
                    "source_url": e.source_url,
                    "chunk_index": str(e.chunk_index),
                }
                for e in entries
            ]
            return RAGAnswer(answer="FastAPI is a modern web framework.", citations=citations)

    app = create_app()
    app.dependency_overrides[chat_router.get_conversation_repo] = lambda: ConversationRepository(
        engine
    )
    app.dependency_overrides[chat_router.get_search_service] = fake_search_service
    app.dependency_overrides[chat_router.get_rag_service] = lambda: FakeRAGService()

    return TestClient(app)


def test_chat_returns_200_with_answer_and_citations(client):
    response = client.post("/api/v1/chat", json={"message": "What is FastAPI?"})

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"]
    assert data["answer"] == "FastAPI is a modern web framework."
    assert data["citations"] == [
        {
            "title": "FastAPI > Performance",
            "source_url": "https://fastapi.tiangolo.com/",
            "chunk_index": "3",
        }
    ]


def test_chat_rejects_empty_message(client):
    response = client.post("/api/v1/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_creates_a_conversation_when_id_is_null(client):
    response = client.post(
        "/api/v1/chat", json={"message": "What is FastAPI?", "conversation_id": None}
    )
    assert response.status_code == 200
    assert response.json()["conversation_id"]


def test_chat_reuses_an_existing_conversation(client):
    first = client.post("/api/v1/chat", json={"message": "What is FastAPI?"}).json()
    second = client.post(
        "/api/v1/chat",
        json={"message": "Tell me more", "conversation_id": first["conversation_id"]},
    ).json()
    assert second["conversation_id"] == first["conversation_id"]


def test_chat_surfaces_retrieval_failures(client):
    """A broken retrieval stack must not be hidden behind a fake answer."""
    import importlib

    chat_router = importlib.import_module("app.api.chat_router")

    class Failing:
        def search(self, *args, **kwargs):
            raise RuntimeError("qdrant unreachable")

    client.app.dependency_overrides[chat_router.get_search_service] = lambda: Failing()

    response = client.post("/api/v1/chat", json={"message": "What is FastAPI?"})

    assert response.status_code == 503
    assert "qdrant unreachable" in response.json()["detail"]
