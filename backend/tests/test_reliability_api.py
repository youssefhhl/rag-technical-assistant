"""Tests for the Phase 10 reliability layer at the Chat API level.

Everything is mocked: no Qdrant, no BM25, no Groq. The endpoint contract
(``answer`` + ``citations``) must stay exactly the same.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.database import init_database
from app.database.repository import ConversationRepository
from app.retrieval.rag_service import RAGAnswer
from app.retrieval.reranker import RerankHit
from app.retrieval.reliability import ABSTENTION_MESSAGE, ReliabilityPipeline


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


class FakeSearchService:
    def __init__(self, hits) -> None:
        self._hits = hits

    def search(self, conversation_id, question, top_k=None):
        from app.retrieval.contextual_search import ContextualOutcome

        return ContextualOutcome(
            question=question,
            rewritten_query=question,
            history_used=0,
            hits=self._hits,
        )


class FakeHybrid:
    def __init__(self, hits=None) -> None:
        self.hits = hits or []
        self.queries: list[str] = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        return list(self.hits)


class FakeRewriter:
    def reformulate(self, question):
        return "reformulated question"


class SpyRAGService:
    """Records the generation calls (must stay empty on an abstention)."""

    def __init__(self, answer="FastAPI is a web framework.", citations=None) -> None:
        self.answer = answer
        self.citations = citations
        self.calls: list = []

    def generate(self, question, hits, top_k=None):
        self.calls.append((question, list(hits)))
        if self.citations is not None:
            return RAGAnswer(answer=self.answer, citations=list(self.citations))
        return RAGAnswer(
            answer=self.answer,
            citations=[
                {
                    "title": hit.metadata.get("title", ""),
                    "source_url": hit.metadata.get("source", ""),
                    "chunk_index": str(hit.metadata.get("chunk_index", 0)),
                }
                for hit in hits
            ],
        )


@pytest.fixture
def api(tmp_path):
    """A TestClient with mocked repository, retrieval, generation and CRAG."""

    def build(hits, rag_service, hybrid=None, rewriter=None):
        chat_router = importlib.import_module("app.api.chat_router")
        engine = create_engine(f"sqlite:///{tmp_path}/test.db")
        init_database(engine)
        repo = ConversationRepository(engine)

        pipeline = ReliabilityPipeline(
            hybrid_service=hybrid,
            rewriter=rewriter,
            rag_service=rag_service,
        )

        from app.main import create_app

        app = create_app()
        app.dependency_overrides[chat_router.get_conversation_repo] = lambda: repo
        app.dependency_overrides[chat_router.get_search_service] = lambda: FakeSearchService(hits)
        app.dependency_overrides[chat_router.get_rag_service] = lambda: rag_service
        app.dependency_overrides[chat_router.get_reliability_pipeline] = lambda: pipeline
        return TestClient(app), repo

    return build


def test_api_answers_a_relevant_question(api):
    rag = SpyRAGService()

    client, _repo = api([scored_hit(8.0)], rag)
    response = client.post("/api/v1/chat", json={"message": "What is FastAPI?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "FastAPI is a web framework."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source_url"] == "https://fastapi.tiangolo.com/"
    assert len(rag.calls) == 1


def test_api_abstains_on_an_irrelevant_context(api):
    rag = SpyRAGService()

    client, _repo = api([scored_hit(-9.0)], rag)
    response = client.post("/api/v1/chat", json={"message": "Unrelated question?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == ABSTENTION_MESSAGE
    assert data["citations"] == []
    assert rag.calls == []  # the generator was never called


def test_api_abstains_on_an_empty_retrieval(api):
    rag = SpyRAGService()

    client, _repo = api([], rag)
    response = client.post("/api/v1/chat", json={"message": "What is FastAPI?"})

    assert response.status_code == 200
    assert response.json()["answer"] == ABSTENTION_MESSAGE
    assert rag.calls == []


def test_api_uses_the_corrective_retrieval_before_abstaining(api):
    rag = SpyRAGService()
    hybrid = FakeHybrid(hits=[scored_hit(7.0)])

    client, _repo = api([scored_hit(-9.0)], rag, hybrid=hybrid, rewriter=FakeRewriter())
    response = client.post("/api/v1/chat", json={"message": "What is FastAPI?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "FastAPI is a web framework."
    assert hybrid.queries == ["reformulated question"]
    assert len(rag.calls) == 1


def test_api_filters_out_invalid_citations(api):
    rag = SpyRAGService(
        answer="FastAPI is a web framework.",
        citations=[
            {
                "title": "FastAPI",
                "source_url": "https://fastapi.tiangolo.com/",
                "chunk_index": "0",
            },
            {
                "title": "Invented",
                "source_url": "https://invented.test/doc",
                "chunk_index": "9",
            },
        ],
    )

    client, _repo = api([scored_hit(8.0)], rag)
    response = client.post("/api/v1/chat", json={"message": "What is FastAPI?"})

    citations = response.json()["citations"]
    assert len(citations) == 1
    assert citations[0]["source_url"] == "https://fastapi.tiangolo.com/"


def test_api_still_persists_the_abstention_in_the_history(api):
    rag = SpyRAGService()

    client, repo = api([scored_hit(-9.0)], rag)
    response = client.post("/api/v1/chat", json={"message": "Unrelated question?"})

    conversation_id = response.json()["conversation_id"]
    messages = repo.recent_messages(conversation_id, limit=10)
    roles = [message.role for message in messages]
    assert roles == ["user", "assistant"]
    assert messages[1].content == ABSTENTION_MESSAGE
