"""Integration: conversation history -> rewriting -> hybrid search (mocked LLM+hybrid)."""

import pytest

from app.core.config import Settings
from app.database import ConversationRepository, create_engine_from_url, init_database
from app.database.models import Message
from app.retrieval.contextual_search import ContextualSearchService
from app.retrieval.query_rewriter import QueryRewriter


def make_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        scrape_request_delay_seconds=0,
    )


class RecordingHybridService:
    """Stands in for HybridSearchService.search (algorithms untouched)."""

    def __init__(self) -> None:
        self.received_queries: list[str] = []
        self.received_top_k: list[int | None] = []

    def search(self, query, top_k=None):
        self.received_queries.append(query)
        self.received_top_k.append(top_k)
        return [f"hit-for-{query}-{n}" for n in range(3)]


class RecordingRewriter:
    """Mimics QueryRewriter: history empty -> question unchanged, no rewrite."""

    def __init__(self, rewritten: str) -> None:
        self.rewritten = rewritten
        self.seen_histories: list[list] = []

    def rewrite(self, question, history):
        self.seen_histories.append(list(history))
        return self.rewritten if history else question


@pytest.fixture()
def repo(tmp_path):
    settings = make_settings(tmp_path)
    engine = create_engine_from_url(settings.database_url)
    init_database(engine)
    return ConversationRepository(engine, settings=settings)


def seed_history(repo: ConversationRepository):
    repo.append_message("conv-1", "user", "what is FastAPI?")
    repo.append_message("conv-1", "assistant", "A modern web framework.")


def test_pipeline_rewrites_then_searches_with_rewritten_query(repo):
    seed_history(repo)
    hybrid = RecordingHybridService()
    rewriter = RecordingRewriter("how do I deploy a FastAPI app?")
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    outcome = service.search("conv-1", "and deployment?")

    # hybrid received the rewritten query, not the original question
    assert hybrid.received_queries == ["how do I deploy a FastAPI app?"]
    assert outcome.rewritten_query == "how do I deploy a FastAPI app?"
    # original question preserved for answer generation
    assert outcome.question == "and deployment?"
    assert outcome.history_used == 2
    assert outcome.hits == [f"hit-for-{outcome.rewritten_query}-{n}" for n in range(3)]


def test_without_history_hybrid_receives_original_question(repo):
    hybrid = RecordingHybridService()
    rewriter = RecordingRewriter("should never be used")  # would fail loudly if called
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    outcome = service.search("conv-unknown", "how are routes defined?")

    assert hybrid.received_queries == ["how are routes defined?"]
    assert outcome.rewritten_query == "how are routes defined?"
    assert outcome.history_used == 0
    assert rewriter.seen_histories == [[]]


def test_histories_passed_to_the_rewriter_are_recent_window(repo, tmp_path):
    for i in range(15):
        repo.append_message("conv-1", "user", f"m{i}")
    hybrid = RecordingHybridService()
    seen = []
    rewriter = QueryRewriter(completion=lambda messages: seen.append(messages) or "rewritten")
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    service.search("conv-1", "latest question?")

    # the rewriter must see exactly the repository's windowed history
    prompt = seen[0][1]["content"]
    assert "m14" in prompt and "m5" in prompt and "m4" not in prompt
    assert hybrid.received_queries == ["rewritten"]


def test_top_k_is_forwarded_to_hybrid_search(repo):
    seed_history(repo)
    hybrid = RecordingHybridService()
    rewriter = QueryRewriter(completion=lambda messages: "standalone query")
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    outcome = service.search("conv-1", "and deployment?", top_k=4)

    assert hybrid.received_top_k == [4]
    assert len(outcome.hits) == 3  # fake hybrid returns 3 hits regardless


def test_rewriter_failure_returns_original_question_but_searches(repo):
    seed_history(repo)

    class FailingRewriter:
        def rewrite(self, question, history):
            raise RuntimeError("boom")

    hybrid = RecordingHybridService()
    service = ContextualSearchService(repo, FailingRewriter(), hybrid, window=10)

    outcome = service.search("conv-1", "and deployment?")

    # QueryRewriter rewrites guarantee this; a direct raiser must not blow up search
    assert hybrid.received_queries == ["and deployment?"]
    assert outcome.rewritten_query == "and deployment?"
