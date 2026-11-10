"""Tests for conversation history storage (SQLAlchemy + SQLite)."""

import pytest
from typing import Any

from app.core.config import Settings
from app.database import (
    ConversationRepository,
    create_engine_from_url,
    init_database,
)
from app.retrieval.query_rewriter import QueryRewriter


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "database_url": f"sqlite:///{tmp_path / 'test.db'}",
        "conversation_history_window": 10,
        "scrape_request_delay_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture()
def repository(tmp_path):
    settings = make_settings(tmp_path)
    engine = create_engine_from_url(settings.database_url)
    init_database(engine)
    return ConversationRepository(engine, settings=settings), settings


# -- schema + writes ------------------------------------------------------------


def test_init_database_creates_tables(tmp_path):
    settings = make_settings(tmp_path)
    engine = create_engine_from_url(settings.database_url)
    init_database(engine)
    from sqlalchemy import inspect

    expected = {"conversations", "messages"}
    assert expected.issubset(set(inspect(engine).get_table_names()))


class TestRepository:
    def test_append_message_with_explicit_id_creates_conversation(self, repository):
        repo, _settings = repository

        message = repo.append_message("conv-1", "user", "how are routes defined?")

        assert message.id > 0
        assert message.role == "user"
        assert message.content == "how are routes defined?"
        history = repo.recent_messages("conv-1")
        assert [m.content for m in history] == ["how are routes defined?"]

    def test_append_message_auto_creates_conversation_when_id_is_none(
        self, repository,
    ):
        """REGRESSION: passing conversation_id=None must auto-create a conversation
        without raising NotNullViolation on the primary key."""
        repo, _settings = repository

        message = repo.append_message(None, "user", "hello")

        assert message.id > 0
        assert message.role == "user"
        assert message.content == "hello"
        history = repo.recent_messages(message.conversation_id)
        assert [m.content for m in history] == ["hello"]

    def test_stores_user_and_assistant_messages_in_order(self, repository):
        repo, _settings = repository

        repo.append_message("conv-1", "user", "what is FastAPI?")
        repo.append_message("conv-1", "assistant", "A modern web framework.")

        history = repo.recent_messages("conv-1")
        assert [(m.role, m.content) for m in history] == [
            ("user", "what is FastAPI?"),
            ("assistant", "A modern web framework."),
        ]

    def test_conversations_are_isolated(self, repository):
        repo, _settings = repository

        repo.append_message("conv-1", "user", "question one")
        repo.append_message("conv-2", "user", "question two")

        assert list(repo.recent_messages("conv-1"))[0].content == "question one"
        assert list(repo.recent_messages("conv-2"))[0].content == "question two"

    # -- window / ordering ------------------------------------------------------

    def test_recent_messages_returns_last_N_in_chronological_order(self, repository):
        repo, _settings = repository

        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            repo.append_message("conv-1", role, f"message {i}")

        history = repo.recent_messages("conv-1")

        assert [m.content for m in history] == [
            "message 0",
            "message 1",
            "message 2",
            "message 3",
            "message 4",
        ]

    def test_default_window_comes_from_settings(self, tmp_path):
        settings = make_settings(tmp_path, conversation_history_window=3)
        engine = create_engine_from_url(f"sqlite:///{tmp_path / 'test.db'}")
        init_database(engine)
        repo = ConversationRepository(engine, settings=settings)
        for i in range(9):
            repo.append_message("conv-1", "user", f"m{i}")

        history = repo.recent_messages("conv-1")

        assert [m.content for m in history] == ["m6", "m7", "m8"]

    def test_explicit_limit_overrides_window(self, repository):
        repo, _settings = repository
        for i in range(10):
            repo.append_message("conv-1", "user", f"m{i}")

        history = repo.recent_messages("conv-1", limit=2)

        assert [m.content for m in history] == ["m8", "m9"]

    def test_unknown_conversation_returns_empty_history(self, repository):
        repo, _settings = repository

        assert repo.recent_messages("nope") == []


# -- integration with auto-creation ----------------------------------------------------------------


def seed_history(repo):
    repo.append_message("conv-1", "user", "what is FastAPI?")
    repo.append_message("conv-1", "assistant", "A modern web framework.")


class RecordingHybridService:
    """Stands in for HybridSearchService.search (algorithms untouched)."""

    def __init__(self) -> None:
        self.received_queries: list[str] = []
        self.received_top_k: list[int] = []

    def search(self, conversation_id: str, query: str, top_k: int = 5) -> dict[str, list[str]]:
        """Record the query and return dummy hits."""
        self.received_queries.append(query)
        self.received_top_k.append(top_k)
        return {"hits": [f"hit-for-{query}-0"]}


class RecordingRewriter:
    """Mimics QueryRewriter: history empty -> question unchanged, no rewrite."""

    def __init__(self, rewritten: str | None = None) -> None:
        self.seen_histories: list[list[dict[str, str]]] = []
        self.rewritten = rewritten or "how are routes defined?"

    def rewrite(self, question: str, history: list[dict[str, str]]) -> str:
        self.seen_histories.append(history)
        # If history is empty, return the original question (matching QueryRewriter behavior)
        if not history:
            return question
        return self.rewritten or question


class ContextualSearchService:
    """Retrieval aware of the conversation: rewrite first, then hybrid search."""

    def __init__(
        self,
        repository: ConversationRepository,
        query_rewriter: QueryRewriter,
        hybrid_service: Any,
        *,
        window: int,
    ) -> None:
        self.repository = repository
        self.query_rewriter = query_rewriter
        self.hybrid_service = hybrid_service
        self.window = window

    def search(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        # Rewrite the query
        history = self.repository.recent_messages(conversation_id, limit=self.window)
        # The ORM Message objects are passed straight to the rewriter: the
        # production QueryRewriter reads ``message.role`` / ``message.content``.
        try:
            rewritten = self.query_rewriter.rewrite(query, history)
        except Exception:  # noqa: BLE001 - rewriter failure, keep original question
            rewritten = query

        # Perform hybrid search with rewritten query (may also raise)
        try:
            hits = self.hybrid_service.search(conversation_id, rewritten, top_k=top_k)
        except Exception:  # noqa: BLE001 - hybrid failure, keep original question
            hits = {"hits": []}

        return {
            "rewritten_query": rewritten,
            "question": query,
            "history_used": len(history),
            "hits": hits.get("hits", []),
        }


def test_pipeline_rewrites_then_searches_with_rewritten_query(repo):
    seed_history(repo)
    hybrid = RecordingHybridService()
    rewriter = RecordingRewriter("how do I deploy a FastAPI app?")
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    outcome = service.search("conv-1", "and deployment?")

    # hybrid received the rewritten query, not the original question
    assert hybrid.received_queries == ["how do I deploy a FastAPI app?"]
    assert outcome["rewritten_query"] == "how do I deploy a FastAPI app?"
    # original question preserved for answer generation
    assert outcome["question"] == "and deployment?"
    assert outcome["history_used"] == 2
    assert outcome["hits"] == ["hit-for-how do I deploy a FastAPI app?-0"]


def test_without_history_hybrid_receives_original_question(repo):
    hybrid = RecordingHybridService()
    rewriter = RecordingRewriter("should never be used")
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    outcome = service.search("conv-unknown", "how are routes defined?")

    assert hybrid.received_queries == ["how are routes defined?"]
    assert outcome["rewritten_query"] == "how are routes defined?"
    assert outcome["history_used"] == 0
    assert rewriter.seen_histories == [[]]


def test_histories_passed_to_the_rewriter_are_recent_window(repo):
    for i in range(15):
        repo.append_message("conv-1", "user", f"m{i}")
    hybrid = RecordingHybridService()
    seen_messages = []
    rewriter = QueryRewriter(completion=lambda messages: seen_messages.append(messages) or "rewritten")
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    service.search("conv-1", "latest question?")

    # the rewriter must see exactly the repository's windowed history
    # build_prompt_messages returns [system_msg, user_msg] where user_msg contains
    # the formatted conversation history
    if len(seen_messages) > 0 and len(seen_messages[0]) > 1:
        prompt = seen_messages[0][1].get("content", "")
    else:
        prompt = ""
    assert "m14" in prompt and "m5" in prompt and "m4" not in prompt, f"prompt={prompt}"
    assert hybrid.received_queries == ["rewritten"]


def test_top_k_is_forwarded_to_hybrid_search(repo):
    seed_history(repo)
    hybrid = RecordingHybridService()
    rewriter = QueryRewriter(completion=lambda messages: "standalone query")
    service = ContextualSearchService(repo, rewriter, hybrid, window=10)

    outcome = service.search("conv-1", "and deployment?", top_k=4)

    assert hybrid.received_top_k == [4]
    assert len(outcome["hits"]) == 1  # hybrid returns 1 hit


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
    assert outcome["rewritten_query"] == "and deployment?"