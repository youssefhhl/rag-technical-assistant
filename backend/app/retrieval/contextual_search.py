"""Contextual retrieval: history -> query rewriting -> hybrid search (Phase 6).

Glues the pieces without touching their algorithms:

1. load recent conversation history (``ConversationRepository``);
2. rewrite a context-dependent question into a standalone query
   (``QueryRewriter``, LiteLLM/Groq — skipped when there is no history);
3. run the existing hybrid search with the query (vector+BM25+RRF+rerank);
4. the retrieval answer keeps the **original question** untouched — answer
   generation happens in a later phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.database.repository import ConversationRepository
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextualOutcome:
    """Rewritten-query retrieval with full traceability."""

    question: str  # the original question (kept for answer generation)
    rewritten_query: str  # what hybrid search actually received
    history_used: int
    hits: list[Any] = field(default_factory=list)


class ContextualSearchService:
    """Retrieval aware of the conversation: rewrite first, then hybrid search."""

    def __init__(
        self,
        repository: ConversationRepository,
        query_rewriter: QueryRewriter,
        hybrid_service: HybridSearchService,
        *,
        window: int | None = None,
    ) -> None:
        self._repository = repository
        self._query_rewriter = query_rewriter
        self._hybrid_service = hybrid_service
        self._window = window or repository.default_window

    def search(
        self, conversation_id: str, question: str, top_k: int | None = None
    ) -> ContextualOutcome:
        history = self._repository.recent_messages(conversation_id)
        try:
            rewritten_query = self._query_rewriter.rewrite(question, history)
        except Exception as exc:  # noqa: BLE001 - rewriting must never break retrieval
            logger.warning("Rewriter raised (%s) - using the original question", exc)
            rewritten_query = question
        hits = self._hybrid_service.search(rewritten_query, top_k=top_k)
        logger.info(
            "Contextual search: history_used=%d rewritten=%r",
            len(history),
            rewritten_query,
        )
        return ContextualOutcome(
            question=question,
            rewritten_query=rewritten_query,
            history_used=len(history),
            hits=hits,
        )
