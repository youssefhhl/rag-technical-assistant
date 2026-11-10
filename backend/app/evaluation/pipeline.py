"""Production-pipeline probe for the evaluation (Phase 9.1).

Reuses the exact service wiring of the live FastAPI chat endpoint
(``app.api.chat_router``) - real Qdrant + E5 dense retrieval, real BM25,
RRF fusion, Cross-Encoder reranking, real LiteLLM/Groq generation:

    Qdrant (dense)  \\                    /-> RAGService (Groq)
                     >- RRF -> cross-encoder ->|
    BM25 (sparse)   /

Differences from a user chat session, by design:

- nothing is persisted: the user message is never appended, and the search
  runs against a throw-away evaluation conversation id that is never written,
  so normal user conversation history stays untouched;
- empty history means the history-aware rewriter leaves the question unchanged
  (no LLM call), which is exactly the production path for a fresh conversation.

Any failure (missing BM25 index, unavailable Qdrant, LLM error) is recorded
on the result - never replaced by a fabricated response.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.api.chat_router import get_conversation_repo, get_search_service
from app.core.config import get_settings
from app.retrieval.rag_service import RAGAnswer, RAGService
from app.evaluation.dataset import ABSTENTION_PLACEHOLDER
from app.evaluation.samples import PipelineRunResult

logger = logging.getLogger(__name__)


class ProductionPipeline:
    """Thin wrapper around the real chat pipeline, read-only for the database."""

    def __init__(self) -> None:
        settings = get_settings()
        self._repo = get_conversation_repo()
        self._search_service = get_search_service(self._repo)
        # Reuses the provider wiring in chat_router (real engines).
        self._rag_service = RAGService()
        # The generation model running the pipeline (reported for traceability).
        self.model = self._rag_service.model
        # A conversation id reserved for evaluation: never written to, so
        # recent_messages() always returns [] -> no user history pollution.
        self._conversation_id = f"evaluation-{uuid.uuid4()}"
        self._top_k = settings.reranker_top_k

    def run_question(self, question: str) -> PipelineRunResult:
        """Full production path for one question, with latency capture."""
        started = time.perf_counter()
        try:
            outcome = self._search_service.search(
                self._conversation_id, question, top_k=self._top_k
            )
            hits: list[Any] = outcome.hits
            rag_answer: RAGAnswer = self._rag_service.generate(question, hits)
        except Exception as exc:  # noqa: BLE001 - recorded, never faked
            latency = time.perf_counter() - started
            logger.error("Pipeline failed for %r: %s", question, exc)
            return PipelineRunResult(
                question=question,
                response="",
                latency_seconds=latency,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency = time.perf_counter() - started
        response = rag_answer.answer.strip()
        return PipelineRunResult(
            question=question,
            response=response,
            retrieved_contexts=[str(hit.content) for hit in hits],
            latency_seconds=latency,
        )


def is_abstained(response: str) -> bool:
    """True when the production generator reported insufficient context.

    The production generator answers with a fixed sentence when the context
    does not contain the information (see ``ABSTENTION_PLACEHOLDER`` in
    ``dataset.py``); the same marker phrase is used here to observe the
    abstention behaviour on unanswerable questions.
    """
    return ABSTENTION_PLACEHOLDER.lower() in (response or "").lower()
