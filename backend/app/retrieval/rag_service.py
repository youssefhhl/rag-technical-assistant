"""Grounded RAG generation (Phase 7).

Wraps LiteLLM to produce answers from the RAG context.
- Accepts a question and a list of retrieved chunks.
- Calls the configured LLM (default: groq/openai/gpt-oss-120b).
- Returns a structured answer with the generated text and source citations.
- Falls back gracefully when the LLM says insufficient information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from litellm import completion as litellm_completion
# LiteLLm expects either ``prompt`` (string) or ``messages`` (list of dict).
# We'll use ``prompt`` for the legacy one-shot mode, but fall back to
# ``messages`` when the provider requires it.

from app.core.config import Settings, get_settings
from app.retrieval.context_builder import ContextBuilder, ContextEntry, RAG_PROMPT_TEMPLATE
from app.retrieval.llm_content import extract_message_content

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Structured answer from the RAG generator."""
    answer: str
    citations: list[dict[str, str]] = field(default_factory=list)
    # Each citation: {"title": str, "source_url": str, "chunk_index": int}


class RAGService:
    """Grounded generation over retrieved chunks using LiteLLM."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = model or self._settings.llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # -- LiteLLM call ------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Invoke LiteLLM and return the assistant message content."""
        # Ensure prompt is never empty - if it is, use a minimal fallback
        if not prompt or not prompt.strip():
            prompt = "You are a helpful assistant. Answer the user's question."
        try:
            # LiteLLM accepts either ``prompt`` (string) or ``messages`` (list).
            # We try ``messages`` first since the Groq/Qwen etc. providers require it.
            response = litellm_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:  # noqa: BLE001 - fallback to prompt mode
            try:
                response = litellm_completion(
                    model=self.model,
                    prompt=prompt,
                )
            except Exception:  # noqa: BLE001 - if both fail, return empty answer
                return ""

        # LiteLLM returns a completion object; extract the text.
        # Support both the legacy `completion` API and the newer chat API.
        return extract_message_content(response)

    # -- generation --------------------------------------------------------

    def generate(self, question: str, hits: list[Any], top_k: int | None = None) -> RAGAnswer:
        """Generate a grounded answer for *question* using the top *k* *hits*.

        The method:
        1. Builds the context block from the hits (via ContextBuilder).
        2. Formats the RAG prompt template.
        3. Calls the LLM.
        4. Parses the answer and extracts citations.
        """
        builder = ContextBuilder(hits, max_chunks=5)
        context_block, entries = builder.build()

        # Format citations from the entries
        citations: list[dict[str, str]] = []
        for e in entries:
            citations.append(
                {
                    "title": e.title,
                    "source_url": e.source_url,
                    "chunk_index": str(e.chunk_index),
                }
            )

        # Build the full prompt - ensure question is always included even if context is empty
        if context_block and context_block.strip():
            prompt = RAG_PROMPT_TEMPLATE.format(context=context_block, question=question)
        else:
            # Context is empty - still include the question so the LLM can respond appropriately
            prompt = RAG_PROMPT_TEMPLATE.format(context="", question=question)

        # Call the LLM
        answer_text = self._call_llm(prompt)

        # If the model admits insufficient information, return that.
        lower = answer_text.lower().strip()
        if any(
            phrase in lower for phrase in ["i don't have enough information", "cannot answer", "not enough context"]
        ):
            answer_text = "I don't have enough information to answer this question based on the provided documents."

        return RAGAnswer(
            answer=answer_text,
            citations=citations,
        )


# Convenience function for the retrieval pipeline.
def rag_query(question: str, hybrid_service: Any, top_k: int | None = None) -> RAGAnswer:
    """End-to-end RAG: hybrid search → generation (algorithms untouched)."""
    # Hybrid search returns HybridHit[]; we pass them straight to RAGService.
    rag = RAGService()
    return rag.generate(question, hybrid_service, top_k)