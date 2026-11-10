"""Contextual query rewriting via LiteLLM/Groq (Phase 6).

Follow-up questions in a conversation are not standalone — "et celui après ?",
"how do I deploy *it*?". The rewriter turns the latest question into a
standalone search query using only the recent conversation history (no summar-
isation, no multi-query, no decomposition).

Behaviour:
- history empty  -> the question is returned unchanged (no LLM call);
- LLM failure    -> the original question (fallback, never raises);
- standalone      -> the LLM is instructed to output it unchanged.

The result only feeds retrieval (hybrid search); the original question is
kept by the caller for answer generation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from app.core.config import Settings, get_settings
from app.retrieval.llm_content import extract_message_content as _extract_message_content

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You rewrite a follow-up question of a conversation into a single, "
    "standalone search query for a retrieval engine. Use the conversation "
    "history only to resolve pronouns and ellipses. Output ONLY the rewritten "
    "query, with no explanation, no quotes, no extra words. If the question "
    "already makes sense standalone, output it unchanged."
)


class HistoryMessage(Protocol):
    """Duck-typed message: ``role`` + ``content`` (ORM Message fits)."""

    role: str
    content: str


class QueryRewriter:
    """Rewrites context-dependent questions into standalone search queries."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        completion: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Injectable LLM call (used by tests); ``None`` -> real LiteLLM call.
        self._completion = completion

    # -- prompt ------------------------------------------------------------------

    def build_prompt_messages(
        self, question: str, history: list[HistoryMessage]
    ) -> list[dict[str, str]]:
        """Chat messages (system + user) invoking the model, LiteLLM shape."""
        history_text = "\n".join(
            f"{'User' if message.role == 'user' else 'Assistant'}: {message.content}"
            for message in history
        )
        user_prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"Rewrite this latest question as a standalone search query:\n"
            f"{question.strip()}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    # -- rewriting ----------------------------------------------------------------

    def rewrite(self, question: str, history: list[HistoryMessage]) -> str:
        """Return a standalone search query; falls back to ``question`` on failure."""
        if not question.strip() or not history:
            return question

        messages = self.build_prompt_messages(question, history)
        try:
            rewritten = (self._completion or self._call_llm)(messages)
        except Exception as exc:  # noqa: BLE001 - any LLM/network error is a fallback
            logger.warning("Query rewriting failed (%s) - keeping original question", exc)
            return question

        rewritten = (rewritten or "").strip()
        if not rewritten:
            logger.info("Query rewriting returned nothing - keeping original question")
            return question
        return rewritten

    # -- LiteLLM -----------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Real LiteLLM/Groq call (lazy heavy import; no key -> provider default)."""
        import litellm  # lazy: keeps application import light

        response: Any = litellm.completion(
            model=self._settings.llm_model,
            messages=messages,
            api_key=self._settings.groq_api_key or None,
            temperature=0.0,
            num_retries=1,
        )
        return _extract_message_content(response)
