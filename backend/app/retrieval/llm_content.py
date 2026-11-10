"""Helpers for reading the assistant text out of a LiteLLM completion response.

LiteLLM returns pydantic objects (``ModelResponse`` / ``Choice`` / ``Message``),
but plain dicts show up with some providers, with ``mock`` style test doubles
and with the legacy ``prompt=`` completion mode.  Reading the response must
work for all of those shapes instead of assuming one of them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["extract_message_content"]


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a mapping or an object ("" when absent)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_message_content(response: Any) -> str:
    """Return the assistant text of a completion response ("" when there is none)."""
    choices = _get(response, "choices") or []
    if not choices:
        return ""

    choice = choices[0]
    message = _get(choice, "message")
    if message is None:
        # Legacy completions expose the text directly on the choice.
        return str(_get(choice, "text") or "").strip()

    content = _get(message, "content")
    if content is None:
        content = _get(choice, "text")
    return str(content or "").strip()
