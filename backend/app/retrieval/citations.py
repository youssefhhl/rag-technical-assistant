"""Citation verification against the documents actually retrieved (Phase 10).

The citations returned by the API must point at documents that were really
retrieved for this question. This module checks every citation against the
retrieved chunks:

- the source metadata exists (a citation without a source URL is not usable);
- the title metadata exists;
- the ``(source_url, chunk_index)`` pair matches one of the retrieved chunks.

Invalid citations are **rejected** (dropped from the response and logged with
the exact reason); they are never silently rewritten into another source. The
citation format itself is unchanged (``title`` / ``source_url`` / ``chunk_index``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.context_builder import (
    resolve_chunk_index,
    resolve_source_url,
)

logger = logging.getLogger(__name__)

#: Reasons a citation can be rejected (kept explicit for the logs/reports).
REASON_MISSING_SOURCE = "citation has no source metadata"
REASON_MISSING_TITLE = "citation has no title metadata"
REASON_INVALID_CHUNK_INDEX = "citation has no valid chunk index"
REASON_UNKNOWN_DOCUMENT = "citation does not match any retrieved document"
REASON_MALFORMED = "citation is not a mapping"


@dataclass(frozen=True, slots=True)
class RejectedCitation:
    """One invalid citation and the reason why it was dropped."""

    citation: Any
    reason: str


@dataclass(frozen=True, slots=True)
class CitationVerdict:
    """The outcome of a verification pass."""

    valid: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[RejectedCitation] = field(default_factory=list)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


class CitationVerifier:
    """Checks citations against the documents that were really retrieved."""

    def __init__(self, hits: list[Any] | None) -> None:
        # (source_url, chunk_index) of every retrieved chunk.
        self._documents: set[tuple[str, int]] = set()
        for hit in hits or []:
            metadata = dict(getattr(hit, "metadata", {}) or {})
            source = resolve_source_url(metadata)
            if not source:
                continue
            self._documents.add((source, resolve_chunk_index(metadata)))

    @property
    def documents(self) -> set[tuple[str, int]]:
        """The retrieved ``(source_url, chunk_index)`` pairs."""
        return set(self._documents)

    def verify(self, citations: list[Any] | None) -> CitationVerdict:
        """Split ``citations`` into the valid ones and the rejected ones."""
        valid: list[dict[str, Any]] = []
        rejected: list[RejectedCitation] = []
        for citation in citations or []:
            reason = self.reject_reason(citation)
            if reason is None:
                valid.append(citation)
            else:
                rejected.append(RejectedCitation(citation=citation, reason=reason))
        return CitationVerdict(valid=valid, rejected=rejected)

    def reject_reason(self, citation: Any) -> str | None:
        """The reason why ``citation`` is invalid (``None`` when it is valid)."""
        if not isinstance(citation, dict):
            return REASON_MALFORMED

        source = str(citation.get("source_url") or "").strip()
        if not source:
            return REASON_MISSING_SOURCE

        title = str(citation.get("title") or "").strip()
        if not title:
            return REASON_MISSING_TITLE

        raw_index = citation.get("chunk_index")
        try:
            chunk_index = int(raw_index)
        except (TypeError, ValueError):
            return REASON_INVALID_CHUNK_INDEX

        if (source, chunk_index) not in self._documents:
            return REASON_UNKNOWN_DOCUMENT
        return None
