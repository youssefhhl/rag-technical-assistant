"""Lightweight CRAG: one corrective retrieval attempt (Phase 10).

When the first retrieval is judged irrelevant, the pipeline may try **one**
corrective round before abstaining:

1. reformulate the question with the existing ``QueryRewriter``
   (``reformulate()`` - same information need, different wording);
2. run the existing hybrid search (Qdrant + BM25 + RRF + Cross-Encoder) once
   with the reformulated query;
3. re-assess the new result with the same thresholds.

Hard guarantees:

- at most **one** corrective attempt per question (no recursive retries);
- the corrective attempt is skipped when the reformulation is identical to the
  original question (the search would return the same documents);
- the original question is never modified - only the retrieval query is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.validation import RetrievalAssessment, RetrievalValidator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorrectiveAttempt:
    """One corrective round: the query used, the hits and their assessment."""

    query: str
    hits: list[Any] = field(default_factory=list)
    assessment: RetrievalAssessment | None = None

    @property
    def sufficient(self) -> bool:
        return self.assessment is not None and self.assessment.sufficient


class CorrectiveRetriever:
    """Runs at most one reformulated hybrid search."""

    def __init__(
        self,
        hybrid_service: Any | None = None,
        rewriter: Any | None = None,
        *,
        validator: RetrievalValidator | None = None,
    ) -> None:
        self._hybrid_service = hybrid_service
        self._rewriter = rewriter
        self._validator = validator or RetrievalValidator()

    def attempt(
        self, question: str, top_k: int | None = None
    ) -> CorrectiveAttempt | None:
        """Try one corrective retrieval; ``None`` when it is not possible."""
        if self._hybrid_service is None or self._rewriter is None:
            logger.info("Corrective retrieval not available (no hybrid service or rewriter)")
            return None

        reformulated = (self._rewriter.reformulate(question) or "").strip()
        if not reformulated:
            return None
        if reformulated.lower() == (question or "").strip().lower():
            logger.info(
                "Corrective reformulation is identical to the original question - skipped"
            )
            return None

        hits = list(self._hybrid_service.search(reformulated, top_k=top_k) or [])
        assessment = self._validator.validate(hits)
        logger.info(
            "Corrective retrieval with %r -> %s (%d hit(s))",
            reformulated,
            assessment.label.value,
            len(hits),
        )
        return CorrectiveAttempt(
            query=reformulated, hits=hits, assessment=assessment
        )
