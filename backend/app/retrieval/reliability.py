"""Reliability layer of the chat pipeline (Phase 10).

Wraps the existing Phase 7 generation with four guarantees, without changing
the retrieval or generation algorithms:

1. **Retrieval validation** - the reranked context is classified by the
   Cross-Encoder scores (``app.retrieval.validation``);
2. **Intelligent abstention** - an irrelevant (or empty) context never reaches
   the generator: the fixed abstention sentence is returned instead, with no
   citation;
3. **Lightweight CRAG** - one corrective retrieval (reformulate -> hybrid
   search -> re-assess) before giving up;
4. **Citation verification** - only citations pointing at documents that were
   really retrieved are returned to the caller.

The abstention sentence is the one already used by ``RAGService`` (Phase 7),
so the API keeps a single stable phrasing and the existing response schema
(``answer`` + ``citations``) is untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.citations import CitationVerifier
from app.retrieval.corrective import CorrectiveRetriever
from app.retrieval.validation import RetrievalLabel, RetrievalValidator

logger = logging.getLogger(__name__)

#: The abstention sentence, identical to the one produced by the generator.
ABSTENTION_MESSAGE = (
    "I don't have enough information to answer this question based on the "
    "provided documents."
)


@dataclass(frozen=True, slots=True)
class ReliabilityOutcome:
    """The result of one reliable answer."""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    label: RetrievalLabel = RetrievalLabel.IRRELEVANT
    abstained: bool = False
    corrected: bool = False
    corrective_query: str | None = None
    rejected_citations: list[dict[str, Any]] = field(default_factory=list)
    #: The chunks actually used as context (the corrected ones after a
    #: corrective retrieval) - what the answer was grounded in.
    hits: list[Any] = field(default_factory=list)


class ReliabilityPipeline:
    """Validate, correct, abstain and verify the citations of one answer."""

    def __init__(
        self,
        hybrid_service: Any | None = None,
        rewriter: Any | None = None,
        rag_service: Any | None = None,
        *,
        validator: RetrievalValidator | None = None,
        corrective: CorrectiveRetriever | None = None,
        crag_enabled: bool = True,
    ) -> None:
        self._validator = validator or RetrievalValidator()
        self._corrective = corrective or CorrectiveRetriever(
            hybrid_service, rewriter, validator=self._validator
        )
        self._rag_service = rag_service
        self._crag_enabled = crag_enabled

    def answer(
        self, question: str, hits: list[Any] | None, top_k: int | None = None
    ) -> ReliabilityOutcome:
        """Answer ``question`` from ``hits`` (already reranked by the caller)."""
        assessment = self._validator.validate(hits)
        used_hits = list(hits or [])
        corrected = False
        corrective_query: str | None = None

        # -- one corrective attempt (lightweight CRAG) ------------------------
        if not assessment.sufficient and self._crag_enabled:
            attempt = self._corrective.attempt(question, top_k=top_k)
            if attempt is not None:
                corrective_query = attempt.query
                if attempt.sufficient:
                    used_hits = attempt.hits
                    assessment = attempt.assessment  # type: ignore[assignment]
                    corrected = True

        # -- abstention when the context is still insufficient ----------------
        if not assessment.sufficient:
            logger.info(
                "Abstaining: retrieval assessed %s (corrective attempt: %s)",
                assessment.label.value,
                "yes" if corrective_query else "no",
            )
            return ReliabilityOutcome(
                answer=ABSTENTION_MESSAGE,
                citations=[],
                label=assessment.label,
                abstained=True,
                corrected=corrected,
                corrective_query=corrective_query,
                hits=used_hits,
            )

        # -- generation + citation verification -------------------------------
        if self._rag_service is None:
            raise RuntimeError("ReliabilityPipeline needs a RAGService to generate an answer")

        rag_answer = self._rag_service.generate(question, used_hits)
        verdict = CitationVerifier(used_hits).verify(rag_answer.citations)
        for rejected in verdict.rejected:
            logger.warning(
                "Rejected citation %r: %s", rejected.citation, rejected.reason
            )

        return ReliabilityOutcome(
            answer=rag_answer.answer,
            citations=verdict.valid,
            label=assessment.label,
            abstained=False,
            corrected=corrected,
            corrective_query=corrective_query,
            rejected_citations=[
                {"citation": rejected.citation, "reason": rejected.reason}
                for rejected in verdict.rejected
            ],
            hits=used_hits,
        )
