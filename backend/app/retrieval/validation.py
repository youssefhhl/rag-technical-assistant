"""Retrieval validation on top of the existing Cross-Encoder scores (Phase 10).

The Cross-Encoder of Phase 5 already scores every reranked chunk
(``RerankHit.rerank_score``); this module turns those scores into a decision
the pipeline can act on, without touching the retrieval algorithms:

    RELEVANT           -> the context can answer the question
    PARTIALLY_RELEVANT -> the context is weak but usable (generate anyway)
    IRRELEVANT         -> the context cannot answer the question: abstain
                          (or try one corrective retrieval)

Thresholds are configurable (``retrieval_relevant_threshold`` /
``retrieval_partial_threshold``) and calibrated on the real logits of
``cross-encoder/ms-marco-MiniLM-L-6-v2``.

Deliberate choice: hits **without** a Cross-Encoder score (a pure-RRF
configuration, no reranker wired) are classified ``PARTIALLY_RELEVANT`` — the
retrieval cannot be judged, so the pipeline stays permissive and the generator
keeps its own "not enough information" behaviour. Only a scored, clearly
irrelevant retrieval triggers an abstention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RetrievalLabel(str, Enum):
    """How usable the retrieved context is for the question."""

    RELEVANT = "RELEVANT"
    PARTIALLY_RELEVANT = "PARTIALLY_RELEVANT"
    IRRELEVANT = "IRRELEVANT"


@dataclass(frozen=True, slots=True)
class RetrievalAssessment:
    """The verdict on one retrieval result."""

    label: RetrievalLabel
    top_score: float | None = None
    n_hits: int = 0
    n_relevant: int = 0

    @property
    def sufficient(self) -> bool:
        """True when the context may be handed to the generator."""
        return self.label is not RetrievalLabel.IRRELEVANT


def collect_scores(hits: list[Any] | None) -> list[float]:
    """The Cross-Encoder scores of ``hits`` (empty when there are none)."""
    scores: list[float] = []
    for hit in hits or []:
        value = getattr(hit, "rerank_score", None)
        if value is None:
            continue
        try:
            scores.append(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return scores


class RetrievalValidator:
    """Classifies a reranked result set as relevant / partial / irrelevant."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        relevant_threshold: float | None = None,
        partial_threshold: float | None = None,
    ) -> None:
        config = settings or get_settings()
        self.relevant_threshold = float(
            relevant_threshold
            if relevant_threshold is not None
            else config.retrieval_relevant_threshold
        )
        self.partial_threshold = float(
            partial_threshold
            if partial_threshold is not None
            else config.retrieval_partial_threshold
        )

    def validate(self, hits: list[Any] | None) -> RetrievalAssessment:
        """Assess ``hits`` (reranked hits carry ``rerank_score``)."""
        hits = list(hits or [])
        if not hits:
            # Nothing retrieved: the context is definitely insufficient.
            return RetrievalAssessment(label=RetrievalLabel.IRRELEVANT)

        scores = collect_scores(hits)
        if not scores:
            # No Cross-Encoder score available -> cannot be judged -> permissive.
            return RetrievalAssessment(
                label=RetrievalLabel.PARTIALLY_RELEVANT,
                n_hits=len(hits),
            )

        top = max(scores)
        n_relevant = sum(1 for score in scores if score >= self.relevant_threshold)
        if top >= self.relevant_threshold:
            label = RetrievalLabel.RELEVANT
        elif top >= self.partial_threshold:
            label = RetrievalLabel.PARTIALLY_RELEVANT
        else:
            label = RetrievalLabel.IRRELEVANT

        assessment = RetrievalAssessment(
            label=label,
            top_score=top,
            n_hits=len(hits),
            n_relevant=n_relevant,
        )
        logger.debug(
            "Retrieval assessed %s (top=%.3f, hits=%d, relevant=%d)",
            label.value,
            top,
            len(hits),
            n_relevant,
        )
        return assessment
