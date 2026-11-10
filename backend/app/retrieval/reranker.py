"""Cross-Encoder reranking (Phase 5).

Re-scores the fused hybrid candidates with an actual Cross-Encoder — the last
step of the pipeline ``Vector 20 + BM25 20 -> RRF(k=60, top 10) -> CrossEncoder
-> top 5``. Retrieval algorithms and the RRF fusion are untouched.

The model (default ``cross-encoder/ms-marco-MiniLM-L-6-v2``) is loaded lazily
on the first ``rerank()`` call and reused afterwards; tests inject a fake.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Initial Cross-Encoder configuration (adjust via settings, not here).
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class RerankHit:
    """One reranked result; original content and metadata are preserved."""

    content: str
    metadata: dict
    rerank_score: float
    rrf_score: float | None = None  # carried over from the fused ranking (if known)
    vector_rank: int | None = None
    bm25_rank: int | None = None

    def format(self) -> str:
        metadata = self.metadata
        title = str(metadata.get("title", ""))
        hierarchy = " > ".join(metadata.get("heading_hierarchy", [])) or title
        trace: list[str] = []
        if self.rrf_score is not None:
            trace.append(f"rrf {self.rrf_score:.4f}")
        if self.vector_rank is not None:
            trace.append(f"vector #{self.vector_rank}")
        if self.bm25_rank is not None:
            trace.append(f"bm25 #{self.bm25_rank}")
        suffix = f" ({', '.join(trace)})" if trace else ""
        return (
            f"[ce {self.rerank_score:+.4f}] {title} — {hierarchy}{suffix}\n"
            f"  source: {metadata.get('source', '')} "
            f"(chunk {metadata.get('chunk_index', 0)})\n"
            f"  {self.content[:300]}\n"
        )


class RerankModel(Protocol):
    """Duck-typed contract shared with ``sentence_transformers.CrossEncoder``."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


class CrossEncoderReranker:
    """Re-scores retrieved chunks with a lazy-loaded CrossEncoder model."""

    def __init__(
        self,
        model: RerankModel | None = None,
        *,
        model_name: str | None = None,
        settings: Settings | None = None,
        top_k: int | None = None,
    ) -> None:
        config = settings or get_settings()
        self._settings = config
        self._model_name = model_name or config.reranker_model
        self.top_k = max(1, top_k if top_k is not None else config.reranker_top_k)
        self._model: RerankModel | None = model
        self._load_lock = threading.Lock()

    # -- model handling ------------------------------------------------------

    @property
    def model(self) -> RerankModel:
        """Load the CrossEncoder on first use (once per instance)."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:  # double-checked lazy load
                    self._model = self._load_model()
        return self._model

    def _load_model(self) -> RerankModel:
        from sentence_transformers import CrossEncoder  # lazy heavy import

        logger.info("Loading Cross-Encoder reranker %s ...", self._model_name)
        return CrossEncoder(self._model_name)  # type: ignore[return-value]

    # -- reranking ------------------------------------------------------------

    def rerank(
        self, query: str, hits: list[Any], top_k: int | None = None
    ) -> list[RerankHit]:
        """Re-score ``hits`` against ``query`` and return the best ``top_k``.

        ``hits`` are duck-typed results exposing ``content`` (e.g. ``HybridHit``
        from the fused ranking); original content, metadata and engine rank
        info are preserved. Empty input/query or ``top_k < 1`` returns ``[]``
        without touching the model.
        """
        limit = self.top_k if top_k is None else top_k
        if not query.strip() or not hits or limit < 1:
            return []

        pairs = [(query, str(getattr(hit, "content", "") or "")) for hit in hits]
        scores = self.model.predict(pairs)  # triggers the lazy model load

        ranked = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        reranked: list[RerankHit] = []
        for index in ranked[:limit]:
            original = hits[index]
            reranked.append(
                RerankHit(
                    content=str(original.content),
                    metadata=dict(getattr(original, "metadata", {}) or {}),
                    rerank_score=float(scores[index]),
                    rrf_score=_optional_float(getattr(original, "rrf_score", None)),
                    vector_rank=getattr(original, "vector_rank", None),
                    bm25_rank=getattr(original, "bm25_rank", None),
                )
            )
        logger.info("Reranked %d chunk(s) with %s", len(hits), self._model_name)
        return reranked
