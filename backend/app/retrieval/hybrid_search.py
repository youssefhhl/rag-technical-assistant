"""Hybrid search: Reciprocal Rank Fusion of dense and sparse retrieval (Phase 4).

Combines the two existing search services without touching their algorithms:

- dense vector search (Qdrant + E5, ``app.vector_store.search``);
- sparse lexical search (BM25, ``app.retrieval.bm25_search``).

Each engine retrieves its own candidate list; documents are merged by a
stable chunk identifier and re-ranked with RRF::

    RRF_score(d) = sum(1 / (k + rank))   over the rankings where d appears

Ranks start at 1; original retrieval scores are ignored (ranks only).
Duplicates appearing in both rankings receive both contributions.

Initial configuration (constructor parameters, easy to adjust):
vector_k=20, bm25_k=20, rrf_k=60, top_k=10.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

#: Initial configuration values (adjust via constructor params, not here).
DEFAULT_VECTOR_K = 20
DEFAULT_BM25_K = 20
DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 10

#: Engine labels used in the fusion bookkeeping.
VECTOR_ENGINE = "vector"
BM25_ENGINE = "bm25"


@dataclass(frozen=True, slots=True)
class HybridHit:
    """One fused result; original content and metadata are preserved."""

    content: str
    metadata: dict
    rrf_score: float
    vector_rank: int | None = None  # 1-based rank in the dense ranking, if listed
    bm25_rank: int | None = None  # 1-based rank in the sparse ranking, if listed

    def format(self) -> str:
        metadata = self.metadata
        title = str(metadata.get("title", ""))
        hierarchy = " > ".join(metadata.get("heading_hierarchy", [])) or title
        return (
            f"[hybrid {self.rrf_score:.4f}] {title} — {hierarchy}\n"
            f"  source: {metadata.get('source', '')} "
            f"(chunk {metadata.get('chunk_index', 0)})\n"
            f"  {self.content[:300]}\n"
        )


class SearchEngine(Protocol):
    """Duck-typed contract shared by SemanticSearchService and Bm25SearchService."""

    def search(self, query: str, k: int = 5) -> list[Any]: ...


def _normalized_content(content: str) -> str:
    """Whitespace-normalised chunk text (robust identity basis)."""
    return " ".join((content or "").split())


def chunk_identity(hit: Any) -> str:
    """Stable identifier for duplicate detection across rankings.

    ``chunk_index`` restarts at 0 for every page of a source, so
    ``source + chunk_index`` collides massively on real corpora (verified on
    data/processed/fastapi: 20 keys for 126 chunks). The identity is therefore
    the **normalised content fingerprint**: truly identical chunks (the same
    point retrieved by both engines) merge; different chunks never merge.
    Falls back to ``source_url`` + ``chunk_index`` only for content-less hits.
    """
    content = _normalized_content(str(getattr(hit, "content", "") or ""))
    if content:
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
        return f"content:{digest}"
    source = str(getattr(hit, "source_url", "") or "")
    if source:
        return f"{source}#chunk{getattr(hit, 'chunk_index', 0)}"
    return "content:" + hashlib.sha1(b"").hexdigest()[:16]


@dataclass(slots=True)
class _FusedDocument:
    """Accumulator for one unique chunk across the rankings."""

    hit: Any
    rrf_score: float = 0.0
    contributions: dict[str, int] = field(default_factory=dict)  # engine -> 1-based rank


def reciprocal_rank_fusion(
    rankings: list[tuple[str, list[Any]]],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[_FusedDocument, float]]:
    """Fuse ``(engine_label, hits)`` rankings with RRF, highest score first.

    The fusion algorithm is engine-agnostic: any candidate type exposing
    ``content``/``metadata`` fields works (SearchHit, Bm25Hit, test fakes).
    Deterministic tie-break: first-seen order when scores are equal.
    """
    fused: dict[str, _FusedDocument] = {}
    order: list[str] = []

    for engine_label, ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = chunk_identity(hit)
            entry = fused.get(key)
            if entry is None:
                entry = _FusedDocument(hit=hit)
                fused[key] = entry
                order.append(key)
            entry.rrf_score += 1.0 / (rrf_k + rank)
            entry.contributions[engine_label] = rank

    ordered_keys = sorted(fused, key=lambda key: (-fused[key].rrf_score, order.index(key)))
    return [(fused[key], fused[key].rrf_score) for key in ordered_keys]


class HybridSearchService:
    """Dense + sparse retrieval, merged with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_search: SearchEngine,
        bm25_search: SearchEngine,
        *,
        vector_k: int = DEFAULT_VECTOR_K,
        bm25_k: int = DEFAULT_BM25_K,
        rrf_k: int = DEFAULT_RRF_K,
        top_k: int = DEFAULT_TOP_K,
        reranker: Any | None = None,
    ) -> None:
        self._vector_search = vector_search
        self._bm25_search = bm25_search
        self.vector_k = max(1, vector_k)
        self.bm25_k = max(1, bm25_k)
        self.rrf_k = rrf_k
        self.top_k = max(1, top_k)
        # Phase 5: optional Cross-Encoder reranker applied after the fusion.
        # None keeps the pure RRF behaviour (list[HybridHit]).
        self._reranker = reranker

    def search(self, query: str, top_k: int | None = None) -> list[HybridHit]:
        """Retrieve from both engines, fuse rankings, return the top ``k`` unique hits."""
        if not query.strip():
            return []
        limit = self.top_k if top_k is None else max(1, top_k)

        rankings = [
            (VECTOR_ENGINE, self._vector_search.search(query, k=self.vector_k)),
            (BM25_ENGINE, self._bm25_search.search(query, k=self.bm25_k)),
        ]
        fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)

        hits: list[HybridHit] = []
        for entry, score in fused[: self.top_k]:
            original = entry.hit
            hits.append(
                HybridHit(
                    content=str(original.content),
                    metadata=dict(getattr(original, "metadata", {}) or {}),
                    rrf_score=score,
                    vector_rank=entry.contributions.get(VECTOR_ENGINE),
                    bm25_rank=entry.contributions.get(BM25_ENGINE),
                )
            )

        if self._reranker is None:
            return hits
        # Phase 5 pipeline: fused candidates -> Cross-Encoder -> final top_k.
        # ``top_k`` stays the *final* count; the fusion cap stays self.top_k.
        if top_k is None:
            return self._reranker.rerank(query, hits)  # reranker default (final top 5)
        return self._reranker.rerank(query, hits, top_k=limit)
