"""Minimal semantic search over the indexed Qdrant collection (Phase 3 only).

Basic vector search: one query embedding with the E5 ``query: `` prefix and
cosine similarity. BM25 / hybrid / RRF / reranking belong to Phase 4 and are
intentionally absent here.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result with retrieval traceability."""

    content: str
    source_url: str
    title: str
    heading_hierarchy: list[str]
    similarity_score: float
    chunk_index: int = 0
    # Full payload metadata (used by downstream steps, e.g. hybrid fusion).
    metadata: dict = dataclasses.field(default_factory=dict)

    def format(self) -> str:
        hierarchy = " > ".join(self.heading_hierarchy) or self.title
        return (
            f"[{self.similarity_score:.4f}] {self.title} — {hierarchy}\n"
            f"  source: {self.source_url} (chunk {self.chunk_index})\n"
            f"  {self.content[:300]}\n"
        )


class SemanticSearchService:
    """Top-K cosine search over the collection."""

    def __init__(self, vector_store: QdrantVectorStore) -> None:
        self._vector_store = vector_store

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Embed the query (``query: `` prefix, via the store's Embeddings) and search."""
        if not query.strip():
            return []
        pairs = self._vector_store.similarity_search_with_score(query, k=k)
        hits: list[SearchHit] = []
        for document, score in pairs:
            metadata = document.metadata
            hits.append(
                SearchHit(
                    content=document.page_content,
                    source_url=str(metadata.get("source", "")),
                    title=str(metadata.get("title", "")),
                    heading_hierarchy=list(metadata.get("heading_hierarchy", [])),
                    similarity_score=float(score),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    metadata=metadata,
                )
            )
        return hits
