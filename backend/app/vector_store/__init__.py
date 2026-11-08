"""Vector store layer: Qdrant collection management, indexing, semantic search.

Modules (lazy-loaded to avoid hard dependency on langchain-qdrant until Phase 3
dependencies are installed):

- ``qdrant_service``: client construction (local QdrantLocal or server mode),
  idempotent collection creation (384 dims / cosine), LangChain store builder
- ``indexing``: Phase 2 chunks.jsonl loading, UUIDv5 stable point IDs,
  batched embedding + upsert pipeline with statistics
- ``search``: minimal top-K semantic search (Phase 3 verification; BM25 /
  hybrid / reranking arrive in Phase 4)
"""

from app.vector_store.indexing import (
    IndexingStats,
    chunk_ids,
    discover_sources,
    index_sources,
    load_chunks,
    stable_point_id,
)

__all__ = [
    "IndexingStats",
    "chunk_ids",
    "discover_sources",
    "index_sources",
    "load_chunks",
    "stable_point_id",
]