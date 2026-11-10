"""Retrieval layer (Phase 4/5): sparse BM25, hybrid RRF fusion, reranking.

- ``bm25_indexing``: BM25 index build/persistence/load over Phase 2 chunks.
- ``bm25_search``: top-k lexical search.
- ``hybrid_search``: dense (Qdrant) + sparse (BM25) fusion via RRF.
- ``reranker``: Cross-Encoder reranking of the fused candidates (Phase 5).

Dense (Qdrant) search lives in ``app.vector_store``.
"""

from app.retrieval.bm25_indexing import (
    Bm25Document,
    Bm25IndexBundle,
    build_bm25_index,
    chunks_path_for,
    discover_sources,
    index_path,
    load_bm25_index,
    load_chunks,
    save_bm25_index,
    tokenize,
)
from app.retrieval.bm25_search import Bm25Hit, Bm25SearchService
from app.retrieval.hybrid_search import (
    DEFAULT_BM25_K,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_K,
    HybridHit,
    HybridSearchService,
    chunk_identity,
    reciprocal_rank_fusion,
)
from app.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_TOP_K as DEFAULT_RERANK_TOP_K,
    CrossEncoderReranker,
    RerankHit,
)

__all__ = [
    "DEFAULT_BM25_K",
    "DEFAULT_RRF_K",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANK_TOP_K",
    "DEFAULT_TOP_K",
    "DEFAULT_VECTOR_K",
    "Bm25Document",
    "Bm25Hit",
    "Bm25IndexBundle",
    "Bm25SearchService",
    "CrossEncoderReranker",
    "HybridHit",
    "HybridSearchService",
    "RerankHit",
    "chunk_identity",
    "reciprocal_rank_fusion",
    "build_bm25_index",
    "chunks_path_for",
    "discover_sources",
    "index_path",
    "load_bm25_index",
    "load_chunks",
    "save_bm25_index",
    "tokenize",
]
