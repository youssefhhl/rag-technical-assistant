"""Retrieval layer (Phase 4): sparse BM25 lexical search.

Currently: BM25 index build/persistence/load (``bm25_indexing``) and top-k
lexical search (``bm25_search``). Dense (Qdrant) search lives in
``app.vector_store``; the RRF/hybrid fusion arrives later.
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

__all__ = [
    "Bm25Document",
    "Bm25Hit",
    "Bm25IndexBundle",
    "Bm25SearchService",
    "build_bm25_index",
    "chunks_path_for",
    "discover_sources",
    "index_path",
    "load_bm25_index",
    "load_chunks",
    "save_bm25_index",
    "tokenize",
]
