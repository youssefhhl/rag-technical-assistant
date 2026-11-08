"""Embedding layer: E5 embedding service + LangChain Embeddings adapter."""

from app.embeddings.service import (
    EMBEDDING_DIMENSION,
    EmbeddingService,
    get_embedding_service,
)

__all__ = ["EMBEDDING_DIMENSION", "EmbeddingService", "get_embedding_service"]
