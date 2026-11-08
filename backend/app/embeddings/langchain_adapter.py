"""LangChain ``Embeddings`` adapter connecting our E5 service to LangChain.

``embed_documents`` applies the required ``passage: `` prefix (through
EmbeddingService) and ``embed_query`` the ``query: `` prefix.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.embeddings.service import EmbeddingService


class E5LangChainEmbeddings(Embeddings):
    """LangChain-facing adapter for the E5 EmbeddingService."""

    def __init__(self, service: EmbeddingService) -> None:
        self._service = service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks with the ``passage: `` prefix."""
        return self._service.embed_texts(texts, for_query=False)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the ``query: `` prefix."""
        return self._service.embed_texts([text], for_query=True)[0]
