"""Qdrant collection setup and vector-store construction.

- Local storage mode (default): embedded QdrantLocal under ``QDRANT_LOCAL_PATH``
  — no service to run during initial development.
- Server mode: the instance at ``QDRANT_URL`` (+ optional API key).
- The collection is created (idempotently) with 384-dim vectors and cosine distance.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from app.core.config import Settings, get_settings
from app.embeddings.langchain_adapter import E5LangChainEmbeddings
from app.embeddings.service import EMBEDDING_DIMENSION, EmbeddingService

logger = logging.getLogger(__name__)


def make_client(settings: Settings | None = None) -> QdrantClient:
    """Create a Qdrant client in the configured mode (local path or server)."""
    config = settings or get_settings()
    if config.qdrant_mode == "server":
        return QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    config.qdrant_local_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.qdrant_local_path))


def ensure_collection(client: QdrantClient, collection_name: str, dimension: int) -> None:
    """Create the collection when missing: ``dimension`` vectors, cosine distance."""
    if client.collection_exists(collection_name):
        logger.debug("Collection %r already exists", collection_name)
        return
    logger.info("Creating Qdrant collection %r (dim=%d, distance=COSINE)", collection_name, dimension)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=dimension,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def build_vector_store(
    settings: Settings | None = None,
    service: EmbeddingService | None = None,
    *,
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    """Build the LangChain Qdrant vector store for the configured collection.

    The LangChain ``Embeddings`` implementation wraps our E5 service so
    ``add_documents`` / ``similarity_search`` automatically get the correct
    ``passage:`` / ``query:`` prefixes.
    """
    config = settings or get_settings()
    if client is None:
        client = make_client(config)
    ensure_collection(client, config.qdrant_collection, EMBEDDING_DIMENSION)

    embeddings: Embeddings = E5LangChainEmbeddings(service or get_settings_service(config))
    return QdrantVectorStore(
        client=client,
        collection_name=config.qdrant_collection,
        embeddings=embeddings,
    )


def get_settings_service(config: Settings) -> EmbeddingService:
    from app.embeddings.service import get_embedding_service

    return get_embedding_service(config)
