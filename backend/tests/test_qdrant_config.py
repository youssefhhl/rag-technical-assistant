"""Tests for Qdrant collection configuration and vector-store construction."""

import pytest
import qdrant_client.models as qdrant_models

from app.core.config import Settings
from app.embeddings.service import EMBEDDING_DIMENSION, EmbeddingService
from app.vector_store.qdrant_service import ensure_collection, make_client


class FakeQdrantClient:
    def __init__(self, exists: bool = False) -> None:
        self._exists = exists
        self.created: list[tuple[str, qdrant_models.VectorParams]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self._exists

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.created.append((collection_name, vectors_config))


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path / "data",
        "qdrant_local_path": tmp_path / "qdrant",
        "scrape_request_delay_seconds": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_ensure_collection_creates_384_dim_cosine_collection():
    client = FakeQdrantClient(exists=False)
    ensure_collection(client, "docs_chunks", EMBEDDING_DIMENSION)

    name, params = client.created[0]
    assert name == "docs_chunks"
    assert params.size == 384
    assert params.distance == qdrant_models.Distance.COSINE


def test_ensure_collection_is_idempotent():
    client = FakeQdrantClient(exists=True)
    ensure_collection(client, "docs_chunks", EMBEDDING_DIMENSION)
    assert client.created == []  # nothing created when collection already exists


def test_ensure_collection_parameterizes_dimension():
    """The dimension is passed through (384 expected for E5 in production)."""
    client = FakeQdrantClient(exists=False)
    ensure_collection(client, "docs", 768)
    assert client.created[0][1].size == 768


def test_make_client_local_mode_writes_under_configured_path(tmp_path):
    settings = make_settings(tmp_path)
    client = make_client(settings)
    try:
        assert client.collection_exists("docs_chunks") in (True, False)
        assert (settings.qdrant_local_path).is_dir()  # storage dir created
    finally:
        client.close()


def test_make_client_uses_url_in_server_mode(tmp_path, monkeypatch):
    captured = {}

    class FakeServerClient(FakeQdrantClient):
        def __init__(self, url=None, api_key=None, **kwargs):
            captured["url"] = url
            captured["api_key"] = api_key
            super().__init__()

    monkeypatch.setattr("app.vector_store.qdrant_service.QdrantClient", FakeServerClient)
    settings = make_settings(
        tmp_path, qdrant_mode="server", qdrant_url="http://localhost:9333", qdrant_api_key="secret"
    )
    make_client(settings)
    assert captured["url"] == "http://localhost:9333"
    assert captured["api_key"] == "secret"


def test_build_vector_store_wraps_service_and_ensures_collection(tmp_path, monkeypatch):
    from app.vector_store import qdrant_service

    client = FakeQdrantClient(exists=False)
    captured_kwargs = {}

    class FakeVectorStore:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(qdrant_service, "QdrantVectorStore", FakeVectorStore)

    embedding_service = EmbeddingService(settings=make_settings(tmp_path))  # lazy, never used
    store = qdrant_service.build_vector_store(
        make_settings(tmp_path), embedding_service, client=client  # type: ignore[arg-type]
    )

    assert isinstance(store, FakeVectorStore)
    assert len(client.created) == 1  # collection ensured exactly once
    embeddings = captured_kwargs["embeddings"]
    assert isinstance(embeddings, qdrant_service.E5LangChainEmbeddings)
    assert embeddings._service is embedding_service
    assert captured_kwargs["collection_name"] == "docs_chunks"
