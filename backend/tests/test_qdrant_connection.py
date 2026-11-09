"""Backend <-> Qdrant Server connection verification (fully mocked / offline).

Settings → ``make_client`` → ``build_vector_store`` → CLI / ``index_sources``
must consistently target the Qdrant Server from Docker Compose
(``http://localhost:6333``, see ``infrastructure/docker/docker-compose.yml``).
No HTTP request, Docker, indexing or embedding model is involved.
"""

import inspect

from app.core.config import Settings
from app.embeddings.service import EmbeddingService
from app.vector_store import qdrant_service
from app.vector_store.qdrant_service import make_client


def make_settings(**overrides) -> Settings:
    values = {"_env_file": None, "scrape_request_delay_seconds": 0}
    values.update(overrides)
    return Settings(**values)


# -- settings -----------------------------------------------------------------


def test_default_settings_target_docker_compose_qdrant():
    settings = make_settings()
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "docs_chunks"
    assert settings.qdrant_api_key is None


def test_environment_overrides_qdrant_url_and_api_key(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    settings = make_settings()
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.qdrant_api_key == "secret"


def test_dotenv_is_loaded_from_repository_root():
    """SettingsConfigDict must reference the repo-root .env file, not the CWD."""
    env_files = Settings.model_config.get("env_file")
    assert env_files is not None
    assert str(env_files).endswith("/rag-assistant/.env") or env_files.name == ".env"


# -- compose / code consistency ---------------------------------------------


def test_compose_file_matches_settings(tmp_path):
    """The compose service must expose the ports the backend settings expect."""
    from pathlib import Path

    compose = (
        Path(__file__).resolve().parents[2]
        / "infrastructure"
        / "docker"
        / "docker-compose.yml"
    )
    text = compose.read_text(encoding="utf-8")
    assert "6333:6333" in text
    assert "6334:6334" in text
    assert "image: qdrant/qdrant" in text
    assert "qdrant-server" in text  # persistent storage directory


def test_make_client_passes_url_and_api_key(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *, url=None, api_key=None):
            captured["url"] = url
            captured["api_key"] = api_key

    monkeypatch.setattr(qdrant_service, "QdrantClient", FakeClient)
    settings = make_settings(qdrant_url="http://10.0.0.5:6333", qdrant_api_key="k")

    make_client(settings)

    assert captured == {"url": "http://10.0.0.5:6333", "api_key": "k"}


def test_make_client_accepts_missing_api_key(monkeypatch):
    """api_key=None must not break construction (unsecured Compose setup)."""

    class FakeClient:
        def __init__(self, *, url=None, api_key=None):
            assert api_key is None

    monkeypatch.setattr(qdrant_service, "QdrantClient", FakeClient)
    make_client(make_settings())


# -- end-to-end wiring --------------------------------------------------------


def test_build_vector_store_connects_without_injected_client(monkeypatch):
    """build_vector_store(settings) must derive its client from qdrant_url."""
    captured = {}

    class FakeClient:
        def __init__(self, *, url=None, api_key=None):
            captured["url"] = url

        def collection_exists(self, name):
            return True  # collection ensured once, nothing created

        def create_collection(self, collection_name, vectors_config):  # pragma: no cover
            raise AssertionError("must not recreate an existing collection")

    class FakeVectorStore:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(qdrant_service, "QdrantClient", FakeClient)
    monkeypatch.setattr(qdrant_service, "QdrantVectorStore", FakeVectorStore)
    service = EmbeddingService(settings=make_settings())  # lazy, never loaded

    settings = make_settings(qdrant_url="http://localhost:6333")
    store = qdrant_service.build_vector_store(settings, service)

    assert captured["url"] == "http://localhost:6333"
    assert captured["kwargs"]["collection_name"] == settings.qdrant_collection
    assert isinstance(store, FakeVectorStore)


def test_index_sources_connects_via_build_vector_store(monkeypatch, tmp_path):
    """The indexing path must go through build_vector_store (server client)."""
    calls: list[Settings] = []

    class RecordingStore:
        def add_documents(self, documents, ids=None):
            return list(ids or [])

    sentinel_store = RecordingStore()

    def fake_build_vector_store(config, service=None, *, client=None):
        calls.append(config)
        return sentinel_store

    def fake_chunks_path_for(source, config):
        path = tmp_path / f"{source}.jsonl"
        path.write_text(
            '{"page_content": "c0", "metadata": {"source": "u", "chunk_index": 0}}\n',
            encoding="utf-8",
        )
        return path

    import app.vector_store.indexing as indexing

    monkeypatch.setattr(qdrant_service, "build_vector_store", fake_build_vector_store)
    monkeypatch.setattr(indexing, "chunks_path_for", fake_chunks_path_for)

    settings = make_settings(qdrant_url="http://localhost:6333")
    stats = indexing.index_sources(["s"], settings=settings)

    assert calls == [settings]
    assert stats.points_upserted == 1


def test_make_client_signature_uses_server_kwargs_not_path():
    """Drift guard: QdrantClient must be constructed with url/api_key (no local path)."""
    import qdrant_client

    init_params = inspect.signature(qdrant_client.QdrantClient.__init__).parameters
    assert "url" in init_params and "api_key" in init_params
    assert "path" in init_params  # exists but must never be passed by make_client
    source = inspect.getsource(qdrant_service.make_client)
    assert "path=" not in source
