"""Tests for the E5 embedding service (model always faked — never downloaded)."""

import pytest

from app.core.config import Settings
from app.embeddings.langchain_adapter import E5LangChainEmbeddings
from app.embeddings.service import EMBEDDING_DIMENSION, EmbeddingService, get_embedding_service


class RecordingModel:
    """Fake SentenceTransformer: records encode() calls, returns valid vectors."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self.calls: list[dict] = []

    def encode(self, sentences, *, batch_size=32, normalize_embeddings=False, show_progress_bar=None):
        if any(not text.strip() for text in sentences):
            raise AssertionError("empty text reached the model")
        self.calls.append(
            {
                "texts": list(sentences),
                "batch_size": batch_size,
                "normalize": normalize_embeddings,
            }
        )
        return [(0.0, 1.0, *([0.0] * (self._dimension - 2))) for _ in sentences]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


def make_settings(**overrides) -> Settings:
    import importlib

    config = importlib.import_module("app.core.config")
    values = {
        "embedding_model": "fake/model",
        "embedding_batch_size": 2,
        "scrape_request_delay_seconds": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_passage_prefix_for_chunks():
    model = RecordingModel()
    service = EmbeddingService(model=model, settings=make_settings())
    service.embed_texts(["chunk text"])

    assert model.calls[0]["texts"] == ["passage: chunk text"]
    assert model.calls[0]["normalize"] is True  # cosine-ready L2 normalization


def test_query_prefix_for_queries():
    model = RecordingModel()
    service = EmbeddingService(model=model, settings=make_settings())
    service.embed_query("how do I declare routes?")

    assert model.calls[0]["texts"] == ["query: how do I declare routes?"]


def test_batches_are_size_configured():
    model = RecordingModel()
    settings = make_settings(embedding_batch_size=2)
    service = EmbeddingService(model=model, settings=settings)
    service.embed_texts([f"t{i}" for i in range(5)])

    assert [len(call["texts"]) for call in model.calls] == [2, 2, 1]


def test_empty_text_maps_to_zero_vector_without_model_call():
    model = RecordingModel()
    service = EmbeddingService(model=model, settings=make_settings())
    vectors = service.embed_texts(["", "   ", "real text"])

    assert vectors[0] == [0.0] * EMBEDDING_DIMENSION
    assert vectors[1] == [0.0] * EMBEDDING_DIMENSION
    # exactly one model call, containing only the non-empty text
    assert len(model.calls) == 1
    assert model.calls[0]["texts"] == ["passage: real text"]


def test_none_text_is_treated_as_empty():
    model = RecordingModel()
    service = EmbeddingService(model=model, settings=make_settings())
    vectors = service.embed_texts([None or "", ])  # never None by type but guard anyway
    assert vectors[0] == [0.0] * EMBEDDING_DIMENSION


def test_output_is_unit_normalized():
    model = RecordingModel()
    service = EmbeddingService(model=model, settings=make_settings())
    vector = service.embed_query("x")
    assert abs(sum(component * component for component in vector) - 1.0) < 1e-5


def test_model_dimension_is_validated_on_init():
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        EmbeddingService(model=RecordingModel(dimension=512), settings=make_settings())


def test_lazy_model_loading_on_first_use():
    service = EmbeddingService(settings=make_settings())  # no model injected

    def fail_load(self):
        raise AssertionError("model unexpectedly loaded")

    import app.embeddings.service as service_module
    monkey = service_module.EmbeddingService._load_model
    service_module.EmbeddingService._load_model = fail_load
    try:
        with pytest.raises(AssertionError):
            _ = service.model
    finally:
        service_module.EmbeddingService._load_model = monkey


def test_get_embedding_service_is_cached(monkeypatch):
    monkeypatch.setattr("app.embeddings.service._SERVICE", None)
    first = get_embedding_service(settings=make_settings())
    second = get_embedding_service()
    assert first is second


def test_langchain_adapter_uses_passage_and_query_prefixes():
    model = RecordingModel()
    service = EmbeddingService(model=model, settings=make_settings())
    adapter = E5LangChainEmbeddings(service)

    adapter.embed_documents(["doc one", "doc two"])
    assert model.calls[0]["texts"] == ["passage: doc one", "passage: doc two"]

    query_vector = adapter.embed_query("a question")
    assert model.calls[1]["texts"] == ["query: a question"]
    assert len(query_vector) == EMBEDDING_DIMENSION


def test_embed_texts_batch_returns_one_vector_per_text():
    """Regression: batch embedding must return n proper vectors, never a flat list.

    The previous flat pre-allocation (`[0.0] * dim * n`) returned a corrupted
    sequence (vectors mixed with leftover floats), breaking both the RAGAS
    answer_relevancy metric and any consumer expecting aligned embeddings.
    """
    service = EmbeddingService(model=RecordingModel(), settings=make_settings())
    vectors = service.embed_texts(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(
        isinstance(vector, list) and len(vector) == EMBEDDING_DIMENSION
        for vector in vectors
    )


def test_embed_texts_mixed_empty_batch_stays_aligned():
    service = EmbeddingService(model=RecordingModel(), settings=make_settings())
    vectors = service.embed_texts(["", "real", ""])
    assert len(vectors) == 3
    assert vectors[0] == [0.0] * EMBEDDING_DIMENSION
    assert isinstance(vectors[1], list) and len(vectors[1]) == EMBEDDING_DIMENSION
    assert vectors[2] == [0.0] * EMBEDDING_DIMENSION
