"""Embedding service for the multilingual E5 model via Sentence Transformers.

Key behaviors:

- E5 prefixes are mandatory: ``passage: `` for document chunks, ``query: `` for
  search queries (see ``docs/TECHNICAL_DECISIONS.md``, Phase 3).
- Embeddings are L2-normalized (``normalize_embeddings=True``) so cosine
  similarity works directly on the vectors.
- The model is loaded lazily on first use and cached in the instance; use
  :func:`get_embedding_service` to keep one instance per process.
- Empty/whitespace-only inputs map to zero vectors — no model call, no crash.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, Protocol

from app.core.config import Settings, get_settings

# Output dimension of intfloat/multilingual-e5-small (fixed by the architecture).
EMBEDDING_DIMENSION = 384

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Structural type for the loaded model (real: SentenceTransformer)."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        show_progress_bar: bool | None = None,
    ): ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


class EmbeddingService:
    """Reusable embedding service around one Sentence-Transformers model."""

    def __init__(
        self,
        model: EmbeddingModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Pass ``model`` to inject a pre-built (or test) model; production
        leaves it unset and the model is lazily loaded on first use.
        """
        self._settings = settings or get_settings()
        self._model_name = self._settings.embedding_model
        self._batch_size = max(1, self._settings.embedding_batch_size)
        self._model = model
        self._load_lock = threading.Lock()
        if model is not None:
            self._check_dimension(model)

    # -- model handling ------------------------------------------------------

    def _load_model(self) -> EmbeddingModel:
        from sentence_transformers import SentenceTransformer  # lazy heavy import

        logger.info("Loading embedding model %s ...", self._model_name)
        model = SentenceTransformer(self._model_name)
        self._check_dimension(model)
        return model

    @staticmethod
    def _check_dimension(model: EmbeddingModel) -> None:
        dimension = model.get_sentence_embedding_dimension()
        if dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding model dimension mismatch: expected {EMBEDDING_DIMENSION}, "
                f"got {dimension}"
            )

    @property
    def model(self) -> EmbeddingModel:
        """The underlying model — loaded once, on first access."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    # -- embedding -----------------------------------------------------------

    def embed_texts(self, texts: Iterable[str], *, for_query: bool = False) -> list[list[float]]:
        """Embed texts in configured batches, with E5 prefix + L2 normalization.

        Empty/whitespace-only original texts map to zero vectors without a model call.
        """
        prefix = _QUERY_PREFIX if for_query else _PASSAGE_PREFIX

        # Materialize once (the type is Iterable; enumerate + a comprehension
        # would otherwise consume it twice).
        texts = list(texts)

        # Separate empty/whitespace-only texts (zero vectors, no model call)
        # from non-empty texts (will be prefixed and embedded).
        empty_indices: list[int] = []
        nonempty_texts: list[str] = []
        nonempty_original_indices: list[int] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                empty_indices.append(i)
            else:
                nonempty_texts.append(prefix + text.strip())
                nonempty_original_indices.append(i)

        # One zero vector per input element (NOT a flat list of floats):
        # results[i] is always a proper vector, replaced by the real
        # embedding below. The previous flat pre-allocation corrupted every
        # multi-text batch (n vectors + leftovers mixed in one list).
        results: list[list[float]] = [
            [0.0] * EMBEDDING_DIMENSION for _ in texts
        ]

        # Fill zero vectors for empty texts (already set)
        for idx in empty_indices:
            results[idx] = [0.0] * EMBEDDING_DIMENSION

        # Embed non-empty texts in batches
        for start in range(0, len(nonempty_texts), self._batch_size):
            batch = nonempty_texts[start : start + self._batch_size]
            batch_original_indices = nonempty_original_indices[
                start : start + self._batch_size
            ]

            vectors = self.model.encode(
                batch,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for j, orig_idx in enumerate(batch_original_indices):
                results[orig_idx] = [float(v) for v in vectors[j]]

        return results

    def embed_chunks(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed document chunks — ``passage: `` prefix."""
        return self.embed_texts(texts, for_query=False)

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query — ``query: `` prefix."""
        return self.embed_texts([query], for_query=True)[0]


_SERVICE: EmbeddingService | None = None
_SERVICE_LOCK = threading.Lock()


def get_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Process-wide cached service so the model is never loaded twice."""
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = EmbeddingService(settings=settings)
    return _SERVICE
