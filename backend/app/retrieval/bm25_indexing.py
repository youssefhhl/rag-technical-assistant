"""BM25 sparse retrieval — inverted index over the Phase 2 chunks.

Independent of the Qdrant/E5 vector store (Phase 4 Hybrid Search groundwork):

- tokenizes and indexes the chunks stored in ``data/processed/<source>/chunks.jsonl``
  (same on-disk format as Phase 2 / Phase 3);
- persists a ready-to-search bundle (rank_bm25 BM25Okapi + the original chunk
  contents and metadata) under ``data/indexes/bm25/`` so searches do not
  re-tokenize the corpus on every query;
- reformatted/re-indexed corpora are rebuilt explicitly via :func:`build_bm25_index`.

No RRF / hybrid fusion here — that is the next phase.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Multilingual-friendly tokenizer: lowercase + Unicode word tokens.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: BM25 bundle file (one pickle, all sources).
BM25_INDEX_FILENAME = "bm25_index.pkl"

#: Version bumped when the on-disk format changes (invalidate old pickles).
BUNDLE_VERSION = 1


def tokenize(text: str) -> list[str]:
    """Lowercase + ``\\w+`` token extraction (works with accented text)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class Bm25Document:
    """One indexed chunk with its **original** content and metadata preserved."""

    content: str
    metadata: dict


# -- chunk loading (kept independent from app.vector_store) --------------------


def chunks_path_for(source_name: str, settings: Settings | None = None) -> Path:
    """Phase 2 chunks.jsonl of the given source."""
    config = settings or get_settings()
    return config.data_dir / "processed" / source_name / "chunks.jsonl"


def discover_sources(settings: Settings | None = None) -> list[str]:
    """Sources with chunks under ``data/processed/`` (alphabetical)."""
    config = settings or get_settings()
    base = config.data_dir / "processed"
    if not base.is_dir():
        return []
    return sorted(
        child.name
        for child in base.iterdir()
        if child.is_dir() and (child / "chunks.jsonl").is_file()
    )


def load_chunks(path: Path) -> list[Bm25Document]:
    """Load one ``chunks.jsonl`` into :class:`Bm25Document` records."""
    documents: list[Bm25Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                documents.append(
                    Bm25Document(
                        content=record["page_content"],
                        metadata=dict(record["metadata"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Malformed chunk at {path}:{line_number}: {exc}"
                ) from exc
    return documents


# -- index build / persistence --------------------------------------------------


@dataclass(slots=True)
class Bm25IndexBundle:
    """Persisted, directly searchable BM25 bundle."""

    documents: list[Bm25Document]
    bm25: object  # rank_bm25.BM25Okapi
    tokenized_corpus: list[list[str]]
    source_names: list[str]
    built_at: str
    version: int = BUNDLE_VERSION
    parameters: dict = field(default_factory=dict)


def index_path(settings: Settings | None = None) -> Path:
    config = settings or get_settings()
    return config.data_dir / "indexes" / "bm25" / BM25_INDEX_FILENAME


def build_bm25_index(
    settings: Settings | None = None,
    source_names: Iterable[str] | None = None,
    *,
    bm25_factory: Callable[[list[list[str]]], object] | None = None,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> Bm25IndexBundle:
    """Index the given (or every discoverable) source and persist the bundle.

    Returns the in-memory bundle. Uses :func:`discover_sources` when
    ``source_names`` is omitted. Both callables are injectable for tests.
    """
    from rank_bm25 import BM25Okapi  # lightweight, pure Python

    config = settings or get_settings()
    names = list(source_names) if source_names is not None else discover_sources(config)
    if not names:
        raise FileNotFoundError(
            "No chunks found under data/processed/ - run Phase 2 ingestion first"
        )

    token_fn = tokenizer or tokenize
    documents: list[Bm25Document] = []
    for source_name in names:
        path = chunks_path_for(source_name, config)
        if not path.is_file():
            raise FileNotFoundError(
                f"No chunks for source {source_name!r}: expected {path} "
                "(run Phase 2 ingestion first)"
            )
        loaded = load_chunks(path)
        if not loaded:
            logger.warning("No chunks found in %s - skipping %s", path, source_name)
            continue
        documents.extend(loaded)
        logger.info("BM25: indexed %d chunk(s) from %s", len(loaded), path)

    if not documents:
        raise ValueError("No indexable chunks found (all sources were empty)")

    tokenized_corpus = [token_fn(document.content) for document in documents]
    bundle = Bm25IndexBundle(
        documents=documents,
        bm25=bm25_factory(tokenized_corpus) if bm25_factory else BM25Okapi(tokenized_corpus),
        tokenized_corpus=tokenized_corpus,
        source_names=[n for n in names if (chunks_path_for(n, config)).is_file()],
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        parameters={
            "architecture": "BM25Okapi",
            "tokenizer": "lowercase+\\w+",
            "corpus_size": len(documents),
        },
    )
    save_bm25_index(bundle, config)
    return bundle


def save_bm25_index(bundle: Bm25IndexBundle, settings: Settings | None = None) -> Path:
    """Persist the bundle under ``data/indexes/bm25/``."""
    path = index_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("BM25 index saved to %s (%d documents)", path, len(bundle.documents))
    return path


def load_bm25_index(
    settings: Settings | None = None,
    *,
    path: Path | None = None,
) -> Bm25IndexBundle:
    """Load the persisted bundle (raises if it was never built)."""
    config = settings or get_settings()
    target = path or index_path(config)
    if not target.is_file():
        raise FileNotFoundError(
            f"BM25 index not found at {target} - build it first with "
            "build_bm25_index()"
        )
    with target.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, Bm25IndexBundle) or bundle.version != BUNDLE_VERSION:
        raise ValueError(
            f"Incompatible BM25 index at {target} - rebuild it via build_bm25_index()"
        )
    logger.info("BM25 index loaded from %s", target)
    return bundle
