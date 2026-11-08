"""Chunk loading, stable point IDs and the indexing pipeline.

Reuse of Phase 2 output:

- reads ``data/processed/<source>/chunks.jsonl`` (one JSON object per line:
  ``{"page_content": ..., "metadata": {...}}``) into LangChain Documents;
- generates **stable IDs** from ``(source URL, chunk_index)`` via UUIDv5, so
  re-indexing the same corpus upserts the same points (idempotent);
- embeds via the Qdrant vector store (which calls our E5 service) and reports
  statistics.

Failures are not silently swallowed: malformed JSONL lines raise a ``ValueError``
pinpointing the file and line; embedding/upsert errors propagate after logging.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# -- loading -----------------------------------------------------------------


def chunks_path_for(source_name: str, settings: Settings | None = None) -> Path:
    """Path of a source's chunks.jsonl file from Phase 2."""
    config = settings or get_settings()
    return config.data_dir / "processed" / source_name / "chunks.jsonl"


def load_chunks(path: Path) -> list[Document]:
    """Load a Phase 2 ``chunks.jsonl`` file into LangChain Documents."""
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                page_content = record["page_content"]
                metadata = record["metadata"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Malformed chunk at {path}:{line_number}: {exc}"
                ) from exc
            documents.append(Document(page_content=page_content, metadata=dict(metadata)))
    return documents


def discover_sources(settings: Settings | None = None) -> list[str]:
    """Sources that produced chunks in ``data/processed/`` (alphabetical)."""
    config = settings or get_settings()
    base = config.data_dir / "processed"
    if not base.is_dir():
        return []
    return sorted(
        child.name
        for child in base.iterdir()
        if child.is_dir() and (child / "chunks.jsonl").is_file()
    )


# -- stable IDs ----------------------------------------------------------------


def stable_point_id(source_url: str, chunk_index: int) -> uuid.UUID:
    """Deterministic UUIDv5 point ID: identical chunks always map to the same ID."""
    key = f"{source_url}::chunk::{chunk_index}"
    return uuid.uuid5(uuid.NAMESPACE_URL, key)


def chunk_ids(documents: list[Document]) -> list[uuid.UUID]:
    """Stable IDs for a list of chunk Documents."""
    ids: list[uuid.UUID] = []
    for document in documents:
        url = str(document.metadata.get("source", "unknown::url"))
        index = int(document.metadata.get("chunk_index", 0))
        ids.append(stable_point_id(url, index))
    return ids


# -- pipeline ------------------------------------------------------------------


@dataclass(slots=True)
class IndexingStats:
    """Outcome of one indexing run."""

    sources: list[str]
    documents_loaded: int = 0
    points_upserted: int = 0
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0


def index_sources(
    source_names: list[str],
    settings=None,
    store=None,
    *,
    service=None,
) -> IndexingStats:
    """Index the chunk JSONL of the given sources into the Qdrant collection."""
    from app.core.config import get_settings
    from app.vector_store.qdrant_service import build_vector_store

    config = settings or get_settings()
    started_at = datetime.now(timezone.utc)
    started_timestamp = started_at.timestamp()

    vector_store = store or build_vector_store(config, service=service)

    total_documents: int = 0
    total_upserted: int = 0
    for source_name in source_names:
        path = chunks_path_for(source_name, config)
        if not path.is_file():
            raise FileNotFoundError(
                f"No chunks for source {source_name!r}: expected {path} "
                "(run Phase 2 ingestion first)"
            )
        documents = load_chunks(path)
        if not documents:
            logger.warning("No chunks found in %s - nothing to index", path)
            continue
        logger.info("Indexing %d chunk(s) from %s ...", len(documents), path)

        # add_documents embeds in batches (E5 service) and upserts with IDs; on
        # error the exception propagates and the remaining batches are NOT
        # falsely reported as indexed.
        upserted_ids = vector_store.add_documents(documents, ids=chunk_ids(documents))
        total_upserted += len(upserted_ids)
        total_documents += len(documents)

    finished = datetime.now(timezone.utc)
    stats = IndexingStats(
        sources=list(source_names),
        documents_loaded=total_documents,
        points_upserted=total_upserted,
        started_at=started_at.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        duration_seconds=round(finished.timestamp() - started_timestamp, 3),
    )
    logger.info("Indexed %d chunk(s) into Qdrant collection", total_upserted)
    return stats
