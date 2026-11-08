"""Ingestion pipeline orchestration.

Connects the stages of the ingestion pipeline:

1. select a documentation source (``sources.py``)
2. crawl its pages (``crawler.py``; already-known URLs are skipped via the
   deduplication registry *before* the request)
3. raw HTML persisted under ``<data_dir>/raw/<source>/`` (crawler)
4. parse & clean (``parser.py``)
5. preserve heading hierarchy (``preprocessor.py``)
6. build LangChain Documents (``document_builder.py``)
7. split into chunks with RecursiveCharacterTextSplitter (``splitter.py``)
8. deduplicate pages (registry) and identical chunk texts (in-run, hash-based)
9. return the chunks and persist them under
   ``<data_dir>/processed/<source>/chunks.jsonl`` for the embedding phase.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.ingestion.crawler import Crawler
from app.ingestion.deduplication import DocumentRegistry, content_hash
from app.ingestion.document_builder import pages_to_documents
from app.ingestion.parser import parse_html
from app.ingestion.preprocessor import build_sections
from app.ingestion.sources import get_source
from app.ingestion.splitter import split_documents

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    """Outcome of one ingestion run for a single source."""

    source_name: str
    pages_fetched: int = 0  # downloaded (raw HTML saved) in this run
    pages_processed: int = 0  # parsed, chunked and registered in this run
    pages_skipped: int = 0  # skipped because unchanged / already registered
    documents_built: int = 0  # section-level LangChain Documents
    chunks_deduplicated: int = 0  # duplicate chunk texts dropped in-run
    chunks: list[Document] = field(default_factory=list)
    chunks_path: Path | None = None


class IngestionPipeline:
    """End-to-end ingestion of one documentation source."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: DocumentRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._external_registry = registry

    def run(self, source_name: str) -> IngestionResult:
        """Ingest a documentation source by name (e.g. ``"fastapi"``)."""
        source = get_source(source_name)
        result = IngestionResult(source_name=source.name)

        registry = self._open_registry()
        owns_registry = self._external_registry is None
        try:
            crawler = Crawler(source, self.settings)
            try:
                records = crawler.crawl(
                    url_filter=lambda url: not registry.has_page(url)
                )
            finally:
                crawler.close()
            result.pages_fetched = len(records)

            all_documents: list[Document] = []
            for record in records:
                parsed = parse_html(record.html, record.url)
                page_text_hash = content_hash(parsed.text_hash_basis)

                stored = registry.stored_hash(record.url)
                if stored == page_text_hash:
                    logger.debug("Unchanged page, skipping: %s", record.url)
                    result.pages_skipped += 1
                    continue

                sections = build_sections(parsed)
                documents = pages_to_documents(parsed, sections, source.name)
                result.documents_built += len(documents)
                all_documents.extend(documents)

                registry.register(
                    url=record.url,
                    source_name=source.name,
                    content_hash_value=page_text_hash,
                    title=parsed.title,
                )
                result.pages_processed += 1

            result.chunks = self._dedupe_chunks(split_documents(all_documents), result)
            if all_documents:
                result.chunks_path = self._persist_chunks(source.name, result.chunks)
            else:
                logger.debug(
                    "No new documents for %r - existing chunks file left untouched",
                    source.name,
                )
        finally:
            if owns_registry:
                registry.close()

        logger.info(
            "Ingestion of %r done: fetched=%d processed=%d skipped=%d "
            "documents=%d chunks=%d (duplicates dropped in-run=%d)",
            source.name,
            result.pages_fetched,
            result.pages_processed,
            result.pages_skipped,
            result.documents_built,
            len(result.chunks),
            result.chunks_deduplicated,
        )
        return result

    def _dedupe_chunks(
        self, chunks: list[Document], result: IngestionResult
    ) -> list[Document]:
        """Drop identical chunk texts within one run (hash-based, isolated)."""
        kept: list[Document] = []
        seen_hashes: set[str] = set()
        for chunk in chunks:
            digest = content_hash(chunk.page_content)
            if digest in seen_hashes:
                result.chunks_deduplicated += 1
                continue
            seen_hashes.add(digest)
            kept.append(chunk)
        return kept

    def _persist_chunks(
        self, source_name: str, chunks: list[Document]
    ) -> Path:
        """Write chunks as JSONL for the embedding phase (Phase 3)."""
        out_dir = self.settings.data_dir / "processed" / source_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(
                    json.dumps(
                        {
                            "page_content": chunk.page_content,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return out_path

    def _open_registry(self) -> DocumentRegistry:
        """Create (or reuse) the registry; only internally-owned ones get closed."""
        if self._external_registry is not None:
            return self._external_registry
        return DocumentRegistry(
            self.settings.data_dir / "processed" / "ingestion_registry.sqlite3"
        )
