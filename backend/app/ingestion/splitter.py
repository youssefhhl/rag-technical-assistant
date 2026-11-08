"""Chunking step: RecursiveCharacterTextSplitter (the one and only algorithm).

Initial configuration (per ``docs/ROADMAP.md`` Phase 2, to be re-evaluated in
the evaluation phase):

- chunk size:      1000 characters
- chunk overlap:   150 characters
- separators:      ``\\n\\n`` -> ``\\n`` -> ``'. '`` -> ``' '``

Code blocks are rendered as fenced blocks (see ``document_builder``); the
``\\n\\n``/``\\n`` separators usually keep short-to-medium code blocks intact,
and long ones are split per line rather than mid-token — the reasonable-promise
"avoid splitting code whenever reasonably possible" within a single splitter.

Splitting is done per parent Document so each child carries:

- all parent metadata (source URL, title, source_name, document_type,
  heading_hierarchy), and
- a fresh ``chunk_index`` (0..n-1 within the parent Document) making each chunk
  uniquely traceable via ``source`` + ``chunk_index`` + ``heading_hierarchy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Initial (default) chunking parameters."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


_DEFAULT_CONFIG = ChunkingConfig()


def build_text_splitter(config: ChunkingConfig | None = None) -> RecursiveCharacterTextSplitter:
    """Create the configured ``RecursiveCharacterTextSplitter``."""
    cfg = config or _DEFAULT_CONFIG
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=list(cfg.separators),
        keep_separator=True,
    )


def split_document(
    document: Document,
    config: ChunkingConfig | None = None,
) -> list[Document]:
    """Split a single Document, stamping per-document ``chunk_index`` values."""
    splitter = build_text_splitter(config)
    chunks = splitter.split_documents([document])
    indexed: list[Document] = []
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        indexed.append(chunk)
    return indexed


def split_documents(
    documents: Iterable[Document],
    config: ChunkingConfig | None = None,
) -> list[Document]:
    """Split each Document independently (chunk_index restarts per Document)."""
    chunks: list[Document] = []
    for document in documents:
        chunks.extend(split_document(document, config))
    return chunks
