"""Tests for the indexing pipeline (fake vector store, no Qdrant, no model)."""

import json

import pytest
from langchain_core.documents import Document

from app.core.config import Settings
from app.vector_store.indexing import (
    IndexingStats,
    chunks_path_for,
    discover_sources,
    index_sources,
    load_chunks,
)


class RecordingStore:
    def __init__(self, fail: bool = False) -> None:
        self.batches: list[tuple[list[Document], list]] = []
        self._fail = fail

    def add_documents(self, documents, ids=None):
        if self._fail:
            raise RuntimeError("qdrant exploded")
        self.batches.append((list(documents), ids))
        return list(ids or [])


def make_settings(tmp_path) -> Settings:
    import importlib

    config = importlib.import_module("app.core.config")
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        scrape_request_delay_seconds=0,
        qdrant_local_path=tmp_path / "qdrant",
    )


def write_chunks(tmp_path, source: str, count: int, content_template="chunk {i}") -> None:
    path = chunks_path_for(source, make_settings(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(count):
            handle.write(
                json.dumps(
                    {
                        "page_content": content_template.format(i=i),
                        "metadata": {
                            "source": f"https://docs.test/{source}#chunk{i}",
                            "title": f"{source} doc",
                            "source_name": source,
                            "document_type": "website",
                            "heading_hierarchy": [f"{source} doc"],
                            "chunk_index": i,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_chunks_path_matches_phase2_layout(tmp_path):
    assert chunks_path_for("fastapi", make_settings(tmp_path)) == (
        tmp_path / "data" / "processed" / "fastapi" / "chunks.jsonl"
    )


def test_load_chunks_returns_documents_with_metadata(tmp_path):
    write_chunks(tmp_path, "test", 2)
    path = chunks_path_for("test", make_settings(tmp_path))
    documents = load_chunks(path)

    assert len(documents) == 2
    assert documents[0].page_content == "chunk 0"
    assert documents[0].metadata["source_name"] == "test"
    assert documents[1].metadata["chunk_index"] == 1


def test_load_chunks_skips_blank_lines_and_rejects_malformed_json(tmp_path):
    write_chunks(tmp_path, "test", 1)
    path = chunks_path_for("test", make_settings(tmp_path))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write("not json at all\n")
        handle.write(json.dumps({"page_content": "missing metadata"}) + "\n")

    documents = load_chunks(path)
    assert len(documents) == 1  # blank line skipped

    with pytest.raises(ValueError, match=r"test.jsonl:3"):
        load_chunks(path)  # malformed line number is pinpointed (no silent loss)


def test_discover_sources_lists_only_processed_sources(tmp_path):
    write_chunks(tmp_path, "alpha", 1)
    write_chunks(tmp_path, "beta", 2)
    empty = make_settings(tmp_path).data_dir / "processed" / "gamma"
    empty.mkdir(parents=True)  # no chunks.jsonl inside

    assert discover_sources(make_settings(tmp_path)) == ["alpha", "beta"]


def test_index_sources_loads_and_upserts_with_stable_ids(tmp_path):
    write_chunks(tmp_path, "alpha", 3)
    store = RecordingStore()

    stats = index_sources(["alpha"], settings=make_settings(tmp_path), store=store)

    assert isinstance(stats, IndexingStats)
    assert stats.sources == ["alpha"]
    assert stats.documents_loaded == 3
    assert stats.points_upserted == 3

    documents, ids = store.batches[0]
    assert [document.page_content for document in documents] == ["chunk 0", "chunk 1", "chunk 2"]
    assert len(set(ids)) == 3  # all unique upsert ids
    # re-calling with the same corpus reuses the same IDs (idempotent upsert)
    index_sources(["alpha"], settings=make_settings(tmp_path), store=store)
    assert store.batches[0][1] == store.batches[1][1]


def test_index_sources_embeds_in_passage_mode(tmp_path):
    """add_documents receives Documents only — embedding happens inside the store."""
    write_chunks(tmp_path, "alpha", 2)
    store = RecordingStore()
    index_sources(["alpha"], settings=make_settings(tmp_path), store=store)
    assert all(document.page_content for document in store.batches[0][0])


def test_index_sources_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Phase 2 ingestion first"):
        index_sources(["does-not-exist"], settings=make_settings(tmp_path), store=RecordingStore())


def test_index_sources_indexing_failure_is_not_swallowed(tmp_path):
    write_chunks(tmp_path, "alpha", 2)
    store = RecordingStore(fail=True)
    with pytest.raises(RuntimeError, match="qdrant exploded"):
        index_sources(["alpha"], settings=make_settings(tmp_path), store=store)
    assert store.batches == []  # nothing falsely reported as indexed


def test_index_sources_multiple_sources_in_order(tmp_path):
    write_chunks(tmp_path, "beta", 1)
    write_chunks(tmp_path, "alpha", 2)
    store = RecordingStore()
    stats = index_sources(["beta", "alpha"], settings=make_settings(tmp_path), store=store)
    assert stats.documents_loaded == 3
    assert stats.points_upserted == 3
    assert stats.sources == ["beta", "alpha"]
