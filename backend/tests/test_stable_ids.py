"""Tests for stable Qdrant point ID generation.

Regression note: real Phase 2 metadata has per-page ``chunk_index`` (restarts
at 0 each page), so the key MUST include the content fingerprint — otherwise
uuid5(source, chunk_index) collides (20 keys for 126 real chunks).
"""

import hashlib
import uuid

from langchain_core.documents import Document

from app.vector_store.indexing import chunk_ids, stable_point_id


def test_stable_id_is_deterministic():
    first = stable_point_id("https://docs.test/a.html", 3, "same content")
    second = stable_point_id("https://docs.test/a.html", 3, "same content")
    assert first == second


def test_stable_id_differs_per_chunk_index_and_url_without_content():
    base = stable_point_id("https://docs.test/a.html", 0)
    assert stable_point_id("https://docs.test/a.html", 1) != base
    assert stable_point_id("https://docs.test/b.html", 0) != base


def test_stable_id_is_a_valid_uuid5():
    point_id = stable_point_id("https://docs.test/a.html", 0)
    assert isinstance(point_id, uuid.UUID)
    expected = uuid.uuid5(uuid.NAMESPACE_URL, "https://docs.test/a.html::chunk::0")
    assert point_id == expected


def test_stable_id_includes_content_fingerprint():
    point_id = stable_point_id("https://docs.test/a.html", 0, "real text")
    normalized = " ".join("real text".split())
    fingerprint = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    expected = uuid.uuid5(
        uuid.NAMESPACE_URL, f"https://docs.test/a.html::chunk::0::{fingerprint}"
    )
    assert point_id == expected


def test_stable_id_ignores_whitespace_differences():
    """Whitespace-insensitive: same chunk re-indexed after reformat keeps its ID."""
    base = stable_point_id("https://docs.test/a.html", 0, "hello   world")
    assert stable_point_id("https://docs.test/a.html", 0, "hello world") == base
    assert stable_point_id("https://docs.test/a.html", 0, "hello\nworld ") == base


def test_chunk_ids_unique_even_when_metadata_collides():
    """REGRESSION: same (source, chunk_index) but different contents -> unique IDs."""
    documents = [
        Document(
            page_content="License page chunk",
            metadata={"source": "https://fastapi.tiangolo.com/", "chunk_index": 0},
        ),
        Document(
            page_content="Dependencies chunk on the same page",
            metadata={"source": "https://fastapi.tiangolo.com/", "chunk_index": 0},
        ),
        Document(
            page_content="Opinions chunk on the same page",
            metadata={"source": "https://fastapi.tiangolo.com/", "chunk_index": 0},
        ),
    ]

    ids = chunk_ids(documents)

    assert len(set(ids)) == 3  # was 1 before the content fingerprint
    assert ids == [
        stable_point_id("https://fastapi.tiangolo.com/", 0, "License page chunk"),
        stable_point_id("https://fastapi.tiangolo.com/", 0, "Dependencies chunk on the same page"),
        stable_point_id("https://fastapi.tiangolo.com/", 0, "Opinions chunk on the same page"),
    ]


def test_chunk_ids_stable_across_reindexing():
    documents_first = [
        Document(page_content="chunk alpha", metadata={"source": "https://x.test", "chunk_index": 7})
    ]
    documents_second = [
        Document(page_content="chunk alpha", metadata={"source": "https://x.test", "chunk_index": 7})
    ]
    assert chunk_ids(documents_first) == chunk_ids(documents_second)


def test_chunk_ids_never_crash_on_missing_metadata():
    documents = [Document(page_content="no metadata content")]
    ids = chunk_ids(documents)
    assert ids == [stable_point_id("unknown::url", 0, "no metadata content")]
