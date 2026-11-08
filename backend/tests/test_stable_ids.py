"""Tests for stable Qdrant point ID generation (UUIDv5 over URL + chunk_index)."""

import uuid

from langchain_core.documents import Document

from app.vector_store.indexing import chunk_ids, stable_point_id


def test_stable_id_is_deterministic():
    first = stable_point_id("https://docs.test/a.html", 3)
    second = stable_point_id("https://docs.test/a.html", 3)
    assert first == second


def test_stable_id_differs_per_chunk_and_url():
    base = stable_point_id("https://docs.test/a.html", 0)
    assert stable_point_id("https://docs.test/a.html", 1) != base
    assert stable_point_id("https://docs.test/b.html", 0) != base


def test_stable_id_is_a_valid_uuid5():
    point_id = stable_point_id("https://docs.test/a.html", 0)
    assert isinstance(point_id, uuid.UUID)
    expected = uuid.uuid5(uuid.NAMESPACE_URL, "https://docs.test/a.html::chunk::0")
    assert point_id == expected


def test_chunk_ids_derive_from_chunk_metadata():
    documents = [
        Document(page_content="a", metadata={"source": "https://docs.test/x", "chunk_index": 0}),
        Document(page_content="b", metadata={"source": "https://docs.test/x", "chunk_index": 1}),
        Document(page_content="c", metadata={"source": "https://docs.test/y", "chunk_index": 0}),
    ]
    ids = chunk_ids(documents)

    assert len(ids) == 3
    assert len(set(ids)) == 3  # all unique
    assert ids[0] == stable_point_id("https://docs.test/x", 0)
    assert ids[2] == stable_point_id("https://docs.test/y", 0)


def test_chunk_ids_never_crash_on_missing_metadata():
    documents = [Document(page_content="no metadata")]
    ids = chunk_ids(documents)
    assert ids == [stable_point_id("unknown::url", 0)]