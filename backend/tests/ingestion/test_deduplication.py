"""Tests for the SQLite document registry (deduplication)."""

import pytest

from app.ingestion.deduplication import DocumentRegistry, content_hash
from app.core.config import get_settings


def test_content_hash_is_stable_and_trailing_whitespace_insensitive():
    assert content_hash("line1\nline2") == content_hash("line1\nline2  ")
    assert content_hash("a  \nb") == content_hash("a\nb")  # per-line rstrip
    assert content_hash("a\nb") != content_hash("a\nb\nextra")  # content matters


def test_register_and_has_page(registry):
    assert not registry.has_page("https://docs.test/a")
    registry.register("https://docs.test/a", "example", "hash1", "Title A")
    assert registry.has_page("https://docs.test/a")
    assert registry.stored_hash("https://docs.test/a") == "hash1"


def test_register_updates_hash_when_content_changes(registry):
    registry.register("https://docs.test/a", "example", "old-hash")
    registry.register("https://docs.test/a", "example", "new-hash", "Better Title")
    assert registry.stored_hash("https://docs.test/a") == "new-hash"
    assert registry.count() == 1  # upsert, no duplicate row


def test_count_per_source(registry):
    registry.register("https://a.test/1", "fastapi", "h1")
    registry.register("https://b.test/1", "qdrant", "h2")
    registry.register("https://a.test/2", "fastapi", "h3")
    assert registry.count() == 3
    assert registry.count("fastapi") == 2
    assert registry.count("react") == 0


def test_registries_are_isolated_per_file(tmp_path):
    reg_a = DocumentRegistry(tmp_path / "a.sqlite3")
    reg_b = DocumentRegistry(tmp_path / "b.sqlite3")
    try:
        reg_a.register("https://x.test/", "fastapi", "h")
        assert reg_a.has_page("https://x.test/")
        assert not reg_b.has_page("https://x.test/")
    finally:
        reg_a.close()
        reg_b.close()


def test_in_memory_registry():
    with DocumentRegistry(":memory:") as registry:
        registry.register("https://docs.test/", "fastapi", "h")
        assert registry.has_page("https://docs.test/")
    # closed registry no longer usable
    with pytest.raises(Exception):
        registry.has_page("https://docs.test/")


def test_default_path_is_created_lazily(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "custom-data"))
    get_settings.cache_clear()  # bypass the cached singleton from earlier tests
    registry = DocumentRegistry()
    try:
        expected = tmp_path / "custom-data" / "processed" / "ingestion_registry.sqlite3"
        assert registry.path == expected
    finally:
        registry.close()
        get_settings.cache_clear()
