"""Persistent document registry (deduplication).

A tiny SQLite-backed registry (stdlib ``sqlite3`` — justified per the task:
lightweight, zero extra dependencies, no PostgreSQL). It prevents:

- re-fetching pages that were already ingested (the pipeline passes
  ``not registry.has_page(url)`` as the crawler's ``url_filter``), and
- re-processing the same URL when its content has not changed (content-hash
  comparison; an updated page is re-processed and the registry updated).

The logic is intentionally isolated in this module so it stays testable and
replaceable (Phase 3 may move registration state into PostgreSQL if needed).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    url            TEXT PRIMARY KEY,
    source_name    TEXT NOT NULL,
    title          TEXT NOT NULL DEFAULT '',
    content_hash   TEXT NOT NULL,
    registered_utc TEXT NOT NULL
)
"""


def content_hash(text: str) -> str:
    """Stable sha256 over line-normalized text (used for change detection)."""
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DocumentRegistry:
    """SQLite-backed registry of already-ingested documentation pages."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            settings = get_settings()
            path = settings.data_dir / "processed" / "ingestion_registry.sqlite3"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=10.0)
        self._connection.execute(_SCHEMA)
        self._connection.commit()

    def __enter__(self) -> "DocumentRegistry":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None  # type: ignore[assignment]

    def has_page(self, url: str) -> bool:
        """True when the URL was registered before (in any ingestion run)."""
        cursor = self._connection.execute(
            "SELECT 1 FROM documents WHERE url = ?", (url,)
        )
        return cursor.fetchone() is not None

    def stored_hash(self, url: str) -> str | None:
        """Return the registered content hash for a URL (or None)."""
        cursor = self._connection.execute(
            "SELECT content_hash FROM documents WHERE url = ?", (url,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def register(
        self,
        url: str,
        source_name: str,
        content_hash_value: str,
        title: str = "",
    ) -> None:
        """Insert or update a page registration."""
        self._connection.execute(
            """
            INSERT INTO documents (url, source_name, title, content_hash, registered_utc)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                source_name = excluded.source_name,
                title       = excluded.title,
                content_hash = excluded.content_hash,
                registered_utc = excluded.registered_utc
            """,
            (
                url,
                source_name,
                title,
                content_hash_value,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._connection.commit()

    def count(self, source_name: str | None = None) -> int:
        """Number of registered pages (optionally per source)."""
        if source_name is None:
            cursor = self._connection.execute("SELECT COUNT(*) FROM documents")
        else:
            cursor = self._connection.execute(
                "SELECT COUNT(*) FROM documents WHERE source_name = ?", (source_name,)
            )
        return int(cursor.fetchone()[0])
