"""BM25 search over the persisted sparse index (Phase 4 - pre-hybrid step).

Loads the bundle built by ``bm25_indexing`` and returns the top-k most
relevant chunks. Fusion with the dense (Qdrant + E5) path happens in the next
phase (RRF / hybrid), intentionally not implemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.retrieval.bm25_indexing import Bm25IndexBundle, load_bm25_index, tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Bm25Hit:
    """One ranked BM25 result; original content and metadata are preserved."""

    content: str
    metadata: dict
    bm25_score: float

    # Convenience fields extracted from the metadata (kept out of the dict).
    @property
    def source_url(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))

    @property
    def chunk_index(self) -> int:
        return int(self.metadata.get("chunk_index", 0))

    @property
    def heading_hierarchy(self) -> list[str]:
        return list(self.metadata.get("heading_hierarchy", []))

    def format(self) -> str:
        hierarchy = " > ".join(self.heading_hierarchy) or self.title
        return (
            f"[bm25 {self.bm25_score:.4f}] {self.title} — {hierarchy}\n"
            f"  source: {self.source_url} (chunk {self.chunk_index})\n"
            f"  {self.content[:300]}\n"
        )


class Bm25SearchService:
    """Top-k BM25 lexical search over the loaded index bundle."""

    def __init__(self, bundle: Bm25IndexBundle) -> None:
        self._bundle = bundle

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, path: Path | None = None
    ) -> "Bm25SearchService":
        """Load the persisted index from ``data/indexes/bm25/``."""
        return cls(load_bm25_index(settings, path=path))

    def search(self, query: str, k: int = 5) -> list[Bm25Hit]:
        """Rank chunks by BM25 score against ``query`` (matches with score 0 excluded)."""
        if not query.strip():
            return []
        if k < 1:
            return []

        scores = self._bundle.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits: list[Bm25Hit] = []
        for index in ranked:
            score = float(scores[index])
            if score <= 0.0:
                break
            document = self._bundle.documents[index]
            hits.append(
                Bm25Hit(content=document.content, metadata=document.metadata, bm25_score=score)
            )
            if len(hits) == k:
                break
        return hits
