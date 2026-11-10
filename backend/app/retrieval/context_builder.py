"""Context Builder for RAG — formats reranked results into a grounded prompt.

Takes the top-5 reranked chunks from HybridSearchService and produces:
- A prompt string for the LLM with clear instructions
- A list of source citations (title, URL, chunk_index, relevance score)
- The raw content preserved for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

from app.retrieval.hybrid_search import HybridHit


# -- metadata mapping ----------------------------------------------------------
#
# The retrieval engines (Qdrant dense, BM25 sparse) both expose the *same flat*
# metadata contract on every hit:
#
#     {"title": str, "source": str, "chunk_index": int,
#      "heading_hierarchy": list[str], ...}
#
# The helpers below map that contract onto the citation fields.  They never
# substitute a placeholder for missing data: an absent field stays empty so a
# caller can tell "unknown" apart from a real value.

def _clean(value: Any) -> str:
    """Normalise a metadata value to a stripped string ("" when absent)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def resolve_document_title(metadata: dict | None) -> str:
    """The document title of a chunk, without any section path."""
    md = metadata or {}
    return _clean(md.get("title"))


def resolve_section(metadata: dict | None) -> str:
    """The section path of a chunk, e.g. ``"FastAPI > Performance"``."""
    md = metadata or {}
    headings = md.get("heading_hierarchy") or []
    if isinstance(headings, str):
        headings = [headings]
    return " > ".join(part for part in (_clean(h) for h in headings) if part)


def resolve_citation_title(metadata: dict | None) -> str:
    """Human-readable title used for citations.

    Uses the real document title and appends the section path when the chunk
    comes from a section of a larger page, so several chunks of the same page
    stay distinguishable.  Never returns a placeholder.
    """
    title = resolve_document_title(metadata)
    section = resolve_section(metadata)
    if not section:
        return title
    if not title:
        return section
    if section == title or section.startswith(f"{title} >"):
        return section
    return f"{title} > {section}"


def resolve_source_url(metadata: dict | None) -> str:
    """The original document URL of a chunk ("" when unknown)."""
    return _clean((metadata or {}).get("source"))


def resolve_chunk_index(metadata: dict | None) -> int:
    """The chunk index inside its source document (0 when unknown)."""
    raw = (metadata or {}).get("chunk_index", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class ContextEntry:
    """One chunk entry in the RAG context."""
    content: str
    title: str
    source_url: str
    chunk_index: int
    rrf_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None

    def attribution_text(self) -> str:
        """Human-readable attribution for this chunk."""
        return f"  {self.title} - {self.source_url} (chunk {self.chunk_index})\n"


@dataclass(slots=True)
class ContextBuilder:
    """Builds a RAG-context block from HybridHit results."""

    hits: list[HybridHit]
    max_chunks: int = 5

    def build(self) -> tuple[str, list[ContextEntry]]:
        """Return (prompt_block, entries) ready for the RAG prompt."""
        entries: list[ContextEntry] = []
        # Take up to max_chunks hits, preserving order
        for hit in self.hits[: self.max_chunks]:
            md = dict(getattr(hit, "metadata", {}) or {})
            entries.append(
                ContextEntry(
                    content=str(getattr(hit, "content", "") or ""),
                    title=resolve_citation_title(md),
                    source_url=resolve_source_url(md),
                    chunk_index=resolve_chunk_index(md),
                    rrf_score=getattr(hit, "rrf_score", None),
                    vector_rank=getattr(hit, "vector_rank", None),
                    bm25_rank=getattr(hit, "bm25_rank", None),
                )
            )

        # Build a compact prompt block: each chunk with its attribution.
        lines: list[str] = []
        for i, entry in enumerate(entries, start=1):
            lines.append(
                f"[chunk {i}] {entry.title}\n"
                f"  source: {entry.source_url} (chunk {entry.chunk_index})\n"
                f"  {entry.content[:500]}\n"
            )
        prompt_block = "\n".join(lines) if lines else ""

        return prompt_block, entries


# -- prompt template -----------------------------------------------------------


RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for the Smart Knowledge Assistant RAG system.

Instructions:
- Answer the user's question using ONLY the context provided below.
- If the context does not contain the answer, clearly state: "I don't have enough information to answer this question based on the provided documents."
- Do not use any outside knowledge or hallucinate.
- Cite your sources after each answer point using the format: [(source: TITLE - chunk N)].
- Keep the answer concise and direct.

---

Context:
{context}

---

User question: {question}

Answer:
"""