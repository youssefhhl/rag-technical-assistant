"""Structure-aware preprocessing: attach heading hierarchy to extracted blocks.

Tracks the current H1/H2/H3 stack while walking the ordered blocks of a
:class:`~app.ingestion.parser.ParsedPage`, grouping every paragraph, list item
and code block with the hierarchy it belongs to::

    H1 Docker Compose / H2 Services / H3 Ports

        -> Section(heading_path=("Docker Compose", "Services", "Ports"))

This module does *not* chunk text; it only preserves document structure so the
chunking step can attach section context to every chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.parser import Block, ParsedPage


@dataclass(frozen=True, slots=True)
class Section:
    """A heading-bounded slice of a documentation page."""

    heading_path: tuple[str, ...]  # e.g. ("Docker Compose", "Services", "Ports")
    blocks: tuple[Block, ...] = field(default=tuple())

    @property
    def is_empty(self) -> bool:
        return not any(block.text.strip() for block in self.blocks)

    @property
    def display_path(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else "(introduction)"


def build_sections(page: ParsedPage) -> list[Section]:
    """Group the page's blocks into sections bounded by H1/H2/H3 headings."""
    sections: list[Section] = []
    heading_stack: list[str] = []
    buffer: list[Block] = []

    def flush() -> None:
        # At least the page title is kept as document context, so even content
        # before any heading has a traceable hierarchy.
        path = tuple(heading_stack) or (page.title,)
        if buffer:
            sections.append(Section(heading_path=path, blocks=tuple(buffer)))
            buffer.clear()

    for block in page.blocks:
        if block.kind == "h1":
            flush()
            heading_stack = [block.text]
        elif block.kind == "h2":
            flush()
            base = [heading_stack[0]] if heading_stack else [page.title]
            heading_stack = [*base, block.text]
        elif block.kind == "h3":
            flush()
            if heading_stack:
                heading_stack = [*heading_stack[:2], block.text]
            else:
                heading_stack = [page.title, block.text]
        else:
            buffer.append(block)
    flush()

    return [section for section in sections if not section.is_empty]
