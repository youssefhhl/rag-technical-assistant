"""Build LangChain Documents from parsed pages and their preprocessed sections.

One :class:`~langchain_core.documents.Document` is produced per non-empty
section. The heading hierarchy is both:

- kept as ``heading_hierarchy`` metadata, and
- prefixed into ``page_content`` ("H1 > H2 > H3\\n\\n content") so the heading
  context travels inside the text and survives chunking.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.ingestion.parser import ParsedPage, Block
from app.ingestion.preprocessor import Section


def render_block(block: Block) -> str:
    """Render one block as markdown-ish text, preserving code blocks."""
    if block.kind == "code":
        fence = block.language or ""
        return f"```{fence}\n{block.text}\n```"
    return block.text


def build_section_document(
    page: ParsedPage, section: Section, source_name: str
) -> Document:
    """Build one LangChain Document for a single section."""
    hierarchy = section.heading_path or (page.title,)
    heading_prefix = " > ".join(hierarchy)
    body = "\n\n".join(
        render_block(block) for block in section.blocks if block.text.strip()
    )
    page_content = f"{heading_prefix}\n\n{body}" if body else heading_prefix

    metadata = {
        "source": page.url,
        "title": page.title,
        "source_name": source_name,
        "document_type": "website",
        "heading_hierarchy": list(hierarchy),
    }
    return Document(page_content=page_content, metadata=metadata)


def pages_to_documents(
    page: ParsedPage, sections: list[Section], source_name: str
) -> list[Document]:
    """Convert one parsed page into one Document per non-empty section."""
    return [
        build_section_document(page, section, source_name)
        for section in sections
        if not section.is_empty
    ]
