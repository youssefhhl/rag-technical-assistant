"""Document ingestion package.

Pipeline modules (ordered, see ``pipeline.py`` for the full flow):

- ``sources``: configured documentation sources (allowlists, path prefixes)
- ``url_utils``: URL normalization, domain allowlist checks, link extraction
- ``crawler``: controlled httpx/BeautifulSoup crawler, raw HTML storage
- ``parser``: BeautifulSoup content extraction into ordered blocks
- ``preprocessor``: heading-hierarchy tracking -> sections
- ``document_builder``: sections -> LangChain Documents (with metadata)
- ``splitter``: RecursiveCharacterTextSplitter chunking (per-spec parameters)
- ``deduplication``: SQLite document registry, content hashing
- ``pipeline``: orchestration  |  ``cli``: `python -m app.ingestion.cli`
"""

from app.ingestion.sources import DOC_SOURCES, DocSource, get_source

__all__ = ["DOC_SOURCES", "DocSource", "get_source"]
