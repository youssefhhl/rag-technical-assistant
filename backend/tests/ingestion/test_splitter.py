"""Tests for RecursiveCharacterTextSplitter chunking (splitter.py)."""

from langchain_core.documents import Document

from app.ingestion.splitter import (
    ChunkingConfig,
    build_text_splitter,
    split_document,
    split_documents,
)


def make_long_document(paragraphs: int = 20, source: str = "https://docs.test/x") -> Document:
    body = "\n\n".join(
        f"Paragraph {i} is a reasonably long sentence with several words in it."
        for i in range(paragraphs)
    )
    return Document(
        page_content="Doc > Section\n\n" + body,
        metadata={
            "source": source,
            "title": "Doc",
            "source_name": "example",
            "document_type": "website",
            "heading_hierarchy": ["Doc", "Section"],
        },
    )


def test_splitter_configuration():
    splitter = build_text_splitter()
    assert splitter._chunk_size == 1000
    assert splitter._chunk_overlap == 150
    assert splitter._separators == ["\n\n", "\n", ". ", " "]


def test_custom_configuration_is_applied():
    splitter = build_text_splitter(ChunkingConfig(chunk_size=200, chunk_overlap=20))
    assert splitter._chunk_size == 200
    assert splitter._chunk_overlap == 20


def test_long_document_is_split_with_expected_parameters():
    chunks = split_document(make_long_document(paragraphs=40))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_parent_metadata_is_preserved_and_chunk_index_is_unique():
    chunks = split_document(make_long_document(paragraphs=40))
    indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))  # unique, sequential within the parent doc

    for chunk in chunks:
        assert chunk.metadata["source"] == "https://docs.test/x"
        assert chunk.metadata["source_name"] == "example"
        assert chunk.metadata["document_type"] == "website"
        assert chunk.metadata["heading_hierarchy"] == ["Doc", "Section"]
        assert chunk.metadata["title"] == "Doc"


def test_short_document_stays_single_chunk():
    doc = Document(page_content="Short text.", metadata={"source": "https://s.test/"})
    chunks = split_document(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0


def test_split_documents_restarts_chunk_index_per_parent():
    docs = [make_long_document(), Document(page_content="tiny", metadata={"source": "u"})]
    chunks = split_documents(docs)
    first_doc_indices = [
        c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "https://docs.test/x"
    ]
    assert first_doc_indices == list(range(len(first_doc_indices)))  # restarts, not global


def test_code_block_is_kept_when_reasonably_possible():
    # ~800 chars of code -> fits inside a single 1000-char chunk
    code_body = "\n".join(f"line {i}: value_{i} = {i}  # st" for i in range(20))
    doc = Document(
        page_content="Guide > Config\n\n```yaml\n" + code_body + "\n```",
        metadata={"source": "https://s.test/", "heading_hierarchy": ["Guide", "Config"]},
    )
    chunks = split_document(doc)
    assert len(chunks) == 1
    assert chunks[0].page_content.count("```") == 2  # fence intact
    assert "line 19" in chunks[0].page_content
