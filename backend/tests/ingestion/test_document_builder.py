"""Tests for LangChain Document creation and metadata (document_builder)."""

from app.ingestion.document_builder import pages_to_documents, render_block
from app.ingestion.parser import Block, ParsedPage, parse_html
from app.ingestion.preprocessor import build_sections


def documents_for(html: str, url: str = "https://docs.test/compose", source_name: str = "docker-compose"):
    """Run parser -> preprocessor -> document builder for one page."""
    parsed = parse_html(html, url)
    return parsed, pages_to_documents(parsed, build_sections(parsed), source_name)


def test_document_metadata_schema():
    html = """<main><h1>Docker Compose</h1><p>Overview.</p></main>"""
    parsed, documents = documents_for(html)

    assert documents, "expected at least one document"
    metadata = documents[0].metadata
    assert metadata["source"] == "https://docs.test/compose"
    assert metadata["title"] == "Docker Compose"
    assert metadata["source_name"] == "docker-compose"
    assert metadata["document_type"] == "website"
    assert metadata["heading_hierarchy"] == ["Docker Compose"]


def test_heading_hierarchy_and_content_prefix():
    html = """<main><h1>Docker Compose</h1><p>Overview.</p>
    <h2>Services</h2><h3>Ports</h3><p>Ports body.</p></main>"""
    _, documents = documents_for(html)

    ports_doc = next(doc for doc in documents if "Ports body." in doc.page_content)
    assert ports_doc.metadata["heading_hierarchy"] == ["Docker Compose", "Services", "Ports"]
    # heading context is prefixed into the content itself (retained by the splitter)
    assert ports_doc.page_content.startswith("Docker Compose > Services > Ports\n\n")


def test_code_block_rendering_with_language():
    html = """<main><h1>Compose</h1>
    <pre><code class="language-yaml">services:\n  web: {}</code></pre></main>"""
    _, documents = documents_for(html)
    contents = "\n\n".join(doc.page_content for doc in documents)
    assert "```yaml\nservices:\n  web: {}\n```" in contents


def test_intro_only_page_uses_title_as_hierarchy():
    parsed = ParsedPage(
        url="https://docs.test/intro",
        title="Intro Page",
        blocks=(Block(kind="text", text="Only an introduction."),),
    )
    documents = pages_to_documents(parsed, build_sections(parsed), "docker-compose")
    assert len(documents) == 1
    assert documents[0].metadata["heading_hierarchy"] == ["Intro Page"]
    assert documents[0].page_content == "Intro Page\n\nOnly an introduction."


def test_render_block_fences_code():
    assert render_block(Block(kind="text", text="plain")) == "plain"
    assert render_block(Block(kind="code", text="x=1", language="python")) == "```python\nx=1\n```"
    assert render_block(Block(kind="code", text="x=1", language=None)) == "```\nx=1\n```"
