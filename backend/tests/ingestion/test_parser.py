"""Tests for HTML parsing and content extraction (parse_html)."""

from app.ingestion.parser import parse_html
from app.ingestion.url_utils import normalize_url


def find_by_kind(blocks, kind, text_substring=None):
    return [
        block
        for block in blocks
        if block.kind == kind and (text_substring is None or text_substring in block.text)
    ]


def test_parse_extracts_headings_paragraphs_lists_and_code():
    page = parse_html(
        """<html><head><title>Guide - Site</title></head><body><main>
        <h1>Guide</h1><p>Intro text.</p>
        <h2>Basics</h2><ul><li>First item</li><li>Second item</li></ul>
        <pre><code class="language-python">def hello():\n    return 2</code></pre>
        </main></body></html>""",
        "https://docs.example.test/guide",
    )
    assert page.title == "Guide"  # suffix after " - " removed

    kinds = [block.kind for block in page.blocks]
    assert kinds == ["h1", "text", "h2", "li", "li", "code"]

    code_block = page.blocks[-1]
    assert code_block.language == "python"
    assert code_block.text == "def hello():\n    return 2"  # newlines preserved

    assert find_by_kind(page.blocks, "li", "First item")


def test_parse_removes_boilerplate():
    page = parse_html(
        """<html><body>
        <nav>Menu item</nav><script>alert('evil')</script><style>.x{}</style>
        <header><p>Skip me</p></header>
        <main><h1>Real</h1><p>Content kept.</p></main>
        <footer><p>Contact us</p></footer>
        </body></html>""",
        "https://docs.example.test/x",
    )
    rendered = " ".join(block.text for block in page.blocks).lower()
    for junk in ("menu item", "evil", "skip me", "contact us", ".x{"):
        assert junk not in rendered
    assert "content kept." in rendered


def test_parse_prefers_main_content_region():
    html = """<html><body>
    <p>Misleading preamble outside main.</p>
    <main><h1>Main Content</h1><p>Inside main.</p></main>
    </body></html>"""
    page = parse_html(html, "https://docs.example.test/y")
    rendered = " ".join(block.text for block in page.blocks)
    assert "Inside main." in rendered
    assert "Misleading preamble outside main." not in rendered


def test_parse_falls_back_to_body_without_main():
    html = """<html><body><h1>Solo Docs</h1><p>Body-level content.</p></body></html>"""
    page = parse_html(html, "https://docs.example.test/solo")
    assert any("Body-level content." in block.text for block in page.blocks)


def test_parse_title_fallbacks():
    # no title tag -> first heading wins
    page = parse_html(
        "<html><body><h1>Fallback Title</h1><p>text</p></body></html>",
        "https://docs.example.test/fb",
    )
    assert page.title == "Fallback Title"

    # nothing at all -> URL is used, no crash
    empty = parse_html("<html><body></body></html>", "https://docs.example.test/empty")
    assert empty.title == normalize_url("https://docs.example.test/empty")
    assert empty.blocks == ()

    # truly malformed input must not raise
    broken = parse_html("<html><head><title>Half", "https://docs.example.test/broken")
    assert broken.title == "Half"
    assert all(isinstance(block.text, str) for block in broken.blocks)


def test_parse_handles_missing_code_language():
    page = parse_html(
        "<main><pre><code>SELECT 1;</code></pre></main>",
        "https://docs.example.test/sql",
    )
    code = page.blocks[-1]
    assert code.kind == "code"
    assert code.language is None
