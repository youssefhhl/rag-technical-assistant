"""Tests for URL normalization and allowlist/path-prefix helpers."""

import pytest

from app.ingestion.url_utils import (
    extract_links,
    is_allowed_domain,
    matches_path_prefixes,
    normalize_url,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://WWW.Example.com/Docs/Intro/", "https://example.com/Docs/Intro"),
        ("https://docs.example.com/page?q=1&ref=x#section", "https://docs.example.com/page"),
        ("https://docs.example.com/", "https://docs.example.com/"),
        ("  https://example.com/page#frag  ", "https://example.com/page"),
        ("http://www.wwwexample.com", "http://wwwexample.com/"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


def test_normalize_url_distinguishes_case_in_path() -> None:
    assert normalize_url("https://x.test/A") != normalize_url("https://x.test/a")


def test_is_allowed_domain_boundary_aware() -> None:
    allowed = ("docs.example.com", "qdrant.tech")
    assert is_allowed_domain("https://docs.example.com/guide", allowed)
    assert is_allowed_domain("https://docs.example.com", allowed)
    # subdomains of an allowed domain are allowed
    assert is_allowed_domain("https://qdrant.tech/documentation", allowed)
    # look-alike domains are NOT allowed (no naive `in` match)
    assert not is_allowed_domain("https://notevildomain-doctest.com", ("doctest.com",))
    assert not is_allowed_domain("https://other.example.org", allowed)


def test_matches_path_prefixes() -> None:
    assert matches_path_prefixes("https://x.test/docs/a", ("/docs",))
    assert matches_path_prefixes("https://x.test/", ("/docs",))  # seed itself
    assert matches_path_prefixes("https://x.test/anywhere", ())  # no restriction
    assert not matches_path_prefixes("https://x.test/compose", ("/docs",))


def test_extract_links_resolves_and_filters() -> None:
    html = """
    <html><body>
      <a href="/docs/a.html">A</a>
      <a href="b.html">relative</a>
      <a href="#anchor">fragment only</a>
      <a href="mailto:x@y.z">mail</a>
      <a href="javascript:void(0)">js</a>
      <a href="https://WWW.other.com/p#f">other host</a>
      <a href="/docs/a.html#dup">duplicate</a>
    </body></html>
    """
    links = extract_links(html, "https://base.test/docs/index.html")
    assert "https://base.test/docs/a.html" in links
    assert "https://base.test/docs/b.html" in links
    assert "https://other.com/p" in links
    # duplicates and non-http(s) targets are removed
    assert links.count("https://base.test/docs/a.html") == 1
    assert all(not link.startswith("#") for link in links)
