"""URL normalization and helper functions shared by the crawler.

All checks are boundary-aware so that, e.g., ``evidoctest.com`` can never
pretend to be part of the ``docs.example.com`` allowlist.
"""

from __future__ import annotations

from typing import Iterable
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup


def normalize_url(url: str) -> str:
    """Canonicalize a URL for crawling and deduplication.

    - lowercase scheme/host, strip a leading ``www.``
    - drop the fragment and any query string (doc pages are stateless)
    - collapse trailing slashes (``/x/`` == ``/x``)
    - default the scheme to ``https``
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return f"{scheme}://{host}{path}"


def is_allowed_domain(url: str, allowed_domains: Iterable[str]) -> bool:
    """Check whether a URL's host belongs to the allowlist.

    Matching is boundary-aware: ``qdrant.tech`` matches ``qdrant.tech`` and
    its subdomains, but never ``notevildomain-qdrant.tech``.
    """
    host = urlsplit(url).netloc.lower().removeprefix("www.")
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)


def matches_path_prefixes(url: str, path_prefixes: Iterable[str]) -> bool:
    """Check whether a URL path falls inside the configured prefixes (if any)."""
    prefixes = tuple(path_prefixes)
    if not prefixes:
        return True
    path = urlsplit(url).path
    return any(path == "/" or path.startswith(prefix) for prefix in prefixes)


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract absolute, normalized ``http(s)`` links from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        if urlsplit(absolute).scheme not in ("http", "https"):
            continue
        normalized = normalize_url(absolute)
        if normalized not in links:
            links.append(normalized)
    return links
