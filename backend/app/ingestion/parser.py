"""HTML parsing and content extraction for documentation pages.

Extracts the main documentation content of a page as an ordered list of
:class:`Block` items (headings, paragraphs, list items, code blocks) while
removing navigation menus, scripts, styles, headers, footers and other
boilerplate. The preprocessor (``preprocessor.py``) then attaches the heading
hierarchy to these blocks.

Parsing is deliberately tolerant: missing or malformed elements degrade to
empty results instead of raising.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup, Tag

from app.ingestion.url_utils import normalize_url

logger = logging.getLogger(__name__)

_ANCHOR_MARKER_RE = re.compile(r"\s*[¶#]+\s*$")

# Tags removed wholesale from the extracted content.
_BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "svg",
    "form",
    "nav",
    "header",
    "footer",
    "aside",
)

# CSS selectors tried, in order, to locate the main content region.
_CONTENT_SELECTORS = (
    "main",
    "article",
    '[role="main"]',
    ".md-content",
    "#content",
    "#main",
)

# HTML tags turned into blocks.
_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"]

BlockKind = Literal["h1", "h2", "h3", "text", "li", "code"]


@dataclass(frozen=True, slots=True)
class Block:
    """One extracted content element, already free of markup."""

    kind: BlockKind
    text: str
    language: str | None = None  # only for code blocks


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Result of parsing one documentation page."""

    url: str
    title: str
    blocks: tuple[Block, ...]

    @property
    def text_hash_basis(self) -> str:
        """Stable text used for content-hash change detection."""
        lines = [self.title, *(block.text for block in self.blocks)]
        return "\n".join(line.rstrip() for line in lines)


def _extract_title(soup: BeautifulSoup, fallback_url: str) -> str:
    title_tag = soup.title
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True).split(" - ")[0].split(" | ")[0].strip()
    first_heading = soup.find(["h1", "h2"])
    if first_heading and first_heading.get_text(strip=True):
        return first_heading.get_text(" ", strip=True)
    return fallback_url


def _select_main_content(soup: BeautifulSoup) -> Tag:
    """Find the main documentation container, falling back to ``<body>``."""
    for selector in _CONTENT_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate is not None and candidate.get_text(strip=True):
            return candidate
    return soup.body if soup.body is not None else soup


def _remove_boilerplate(container: Tag) -> None:
    for tag in container.find_all(_BOILERPLATE_TAGS):
        tag.decompose()
    for tag in container.find_all("div", attrs={"class": "header"}):
        tag.decompose()


def _code_language(pre: Tag) -> str | None:
    """Read a language hint from ``<pre><code class="language-x">``."""
    code_tag = pre.find("code", class_=True)
    if code_tag is None:
        return None
    for css_class in code_tag.get("class", []):
        if css_class.startswith("language-"):
            return css_class.removeprefix("language-")
        if css_class.startswith("highlight-"):
            return css_class.removeprefix("highlight-").removeprefix("source-")
    return None


def _skipped_container(tag: Tag, handled: set[Tag]) -> bool:
    """Return True when an already-processed ancestor covers this tag."""
    ancestor = tag.find_parent(_BLOCK_TAGS)
    return ancestor is not None and ancestor in handled


def _strip_anchor_marker(text: str) -> str:
    """Remove trailing heading-anchor markers ("Heading ¶" -> "Heading")."""
    return _ANCHOR_MARKER_RE.sub("", text).strip() or text


def _iter_blocks(container: Tag) -> list[Block]:
    """Walk the container in document order and emit content blocks.

    Notes (deliberate simplifications):
    - list items are flattened (nested lists end up as text of the parent item);
    - if a paragraph/list item *contains* a code block, the wrapper text is
      dropped and the code blocks are extracted on their own, avoiding
      duplicated code text downstream.
    """
    blocks: list[Block] = []
    handled: set[Tag] = set()

    for tag in container.find_all(_BLOCK_TAGS):
        if tag in handled or _skipped_container(tag, handled):
            continue
        text = tag.get_text(" ", strip=True)

        if tag.name == "pre":
            raw_code = tag.get_text()
            code = "\n".join(line.rstrip() for line in raw_code.strip().splitlines())
            if code:
                blocks.append(
                    Block(kind="code", text=code, language=_code_language(tag))
                )
            handled.add(tag)
            continue
        if tag.name == "li" or tag.name == "p":
            if tag.find("pre") is not None:
                # Let the inner <pre> elements down the loop handle themselves.
                continue
            prefix = "- " if tag.name == "li" else ""
            if text:
                blocks.append(Block(kind="li" if tag.name == "li" else "text", text=prefix + text))
            handled.add(tag)
            continue
        if tag.name in ("h1", "h2", "h3"):
            if text:
                # strip trailing anchor markers like "Heading ¶" common on MkDocs sites
                blocks.append(Block(kind=tag.name, text=_strip_anchor_marker(text)))
            handled.add(tag)
            continue
        # h4..h6: keep the text, but do not create hierarchy levels.
        if text:
            blocks.append(Block(kind="text", text=text))
        handled.add(tag)

    return blocks


def parse_html(html: str, url: str) -> ParsedPage:
    """Parse one raw HTML page into a :class:`ParsedPage`."""
    normalized_url = normalize_url(url or "https://unknown.invalid/")
    soup = BeautifulSoup(html or "", "html.parser")

    title = _extract_title(soup, normalized_url)
    container = _select_main_content(soup)
    _remove_boilerplate(container)
    blocks = _iter_blocks(container)

    if not blocks:
        logger.debug("No content blocks extracted from %s", normalized_url)

    return ParsedPage(url=normalized_url, title=title, blocks=tuple(blocks))
