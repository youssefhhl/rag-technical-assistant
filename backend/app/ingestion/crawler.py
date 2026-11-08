"""Controlled web crawler for official documentation sites.

Controls (deliberately *not* a general-purpose crawler):

- visits only the domains and path prefixes of its :class:`~app.ingestion.sources.DocSource`;
- honours ``robots.txt`` when it is reachable; if the robots.txt fetch fails,
  the site is treated as accessible (graceful fallback, logged);
- bounded by ``scrape_max_pages``, ``scrape_max_depth`` and
  ``scrape_request_delay_seconds`` from application settings;
- keeps a per-run visited-set so no URL is fetched twice;
- stores each raw page under ``<data_dir>/raw/<source>/<hash>.html`` with a
  sidecar ``.json`` containing the source URL and fetch metadata (audit trail,
  per ``docs/ROADMAP.md`` Phase 2).

HTTP failures, timeouts and invalid content are logged and skipped; a failing
page never aborts the whole crawl.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
import urllib.robotparser
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx

from app.core.config import Settings, get_settings
from app.ingestion.sources import DocSource
from app.ingestion.url_utils import extract_links, is_allowed_domain, matches_path_prefixes, normalize_url

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class PageRecord:
    """One successfully fetched raw page."""

    url: str  # normalized URL
    html: str
    status_code: int
    fetched_at: str  # ISO-8601 UTC
    html_hash: str  # sha256 of the raw HTML (audit / change detection)

    @property
    def file_stem(self) -> str:
        """Stable filesystem stem derived from the URL."""
        return hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:16]


class _RobotsChecker:
    """Per-host robots.txt cache; treats unreachable robots.txt as permissive."""

    def __init__(self, client: httpx.Client, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        netloc = parts.netloc.lower()
        if netloc not in self._parsers:
            robots_url = f"{parts.scheme or 'https'}://{netloc}/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            try:
                response = self._client.get(
                    robots_url, headers={"User-Agent": self._user_agent}
                )
                if response.status_code == 200 and response.text.strip():
                    parser.parse(response.text.splitlines())
                else:
                    # Missing/empty robots.txt: explicitly allow all. (A parser
                    # that never parsed any rule may deny-by-default.)
                    parser.allow_all = True
            except httpx.HTTPError as exc:
                # Graceful fallback: unreachable robots.txt does not block the crawl.
                logger.debug("robots.txt unreachable for %s (%s) - allowing", netloc, exc)
                parser.allow_all = True
            self._parsers[netloc] = parser
        return self._parsers[netloc].can_fetch(self._user_agent, url)


class Crawler:
    """BFS crawler bounded by the settings of application configuration."""

    def __init__(
        self,
        source: DocSource,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.source = source
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": self.settings.scrape_user_agent},
            timeout=_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "Crawler":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def crawl(
        self, url_filter: Callable[[str], bool] | None = None
    ) -> list[PageRecord]:
        """Crawl the source, returning raw page records.

        ``url_filter`` lets the caller cheaply reject URLs *before* fetching
        (e.g. pages already present in the deduplication registry).
        """
        max_pages = self.settings.scrape_max_pages
        max_depth = self.settings.scrape_max_depth
        delay = self.settings.scrape_request_delay_seconds
        robots = _RobotsChecker(self._client, self.settings.scrape_user_agent)

        records: list[PageRecord] = []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque(
            [(normalize_url(self.source.seed_url), 0)]
        )

        while queue and len(records) < max_pages:
            url, depth = queue.popleft()
            if url in visited or (url_filter is not None and not url_filter(url)):
                continue
            visited.add(url)

            if depth > 0 and not is_allowed_domain(url, self.source.allowed_domains):
                logger.warning("Blocked by domain allowlist: %s", url)
                continue
            if not matches_path_prefixes(url, self.source.path_prefixes):
                logger.debug("Outside configured path prefixes: %s", url)
                continue
            if not robots.is_allowed(url):
                logger.info("Blocked by robots.txt: %s", url)
                continue

            try:
                response = self._client.get(url)
            except httpx.HTTPError as exc:
                logger.warning("Fetch failed for %s: %s", url, exc)
                continue

            if response.status_code >= 400:
                logger.warning(
                    "HTTP %s for %s - skipping", response.status_code, url
                )
                continue
            if "text/html" not in response.headers.get("content-type", ""):
                logger.debug("Non-HTML response, skipping: %s", url)
                continue

            record = PageRecord(
                url=normalize_url(str(response.url)),
                html=response.text,
                status_code=response.status_code,
                fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                html_hash=hashlib.sha256(response.text.encode()).hexdigest(),
            )
            records.append(record)
            self._save_raw_file(record)

            if depth < max_depth:
                for link in extract_links(response.text, str(response.url)):
                    if (
                        link not in visited
                        and is_allowed_domain(link, self.source.allowed_domains)
                        and matches_path_prefixes(link, self.source.path_prefixes)
                    ):
                        queue.append((link, depth + 1))

            if delay > 0:
                time.sleep(delay)

        logger.info(
            "Crawl of %r finished: %d page(s) fetched, %d queued after cap",
            self.source.name,
            len(records),
            len(queue),
        )
        return records

    def _save_raw_file(self, record: PageRecord) -> None:
        """Store raw HTML + a sidecar JSON metadata file under ``data/raw``."""
        page_dir = self.settings.data_dir / "raw" / self.source.name
        page_dir.mkdir(parents=True, exist_ok=True)

        html_path = page_dir / f"{record.file_stem}.html"
        html_path.write_text(record.html, encoding="utf-8")

        meta = {
            "url": record.url,
            "source_name": self.source.name,
            "status_code": record.status_code,
            "fetched_at": record.fetched_at,
            "html_hash": record.html_hash,
        }
        meta_path = page_dir / f"{record.file_stem}.json"
        with contextlib.suppress(OSError):
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )


__all__ = ["Crawler", "PageRecord", "normalize_url"]
