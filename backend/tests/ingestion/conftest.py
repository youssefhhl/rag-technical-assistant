"""Shared fixtures for ingestion tests.

All HTTP access in tests goes through ``httpx.MockTransport`` — no external
website is ever contacted, and everything is deterministic.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx
import pytest

from app.core.config import Settings
from app.ingestion.deduplication import DocumentRegistry
from app.ingestion.sources import DocSource

FAKE_HOST = "docs.example.test"

URL_HOME = f"https://{FAKE_HOST}/"
URL_SERVICES = f"https://{FAKE_HOST}/docs/services.html"
URL_PORTS = f"https://{FAKE_HOST}/docs/ports.html"
URL_EXTERNAL = f"https://other-site.test/docs/x.html"

PAGE_HOME = f"""<!doctype html><html><head><title>Example Docs</title></head><body>
<nav><a href="/unlinked.html">ignore me</a></nav>
<main>
<h1>Example Docs</h1>
<p>Intro paragraph with <a href="{URL_SERVICES}">a link to services</a>.</p>
<h2>Services</h2>
<p>Services define containers.</p>
<ul><li>One service per container</li><li>Declared under <code>services:</code></li></ul>
<h3>Ports</h3>
<p>Ports allow services to be exposed on the host machine.</p>
<pre><code class="language-yaml">services:
  web:
    ports:
      - "8080:80"</code></pre>
<a href="{URL_PORTS}">ports docs</a>
<a href="/docs/ports.html?tracked=1">ports docs again</a>
<a href="{URL_EXTERNAL}">ext</a>
<a href="#fragment">fragment</a>
</main>
<footer><p>footer junk</p></footer>
</body></html>"""

PAGE_SERVICES = """<!doctype html><html><head><title>Example Services</title></head><body>
<main>
<h1>Services</h1><p>Every service runs its own container.</p>
<h2>Replicas</h2><p>Replicas scale a service independently.</p>
</main></body></html>"""

PAGE_PORTS = """<!doctype html><html><head><title>Example Ports</title></head><body>
<main><h1>Ports</h1>
<p>Ports map container ports to host ports.</p>
<pre><code>ports:
  - "8080:80"</code></pre>
</main></body></html>"""


def make_handler(
    pages: dict[str, str], *, requested: list[str] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a ``MockTransport`` handler for the fake docs site.

    ``pages`` maps full URLs (or ``"robots.txt"``) to bodies. Values starting
    with ``"__status:"`` return that HTTP status instead of content. All
    requests are optionally recorded into ``requested``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if requested is not None:
            requested.append(str(request.url))
        url = str(request.url)
        if url.endswith("/robots.txt"):
            robots = pages.get("robots.txt")
            if robots is None:
                return httpx.Response(404, text="no robots")
            return httpx.Response(200, text=robots)
        body = pages.get(url)
        if body is None:
            return httpx.Response(404, text="not found")
        if body.startswith("__status:"):
            return httpx.Response(int(body.removeprefix("__status:")), text="")
        return httpx.Response(
            200, text=body, headers={"content-type": "text/html; charset=utf-8"}
        )

    return handler


@pytest.fixture
def fake_source() -> DocSource:
    return DocSource(
        name="example",
        display_name="Example Docs",
        seed_url=URL_HOME,
        allowed_domains=(FAKE_HOST,),
        description="Test-only documentation source.",
    )


@pytest.fixture
def make_settings(tmp_path: Any) -> Callable[..., Settings]:
    """Build Settings pointed at a temp data dir with zero crawl delay."""

    def _make(**overrides: Any) -> Settings:
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "data_dir": data_dir,
            "scrape_request_delay_seconds": 0,
            "scrape_max_pages": 50,
            "scrape_max_depth": 3,
            "scrape_user_agent": "pytest-agent",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return _make


# Snapshot the real httpx.Client at import time so fixtures keep working even
# when `httpx.Client` is monkeypatched elsewhere (pipeline tests).
RealHttpClient = httpx.Client


@pytest.fixture
def mock_client_factory() -> Callable[..., Any]:
    def _make(pages: dict[str, str], requested: list[str] | None = None) -> httpx.Client:
        return RealHttpClient(transport=httpx.MockTransport(make_handler(pages, requested=requested)))

    return _make


@pytest.fixture
def registry(tmp_path: Any) -> DocumentRegistry:
    with DocumentRegistry(tmp_path / "data" / "processed" / "registry.sqlite3") as reg:
        yield reg
