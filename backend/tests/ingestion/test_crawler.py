"""Tests for the controlled crawler (all HTTP via httpx.MockTransport)."""

import json

import httpx

from app.ingestion.crawler import Crawler
from tests.ingestion.conftest import (
    FAKE_HOST,
    PAGE_HOME,
    PAGE_PORTS,
    PAGE_SERVICES,
    URL_EXTERNAL,
    URL_HOME,
    URL_PORTS,
    URL_SERVICES,
    make_handler,
)

FAKE_SITE = {
    URL_HOME: PAGE_HOME,
    URL_SERVICES: PAGE_SERVICES,
    URL_PORTS: PAGE_PORTS,
}


def run_crawl(
    fake_source,
    make_settings,
    pages: dict[str, str] | None = None,
    requested: list[str] | None = None,
    url_filter=None,
    **settings_overrides,
):
    """Build a crawler against the fake site and run one crawl."""
    crawler = Crawler(
        fake_source,
        make_settings(**settings_overrides),
        client=httpx.Client(
            transport=httpx.MockTransport(
                make_handler(FAKE_SITE if pages is None else pages, requested=requested)
            )
        ),
    )
    try:
        return crawler, crawler.crawl(url_filter=url_filter)
    finally:
        crawler.close()


def test_crawl_follows_links_within_allowed_domain(fake_source, make_settings):
    _, records = run_crawl(fake_source, make_settings, scrape_max_pages=10)
    urls = {record.url for record in records}
    assert urls == {URL_HOME, URL_SERVICES, URL_PORTS}
    assert all(record.status_code == 200 for record in records)


def test_crawl_never_fetches_disallowed_domain(fake_source, make_settings):
    requested: list[str] = []
    run_crawl(fake_source, make_settings, requested=requested, scrape_max_pages=10)
    assert not any(URL_EXTERNAL in url for url in requested)


def test_crawl_respects_max_pages(fake_source, make_settings):
    _, records = run_crawl(fake_source, make_settings, scrape_max_pages=1)
    assert len(records) == 1

    _, records = run_crawl(fake_source, make_settings, scrape_max_pages=2)
    assert len(records) == 2


def test_crawl_respects_max_depth(fake_source, make_settings):
    _, records = run_crawl(fake_source, make_settings, scrape_max_pages=10, scrape_max_depth=0)
    assert [record.url for record in records] == [URL_HOME]  # links not followed

    _, records = run_crawl(fake_source, make_settings, scrape_max_pages=10, scrape_max_depth=1)
    assert len(records) == 3  # home + its links


def test_crawl_normalizes_and_never_refetches_duplicate_url(fake_source, make_settings):
    """PAGE_HOME links to the ports page twice, once with a query string."""
    requested: list[str] = []
    run_crawl(fake_source, make_settings, requested=requested, scrape_max_pages=10)

    assert all(url != "https://docs.example.test/robots.txt?" for url in requested)
    # fetched exactly once despite two source links
    assert requested.count("https://docs.example.test/docs/ports.html") == 1
    # the ?tracked=1 variant normalizes to the same page and is never fetched
    assert sum("/docs/ports.html?" in url for url in requested) == 0


def test_crawl_respects_robots_txt(fake_source, make_settings):
    pages = {"robots.txt": "User-agent: *\nDisallow: /docs\n", **FAKE_SITE}
    requested: list[str] = []
    _, records = run_crawl(
        fake_source, make_settings, pages=pages, requested=requested, scrape_max_pages=10
    )

    assert [record.url for record in records] == [URL_HOME]
    assert not any("/docs/" in url for url in requested)


def test_crawl_tolerates_http_errors(fake_source, make_settings):
    pages = {**FAKE_SITE, URL_PORTS: "__status:500"}
    _, records = run_crawl(
        fake_source, make_settings, pages=pages, scrape_max_pages=10
    )

    urls = {record.url for record in records}
    assert URL_HOME in urls
    assert URL_SERVICES in urls
    assert URL_PORTS not in urls


def test_crawl_missing_page_is_skipped(fake_source, make_settings):
    """A linked page that 404s doesn't abort the crawl."""
    pages = {URL_HOME: PAGE_HOME, URL_SERVICES: PAGE_SERVICES}  # /docs/ports.html -> 404
    _, records = run_crawl(
        fake_source, make_settings, pages=pages, scrape_max_pages=10
    )
    assert [record.url for record in records] == [URL_HOME, URL_SERVICES]


def test_crawl_tolerates_transport_exceptions(fake_source, make_settings):
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    crawler = Crawler(
        fake_source,
        make_settings(),
        client=httpx.Client(transport=httpx.MockTransport(failing_handler)),
    )
    try:
        records = crawler.crawl()
    finally:
        crawler.close()
    assert records == []


def test_crawl_saves_raw_html_with_sidecar(fake_source, make_settings):
    crawler, _ = run_crawl(fake_source, make_settings)
    raw_dir = crawler.settings.data_dir / "raw" / "example"

    html_files = sorted(raw_dir.glob("*.html"))
    assert len(html_files) == 3

    meta = json.loads((raw_dir / (html_files[0].stem + ".json")).read_text())
    assert meta["source_name"] == "example"
    assert meta["url"].startswith(f"https://{FAKE_HOST}")
    assert meta["status_code"] == 200
    assert len(meta["html_hash"]) == 64


def test_crawl_url_filter_skips_fetching_registered_pages(
    fake_source, make_settings, registry
):
    """Registry filter: pages already ingested are never requested."""
    registry.register(URL_SERVICES, "example", "hash", "Example Services")
    requested: list[str] = []
    _, records = run_crawl(
        fake_source,
        make_settings,
        requested=requested,
        scrape_max_pages=10,
        url_filter=lambda url: not registry.has_page(url),
    )

    assert all(URL_SERVICES not in url for url in requested)  # not fetched
    assert all(record.url != URL_SERVICES for record in records)
