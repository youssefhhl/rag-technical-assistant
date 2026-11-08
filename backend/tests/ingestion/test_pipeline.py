"""End-to-end pipeline tests (crawl -> parse -> chunks).

The pipeline builds its own ``httpx.Client``; the ``mock_http`` fixture patches
``httpx.Client`` (and the pipeline's source lookup) so everything runs fully
offline against the fake in-memory documentation site from ``conftest.py``.
"""

import json

import httpx
import pytest

from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.deduplication import DocumentRegistry
from tests.ingestion.conftest import (
    PAGE_HOME,
    PAGE_PORTS,
    PAGE_SERVICES,
    URL_HOME,
    URL_PORTS,
    URL_SERVICES,
)

CLONE_URL = "https://docs.example.test/docs/services-clone.html"


def fake_site_pages():
    return {URL_HOME: PAGE_HOME, URL_SERVICES: PAGE_SERVICES, URL_PORTS: PAGE_PORTS}


@pytest.fixture
def pipeline(make_settings, registry):
    return IngestionPipeline(settings=make_settings(), registry=registry)


@pytest.fixture
def mock_http(monkeypatch, mock_client_factory, fake_source):
    """Patch httpx.Client + the pipeline's source lookup: fully offline runs."""
    pages_container = {"pages": fake_site_pages()}

    class FakeClient:
        def __new__(cls, *args, **kwargs):
            return mock_client_factory(pages_container["pages"])

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr("app.ingestion.pipeline.get_source", lambda name: fake_source)
    return pages_container


def test_pipeline_end_to_end_docs_chunks_metadata(pipeline, mock_http):
    result = pipeline.run("example")

    assert result.pages_fetched == 3
    assert result.pages_processed == 3
    assert result.documents_built == 6  # 3 sections on home + 2 + 1
    assert len(result.chunks) == 6

    for chunk in result.chunks:
        metadata = chunk.metadata
        assert metadata["source"].startswith("https://docs.example.test/")
        assert metadata["source_name"] == "example"
        assert metadata["document_type"] == "website"
        assert metadata["title"]
        assert metadata["heading_hierarchy"]
        assert isinstance(metadata["chunk_index"], int)

    chunks_path = result.chunks_path
    assert chunks_path is not None and chunks_path.exists()
    lines = chunks_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(result.chunks)
    first = json.loads(lines[0])
    assert set(first) == {"page_content", "metadata"}
    assert first["metadata"]["source_name"] == "example"


def test_second_run_skips_already_ingested_pages(pipeline, mock_http):
    first = pipeline.run("example")
    second = pipeline.run("example")

    assert first.pages_processed == 3
    assert second.pages_fetched == 0  # registry filter: no re-fetch, no re-processing
    assert second.pages_processed == 0
    assert second.chunks == []
    assert second.chunks_deduplicated == 0


def test_identical_content_on_two_pages_is_chunk_deduplicated(pipeline, mock_http, registry):
    home = mock_http["pages"][URL_HOME]
    home_with_extra_link = home.replace(
        '<a href="#fragment">fragment</a>',  # see conftest PAGE_HOME
        f'<a href="{CLONE_URL}">clone</a>',
    )
    mock_http["pages"][URL_HOME] = home_with_extra_link
    mock_http["pages"][CLONE_URL] = PAGE_SERVICES  # identical content to services page

    result = pipeline.run("example")

    # the clone page was crawled and processed but contributed no new chunks
    assert result.pages_processed == 4
    assert result.documents_built > result.pages_processed  # sections > pages
    assert result.chunks_deduplicated >= 2  # every section text of the clone is a duplicate
    sources = {chunk.metadata["source"] for chunk in result.chunks}
    assert CLONE_URL not in sources
    assert URL_PORTS in sources


def test_registry_only_counts_processed_pages(pipeline, mock_http, make_settings):
    result = pipeline.run("example")
    assert result.pages_processed == 3

    registry_path = make_settings().data_dir / "processed" / "registry.sqlite3"
    with DocumentRegistry(registry_path) as fresh_registry:
        # registered pages == processed pages; metadata stored alongside
        assert fresh_registry.count("example") == 3
        assert fresh_registry.has_page("https://docs.example.test/")


def test_unknown_source_name_raises(pipeline):
    with pytest.raises(ValueError, match="Unknown documentation source"):
        pipeline.run("does-not-exist")
