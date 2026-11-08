"""Tests for the ingestion CLI (no network access)."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ingestion.cli import main
from app.ingestion.pipeline import IngestionResult


class StubPipeline:
    """Stands in for IngestionPipeline inside CLI tests."""

    last_settings = None

    def __init__(self, settings=None, registry=None):
        StubPipeline.last_settings = settings

    def run(self, source_name: str) -> IngestionResult:
        return IngestionResult(
            source_name=source_name,
            pages_fetched=3,
            pages_processed=3,
            documents_built=6,
            chunks=[Document(page_content="c", metadata={"source": "https://x.test/"})],
            chunks_path=Path("/tmp/chunks.jsonl"),
        )


def test_list_sources_prints_all_sources(capsys):
    exit_code = main(["--list"])

    output = capsys.readouterr().out
    assert exit_code == 0
    for source in ("fastapi", "qdrant", "docker-compose", "react", "postgresql"):
        assert source in output


def test_cli_runs_ingestion_with_stub(monkeypatch, capsys):
    monkeypatch.setattr("app.ingestion.cli.IngestionPipeline", StubPipeline)

    exit_code = main(["--source", "fastapi", "--max-pages", "2", "--depth", "1"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Ingestion of 'fastapi'" in output
    assert "pages fetched (raw HTML saved): 3" in output
    assert "chunks (after dedup):           1" in output
    # CLI overrides were applied to a fresh Settings instance
    assert StubPipeline.last_settings is not None
    assert StubPipeline.last_settings.scrape_max_pages == 2
    assert StubPipeline.last_settings.scrape_max_depth == 1


def test_cli_without_source_exits_2(capsys):
    assert main([]) == 2
    assert "No source given" in capsys.readouterr().err


def test_cli_unknown_source_is_rejected_by_argparse():
    with pytest.raises(SystemExit) as excinfo:
        main(["--source", "not-a-source"])
    assert excinfo.value.code == 2
