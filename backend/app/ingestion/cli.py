"""CLI entry point for the ingestion pipeline.

Usage::

    uv run python -m app.ingestion.cli --list
    uv run python -m app.ingestion.cli --source fastapi
    uv run python -m app.ingestion.cli --source fastapi --max-pages 10 --depth 1

Scraper limits default to the ``SCRAPE_*`` settings (see ``.env.example``) and
can be adjusted per run with the options below.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.core.config import Settings, get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.sources import DOC_SOURCES


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        prog="app.ingestion.cli",
        description="Ingest an official documentation source into LangChain chunks.",
    )
    argument_parser.add_argument(
        "--source",
        choices=sorted(source.name for source in DOC_SOURCES),
        help="Documentation source to ingest (see --list).",
    )
    argument_parser.add_argument(
        "--list",
        action="store_true",
        help="List the configured documentation sources and exit.",
    )
    argument_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Override SCRAPE_MAX_PAGES for this run.",
    )
    argument_parser.add_argument(
        "--depth",
        type=int,
        default=None,
        metavar="D",
        help="Override SCRAPE_MAX_DEPTH for this run.",
    )
    argument_parser.add_argument(
        "--delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override SCRAPE_REQUEST_DELAY_SECONDS for this run.",
    )
    argument_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return argument_parser


def _settings_with_overrides(args: argparse.Namespace) -> Settings:
    """Rebuild settings with per-run CLI overrides applied."""
    data = get_settings().model_dump()
    if args.max_pages is not None:
        data["scrape_max_pages"] = args.max_pages
    if args.depth is not None:
        data["scrape_max_depth"] = args.depth
    if args.delay is not None:
        data["scrape_request_delay_seconds"] = args.delay
    return Settings(_env_file=None, **data)


def _print_sources() -> None:
    for source in DOC_SOURCES:
        scope = ", ".join(source.allowed_domains)
        if source.path_prefixes:
            scope += f" [paths: {', '.join(source.path_prefixes)}]"
        print(f"{source.name:<16} {scope:<40} {source.description}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.list:
        _print_sources()
        return 0
    if not args.source:
        print("No source given. Use --source <name> or --list.", file=sys.stderr)
        return 2

    settings = _settings_with_overrides(args)
    result = IngestionPipeline(settings=settings).run(args.source)

    print(f"Ingestion of {result.source_name!r} finished:")
    print(f"  pages fetched (raw HTML saved): {result.pages_fetched}")
    print(f"  pages processed:                {result.pages_processed}")
    print(f"  pages skipped (unchanged):      {result.pages_skipped}")
    print(f"  LangChain Documents (sections): {result.documents_built}")
    print(f"  chunks (after dedup):           {len(result.chunks)}")
    print(f"  duplicate chunks dropped:       {result.chunks_deduplicated}")
    if result.chunks_path is not None:
        print(f"  chunks written to:              {result.chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
