"""CLI for indexing Phase 2 chunks into Qdrant and semantic search verification.

Usage (the USER executes these; they download the model and populate Qdrant)::

    uv run python -m app.vector_store.cli index --source fastapi
    uv run python -m app.vector_store.cli index --all
    uv run python -m app.vector_store.cli search --query "how do I define 国际 routes?" -k 5
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.core.config import get_settings
from app.vector_store.indexing import discover_sources, index_sources
from app.vector_store.qdrant_service import build_vector_store
from app.vector_store.search import SemanticSearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.vector_store.cli",
        description="Index Phase 2 chunks in Qdrant / run semantic search.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Generate embeddings and index chunks.")
    target = index.add_mutually_exclusive_group(required=True)
    target.add_argument("--source", metavar="NAME", help="Index a single processed source.")
    target.add_argument("--all", action="store_true", help="Index every processed source.")

    search = subparsers.add_parser("search", help="Vector search (verification).")
    search.add_argument("--query", required=True, metavar="TEXT")
    search.add_argument("-k", type=int, default=5, metavar="K", help="Top K results.")

    return parser


def _run_index(argv: list[str]) -> int:
    config = get_settings()
    if argv.source:
        source_names = [argv.source]
    else:
        source_names = discover_sources(config)
        if not source_names:
            print("No processed sources found under data/processed/. Run Phase 2 ingestion first.")
            return 1

    stats = index_sources(source_names, settings=config)
    print("Indexing finished:")
    print(f"  sources:          {', '.join(stats.sources)}")
    print(f"  chunks loaded:    {stats.documents_loaded}")
    print(f"  points upserted:  {stats.points_upserted} (collection: {config.qdrant_collection})")
    print(f"  duration:         {stats.duration_seconds:.3f}s")
    return 0


def _run_search(argv: argparse.Namespace) -> int:
    config = get_settings()
    vector_store = build_vector_store(config)
    service = SemanticSearchService(vector_store)
    hits = service.search(argv.query, k=argv.k)
    if not hits:
        print("No results. Is the collection populated? Run the 'index' command first.")
        return 1
    for hit in hits:
        print(hit.format())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.command == "index":
        return _run_index(args)
    if args.command == "search":
        return _run_search(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
