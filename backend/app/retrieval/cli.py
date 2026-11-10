"""CLI for real Hybrid (RRF) search verification against live services.

Usage (the USER executes these; they require the Qdrant server running and the
BM25 index built, and load the E5 embedding model on first query)::

    uv run python -m app.retrieval.cli hybrid --query "how are routes defined?"
    uv run python -m app.retrieval.cli hybrid --query "docker compose ports" --k 10
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.core.config import get_settings
from app.database import ConversationRepository, create_engine_from_url, init_database
from app.retrieval.bm25_search import Bm25SearchService
from app.retrieval.contextual_search import ContextualSearchService, ContextualOutcome
from app.retrieval.hybrid_search import (
    DEFAULT_BM25_K,
    DEFAULT_RRF_K,
    DEFAULT_VECTOR_K,
    HybridHit,
    HybridSearchService,
)
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import CrossEncoderReranker, RerankHit
from app.vector_store.qdrant_service import build_vector_store
from app.vector_store.search import SemanticSearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.retrieval.cli",
        description="Real hybrid search (dense Qdrant + sparse BM25, fused with RRF).",
    )
    hybrid = parser.add_subparsers(dest="command", required=True)
    search = hybrid.add_parser("hybrid", help="RRF fusion of vector + BM25 search.")
    search.add_argument("--query", required=True, metavar="TEXT")
    search.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="K",
        help="Final top K (default: 5 with reranker, 10 pure RRF).",
    )
    search.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the Cross-Encoder reranking step (pure RRF).",
    )
    search.add_argument(
        "--conversation",
        metavar="ID",
        default=None,
        help="Conversation id: loads history and rewrites the query (Phase 6).",
    )
    return parser


def _run_hybrid(args: argparse.Namespace) -> int:
    config = get_settings()
    vector_service = SemanticSearchService(build_vector_store(config))
    bm25_service = Bm25SearchService.from_settings(config)

    reranker = None
    if args.no_rerank:
        reranker_label = "off (--no-rerank)"
    else:
        reranker = CrossEncoderReranker(settings=config)
        reranker_label = f"{config.reranker_model} (final top {config.reranker_top_k})"
    hybrid = HybridSearchService(vector_service, bm25_service, reranker=reranker)

    contextual = None
    if args.conversation:
        engine = create_engine_from_url(config.database_url)
        init_database(engine)
        repository = ConversationRepository(engine, settings=config)
        contextual = ContextualSearchService(
            repository, QueryRewriter(settings=config), hybrid
        )

    final_top = args.k
    if final_top is None:
        final_top = 10 if reranker is None else config.reranker_top_k

    if contextual is not None:
        outcome = contextual.search(args.conversation, args.query)
        print("=== Hybrid search (RRF + context) ===")
        print(f"original question:   {outcome.question}")
        print(f"rewritten query:     {outcome.rewritten_query}")
        print(f"history used:        {outcome.history_used} message(s)")
    else:
        outcome = None
        print("=== Hybrid search (RRF) ===")
        print(f"query:   {args.query}")
    print(
        f"params:  vector_k={DEFAULT_VECTOR_K} bm25_k={DEFAULT_BM25_K} "
        f"rrf_k={DEFAULT_RRF_K} | reranker: {reranker_label} "
        f"(collection: {config.qdrant_collection})"
    )
    print(f"top {final_top} results:")
    print()

    hits = outcome.hits if outcome is not None else hybrid.search(args.query, top_k=args.k)
    if not hits:
        print("No results. Is the collection indexed and the BM25 index built?")
        return 1

    for position, hit in enumerate(hits, start=1):
        if isinstance(hit, RerankHit):
            body = hit.format()
        else:
            metadata = hit.metadata
            ranks = []
            if hit.vector_rank is not None:
                ranks.append(f"vector #{hit.vector_rank}")
            if hit.bm25_rank is not None:
                ranks.append(f"bm25 #{hit.bm25_rank}")
            title = str(metadata.get("title", ""))
            hierarchy = " > ".join(metadata.get("heading_hierarchy", [])) or title
            body = (
                f"[hybrid {hit.rrf_score:.4f}] ({', '.join(ranks)}) {title} — {hierarchy}\n"
                f"  source: {metadata.get('source', '')} "
                f"(chunk {metadata.get('chunk_index', 0)})\n"
                f"  {hit.content[:300]}\n"
            )
        print(f"{position}. {body}", end="\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.command == "hybrid":
        return _run_hybrid(args)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
