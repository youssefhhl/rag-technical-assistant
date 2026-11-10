"""Tests for the hybrid search CLI (fake engines, no Qdrant server, no model)."""

import pytest

from app.retrieval.bm25_search import Bm25Hit
from app.retrieval.cli import build_parser, main
from app.vector_store.search import SearchHit


def search_hit(i: int) -> SearchHit:
    return SearchHit(
        content=f"dense content {i} about middlewares and dependencies.",
        source_url=f"https://docs.test/page{i}",
        title=f"Doc {i}",
        heading_hierarchy=[f"Doc {i}", "Section"],
        similarity_score=0.9 - 0.01 * i,
        chunk_index=4 + i,
        metadata={"source": f"https://docs.test/page{i}", "title": f"Doc {i}", "chunk_index": 4 + i},
    )  # type: ignore[call-arg]


def bm25_hit(i: int) -> Bm25Hit:
    return Bm25Hit(
        content=f"sparse content {i} about middlewares and dependencies.",
        metadata={"source": f"https://docs.test/page{i}", "title": f"Doc {i}", "chunk_index": 4 + i},
        bm25_score=5.0 - 0.1 * i,
    )  # type: ignore[call-arg]


class FakeRealService:
    """Shared shape with SemanticSearchService / Bm25SearchService."""

    def __init__(self, hits) -> None:
        self._hits = hits

    def search(self, query, k=5):
        if not query.strip():
            raise AssertionError("empty query reached an engine")
        return list(self._hits[:k])


class FakeRerankService:
    def __init__(self, hits) -> None:
        self.calls: list[int] = []

    def rerank(self, query, hits, top_k=None):
        from app.retrieval.reranker import RerankHit

        self.calls.append(top_k)
        reranked = [
            RerankHit(
                content=hit.content,
                metadata=dict(hit.metadata),
                rerank_score=0.99 - 0.01 * position,
                rrf_score=getattr(hit, "rrf_score", None),
                vector_rank=getattr(hit, "vector_rank", None),
                bm25_rank=getattr(hit, "bm25_rank", None),
            )
            for position, hit in enumerate(hits)
        ]
        return reranked[: (top_k if top_k is not None else 5)]


# update tests in this module via a shared helper as before
def make_fake_services(monkeypatch, vector_hits, bm25_hits, *, rerank=True):
    import app.retrieval.cli as cli_module

    vector_service = FakeRealService(vector_hits)
    bm25_service = FakeRealService(bm25_hits)
    monkeypatch.setattr(cli_module, "build_vector_store", lambda config: object())
    monkeypatch.setattr(cli_module, "SemanticSearchService", lambda store: vector_service)
    monkeypatch.setattr(
        cli_module, "Bm25SearchService", type("B", (), {"from_settings": staticmethod(lambda config: bm25_service)})
    )
    reranker_used = {"value": None}

    if rerank:
        fake_reranker = FakeRerankService(None)
        reranker_used["value"] = fake_reranker
        monkeypatch.setattr(
            cli_module,
            "CrossEncoderReranker",
            lambda settings: fake_reranker,
        )
    else:
        reranker_used["value"] = None
    return vector_service, bm25_service, reranker_used


def test_parser_accepts_query_and_k():
    args = build_parser().parse_args(["hybrid", "--query", "ports", "--k", "5"])

    assert args.query == "ports"
    assert args.k == 5


def test_parser_defaults_match_phase5_pipeline():
    args = build_parser().parse_args(["hybrid", "--query", "ports"])

    assert args.k is None  # final counts come from the pipeline defaults
    assert args.no_rerank is False


def test_main_prints_query_top_results_scores_sources_and_chunks(monkeypatch, capsys):
    make_fake_services(
        monkeypatch,
        [search_hit(0), search_hit(1)],
        [bm25_hit(1), bm25_hit(2)],
    )
    argv = ["hybrid", "--query", "how are routes defined?"]

    exit_code = main(argv)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "query:   how are routes defined?" in output
    assert "vector_k=20" in output and "bm25_k=20" in output and "rrf_k=60" in output
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in output  # reranker wired by default
    assert "ce " in output  # reranker-tagged scores are printed
    for i in range(3):  # vector-only 0 + both 1 + bm25-only 2 = chunk 4,5,6
        assert f"chunk {4 + i}" in output
    assert "https://docs.test/page0" in output
    assert "https://docs.test/page1" in output
    assert "https://docs.test/page2" in output


def test_main_no_rerank_flag_prints_hybrid_scores(monkeypatch, capsys):
    make_fake_services(
        monkeypatch,
        [search_hit(0), search_hit(1)],
        [bm25_hit(1), bm25_hit(2)],
        rerank=False,
    )
    exit_code = main(["hybrid", "--query", "routes", "--no-rerank"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "reranker: off" in output
    assert "hybrid" in output  # pure RRF scores and engine ranks


def test_main_respects_custom_k(monkeypatch, capsys):
    make_fake_services(
        monkeypatch,
        [search_hit(i) for i in range(6)],
        [bm25_hit(i) for i in range(6)],
    )

    exit_code = main(["hybrid", "--query", "dependencies", "--k", "3"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "top 3 results:" in output
    assert "4. [ce" not in output  # only 3 result blocks


def test_main_uses_rrf_top10_when_rerank_disabled_without_k(monkeypatch, capsys):
    make_fake_services(monkeypatch, [search_hit(i) for i in range(12)], [], rerank=False)

    exit_code = main(["hybrid", "--query", "guarded", "--no-rerank"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "top 10 results:" in output  # pure RRF default final count


def test_main_conversation_mode_prints_context_blocks(monkeypatch, capsys, tmp_path):
    make_fake_services(monkeypatch, [search_hit(0)], [bm25_hit(0)])
    import app.retrieval.cli as cli_module
    from app.database import ConversationRepository, create_engine_from_url, init_database

    engine = create_engine_from_url(f"sqlite:///{tmp_path / 'conv.db'}")
    init_database(engine)
    ConversationRepository(engine, settings=None).append_message("sess-1", "user", "what is FastAPI?")

    class FakeRewriter:
        def rewrite(self, question, history):
            return "how do I declare routes in FastAPI?"

    monkeypatch.setattr(cli_module, "create_engine_from_url", lambda url: engine)
    monkeypatch.setattr(cli_module, "QueryRewriter", lambda settings: FakeRewriter())

    exit_code = main(["hybrid", "--query", "and routes?", "--conversation", "sess-1", "--no-rerank"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "original question:   and routes?" in output
    assert "rewritten query:     how do I declare routes in FastAPI?" in output
    assert "history used:        1 message(s)" in output
    assert "hybrid " in output  # hybrid results still printed with RRF scores


def test_main_returns_error_code_when_no_results(monkeypatch, capsys):
    make_fake_services(monkeypatch, [], [])

    exit_code = main(["hybrid", "--query", "anything"])

    assert exit_code == 1
    assert "No results" in capsys.readouterr().out
