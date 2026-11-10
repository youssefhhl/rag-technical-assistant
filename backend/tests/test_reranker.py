"""Tests for Cross-Encoder reranking (Phase 5) — model fully mocked."""

import pytest

from app.core.config import Settings
from app.retrieval.hybrid_search import HybridHit, HybridSearchService
from app.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankHit,
)
from app.retrieval.bm25_search import Bm25Hit


def make_fused_hit(i: int) -> HybridHit:
    return HybridHit(
        content=f"fused content {i} about routers and dependencies",
        metadata={"source": f"https://docs.test/{i}", "title": f"Doc {i}", "chunk_index": i},
        rrf_score=0.03 - 0.001 * i,
        vector_rank=(i + 1) if i < 3 else None,
        bm25_rank=None,
    )


class FakeCrossEncoder:
    """Records pairs; deterministic scores (or a provided mapping)."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self.pairs_seen: list[list[tuple[str, str]]] = []
        self._scores = scores

    def predict(self, pairs):
        self.pairs_seen.append(list(pairs))
        if self._scores is not None:
            return list(self._scores)
        # deterministic: score = fraction of query tokens present in content
        query_tokens = set(pairs[0][0].lower().split())
        return [
            float(len(set(str(content).lower().split()) & query_tokens)) / 10.0
            for _, content in pairs
        ]


def make_reranker(model=None, **kwargs) -> tuple[CrossEncoderReranker, FakeCrossEncoder | None]:
    fake = model
    if fake is None and "settings" not in kwargs:
        fake = FakeCrossEncoder()
    return CrossEncoderReranker(model=fake, **kwargs), fake


def make_settings(**overrides) -> Settings:
    values = {"_env_file": None, "scrape_request_delay_seconds": 0}
    values.update(overrides)
    return Settings(**values)


def make_bare_hit(i: int) -> Bm25Hit:
    return Bm25Hit(
        content=f"fused content {i} about routers and dependencies",
        metadata={"source": f"https://docs.test/{i}", "title": f"Doc {i}", "chunk_index": i},
        bm25_score=1.0,
    )


# -- pair construction / scoring -----------------------------------------------


def test_builds_query_content_pairs_and_scores_assigned():
    reranker, fake = make_reranker()
    hits = [make_fused_hit(0), make_fused_hit(1)]

    reranker.rerank("routers dependencies?", hits)

    assert fake.pairs_seen == [
        [
            ("routers dependencies?", "fused content 0 about routers and dependencies"),
            ("routers dependencies?", "fused content 1 about routers and dependencies"),
        ]
    ]


def test_scores_map_back_to_the_right_chunks():
    reranker, _ = make_reranker(model=FakeCrossEncoder(scores=[0.2, 0.9, 0.4]))
    hits = [make_fused_hit(i) for i in range(3)]

    reranked = reranker.rerank("anything", hits)

    assert [hit.rerank_score for hit in reranked] == pytest.approx([0.9, 0.4, 0.2])
    assert reranked[0].content == "fused content 1 about routers and dependencies"
    assert reranked[2].content == "fused content 0 about routers and dependencies"


def test_scores_are_in_descending_order():
    reranker, _ = make_reranker()
    hits = [make_fused_hit(i) for i in range(6)]  # shared tokens -> identical scores?

    scores = [hit.rerank_score for hit in reranker.rerank("routers and dependencies", hits)]

    assert scores == sorted(scores, reverse=True)


# -- selection limits -------------------------------------------------------------


def test_default_top_k_is_five():
    reranker, _ = make_reranker()
    hits = [make_fused_hit(i) for i in range(8)]

    reranked = reranker.rerank("routers and dependencies", hits)

    assert len(reranked) == 5
    assert DEFAULT_RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_custom_top_k():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(), top_k=2
    )
    hits = [make_fused_hit(i) for i in range(4)]

    assert len(reranker.rerank("query", hits)) == 2
    assert reranker.rerank("query", hits, top_k=3) == reranker.rerank("query", hits, top_k=3)
    assert len(reranker.rerank("query", hits, top_k=3)) == 3


def test_top_k_larger_than_candidates_returns_all():
    reranker, _ = make_reranker(top_k=10)
    hits = [make_fused_hit(i) for i in range(3)]

    assert len(reranker.rerank("routers and dependencies", hits)) == 3


# -- empty / invalid input -------------------------------------------------------


def test_empty_hits_returns_empty_without_model_call():
    reranker, fake = make_reranker()

    assert reranker.rerank("anything", []) == []
    assert fake.pairs_seen == []


def test_empty_query_returns_empty_list():
    reranker, fake = make_reranker()
    hits = [make_fused_hit(0)]

    assert reranker.rerank("", hits) == []
    assert reranker.rerank("   ", hits) == []
    assert fake.pairs_seen == []


def test_top_k_below_one_returns_empty():
    reranker, _ = make_reranker()
    hits = [make_fused_hit(0)]

    assert reranker.rerank("query", hits, top_k=0) == []


# -- preservation ----------------------------------------------------------------


def test_original_content_and_metadata_preserved():
    reranker, _ = make_reranker()
    hits = [make_fused_hit(i) for i in range(3)]

    reranked = reranker.rerank("routers and dependencies", hits)

    original = {(hit.content, hit.metadata["title"]) for hit in hits}
    assert {(hit.content, hit.metadata["title"]) for hit in reranked} == original
    carried = reranked[0]
    assert isinstance(carried, RerankHit)
    assert carried.metadata["chunk_index"] in {0, 1, 2}


def test_fusion_trace_carried_over_from_hybrid_hits():
    reranker, _ = make_reranker()
    hits = [make_fused_hit(0), make_fused_hit(3)]  # (3) has vector_rank None

    reranked = reranker.rerank("routers and dependencies", hits)

    by_content = {hit.content: hit for hit in reranked}
    assert by_content["fused content 0 about routers and dependencies"].vector_rank == 1
    assert by_content["fused content 3 about routers and dependencies"].vector_rank is None
    assert by_content["fused content 0 about routers and dependencies"].rrf_score > 0


def test_works_with_non_hybrid_hit_types():
    reranker, _ = make_reranker()

    reranked = reranker.rerank("routers", [make_bare_hit(0), make_bare_hit(1)])

    assert len(reranked) == 2
    assert reranked[0].vector_rank is None and reranked[0].rrf_score is None


# -- lazy loading + configuration -------------------------------------------------


def test_model_is_lazy_and_loaded_once(monkeypatch):
    load_calls: list[str] = []
    calls = []
    fake = FakeCrossEncoder()

    def fake_model_constructor(self):
        load_calls.append(self._model_name)
        return fake

    monkeypatch.setattr(CrossEncoderReranker, "_load_model", fake_model_constructor)
    reranker = CrossEncoderReranker(settings=make_settings())
    hits = [make_fused_hit(0), make_fused_hit(1)]

    assert load_calls == []  # nothing loaded at construction
    reranker.rerank("query", hits)  # type: ignore[arg-type]
    reranker.rerank("query", hits)  # type: ignore[arg-type]

    assert load_calls == [DEFAULT_RERANKER_MODEL]  # loaded exactly once, lazily
    assert len(fake.pairs_seen) == 2


def test_model_name_and_top_k_read_from_settings():
    settings = make_settings()
    reranker = CrossEncoderReranker(settings=settings)

    assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.top_k == 5


def test_settings_overridden_by_explicit_arguments():
    settings = make_settings(reranker_model="custom/reranker", reranker_top_k=7)
    reranker = CrossEncoderReranker(settings=settings, model_name="explicit/model", top_k=3)

    assert reranker._model_name == "explicit/model"
    assert reranker.top_k == 3


def test_injected_model_never_triggers_loading(monkeypatch):
    monkeypatch.setattr(
        CrossEncoderReranker,
        "_load_model",
        lambda self: (_ for _ in ()).throw(AssertionError("model must not load")),
    )
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    reranker.rerank("query", [make_fused_hit(0)])  # type: ignore[list-item]

    assert reranker.model is reranker._model


# -- integration: hybrid pipeline with reranker ------------------------------------


class FakeEngine:
    def __init__(self, hits) -> None:
        self._hits = hits

    def search(self, query, k=5):
        return list(self._hits[:k])


def _hybrid_hits(count):
    return [
        HybridHit(
            content=f"candidate {i} distinct text {i}",
            metadata={"source": f"https://h.test/{i}", "chunk_index": i},
            rrf_score=0.03 - 0.001 * i,
            vector_rank=i + 1,
            bm25_rank=None,
        )
        for i in range(count)
    ]


def test_hybrid_pipeline_applies_reranker_after_rrf():
    from app.vector_store.search import SearchHit

    vector_search = FakeEngine(
        [
            SearchHit(
                content=f"candidate {i} distinct text {i}",
                source_url=f"https://h.test/{i}",
                title=f"Doc {i}",
                heading_hierarchy=[],
                similarity_score=0.9 - i * 0.1,
                chunk_index=i,
                metadata={"source": f"https://h.test/{i}", "chunk_index": i},
            )
            for i in range(20)
        ]
    )
    bm25_search = FakeEngine(
        [
            Bm25Hit(
                content=f"candidate {i} distinct text {i}",
                metadata={"source": f"https://h.test/{i}", "chunk_index": i},
                bm25_score=5.0 - i,
            )
            for i in range(20)
        ]
    )
    fake_reranker = FakeCrossEncoder()
    hybrid = HybridSearchService(
        vector_search,
        bm25_search,
        reranker=CrossEncoderReranker(model=fake_reranker),
    )

    hits = hybrid.search("candidate query")

    # final fan-out = reranker default (5), from 10 fused candidates
    assert all(isinstance(hit, RerankHit) for hit in hits)
    assert len(hits) == 5
    assert len(fake_reranker.pairs_seen[0]) == 10  # reranked the fused top-10


def test_hybrid_without_reranker_keeps_pure_rrf_output():
    engine_hits = _hybrid_hits(2)
    hybrid = HybridSearchService(FakeEngine(engine_hits), FakeEngine([]))

    hits = hybrid.search("query")

    assert all(not isinstance(hit, RerankHit) for hit in hits)  # HybridHit as before
