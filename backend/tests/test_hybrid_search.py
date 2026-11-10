"""Tests for Reciprocal Rank Fusion of dense + sparse rankings (all fakes)."""

import pytest

from app.retrieval.bm25_search import Bm25Hit
from app.retrieval.hybrid_search import (
    DEFAULT_BM25_K,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_K,
    HybridSearchService,
    reciprocal_rank_fusion,
)
from app.vector_store.search import SearchHit


class FakeEngine:
    """Records requested k, returns a fixed ranking."""

    def __init__(self, hits: list) -> None:
        self._hits = hits
        self.calls: list[int] = []

    def search(self, query: str, k: int = 5) -> list:
        if not query.strip():
            raise AssertionError("empty query reached an engine")
        self.calls.append(k)
        return list(self._hits[:k])


def vector_hit(uid: str, i: int = 0, content: str | None = None) -> SearchHit:
    return SearchHit(
        content=content if content is not None else f"vector content {uid}",
        source_url=f"https://docs.test/{uid}",
        title=uid,
        heading_hierarchy=[uid],
        similarity_score=0.9 - 0.1 * i,
        chunk_index=i,
        metadata={"source": f"https://docs.test/{uid}", "title": uid, "chunk_index": i},
    )  # type: ignore[call-arg]


def bm25_hit(uid: str, i: int = 0, content: str | None = None) -> Bm25Hit:
    return Bm25Hit(
        content=content if content is not None else f"bm25 content {uid}",
        metadata={"source": f"https://docs.test/{uid}", "title": uid, "chunk_index": i},
        bm25_score=10.0 - i,
    )  # type: ignore[call-arg]


def shared_hit_pair(uid: str, i: int = 0, content: str | None = None):
    """Same chunk as seen by both engines (identical content -> same identity)."""
    text = content if content is not None else f"shared content {uid}"
    return vector_hit(uid, i, content=text), bm25_hit(uid, i, content=text)


def fused_service(vector_hits, bm25_hits, **overrides):
    return HybridSearchService(
        FakeEngine(vector_hits), FakeEngine(bm25_hits), **overrides
    )


# -- core fusion ----------------------------------------------------------------


def test_document_in_both_rankings_gets_both_contributions():
    """rank 1 in vector + rank 1 in bm25 -> 1/61 + 1/61."""
    dense, sparse = shared_hit_pair("a")
    service = fused_service([dense], [sparse])

    hits = service.search("query")

    assert len(hits) == 1
    assert hits[0].rrf_score == pytest.approx(1 / (DEFAULT_RRF_K + 1) + 1 / (DEFAULT_RRF_K + 1))
    assert hits[0].vector_rank == 1
    assert hits[0].bm25_rank == 1


def test_document_in_one_ranking_only_keeps_single_contribution():
    service = fused_service([vector_hit("only-dense")], [])

    hits = service.search("query")

    assert len(hits) == 1
    assert hits[0].rrf_score == pytest.approx(1 / (DEFAULT_RRF_K + 1))
    assert hits[0].vector_rank == 1
    assert hits[0].bm25_rank is None


def test_exact_rrf_score_computation():
    """chunk 'x': rank 1 (vector) + rank 3 (bm25) -> 1/61 + 1/63."""
    dense_x, sparse_x = shared_hit_pair("x")
    dense_other, _ = shared_hit_pair("other", i=1)
    service = fused_service(
        [dense_x, dense_other],
        [bm25_hit("y", i=0), bm25_hit("z", i=1), sparse_x],
    )

    hits = service.search("query", top_k=4)

    assert hits[0].rrf_score == pytest.approx(1 / 61 + 1 / 63)
    assert hits[0].vector_rank == 1
    assert hits[0].bm25_rank == 3
    # others are single-ranking documents: 'y' rank 1 -> 1/61
    assert hits[1].rrf_score == pytest.approx(1 / 61)
    assert hits[1].bm25_rank == 1 and hits[1].vector_rank is None
    # 'other' (first seen in the vector ranking) and 'z' tie at 1/62
    assert hits[2].rrf_score == pytest.approx(1 / 62)
    assert hits[3].rrf_score == pytest.approx(1 / 62)


def test_results_ordered_by_descending_rrf_score():
    dense_b, sparse_b = shared_hit_pair("b", i=1)
    service = fused_service(
        [vector_hit("a"), dense_b, vector_hit("c", i=2)],
        [sparse_b, bm25_hit("d", i=0)],
    )

    hits = service.search("query")

    scores = [hit.rrf_score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    # document in both rankings outranks all single-ranking documents
    assert hits[0].metadata["title"] == "b"
    assert hits[0].vector_rank == 2 and hits[0].bm25_rank == 1


def test_same_source_and_chunk_index_but_different_content_never_merges():
    """REGRESSION (real corpus): source#chunk0 keys collide across pages.

    Two DIFFERENT chunks (License vs Dependencies of the same homepage) must
    stay separate — previously they merged and each ranking's hits at different
    ranks accumulated into one inflated RRF entry (0.0928 instead of ~0.0308).
    """
    license_hit = SearchHit(
        content="This project is licensed.",
        source_url="https://fastapi.tiangolo.com/",
        title="License",
        heading_hierarchy=["FastAPI", "License"],
        similarity_score=0.8,
        chunk_index=0,
        metadata={"source": "https://fastapi.tiangolo.com/", "chunk_index": 0, "title": "License"},
    )  # type: ignore[call-arg]
    dependencies_bm25 = Bm25Hit(
        content="FastAPI depends on Pydantic and Starlette.",
        metadata={"source": "https://fastapi.tiangolo.com/", "chunk_index": 0, "title": "Dependencies"},
        bm25_score=2.0,
    )  # type: ignore[call-arg]

    service = fused_service([license_hit], [dependencies_bm25])

    hits = service.search("query", top_k=2)

    assert len(hits) == 2  # was 1 (false merge) before the content-based identity
    assert {hit.content for hit in hits} == {"This project is licensed.", "FastAPI depends on Pydantic and Starlette."}
    for hit in hits:
        assert hit.rrf_score == pytest.approx(1 / (DEFAULT_RRF_K + 1))
        assert (hit.vector_rank == 1) != (hit.bm25_rank == 1)
        if hit.bm25_rank == 1:
            assert hit.vector_rank is None
        else:
            assert hit.bm25_rank is None


def test_bm25_internal_same_key_hits_are_now_separate_entries():
    """REGRESSION: BM25 ranked 5 different chunks that shared key #c0 (ranks 3..7):
    their contributions accumulated on one entry -> RRF 0.0928 instead of 0.0308.
    Each rank must now contribute to its own entry only.
    """
    # two unrelated chunks ranked before (vector side only, dense ranking),
    # then five bm25 chunks that used to collide on "https://host/#chunk0"
    dense = [vector_hit(f"d{i}", i=i) for i in range(2)]
    shared_key_chunks = [
        bm25_hit(
            "host", i=0, content=f"chunk number {n} about routes"
        )  # same source, same chunk_index
        for n in range(5)
    ]
    service = fused_service(dense, shared_key_chunks)

    hits = service.search("query", top_k=20)

    # every hit has exactly ONE bm25 contribution (no multi-rank accumulation)
    bm25_hits = [hit for hit in hits if hit.vector_rank is None]
    assert len(bm25_hits) == 5
    for hit in bm25_hits:
        assert hit.bm25_rank in (1, 2, 3, 4, 5)
        assert hit.rrf_score == pytest.approx(1 / (DEFAULT_RRF_K + hit.bm25_rank))
    # order preserved: earlier bm25 ranks score higher
    ranks = [hit.bm25_rank for hit in bm25_hits]
    assert ranks == sorted(ranks)


def test_content_hash_identity_when_metadata_missing():
    generic = type(
        "Hit",
        (),
        {"content": "same content exactly", "metadata": {}, "source_url": "", "chunk_index": 0},
    )
    service = fused_service([generic()], [generic()])

    hits = service.search("query")

    assert len(hits) == 1  # merged via content hash
    assert hits[0].vector_rank == 1 and hits[0].bm25_rank == 1


def test_original_content_and_metadata_preserved():
    service = fused_service([vector_hit("v")], [bm25_hit("m")])

    hits = service.search("query")

    by_title = {hit.metadata["title"]: hit for hit in hits}
    assert by_title["v"].content == "vector content v"
    assert by_title["m"].content == "bm25 content m"
    assert by_title["m"].metadata["chunk_index"] == 0


def test_both_rankings_empty_returns_empty_list():
    service = fused_service([], [])

    assert service.search("anything") == []


def test_empty_query_returns_empty_list_without_calling_engines():
    engine = FakeEngine([vector_hit("a")])
    service = HybridSearchService(engine, FakeEngine([bm25_hit("a")]))

    assert service.search("   ") == []
    assert engine.calls == []


# -- configuration ----------------------------------------------------------------


def test_initial_configuration_defaults():
    service = HybridSearchService(FakeEngine([]), FakeEngine([]))

    assert service.vector_k == DEFAULT_VECTOR_K == 20
    assert service.bm25_k == DEFAULT_BM25_K == 20
    assert service.rrf_k == DEFAULT_RRF_K == 60
    assert service.top_k == DEFAULT_TOP_K == 10


def test_engines_retrieve_k_chunks_each():
    vector_engine = FakeEngine([vector_hit(f"v{i}") for i in range(25)])
    bm25_engine = FakeEngine([bm25_hit(f"b{i}") for i in range(25)])
    service = HybridSearchService(vector_engine, bm25_engine)

    service.search("query")

    assert vector_engine.calls == [20]
    assert bm25_engine.calls == [20]


def test_custom_parameter_values_are_used():
    top_dense = vector_hit("v0", content="the shared dual-listing chunk")
    sparse = bm25_hit("v0", content="the shared dual-listing chunk")
    vector_engine = FakeEngine([top_dense] + [vector_hit(f"v{i}") for i in range(1, 30)])
    bm25_engine = FakeEngine([sparse] + [bm25_hit(f"m{i}") for i in range(1, 30)])
    service = HybridSearchService(
        vector_engine,
        bm25_engine,
        vector_k=5,
        bm25_k=4,
        rrf_k=10,
        top_k=3,
    )

    hits = service.search("query")

    assert vector_engine.calls == [5]
    assert bm25_engine.calls == [4]
    assert len(hits) == 3  # top_k overrides the default of 10
    # with rrf_k=10, a chunk ranked 1st by both engines scores 1/11 + 1/11
    assert hits[0].rrf_score == pytest.approx(1 / 11 + 1 / 11)


def test_search_time_top_k_limits_final_results():
    service = fused_service(
        [vector_hit(f"v{i}", i=i) for i in range(12)], [],  # 12 unique
    )

    assert len(service.search("query")) == 10  # default


def test_default_top_k_caps_more_than_ten_unique_chunks():
    service = fused_service(
        [vector_hit(f"v{i}", i=i) for i in range(15)],
        [bm25_hit(f"m{i}", i=i) for i in range(15)],
    )

    hits = service.search("query")

    assert len(hits) == 10


# -- engine-agnostic fusion ------------------------------------------------------


def test_reciprocal_rank_fusion_is_engine_agnostic():
    """Fusion works on raw fake hits: content/metadata + identity fallback."""
    ranking1 = [
        type("H", (), {"content": "alpha text", "metadata": {}, "source_url": "", "chunk_index": 0})(),
        type("H", (), {"content": "beta text", "metadata": {}, "source_url": "", "chunk_index": 0})(),
    ]
    ranking2 = [
        type("H", (), {"content": "beta text", "metadata": {}, "source_url": "", "chunk_index": 0})(),
    ]

    fused = reciprocal_rank_fusion([("vector", ranking1), ("bm25", ranking2)], rrf_k=60)

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0][0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1][0].rrf_score == pytest.approx(1 / 61)
