"""Tests for the retrieval validation (Phase 10) - scores are mocked."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.retrieval.hybrid_search import HybridHit
from app.retrieval.reranker import RerankHit
from app.retrieval.validation import (
    RetrievalLabel,
    RetrievalValidator,
    collect_scores,
)


def make_settings(**overrides) -> Settings:
    values = {"_env_file": None, "scrape_request_delay_seconds": 0}
    values.update(overrides)
    return Settings(**values)


def scored_hit(score: float, index: int = 0) -> RerankHit:
    """A reranked hit with a Cross-Encoder score."""
    return RerankHit(
        content=f"chunk {index}",
        metadata={
            "source": f"https://docs.test/{index}",
            "title": "FastAPI",
            "chunk_index": index,
        },
        rerank_score=score,
    )


# -- classification ------------------------------------------------------------


def test_high_score_is_relevant():
    validator = RetrievalValidator(settings=make_settings())

    assessment = validator.validate([scored_hit(8.0)])

    assert assessment.label is RetrievalLabel.RELEVANT
    assert assessment.sufficient is True
    assert assessment.top_score == pytest.approx(8.0)


def test_medium_score_is_partially_relevant():
    validator = RetrievalValidator(settings=make_settings())

    assessment = validator.validate([scored_hit(-1.0)])

    assert assessment.label is RetrievalLabel.PARTIALLY_RELEVANT
    assert assessment.sufficient is True


def test_low_score_is_irrelevant():
    validator = RetrievalValidator(settings=make_settings())

    assessment = validator.validate([scored_hit(-9.0)])

    assert assessment.label is RetrievalLabel.IRRELEVANT
    assert assessment.sufficient is False


def test_classification_uses_the_best_hit_not_the_average():
    """One strong hit is enough: the answer may be in that chunk."""
    validator = RetrievalValidator(settings=make_settings())

    assessment = validator.validate(
        [scored_hit(-9.0, 0), scored_hit(6.0, 1), scored_hit(-8.0, 2)]
    )

    assert assessment.label is RetrievalLabel.RELEVANT
    assert assessment.top_score == pytest.approx(6.0)
    assert assessment.n_hits == 3
    assert assessment.n_relevant == 1


def test_score_exactly_on_the_threshold_is_relevant():
    validator = RetrievalValidator(settings=make_settings())

    assert (
        validator.validate([scored_hit(2.0)]).label is RetrievalLabel.RELEVANT
    )
    assert (
        validator.validate([scored_hit(-3.0)]).label
        is RetrievalLabel.PARTIALLY_RELEVANT
    )


# -- empty / unscored results ----------------------------------------------------


def test_empty_retrieval_is_irrelevant():
    validator = RetrievalValidator(settings=make_settings())

    assessment = validator.validate([])

    assert assessment.label is RetrievalLabel.IRRELEVANT
    assert assessment.sufficient is False
    assert assessment.n_hits == 0


def test_none_retrieval_is_irrelevant():
    validator = RetrievalValidator(settings=make_settings())

    assert validator.validate(None).label is RetrievalLabel.IRRELEVANT


def test_hits_without_cross_encoder_score_stay_permissive():
    """A pure-RRF configuration cannot be judged: never abstain on it."""
    validator = RetrievalValidator(settings=make_settings())
    hits = [
        HybridHit(content="chunk", metadata={"source": "https://x.test/"}, rrf_score=0.03)
    ]

    assessment = validator.validate(hits)

    assert assessment.label is RetrievalLabel.PARTIALLY_RELEVANT
    assert assessment.sufficient is True
    assert assessment.top_score is None


def test_collect_scores_skips_hits_without_score():
    hits = [scored_hit(1.5), HybridHit(content="c", metadata={}, rrf_score=0.03)]

    assert collect_scores(hits) == [pytest.approx(1.5)]


# -- configurable thresholds -------------------------------------------------------


def test_thresholds_come_from_the_settings():
    validator = RetrievalValidator(
        settings=make_settings(
            retrieval_relevant_threshold=5.0, retrieval_partial_threshold=1.0
        )
    )

    assert validator.relevant_threshold == 5.0
    assert validator.partial_threshold == 1.0
    assert validator.validate([scored_hit(3.0)]).label is RetrievalLabel.PARTIALLY_RELEVANT
    assert validator.validate([scored_hit(6.0)]).label is RetrievalLabel.RELEVANT


def test_thresholds_can_be_overridden_explicitly():
    validator = RetrievalValidator(settings=make_settings(), relevant_threshold=0.0)

    assert validator.validate([scored_hit(0.5)]).label is RetrievalLabel.RELEVANT


def test_default_thresholds_match_the_calibrated_values():
    """Calibrated on cross-encoder/ms-marco-MiniLM-L-6-v2 real logits."""
    validator = RetrievalValidator(settings=make_settings())

    assert validator.relevant_threshold == 2.0
    assert validator.partial_threshold == -3.0
