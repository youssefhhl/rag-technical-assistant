"""Tests for the Golden Dataset loader (Phase 9.1).

Unit tests only: no services, no LLM. The default dataset file is validated
as a whole so a bad entry cannot silently reach an evaluation run.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    GoldenQuestion,
    load_golden_dataset,
)


# -- default dataset ------------------------------------------------------------


def test_default_dataset_loads_and_validates():
    dataset = load_golden_dataset()
    assert dataset.name == "golden_dataset"
    assert len(dataset) >= 30, "the dataset should contain 30-50 questions"
    assert 30 <= len(dataset.answerable) <= 50
    assert len(dataset.unanswerable) >= 1


def test_all_answerable_entries_are_grounded():
    dataset = load_golden_dataset()
    for question in dataset.answerable:
        assert question.reference.strip(), f"missing reference: {question.question!r}"
        assert question.reference_contexts, f"missing contexts: {question.question!r}"


def test_unanswerable_questions_have_empty_reference():
    dataset = load_golden_dataset()
    for question in dataset.unanswerable:
        assert question.answerable is False
        assert not question.reference.strip()
        assert question.reference_contexts == []


def test_questions_are_unique():
    dataset = load_golden_dataset()
    questions = [q.question for q in dataset.questions]
    assert len(questions) == len(set(questions)), "duplicate/near-identical questions"


# -- loader edge cases ----------------------------------------------------------


def test_missing_file_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_golden_dataset(Path("/nonexistent/dataset.json"))


def test_wrong_top_level_shape_raises():
    import json, tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps({"questions": "not a list"}), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level list"):
            load_golden_dataset(path)


def test_empty_dataset_raises():
    import json, tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.json"
        path.write_text(json.dumps([]), encoding="utf-8")
        with pytest.raises(ValueError, match="no questions"):
            load_golden_dataset(path)


# -- entry validation -----------------------------------------------------------


def test_answerable_entry_without_reference_is_rejected():
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate(
            {"question": "Q?", "reference": "", "reference_contexts": ["ctx"]}
        )


def test_answerable_entry_without_reference_contexts_is_rejected():
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate({"question": "Q?", "reference": "answer"})


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate({"question": "   ", "reference": "ans"})


def test_contexts_are_stripped_at_load():
    entry = GoldenQuestion.model_validate(
        {"question": "Q?", "reference": "ref", "reference_contexts": ["  ctx  ", "   ", "ctx2"]}
    )
    assert entry.reference_contexts == ["ctx", "ctx2"]


def test_default_dataset_path_exists():
    assert DEFAULT_DATASET_PATH.is_file()
