"""Golden dataset loading and validation (Phase 9.1).

The golden dataset lives at ``datasets/golden_dataset.json`` and contains
questions grounded in the documentation actually indexed today (FastAPI docs
only). Two kinds of entries are supported:

- answerable  (default): has a manually verified ``reference`` answer and
  ``reference_contexts`` excerpts from the official docs;
- unanswerable: ``"answerable": false``, used to observe the pipeline's
  abstention behaviour. They never take part in reference-based RAGAS metric
  computation.

Validation is deliberately strict: a malformed or placeholder entry raises
``pydantic.ValidationError`` at load time instead of silently degrading an
evaluation run.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "golden_dataset.json"

ABSTENTION_PLACEHOLDER = (
    "I don't have enough information to answer this question "
    "based on the provided documents."
)


class GoldenQuestion(BaseModel):
    """One Golden Dataset entry (strictly validated)."""

    question: str
    reference: str = ""
    reference_contexts: list[str] = Field(default_factory=list)
    answerable: bool = True

    @field_validator("question")
    @classmethod
    def _question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value

    @model_validator(mode="after")
    def _answerable_entries_are_grounded(self) -> "GoldenQuestion":
        """Answerable entries must be manually grounded (verified answer + excerpts)."""
        if self.answerable:
            if not self.reference.strip():
                raise ValueError("answerable questions require a non-empty reference")
            if not any(c.strip() for c in self.reference_contexts):
                raise ValueError(
                    "answerable questions require at least one reference_context"
                )
        self.reference = self.reference.strip()
        self.reference_contexts = [c.strip() for c in self.reference_contexts if c.strip()]
        return self


class GoldenDataset:
    """A loaded and validated Golden Dataset."""

    def __init__(self, questions: list[GoldenQuestion], source: Path) -> None:
        self.questions = questions
        self.source = source

    @property
    def name(self) -> str:
        return self.source.stem

    @property
    def answerable(self) -> list[GoldenQuestion]:
        """Questions evaluated against a reference with RAGAS metrics."""
        return [q for q in self.questions if q.answerable]

    @property
    def unanswerable(self) -> list[GoldenQuestion]:
        """Questions outside the indexed corpus, probing abstention only."""
        return [q for q in self.questions if not q.answerable]

    def __len__(self) -> int:
        return len(self.questions)


def _validate_shape(data: object, source: Path) -> list[GoldenQuestion]:
    if isinstance(data, dict):
        data = data.get("questions")
    if not isinstance(data, list):
        raise ValueError(
            f"{source}: expected a top-level list of question objects "
            "(or a dict with a 'questions' key), got "
            f"{type(data).__name__}"
        )
    return [GoldenQuestion.model_validate(entry) for entry in data]


def load_golden_dataset(path: Path | str | None = None) -> GoldenDataset:
    """Load and validate the Golden Dataset JSON.

    Raises a clear ``FileNotFoundError`` when the file is missing and
    ``pydantic.ValidationError`` / ``ValueError`` for malformed content.
    """
    source = Path(path) if path is not None else DEFAULT_DATASET_PATH
    if not source.is_file():
        raise FileNotFoundError(f"Golden dataset not found at {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    questions = _validate_shape(raw, source)
    if not questions:
        raise ValueError(f"{source}: dataset contains no questions")
    return GoldenDataset(questions=questions, source=source)
