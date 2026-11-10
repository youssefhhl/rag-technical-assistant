"""Evaluation report building and persistence (Phase 9.1).

One JSON report per run under ``reports/``: metadata, aggregate scores per
metric (averaged only over samples that actually scored), per-question rows,
failures with their error details, and the performance section. RAGAS scores
and latency/cost measurements are kept in separate blocks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation import ragas_compat  # noqa: F401  # shim before any ragas import
from app.evaluation.dataset import ABSTENTION_PLACEHOLDER, GoldenDataset
from app.evaluation.samples import PipelineRunResult

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

import ragas  # noqa: E402

RAGAS_VERSION = ragas.__version__


@dataclass
class ReportBuilder:
    """Accumulates sample scores + failures, then emits the JSON report."""

    dataset: GoldenDataset
    evaluator_model: str
    limit: int | None = None
    generator_model: str | None = None
    started_utc: float = field(default_factory=time.time)
    per_question: list[dict] = field(default_factory=list)
    metric_errors: dict[str, list[str]] = field(default_factory=dict)
    metric_error_messages: list[tuple[str, str]] = field(default_factory=list)

    def add_sample(
        self,
        result: PipelineRunResult,
        scores: dict[str, float | None] | None,
        reference: str = "",
        metric_errors: dict[str, str] | None = None,
        *,
        is_probe: bool = False,
    ) -> None:
        """Record one sample: pipeline row plus its metric scores (or errors).

        ``metric_errors`` maps failed metric names to their exact error
        message (kept in the report for debugging). ``is_probe`` marks
        unanswerable questions: they are recorded (response + latency +
        abstention) but never counted as metric failures, since no
        reference-based metric is computed for them.
        """
        metric_errors = metric_errors or {}
        row: dict = {
            "question": result.question,
            "latency_seconds": (
                round(result.latency_seconds, 3)
                if result.latency_seconds is not None
                else None
            ),
            "pipeline_error": result.error,
            "response": result.response,
            "abstained": result.abstained,
        }
        if result.error is None and scores is not None:
            row["scores"] = {
                name: (round(value, 4) if value is not None else None)
                for name, value in scores.items()
            }
            if not is_probe:
                for name, value in scores.items():
                    if value is None:
                        self.metric_errors.setdefault(name, []).append(result.question)
                        if name in metric_errors:
                            self.metric_error_messages.append(
                                {"metric": name, "question": result.question, "error": metric_errors[name]}
                            )
        else:
            # A failed pipeline run cannot be scored - recorded as failed.
            row["scores"] = {}
            if not is_probe:
                for name in (
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ):
                    self.metric_errors.setdefault(name, []).append(row["question"])
        row["reference"] = reference
        # Phase 10 reliability layer (absent when it is disabled, so the
        # Phase 9 report format stays unchanged).
        if result.retrieval_label is not None:
            row["retrieval_label"] = result.retrieval_label
        if result.corrective_used:
            row["corrective_used"] = True
        self.per_question.append(row)

    # -- aggregation ------------------------------------------------------------

    def aggregate_scores(self, metric_names: tuple[str, ...]) -> dict[str, dict]:
        """Mean score per metric over samples that produced a real score."""
        aggregates: dict[str, dict] = {}
        for name in metric_names:
            values = [
                row["scores"][name]
                for row in self.per_question
                if name in row.get("scores", {}) and row["scores"][name] is not None
            ]
            aggregates[name] = {
                "mean": round(sum(values) / len(values), 4) if values else None,
                "n_evaluated": len(values),
                "n_failed": len(self.metric_errors.get(name, [])),
            }
        return aggregates

    # -- report -----------------------------------------------------------------

    def to_report(self, latency_samples: list[float]) -> dict:
        """Assemble the report dict (no filesystem writes)."""
        evaluated = [
            row for row in self.per_question if "scores" in row and row["scores"]
        ]
        answerable_rows = [
            row for row in evaluated if row["question"] in
            {q.question for q in self.dataset.answerable}
        ]
        unanswerable_rows = [
            row for row in self.per_question
            if row["question"] in {q.question for q in self.dataset.unanswerable}
        ]
        n_failed_pipelines = sum(1 for row in self.per_question if row["pipeline_error"])
        n_failed_metrics = sum(
            1 for row in evaluated for value in row["scores"].values() if value is None
        )
        timestamp = datetime.fromtimestamp(self.started_utc, tz=timezone.utc)
        return {
            "timestamp_utc": timestamp.isoformat(timespec="seconds"),
            "dataset": {
                "name": self.dataset.name,
                "num_questions": len(self.dataset),
                "answerable": len(self.dataset.answerable),
                "unanswerable": len(self.dataset.unanswerable),
            },
            "limit": self.limit,
            "ragas_version": RAGAS_VERSION,
            "evaluator_model": self.evaluator_model,
            "generator_model": self.generator_model,
            "metrics": self.aggregate_scores(
                ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
            ),
            "samples_evaluated": len(answerable_rows),
            "failed_or_skipped": {
                "pipeline_failures": n_failed_pipelines,
                "metric_failures": n_failed_metrics,
                "errors": {
                    row["question"]: row["pipeline_error"] or "metric failure"
                    for row in self.per_question
                    if row["pipeline_error"] or (
                        row.get("scores")
                        and any(value is None for value in row["scores"].values())
                    )
                },
                "metric_error_details": self.metric_error_messages,
            },
            "abstention": {
                "unanswerable_run": len(unanswerable_rows),
                "abstained": sum(1 for r in unanswerable_rows if r["abstained"]),
            },
            "performance": {
                "total_query_latency_seconds": (
                    round(sum(latency_samples), 3) if latency_samples else None
                ),
                "average_query_latency_seconds": (
                    round(sum(latency_samples) / len(latency_samples), 3)
                    if latency_samples
                    else None
                ),
                "estimated_cost_per_query": None,
                "estimated_cost_note": (
                    "unavailable - the production LiteLLM call does not expose "
                    "reliable token usage/pricing without modifying production code"
                ),
            },
            "per_question": self.per_question,
        }


def report_path(reports_dir: Path | None = None) -> Path:
    """Timestamped report file path (the caller decides whether to write it)."""
    directory = reports_dir or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return directory / f"evaluation_{stamp}.json"


def save_report(report: dict, path: Path) -> Path:
    """Write the JSON report and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info("Report written to %s", path)
    return path
