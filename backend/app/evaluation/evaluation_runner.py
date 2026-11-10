"""Evaluation runner: pipeline -> RAGAS -> JSON report (Phase 9.1), with a CLI.

Orchestration, in order:

1. load + validate the Golden Dataset (FastAPI-docs grounded);
2. iterate the (optionally limited) answerable questions;
3. run the REAL production pipeline per question (Qdrant + BM25 + RRF +
   Cross-Encoder + Groq), nothing mocked - see ``pipeline.py``;
4. run the unanswerable questions to observe abstention only;
5. map each sample to the RAGAS ``SingleTurnSample`` schema (retrieved vs
   reference material strictly separated - see ``samples.py``);
6. score the four RAGAS metrics sequentially (no concurrent bursts against
   Groq), collecting per-sample failures explicitly;
7. aggregate and persist the JSON report under ``reports/``.

Runs do not write to the conversations database (evaluation conversation id is
never appended to), and no failed sample is ever turned into a fabricated
score: it lands in ``failed_or_skipped`` with its error details.

CLI::

    uv run python -m app.evaluation.evaluation_runner --limit 5   # smoke run
    uv run python -m app.evaluation.evaluation_runner             # full run

requires GROQ_API_KEY plus the live services (PostgreSQL, Qdrant, BM25 index).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.evaluation import ragas_compat  # noqa: F401  # shim before ragas
from app.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    GoldenDataset,
    load_golden_dataset,
)
from app.evaluation.evaluator import METRIC_NAMES, RagasEvaluator, ScoreOutcome
from app.evaluation.pipeline import ProductionPipeline, is_abstained
from app.core.config import get_settings
from app.evaluation.report import ReportBuilder, report_path, save_report
from app.evaluation.samples import build_single_turn_sample

logger = logging.getLogger(__name__)


async def run_evaluation(
    dataset: GoldenDataset,
    *,
    dataset_limit: int | None = None,
    evaluator_model: str | None = None,
    pipeline: ProductionPipeline | None = None,
    save: bool = True,
    reports_dir: Path | None = None,
) -> dict:
    """Execute the full evaluation and return the report dict.

    ``pipeline`` / evaluator metrics are injectable for tests; production runs
    use the real services. Per-sample errors are recorded, never faked.
    """
    pipeline = pipeline or ProductionPipeline()
    evaluator = RagasEvaluator(evaluator_model=evaluator_model)
    builder = ReportBuilder(
        dataset=dataset,
        evaluator_model=evaluator.evaluator_model or "settings.llm_model",
        limit=dataset_limit,
        generator_model=getattr(pipeline, "model", None),
    )

    questions = dataset.answerable
    if dataset_limit is not None:
        questions = questions[: max(0, dataset_limit)]
    total = len(questions) + len(dataset.unanswerable)

    outputs: list = []
    for index, question in enumerate(questions, start=1):
        logger.info("[%d/%d] %s", index, total, question.question)
        result = pipeline.run_question(question.question)
        if result.error is None and not result.response.strip():
            # An empty answer means the LLM generation silently failed
            # (typically a provider rate limit): retry once before recording
            # it as an evaluation failure. An empty string must never be
            # scored as if it were a real answer.
            logger.warning("Empty generation for %r - retrying once", question.question)
            print("  -> empty answer, retrying once...")
            result = pipeline.run_question(question.question)
            if not result.response.strip():
                result.error = (
                    "generation returned an empty answer after one retry "
                    "(provider rate limit or outage)"
                )
                result.response = ""
                result.retrieved_contexts = []
        outputs.append(result)
        if result.error is not None:
            builder.add_sample(result, scores=None)
            print(f"  -> pipeline FAILED: {result.error}")
            continue
        print(f"  -> answered in {result.latency_seconds:.1f}s, scoring...")
        sample = build_single_turn_sample(
            question=question.question,
            response=result.response,
            retrieved_contexts=result.retrieved_contexts,
            reference=question.reference,
            reference_contexts=question.reference_contexts,
        )
        scores = await evaluator.score_sample(sample)
        builder.add_sample(
            result, scores=scores.scores, metric_errors=scores.errors,
            reference=question.reference,
        )
        failed = [name for name, s in scores.scores.items() if s is None]
        if failed:
            print(f"  -> metric failures: {', '.join(failed)}")

    # Unanswerable questions: abstention observation only, no RAGAS scoring.
    unanswerable = dataset.unanswerable
    for offset, question in enumerate(unanswerable, start=len(questions) + 1):
        logger.info("[%d/%d] (unanswerable probe) %s", offset, total, question.question)
        result = pipeline.run_question(question.question)
        outputs.append(result)
        result.abstained = is_abstained(result.response)
        builder.add_sample(result, scores=None, is_probe=True)

    latencies = [
        result.latency_seconds
        for result in outputs
        if result.latency_seconds is not None
    ]
    report = builder.to_report(latencies)
    if save:
        path = report_path(reports_dir)
        save_report(report, path)
        print(f"\nReport saved: {path}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments: dataset, limit, output dir, evaluator model override."""
    parser = argparse.ArgumentParser(
        prog="app.evaluation.evaluation_runner",
        description=(
            "Evaluate the production RAG pipeline with RAGAS (faithfulness, "
            "answer relevancy, context precision, context recall). Requires "
            "GROQ_API_KEY and the live services (PostgreSQL, Qdrant, BM25)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="evaluate only the first N answerable questions (development)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"path to the golden dataset JSON (default: {DEFAULT_DATASET_PATH.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="directory for the JSON report (default: app/evaluation/reports)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="override the RAGAS evaluator model (LiteLLM-style 'groq/<model>')",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: full evaluation against the live services."""
    args = parse_args(argv)
    dataset = load_golden_dataset(args.dataset)
    print(
        f"Golden dataset: {dataset.name} "
        f"({len(dataset.answerable)} answerable, {len(dataset.unanswerable)} unanswerable)"
    )
    report = asyncio.run(
        run_evaluation(
            dataset,
            dataset_limit=args.limit,
            evaluator_model=args.model,
            reports_dir=args.output,
        )
    )
    print("\n=== Aggregate RAGAS scores ===")
    for name, stats in report["metrics"].items():
        print(
            f"  {name:>18}: mean={stats['mean']} "
            f"(n={stats['n_evaluated']}, failed={stats['n_failed']})"
        )
    print("\n=== Performance ===")
    perf = report["performance"]
    print(f"  avg latency per query: {perf['average_query_latency_seconds']}s")
    print(f"  estimated cost per query: {perf['estimated_cost_per_query']}")
    failures = report["failed_or_skipped"]
    print(
        f"  failed/skipped: {failures['pipeline_failures']} pipeline, "
        f"{failures['metric_failures']} metric"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
