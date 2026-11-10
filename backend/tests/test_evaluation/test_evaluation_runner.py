"""Tests for the CLI + orchestration of the evaluation runner (Phase 9.1).

The production pipeline and the RAGAS evaluator are faked here (monkeypatched
``ProductionPipeline`` / ``RagasEvaluator``): the orchestration, mapping,
failure handling, abstention probes and report persistence are real logic.
No service, no API key required.
"""

import asyncio
import json
from pathlib import Path

import pytest

import app.evaluation.evaluation_runner as runner_module
from app.evaluation.dataset import ABSTENTION_PLACEHOLDER, load_golden_dataset
from app.evaluation.evaluation_runner import main, parse_args, run_evaluation
from app.evaluation.evaluator import ScoreOutcome
from app.evaluation.samples import (
    PipelineRunResult,
    responses_from_errors,
)


def fake_pipeline(responses: dict[str, PipelineRunResult]):
    """A duck-typed ProductionPipeline serving canned results.

    Unknown questions (unanswerable probes) get an abstention response, which
    is exactly how the pipeline should behave for out-of-corpus questions.
    """

    class FakePipeline:
        def run_question(self, question: str) -> PipelineRunResult:
            return responses.get(
                question,
                PipelineRunResult(
                    question=question,
                    response=ABSTENTION_PLACEHOLDER,
                    retrieved_contexts=["unrelated chunk"],
                    latency_seconds=1.0,
                ),
            )

    return FakePipeline()


# -- CLI argument handling ------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args([])
    assert args.limit is None
    assert args.model is None
    assert args.output is None
    assert args.dataset.name == "golden_dataset.json"


def test_parse_args_limit_and_dataset_backward_compatibility():
    args = parse_args(["--limit", "5", "--dataset", "custom.json", "--model", "groq/x"])
    assert args.limit == 5
    assert args.dataset == Path("custom.json")
    assert args.model == "groq/x"


# -- orchestration --------------------------------------------------------------


def test_run_evaluation_scores_answerable_questions(monkeypatch, tmp_path):
    dataset = load_golden_dataset()
    answerable = dataset.answerable[:2]

    expected_references = {q.question: q.reference for q in answerable}

    class FakeEvaluator:
        def __init__(self, evaluator_model=None):
            self.evaluator_model = evaluator_model or "fake-judge"

        async def score_sample(self, sample) -> ScoreOutcome:
            assert sample.retrieved_contexts, "pipeline contexts must flow through"
            assert sample.reference == expected_references[sample.user_input]
            assert set(sample.retrieved_contexts) == {"chunk-1", "chunk-2"}
            return ScoreOutcome(
                scores={
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                    "context_precision": 1.0,
                    "context_recall": 0.7,
                }
            )

    monkeypatch.setattr(runner_module, "RagasEvaluator", FakeEvaluator)
    responses = {
        q.question: PipelineRunResult(
            question=q.question,
            response="generated answer",
            retrieved_contexts=["chunk-1", "chunk-2"],
            latency_seconds=1.5,
        )
        for q in answerable
    }
    report = asyncio.run(run_evaluation(
        dataset,
        dataset_limit=2,
        pipeline=fake_pipeline(responses),
        save=True,
        reports_dir=tmp_path,
    ))
    assert list(report["metrics"]) == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    assert report["samples_evaluated"] == 2
    assert report["metrics"]["faithfulness"]["mean"] == 0.9
    # average over all successful queries (2 answerable @1.5s + 5 probes @1.0s)
    expected_avg = round((1.5 + 1.5 + 5 * 1.0) / 7, 3)
    assert report["performance"]["average_query_latency_seconds"] == expected_avg
    assert report["failed_or_skipped"]["pipeline_failures"] == 0
    report_files = list(tmp_path.glob("evaluation_*.json"))
    assert len(report_files) == 1


def test_run_evaluation_records_pipeline_failure_without_faking(monkeypatch, tmp_path):
    dataset = load_golden_dataset()
    q = dataset.answerable[0]
    failure = responses_from_errors(q.question, "ConnectionError: ref", latency_seconds=0.4)

    class FakeEvaluator:
        def __init__(self, evaluator_model=None):
            self.evaluator_model = "fake-judge"

        async def score_sample(self, sample) -> ScoreOutcome:  # pragma: no cover
            raise AssertionError("a failed sample must never be scored")

    monkeypatch.setattr(runner_module, "RagasEvaluator", FakeEvaluator)
    report = asyncio.run(run_evaluation(
        dataset,
        dataset_limit=1,
        pipeline=fake_pipeline({q.question: failure}),
        save=False))
    assert report["failed_or_skipped"]["pipeline_failures"] == 1
    assert report["metrics"]["faithfulness"]["mean"] is None
    assert report["metrics"]["faithfulness"]["n_evaluated"] == 0
    assert any("ConnectionError: ref" in str(e) for e in report["failed_or_skipped"]["errors"].values())


def test_run_evaluation_probes_abstention_for_unanswerable(monkeypatch, tmp_path):
    dataset = load_golden_dataset()
    answerable = dataset.answerable[:1]
    unanswerable = dataset.unanswerable

    class FakeEvaluator:
        def __init__(self, evaluator_model=None):
            self.evaluator_model = "fake-judge"

        async def score_sample(self, sample) -> ScoreOutcome:
            return ScoreOutcome(
                scores={
                    "faithfulness": 1.0,
                    "answer_relevancy": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                }
            )

    monkeypatch.setattr(runner_module, "RagasEvaluator", FakeEvaluator)
    responses = {
        **{
            q.question: PipelineRunResult(
                question=q.question, response="grounded answer",
                retrieved_contexts=["c"], latency_seconds=1.0,
            )
            for q in answerable
        },
        **{
            q.question: PipelineRunResult(
                question=q.question,
                response=ABSTENTION_PLACEHOLDER,
                retrieved_contexts=["c"],
                latency_seconds=1.0,
            )
            for q in unanswerable
        },
    }
    report = asyncio.run(run_evaluation(
        dataset,
        dataset_limit=1,
        pipeline=fake_pipeline(responses),
        save=False))
    # Unanswerable questions are probed, never scored against a reference.
    assert report["samples_evaluated"] == 1
    assert report["abstention"]["unanswerable_run"] == len(unanswerable)
    assert report["abstention"]["abstained"] == len(unanswerable)


def test_limit_selects_first_n_answerable(monkeypatch, tmp_path):
    dataset = load_golden_dataset()

    class FakeEvaluator:
        def __init__(self, evaluator_model=None):
            self.evaluator_model = "fake-judge"

        async def score_sample(self, sample) -> dict:
            return ScoreOutcome(
                scores={
                    "faithfulness": 1.0, "answer_relevancy": 1.0,
                    "context_precision": 1.0, "context_recall": 1.0,
                }
            )

    monkeypatch.setattr(runner_module, "RagasEvaluator", FakeEvaluator)
    seen = []

    class RecordingPipeline:
        def run_question(self, question):
            seen.append(question)
            return PipelineRunResult(question=question, response="a",
                                     retrieved_contexts=["c"], latency_seconds=0.1)

    asyncio.run(run_evaluation(
        dataset, dataset_limit=3, pipeline=RecordingPipeline(), save=False,
    ))
    answerable_three = {q.question for q in dataset.answerable[:3]}
    assert set(seen) == answerable_three | {q.question for q in dataset.unanswerable}


# -- main() exit path (uppy patched to avoid services) ---------------------------


def test_main_writes_report_and_returns_zero(monkeypatch, tmp_path):
    dataset = load_golden_dataset()

    class FakeEvaluator:
        def __init__(self, evaluator_model=None):
            self.evaluator_model = "fake-judge"

        async def score_sample(self, sample) -> dict:
            return ScoreOutcome(
                scores={
                    "faithfulness": 0.5, "answer_relevancy": 0.5,
                    "context_precision": 0.5, "context_recall": 0.5,
                }
            )

    monkeypatch.setattr(runner_module, "RagasEvaluator", FakeEvaluator)
    responses = {
        q.question: PipelineRunResult(
            question=q.question, response="a", retrieved_contexts=["c"],
            latency_seconds=0.2,
        )
        for q in dataset.answerable[:1]
    }
    responses.update(
        {
            q.question: PipelineRunResult(
                question=q.question, response=ABSTENTION_PLACEHOLDER,
                retrieved_contexts=["c"], latency_seconds=0.2,
            )
            for q in dataset.unanswerable
        }
    )
    fake = fake_pipeline(responses)
    monkeypatch.setattr(
        runner_module, "ProductionPipeline", lambda: fake
    )
    exit_code = main(["--limit", "1", "--output", str(tmp_path)])
    assert exit_code == 0
    files = list(tmp_path.glob("evaluation_*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text(encoding="utf-8"))
    assert report["limit"] == 1
