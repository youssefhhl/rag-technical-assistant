"""Tests for report generation and metric aggregation (Phase 9.1).

No services, no LLM: the ReportBuilder is exercised directly. RAGAS scores
and performance/cost measurements stay in separate report blocks, and cost is
explicitly reported as unavailable.
"""

import json

from app.evaluation import ragas_compat  # noqa: F401  # shim before ragas
from app.evaluation.dataset import load_golden_dataset
from app.evaluation.evaluator import METRIC_NAMES
from app.evaluation.pipeline import is_abstained
from app.evaluation.report import ReportBuilder, report_path, save_report
from app.evaluation.samples import PipelineRunResult, responses_from_errors


def make_result(question: str, response: str = "ok", latency: float | None = 1.0):
    return PipelineRunResult(
        question=question, response=response, latency_seconds=latency
    )


def test_aggregate_averages_only_real_scores():
    dataset = load_golden_dataset()
    builder = ReportBuilder(dataset=dataset, evaluator_model="test-judge")
    q1, q2, q3 = [q.question for q in dataset.answerable[:3]]
    builder.add_sample(make_result(q1), {"faithfulness": 0.8, "answer_relevancy": None, "context_precision": None, "context_recall": None})
    builder.add_sample(make_result(q2), {"faithfulness": 1.0, "answer_relevancy": None, "context_precision": None, "context_recall": None})
    builder.add_sample(make_result(q3), {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None})
    aggregate = builder.aggregate_scores(METRIC_NAMES)
    assert aggregate["faithfulness"]["mean"] == (0.8 + 1.0) / 2
    assert aggregate["faithfulness"]["n_evaluated"] == 2
    assert aggregate["faithfulness"]["n_failed"] == 1


def test_report_contains_metadata_and_ragas_version():
    dataset = load_golden_dataset()
    builder = ReportBuilder(dataset=dataset, evaluator_model="groq/test", limit=5)
    report = builder.to_report([1.5, 2.5])
    assert report["ragas_version"]
    assert report["evaluator_model"] == "groq/test"
    assert report["dataset"]["name"] == "golden_dataset"
    assert report["dataset"]["num_questions"] == len(dataset)
    assert report["limit"] == 5


def test_performance_section_has_latency_and_unavailable_cost():
    dataset = load_golden_dataset()
    builder = ReportBuilder(dataset=dataset, evaluator_model="m")
    report = builder.to_report([2.0, 4.0])
    perf = report["performance"]
    assert perf["total_query_latency_seconds"] == 6.0
    assert perf["average_query_latency_seconds"] == 3.0
    assert perf["estimated_cost_per_query"] is None
    assert "unavailable" in perf["estimated_cost_note"]


def test_failed_pipeline_sample_is_reported_not_faked():
    dataset = load_golden_dataset()
    builder = ReportBuilder(dataset=dataset, evaluator_model="m")
    failure = responses_from_errors("q?", "HTTPError: 503", latency_seconds=0.5)
    builder.add_sample(failure, scores=None)
    report = builder.to_report([0.5])
    assert report["failed_or_skipped"]["pipeline_failures"] == 1
    assert report["failed_or_skipped"]["metric_failures"] == 0
    # the failure detail must show up with the exact error
    assert any("HTTPError: 503" in str(err) for err in report["failed_or_skipped"]["errors"].values())
    assert report["per_question"][0]["pipeline_error"] == "HTTPError: 503"
    assert report["per_question"][0]["response"] == ""


def test_abstention_observations_are_recorded_for_unanswerable_probes():
    dataset = load_golden_dataset()
    builder = ReportBuilder(dataset=dataset, evaluator_model="m")
    unanswerable_questions = [q.question for q in dataset.unanswerable]
    for question in unanswerable_questions:
        result = make_result(
            question,
            response="I don't have enough information to answer this question based on the provided documents.",
        )
        result.abstained = is_abstained(result.response)
        builder.add_sample(result, scores=None, is_probe=True)
    report = builder.to_report([])
    assert report["abstention"]["unanswerable_run"] == len(unanswerable_questions)
    assert report["abstention"]["abstained"] == len(unanswerable_questions)


def test_save_report_writes_valid_json(tmp_path):
    dataset = load_golden_dataset()
    builder = ReportBuilder(dataset=dataset, evaluator_model="m")
    report = builder.to_report([1.0])
    path = save_report(report, tmp_path / "rep.json")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["evaluator_model"] == "m"
    assert path == tmp_path / "rep.json"


def test_report_path_is_timestamped_json():
    path = report_path()
    assert path.suffix == ".json"
    assert path.name.startswith("evaluation_")
