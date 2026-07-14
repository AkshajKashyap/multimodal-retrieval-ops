from dataclasses import replace

import pytest

from multimodal_retrieval_ops.retrieval_monitoring import (
    HealthThresholds,
    RetrievalMonitoringError,
    TelemetryReadResult,
    analyze_events,
    render_monitoring_report,
)
from test_milestone_eleven_a import event


def read(events):
    return TelemetryReadResult(events, 0, 0, 0, 1)


def labeled_event(*, recall: float = 1.0, reciprocal_rank: float = 1.0):
    return event(
        ground_truth_relevance_available=True,
        recall_at_1=recall,
        recall_at_5=recall,
        recall_at_10=recall,
        reciprocal_rank=reciprocal_rank,
    )


def readiness_event(*, failed: bool = False):
    return event(
        endpoint="readiness",
        retrieval_direction="operational",
        request_status="failed" if failed else "success",
        error_category="readiness_failure" if failed else None,
    )


def failed_event():
    return event(
        request_status="failed",
        http_status_code=404,
        error_category="unknown_cached_id",
    )


def five_event_window():
    return [
        event(endpoint="health", retrieval_direction="operational"),
        readiness_event(),
        labeled_event(recall=0.0, reciprocal_rank=0.5),
        labeled_event(recall=1.0, reciprocal_rank=0.5),
        failed_event(),
    ]


def test_five_event_smoke_window_is_insufficient_data() -> None:
    summary = analyze_events(read(five_event_window()))
    checks = summary["health"]["threshold_results"]
    assert summary["health"]["decision"] == "insufficient_data"
    assert checks["maximum_error_rate"]["result"] == "insufficient_data"
    assert checks["maximum_error_rate"]["observation_count"] == 5
    assert checks["maximum_readiness_failures"]["result"] == "pass"
    assert checks["minimum_labeled_recall_at_10"]["result"] == "insufficient_data"
    assert checks["minimum_labeled_mrr"]["result"] == "insufficient_data"
    report = render_monitoring_report(summary)
    assert "No production-health conclusion can be drawn" in report
    assert "validates telemetry collection and analysis only" in report


def test_two_labeled_queries_do_not_evaluate_quality_thresholds() -> None:
    summary = analyze_events(read([readiness_event(), labeled_event(), labeled_event()]))
    checks = summary["health"]["threshold_results"]
    recall = checks["minimum_labeled_recall_at_10"]
    mrr = checks["minimum_labeled_mrr"]
    assert recall["observation_count"] == mrr["observation_count"] == 2
    assert recall["minimum_required_observations"] == 50
    assert mrr["minimum_required_observations"] == 50
    assert recall["sufficient_data"] is mrr["sufficient_data"] is False
    assert recall["result"] == mrr["result"] == "insufficient_data"


def test_low_sample_error_rate_is_insufficient_data() -> None:
    summary = analyze_events(read([failed_event()] * 5))
    check = summary["health"]["threshold_results"]["maximum_error_rate"]
    assert check["observed_value"] == 1.0
    assert check["observation_count"] == 5
    assert check["result"] == "insufficient_data"
    assert summary["health"]["decision"] == "insufficient_data"


@pytest.mark.parametrize(
    ("failure_count", "expected_result", "expected_decision"),
    [(4, "pass", "insufficient_data"), (6, "fail", "warning")],
)
def test_sufficient_error_rate_pass_and_failure(
    failure_count: int, expected_result: str, expected_decision: str
) -> None:
    events = [failed_event()] * failure_count + [event()] * (20 - failure_count)
    summary = analyze_events(read(events))
    check = summary["health"]["threshold_results"]["maximum_error_rate"]
    assert check["observation_count"] == 20
    assert check["sufficient_data"] is True
    assert check["result"] == expected_result
    assert summary["health"]["decision"] == expected_decision


def test_latency_sufficiency_pass_and_failure() -> None:
    thresholds = HealthThresholds(
        maximum_p95_latency_ms=15.0,
        min_latency_observations=3,
    )
    insufficient = analyze_events(read([event(latency_ms=10.0)] * 2), thresholds)
    check = insufficient["health"]["threshold_results"]["maximum_p95_latency_ms"]
    assert check["result"] == "insufficient_data"
    passing = analyze_events(
        read([event(latency_ms=value) for value in (5.0, 10.0, 15.0)]), thresholds
    )
    assert (
        passing["health"]["threshold_results"]["maximum_p95_latency_ms"]["result"]
        == "pass"
    )
    failing = analyze_events(
        read([event(latency_ms=value) for value in (5.0, 10.0, 16.0)]), thresholds
    )
    assert (
        failing["health"]["threshold_results"]["maximum_p95_latency_ms"]["result"]
        == "fail"
    )
    assert failing["health"]["decision"] == "warning"


def test_readiness_minimum_pass_and_sufficient_failure() -> None:
    passing = analyze_events(read([readiness_event()]))
    check = passing["health"]["threshold_results"]["maximum_readiness_failures"]
    assert check["observation_count"] == 1
    assert check["result"] == "pass"
    failing = analyze_events(read([readiness_event(failed=True)]))
    failed_check = failing["health"]["threshold_results"]["maximum_readiness_failures"]
    assert failed_check["observed_value"] == 1
    assert failed_check["result"] == "fail"
    assert failing["health"]["decision"] == "warning"


def test_healthy_requires_every_enabled_check_to_pass_with_enough_data() -> None:
    events = [readiness_event(), *[labeled_event() for _ in range(50)]]
    summary = analyze_events(read(events))
    assert summary["health"]["decision"] == "healthy"
    enabled = [
        check
        for check in summary["health"]["threshold_results"].values()
        if check["result"] != "disabled"
    ]
    assert enabled and all(check["result"] == "pass" for check in enabled)


def test_warning_when_one_sufficiently_sampled_check_fails() -> None:
    events = [
        readiness_event(),
        *[labeled_event(recall=0.0, reciprocal_rank=1.0) for _ in range(50)],
    ]
    summary = analyze_events(read(events))
    assert summary["health"]["decision"] == "warning"
    assert (
        summary["health"]["threshold_results"]["minimum_labeled_recall_at_10"][
            "result"
        ]
        == "fail"
    )


def test_insufficient_when_no_failures_but_one_check_lacks_data() -> None:
    summary = analyze_events(read([labeled_event() for _ in range(50)]))
    assert summary["health"]["decision"] == "insufficient_data"
    assert (
        summary["health"]["threshold_results"]["maximum_readiness_failures"]["result"]
        == "insufficient_data"
    )


def test_disabled_optional_thresholds_do_not_block_healthy() -> None:
    thresholds = HealthThresholds(
        maximum_p95_latency_ms=None,
        minimum_labeled_recall_at_10=None,
        minimum_labeled_mrr=None,
    )
    summary = analyze_events(
        read([readiness_event(), *[event(endpoint="health") for _ in range(19)]]),
        thresholds,
    )
    assert summary["health"]["decision"] == "healthy"
    checks = summary["health"]["threshold_results"]
    assert checks["maximum_p95_latency_ms"]["result"] == "disabled"
    assert checks["minimum_labeled_recall_at_10"]["result"] == "disabled"
    assert checks["minimum_labeled_mrr"]["result"] == "disabled"


@pytest.mark.parametrize(
    "field",
    [
        "min_error_rate_observations",
        "min_latency_observations",
        "min_labeled_recall_observations",
        "min_labeled_mrr_observations",
        "min_readiness_observations",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_minimum_observations_are_rejected(field: str, value: int) -> None:
    with pytest.raises(RetrievalMonitoringError, match="minimum observation"):
        replace(HealthThresholds(), **{field: value}).validate()


def test_arbitrary_queries_do_not_count_as_labeled_quality() -> None:
    arbitrary = [
        event(
            endpoint="arbitrary_text_to_image",
            retrieval_direction="text_to_image",
            ground_truth_relevance_available=False,
        )
        for _ in range(60)
    ]
    summary = analyze_events(read([readiness_event(), *arbitrary]))
    assert summary["known_label_quality"]["labeled_query_count"] == 0
    recall = summary["health"]["threshold_results"]["minimum_labeled_recall_at_10"]
    mrr = summary["health"]["threshold_results"]["minimum_labeled_mrr"]
    assert recall["observation_count"] == mrr["observation_count"] == 0
    assert recall["result"] == mrr["result"] == "insufficient_data"
