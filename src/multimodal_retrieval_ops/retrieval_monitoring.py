"""Deterministic offline aggregation of privacy-safe retrieval telemetry."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .api.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    RetrievalEvent,
    TelemetryValidationError,
    event_from_mapping,
)

MONITORING_RUN_STATES = {
    "success",
    "not_run",
    "dependency_unavailable",
    "artifact_unavailable",
    "telemetry_unavailable",
    "telemetry_invalid",
    "execution_failed",
}


class RetrievalMonitoringError(ValueError):
    """Expected telemetry availability, parsing, or analysis error."""


class TelemetryUnavailableError(RetrievalMonitoringError):
    pass


@dataclass(frozen=True)
class HealthThresholds:
    maximum_error_rate: float = 0.25
    maximum_readiness_failures: int = 0
    maximum_p95_latency_ms: float | None = None
    minimum_labeled_recall_at_10: float | None = 0.80
    minimum_labeled_mrr: float | None = 0.50
    min_error_rate_observations: int = 20
    min_latency_observations: int = 20
    min_labeled_recall_observations: int = 50
    min_labeled_mrr_observations: int = 50
    min_readiness_observations: int = 1

    def validate(self) -> None:
        probabilities = (
            self.maximum_error_rate,
            self.minimum_labeled_recall_at_10,
            self.minimum_labeled_mrr,
        )
        if any(value is not None and not 0.0 <= value <= 1.0 for value in probabilities):
            raise RetrievalMonitoringError("probability thresholds must be between 0 and 1")
        if self.maximum_readiness_failures < 0:
            raise RetrievalMonitoringError("maximum readiness failures must be non-negative")
        if self.maximum_p95_latency_ms is not None and self.maximum_p95_latency_ms <= 0:
            raise RetrievalMonitoringError("maximum p95 latency must be positive")
        minimums = (
            self.min_error_rate_observations,
            self.min_latency_observations,
            self.min_labeled_recall_observations,
            self.min_labeled_mrr_observations,
            self.min_readiness_observations,
        )
        if any(value <= 0 for value in minimums):
            raise RetrievalMonitoringError("minimum observation counts must be positive")


@dataclass(frozen=True)
class TelemetryReadResult:
    events: list[RetrievalEvent]
    malformed_json_line_count: int
    invalid_event_record_count: int
    unsupported_schema_record_count: int
    file_count: int

    @property
    def parsing_error_count(self) -> int:
        return (
            self.malformed_json_line_count
            + self.invalid_event_record_count
            + self.unsupported_schema_record_count
        )


def _rotated_paths(path: Path) -> list[Path]:
    siblings = []
    for candidate in path.parent.glob(f"{path.name}.*"):
        suffix = candidate.name.removeprefix(f"{path.name}.")
        if suffix.isdigit() and candidate.is_file():
            siblings.append((int(suffix), candidate))
    return [candidate for _, candidate in sorted(siblings, reverse=True)]


def read_telemetry(path: Path, *, include_rotated: bool = False) -> TelemetryReadResult:
    paths = ([*_rotated_paths(path), path] if include_rotated else [path])
    existing = [candidate for candidate in paths if candidate.is_file()]
    if not existing:
        raise TelemetryUnavailableError("telemetry input is unavailable")
    events: list[RetrievalEvent] = []
    malformed = 0
    invalid = 0
    unsupported = 0
    for candidate in existing:
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            invalid += 1
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(data, dict):
                invalid += 1
                continue
            if data.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
                unsupported += 1
                continue
            try:
                events.append(event_from_mapping(data))
            except TelemetryValidationError:
                invalid += 1
    return TelemetryReadResult(events, malformed, invalid, unsupported, len(existing))


def _counts(values: list[str]) -> dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _threshold_result(
    *,
    metric_name: str,
    observed: float | int | None,
    threshold: float | int | None,
    comparison: str,
    observation_count: int,
    minimum_observations: int,
) -> dict[str, Any]:
    if threshold is None:
        return {
            "comparison": comparison,
            "configured_threshold": None,
            "metric_name": metric_name,
            "minimum_required_observations": minimum_observations,
            "observation_count": observation_count,
            "observed_value": observed,
            "result": "disabled",
            "sufficient_data": False,
        }
    sufficient = observation_count >= minimum_observations and observed is not None
    if not sufficient:
        return {
            "comparison": comparison,
            "configured_threshold": threshold,
            "metric_name": metric_name,
            "minimum_required_observations": minimum_observations,
            "observation_count": observation_count,
            "observed_value": observed,
            "result": "insufficient_data",
            "sufficient_data": False,
        }
    passed = None if observed is None else (
        observed <= threshold if comparison == "maximum" else observed >= threshold
    )
    return {
        "comparison": comparison,
        "configured_threshold": threshold,
        "metric_name": metric_name,
        "minimum_required_observations": minimum_observations,
        "observation_count": observation_count,
        "observed_value": observed,
        "result": "pass" if passed else "fail",
        "sufficient_data": True,
    }


def analyze_events(
    read_result: TelemetryReadResult,
    thresholds: HealthThresholds = HealthThresholds(),
) -> dict[str, Any]:
    thresholds.validate()
    events = read_result.events
    successes = [event for event in events if event.request_status == "success"]
    failures = [event for event in events if event.request_status == "failed"]
    latencies = [event.latency_ms for event in events]
    cache_values = [event.cache_hit for event in events if event.cache_hit is not None]
    top1_scores = [event.top1_score for event in events if event.top1_score is not None]
    margins = [
        event.top1_top2_margin
        for event in events
        if event.top1_top2_margin is not None
    ]
    labeled = [event for event in events if event.ground_truth_relevance_available]
    recall1 = [event.recall_at_1 for event in labeled if event.recall_at_1 is not None]
    recall5 = [event.recall_at_5 for event in labeled if event.recall_at_5 is not None]
    recall10 = [event.recall_at_10 for event in labeled if event.recall_at_10 is not None]
    reciprocal_ranks = [
        event.reciprocal_rank for event in labeled if event.reciprocal_rank is not None
    ]
    readiness_failures = sum(event.error_category == "readiness_failure" for event in events)
    readiness_observations = sum(event.endpoint == "readiness" for event in events)
    error_rate = len(failures) / len(events) if events else None
    latency = {
        "observation_count": len(latencies),
        "mean_ms": _mean(latencies),
        "median_ms": _median(latencies),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "maximum_ms": max(latencies) if latencies else None,
    }
    quality = {
        "labeled_query_count": len(labeled),
        "recall_at_1_observation_count": len(recall1),
        "recall_at_5_observation_count": len(recall5),
        "recall_at_10_observation_count": len(recall10),
        "reciprocal_rank_observation_count": len(reciprocal_ranks),
        "mean_recall_at_1": _mean(recall1),
        "mean_recall_at_5": _mean(recall5),
        "mean_recall_at_10": _mean(recall10),
        "mean_reciprocal_rank": _mean(reciprocal_ranks),
    }
    checks = {
        "maximum_error_rate": _threshold_result(
            metric_name="overall_error_rate",
            observed=error_rate,
            threshold=thresholds.maximum_error_rate,
            comparison="maximum",
            observation_count=len(events),
            minimum_observations=thresholds.min_error_rate_observations,
        ),
        "maximum_readiness_failures": _threshold_result(
            metric_name="readiness_failure_count",
            observed=readiness_failures,
            threshold=thresholds.maximum_readiness_failures,
            comparison="maximum",
            observation_count=readiness_observations,
            minimum_observations=thresholds.min_readiness_observations,
        ),
        "maximum_p95_latency_ms": _threshold_result(
            metric_name="p95_latency_ms",
            observed=latency["p95_ms"],
            threshold=thresholds.maximum_p95_latency_ms,
            comparison="maximum",
            observation_count=len(latencies),
            minimum_observations=thresholds.min_latency_observations,
        ),
        "minimum_labeled_recall_at_10": _threshold_result(
            metric_name="mean_labeled_recall_at_10",
            observed=quality["mean_recall_at_10"],
            threshold=thresholds.minimum_labeled_recall_at_10,
            comparison="minimum",
            observation_count=len(labeled),
            minimum_observations=thresholds.min_labeled_recall_observations,
        ),
        "minimum_labeled_mrr": _threshold_result(
            metric_name="mean_labeled_reciprocal_rank",
            observed=quality["mean_reciprocal_rank"],
            threshold=thresholds.minimum_labeled_mrr,
            comparison="minimum",
            observation_count=len(labeled),
            minimum_observations=thresholds.min_labeled_mrr_observations,
        ),
    }
    enabled_checks = [value for value in checks.values() if value["result"] != "disabled"]
    if any(value["result"] == "fail" for value in enabled_checks):
        health = "warning"
    elif enabled_checks and all(value["result"] == "pass" for value in enabled_checks):
        health = "healthy"
    else:
        health = "insufficient_data"
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "source_file_count": read_result.file_count,
        "traffic": {
            "total_event_count": len(events),
            "request_count_by_endpoint": _counts([event.endpoint for event in events]),
            "request_count_by_backend": _counts([event.backend for event in events]),
            "request_count_by_retrieval_direction": _counts(
                [event.retrieval_direction for event in events]
            ),
            "successful_request_count": len(successes),
            "failed_request_count": len(failures),
        },
        "reliability": {
            "overall_error_rate": error_rate,
            "http_status_counts": _counts([str(event.http_status_code) for event in events]),
            "error_category_counts": _counts(
                [event.error_category for event in failures if event.error_category is not None]
            ),
            "readiness_failure_count": readiness_failures,
            "telemetry_parsing_error_count": read_result.parsing_error_count,
            "malformed_json_line_count": read_result.malformed_json_line_count,
            "invalid_event_record_count": read_result.invalid_event_record_count,
            "unsupported_schema_record_count": read_result.unsupported_schema_record_count,
        },
        "latency": latency,
        "cache": {
            "observation_count": len(cache_values),
            "cache_hit_count": sum(cache_values),
            "cache_miss_count": len(cache_values) - sum(cache_values),
            "cache_hit_rate": sum(cache_values) / len(cache_values) if cache_values else None,
        },
        "score_confidence": {
            "observation_count": len(top1_scores),
            "mean_top1_score": _mean(top1_scores),
            "median_top1_score": _median(top1_scores),
            "margin_observation_count": len(margins),
            "mean_top1_top2_margin": _mean(margins),
            "median_top1_top2_margin": _median(margins),
            "non_positive_margin_rate": (
                sum(value <= 0 for value in margins) / len(margins) if margins else None
            ),
        },
        "known_label_quality": quality,
        "health": {
            "decision": health,
            "thresholds": asdict(thresholds),
            "threshold_results": checks,
        },
    }


def _display(value: Any, digits: int = 4) -> str:
    return "unavailable" if value is None else (
        f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    )


def render_monitoring_report(summary: dict[str, Any]) -> str:
    traffic = summary["traffic"]
    reliability = summary["reliability"]
    latency = summary["latency"]
    cache = summary["cache"]
    score = summary["score_confidence"]
    quality = summary["known_label_quality"]
    health = summary["health"]
    return "\n".join(
        [
            "# Retrieval Monitoring Report",
            "",
            "Run state: **success**",
            "",
            f"- Telemetry schema version: {summary['schema_version']}",
            f"- Event count: {traffic['total_event_count']}",
            f"- Source files: {summary['source_file_count']}",
            "",
            "## Traffic and reliability",
            "",
            f"- Requests by endpoint: `{json.dumps(traffic['request_count_by_endpoint'], sort_keys=True)}`",
            f"- Requests by backend: `{json.dumps(traffic['request_count_by_backend'], sort_keys=True)}`",
            f"- Requests by direction: `{json.dumps(traffic['request_count_by_retrieval_direction'], sort_keys=True)}`",
            f"- Successful / failed: {traffic['successful_request_count']} / {traffic['failed_request_count']}",
            f"- Overall error rate: {_display(reliability['overall_error_rate'])}",
            f"- HTTP statuses: `{json.dumps(reliability['http_status_counts'], sort_keys=True)}`",
            f"- Error categories: `{json.dumps(reliability['error_category_counts'], sort_keys=True)}`",
            f"- Readiness failures: {reliability['readiness_failure_count']}",
            f"- Telemetry parsing errors: {reliability['telemetry_parsing_error_count']}",
            "",
            "## Latency and cache",
            "",
            f"- Latency observations: {latency['observation_count']}",
            f"- Mean / median / p95 / maximum milliseconds: {_display(latency['mean_ms'])} / "
            f"{_display(latency['median_ms'])} / {_display(latency['p95_ms'])} / "
            f"{_display(latency['maximum_ms'])}",
            f"- Cache hits / misses / hit rate: {cache['cache_hit_count']} / "
            f"{cache['cache_miss_count']} / {_display(cache['cache_hit_rate'])}",
            "",
            "## Score confidence",
            "",
            f"- Top-1 score observations: {score['observation_count']}",
            f"- Mean / median top-1 score: {_display(score['mean_top1_score'])} / "
            f"{_display(score['median_top1_score'])}",
            f"- Mean / median top-1-to-top-2 margin: "
            f"{_display(score['mean_top1_top2_margin'])} / "
            f"{_display(score['median_top1_top2_margin'])}",
            f"- Non-positive margin rate: {_display(score['non_positive_margin_rate'])}",
            "",
            "## Known-label quality",
            "",
            f"- Labeled cached-ID queries: {quality['labeled_query_count']}",
            f"- Mean Recall@1 / @5 / @10: {_display(quality['mean_recall_at_1'])} / "
            f"{_display(quality['mean_recall_at_5'])} / {_display(quality['mean_recall_at_10'])}",
            f"- Mean reciprocal rank: {_display(quality['mean_reciprocal_rank'])}",
            "",
            "Known-label quality applies only to cached-ID queries. Arbitrary text and uploaded",
            "image queries are explicitly unlabeled; no relevance judgments are invented.",
            "",
            "## Health decision",
            "",
            f"Decision: **{health['decision']}**",
            "",
            "| Check | Observations | Minimum required | Observed value | Threshold | Sufficiency | Result |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            *(
                f"| {name} | {value['observation_count']} | "
                f"{value['minimum_required_observations']} | "
                f"{_display(value['observed_value'])} | "
                f"{_display(value['configured_threshold'])} | "
                f"{'sufficient' if value['sufficient_data'] else 'insufficient'} | "
                f"{value['result']} |"
                for name, value in health["threshold_results"].items()
            ),
            "",
            *(
                [
                    "No production-health conclusion can be drawn from this undersized window.",
                    "The smoke window successfully validates telemetry collection and analysis only.",
                    "",
                ]
                if health["decision"] == "insufficient_data"
                else []
            ),
            "## Privacy and operational limitations",
            "",
            "Telemetry excludes raw text, caption text, image bytes, uploaded filenames, absolute",
            "paths, headers, model-cache paths, stack traces, and ranked payloads. Query identity is",
            "represented only by SHA-256. Reports exclude timestamps, event IDs, query hashes, and",
            "raw JSONL records. Monitoring is process-local and single-window; local file rotation",
            "is not a durable or multi-instance observability system. Telemetry write failures do",
            "not fail retrieval and are visible only in process-local metrics. Retrieval rankings,",
            "models, indexes, and API response contracts are unchanged.",
            "",
        ]
    )


def render_monitoring_decision(summary: dict[str, Any]) -> str:
    health = summary["health"]
    return "\n".join(
        [
            "# Retrieval Monitoring Decision",
            "",
            f"Decision: **{health['decision']}**",
            "",
            *(
                f"- {name}: {value['result']} "
                f"({value['observation_count']}/{value['minimum_required_observations']} observations)"
                for name, value in health["threshold_results"].items()
            ),
            "",
            *(
                [
                    "No production-health conclusion can be drawn. The window validates the",
                    "telemetry pipeline, but it does not meet all enabled minimum sample sizes.",
                ]
                if health["decision"] == "insufficient_data"
                else [
                    "This is a bounded single-window service-health decision. Known-label",
                    "thresholds use cached-ID queries only; arbitrary queries have no inferred",
                    "relevance labels.",
                ]
            ),
            "",
        ]
    )


def write_monitoring_outputs(
    summary: dict[str, Any], report_path: Path, metrics_path: Path, decision_path: Path
) -> None:
    for path in (report_path, metrics_path, decision_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_monitoring_report(summary), encoding="utf-8")
    metrics_path.write_text(
        json.dumps({"run_state": "success", **summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision_path.write_text(render_monitoring_decision(summary), encoding="utf-8")


def write_monitoring_failure(
    state: str,
    detail: str,
    report_path: Path,
    metrics_path: Path,
    decision_path: Path,
) -> None:
    if state not in MONITORING_RUN_STATES:
        raise ValueError(f"unsupported monitoring run state: {state}")
    for path in (report_path, metrics_path, decision_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Retrieval Monitoring Report\n\nRun state: **{state}**\n\nDetail: {detail}\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": detail, "run_state": state}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision_path.write_text(
        "# Retrieval Monitoring Decision\n\nDecision: **insufficient_data**\n\n"
        f"Run state: **{state}**. {detail}\n",
        encoding="utf-8",
    )
