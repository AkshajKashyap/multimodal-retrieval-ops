from dataclasses import asdict, replace
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from multimodal_retrieval_ops.api.app import create_app
from multimodal_retrieval_ops.api.settings import ServiceSettings
from multimodal_retrieval_ops.api.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    JsonlTelemetrySink,
    cached_relevance_metrics,
    make_retrieval_event,
    safe_sha256,
    unlabeled_relevance_metrics,
)
from multimodal_retrieval_ops.api.telemetry_smoke import run_telemetry_smoke
from multimodal_retrieval_ops.retrieval_monitoring import (
    HealthThresholds,
    TelemetryReadResult,
    analyze_events,
    read_telemetry,
)
from test_milestone_eight_a import FakeTextEncoder, enabled_settings
from test_milestone_eight_b import FakeImageEncoder, factory, image_bytes, upload
from test_milestone_one import run_cli
from test_milestone_seven_c import _settings


def telemetry_settings(tmp_path: Path, **changes: object) -> ServiceSettings:
    return replace(
        _settings(tmp_path),
        telemetry_enabled=True,
        telemetry_path=tmp_path / "telemetry" / "events.jsonl",
        telemetry_max_bytes=50_000,
        telemetry_backup_count=2,
        **changes,
    )


def event(**changes: object):
    values = {
        "endpoint": "cached_caption_to_image",
        "retrieval_direction": "caption_to_image",
        "backend": "flat",
        "request_status": "success",
        "http_status_code": 200,
        "latency_ms": 10.0,
        **changes,
    }
    return make_retrieval_event(**values)


def test_telemetry_disabled_by_default_and_creates_nothing(tmp_path: Path) -> None:
    assert ServiceSettings().telemetry_enabled is False
    output = tmp_path / "missing" / "events.jsonl"
    settings = replace(_settings(tmp_path), telemetry_path=output)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        metrics = client.get("/metrics").json()
    assert not output.parent.exists()
    assert metrics["telemetry_enabled"] is False
    assert metrics["telemetry_event_count"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"telemetry_max_bytes": 0},
        {"telemetry_backup_count": -1},
        {"telemetry_path": Path("events.txt")},
    ],
)
def test_invalid_telemetry_settings_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="telemetry"):
        replace(ServiceSettings(), **changes).validate()


def test_valid_event_serialization_and_deterministic_hash(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    sink = JsonlTelemetrySink(output, 10_000, 1)
    query_hash = safe_sha256("normalized query")
    assert query_hash == safe_sha256("normalized query")
    assert query_hash != safe_sha256("other query")
    assert sink.write(event(safe_query_hash=query_hash))
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == TELEMETRY_SCHEMA_VERSION
    assert record["safe_query_hash"] == query_hash
    assert set(record) == set(asdict(event()))


def test_cached_relevance_and_unlabeled_contracts() -> None:
    labeled = cached_relevance_metrics([False, True, False], 10)
    assert labeled == {
        "ground_truth_relevance_available": True,
        "recall_at_1": 0.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "reciprocal_rank": 0.5,
    }
    assert cached_relevance_metrics([False], 1)["recall_at_5"] is None
    assert unlabeled_relevance_metrics() == {
        "ground_truth_relevance_available": False,
        "recall_at_1": None,
        "recall_at_5": None,
        "recall_at_10": None,
        "reciprocal_rank": None,
    }


def test_cached_service_events_exclude_ids_captions_and_absolute_paths(tmp_path: Path) -> None:
    settings = telemetry_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        images = client.post(
            "/retrieve/images", json={"caption_id": "caption-a1", "top_k": 2}
        )
        captions = client.post(
            "/retrieve/captions", json={"image_id": "image-a", "top_k": 3}
        )
    assert images.status_code == captions.status_code == 200
    raw = settings.telemetry_path.read_text(encoding="utf-8")
    assert "caption-a1" not in raw
    assert "image-a" not in raw
    assert '"alpha"' not in raw
    assert str(tmp_path) not in raw
    records = read_telemetry(settings.telemetry_path).events
    assert len(records) == 2
    assert all(record.ground_truth_relevance_available for record in records)
    assert records[0].recall_at_1 == 1.0


def test_arbitrary_text_event_contains_hash_not_raw_text(tmp_path: Path) -> None:
    base = enabled_settings(tmp_path)
    settings = replace(
        base,
        telemetry_enabled=True,
        telemetry_path=tmp_path / "telemetry" / "text.jsonl",
    )
    with TestClient(
        create_app(settings, text_encoder_factory=lambda _: FakeTextEncoder())
    ) as client:
        response = client.post(
            "/search/text", json={"query": "  Secret   Caption Text ", "top_k": 2}
        )
    assert response.status_code == 200
    raw = settings.telemetry_path.read_text(encoding="utf-8")
    assert "Secret" not in raw
    assert "Caption Text" not in raw
    record = read_telemetry(settings.telemetry_path).events[0]
    assert record.safe_query_hash == safe_sha256("secret caption text")
    assert record.ground_truth_relevance_available is False
    assert record.cache_hit is False


def test_uploaded_image_event_excludes_filename_and_bytes(tmp_path: Path) -> None:
    base = replace(
        _settings(tmp_path),
        enable_image_inference=True,
        image_model_name="synthetic-model",
        image_model_revision="test",
        maximum_upload_bytes=4096,
        maximum_pixel_count=100,
        image_query_cache_size=2,
    )
    settings = replace(
        base,
        telemetry_enabled=True,
        telemetry_path=tmp_path / "telemetry" / "image.jsonl",
    )
    payload = image_bytes()
    with TestClient(
        create_app(settings, image_encoder_factory=factory(FakeImageEncoder()))
    ) as client:
        response = upload(client, payload, filename="private-upload-name.png")
    assert response.status_code == 200
    raw = settings.telemetry_path.read_text(encoding="utf-8")
    assert "private-upload-name" not in raw
    assert payload.hex() not in raw
    record = read_telemetry(settings.telemetry_path).events[0]
    assert record.safe_query_hash == safe_sha256(payload)
    assert record.ground_truth_relevance_available is False


def test_rotation_and_backup_count_are_bounded(tmp_path: Path) -> None:
    rotations = 0

    def rotated() -> None:
        nonlocal rotations
        rotations += 1

    output = tmp_path / "events.jsonl"
    sample = event()
    encoded_size = len(json.dumps(asdict(sample), sort_keys=True)) + 1
    sink = JsonlTelemetrySink(
        output, encoded_size + 20, 2, on_rotation=rotated
    )
    for _ in range(6):
        assert sink.write(event())
    backups = sorted(tmp_path.glob("events.jsonl.*"))
    assert [path.name for path in backups] == ["events.jsonl.1", "events.jsonl.2"]
    assert rotations >= 4


def test_telemetry_failure_does_not_fail_retrieval_and_is_counted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    settings = replace(
        settings,
        telemetry_enabled=True,
        telemetry_path=blocked / "events.jsonl",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/retrieve/images", json={"caption_id": "caption-a1", "top_k": 1}
        )
        metrics = client.get("/metrics").json()
    assert response.status_code == 200
    assert metrics["telemetry_event_count"] == 0
    assert metrics["telemetry_write_failure_count"] == 1


def test_invalid_requests_have_typed_safe_events(tmp_path: Path) -> None:
    settings = telemetry_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        unknown = client.post(
            "/retrieve/images", json={"caption_id": "private-missing-id", "top_k": 1}
        )
        invalid = client.post(
            "/retrieve/images", json={"caption_id": "caption-a1", "top_k": 0}
        )
    assert unknown.status_code == 404
    assert invalid.status_code == 422
    records = read_telemetry(settings.telemetry_path).events
    assert [record.error_category for record in records] == [
        "unknown_cached_id",
        "invalid_top_k",
    ]
    assert "private-missing-id" not in settings.telemetry_path.read_text(encoding="utf-8")


def test_malformed_unsupported_and_invalid_records_are_counted(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    valid = json.dumps(asdict(event()))
    unsupported = json.dumps({**asdict(event()), "schema_version": 999})
    invalid = json.dumps({"schema_version": TELEMETRY_SCHEMA_VERSION})
    output.write_text(f"\n{valid}\nnot-json\n{unsupported}\n{invalid}\n", encoding="utf-8")
    result = read_telemetry(output)
    assert len(result.events) == 1
    assert result.malformed_json_line_count == 1
    assert result.unsupported_schema_record_count == 1
    assert result.invalid_event_record_count == 1
    assert result.parsing_error_count == 3


def test_offline_aggregation_latency_cache_scores_and_quality() -> None:
    events = [
        event(
            endpoint="arbitrary_text_to_image",
            cache_hit=True,
            top1_score=0.8,
            top1_top2_margin=0.2,
            latency_ms=10.0,
        ),
        event(
            endpoint="arbitrary_text_to_image",
            cache_hit=False,
            top1_score=0.6,
            top1_top2_margin=0.0,
            latency_ms=20.0,
        ),
        event(
            ground_truth_relevance_available=True,
            recall_at_1=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            reciprocal_rank=1.0,
            latency_ms=30.0,
        ),
    ]
    result = TelemetryReadResult(events, 1, 0, 0, 1)
    summary = analyze_events(result)
    assert summary["traffic"]["request_count_by_endpoint"] == {
        "arbitrary_text_to_image": 2,
        "cached_caption_to_image": 1,
    }
    assert summary["traffic"]["request_count_by_backend"] == {"flat": 3}
    assert summary["latency"]["mean_ms"] == 20.0
    assert summary["latency"]["median_ms"] == 20.0
    assert summary["latency"]["p95_ms"] == 30.0
    assert summary["cache"]["cache_hit_rate"] == 0.5
    assert summary["score_confidence"]["mean_top1_score"] == pytest.approx(0.7)
    assert summary["score_confidence"]["non_positive_margin_rate"] == 0.5
    assert summary["known_label_quality"]["mean_recall_at_10"] == 1.0
    assert summary["known_label_quality"]["mean_reciprocal_rank"] == 1.0
    assert summary["reliability"]["telemetry_parsing_error_count"] == 1


def test_health_decisions_cover_healthy_warning_and_insufficient_data() -> None:
    controlled = HealthThresholds(
        min_error_rate_observations=1,
        min_latency_observations=1,
        min_labeled_recall_observations=1,
        min_labeled_mrr_observations=1,
        min_readiness_observations=1,
    )
    readiness = event(endpoint="readiness", retrieval_direction="operational")
    labeled = event(
        ground_truth_relevance_available=True,
        recall_at_1=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        reciprocal_rank=1.0,
    )
    healthy = analyze_events(
        TelemetryReadResult([readiness, labeled], 0, 0, 0, 1), controlled
    )
    assert healthy["health"]["decision"] == "healthy"
    failed = event(
        request_status="failed",
        http_status_code=500,
        error_category="internal_retrieval_failure",
    )
    warning = analyze_events(
        TelemetryReadResult([readiness, labeled, failed], 0, 0, 0, 1), controlled
    )
    assert warning["health"]["decision"] == "warning"
    unlabeled = event()
    insufficient = analyze_events(
        TelemetryReadResult([readiness, unlabeled], 0, 0, 0, 1), controlled
    )
    assert insufficient["health"]["decision"] == "insufficient_data"
    latency_warning = analyze_events(
        TelemetryReadResult([readiness, labeled], 0, 0, 0, 1),
        HealthThresholds(
            maximum_p95_latency_ms=1.0,
            min_error_rate_observations=1,
            min_latency_observations=1,
            min_labeled_recall_observations=1,
            min_labeled_mrr_observations=1,
            min_readiness_observations=1,
        ),
    )
    assert latency_warning["health"]["decision"] == "warning"


def test_in_process_telemetry_smoke_uses_cached_ids_only(tmp_path: Path) -> None:
    settings = telemetry_settings(tmp_path)
    result = run_telemetry_smoke(settings)
    assert result["event_count"] == 5
    assert result["health_status"] == "alive"
    assert result["ready_status"] == "ready"
    assert result["service_metrics"]["telemetry_event_count"] == 5
    records = read_telemetry(settings.telemetry_path).events
    assert {record.endpoint for record in records} == {
        "health",
        "readiness",
        "cached_caption_to_image",
        "cached_image_to_caption",
    }
    assert not any("arbitrary" in record.endpoint for record in records)


@pytest.mark.parametrize(
    "command",
    [
        "retrieval-telemetry-info",
        "analyze-retrieval-telemetry",
        "retrieval-telemetry-smoke",
    ],
)
def test_telemetry_cli_help_smoke(command: str) -> None:
    completed = run_cli(command, "--help")
    assert completed.returncode == 0, completed.stderr
