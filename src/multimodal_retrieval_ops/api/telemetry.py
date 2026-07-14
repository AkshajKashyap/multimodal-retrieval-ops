"""Privacy-safe, bounded local JSONL telemetry for retrieval requests."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable
import uuid

TELEMETRY_SCHEMA_VERSION = 1


class TelemetryValidationError(ValueError):
    """An event or sink configuration is invalid."""


@dataclass(frozen=True)
class RetrievalEvent:
    schema_version: int
    event_id: str
    event_timestamp: str
    endpoint: str
    retrieval_direction: str
    backend: str
    request_status: str
    http_status_code: int
    error_category: str | None
    top_k: int | None
    result_count: int | None
    latency_ms: float
    cache_hit: bool | None
    embedding_dimension: int | None
    model_identity_fingerprint: str | None
    artifact_fingerprint: str | None
    candidate_count: int | None
    top1_score: float | None
    top1_top2_margin: float | None
    minimum_returned_score: float | None
    maximum_returned_score: float | None
    ground_truth_relevance_available: bool
    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    reciprocal_rank: float | None
    safe_query_hash: str | None


def safe_sha256(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_identity_fingerprint(
    backend_name: str, model_name: str, model_revision: str, dimension: int
) -> str:
    return safe_sha256(
        f"{backend_name}\n{model_name}\n{model_revision}\n{dimension}"
    )


def score_summary(scores: list[float]) -> dict[str, float | None]:
    if not scores:
        return {
            "top1_score": None,
            "top1_top2_margin": None,
            "minimum_returned_score": None,
            "maximum_returned_score": None,
        }
    if not all(math.isfinite(score) for score in scores):
        raise TelemetryValidationError("returned scores must be finite")
    return {
        "top1_score": scores[0],
        "top1_top2_margin": scores[0] - scores[1] if len(scores) >= 2 else None,
        "minimum_returned_score": min(scores),
        "maximum_returned_score": max(scores),
    }


def cached_relevance_metrics(
    relevant_flags: list[bool], requested_top_k: int
) -> dict[str, float | bool | None]:
    rank = next((index for index, relevant in enumerate(relevant_flags, 1) if relevant), None)

    def recall_at(k: int) -> float | None:
        if rank is not None:
            return float(rank <= k)
        return 0.0 if requested_top_k >= k else None

    return {
        "ground_truth_relevance_available": True,
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "reciprocal_rank": 1.0 / rank if rank is not None else None,
    }


def unlabeled_relevance_metrics() -> dict[str, float | bool | None]:
    return {
        "ground_truth_relevance_available": False,
        "recall_at_1": None,
        "recall_at_5": None,
        "recall_at_10": None,
        "reciprocal_rank": None,
    }


def make_retrieval_event(**values: Any) -> RetrievalEvent:
    defaults: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_direction": "operational",
        "backend": "unavailable",
        "request_status": "success",
        "http_status_code": 200,
        "error_category": None,
        "top_k": None,
        "result_count": None,
        "latency_ms": 0.0,
        "cache_hit": None,
        "embedding_dimension": None,
        "model_identity_fingerprint": None,
        "artifact_fingerprint": None,
        "candidate_count": None,
        "top1_score": None,
        "top1_top2_margin": None,
        "minimum_returned_score": None,
        "maximum_returned_score": None,
        **unlabeled_relevance_metrics(),
        "safe_query_hash": None,
    }
    defaults.update(values)
    event = RetrievalEvent(**defaults)
    validate_event(event)
    return event


def validate_event(event: RetrievalEvent) -> None:
    if event.schema_version != TELEMETRY_SCHEMA_VERSION:
        raise TelemetryValidationError(
            f"unsupported telemetry schema version: {event.schema_version}"
        )
    if not event.event_id or not event.event_timestamp or not event.endpoint:
        raise TelemetryValidationError("event identity, timestamp, and endpoint are required")
    if event.request_status not in {"success", "failed"}:
        raise TelemetryValidationError("request_status must be success or failed")
    if not 100 <= event.http_status_code <= 599:
        raise TelemetryValidationError("HTTP status code is invalid")
    if event.top_k is not None and event.top_k <= 0:
        raise TelemetryValidationError("top_k must be positive when present")
    if event.result_count is not None and event.result_count < 0:
        raise TelemetryValidationError("result_count must be non-negative")
    numeric = (
        event.latency_ms,
        event.top1_score,
        event.top1_top2_margin,
        event.minimum_returned_score,
        event.maximum_returned_score,
        event.recall_at_1,
        event.recall_at_5,
        event.recall_at_10,
        event.reciprocal_rank,
    )
    if any(value is not None and not math.isfinite(value) for value in numeric):
        raise TelemetryValidationError("event contains a non-finite numeric value")
    if event.latency_ms < 0:
        raise TelemetryValidationError("latency must be non-negative")
    if event.safe_query_hash is not None and (
        len(event.safe_query_hash) != 64
        or any(character not in "0123456789abcdef" for character in event.safe_query_hash)
    ):
        raise TelemetryValidationError("safe_query_hash must be lowercase SHA-256")


def event_from_mapping(data: dict[str, Any]) -> RetrievalEvent:
    try:
        event = RetrievalEvent(**data)
    except (TypeError, ValueError) as error:
        raise TelemetryValidationError("event record does not match the schema") from error
    validate_event(event)
    return event


class JsonlTelemetrySink:
    """Thread-safe JSONL append sink with bounded local rotation."""

    def __init__(
        self,
        path: Path,
        maximum_bytes: int,
        backup_count: int,
        *,
        flush_each_event: bool = True,
        on_event: Callable[[], None] | None = None,
        on_failure: Callable[[], None] | None = None,
        on_rotation: Callable[[], None] | None = None,
    ) -> None:
        if maximum_bytes <= 0:
            raise TelemetryValidationError("telemetry maximum bytes must be positive")
        if backup_count < 0:
            raise TelemetryValidationError("telemetry backup count must be non-negative")
        if path.name in {"", ".", ".."} or path.suffix.lower() != ".jsonl":
            raise TelemetryValidationError("telemetry path must name a .jsonl file")
        self.path = path
        self.maximum_bytes = maximum_bytes
        self.backup_count = backup_count
        self.flush_each_event = flush_each_event
        self.on_event = on_event
        self.on_failure = on_failure
        self.on_rotation = on_rotation
        self._lock = Lock()

    def _rotate(self) -> None:
        if self.backup_count == 0:
            self.path.write_text("", encoding="utf-8")
        else:
            oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
            if oldest.exists():
                oldest.unlink()
            for number in range(self.backup_count - 1, 0, -1):
                source = self.path.with_name(f"{self.path.name}.{number}")
                if source.exists():
                    os.replace(source, self.path.with_name(f"{self.path.name}.{number + 1}"))
            if self.path.exists():
                os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        if self.on_rotation is not None:
            self.on_rotation()

    def write(self, event: RetrievalEvent) -> bool:
        try:
            validate_event(event)
            encoded = (json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            if len(encoded) > self.maximum_bytes:
                raise TelemetryValidationError("one telemetry event exceeds maximum file size")
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                current_size = self.path.stat().st_size if self.path.exists() else 0
                if current_size and current_size + len(encoded) > self.maximum_bytes:
                    self._rotate()
                with self.path.open("ab") as destination:
                    destination.write(encoded)
                    if self.flush_each_event:
                        destination.flush()
            if self.on_event is not None:
                self.on_event()
            return True
        except Exception:
            if self.on_failure is not None:
                self.on_failure()
            return False
