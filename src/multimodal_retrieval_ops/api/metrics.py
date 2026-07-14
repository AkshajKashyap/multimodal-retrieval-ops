"""Bounded process-local observability for retrieval requests."""

from dataclasses import dataclass, field
import math
import statistics
from threading import Lock

MAX_LATENCY_OBSERVATIONS = 1000


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


@dataclass
class ServiceMetrics:
    """Thread-safe counters scoped to one application process."""

    total_requests: int = 0
    retrieval_requests_by_direction: dict[str, int] = field(
        default_factory=lambda: {"caption_to_image": 0, "image_to_caption": 0}
    )
    requests_by_backend: dict[str, int] = field(
        default_factory=lambda: {"flat": 0, "hnsw": 0}
    )
    error_count: int = 0
    unknown_query_id_count: int = 0
    invalid_top_k_count: int = 0
    arbitrary_text_request_count: int = 0
    text_encoder_invocation_count: int = 0
    text_query_cache_hits: int = 0
    text_query_cache_misses: int = 0
    text_inference_errors: int = 0
    arbitrary_image_request_count: int = 0
    image_encoder_invocation_count: int = 0
    image_query_cache_hits: int = 0
    image_query_cache_misses: int = 0
    image_validation_errors: int = 0
    image_inference_errors: int = 0
    uploaded_byte_total: int = 0
    telemetry_enabled: bool = False
    telemetry_event_count: int = 0
    telemetry_write_failure_count: int = 0
    telemetry_rotation_count: int = 0
    _latencies: list[float] = field(default_factory=list)
    _text_latencies: list[float] = field(default_factory=list)
    _image_latencies: list[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_request(self, is_error: bool) -> None:
        with self._lock:
            self.total_requests += 1
            if is_error:
                self.error_count += 1

    def record_retrieval(self, direction: str, backend: str, latency_seconds: float) -> None:
        with self._lock:
            self.retrieval_requests_by_direction[direction] += 1
            self.requests_by_backend[backend] += 1
            self._latencies.append(latency_seconds)
            if len(self._latencies) > MAX_LATENCY_OBSERVATIONS:
                del self._latencies[: len(self._latencies) - MAX_LATENCY_OBSERVATIONS]

    def record_unknown_query(self) -> None:
        with self._lock:
            self.unknown_query_id_count += 1

    def record_invalid_top_k(self) -> None:
        with self._lock:
            self.invalid_top_k_count += 1

    def record_text_request(self) -> None:
        with self._lock:
            self.arbitrary_text_request_count += 1

    def record_text_encoder_invocation(self) -> None:
        with self._lock:
            self.text_encoder_invocation_count += 1

    def record_text_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.text_query_cache_hits += 1
            else:
                self.text_query_cache_misses += 1

    def record_text_error(self) -> None:
        with self._lock:
            self.text_inference_errors += 1

    def record_text_latency(self, latency_seconds: float, backend: str) -> None:
        with self._lock:
            self.requests_by_backend[backend] += 1
            self._text_latencies.append(latency_seconds)
            if len(self._text_latencies) > MAX_LATENCY_OBSERVATIONS:
                del self._text_latencies[
                    : len(self._text_latencies) - MAX_LATENCY_OBSERVATIONS
                ]

    def record_image_request(self) -> None:
        with self._lock:
            self.arbitrary_image_request_count += 1

    def record_image_encoder_invocation(self) -> None:
        with self._lock:
            self.image_encoder_invocation_count += 1

    def record_image_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.image_query_cache_hits += 1
            else:
                self.image_query_cache_misses += 1

    def record_image_validation_error(self) -> None:
        with self._lock:
            self.image_validation_errors += 1

    def record_image_inference_error(self) -> None:
        with self._lock:
            self.image_inference_errors += 1

    def record_uploaded_bytes(self, byte_count: int) -> None:
        with self._lock:
            self.uploaded_byte_total += byte_count

    def record_image_latency(self, latency_seconds: float, backend: str) -> None:
        with self._lock:
            self.requests_by_backend[backend] += 1
            self._image_latencies.append(latency_seconds)
            if len(self._image_latencies) > MAX_LATENCY_OBSERVATIONS:
                del self._image_latencies[
                    : len(self._image_latencies) - MAX_LATENCY_OBSERVATIONS
                ]

    def configure_telemetry(self, enabled: bool) -> None:
        with self._lock:
            self.telemetry_enabled = enabled

    def record_telemetry_event(self) -> None:
        with self._lock:
            self.telemetry_event_count += 1

    def record_telemetry_failure(self) -> None:
        with self._lock:
            self.telemetry_write_failure_count += 1

    def record_telemetry_rotation(self) -> None:
        with self._lock:
            self.telemetry_rotation_count += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latencies = list(self._latencies)
            text_latencies = list(self._text_latencies)
            image_latencies = list(self._image_latencies)
            return {
                "total_requests": self.total_requests,
                "retrieval_requests_by_direction": dict(self.retrieval_requests_by_direction),
                "requests_by_backend": dict(self.requests_by_backend),
                "error_count": self.error_count,
                "unknown_query_id_count": self.unknown_query_id_count,
                "invalid_top_k_count": self.invalid_top_k_count,
                "latency_observation_count": len(latencies),
                "mean_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
                "p50_latency_seconds": _percentile(latencies, 0.50),
                "p95_latency_seconds": _percentile(latencies, 0.95),
                "arbitrary_text_request_count": self.arbitrary_text_request_count,
                "text_encoder_invocation_count": self.text_encoder_invocation_count,
                "text_query_cache_hits": self.text_query_cache_hits,
                "text_query_cache_misses": self.text_query_cache_misses,
                "text_inference_errors": self.text_inference_errors,
                "text_inference_latency_count": len(text_latencies),
                "text_inference_latency_mean_seconds": (
                    statistics.mean(text_latencies) if text_latencies else 0.0
                ),
                "text_inference_latency_p50_seconds": _percentile(text_latencies, 0.50),
                "text_inference_latency_p95_seconds": _percentile(text_latencies, 0.95),
                "arbitrary_image_request_count": self.arbitrary_image_request_count,
                "image_encoder_invocation_count": self.image_encoder_invocation_count,
                "image_query_cache_hits": self.image_query_cache_hits,
                "image_query_cache_misses": self.image_query_cache_misses,
                "image_validation_errors": self.image_validation_errors,
                "image_inference_errors": self.image_inference_errors,
                "uploaded_byte_total": self.uploaded_byte_total,
                "image_inference_latency_count": len(image_latencies),
                "image_inference_latency_mean_seconds": (
                    statistics.mean(image_latencies) if image_latencies else 0.0
                ),
                "image_inference_latency_p50_seconds": _percentile(image_latencies, 0.50),
                "image_inference_latency_p95_seconds": _percentile(image_latencies, 0.95),
                "telemetry_enabled": self.telemetry_enabled,
                "telemetry_event_count": self.telemetry_event_count,
                "telemetry_write_failure_count": self.telemetry_write_failure_count,
                "telemetry_rotation_count": self.telemetry_rotation_count,
            }
