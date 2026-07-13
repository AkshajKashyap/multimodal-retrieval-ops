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
    _latencies: list[float] = field(default_factory=list)
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

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latencies = list(self._latencies)
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
            }
