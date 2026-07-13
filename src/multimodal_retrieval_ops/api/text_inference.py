"""Bounded arbitrary-text inference primitives for the retrieval service."""

from collections import OrderedDict
import math
from typing import Protocol

from ..clip_backend import ClipEmbeddingBackend
from .settings import ServiceSettings


class TextEncoder(Protocol):
    model_name: str
    model_revision: str | None
    backend_name: str
    backend_version: str
    dimension: int

    def ensure_loaded(self) -> None: ...

    def encode_text(self, text: str) -> list[float]: ...


def create_default_text_encoder(settings: ServiceSettings) -> TextEncoder:
    """Create the lazy Hugging Face encoder without downloading model files."""
    return ClipEmbeddingBackend(
        model_name=settings.text_model_name,
        model_revision=settings.text_model_revision,
        device=settings.text_device,
        batch_size=1,
        allow_download=not settings.local_files_only,
    )


def normalized_query(query: str) -> tuple[str, str]:
    display = " ".join(query.split())
    return display, display.casefold()


def normalize_vector(vector: list[float], expected_dimension: int) -> list[float]:
    if len(vector) != expected_dimension:
        raise ValueError(
            f"text embedding dimension {len(vector)} does not match index dimension "
            f"{expected_dimension}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("text embedding contains non-finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("text embedding has zero or invalid norm")
    normalized = [value / norm for value in vector]
    normalized_norm = math.sqrt(sum(value * value for value in normalized))
    if not math.isclose(normalized_norm, 1.0, abs_tol=1e-6):
        raise ValueError("text embedding could not be L2-normalized")
    return normalized


class QueryEmbeddingCache:
    """Process-local deterministic LRU cache with a hard capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("text query cache size must be positive")
        self.capacity = capacity
        self._values: OrderedDict[str, list[float]] = OrderedDict()

    def get(self, key: str) -> list[float] | None:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: str, value: list[float]) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)
