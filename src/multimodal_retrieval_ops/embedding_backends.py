"""Interfaces and shared utilities for pluggable embedding backends."""

from collections.abc import Iterable
import hashlib
import math
from typing import Protocol


class TextEncoder(Protocol):
    """Contract implemented by text embedding backends."""

    backend_name: str
    backend_version: str
    dimension: int

    def encode_text(self, text: str) -> list[float]: ...


class ImageEncoder(Protocol):
    """Contract implemented by image embedding backends."""

    backend_name: str
    backend_version: str
    dimension: int

    def encode_image(self, image_path: str) -> list[float]: ...


def normalized_hashed_vector(features: Iterable[str], dimension: int) -> list[float]:
    """Hash string features into a deterministic signed, normalized vector."""
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    vector = [0.0] * dimension
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
