"""Deterministic hashed text encoder placeholder."""

from dataclasses import dataclass

from .embedding_backends import normalized_hashed_vector
from .text_baseline import tokenize


@dataclass(frozen=True)
class DeterministicTextEncoder:
    """Encode tokens into a fixed-size shared hashing space."""

    dimension: int = 64
    backend_name: str = "deterministic-hashed-text"
    backend_version: str = "1"

    def encode_text(self, text: str) -> list[float]:
        return normalized_hashed_vector(tokenize(text), self.dimension)
