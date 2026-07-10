"""Deterministic train-vocabulary bag-of-words embeddings."""

from collections import Counter
import math
import re

from .manifest import ManifestItem

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text with a stable, locale-independent rule."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(items: list[ManifestItem]) -> list[str]:
    """Build a sorted vocabulary exclusively from training captions."""
    return sorted(
        {
            token
            for item in items
            if item.split == "train"
            for token in tokenize(item.caption)
        }
    )


def encode_text(text: str, vocabulary: list[str]) -> list[float]:
    """Encode text as an L2-normalized term-frequency vector."""
    counts = Counter(tokenize(text))
    vector = [float(counts[token]) for token in vocabulary]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
