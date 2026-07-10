"""Seeded deterministic manifest splitting."""

from dataclasses import replace
import random

from .manifest import ManifestItem, VALID_SPLITS


def _split_counts(row_count: int, fractions: tuple[float, float, float]) -> list[int]:
    if row_count < len(VALID_SPLITS):
        raise ValueError("at least three rows are required to populate every split")
    if any(fraction <= 0 for fraction in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must be positive and sum to 1.0")
    remaining = row_count - len(VALID_SPLITS)
    raw = [remaining * fraction for fraction in fractions]
    counts = [1 + int(value) for value in raw]
    unassigned = row_count - sum(counts)
    order = sorted(range(3), key=lambda index: (-(raw[index] % 1), index))
    for index in order[:unassigned]:
        counts[index] += 1
    return counts


def assign_splits(
    items: list[ManifestItem],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> list[ManifestItem]:
    """Assign each unique item to exactly one split using a seeded shuffle."""
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("cannot split a manifest with duplicate item_id values")
    shuffled = sorted(items, key=lambda item: item.item_id)
    random.Random(seed).shuffle(shuffled)
    counts = _split_counts(len(shuffled), fractions)
    split_by_id: dict[str, str] = {}
    start = 0
    for split, count in zip(VALID_SPLITS, counts, strict=True):
        for item in shuffled[start : start + count]:
            split_by_id[item.item_id] = split
        start += count
    return [replace(item, split=split_by_id[item.item_id]) for item in items]
