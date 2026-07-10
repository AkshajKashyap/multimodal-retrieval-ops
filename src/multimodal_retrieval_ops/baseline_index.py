"""Serializable exact-search index for the lexical baseline."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .manifest import ManifestRecord, image_identity
from .text_baseline import build_vocabulary, encode_text


@dataclass(frozen=True)
class IndexEntry:
    item_id: str
    image_path: str
    caption: str
    split: str
    source: str
    vector: list[float]
    query_captions: list[str] | None = None


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float
    caption: str
    image_path: str
    split: str


def build_index(items: list[ManifestRecord]) -> tuple[list[str], list[IndexEntry]]:
    """Fit the train vocabulary and encode every registered candidate."""
    vocabulary = build_vocabulary(items)
    grouped: dict[str, list[ManifestRecord]] = {}
    for item in items:
        grouped.setdefault(image_identity(item), []).append(item)
    entries = [
        IndexEntry(
            item_id=image_id,
            image_path=group[0].image_path,
            caption=" ".join(item.caption for item in group),
            split=group[0].split,
            source=group[0].source,
            vector=encode_text(" ".join(item.caption for item in group), vocabulary),
            query_captions=[item.caption for item in group],
        )
        for image_id, group in grouped.items()
    ]
    return vocabulary, entries


def write_index(
    vocabulary: list[str], entries: list[IndexEntry], index_path: Path, vocab_path: Path
) -> None:
    """Write deterministic JSON index and vocabulary artifacts."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = {"entries": [asdict(entry) for entry in entries], "format_version": 1}
    vocab_data = {"tokens": vocabulary}
    index_path.write_text(json.dumps(index_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vocab_path.write_text(json.dumps(vocab_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_index(index_path: Path, vocab_path: Path) -> tuple[list[str], list[IndexEntry]]:
    """Load baseline artifacts from disk."""
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
    return vocab_data["tokens"], [IndexEntry(**entry) for entry in index_data["entries"]]


def exact_search(
    query: str,
    vocabulary: list[str],
    entries: list[IndexEntry],
    *,
    k: int = 5,
    allowed_splits: set[str] | None = None,
) -> list[SearchResult]:
    """Rank candidates by exact cosine similarity with stable tie-breaking."""
    if k <= 0:
        raise ValueError("k must be positive")
    query_vector = encode_text(query, vocabulary)
    candidates = entries
    if allowed_splits is not None:
        candidates = [entry for entry in entries if entry.split in allowed_splits]
    scored = [
        SearchResult(
            item_id=entry.item_id,
            score=sum(left * right for left, right in zip(query_vector, entry.vector, strict=True)),
            caption=entry.caption,
            image_path=entry.image_path,
            split=entry.split,
        )
        for entry in candidates
    ]
    return sorted(scored, key=lambda result: (-result.score, result.item_id))[:k]
