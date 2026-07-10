"""Serializable exact-search index for the lexical baseline."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .manifest import ManifestItem
from .text_baseline import build_vocabulary, encode_text


@dataclass(frozen=True)
class IndexEntry:
    item_id: str
    image_path: str
    caption: str
    split: str
    source: str
    vector: list[float]


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float
    caption: str
    image_path: str
    split: str


def build_index(items: list[ManifestItem]) -> tuple[list[str], list[IndexEntry]]:
    """Fit the train vocabulary and encode every registered candidate."""
    vocabulary = build_vocabulary(items)
    entries = [
        IndexEntry(
            item_id=item.item_id,
            image_path=item.image_path,
            caption=item.caption,
            split=item.split,
            source=item.source,
            vector=encode_text(item.caption, vocabulary),
        )
        for item in items
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
