"""Shared deterministic image embedding index and exact text-to-image search."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .embedding_backends import ImageEncoder, TextEncoder
from .manifest import ManifestRecord, caption_identity, image_identity


@dataclass(frozen=True)
class MultimodalIndexEntry:
    item_id: str
    image_path: str
    caption: str
    split: str
    source: str
    vector: list[float]


@dataclass(frozen=True)
class CaptionQuery:
    caption_id: str
    image_id: str
    caption: str
    split: str


@dataclass(frozen=True)
class MultimodalSearchResult:
    item_id: str
    score: float
    caption: str
    image_path: str
    split: str


@dataclass(frozen=True)
class MultimodalIndex:
    backend_name: str
    backend_version: str
    dimension: int
    entries: list[MultimodalIndexEntry]
    model_name: str | None = None
    queries: list[CaptionQuery] | None = None


def build_multimodal_index(
    items: list[ManifestRecord], image_encoder: ImageEncoder, text_encoder: TextEncoder
) -> MultimodalIndex:
    """Encode exactly one image candidate for every unique manifest item."""
    if image_encoder.dimension != text_encoder.dimension:
        raise ValueError("text and image encoder dimensions must match")
    candidates: dict[str, ManifestRecord] = {}
    for item in items:
        candidates.setdefault(image_identity(item), item)
    entries = [
        MultimodalIndexEntry(
            item_id=image_id,
            image_path=item.image_path,
            caption=item.caption,
            split=item.split,
            source=item.source,
            vector=image_encoder.encode_image(item.image_path),
        )
        for image_id, item in candidates.items()
    ]
    queries = [
        CaptionQuery(
            caption_id=caption_identity(item),
            image_id=image_identity(item),
            caption=item.caption,
            split=item.split,
        )
        for item in items
    ]
    return MultimodalIndex(
        backend_name=f"{text_encoder.backend_name}+{image_encoder.backend_name}",
        backend_version=f"{text_encoder.backend_version}+{image_encoder.backend_version}",
        dimension=text_encoder.dimension,
        entries=entries,
        model_name=None,
        queries=queries,
    )


def write_multimodal_index(index: MultimodalIndex, path: Path) -> None:
    """Write a stable JSON index artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(index) | {"format_version": 1}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_multimodal_index(path: Path) -> MultimodalIndex:
    """Load a deterministic multimodal index artifact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return MultimodalIndex(
        backend_name=data["backend_name"],
        backend_version=data["backend_version"],
        dimension=data["dimension"],
        entries=[MultimodalIndexEntry(**entry) for entry in data["entries"]],
        model_name=data.get("model_name"),
        queries=(
            [CaptionQuery(**query) for query in data["queries"]]
            if data.get("queries") is not None
            else None
        ),
    )


def search_multimodal_index(
    query: str,
    text_encoder: TextEncoder,
    index: MultimodalIndex,
    *,
    k: int = 5,
    allowed_splits: set[str] | None = None,
) -> list[MultimodalSearchResult]:
    """Rank image candidates by exact cosine similarity to encoded text."""
    if k <= 0:
        raise ValueError("k must be positive")
    if text_encoder.dimension != index.dimension:
        raise ValueError("text encoder dimension does not match index dimension")
    query_vector = text_encoder.encode_text(query)
    candidates = index.entries
    if allowed_splits is not None:
        candidates = [entry for entry in candidates if entry.split in allowed_splits]
    results = [
        MultimodalSearchResult(
            item_id=entry.item_id,
            score=sum(left * right for left, right in zip(query_vector, entry.vector, strict=True)),
            caption=entry.caption,
            image_path=entry.image_path,
            split=entry.split,
        )
        for entry in candidates
    ]
    return sorted(results, key=lambda result: (-result.score, result.item_id))[:k]
