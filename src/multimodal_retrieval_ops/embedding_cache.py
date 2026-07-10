"""Deterministic JSON embedding cache with staleness detection."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .manifest import ManifestItem


@dataclass(frozen=True)
class EmbeddingCacheMetadata:
    backend_name: str
    backend_version: str
    model_name: str
    manifest_hash: str
    item_count: int
    embedding_dimension: int


@dataclass(frozen=True)
class EmbeddingCache:
    metadata: EmbeddingCacheMetadata
    text_embeddings: dict[str, list[float]]
    image_embeddings: dict[str, list[float]]


def manifest_digest(items: list[ManifestItem]) -> str:
    """Hash canonical manifest contents independent of CSV formatting."""
    rows = [asdict(item) for item in items]
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_cache_metadata(
    items: list[ManifestItem],
    *,
    backend_name: str,
    backend_version: str,
    model_name: str,
    embedding_dimension: int,
) -> EmbeddingCacheMetadata:
    return EmbeddingCacheMetadata(
        backend_name=backend_name,
        backend_version=backend_version,
        model_name=model_name,
        manifest_hash=manifest_digest(items),
        item_count=len(items),
        embedding_dimension=embedding_dimension,
    )


def cache_is_stale(cache: EmbeddingCache, expected: EmbeddingCacheMetadata) -> bool:
    """Detect any backend, model, manifest, count, or dimension mismatch."""
    return cache.metadata != expected


def write_embedding_cache(cache: EmbeddingCache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cache), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_embedding_cache(path: Path) -> EmbeddingCache:
    data = json.loads(path.read_text(encoding="utf-8"))
    return EmbeddingCache(
        metadata=EmbeddingCacheMetadata(**data["metadata"]),
        text_embeddings=data["text_embeddings"],
        image_embeddings=data["image_embeddings"],
    )
