"""CLIP index construction backed by a deterministic embedding cache."""

from pathlib import Path

from .clip_backend import ClipEmbeddingBackend
from .embedding_cache import (
    EmbeddingCache,
    cache_is_stale,
    load_embedding_cache,
    make_cache_metadata,
    write_embedding_cache,
)
from .manifest import ManifestItem
from .multimodal_index import MultimodalIndex, MultimodalIndexEntry


def _cache_matches_configuration(
    cache: EmbeddingCache, items: list[ManifestItem], backend: ClipEmbeddingBackend
) -> bool:
    expected = make_cache_metadata(
        items,
        backend_name=backend.backend_name,
        backend_version=backend.backend_version,
        model_name=backend.model_name,
        embedding_dimension=cache.metadata.embedding_dimension,
    )
    return not cache_is_stale(cache, expected)


def build_clip_index(
    items: list[ManifestItem], backend: ClipEmbeddingBackend, cache_path: Path
) -> tuple[MultimodalIndex, bool]:
    """Build an image index, reusing a compatible complete embedding cache."""
    cache: EmbeddingCache | None = None
    if cache_path.is_file():
        candidate = load_embedding_cache(cache_path)
        if _cache_matches_configuration(candidate, items, backend):
            expected_ids = {item.item_id for item in items}
            if set(candidate.text_embeddings) == expected_ids and set(
                candidate.image_embeddings
            ) == expected_ids:
                cache = candidate
    cache_reused = cache is not None
    if cache is None:
        text_embeddings = {item.item_id: backend.encode_text(item.caption) for item in items}
        image_embeddings = {
            item.item_id: backend.encode_image(item.image_path) for item in items
        }
        metadata = make_cache_metadata(
            items,
            backend_name=backend.backend_name,
            backend_version=backend.backend_version,
            model_name=backend.model_name,
            embedding_dimension=backend.dimension,
        )
        cache = EmbeddingCache(metadata, text_embeddings, image_embeddings)
        write_embedding_cache(cache, cache_path)
    entries = [
        MultimodalIndexEntry(
            item_id=item.item_id,
            image_path=item.image_path,
            caption=item.caption,
            split=item.split,
            source=item.source,
            vector=cache.image_embeddings[item.item_id],
        )
        for item in items
    ]
    return (
        MultimodalIndex(
            backend_name=backend.backend_name,
            backend_version=backend.backend_version,
            dimension=cache.metadata.embedding_dimension,
            entries=entries,
            model_name=backend.model_name,
        ),
        cache_reused,
    )
