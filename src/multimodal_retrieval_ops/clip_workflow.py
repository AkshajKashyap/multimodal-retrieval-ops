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
from .manifest import ManifestRecord, caption_identity, image_identity
from .multimodal_index import CaptionQuery, MultimodalIndex, MultimodalIndexEntry


def _cache_matches_configuration(
    cache: EmbeddingCache, items: list[ManifestRecord], backend: ClipEmbeddingBackend
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
    items: list[ManifestRecord], backend: ClipEmbeddingBackend, cache_path: Path
) -> tuple[MultimodalIndex, bool]:
    """Build an image index, reusing a compatible complete embedding cache."""
    cache: EmbeddingCache | None = None
    if cache_path.is_file():
        candidate = load_embedding_cache(cache_path)
        if _cache_matches_configuration(candidate, items, backend):
            expected_image_ids = {image_identity(item) for item in items}
            expected_caption_ids = {caption_identity(item) for item in items}
            if set(candidate.text_embeddings) == expected_caption_ids and set(
                candidate.image_embeddings
            ) == expected_image_ids:
                cache = candidate
    cache_reused = cache is not None
    if cache is None:
        text_vectors = backend.encode_texts([item.caption for item in items])
        candidate_items: dict[str, ManifestRecord] = {}
        for item in items:
            candidate_items.setdefault(image_identity(item), item)
        image_vectors = backend.encode_images(
            [item.image_path for item in candidate_items.values()]
        )
        text_embeddings = {
            caption_identity(item): vector for item, vector in zip(items, text_vectors, strict=True)
        }
        image_embeddings = {
            image_id: vector
            for (image_id, _), vector in zip(candidate_items.items(), image_vectors, strict=True)
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
    candidate_items = {}
    for item in items:
        candidate_items.setdefault(image_identity(item), item)
    entries = [
        MultimodalIndexEntry(
            item_id=image_id,
            image_path=item.image_path,
            caption=item.caption,
            split=item.split,
            source=item.source,
            vector=cache.image_embeddings[image_id],
        )
        for image_id, item in candidate_items.items()
    ]
    queries = [
        CaptionQuery(caption_identity(item), image_identity(item), item.caption, item.split)
        for item in items
    ]
    return (
        MultimodalIndex(
            backend_name=backend.backend_name,
            backend_version=backend.backend_version,
            dimension=cache.metadata.embedding_dimension,
            entries=entries,
            model_name=backend.model_name,
            queries=queries,
        ),
        cache_reused,
    )
