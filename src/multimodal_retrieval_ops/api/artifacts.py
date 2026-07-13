"""Read-only loading and compatibility validation for service artifacts."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from ..embedding_cache import manifest_digest
from ..faiss_flat import (
    FAISS_INDEX_TYPE,
    FaissDependencyError,
    FaissIndexArtifact,
    expected_artifact_metadata,
    file_sha256,
    load_faiss_artifact,
    ordered_embeddings,
    require_faiss,
)
from ..faiss_hnsw import (
    HNSW_INDEX_TYPE,
    HNSWIndexArtifact,
    load_hnsw_artifact,
    make_hnsw_metadata,
    set_ef_search,
)
from ..hf_clip_benchmark import HFBenchmarkCache, load_hf_benchmark_cache
from ..manifest import ManifestItemV2, read_manifest
from .settings import ServiceSettings


class ServiceArtifactError(RuntimeError):
    """A safe, path-free readiness failure."""

    def __init__(self, state: str, reason: str) -> None:
        self.state = state
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class ServiceArtifacts:
    cache: HFBenchmarkCache
    text_to_image: FaissIndexArtifact | HNSWIndexArtifact
    image_to_text: FaissIndexArtifact | HNSWIndexArtifact
    caption_targets: dict[str, str]
    caption_text: dict[str, str]
    image_captions: dict[str, list[str]]
    safe_image_paths: dict[str, str]
    image_splits: dict[str, str]
    backend: str
    ef_search: int | None


def _missing_components(settings: ServiceSettings) -> list[str]:
    index_root = settings.index_artifacts_path
    paths = {
        "embedding cache": settings.embedding_cache_path,
        "manifest": settings.manifest_path,
        "text-to-image index": index_root / "text_to_image.faiss",
        "text-to-image metadata": index_root / "text_to_image.json",
        "image-to-text index": index_root / "image_to_text.faiss",
        "image-to-text metadata": index_root / "image_to_text.json",
    }
    return [name for name, path in paths.items() if not path.is_file()]


def _validate_vectors(cache: HFBenchmarkCache) -> None:
    dimension = cache.metadata.embedding_dimension
    for collection_name, embeddings in (
        ("image", cache.image_embeddings),
        ("caption", cache.caption_embeddings),
    ):
        expected_count = (
            cache.metadata.image_count
            if collection_name == "image"
            else cache.metadata.caption_count
        )
        if len(embeddings) != expected_count:
            raise ServiceArtifactError(
                "artifact_incompatible", f"{collection_name} embedding count is incompatible"
            )
        for vector in embeddings.values():
            if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
                raise ServiceArtifactError(
                    "artifact_incompatible",
                    f"{collection_name} embedding dimension or values are incompatible",
                )
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(norm, 1.0, abs_tol=1e-4):
                raise ServiceArtifactError(
                    "artifact_incompatible", f"{collection_name} embeddings are not normalized"
                )


def _load_manifest_metadata(
    manifest_path: Path, cache: HFBenchmarkCache
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, list[str]],
    dict[str, str],
    dict[str, str],
]:
    try:
        raw_rows = read_manifest(manifest_path)
    except Exception as error:
        raise ServiceArtifactError("artifact_incompatible", "manifest validation failed") from error
    rows = [
        row
        for row in raw_rows
        if isinstance(row, ManifestItemV2) and row.split == cache.metadata.split
    ]
    if len(rows) != cache.metadata.caption_count:
        raise ServiceArtifactError(
            "artifact_incompatible", "manifest split caption count does not match the cache"
        )
    if manifest_digest(rows) != cache.metadata.manifest_fingerprint:
        raise ServiceArtifactError(
            "artifact_incompatible", "manifest fingerprint does not match the cache"
        )
    caption_targets = {row.caption_id: row.image_id for row in rows}
    caption_text = {row.caption_id: row.caption for row in rows}
    if set(caption_targets) != set(cache.caption_embeddings):
        raise ServiceArtifactError(
            "artifact_incompatible", "manifest caption IDs do not match the cache"
        )
    if set(caption_targets.values()) != set(cache.image_embeddings):
        raise ServiceArtifactError(
            "artifact_incompatible", "manifest image IDs do not match the cache"
        )
    image_captions: dict[str, list[tuple[str, str]]] = {}
    safe_image_paths: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    for row in rows:
        image_captions.setdefault(row.image_id, []).append((row.caption_id, row.caption))
        safe_image_paths[row.image_id] = f"{row.split}/{Path(row.image_path).name}"
        image_splits[row.image_id] = row.split
    ordered_captions = {
        image_id: [caption for _, caption in sorted(captions)]
        for image_id, captions in image_captions.items()
    }
    return (
        caption_targets,
        caption_text,
        ordered_captions,
        safe_image_paths,
        image_splits,
    )


def _validate_loaded_indexes(
    settings: ServiceSettings,
    cache: HFBenchmarkCache,
    text_index: Any,
    image_index: Any,
) -> None:
    faiss = require_faiss()
    expected_type = FAISS_INDEX_TYPE if settings.backend == "flat" else HNSW_INDEX_TYPE
    for direction, artifact, expected_ids in (
        ("text_to_image", text_index, sorted(cache.image_embeddings)),
        ("image_to_text", image_index, sorted(cache.caption_embeddings)),
    ):
        metadata = artifact.metadata
        if metadata.direction != direction or metadata.index_type != expected_type:
            raise ServiceArtifactError(
                "artifact_incompatible", f"{direction} index type or direction is incompatible"
            )
        if type(artifact.index).__name__ != expected_type:
            raise ServiceArtifactError(
                "artifact_incompatible", f"{direction} binary index type is incompatible"
            )
        if metadata.candidate_ids != expected_ids:
            raise ServiceArtifactError(
                "artifact_incompatible", f"{direction} candidate ordering is incompatible"
            )
        if artifact.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ServiceArtifactError(
                "artifact_incompatible", f"{direction} index metric is incompatible"
            )


def _load_flat(
    settings: ServiceSettings, cache: HFBenchmarkCache
) -> tuple[FaissIndexArtifact, FaissIndexArtifact]:
    root = settings.index_artifacts_path
    artifacts = []
    for direction in ("text_to_image", "image_to_text"):
        expected = expected_artifact_metadata(cache, settings.embedding_cache_path, direction)
        artifacts.append(
            load_faiss_artifact(
                root / f"{direction}.faiss", root / f"{direction}.json", expected
            )
        )
    return artifacts[0], artifacts[1]


def _load_hnsw(
    settings: ServiceSettings, cache: HFBenchmarkCache
) -> tuple[HNSWIndexArtifact, HNSWIndexArtifact]:
    faiss = require_faiss()
    root = settings.index_artifacts_path
    fingerprint = file_sha256(settings.embedding_cache_path)
    artifacts = []
    for direction, embeddings in (
        ("text_to_image", cache.image_embeddings),
        ("image_to_text", cache.caption_embeddings),
    ):
        ids, _ = ordered_embeddings(embeddings)
        expected = make_hnsw_metadata(
            cache,
            direction=direction,
            candidate_ids=ids,
            source_cache_fingerprint=fingerprint,
            faiss_version=faiss.__version__,
        )
        artifact = load_hnsw_artifact(
            root / f"{direction}.faiss", root / f"{direction}.json", expected
        )
        set_ef_search(artifact.index, settings.ef_search)
        artifacts.append(artifact)
    return artifacts[0], artifacts[1]


def load_service_artifacts(settings: ServiceSettings) -> ServiceArtifacts:
    """Load and validate persisted artifacts without rebuilding any of them."""
    try:
        settings.validate()
    except ValueError as error:
        raise ServiceArtifactError("artifact_incompatible", str(error)) from error
    missing = _missing_components(settings)
    if missing:
        raise ServiceArtifactError(
            "artifact_unavailable", "missing required artifacts: " + ", ".join(missing)
        )
    try:
        cache = load_hf_benchmark_cache(settings.embedding_cache_path)
        _validate_vectors(cache)
        (
            caption_targets,
            caption_text,
            image_captions,
            safe_image_paths,
            image_splits,
        ) = _load_manifest_metadata(settings.manifest_path, cache)
        if settings.backend == "flat":
            text_index, image_index = _load_flat(settings, cache)
        else:
            text_index, image_index = _load_hnsw(settings, cache)
        _validate_loaded_indexes(settings, cache, text_index, image_index)
    except ServiceArtifactError:
        raise
    except FaissDependencyError as error:
        raise ServiceArtifactError(
            "dependency_unavailable", "optional FAISS CPU dependency is unavailable"
        ) from error
    except Exception as error:
        raise ServiceArtifactError(
            "artifact_incompatible", "persisted artifact compatibility validation failed"
        ) from error
    return ServiceArtifacts(
        cache=cache,
        text_to_image=text_index,
        image_to_text=image_index,
        caption_targets=caption_targets,
        caption_text=caption_text,
        image_captions=image_captions,
        safe_image_paths=safe_image_paths,
        image_splits=image_splits,
        backend=settings.backend,
        ef_search=settings.ef_search if settings.backend == "hnsw" else None,
    )
