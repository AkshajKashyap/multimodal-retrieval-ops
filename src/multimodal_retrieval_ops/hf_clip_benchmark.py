"""Bidirectional exact CLIP evaluation for official Flickr8k splits."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

from .clip_backend import ClipEmbeddingBackend
from .embedding_cache import manifest_digest
from .evaluation import RetrievalMetrics, metrics_from_ranks
from .hf_flickr8k import HFFlickr8kProvenance
from .manifest import ManifestItemV2
from .multimodal_index import (
    CaptionQuery,
    MultimodalIndex,
    MultimodalIndexEntry,
    write_multimodal_index,
)


@dataclass(frozen=True)
class HFBenchmarkCacheMetadata:
    backend_name: str
    backend_version: str
    model_name: str
    model_revision: str
    dataset_fingerprint: str
    manifest_fingerprint: str
    split: str
    max_images: int | None
    seed: int
    image_count: int
    caption_count: int
    embedding_dimension: int
    dataset_revision: str = "default"


@dataclass(frozen=True)
class HFBenchmarkCache:
    metadata: HFBenchmarkCacheMetadata
    image_embeddings: dict[str, list[float]]
    caption_embeddings: dict[str, list[float]]


@dataclass(frozen=True)
class DirectionResult:
    metrics: RetrievalMetrics
    candidate_count: int


@dataclass(frozen=True)
class BidirectionalResult:
    text_to_image: DirectionResult
    image_to_text: DirectionResult


def select_official_split(
    rows: list[ManifestItemV2], split: str, max_images: int | None, seed: int
) -> list[ManifestItemV2]:
    split_rows = [row for row in rows if row.split == split]
    image_ids = sorted({row.image_id for row in split_rows})
    if max_images is not None:
        random.Random(seed).shuffle(image_ids)
        selected = set(image_ids[:max_images])
        split_rows = [row for row in split_rows if row.image_id in selected]
    return sorted(split_rows, key=lambda row: (row.image_id, row.caption_id))


def make_hf_cache_metadata(
    rows: list[ManifestItemV2],
    backend: ClipEmbeddingBackend,
    provenance: HFFlickr8kProvenance,
    *,
    split: str,
    max_images: int | None,
    seed: int,
    dimension: int,
) -> HFBenchmarkCacheMetadata:
    return HFBenchmarkCacheMetadata(
        backend_name=backend.backend_name,
        backend_version=backend.backend_version,
        model_name=backend.model_name,
        model_revision=backend.model_revision or "default",
        dataset_fingerprint=provenance.resolved_fingerprint,
        manifest_fingerprint=manifest_digest(rows),
        split=split,
        max_images=max_images,
        seed=seed,
        image_count=len({row.image_id for row in rows}),
        caption_count=len(rows),
        embedding_dimension=dimension,
        dataset_revision=provenance.requested_revision,
    )


def hf_benchmark_cache_is_stale(
    cache: HFBenchmarkCache, expected: HFBenchmarkCacheMetadata
) -> bool:
    return cache.metadata != expected


def write_hf_benchmark_cache(cache: HFBenchmarkCache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cache), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_hf_benchmark_cache(path: Path) -> HFBenchmarkCache:
    data = json.loads(path.read_text(encoding="utf-8"))
    return HFBenchmarkCache(
        metadata=HFBenchmarkCacheMetadata(**data["metadata"]),
        image_embeddings=data["image_embeddings"],
        caption_embeddings=data["caption_embeddings"],
    )


def build_hf_benchmark_cache(
    rows: list[ManifestItemV2],
    backend: ClipEmbeddingBackend,
    provenance: HFFlickr8kProvenance,
    *,
    split: str,
    max_images: int | None,
    seed: int,
    cache_path: Path,
) -> tuple[HFBenchmarkCache, bool]:
    if cache_path.is_file():
        candidate = load_hf_benchmark_cache(cache_path)
        expected = make_hf_cache_metadata(
            rows,
            backend,
            provenance,
            split=split,
            max_images=max_images,
            seed=seed,
            dimension=candidate.metadata.embedding_dimension,
        )
        if not hf_benchmark_cache_is_stale(candidate, expected):
            if set(candidate.image_embeddings) == {row.image_id for row in rows} and set(
                candidate.caption_embeddings
            ) == {row.caption_id for row in rows}:
                return candidate, True
    image_rows: dict[str, ManifestItemV2] = {}
    for row in rows:
        image_rows.setdefault(row.image_id, row)
    image_vectors = backend.encode_images([row.image_path for row in image_rows.values()])
    caption_vectors = backend.encode_texts([row.caption for row in rows])
    image_embeddings = {
        image_id: vector
        for (image_id, _), vector in zip(image_rows.items(), image_vectors, strict=True)
    }
    caption_embeddings = {
        row.caption_id: vector for row, vector in zip(rows, caption_vectors, strict=True)
    }
    metadata = make_hf_cache_metadata(
        rows,
        backend,
        provenance,
        split=split,
        max_images=max_images,
        seed=seed,
        dimension=backend.dimension,
    )
    cache = HFBenchmarkCache(metadata, image_embeddings, caption_embeddings)
    write_hf_benchmark_cache(cache, cache_path)
    return cache, False


def bidirectional_ranks_reference(
    similarity: list[list[float]], caption_image_indices: list[int]
) -> tuple[list[int], list[int]]:
    """Small download-free correctness reference supporting multiple relevant captions."""
    text_ranks = []
    for scores, target in zip(similarity, caption_image_indices, strict=True):
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        text_ranks.append(order.index(target) + 1)
    image_count = len(similarity[0]) if similarity else 0
    image_ranks = []
    for image_index in range(image_count):
        order = sorted(
            range(len(similarity)),
            key=lambda caption_index: (-similarity[caption_index][image_index], caption_index),
        )
        relevant_positions = [
            position
            for position, caption_index in enumerate(order, start=1)
            if caption_image_indices[caption_index] == image_index
        ]
        image_ranks.append(min(relevant_positions))
    return text_ranks, image_ranks


def evaluate_bidirectional_vectorized(
    image_vectors: list[list[float]],
    caption_vectors: list[list[float]],
    caption_image_indices: list[int],
) -> BidirectionalResult:
    """Compute exact cosine ranks using vectorized Torch matrix multiplication and sorting."""
    try:
        import torch
    except ImportError as error:
        raise ValueError("Vectorized CLIP evaluation requires the optional clip extra") from error
    images = torch.tensor(image_vectors, dtype=torch.float32)
    captions = torch.tensor(caption_vectors, dtype=torch.float32)
    similarity = captions @ images.T
    targets = torch.tensor(caption_image_indices, dtype=torch.long)
    text_order = torch.argsort(similarity, dim=1, descending=True, stable=True)
    text_ranks = (
        (text_order == targets[:, None]).to(torch.int64).argmax(dim=1) + 1
    ).tolist()
    image_order = torch.argsort(similarity.T, dim=1, descending=True, stable=True)
    ordered_targets = targets[image_order]
    image_indices = torch.arange(images.shape[0], dtype=torch.long)[:, None]
    image_ranks = (
        (ordered_targets == image_indices).to(torch.int64).argmax(dim=1) + 1
    ).tolist()
    return BidirectionalResult(
        text_to_image=DirectionResult(metrics_from_ranks(text_ranks), images.shape[0]),
        image_to_text=DirectionResult(metrics_from_ranks(image_ranks), captions.shape[0]),
    )


def _direction_dict(result: DirectionResult) -> dict[str, Any]:
    return asdict(result.metrics) | {"candidate_count": result.candidate_count}


def render_hf_clip_report(
    result: BidirectionalResult,
    metadata: HFBenchmarkCacheMetadata,
    *,
    device: str,
    batch_size: int,
    cache_hit: bool,
    mode: str,
    dataset_name: str,
    resolved_dataset_revision: str,
) -> str:
    def row(name: str, direction: DirectionResult) -> str:
        metrics = direction.metrics
        return (
            f"| {name} | {metrics.recall_at_1:.4f} | {metrics.recall_at_5:.4f} | "
            f"{metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | "
            f"{metrics.median_rank:.2f} | {metrics.mean_rank:.2f} | "
            f"{metrics.query_count} | {direction.candidate_count} |"
        )

    return "\n".join(
        [
            "# CLIP Flickr8k Bidirectional Retrieval Report",
            "",
            "Run state: **success**",
            "",
            f"- Dataset source: `{dataset_name}`",
            f"- Resolved dataset revision: `{resolved_dataset_revision}`",
            f"- Dataset fingerprint: `{metadata.dataset_fingerprint}`",
            f"- Dataset revision: `{metadata.dataset_revision}`",
            f"- Manifest fingerprint: `{metadata.manifest_fingerprint}`",
            f"- Official split: `{metadata.split}`",
            f"- Benchmark mode: `{mode}`",
            f"- Model: `{metadata.model_name}`",
            f"- Model revision: `{metadata.model_revision}`",
            f"- Device: `{device}`",
            f"- Embedding dimension: {metadata.embedding_dimension}",
            f"- Batch size: {batch_size}",
            f"- Unique images: {metadata.image_count}",
            f"- Captions: {metadata.caption_count}",
            f"- Cache: {'hit' if cache_hit else 'miss'}",
            "",
            "| Direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank | Queries | Candidates |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            row("Text to image", result.text_to_image),
            row("Image to text", result.image_to_text),
            "",
            "Image-to-text rank uses the highest-ranked caption relevant to each image.",
            "The dataset source does not clearly expose licensing information; status is unresolved.",
            "",
        ]
    )


def run_hf_clip_benchmark(
    rows: list[ManifestItemV2],
    provenance: HFFlickr8kProvenance,
    backend: ClipEmbeddingBackend,
    *,
    split: str,
    max_images: int | None,
    seed: int,
    cache_path: Path,
    index_path: Path,
    report_path: Path,
    metrics_path: Path,
) -> tuple[BidirectionalResult, bool]:
    selected = select_official_split(rows, split, max_images, seed)
    if not selected:
        raise ValueError(f"manifest has no rows for official split '{split}'")
    cache, cache_hit = build_hf_benchmark_cache(
        selected,
        backend,
        provenance,
        split=split,
        max_images=max_images,
        seed=seed,
        cache_path=cache_path,
    )
    image_ids = sorted(cache.image_embeddings)
    caption_rows = sorted(selected, key=lambda row: row.caption_id)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    result = evaluate_bidirectional_vectorized(
        [cache.image_embeddings[image_id] for image_id in image_ids],
        [cache.caption_embeddings[row.caption_id] for row in caption_rows],
        [image_index[row.image_id] for row in caption_rows],
    )
    first_rows = {row.image_id: row for row in selected}
    index = MultimodalIndex(
        backend_name=backend.backend_name,
        backend_version=backend.backend_version,
        dimension=cache.metadata.embedding_dimension,
        model_name=backend.model_name,
        entries=[
            MultimodalIndexEntry(
                image_id,
                first_rows[image_id].image_path,
                first_rows[image_id].caption,
                split,
                first_rows[image_id].source,
                cache.image_embeddings[image_id],
            )
            for image_id in image_ids
        ],
        queries=[
            CaptionQuery(row.caption_id, row.image_id, row.caption, row.split)
            for row in caption_rows
        ],
    )
    write_multimodal_index(index, index_path)
    mode = "integration_subset" if max_images is not None else "complete_test_split"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_hf_clip_report(
            result,
            cache.metadata,
            device=backend.device,
            batch_size=backend.batch_size,
            cache_hit=cache_hit,
            mode=mode,
            dataset_name=provenance.dataset_name,
            resolved_dataset_revision=provenance.resolved_revision,
        ),
        encoding="utf-8",
    )
    metrics = {
        "run_state": "success",
        "dataset_name": provenance.dataset_name,
        "resolved_dataset_revision": provenance.resolved_revision,
        "dataset_fingerprint": cache.metadata.dataset_fingerprint,
        "dataset_revision": cache.metadata.dataset_revision,
        "manifest_fingerprint": cache.metadata.manifest_fingerprint,
        "split": split,
        "mode": mode,
        "model_name": backend.model_name,
        "model_revision": backend.model_revision or "default",
        "device": backend.device,
        "embedding_dimension": cache.metadata.embedding_dimension,
        "batch_size": backend.batch_size,
        "unique_image_count": cache.metadata.image_count,
        "caption_count": cache.metadata.caption_count,
        "cache_hit": cache_hit,
        "text_to_image": _direction_dict(result.text_to_image),
        "image_to_text": _direction_dict(result.image_to_text),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, cache_hit


def write_hf_clip_failure(path: Path, metrics_path: Path, state: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# CLIP Flickr8k Bidirectional Retrieval Report\n\nRun state: **{state}**\n\nDetail: {detail}\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": detail, "run_state": state}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
