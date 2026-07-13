"""Bounded FAISS HNSW comparison over the existing Flickr8k embedding cache."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

from .evaluation import RetrievalMetrics, metrics_from_ranks
from .faiss_flat import (
    FAISS_BACKEND_NAME,
    PREPROCESSING_VERSION,
    FaissCacheError,
    FaissFlatError,
    FaissIndexStaleError,
    FaissIndexArtifact,
    _as_normalized_matrix,
    file_sha256,
    load_flickr8k_artifacts,
    ordered_embeddings,
    require_faiss,
)
from .hf_clip_benchmark import HFBenchmarkCache, load_hf_benchmark_cache
from .manifest import ManifestItemV2, read_manifest

HNSW_BACKEND_VERSION = "1"
HNSW_FORMAT_VERSION = 1
HNSW_INDEX_TYPE = "IndexHNSWFlat"
HNSW_METRIC = "inner_product"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 100
ALLOWED_EF_SEARCH = (16, 32, 64)
TIMING_REPETITIONS = 5


@dataclass(frozen=True)
class HNSWIndexMetadata:
    format_version: int
    backend_name: str
    backend_version: str
    faiss_version: str
    index_type: str
    metric: str
    direction: str
    embedding_dimension: int
    m: int
    ef_construction: int
    model_name: str
    model_revision: str
    dataset_fingerprint: str
    manifest_fingerprint: str
    split: str
    candidate_count: int
    candidate_ids: list[str]
    preprocessing_version: str
    source_cache_fingerprint: str


@dataclass(frozen=True)
class HNSWIndexArtifact:
    metadata: HNSWIndexMetadata
    index: Any


@dataclass(frozen=True)
class LatencyResult:
    warmup_count: int
    measured_repetitions: int
    median_batch_seconds: float
    mean_batch_seconds: float
    approximate_queries_per_second: float


@dataclass(frozen=True)
class ApproximationResult:
    ef_search: int
    top1_agreement: float
    top5_reference_set_recall: float
    top10_reference_set_recall: float
    mean_overlap_at_5: float
    mean_overlap_at_10: float
    metrics: RetrievalMetrics
    metric_absolute_differences: dict[str, float]
    latency: LatencyResult


@dataclass(frozen=True)
class DirectionBenchmark:
    query_count: int
    candidate_count: int
    flat_metrics: RetrievalMetrics
    flat_latency: LatencyResult
    configurations: list[ApproximationResult]


@dataclass(frozen=True)
class HNSWBenchmarkResult:
    text_to_image: DirectionBenchmark
    image_to_text: DirectionBenchmark
    recommendation: str
    recommendation_reason: str


def validate_hnsw_parameters(m: int, ef_construction: int, ef_search: int | None = None) -> None:
    if m != HNSW_M or ef_construction != HNSW_EF_CONSTRUCTION:
        raise FaissFlatError(
            f"HNSW construction is fixed at M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}"
        )
    if ef_search is not None and ef_search not in ALLOWED_EF_SEARCH:
        allowed = ", ".join(str(value) for value in ALLOWED_EF_SEARCH)
        raise FaissFlatError(f"efSearch must be one of: {allowed}")


def build_hnsw_index(
    vectors: list[list[float]],
    expected_dimension: int | None = None,
    *,
    m: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
) -> Any:
    validate_hnsw_parameters(m, ef_construction)
    faiss = require_faiss()
    matrix = _as_normalized_matrix(vectors, expected_dimension)
    index = faiss.IndexHNSWFlat(int(matrix.shape[1]), m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(matrix)
    return index


def set_ef_search(index: Any, ef_search: int) -> None:
    validate_hnsw_parameters(HNSW_M, HNSW_EF_CONSTRUCTION, ef_search)
    if not hasattr(index, "hnsw"):
        raise FaissFlatError("efSearch can only be applied to an HNSW index")
    index.hnsw.efSearch = ef_search


def make_hnsw_metadata(
    cache: HFBenchmarkCache,
    *,
    direction: str,
    candidate_ids: list[str],
    source_cache_fingerprint: str,
    faiss_version: str,
) -> HNSWIndexMetadata:
    metadata = cache.metadata
    return HNSWIndexMetadata(
        format_version=HNSW_FORMAT_VERSION,
        backend_name=FAISS_BACKEND_NAME,
        backend_version=HNSW_BACKEND_VERSION,
        faiss_version=faiss_version,
        index_type=HNSW_INDEX_TYPE,
        metric=HNSW_METRIC,
        direction=direction,
        embedding_dimension=metadata.embedding_dimension,
        m=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
        model_name=metadata.model_name,
        model_revision=metadata.model_revision,
        dataset_fingerprint=metadata.dataset_fingerprint,
        manifest_fingerprint=metadata.manifest_fingerprint,
        split=metadata.split,
        candidate_count=len(candidate_ids),
        candidate_ids=candidate_ids,
        preprocessing_version=PREPROCESSING_VERSION,
        source_cache_fingerprint=source_cache_fingerprint,
    )


def hnsw_metadata_is_stale(actual: HNSWIndexMetadata, expected: HNSWIndexMetadata) -> bool:
    return actual != expected


def save_hnsw_artifact(
    artifact: HNSWIndexArtifact, index_path: Path, metadata_path: Path
) -> None:
    faiss = require_faiss()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_path.with_name(f".{index_path.name}.tmp")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    faiss.write_index(artifact.index, str(temporary_index))
    temporary_metadata.write_text(
        json.dumps(asdict(artifact.metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_index, index_path)
    os.replace(temporary_metadata, metadata_path)


def load_hnsw_artifact(
    index_path: Path,
    metadata_path: Path,
    expected: HNSWIndexMetadata | None = None,
) -> HNSWIndexArtifact:
    faiss = require_faiss()
    if not index_path.is_file() or not metadata_path.is_file():
        raise FaissIndexStaleError(
            f"FAISS HNSW artifact is missing: {index_path} or {metadata_path}"
        )
    metadata = HNSWIndexMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
    validate_hnsw_parameters(metadata.m, metadata.ef_construction)
    if expected is not None and hnsw_metadata_is_stale(metadata, expected):
        raise FaissIndexStaleError(f"FAISS HNSW metadata is stale for {metadata.direction}")
    index = faiss.read_index(str(index_path))
    if (
        index.d != metadata.embedding_dimension
        or index.ntotal != metadata.candidate_count
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
        or not hasattr(index, "hnsw")
        or index.hnsw.efConstruction != metadata.ef_construction
    ):
        raise FaissIndexStaleError("FAISS HNSW binary does not match companion metadata")
    return HNSWIndexArtifact(metadata, index)


def _validate_full_cache(cache: HFBenchmarkCache) -> None:
    if (
        cache.metadata.split != "test"
        or cache.metadata.max_images is not None
        or cache.metadata.image_count != 1000
        or cache.metadata.caption_count != 5000
    ):
        raise FaissCacheError(
            "embedding cache is incompatible; expected full test split with 1000 images and 5000 captions"
        )


def build_flickr8k_hnsw_artifacts(
    cache_path: Path, artifacts_dir: Path
) -> tuple[HNSWIndexArtifact, HNSWIndexArtifact]:
    faiss = require_faiss()
    if not cache_path.is_file():
        raise FaissCacheError(f"required embedding cache is missing: {cache_path}")
    cache = load_hf_benchmark_cache(cache_path)
    _validate_full_cache(cache)
    fingerprint = file_sha256(cache_path)
    artifacts: list[HNSWIndexArtifact] = []
    for direction, embeddings in (
        ("text_to_image", cache.image_embeddings),
        ("image_to_text", cache.caption_embeddings),
    ):
        ids, vectors = ordered_embeddings(embeddings)
        metadata = make_hnsw_metadata(
            cache,
            direction=direction,
            candidate_ids=ids,
            source_cache_fingerprint=fingerprint,
            faiss_version=faiss.__version__,
        )
        artifact = HNSWIndexArtifact(
            metadata, build_hnsw_index(vectors, cache.metadata.embedding_dimension)
        )
        save_hnsw_artifact(
            artifact,
            artifacts_dir / f"{direction}.faiss",
            artifacts_dir / f"{direction}.json",
        )
        artifacts.append(artifact)
    return artifacts[0], artifacts[1]


def _expected_hnsw_metadata(
    cache: HFBenchmarkCache, cache_path: Path, direction: str
) -> HNSWIndexMetadata:
    faiss = require_faiss()
    embeddings = (
        cache.image_embeddings if direction == "text_to_image" else cache.caption_embeddings
    )
    ids, _ = ordered_embeddings(embeddings)
    return make_hnsw_metadata(
        cache,
        direction=direction,
        candidate_ids=ids,
        source_cache_fingerprint=file_sha256(cache_path),
        faiss_version=faiss.__version__,
    )


def load_flickr8k_hnsw_artifacts(
    cache_path: Path, artifacts_dir: Path
) -> tuple[HFBenchmarkCache, HNSWIndexArtifact, HNSWIndexArtifact]:
    if not cache_path.is_file():
        raise FaissCacheError(f"required embedding cache is missing: {cache_path}")
    cache = load_hf_benchmark_cache(cache_path)
    _validate_full_cache(cache)
    artifacts = []
    for direction in ("text_to_image", "image_to_text"):
        artifacts.append(
            load_hnsw_artifact(
                artifacts_dir / f"{direction}.faiss",
                artifacts_dir / f"{direction}.json",
                _expected_hnsw_metadata(cache, cache_path, direction),
            )
        )
    return cache, artifacts[0], artifacts[1]


def search_hnsw_embedding(
    query_id: str,
    query_embeddings: dict[str, list[float]],
    artifact: HNSWIndexArtifact,
    k: int,
    ef_search: int,
) -> list[dict[str, float | str]]:
    if query_id not in query_embeddings:
        raise FaissCacheError(f"cached query embedding is missing: {query_id}")
    if k <= 0:
        raise FaissFlatError("k must be positive")
    set_ef_search(artifact.index, ef_search)
    query = _as_normalized_matrix(
        [query_embeddings[query_id]], artifact.metadata.embedding_dimension
    )
    scores, indices = artifact.index.search(query, min(k, artifact.metadata.candidate_count))
    valid = [(float(score), int(index)) for score, index in zip(scores[0], indices[0]) if index >= 0]
    if len({index for _, index in valid}) != len(valid):
        raise FaissFlatError("HNSW returned duplicate candidate indices")
    return [
        {"candidate_id": artifact.metadata.candidate_ids[index], "score": score}
        for score, index in valid
    ]


def _ranks(order: Any, relevant: list[set[int]]) -> list[int]:
    ranks = []
    for row, targets in zip(order.tolist(), relevant, strict=True):
        rank = next((position for position, candidate in enumerate(row, 1) if candidate in targets), None)
        if rank is None:
            rank = len(row) + 1
        ranks.append(rank)
    return ranks


def _metric_differences(reference: RetrievalMetrics, actual: RetrievalMetrics) -> dict[str, float]:
    return {
        name: abs(float(getattr(reference, name)) - float(getattr(actual, name)))
        for name in (
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "median_rank",
            "mean_rank",
            "query_count",
        )
    }


def approximation_metrics(reference_order: Any, approximate_order: Any) -> dict[str, float]:
    if reference_order.shape[0] != approximate_order.shape[0]:
        raise FaissFlatError("reference and approximate query counts differ")
    count = reference_order.shape[0]
    values: dict[str, float] = {}
    for k in (1, 5, 10):
        width = min(k, reference_order.shape[1], approximate_order.shape[1])
        overlaps = [
            len(set(reference_order[row, :width]) & set(approximate_order[row, :width]))
            for row in range(count)
        ]
        if k == 1:
            values["top1_agreement"] = sum(value == 1 for value in overlaps) / count
        else:
            values[f"top{k}_reference_set_recall"] = sum(overlaps) / (count * width)
            values[f"mean_overlap_at_{k}"] = statistics.mean(overlaps)
    return values


def measure_search_latency(index: Any, queries: Any, *, repetitions: int = TIMING_REPETITIONS) -> LatencyResult:
    if not 1 <= repetitions <= TIMING_REPETITIONS:
        raise FaissFlatError(f"timing repetitions must be between 1 and {TIMING_REPETITIONS}")
    index.search(queries, min(10, index.ntotal))  # one warmup
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        index.search(queries, min(10, index.ntotal))
        samples.append(time.perf_counter() - started)
    mean_seconds = statistics.mean(samples)
    return LatencyResult(
        warmup_count=1,
        measured_repetitions=repetitions,
        median_batch_seconds=statistics.median(samples),
        mean_batch_seconds=mean_seconds,
        approximate_queries_per_second=len(queries) / mean_seconds,
    )


def _benchmark_direction(
    queries: list[list[float]],
    relevant: list[set[int]],
    flat: FaissIndexArtifact,
    hnsw: HNSWIndexArtifact,
) -> DirectionBenchmark:
    query_matrix = _as_normalized_matrix(queries, flat.metadata.embedding_dimension)
    candidate_count = flat.metadata.candidate_count
    _, reference_order = flat.index.search(query_matrix, candidate_count)
    reference_metrics = metrics_from_ranks(_ranks(reference_order, relevant))
    flat_latency = measure_search_latency(flat.index, query_matrix)
    configurations = []
    for ef_search in ALLOWED_EF_SEARCH:
        set_ef_search(hnsw.index, ef_search)
        _, approximate_order = hnsw.index.search(query_matrix, candidate_count)
        metrics = metrics_from_ranks(_ranks(approximate_order, relevant))
        agreement = approximation_metrics(reference_order[:, :10], approximate_order[:, :10])
        configurations.append(
            ApproximationResult(
                ef_search=ef_search,
                metrics=metrics,
                metric_absolute_differences=_metric_differences(reference_metrics, metrics),
                latency=measure_search_latency(hnsw.index, query_matrix),
                **agreement,
            )
        )
    return DirectionBenchmark(
        query_count=len(queries),
        candidate_count=candidate_count,
        flat_metrics=reference_metrics,
        flat_latency=flat_latency,
        configurations=configurations,
    )


def recommend_configuration(
    text_to_image: DirectionBenchmark, image_to_text: DirectionBenchmark
) -> tuple[str, str]:
    by_direction = (
        {result.ef_search: result for result in text_to_image.configurations},
        {result.ef_search: result for result in image_to_text.configurations},
    )
    passing = []
    for ef_search in ALLOWED_EF_SEARCH:
        if all(
            direction[ef_search].metric_absolute_differences["recall_at_10"] <= 0.005
            and direction[ef_search].metric_absolute_differences["mrr"] <= 0.005
            and direction[ef_search].top10_reference_set_recall >= 0.98
            for direction in by_direction
        ):
            passing.append(ef_search)
    if not passing:
        return "FlatIP", "No HNSW configuration passed the conservative two-direction gate."
    selected = min(passing)
    return (
        f"HNSW efSearch={selected}",
        "Selected the lowest efSearch passing the bounded two-direction accuracy gate.",
    )


def evaluate_flickr8k_hnsw(
    cache_path: Path,
    manifest_path: Path,
    flat_artifacts_dir: Path,
    hnsw_artifacts_dir: Path,
) -> tuple[HNSWBenchmarkResult, HNSWIndexMetadata]:
    cache, flat_text, flat_image = load_flickr8k_artifacts(cache_path, flat_artifacts_dir)
    hnsw_cache, hnsw_text, hnsw_image = load_flickr8k_hnsw_artifacts(
        cache_path, hnsw_artifacts_dir
    )
    if cache.metadata != hnsw_cache.metadata:
        raise FaissIndexStaleError("FlatIP and HNSW indexes do not reference the same cache")
    rows = [
        row
        for row in read_manifest(manifest_path)
        if isinstance(row, ManifestItemV2) and row.split == "test"
    ]
    caption_targets = {row.caption_id: row.image_id for row in rows}
    image_ids = flat_text.metadata.candidate_ids
    caption_ids = flat_image.metadata.candidate_ids
    if set(caption_targets) != set(caption_ids):
        raise FaissCacheError("manifest caption IDs do not match cached caption embeddings")
    image_positions = {value: index for index, value in enumerate(image_ids)}
    caption_positions = {value: index for index, value in enumerate(caption_ids)}
    text_relevant = [{image_positions[caption_targets[value]]} for value in caption_ids]
    captions_by_image = {image_id: set() for image_id in image_ids}
    for caption_id, image_id in caption_targets.items():
        captions_by_image[image_id].add(caption_positions[caption_id])
    text = _benchmark_direction(
        [cache.caption_embeddings[value] for value in caption_ids],
        text_relevant,
        flat_text,
        hnsw_text,
    )
    image = _benchmark_direction(
        [cache.image_embeddings[value] for value in image_ids],
        [captions_by_image[value] for value in image_ids],
        flat_image,
        hnsw_image,
    )
    recommendation, reason = recommend_configuration(text, image)
    return HNSWBenchmarkResult(text, image, recommendation, reason), hnsw_text.metadata


def _metrics_row(name: str, result: ApproximationResult) -> str:
    metrics = result.metrics
    return (
        f"| {name} | {result.top1_agreement:.4f} | {result.top5_reference_set_recall:.4f} | "
        f"{result.top10_reference_set_recall:.4f} | {result.mean_overlap_at_5:.3f} | "
        f"{result.mean_overlap_at_10:.3f} | {metrics.recall_at_1:.4f} | "
        f"{metrics.recall_at_5:.4f} | {metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | "
        f"{metrics.median_rank:.2f} | {metrics.mean_rank:.2f} |"
    )


def render_hnsw_report(result: HNSWBenchmarkResult, metadata: HNSWIndexMetadata) -> str:
    def direction(name: str, value: DirectionBenchmark) -> list[str]:
        flat = value.flat_metrics
        lines = [
            f"## {name}",
            "",
            f"- Queries: {value.query_count}",
            f"- Candidates: {value.candidate_count}",
            f"- FlatIP metrics (R@1/R@5/R@10/MRR/median/mean): "
            f"{flat.recall_at_1:.4f}/{flat.recall_at_5:.4f}/{flat.recall_at_10:.4f}/"
            f"{flat.mrr:.4f}/{flat.median_rank:.2f}/{flat.mean_rank:.2f}",
            "",
            "| Setting | Top-1 agreement | Top-5 ref recall | Top-10 ref recall | Mean overlap@5 | Mean overlap@10 | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(_metrics_row(f"efSearch={item.ef_search}", item) for item in value.configurations)
        lines.extend(
            [
                "",
                "| Setting | Absolute ΔR@1 | ΔR@5 | ΔR@10 | ΔMRR | ΔMedian rank | ΔMean rank |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *(
                    f"| efSearch={item.ef_search} | "
                    f"{item.metric_absolute_differences['recall_at_1']:.4f} | "
                    f"{item.metric_absolute_differences['recall_at_5']:.4f} | "
                    f"{item.metric_absolute_differences['recall_at_10']:.4f} | "
                    f"{item.metric_absolute_differences['mrr']:.4f} | "
                    f"{item.metric_absolute_differences['median_rank']:.2f} | "
                    f"{item.metric_absolute_differences['mean_rank']:.2f} |"
                    for item in value.configurations
                ),
                "",
                "| Setting | Median batch (s) | Mean batch (s) | Approx. queries/s |",
                "| --- | ---: | ---: | ---: |",
                f"| FlatIP | {value.flat_latency.median_batch_seconds:.6f} | "
                f"{value.flat_latency.mean_batch_seconds:.6f} | "
                f"{value.flat_latency.approximate_queries_per_second:.2f} |",
            ]
        )
        lines.extend(
            f"| efSearch={item.ef_search} | {item.latency.median_batch_seconds:.6f} | "
            f"{item.latency.mean_batch_seconds:.6f} | "
            f"{item.latency.approximate_queries_per_second:.2f} |"
            for item in value.configurations
        )
        lines.append("")
        return lines

    return "\n".join(
        [
            "# FAISS HNSW Comparison Report",
            "",
            "Run state: **success**",
            "",
            f"- Dataset fingerprint: `{metadata.dataset_fingerprint}`",
            f"- Manifest fingerprint: `{metadata.manifest_fingerprint}`",
            f"- Source cache fingerprint: `{metadata.source_cache_fingerprint}`",
            f"- Split: `{metadata.split}`",
            "- Image count: 1000",
            "- Caption count: 5000",
            f"- Model: `{metadata.model_name}` (`{metadata.model_revision}`)",
            f"- Embedding dimension: {metadata.embedding_dimension}",
            f"- FAISS/index: `{metadata.faiss_version}` / `{metadata.index_type}` / `{metadata.metric}`",
            f"- Construction: M={metadata.m}, efConstruction={metadata.ef_construction}",
            f"- Bounded search settings: {', '.join(str(value) for value in ALLOWED_EF_SEARCH)}",
            "",
            *direction("Text to image", result.text_to_image),
            *direction("Image to text", result.image_to_text),
            "## Recommendation",
            "",
            f"**{result.recommendation}.** {result.recommendation_reason}",
            "",
            "## Limitations",
            "",
            "Timing is machine-specific and non-authoritative. It measures search only, with one",
            "warmup and five measured whole-query-batch repetitions. Mean overlap@k is the mean",
            "intersection count; reference-set recall divides that count by k.",
            "This benchmark has only 1,000 image and 5,000 caption candidates, so FlatIP may be",
            "faster. HNSW is not inherently better at this scale; these results do not establish",
            "production-scale acceleration or model quality.",
            "",
        ]
    )


def write_hnsw_outputs(
    result: HNSWBenchmarkResult,
    metadata: HNSWIndexMetadata,
    report_path: Path,
    metrics_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_hnsw_report(result, metadata), encoding="utf-8")
    payload = {
        "run_state": "success",
        "metadata": asdict(metadata),
        **asdict(result),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_hnsw_failure(report_path: Path, metrics_path: Path, state: str, detail: str) -> None:
    valid_states = {
        "not_run",
        "dependency_unavailable",
        "cache_unavailable",
        "cache_incompatible",
        "execution_failed",
    }
    if state not in valid_states:
        raise ValueError(f"unsupported HNSW report state: {state}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# FAISS HNSW Comparison Report\n\nRun state: **{state}**\n\nDetail: {detail}\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": detail, "run_state": state}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
