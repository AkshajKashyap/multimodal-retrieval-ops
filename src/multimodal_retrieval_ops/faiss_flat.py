"""Optional FAISS FlatIP indexes over cached normalized embeddings."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

from .evaluation import RetrievalMetrics, metrics_from_ranks
from .hf_clip_benchmark import HFBenchmarkCache, load_hf_benchmark_cache
from .manifest import ManifestItemV2, read_manifest

FAISS_BACKEND_NAME = "faiss-cpu"
FAISS_BACKEND_VERSION = "1"
FAISS_INDEX_TYPE = "IndexFlatIP"
FAISS_FORMAT_VERSION = 1
PREPROCESSING_VERSION = "normalized-clip-v1"


class FaissFlatError(ValueError):
    """Expected FAISS dependency, cache, index, or execution error."""


class FaissDependencyError(FaissFlatError):
    pass


class FaissCacheError(FaissFlatError):
    pass


class FaissIndexStaleError(FaissFlatError):
    pass


def faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None


def faiss_dependency_message() -> str:
    return (
        "FAISS is unavailable. Install the optional CPU backend with "
        '`python -m pip install -e ".[dev,faiss]"`.'
    )


def require_faiss() -> Any:
    if not faiss_available():
        raise FaissDependencyError(faiss_dependency_message())
    try:
        import faiss

        return faiss
    except Exception as error:
        raise FaissDependencyError(faiss_dependency_message()) from error


@dataclass(frozen=True)
class FaissIndexMetadata:
    format_version: int
    backend_name: str
    backend_version: str
    faiss_version: str
    index_type: str
    direction: str
    embedding_dimension: int
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
class FaissIndexArtifact:
    metadata: FaissIndexMetadata
    index: Any


@dataclass(frozen=True)
class DirectionComparison:
    reference_metrics: RetrievalMetrics
    faiss_metrics: RetrievalMetrics
    metric_absolute_differences: dict[str, float]
    top1_agreement_rate: float
    top5_set_agreement_rate: float
    top10_set_agreement_rate: float
    top1_tie_explained_disagreements: int
    top5_tie_explained_disagreements: int
    top10_tie_explained_disagreements: int
    maximum_score_difference: float
    query_count: int
    candidate_count: int
    correctness_gate_passed: bool


@dataclass(frozen=True)
class FaissCorrectnessResult:
    text_to_image: DirectionComparison
    image_to_text: DirectionComparison


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_normalized_matrix(vectors: list[list[float]], expected_dimension: int | None = None) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise FaissDependencyError("FAISS Flat integration requires NumPy") from error
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise FaissFlatError("embedding matrix must be non-empty and two-dimensional")
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise FaissFlatError(
            f"embedding dimension {matrix.shape[1]} does not match expected {expected_dimension}"
        )
    if not np.isfinite(matrix).all():
        raise FaissFlatError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise FaissFlatError("IndexFlatIP requires L2-normalized cached embeddings")
    return matrix


def build_flat_ip_index(vectors: list[list[float]], expected_dimension: int | None = None) -> Any:
    faiss = require_faiss()
    matrix = _as_normalized_matrix(vectors, expected_dimension)
    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    return index


def ordered_embeddings(embeddings: dict[str, list[float]]) -> tuple[list[str], list[list[float]]]:
    candidate_ids = sorted(embeddings)
    return candidate_ids, [embeddings[candidate_id] for candidate_id in candidate_ids]


def make_index_metadata(
    cache: HFBenchmarkCache,
    *,
    direction: str,
    candidate_ids: list[str],
    source_cache_fingerprint: str,
    faiss_version: str,
) -> FaissIndexMetadata:
    metadata = cache.metadata
    return FaissIndexMetadata(
        format_version=FAISS_FORMAT_VERSION,
        backend_name=FAISS_BACKEND_NAME,
        backend_version=FAISS_BACKEND_VERSION,
        faiss_version=faiss_version,
        index_type=FAISS_INDEX_TYPE,
        direction=direction,
        embedding_dimension=metadata.embedding_dimension,
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


def index_metadata_is_stale(actual: FaissIndexMetadata, expected: FaissIndexMetadata) -> bool:
    return actual != expected


def save_faiss_artifact(artifact: FaissIndexArtifact, index_path: Path, metadata_path: Path) -> None:
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


def load_faiss_artifact(
    index_path: Path,
    metadata_path: Path,
    expected: FaissIndexMetadata | None = None,
) -> FaissIndexArtifact:
    faiss = require_faiss()
    if not index_path.is_file() or not metadata_path.is_file():
        raise FaissIndexStaleError(
            f"FAISS artifact is missing: {index_path} or {metadata_path}"
        )
    metadata = FaissIndexMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
    if expected is not None and index_metadata_is_stale(metadata, expected):
        raise FaissIndexStaleError(f"FAISS metadata is stale for {metadata.direction}")
    index = faiss.read_index(str(index_path))
    if index.d != metadata.embedding_dimension or index.ntotal != metadata.candidate_count:
        raise FaissIndexStaleError("FAISS binary shape does not match companion metadata")
    return FaissIndexArtifact(metadata, index)


def build_flickr8k_flat_artifacts(
    cache_path: Path, artifacts_dir: Path
) -> tuple[FaissIndexArtifact, FaissIndexArtifact]:
    faiss = require_faiss()
    if not cache_path.is_file():
        raise FaissCacheError(f"required embedding cache is missing: {cache_path}")
    cache_fingerprint = file_sha256(cache_path)
    cache = load_hf_benchmark_cache(cache_path)
    if (
        cache.metadata.split != "test"
        or cache.metadata.max_images is not None
        or cache.metadata.image_count != 1000
        or cache.metadata.caption_count != 5000
    ):
        raise FaissCacheError(
            "embedding cache is incompatible; expected full test split with 1000 images and 5000 captions"
        )
    image_ids, image_vectors = ordered_embeddings(cache.image_embeddings)
    caption_ids, caption_vectors = ordered_embeddings(cache.caption_embeddings)
    artifacts = []
    for direction, ids, vectors in (
        ("text_to_image", image_ids, image_vectors),
        ("image_to_text", caption_ids, caption_vectors),
    ):
        metadata = make_index_metadata(
            cache,
            direction=direction,
            candidate_ids=ids,
            source_cache_fingerprint=cache_fingerprint,
            faiss_version=faiss.__version__,
        )
        artifact = FaissIndexArtifact(
            metadata,
            build_flat_ip_index(vectors, cache.metadata.embedding_dimension),
        )
        save_faiss_artifact(
            artifact,
            artifacts_dir / f"{direction}.faiss",
            artifacts_dir / f"{direction}.json",
        )
        artifacts.append(artifact)
    return artifacts[0], artifacts[1]


def expected_artifact_metadata(
    cache: HFBenchmarkCache, cache_path: Path, direction: str
) -> FaissIndexMetadata:
    faiss = require_faiss()
    embeddings = (
        cache.image_embeddings if direction == "text_to_image" else cache.caption_embeddings
    )
    ids, _ = ordered_embeddings(embeddings)
    return make_index_metadata(
        cache,
        direction=direction,
        candidate_ids=ids,
        source_cache_fingerprint=file_sha256(cache_path),
        faiss_version=faiss.__version__,
    )


def load_flickr8k_artifacts(
    cache_path: Path, artifacts_dir: Path
) -> tuple[HFBenchmarkCache, FaissIndexArtifact, FaissIndexArtifact]:
    if not cache_path.is_file():
        raise FaissCacheError(f"required embedding cache is missing: {cache_path}")
    cache = load_hf_benchmark_cache(cache_path)
    artifacts = []
    for direction in ("text_to_image", "image_to_text"):
        expected = expected_artifact_metadata(cache, cache_path, direction)
        artifacts.append(
            load_faiss_artifact(
                artifacts_dir / f"{direction}.faiss",
                artifacts_dir / f"{direction}.json",
                expected,
            )
        )
    return cache, artifacts[0], artifacts[1]


def _metric_differences(reference: RetrievalMetrics, actual: RetrievalMetrics) -> dict[str, float]:
    return {
        field: abs(float(getattr(reference, field)) - float(getattr(actual, field)))
        for field in (
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "median_rank",
            "mean_rank",
            "query_count",
        )
    }


def _compare_direction(
    query_vectors: list[list[float]],
    candidate_vectors: list[list[float]],
    relevant_candidate_indices: list[set[int]],
    index: Any,
    tie_tolerance: float = 1e-6,
) -> DirectionComparison:
    import numpy as np

    queries = _as_normalized_matrix(query_vectors)
    candidates = _as_normalized_matrix(candidate_vectors, queries.shape[1])
    similarity = queries @ candidates.T
    reference_order = np.argsort(-similarity, axis=1, kind="stable")
    faiss_scores, faiss_order = index.search(queries, candidates.shape[0])

    def ranks(order: Any) -> list[int]:
        return [
            min(position for position, candidate in enumerate(row, 1) if candidate in relevant)
            for row, relevant in zip(order.tolist(), relevant_candidate_indices, strict=True)
        ]

    reference_metrics = metrics_from_ranks(ranks(reference_order))
    faiss_metrics = metrics_from_ranks(ranks(faiss_order))
    top_rates: dict[int, float] = {}
    tie_counts: dict[int, int] = {}
    for k in (1, 5, 10):
        width = min(k, candidates.shape[0])
        agreements = 0
        tie_explained = 0
        for query_index in range(queries.shape[0]):
            reference_set = set(reference_order[query_index, :width].tolist())
            faiss_set = set(faiss_order[query_index, :width].tolist())
            if reference_set == faiss_set:
                agreements += 1
                continue
            boundary = similarity[query_index, reference_order[query_index, width - 1]]
            differing = reference_set.symmetric_difference(faiss_set)
            if differing and all(
                math.isclose(
                    float(similarity[query_index, candidate]),
                    float(boundary),
                    abs_tol=tie_tolerance,
                )
                for candidate in differing
            ):
                tie_explained += 1
        top_rates[k] = agreements / queries.shape[0]
        tie_counts[k] = tie_explained
    gathered_reference_scores = np.take_along_axis(similarity, faiss_order, axis=1)
    maximum_score_difference = float(np.max(np.abs(gathered_reference_scores - faiss_scores)))
    differences = _metric_differences(reference_metrics, faiss_metrics)
    all_top_disagreements_explained = all(
        round((1.0 - top_rates[k]) * queries.shape[0]) <= tie_counts[k] for k in (1, 5, 10)
    )
    gate = max(differences.values()) <= 1e-12 and all_top_disagreements_explained
    return DirectionComparison(
        reference_metrics=reference_metrics,
        faiss_metrics=faiss_metrics,
        metric_absolute_differences=differences,
        top1_agreement_rate=top_rates[1],
        top5_set_agreement_rate=top_rates[5],
        top10_set_agreement_rate=top_rates[10],
        top1_tie_explained_disagreements=tie_counts[1],
        top5_tie_explained_disagreements=tie_counts[5],
        top10_tie_explained_disagreements=tie_counts[10],
        maximum_score_difference=maximum_score_difference,
        query_count=queries.shape[0],
        candidate_count=candidates.shape[0],
        correctness_gate_passed=gate,
    )


def evaluate_flickr8k_faiss(
    cache: HFBenchmarkCache,
    text_to_image: FaissIndexArtifact,
    image_to_text: FaissIndexArtifact,
    manifest_path: Path,
) -> FaissCorrectnessResult:
    raw_rows = read_manifest(manifest_path)
    rows = [
        row for row in raw_rows if isinstance(row, ManifestItemV2) and row.split == "test"
    ]
    image_ids = text_to_image.metadata.candidate_ids
    caption_ids = image_to_text.metadata.candidate_ids
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    caption_index = {caption_id: index for index, caption_id in enumerate(caption_ids)}
    caption_targets = {row.caption_id: row.image_id for row in rows}
    if set(caption_targets) != set(caption_ids):
        raise FaissCacheError("manifest caption IDs do not match cached caption embeddings")
    text_relevant = [{image_index[caption_targets[caption_id]]} for caption_id in caption_ids]
    captions_by_image: dict[str, set[int]] = {image_id: set() for image_id in image_ids}
    for caption_id, image_id in caption_targets.items():
        captions_by_image[image_id].add(caption_index[caption_id])
    return FaissCorrectnessResult(
        text_to_image=_compare_direction(
            [cache.caption_embeddings[caption_id] for caption_id in caption_ids],
            [cache.image_embeddings[image_id] for image_id in image_ids],
            text_relevant,
            text_to_image.index,
        ),
        image_to_text=_compare_direction(
            [cache.image_embeddings[image_id] for image_id in image_ids],
            [cache.caption_embeddings[caption_id] for caption_id in caption_ids],
            [captions_by_image[image_id] for image_id in image_ids],
            image_to_text.index,
        ),
    )


def search_cached_embedding(
    query_id: str,
    query_embeddings: dict[str, list[float]],
    artifact: FaissIndexArtifact,
    k: int,
) -> list[dict[str, float | str]]:
    if query_id not in query_embeddings:
        raise FaissCacheError(f"cached query embedding is missing: {query_id}")
    if k <= 0:
        raise FaissFlatError("k must be positive")
    query = _as_normalized_matrix(
        [query_embeddings[query_id]], artifact.metadata.embedding_dimension
    )
    scores, indices = artifact.index.search(query, min(k, artifact.metadata.candidate_count))
    return [
        {"candidate_id": artifact.metadata.candidate_ids[index], "score": float(score)}
        for score, index in zip(scores[0], indices[0], strict=True)
    ]


def render_correctness_report(
    result: FaissCorrectnessResult,
    metadata: FaissIndexMetadata,
) -> str:
    def direction_lines(name: str, comparison: DirectionComparison) -> list[str]:
        return [
            f"## {name}",
            "",
            f"- Queries: {comparison.query_count}",
            f"- Candidates: {comparison.candidate_count}",
            f"- Correctness gate: **{'pass' if comparison.correctness_gate_passed else 'fail'}**",
            f"- Top-1 agreement: {comparison.top1_agreement_rate:.6f}",
            f"- Top-5 set agreement: {comparison.top5_set_agreement_rate:.6f}",
            f"- Top-10 set agreement: {comparison.top10_set_agreement_rate:.6f}",
            f"- Maximum score difference: {comparison.maximum_score_difference:.10f}",
            f"- Tie-explained disagreements (1/5/10): "
            f"{comparison.top1_tie_explained_disagreements}/"
            f"{comparison.top5_tie_explained_disagreements}/"
            f"{comparison.top10_tie_explained_disagreements}",
            "",
            "| Backend | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _metrics_row("Reference", comparison.reference_metrics),
            _metrics_row("FAISS FlatIP", comparison.faiss_metrics),
            "",
        ]

    return "\n".join(
        [
            "# FAISS Flat Correctness Report",
            "",
            "Run state: **success**",
            "",
            f"- FAISS version: `{metadata.faiss_version}`",
            f"- Index type: `{metadata.index_type}`",
            f"- Source cache fingerprint: `{metadata.source_cache_fingerprint}`",
            f"- Dataset fingerprint: `{metadata.dataset_fingerprint}`",
            f"- Manifest fingerprint: `{metadata.manifest_fingerprint}`",
            f"- Split: `{metadata.split}`",
            f"- Model: `{metadata.model_name}`",
            f"- Model revision: `{metadata.model_revision}`",
            f"- Embedding dimension: {metadata.embedding_dimension}",
            "- Image count: 1000",
            "- Caption count: 5000",
            "",
            *direction_lines("Text to image", result.text_to_image),
            *direction_lines("Image to text", result.image_to_text),
            "## Tie handling and limitations",
            "",
            "Set disagreements are classified as tie-explained only when every differing candidate",
            "is within the score tolerance of the reference top-k boundary.",
            "IndexFlatIP is exact; this validates retrieval-backend correctness, not model quality.",
            "No latency claims, approximate indexes, GPU FAISS, or neural inference are included.",
            "",
        ]
    )


def _metrics_row(name: str, metrics: RetrievalMetrics) -> str:
    return (
        f"| {name} | {metrics.recall_at_1:.4f} | {metrics.recall_at_5:.4f} | "
        f"{metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | {metrics.median_rank:.2f} | "
        f"{metrics.mean_rank:.2f} |"
    )


def write_correctness_outputs(
    result: FaissCorrectnessResult,
    metadata: FaissIndexMetadata,
    report_path: Path,
    metrics_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_correctness_report(result, metadata), encoding="utf-8")
    metrics = {
        "run_state": "success",
        "faiss_version": metadata.faiss_version,
        "index_type": metadata.index_type,
        "source_cache_fingerprint": metadata.source_cache_fingerprint,
        "dataset_fingerprint": metadata.dataset_fingerprint,
        "manifest_fingerprint": metadata.manifest_fingerprint,
        "split": metadata.split,
        "model_name": metadata.model_name,
        "model_revision": metadata.model_revision,
        "embedding_dimension": metadata.embedding_dimension,
        "image_count": 1000,
        "caption_count": 5000,
        "text_to_image": asdict(result.text_to_image),
        "image_to_text": asdict(result.image_to_text),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_faiss_failure(report_path: Path, metrics_path: Path, state: str, detail: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# FAISS Flat Correctness Report\n\nRun state: **{state}**\n\nDetail: {detail}\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": detail, "run_state": state}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
