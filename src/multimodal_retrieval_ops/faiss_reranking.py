"""Bounded exact reranking over persisted Flickr8k HNSW candidates."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

from .evaluation import RetrievalMetrics, metrics_from_ranks
from .faiss_flat import (
    FaissCacheError,
    FaissFlatError,
    FaissIndexArtifact,
    FaissIndexStaleError,
    _as_normalized_matrix,
    load_flickr8k_artifacts,
)
from .faiss_hnsw import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    HNSWIndexArtifact,
    load_flickr8k_hnsw_artifacts,
    set_ef_search,
)
from .hf_clip_benchmark import HFBenchmarkCache
from .manifest import ManifestItemV2, read_manifest

CANDIDATE_K = 50
EF_SEARCH = 64
TIMING_REPETITIONS = 3
RERANKING_FORMAT_VERSION = 1
SUPPORTED_RUN_STATES = {
    "success",
    "not_run",
    "dependency_unavailable",
    "cache_unavailable",
    "artifact_unavailable",
    "artifact_incompatible",
    "execution_failed",
}


class FaissRerankingError(FaissFlatError):
    """Expected persisted-artifact or exact-reranking error."""


class RerankingArtifactUnavailableError(FaissRerankingError):
    pass


class RerankingArtifactIncompatibleError(FaissIndexStaleError, FaissRerankingError):
    pass


@dataclass(frozen=True)
class TimingResult:
    warmup_count: int
    measured_repetitions: int
    median_batch_seconds: float
    approximate_queries_per_second: float


@dataclass(frozen=True)
class ShortlistCoverage:
    flat_top1_present_rate: float
    mean_flat_top5_fraction: float
    mean_flat_top10_fraction: float
    complete_flat_top10_query_count: int
    complete_flat_top10_query_rate: float
    missing_flat_top1_query_count: int


@dataclass(frozen=True)
class RankedAgreement:
    top1_agreement: float
    top5_set_agreement: float
    top10_set_agreement: float
    mean_overlap_at_5: float
    mean_overlap_at_10: float


@dataclass(frozen=True)
class RankChanges:
    improved_query_count: int
    unchanged_query_count: int
    worsened_query_count: int
    mean_rank_change: float
    median_rank_change: float
    largest_improvement: int
    largest_regression: int


@dataclass(frozen=True)
class DirectionTiming:
    raw_hnsw: TimingResult
    exact_shortlist_rescoring: TimingResult
    combined_hnsw_reranking: TimingResult
    flat_ip: TimingResult


@dataclass(frozen=True)
class DirectionRerankingResult:
    query_count: int
    candidate_count: int
    shortlist_coverage: ShortlistCoverage
    flat_metrics: RetrievalMetrics
    raw_hnsw_metrics: RetrievalMetrics
    reranked_metrics: RetrievalMetrics
    raw_agreement: RankedAgreement
    reranked_agreement: RankedAgreement
    rank_changes: RankChanges
    maximum_exact_score_difference: float
    timing: DirectionTiming


@dataclass(frozen=True)
class PromotionGate:
    approved: bool
    flat_top10_coverage_passed: bool
    top10_agreement_improved: bool
    mrr_preserved: bool
    recall_at_10_preserved: bool
    artifact_compatibility_passed: bool
    rejected_adapter_embeddings_used: bool
    recommendation: str


@dataclass(frozen=True)
class RerankingMetadata:
    format_version: int
    model_name: str
    model_revision: str
    embedding_dimension: int
    dataset_fingerprint: str
    manifest_fingerprint: str
    source_cache_fingerprint: str
    split: str
    image_count: int
    caption_count: int
    flat_index_type: str
    hnsw_index_type: str
    hnsw_m: int
    hnsw_ef_construction: int
    ef_search: int
    candidate_k: int
    artifact_compatibility: str
    rejected_adapter_embeddings_used: bool


@dataclass(frozen=True)
class RerankingEvaluationResult:
    text_to_image: DirectionRerankingResult
    image_to_text: DirectionRerankingResult
    promotion_gate: PromotionGate


@dataclass(frozen=True)
class RerankingArtifacts:
    cache: HFBenchmarkCache
    flat_text_to_image: FaissIndexArtifact
    flat_image_to_text: FaissIndexArtifact
    hnsw_text_to_image: HNSWIndexArtifact
    hnsw_image_to_text: HNSWIndexArtifact
    metadata: RerankingMetadata


def validate_fixed_configuration(candidate_k: int, ef_search: int) -> None:
    if candidate_k != CANDIDATE_K:
        raise FaissRerankingError(f"candidate_k is fixed at {CANDIDATE_K}")
    if ef_search != EF_SEARCH:
        raise FaissRerankingError(f"efSearch is fixed at {EF_SEARCH}")


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise FaissRerankingError("exact FAISS reranking requires NumPy") from error
    return np


def exact_rescore_shortlist(
    query_vector: list[float],
    candidate_ids: list[str],
    candidate_embeddings: dict[str, list[float]],
) -> list[dict[str, float | int | str]]:
    """Exactly rescore a unique shortlist, preserving shortlist order for ties."""
    if not candidate_ids:
        raise FaissRerankingError("candidate shortlist must not be empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise FaissRerankingError("candidate shortlist contains duplicate IDs")
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidate_embeddings]
    if missing:
        raise FaissRerankingError(f"cached candidate embedding is missing: {missing[0]}")
    try:
        query = _as_normalized_matrix([query_vector])
        candidates = _as_normalized_matrix(
            [candidate_embeddings[candidate_id] for candidate_id in candidate_ids],
            int(query.shape[1]),
        )
    except FaissFlatError as error:
        raise FaissRerankingError(str(error)) from error
    scores = candidates @ query[0]
    if not _numpy().isfinite(scores).all():
        raise FaissRerankingError("exact reranking produced a non-finite score")
    order = sorted(range(len(candidate_ids)), key=lambda index: (-float(scores[index]), index))
    return [
        {
            "candidate_id": candidate_ids[index],
            "score": float(scores[index]),
            "shortlist_rank": index + 1,
        }
        for index in order
    ]


def _compatibility_value(metadata: Any, name: str) -> Any:
    return getattr(metadata, name)


def _validate_artifact_pair(flat: FaissIndexArtifact, hnsw: HNSWIndexArtifact) -> None:
    names = (
        "direction",
        "embedding_dimension",
        "model_name",
        "model_revision",
        "dataset_fingerprint",
        "manifest_fingerprint",
        "split",
        "candidate_count",
        "candidate_ids",
        "preprocessing_version",
        "source_cache_fingerprint",
    )
    for name in names:
        if _compatibility_value(flat.metadata, name) != _compatibility_value(hnsw.metadata, name):
            raise RerankingArtifactIncompatibleError(
                f"FlatIP and HNSW {flat.metadata.direction} metadata differ: {name}"
            )
    if hnsw.metadata.m != HNSW_M or hnsw.metadata.ef_construction != HNSW_EF_CONSTRUCTION:
        raise RerankingArtifactIncompatibleError("persisted HNSW construction metadata is incompatible")


def load_reranking_artifacts(
    cache_path: Path,
    flat_artifacts_dir: Path,
    hnsw_artifacts_dir: Path,
    *,
    candidate_k: int = CANDIDATE_K,
    ef_search: int = EF_SEARCH,
) -> RerankingArtifacts:
    validate_fixed_configuration(candidate_k, ef_search)
    try:
        cache, flat_text, flat_image = load_flickr8k_artifacts(cache_path, flat_artifacts_dir)
        hnsw_cache, hnsw_text, hnsw_image = load_flickr8k_hnsw_artifacts(
            cache_path, hnsw_artifacts_dir
        )
    except FaissIndexStaleError as error:
        if "missing" in str(error).lower():
            raise RerankingArtifactUnavailableError(str(error)) from error
        raise RerankingArtifactIncompatibleError(str(error)) from error
    if cache.metadata != hnsw_cache.metadata:
        raise RerankingArtifactIncompatibleError("FlatIP and HNSW caches have different metadata")
    _validate_artifact_pair(flat_text, hnsw_text)
    _validate_artifact_pair(flat_image, hnsw_image)
    if flat_text.metadata.candidate_ids != sorted(cache.image_embeddings):
        raise RerankingArtifactIncompatibleError("text-to-image candidate ordering is incompatible")
    if flat_image.metadata.candidate_ids != sorted(cache.caption_embeddings):
        raise RerankingArtifactIncompatibleError("image-to-text candidate ordering is incompatible")
    if candidate_k > min(flat_text.metadata.candidate_count, flat_image.metadata.candidate_count):
        raise RerankingArtifactIncompatibleError("candidate_k exceeds a persisted candidate count")
    set_ef_search(hnsw_text.index, ef_search)
    set_ef_search(hnsw_image.index, ef_search)
    metadata = RerankingMetadata(
        format_version=RERANKING_FORMAT_VERSION,
        model_name=cache.metadata.model_name,
        model_revision=cache.metadata.model_revision,
        embedding_dimension=cache.metadata.embedding_dimension,
        dataset_fingerprint=cache.metadata.dataset_fingerprint,
        manifest_fingerprint=cache.metadata.manifest_fingerprint,
        source_cache_fingerprint=flat_text.metadata.source_cache_fingerprint,
        split=cache.metadata.split,
        image_count=cache.metadata.image_count,
        caption_count=cache.metadata.caption_count,
        flat_index_type=flat_text.metadata.index_type,
        hnsw_index_type=hnsw_text.metadata.index_type,
        hnsw_m=hnsw_text.metadata.m,
        hnsw_ef_construction=hnsw_text.metadata.ef_construction,
        ef_search=ef_search,
        candidate_k=candidate_k,
        artifact_compatibility="passed",
        rejected_adapter_embeddings_used=False,
    )
    return RerankingArtifacts(cache, flat_text, flat_image, hnsw_text, hnsw_image, metadata)


def search_reranked_embedding(
    query_id: str,
    query_embeddings: dict[str, list[float]],
    candidate_embeddings: dict[str, list[float]],
    hnsw: HNSWIndexArtifact,
    *,
    k: int = 10,
    candidate_k: int = CANDIDATE_K,
    ef_search: int = EF_SEARCH,
) -> list[dict[str, float | int | str]]:
    validate_fixed_configuration(candidate_k, ef_search)
    if query_id not in query_embeddings:
        raise FaissCacheError(f"cached query embedding is missing: {query_id}")
    if k <= 0 or k > candidate_k:
        raise FaissRerankingError(f"k must be between 1 and {candidate_k}")
    set_ef_search(hnsw.index, ef_search)
    query = _as_normalized_matrix([query_embeddings[query_id]], hnsw.metadata.embedding_dimension)
    _, indices = hnsw.index.search(query, candidate_k)
    shortlist = [hnsw.metadata.candidate_ids[int(index)] for index in indices[0] if index >= 0]
    if len(shortlist) != candidate_k:
        raise FaissRerankingError(f"HNSW returned {len(shortlist)} candidates; expected {candidate_k}")
    return exact_rescore_shortlist(query_embeddings[query_id], shortlist, candidate_embeddings)[:k]


def shortlist_coverage(reference_order: Any, shortlist_order: Any) -> ShortlistCoverage:
    if reference_order.shape[0] == 0 or reference_order.shape[0] != shortlist_order.shape[0]:
        raise FaissRerankingError("reference and shortlist query counts must match and be non-zero")
    top1_present = []
    top5_fractions = []
    top10_fractions = []
    complete_top10 = []
    for reference, shortlist in zip(reference_order, shortlist_order, strict=True):
        candidates = set(int(value) for value in shortlist)
        top1_present.append(int(reference[0]) in candidates)
        top5 = {int(value) for value in reference[:5]}
        top10 = {int(value) for value in reference[:10]}
        top5_fractions.append(len(top5 & candidates) / len(top5))
        top10_fractions.append(len(top10 & candidates) / len(top10))
        complete_top10.append(top10 <= candidates)
    count = len(top1_present)
    return ShortlistCoverage(
        sum(top1_present) / count,
        statistics.mean(top5_fractions),
        statistics.mean(top10_fractions),
        sum(complete_top10),
        sum(complete_top10) / count,
        count - sum(top1_present),
    )


def ranked_agreement(reference_order: Any, actual_order: Any) -> RankedAgreement:
    if reference_order.shape[0] == 0 or reference_order.shape[0] != actual_order.shape[0]:
        raise FaissRerankingError("ranked result query counts must match and be non-zero")
    top1 = []
    exact5 = []
    exact10 = []
    overlap5 = []
    overlap10 = []
    for reference, actual in zip(reference_order, actual_order, strict=True):
        top1.append(int(reference[0]) == int(actual[0]))
        ref5, actual5 = set(reference[:5]), set(actual[:5])
        ref10, actual10 = set(reference[:10]), set(actual[:10])
        exact5.append(ref5 == actual5)
        exact10.append(ref10 == actual10)
        overlap5.append(len(ref5 & actual5))
        overlap10.append(len(ref10 & actual10))
    count = len(top1)
    return RankedAgreement(
        sum(top1) / count,
        sum(exact5) / count,
        sum(exact10) / count,
        statistics.mean(overlap5),
        statistics.mean(overlap10),
    )


def relevant_ranks(order: Any, relevant: list[set[int]], missing_rank: int) -> list[int]:
    ranks = []
    for row, targets in zip(order.tolist(), relevant, strict=True):
        rank = next(
            (position for position, candidate in enumerate(row, 1) if candidate in targets),
            missing_rank,
        )
        ranks.append(rank)
    return ranks


def classify_rank_changes(raw_ranks: list[int], reranked_ranks: list[int]) -> RankChanges:
    if not raw_ranks or len(raw_ranks) != len(reranked_ranks):
        raise FaissRerankingError("rank lists must match and be non-empty")
    changes = [raw - reranked for raw, reranked in zip(raw_ranks, reranked_ranks, strict=True)]
    return RankChanges(
        sum(value > 0 for value in changes),
        sum(value == 0 for value in changes),
        sum(value < 0 for value in changes),
        statistics.mean(changes),
        float(statistics.median(changes)),
        max(changes),
        min(changes),
    )


def _rerank_orders(
    query_matrix: Any, candidate_matrix: Any, shortlist_order: Any
) -> tuple[Any, float]:
    np = _numpy()
    output = np.empty_like(shortlist_order)
    maximum_difference = 0.0
    for row_index, (query, candidates) in enumerate(
        zip(query_matrix, shortlist_order, strict=True)
    ):
        unique = np.unique(candidates)
        if len(unique) != len(candidates):
            raise FaissRerankingError("HNSW returned duplicate candidate indices")
        vectors = candidate_matrix[candidates]
        scores = vectors @ query
        if not np.isfinite(scores).all():
            raise FaissRerankingError("exact reranking produced a non-finite score")
        direct = np.einsum("ij,j->i", vectors, query, dtype="float32")
        maximum_difference = max(maximum_difference, float(np.max(np.abs(scores - direct))))
        local = np.argsort(-scores, kind="stable")
        output[row_index] = candidates[local]
    return output, maximum_difference


def _measure(operation: Callable[[], Any], query_count: int) -> TimingResult:
    operation()  # one warmup
    samples = []
    for _ in range(TIMING_REPETITIONS):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    return TimingResult(1, TIMING_REPETITIONS, median, query_count / median)


def _evaluate_direction(
    queries: list[list[float]],
    candidates: list[list[float]],
    relevant: list[set[int]],
    flat: FaissIndexArtifact,
    hnsw: HNSWIndexArtifact,
) -> DirectionRerankingResult:
    query_matrix = _as_normalized_matrix(queries, flat.metadata.embedding_dimension)
    candidate_matrix = _as_normalized_matrix(candidates, flat.metadata.embedding_dimension)
    candidate_count = flat.metadata.candidate_count
    _, flat_order = flat.index.search(query_matrix, candidate_count)
    set_ef_search(hnsw.index, EF_SEARCH)
    _, raw_order = hnsw.index.search(query_matrix, CANDIDATE_K)
    if raw_order.shape != (len(queries), CANDIDATE_K):
        raise FaissRerankingError("HNSW did not return the fixed top-50 shortlist")
    reranked_order, maximum_difference = _rerank_orders(
        query_matrix, candidate_matrix, raw_order
    )
    missing_rank = candidate_count + 1
    flat_ranks = relevant_ranks(flat_order, relevant, missing_rank)
    raw_ranks = relevant_ranks(raw_order, relevant, missing_rank)
    reranked_ranks = relevant_ranks(reranked_order, relevant, missing_rank)

    def raw_search() -> Any:
        return hnsw.index.search(query_matrix, CANDIDATE_K)

    def rescore() -> Any:
        return _rerank_orders(query_matrix, candidate_matrix, raw_order)

    def combined() -> Any:
        _, order = hnsw.index.search(query_matrix, CANDIDATE_K)
        return _rerank_orders(query_matrix, candidate_matrix, order)

    def flat_search() -> Any:
        return flat.index.search(query_matrix, CANDIDATE_K)

    return DirectionRerankingResult(
        query_count=len(queries),
        candidate_count=candidate_count,
        shortlist_coverage=shortlist_coverage(flat_order, raw_order),
        flat_metrics=metrics_from_ranks(flat_ranks),
        raw_hnsw_metrics=metrics_from_ranks(raw_ranks),
        reranked_metrics=metrics_from_ranks(reranked_ranks),
        raw_agreement=ranked_agreement(flat_order, raw_order),
        reranked_agreement=ranked_agreement(flat_order, reranked_order),
        rank_changes=classify_rank_changes(raw_ranks, reranked_ranks),
        maximum_exact_score_difference=maximum_difference,
        timing=DirectionTiming(
            raw_hnsw=_measure(raw_search, len(queries)),
            exact_shortlist_rescoring=_measure(rescore, len(queries)),
            combined_hnsw_reranking=_measure(combined, len(queries)),
            flat_ip=_measure(flat_search, len(queries)),
        ),
    )


def evaluate_promotion_gate(
    text: DirectionRerankingResult,
    image: DirectionRerankingResult,
    *,
    artifact_compatibility_passed: bool = True,
    rejected_adapter_embeddings_used: bool = False,
) -> PromotionGate:
    directions = (text, image)
    coverage = all(value.shortlist_coverage.mean_flat_top10_fraction >= 0.99 for value in directions)
    agreement = all(
        value.reranked_agreement.top10_set_agreement > value.raw_agreement.top10_set_agreement
        for value in directions
    )
    mrr = all(
        value.reranked_metrics.mrr >= value.raw_hnsw_metrics.mrr - 0.001
        for value in directions
    )
    recall = all(
        value.reranked_metrics.recall_at_10
        >= value.raw_hnsw_metrics.recall_at_10 - 0.001
        for value in directions
    )
    approved = all((coverage, agreement, mrr, recall, artifact_compatibility_passed)) and not (
        rejected_adapter_embeddings_used
    )
    recommendation = (
        "Approve later service-integration consideration; this milestone does not change serving."
        if approved
        else "Keep the existing raw FlatIP and HNSW serving behavior."
    )
    return PromotionGate(
        approved,
        coverage,
        agreement,
        mrr,
        recall,
        artifact_compatibility_passed,
        rejected_adapter_embeddings_used,
        recommendation,
    )


def evaluate_hnsw_reranking(
    cache_path: Path,
    manifest_path: Path,
    flat_artifacts_dir: Path,
    hnsw_artifacts_dir: Path,
    *,
    candidate_k: int = CANDIDATE_K,
    ef_search: int = EF_SEARCH,
) -> tuple[RerankingEvaluationResult, RerankingMetadata]:
    artifacts = load_reranking_artifacts(
        cache_path,
        flat_artifacts_dir,
        hnsw_artifacts_dir,
        candidate_k=candidate_k,
        ef_search=ef_search,
    )
    rows = [
        row
        for row in read_manifest(manifest_path)
        if isinstance(row, ManifestItemV2) and row.split == "test"
    ]
    caption_to_image = {row.caption_id: row.image_id for row in rows}
    image_ids = artifacts.flat_text_to_image.metadata.candidate_ids
    caption_ids = artifacts.flat_image_to_text.metadata.candidate_ids
    if set(caption_to_image) != set(caption_ids):
        raise FaissCacheError("manifest caption IDs do not match persisted candidate IDs")
    image_positions = {value: index for index, value in enumerate(image_ids)}
    caption_positions = {value: index for index, value in enumerate(caption_ids)}
    text_relevant = [{image_positions[caption_to_image[value]]} for value in caption_ids]
    captions_by_image = {image_id: set() for image_id in image_ids}
    for caption_id, image_id in caption_to_image.items():
        captions_by_image[image_id].add(caption_positions[caption_id])
    text = _evaluate_direction(
        [artifacts.cache.caption_embeddings[value] for value in caption_ids],
        [artifacts.cache.image_embeddings[value] for value in image_ids],
        text_relevant,
        artifacts.flat_text_to_image,
        artifacts.hnsw_text_to_image,
    )
    image = _evaluate_direction(
        [artifacts.cache.image_embeddings[value] for value in image_ids],
        [artifacts.cache.caption_embeddings[value] for value in caption_ids],
        [captions_by_image[value] for value in image_ids],
        artifacts.flat_image_to_text,
        artifacts.hnsw_image_to_text,
    )
    gate = evaluate_promotion_gate(text, image)
    return RerankingEvaluationResult(text, image, gate), artifacts.metadata


def _metrics_line(metrics: RetrievalMetrics) -> str:
    return (
        f"{metrics.recall_at_1:.4f}/{metrics.recall_at_5:.4f}/"
        f"{metrics.recall_at_10:.4f}/{metrics.mrr:.4f}/"
        f"{metrics.median_rank:.2f}/{metrics.mean_rank:.4f}"
    )


def render_reranking_report(
    result: RerankingEvaluationResult, metadata: RerankingMetadata
) -> str:
    def direction(name: str, value: DirectionRerankingResult) -> list[str]:
        coverage = value.shortlist_coverage
        changes = value.rank_changes
        return [
            f"## {name}",
            "",
            f"- Queries / candidates: {value.query_count} / {value.candidate_count}",
            f"- Flat top-1 in shortlist: {coverage.flat_top1_present_rate:.4f} "
            f"(missing {coverage.missing_flat_top1_query_count})",
            f"- Mean Flat top-5 / top-10 shortlist fraction: "
            f"{coverage.mean_flat_top5_fraction:.4f} / {coverage.mean_flat_top10_fraction:.4f}",
            f"- Complete Flat top-10 shortlists: {coverage.complete_flat_top10_query_count} "
            f"({coverage.complete_flat_top10_query_rate:.4f})",
            "",
            "| Method | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *(
                f"| {method} | {metrics.recall_at_1:.4f} | {metrics.recall_at_5:.4f} | "
                f"{metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | "
                f"{metrics.median_rank:.2f} | {metrics.mean_rank:.4f} |"
                for method, metrics in (
                    ("FlatIP", value.flat_metrics),
                    ("Raw HNSW", value.raw_hnsw_metrics),
                    ("Reranked HNSW", value.reranked_metrics),
                )
            ),
            "",
            "| Method | Top-1 agreement | Top-5 set | Top-10 set | Mean overlap@5 | Mean overlap@10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *(
                f"| {method} | {agreement.top1_agreement:.4f} | "
                f"{agreement.top5_set_agreement:.4f} | {agreement.top10_set_agreement:.4f} | "
                f"{agreement.mean_overlap_at_5:.4f} | {agreement.mean_overlap_at_10:.4f} |"
                for method, agreement in (
                    ("Raw HNSW", value.raw_agreement),
                    ("Reranked HNSW", value.reranked_agreement),
                )
            ),
            "",
            f"- Target-rank changes improved / unchanged / worsened: "
            f"{changes.improved_query_count} / {changes.unchanged_query_count} / "
            f"{changes.worsened_query_count}",
            f"- Mean / median rank change (positive is improvement): "
            f"{changes.mean_rank_change:.4f} / {changes.median_rank_change:.2f}",
            f"- Largest improvement / regression: "
            f"{changes.largest_improvement} / {changes.largest_regression}",
            f"- Maximum exact-score verification difference: "
            f"{value.maximum_exact_score_difference:.10g}",
            "",
            "| Timed method | Median batch (s) | Approx. queries/s |",
            "| --- | ---: | ---: |",
            *(
                f"| {method} | {timing.median_batch_seconds:.6f} | "
                f"{timing.approximate_queries_per_second:.2f} |"
                for method, timing in (
                    ("Raw HNSW", value.timing.raw_hnsw),
                    ("Exact shortlist rescoring", value.timing.exact_shortlist_rescoring),
                    ("Combined HNSW + reranking", value.timing.combined_hnsw_reranking),
                    ("FlatIP", value.timing.flat_ip),
                )
            ),
            "",
        ]

    return "\n".join(
        [
            "# HNSW Exact Reranking Report",
            "",
            "Run state: **success**",
            "",
            f"- Model: `{metadata.model_name}` (`{metadata.model_revision}`)",
            f"- Dataset fingerprint: `{metadata.dataset_fingerprint}`",
            f"- Manifest fingerprint: `{metadata.manifest_fingerprint}`",
            f"- Source cache fingerprint: `{metadata.source_cache_fingerprint}`",
            f"- Split / dimension: `{metadata.split}` / {metadata.embedding_dimension}",
            f"- Images / captions: {metadata.image_count} / {metadata.caption_count}",
            f"- Indexes: `{metadata.flat_index_type}` and `{metadata.hnsw_index_type}`",
            f"- Fixed HNSW: M={metadata.hnsw_m}, "
            f"efConstruction={metadata.hnsw_ef_construction}, efSearch={metadata.ef_search}",
            f"- Fixed exact-reranking shortlist: candidate_k={metadata.candidate_k}",
            f"- Artifact compatibility: **{metadata.artifact_compatibility}**",
            "- Rejected adapter embeddings used: **no**",
            "",
            *direction("Text to image", result.text_to_image),
            *direction("Image to text", result.image_to_text),
            "## Promotion gate",
            "",
            f"Decision: **{'pass' if result.promotion_gate.approved else 'fail'}**",
            "",
            f"{result.promotion_gate.recommendation}",
            "",
            "## Limitations",
            "",
            "This is retrieval-backend fidelity evaluation over the already-established official",
            "Flickr8k test artifact, not a new unbiased model-generalization estimate. Rescoring uses",
            "only persisted normalized CLIP vectors; it does not load CLIP or rejected adapters.",
            "Timings use one warmup and three measured whole-query-batch repetitions and are",
            "machine-specific and non-authoritative. FlatIP may remain faster at this dataset scale.",
            "No serving behavior is changed by this offline milestone.",
            "",
        ]
    )


def render_decision(result: RerankingEvaluationResult) -> str:
    gate = result.promotion_gate
    return "\n".join(
        [
            "# HNSW Exact Reranking Decision",
            "",
            f"Decision: **{'approve later service consideration' if gate.approved else 'do not promote'}**",
            "",
            f"- Flat top-10 shortlist coverage >= 0.99 in both directions: {gate.flat_top10_coverage_passed}",
            f"- Reranked top-10 agreement improved in both directions: {gate.top10_agreement_improved}",
            f"- Reranked MRR preserved within 0.001: {gate.mrr_preserved}",
            f"- Reranked Recall@10 preserved within 0.001: {gate.recall_at_10_preserved}",
            f"- Artifact compatibility passed: {gate.artifact_compatibility_passed}",
            f"- Rejected adapter embeddings used: {gate.rejected_adapter_embeddings_used}",
            "",
            gate.recommendation,
            "",
            "This decision applies only to possible later service work. Milestone 10A does not",
            "modify the retrieval service.",
            "",
        ]
    )


def write_reranking_outputs(
    result: RerankingEvaluationResult,
    metadata: RerankingMetadata,
    report_path: Path,
    metrics_path: Path,
    decision_path: Path,
) -> None:
    for path in (report_path, metrics_path, decision_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_reranking_report(result, metadata), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "run_state": "success",
                "metadata": asdict(metadata),
                **asdict(result),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    decision_path.write_text(render_decision(result), encoding="utf-8")


def write_reranking_failure(
    report_path: Path,
    metrics_path: Path,
    decision_path: Path,
    state: str,
    detail: str,
) -> None:
    if state not in SUPPORTED_RUN_STATES:
        raise ValueError(f"unsupported reranking report state: {state}")
    for path in (report_path, metrics_path, decision_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# HNSW Exact Reranking Report\n\nRun state: **{state}**\n\nDetail: {detail}\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": detail, "run_state": state}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision_path.write_text(
        "# HNSW Exact Reranking Decision\n\nDecision: **do not promote**\n\n"
        f"Run state: **{state}**. {detail}\n",
        encoding="utf-8",
    )


def load_persisted_reranking_info(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        raise RerankingArtifactUnavailableError(
            f"persisted reranking result is missing: {metrics_path}"
        )
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RerankingArtifactIncompatibleError("persisted reranking result is unreadable") from error
    if payload.get("run_state") != "success" or "metadata" not in payload:
        raise RerankingArtifactIncompatibleError("persisted reranking result is not successful")
    return payload
