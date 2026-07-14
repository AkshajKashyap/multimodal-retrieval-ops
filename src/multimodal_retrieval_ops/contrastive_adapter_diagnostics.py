"""Validation-only diagnostics for the rejected frozen-embedding adapters."""

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from .contrastive_adapters import (
    ADAPTER_ARCHITECTURE,
    AdapterEmbeddingCache,
    file_digest,
    load_adapter_cache,
    load_adapter_checkpoint,
    require_torch,
    validate_checkpoint_cache_identity,
)
from .evaluation import RetrievalMetrics, metrics_from_ranks
from .hf_clip_benchmark import BidirectionalResult, DirectionResult

METRIC_TOLERANCE = 1e-6
DIAGNOSTIC_VERSION = "validation-failure-analysis-v1"


class AdapterDiagnosticError(ValueError):
    """Base expected error for validation-only failure analysis."""


class DiagnosticArtifactUnavailableError(AdapterDiagnosticError):
    """A required cache, checkpoint, manifest, or metric record is unavailable."""


class DiagnosticCheckpointUnavailableError(AdapterDiagnosticError):
    """The selected Milestone 9A checkpoint is unavailable."""


class DiagnosticArtifactIncompatibleError(AdapterDiagnosticError):
    """Existing artifacts do not describe the same validation experiment."""


class DiagnosticExecutionError(AdapterDiagnosticError):
    """Compatible artifacts could not be analyzed."""


@dataclass(frozen=True)
class ValidationCaption:
    caption_id: str
    image_id: str
    caption_text: str


@dataclass(frozen=True)
class TextQueryDiagnostic:
    caption_id: str
    target_image_id: str
    zero_shot_rank: int
    adapted_rank: int
    rank_change: int
    zero_shot_reciprocal_rank: float
    adapted_reciprocal_rank: float
    caption_text: str
    caption_token_count: int
    zero_shot_relevant_similarity: float
    zero_shot_highest_irrelevant_similarity: float
    zero_shot_margin: float
    adapted_relevant_similarity: float
    adapted_highest_irrelevant_similarity: float
    adapted_margin: float
    margin_change: float


@dataclass(frozen=True)
class ImageQueryDiagnostic:
    image_id: str
    zero_shot_best_relevant_rank: int
    adapted_best_relevant_rank: int
    rank_change: int
    zero_shot_reciprocal_rank: float
    adapted_reciprocal_rank: float
    relevant_caption_ids: list[str]
    representative_caption_text: str
    zero_shot_relevant_similarity: float
    zero_shot_highest_irrelevant_similarity: float
    zero_shot_margin: float
    adapted_relevant_similarity: float
    adapted_highest_irrelevant_similarity: float
    adapted_margin: float
    margin_change: float


@dataclass(frozen=True)
class QueryOutcomeSummary:
    query_count: int
    improved_count: int
    improved_percentage: float
    unchanged_count: int
    unchanged_percentage: float
    worsened_count: int
    worsened_percentage: float
    mean_rank_change: float
    median_rank_change: float
    largest_improvement: int
    largest_regression: int


@dataclass(frozen=True)
class MarginSummary:
    query_count: int
    zero_shot_mean_margin: float
    zero_shot_median_margin: float
    adapted_mean_margin: float
    adapted_median_margin: float
    zero_shot_positive_margin_percentage: float
    adapted_positive_margin_percentage: float
    margin_improved_percentage: float
    margin_worsened_percentage: float
    largest_margin_gain: float
    largest_margin_loss: float


@dataclass(frozen=True)
class MovementStatistics:
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    percentile_5: float
    percentile_95: float


@dataclass(frozen=True)
class CaptionSliceResult:
    group: str
    query_count: int
    zero_shot: RetrievalMetrics
    adapted: RetrievalMetrics


@dataclass(frozen=True)
class TrainingHistoryInterpretation:
    selected_epoch: int
    stopping_epoch: int
    best_validation_mean_bidirectional_mrr: float
    final_validation_mean_bidirectional_mrr: float
    initial_training_loss: float
    selected_epoch_training_loss: float
    final_training_loss: float
    training_loss_trend: str
    validation_before_selected_epoch: str
    validation_after_selected_epoch: str
    supported_behavior: str


@dataclass(frozen=True)
class DiagnosisConclusion:
    conclusion: str
    evidence: str


@dataclass(frozen=True)
class ArtifactCompatibility:
    compatible: bool
    model_name: str
    model_revision: str
    embedding_dimension: int
    validation_subset_fingerprint: str
    validation_image_count: int
    validation_caption_count: int
    checkpoint_architecture: str
    adapter_parameter_count: int
    train_cache_fingerprint: str
    validation_cache_fingerprint: str


@dataclass(frozen=True)
class AdapterFailureAnalysis:
    artifact_compatibility: ArtifactCompatibility
    reproduced_zero_shot: BidirectionalResult
    reproduced_adapted: BidirectionalResult
    text_queries: list[TextQueryDiagnostic]
    image_queries: list[ImageQueryDiagnostic]
    text_outcomes: QueryOutcomeSummary
    image_outcomes: QueryOutcomeSummary
    text_margins: MarginSummary
    image_margins: MarginSummary
    image_movement: MovementStatistics
    text_movement: MovementStatistics
    text_moved_more_aggressively: bool
    caption_length_slices: list[CaptionSliceResult]
    training_interpretation: TrainingHistoryInterpretation
    diagnoses: list[DiagnosisConclusion]
    examples: dict[str, list[dict[str, Any]]]


def read_validation_captions(path: Path) -> dict[str, ValidationCaption]:
    """Read validation metadata without materializing official test records."""
    if not path.is_file():
        raise DiagnosticArtifactUnavailableError("schema-v2 manifest is unavailable")
    required = {"image_id", "caption_id", "caption", "split"}
    captions: dict[str, ValidationCaption] = {}
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise DiagnosticArtifactIncompatibleError(
                    f"manifest is missing diagnostic fields: {', '.join(missing)}"
                )
            for row in reader:
                split = row["split"]
                if split == "test":
                    break
                if split != "validation":
                    continue
                caption_id = row["caption_id"]
                if caption_id in captions:
                    raise DiagnosticArtifactIncompatibleError(
                        "validation manifest contains duplicate caption IDs"
                    )
                captions[caption_id] = ValidationCaption(
                    caption_id=caption_id,
                    image_id=row["image_id"],
                    caption_text=row["caption"],
                )
    except AdapterDiagnosticError:
        raise
    except Exception as error:
        raise DiagnosticArtifactIncompatibleError(
            "validation manifest could not be read safely"
        ) from error
    return captions


def caption_token_count(caption: str) -> int:
    return len(caption.split())


def caption_length_group(token_count: int) -> str:
    if token_count <= 0:
        raise AdapterDiagnosticError("caption token count must be positive")
    if token_count <= 7:
        return "short"
    if token_count <= 12:
        return "medium"
    return "long"


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise AdapterDiagnosticError("at least one value is required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def movement_statistics(
    original_vectors: Iterable[Iterable[float]],
    adapted_vectors: Iterable[Iterable[float]],
) -> MovementStatistics:
    similarities = []
    for original, adapted in zip(original_vectors, adapted_vectors, strict=True):
        original_values = list(original)
        adapted_values = list(adapted)
        if len(original_values) != len(adapted_values):
            raise DiagnosticArtifactIncompatibleError("representation dimensions differ")
        similarity = sum(a * b for a, b in zip(original_values, adapted_values, strict=True))
        similarities.append(max(-1.0, min(1.0, similarity)))
    if not similarities:
        raise AdapterDiagnosticError("movement analysis requires embeddings")
    return MovementStatistics(
        count=len(similarities),
        mean=statistics.mean(similarities),
        median=float(statistics.median(similarities)),
        minimum=min(similarities),
        maximum=max(similarities),
        percentile_5=_percentile(similarities, 0.05),
        percentile_95=_percentile(similarities, 0.95),
    )


def _stable_rank(scores: list[float], target_indices: set[int]) -> tuple[int, int]:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    ranked = [(rank, index) for rank, index in enumerate(order, start=1) if index in target_indices]
    return min(ranked)[0], min(ranked)[1]


def _margin(scores: list[float], relevant_indices: set[int]) -> tuple[float, float, float]:
    relevant = max(scores[index] for index in relevant_indices)
    irrelevant = max(
        score for index, score in enumerate(scores) if index not in relevant_indices
    )
    return relevant, irrelevant, relevant - irrelevant


def calculate_query_diagnostics(
    zero_similarity: list[list[float]],
    adapted_similarity: list[list[float]],
    image_ids: list[str],
    caption_ids: list[str],
    caption_image_ids: dict[str, str],
    caption_text: dict[str, str],
) -> tuple[list[TextQueryDiagnostic], list[ImageQueryDiagnostic]]:
    if len(zero_similarity) != len(caption_ids) or len(adapted_similarity) != len(caption_ids):
        raise DiagnosticArtifactIncompatibleError("similarity row count is incompatible")
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    text_queries = []
    for caption_index, caption_id in enumerate(caption_ids):
        target_image_id = caption_image_ids[caption_id]
        target = image_index[target_image_id]
        zero_scores = zero_similarity[caption_index]
        adapted_scores = adapted_similarity[caption_index]
        zero_rank, _ = _stable_rank(zero_scores, {target})
        adapted_rank, _ = _stable_rank(adapted_scores, {target})
        zero_relevant, zero_irrelevant, zero_margin = _margin(zero_scores, {target})
        adapted_relevant, adapted_irrelevant, adapted_margin = _margin(
            adapted_scores, {target}
        )
        text_queries.append(
            TextQueryDiagnostic(
                caption_id=caption_id,
                target_image_id=target_image_id,
                zero_shot_rank=zero_rank,
                adapted_rank=adapted_rank,
                rank_change=adapted_rank - zero_rank,
                zero_shot_reciprocal_rank=1.0 / zero_rank,
                adapted_reciprocal_rank=1.0 / adapted_rank,
                caption_text=caption_text[caption_id],
                caption_token_count=caption_token_count(caption_text[caption_id]),
                zero_shot_relevant_similarity=zero_relevant,
                zero_shot_highest_irrelevant_similarity=zero_irrelevant,
                zero_shot_margin=zero_margin,
                adapted_relevant_similarity=adapted_relevant,
                adapted_highest_irrelevant_similarity=adapted_irrelevant,
                adapted_margin=adapted_margin,
                margin_change=adapted_margin - zero_margin,
            )
        )
    image_queries = []
    for image_index_value, image_id in enumerate(image_ids):
        relevant_ids = sorted(
            caption_id
            for caption_id in caption_ids
            if caption_image_ids[caption_id] == image_id
        )
        relevant_indices = {caption_ids.index(caption_id) for caption_id in relevant_ids}
        zero_scores = [row[image_index_value] for row in zero_similarity]
        adapted_scores = [row[image_index_value] for row in adapted_similarity]
        zero_rank, zero_best = _stable_rank(zero_scores, relevant_indices)
        adapted_rank, adapted_best = _stable_rank(adapted_scores, relevant_indices)
        zero_relevant, zero_irrelevant, zero_margin = _margin(zero_scores, relevant_indices)
        adapted_relevant, adapted_irrelevant, adapted_margin = _margin(
            adapted_scores, relevant_indices
        )
        representative_index = adapted_best if adapted_rank <= zero_rank else zero_best
        image_queries.append(
            ImageQueryDiagnostic(
                image_id=image_id,
                zero_shot_best_relevant_rank=zero_rank,
                adapted_best_relevant_rank=adapted_rank,
                rank_change=adapted_rank - zero_rank,
                zero_shot_reciprocal_rank=1.0 / zero_rank,
                adapted_reciprocal_rank=1.0 / adapted_rank,
                relevant_caption_ids=relevant_ids,
                representative_caption_text=caption_text[caption_ids[representative_index]],
                zero_shot_relevant_similarity=zero_relevant,
                zero_shot_highest_irrelevant_similarity=zero_irrelevant,
                zero_shot_margin=zero_margin,
                adapted_relevant_similarity=adapted_relevant,
                adapted_highest_irrelevant_similarity=adapted_irrelevant,
                adapted_margin=adapted_margin,
                margin_change=adapted_margin - zero_margin,
            )
        )
    return text_queries, image_queries


def summarize_rank_changes(records: Iterable[Any]) -> QueryOutcomeSummary:
    changes = [record.rank_change for record in records]
    if not changes:
        raise AdapterDiagnosticError("rank-change analysis requires queries")
    count = len(changes)
    improved = sum(change < 0 for change in changes)
    unchanged = sum(change == 0 for change in changes)
    worsened = sum(change > 0 for change in changes)
    return QueryOutcomeSummary(
        query_count=count,
        improved_count=improved,
        improved_percentage=improved / count,
        unchanged_count=unchanged,
        unchanged_percentage=unchanged / count,
        worsened_count=worsened,
        worsened_percentage=worsened / count,
        mean_rank_change=statistics.mean(changes),
        median_rank_change=float(statistics.median(changes)),
        largest_improvement=min(changes),
        largest_regression=max(changes),
    )


def summarize_margins(records: Iterable[Any]) -> MarginSummary:
    items = list(records)
    if not items:
        raise AdapterDiagnosticError("margin analysis requires queries")
    zero = [record.zero_shot_margin for record in items]
    adapted = [record.adapted_margin for record in items]
    changes = [record.margin_change for record in items]
    count = len(items)
    return MarginSummary(
        query_count=count,
        zero_shot_mean_margin=statistics.mean(zero),
        zero_shot_median_margin=float(statistics.median(zero)),
        adapted_mean_margin=statistics.mean(adapted),
        adapted_median_margin=float(statistics.median(adapted)),
        zero_shot_positive_margin_percentage=sum(value > 0 for value in zero) / count,
        adapted_positive_margin_percentage=sum(value > 0 for value in adapted) / count,
        margin_improved_percentage=sum(value > 0 for value in changes) / count,
        margin_worsened_percentage=sum(value < 0 for value in changes) / count,
        largest_margin_gain=max(changes),
        largest_margin_loss=min(changes),
    )


def caption_slice_metrics(
    text_queries: list[TextQueryDiagnostic],
) -> list[CaptionSliceResult]:
    output = []
    for group in ("short", "medium", "long"):
        members = [
            record
            for record in text_queries
            if caption_length_group(record.caption_token_count) == group
        ]
        if not members:
            raise AdapterDiagnosticError(f"caption-length group '{group}' has no queries")
        output.append(
            CaptionSliceResult(
                group=group,
                query_count=len(members),
                zero_shot=metrics_from_ranks([record.zero_shot_rank for record in members]),
                adapted=metrics_from_ranks([record.adapted_rank for record in members]),
            )
        )
    return output


def interpret_training_history(metadata: dict[str, Any]) -> TrainingHistoryInterpretation:
    history = metadata.get("history", [])
    selected_epoch = int(metadata.get("selected_epoch", 0))
    if not history or selected_epoch <= 0:
        raise DiagnosticArtifactIncompatibleError("checkpoint training history is incomplete")
    by_epoch = {int(record["epoch"]): record for record in history}
    if selected_epoch not in by_epoch:
        raise DiagnosticArtifactIncompatibleError("selected epoch is absent from history")
    selected = by_epoch[selected_epoch]
    final = history[-1]
    initial = history[0]
    before = [
        record["validation_mean_bidirectional_mrr"]
        for record in history
        if record["epoch"] < selected_epoch
    ]
    after = [
        record["validation_mean_bidirectional_mrr"]
        for record in history
        if record["epoch"] > selected_epoch
    ]
    loss_trend = (
        "decreased throughout the recorded run"
        if all(
            current["training_loss"] <= previous["training_loss"]
            for previous, current in zip(history, history[1:], strict=False)
        )
        else "decreased overall with fluctuations"
        if final["training_loss"] < initial["training_loss"]
        else "did not decrease overall"
    )
    before_text = (
        f"validation mean MRR ranged from {min(before):.6f} to {max(before):.6f} "
        f"before epoch {selected_epoch}"
        if before
        else "the first epoch was selected"
    )
    after_text = (
        f"all {len(after)} later validation values were below the selected value"
        if after and max(after) < selected["validation_mean_bidirectional_mrr"]
        else f"later validation values reached {max(after):.6f}"
        if after
        else "training stopped at the selected epoch"
    )
    overfit_evidence = (
        int(final["epoch"]) > selected_epoch
        and final["training_loss"] < selected["training_loss"]
        and bool(after)
        and max(after) < selected["validation_mean_bidirectional_mrr"]
    )
    behavior = (
        "evidence consistent with overfitting after the selected epoch"
        if overfit_evidence
        else "mixed or inconclusive training-history evidence"
    )
    return TrainingHistoryInterpretation(
        selected_epoch=selected_epoch,
        stopping_epoch=int(final["epoch"]),
        best_validation_mean_bidirectional_mrr=float(
            selected["validation_mean_bidirectional_mrr"]
        ),
        final_validation_mean_bidirectional_mrr=float(
            final["validation_mean_bidirectional_mrr"]
        ),
        initial_training_loss=float(initial["training_loss"]),
        selected_epoch_training_loss=float(selected["training_loss"]),
        final_training_loss=float(final["training_loss"]),
        training_loss_trend=loss_trend,
        validation_before_selected_epoch=before_text,
        validation_after_selected_epoch=after_text,
        supported_behavior=behavior,
    )


def diagnose_failure(
    *,
    mean_mrr_difference: float,
    text_mrr_difference: float,
    image_mrr_difference: float,
    image_movement_mean: float,
    text_movement_mean: float,
    training: TrainingHistoryInterpretation,
) -> list[DiagnosisConclusion]:
    conclusions = []
    if training.supported_behavior.startswith("evidence consistent with overfitting"):
        conclusions.append(
            DiagnosisConclusion(
                "likely overfitting",
                f"training loss fell from {training.selected_epoch_training_loss:.6f} at "
                f"selected epoch {training.selected_epoch} to {training.final_training_loss:.6f} "
                f"at epoch {training.stopping_epoch}, while all later validation values stayed "
                "below the selected value",
            )
        )
    directional_gap = abs(text_mrr_difference - image_mrr_difference)
    if directional_gap >= 0.01:
        conclusions.append(
            DiagnosisConclusion(
                "likely optimization imbalance between modalities",
                f"text-to-image and image-to-text MRR changes differed by "
                f"{directional_gap:.6f} ({text_mrr_difference:+.6f} versus "
                f"{image_mrr_difference:+.6f})",
            )
        )
    if min(image_movement_mean, text_movement_mean) < 0.95:
        conclusions.append(
            DiagnosisConclusion(
                "representation movement too aggressive",
                f"mean original-to-adapted cosine was {image_movement_mean:.6f} for images "
                f"and {text_movement_mean:.6f} for text; at least one fell below 0.95",
            )
        )
    if mean_mrr_difference < 0.005:
        conclusions.append(
            DiagnosisConclusion(
                "insufficient improvement signal",
                f"mean bidirectional MRR changed by {mean_mrr_difference:+.6f}, below the "
                "+0.005 promotion requirement",
            )
        )
    if not conclusions:
        conclusions.append(
            DiagnosisConclusion(
                "mixed or inconclusive evidence",
                "none of the bounded diagnostic thresholds supplied a supported explanation",
            )
        )
    return conclusions


def _direction_from_ranks(ranks: list[int], candidates: int) -> DirectionResult:
    return DirectionResult(metrics_from_ranks(ranks), candidates)


def _result_from_queries(
    text_queries: list[TextQueryDiagnostic],
    image_queries: list[ImageQueryDiagnostic],
    *,
    adapted: bool,
) -> BidirectionalResult:
    text_ranks = [
        record.adapted_rank if adapted else record.zero_shot_rank for record in text_queries
    ]
    image_ranks = [
        record.adapted_best_relevant_rank
        if adapted
        else record.zero_shot_best_relevant_rank
        for record in image_queries
    ]
    return BidirectionalResult(
        text_to_image=_direction_from_ranks(text_ranks, len(image_queries)),
        image_to_text=_direction_from_ranks(image_ranks, len(text_queries)),
    )


def _verify_direction(
    reproduced: DirectionResult,
    recorded: dict[str, Any],
    label: str,
    tolerance: float,
) -> None:
    values = asdict(reproduced.metrics) | {"candidate_count": reproduced.candidate_count}
    for name, value in values.items():
        recorded_value = recorded.get(name)
        if isinstance(value, int):
            matches = value == recorded_value
        else:
            matches = isinstance(recorded_value, (int, float)) and math.isclose(
                value, recorded_value, abs_tol=tolerance
            )
        if not matches:
            raise DiagnosticArtifactIncompatibleError(
                f"reproduced {label} {name} does not match the Milestone 9A record"
            )


def verify_recorded_metrics(
    zero_shot: BidirectionalResult,
    adapted: BidirectionalResult,
    recorded: dict[str, Any],
    tolerance: float = METRIC_TOLERANCE,
) -> None:
    if recorded.get("run_state") != "success":
        raise DiagnosticArtifactIncompatibleError("Milestone 9A metric record was not successful")
    for representation, result in (("zero_shot", zero_shot), ("adapted", adapted)):
        document = recorded.get(representation, {})
        _verify_direction(
            result.text_to_image,
            document.get("text_to_image", {}),
            f"{representation} text-to-image",
            tolerance,
        )
        _verify_direction(
            result.image_to_text,
            document.get("image_to_text", {}),
            f"{representation} image-to-text",
            tolerance,
        )


def validate_diagnostic_artifacts(
    train_cache: AdapterEmbeddingCache,
    validation_cache: AdapterEmbeddingCache,
    checkpoint_metadata: dict[str, Any],
    recorded_metrics: dict[str, Any],
    validation_captions: dict[str, ValidationCaption],
    adapters: Any,
    train_cache_path: Path,
    validation_cache_path: Path,
) -> ArtifactCompatibility:
    if train_cache.metadata.split != "train" or validation_cache.metadata.split != "validation":
        raise DiagnosticArtifactIncompatibleError(
            "diagnostics require the original train and validation caches"
        )
    identity_fields = (
        "backend_name",
        "backend_version",
        "preprocessing_identity",
        "model_name",
        "model_revision",
        "dataset_fingerprint",
        "dataset_revision",
        "embedding_dimension",
    )
    if any(
        getattr(train_cache.metadata, field) != getattr(validation_cache.metadata, field)
        for field in identity_fields
    ):
        raise DiagnosticArtifactIncompatibleError(
            "training and validation cache source identities differ"
        )
    validate_checkpoint_cache_identity(
        checkpoint_metadata, train_cache_path, validation_cache_path
    )
    if checkpoint_metadata.get("architecture") != ADAPTER_ARCHITECTURE:
        raise DiagnosticArtifactIncompatibleError("checkpoint architecture is incompatible")
    source = checkpoint_metadata.get("source_model", {})
    expected_source = {
        "backend_name": validation_cache.metadata.backend_name,
        "backend_version": validation_cache.metadata.backend_version,
        "model_name": validation_cache.metadata.model_name,
        "model_revision": validation_cache.metadata.model_revision,
        "preprocessing_identity": validation_cache.metadata.preprocessing_identity,
    }
    if source != expected_source:
        raise DiagnosticArtifactIncompatibleError("checkpoint source model identity differs")
    if checkpoint_metadata.get("input_dimension") != validation_cache.metadata.embedding_dimension:
        raise DiagnosticArtifactIncompatibleError("checkpoint embedding dimension differs")
    if (
        checkpoint_metadata.get("validation_subset_fingerprint")
        != validation_cache.metadata.subset_fingerprint
    ):
        raise DiagnosticArtifactIncompatibleError("validation subset fingerprint differs")
    selected_caption_ids = set(validation_cache.caption_embeddings)
    if not selected_caption_ids <= set(validation_captions):
        raise DiagnosticArtifactIncompatibleError(
            "selected validation caption IDs are missing from the manifest"
        )
    selected_captions = {
        caption_id: validation_captions[caption_id] for caption_id in selected_caption_ids
    }
    if {item.image_id for item in selected_captions.values()} != set(
        validation_cache.metadata.selected_image_ids
    ):
        raise DiagnosticArtifactIncompatibleError(
            "selected validation image IDs differ from the manifest"
        )
    for caption_id, item in selected_captions.items():
        if validation_cache.caption_image_ids.get(caption_id) != item.image_id:
            raise DiagnosticArtifactIncompatibleError(
                "validation image-caption relationships differ from the manifest"
            )
    parameter_count = sum(parameter.numel() for parameter in adapters.parameters())
    if parameter_count != checkpoint_metadata.get("parameter_count"):
        raise DiagnosticArtifactIncompatibleError("adapter parameter metadata differs")
    recorded_boundaries = recorded_metrics.get("data_boundaries", {})
    if recorded_boundaries.get("official_test_accessed") is not False:
        raise DiagnosticArtifactIncompatibleError("Milestone 9A did not preserve the test boundary")
    if (
        recorded_boundaries.get("validation_subset_fingerprint")
        != validation_cache.metadata.subset_fingerprint
    ):
        raise DiagnosticArtifactIncompatibleError("recorded validation subset differs")
    if recorded_metrics.get("promotion", {}).get("promote") is not False:
        raise DiagnosticArtifactIncompatibleError("Milestone 9A promotion decision is not retained")
    if recorded_metrics.get("source_model") != source:
        raise DiagnosticArtifactIncompatibleError("recorded source model identity differs")
    recorded_architecture = recorded_metrics.get("architecture", {})
    if (
        recorded_architecture.get("name") != ADAPTER_ARCHITECTURE
        or recorded_architecture.get("input_dimension")
        != validation_cache.metadata.embedding_dimension
        or recorded_architecture.get("parameter_count") != parameter_count
    ):
        raise DiagnosticArtifactIncompatibleError("recorded adapter architecture differs")
    return ArtifactCompatibility(
        compatible=True,
        model_name=validation_cache.metadata.model_name,
        model_revision=validation_cache.metadata.model_revision,
        embedding_dimension=validation_cache.metadata.embedding_dimension,
        validation_subset_fingerprint=validation_cache.metadata.subset_fingerprint,
        validation_image_count=validation_cache.metadata.image_count,
        validation_caption_count=validation_cache.metadata.caption_count,
        checkpoint_architecture=checkpoint_metadata["architecture"],
        adapter_parameter_count=parameter_count,
        train_cache_fingerprint=file_digest(train_cache_path),
        validation_cache_fingerprint=file_digest(validation_cache_path),
    )


def deterministic_examples(
    text_queries: list[TextQueryDiagnostic], image_queries: list[ImageQueryDiagnostic]
) -> dict[str, list[dict[str, Any]]]:
    def text_document(record: TextQueryDiagnostic) -> dict[str, Any]:
        return {
            "caption_id": record.caption_id,
            "target_image_id": record.target_image_id,
            "caption_text": record.caption_text,
            "zero_shot_rank": record.zero_shot_rank,
            "adapted_rank": record.adapted_rank,
            "rank_change": record.rank_change,
            "margin_change": record.margin_change,
        }

    def image_document(record: ImageQueryDiagnostic) -> dict[str, Any]:
        return {
            "image_id": record.image_id,
            "relevant_caption_ids": record.relevant_caption_ids,
            "caption_text": record.representative_caption_text,
            "zero_shot_rank": record.zero_shot_best_relevant_rank,
            "adapted_rank": record.adapted_best_relevant_rank,
            "rank_change": record.rank_change,
            "margin_change": record.margin_change,
        }

    text_regressions = sorted(
        (record for record in text_queries if record.rank_change > 0),
        key=lambda record: (-record.rank_change, record.margin_change, record.caption_id),
    )[:5]
    text_improvements = sorted(
        (record for record in text_queries if record.rank_change < 0),
        key=lambda record: (record.rank_change, -record.margin_change, record.caption_id),
    )[:5]
    image_regressions = sorted(
        (record for record in image_queries if record.rank_change > 0),
        key=lambda record: (-record.rank_change, record.margin_change, record.image_id),
    )[:5]
    image_improvements = sorted(
        (record for record in image_queries if record.rank_change < 0),
        key=lambda record: (record.rank_change, -record.margin_change, record.image_id),
    )[:5]
    return {
        "text_to_image_regressions": [text_document(record) for record in text_regressions],
        "text_to_image_improvements": [text_document(record) for record in text_improvements],
        "image_to_text_regressions": [image_document(record) for record in image_regressions],
        "image_to_text_improvements": [image_document(record) for record in image_improvements],
    }


def analyze_loaded_artifacts(
    train_cache: AdapterEmbeddingCache,
    validation_cache: AdapterEmbeddingCache,
    adapters: Any,
    checkpoint_metadata: dict[str, Any],
    recorded_metrics: dict[str, Any],
    validation_captions: dict[str, ValidationCaption],
    compatibility: ArtifactCompatibility,
    device: str = "cpu",
) -> AdapterFailureAnalysis:
    torch = require_torch()
    image_ids = sorted(validation_cache.image_embeddings)
    caption_ids = sorted(validation_cache.caption_embeddings)
    original_images = torch.tensor(
        [validation_cache.image_embeddings[item_id] for item_id in image_ids],
        dtype=torch.float32,
        device=device,
    )
    original_text = torch.tensor(
        [validation_cache.caption_embeddings[item_id] for item_id in caption_ids],
        dtype=torch.float32,
        device=device,
    )
    adapters.eval()
    with torch.inference_mode():
        adapted_images = adapters["image"](original_images)
        adapted_text = adapters["text"](original_text)
        zero_similarity = (original_text @ original_images.T).cpu().tolist()
        adapted_similarity = (adapted_text @ adapted_images.T).cpu().tolist()
    text_queries, image_queries = calculate_query_diagnostics(
        zero_similarity,
        adapted_similarity,
        image_ids,
        caption_ids,
        validation_cache.caption_image_ids,
        {caption_id: validation_captions[caption_id].caption_text for caption_id in caption_ids},
    )
    reproduced_zero = _result_from_queries(text_queries, image_queries, adapted=False)
    reproduced_adapted = _result_from_queries(text_queries, image_queries, adapted=True)
    verify_recorded_metrics(reproduced_zero, reproduced_adapted, recorded_metrics)
    image_movement = movement_statistics(
        original_images.cpu().tolist(), adapted_images.cpu().tolist()
    )
    text_movement = movement_statistics(
        original_text.cpu().tolist(), adapted_text.cpu().tolist()
    )
    training = interpret_training_history(checkpoint_metadata)
    mean_difference = (
        reproduced_adapted.text_to_image.metrics.mrr
        + reproduced_adapted.image_to_text.metrics.mrr
        - reproduced_zero.text_to_image.metrics.mrr
        - reproduced_zero.image_to_text.metrics.mrr
    ) / 2
    text_difference = (
        reproduced_adapted.text_to_image.metrics.mrr
        - reproduced_zero.text_to_image.metrics.mrr
    )
    image_difference = (
        reproduced_adapted.image_to_text.metrics.mrr
        - reproduced_zero.image_to_text.metrics.mrr
    )
    return AdapterFailureAnalysis(
        artifact_compatibility=compatibility,
        reproduced_zero_shot=reproduced_zero,
        reproduced_adapted=reproduced_adapted,
        text_queries=text_queries,
        image_queries=image_queries,
        text_outcomes=summarize_rank_changes(text_queries),
        image_outcomes=summarize_rank_changes(image_queries),
        text_margins=summarize_margins(text_queries),
        image_margins=summarize_margins(image_queries),
        image_movement=image_movement,
        text_movement=text_movement,
        text_moved_more_aggressively=text_movement.mean < image_movement.mean,
        caption_length_slices=caption_slice_metrics(text_queries),
        training_interpretation=training,
        diagnoses=diagnose_failure(
            mean_mrr_difference=mean_difference,
            text_mrr_difference=text_difference,
            image_mrr_difference=image_difference,
            image_movement_mean=image_movement.mean,
            text_movement_mean=text_movement.mean,
            training=training,
        ),
        examples=deterministic_examples(text_queries, image_queries),
    )


def load_recorded_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticArtifactUnavailableError("Milestone 9A metric record is unavailable")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise DiagnosticArtifactIncompatibleError(
            "Milestone 9A metric record is invalid"
        ) from error


def run_adapter_failure_analysis(
    *,
    train_cache_path: Path,
    validation_cache_path: Path,
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    manifest_path: Path,
    recorded_metrics_path: Path,
    device: str = "cpu",
) -> AdapterFailureAnalysis:
    """Load only existing artifacts and reproduce validation diagnostics once."""
    for path, label in (
        (train_cache_path, "training cache"),
        (validation_cache_path, "validation cache"),
        (manifest_path, "schema-v2 manifest"),
        (recorded_metrics_path, "Milestone 9A metrics"),
    ):
        if not path.is_file():
            raise DiagnosticArtifactUnavailableError(f"required {label} is unavailable")
    if not checkpoint_path.is_file() or not checkpoint_metadata_path.is_file():
        raise DiagnosticCheckpointUnavailableError(
            "selected Milestone 9A checkpoint artifacts are unavailable"
        )
    try:
        train_cache = load_adapter_cache(train_cache_path)
        validation_cache = load_adapter_cache(validation_cache_path)
        adapters, checkpoint_metadata = load_adapter_checkpoint(
            checkpoint_path, checkpoint_metadata_path, device
        )
        recorded = load_recorded_metrics(recorded_metrics_path)
        captions = read_validation_captions(manifest_path)
        compatibility = validate_diagnostic_artifacts(
            train_cache,
            validation_cache,
            checkpoint_metadata,
            recorded,
            captions,
            adapters,
            train_cache_path,
            validation_cache_path,
        )
        return analyze_loaded_artifacts(
            train_cache,
            validation_cache,
            adapters,
            checkpoint_metadata,
            recorded,
            captions,
            compatibility,
            device,
        )
    except AdapterDiagnosticError:
        raise
    except ValueError as error:
        raise DiagnosticArtifactIncompatibleError(str(error)) from error
    except Exception as error:
        raise DiagnosticExecutionError("validation-only adapter analysis failed") from error
