"""Deterministic reports for the bounded frozen-embedding adapter experiment."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .contrastive_adapters import (
    AdapterEmbeddingCache,
    AdapterEvaluationComparison,
    metrics_payload,
)

ADAPTER_REPORT_STATES = frozenset(
    {
        "success",
        "not_run",
        "dependency_unavailable",
        "model_unavailable",
        "dataset_unavailable",
        "cache_incompatible",
        "training_failed",
        "evaluation_failed",
    }
)


def _metric_difference(adapted: Any, zero_shot: Any) -> dict[str, float | int]:
    return {
        "recall_at_1": adapted.recall_at_1 - zero_shot.recall_at_1,
        "recall_at_5": adapted.recall_at_5 - zero_shot.recall_at_5,
        "recall_at_10": adapted.recall_at_10 - zero_shot.recall_at_10,
        "mrr": adapted.mrr - zero_shot.mrr,
        "median_rank": adapted.median_rank - zero_shot.median_rank,
        "mean_rank": adapted.mean_rank - zero_shot.mean_rank,
        "query_count": adapted.query_count - zero_shot.query_count,
    }


def _metric_row(label: str, metrics: Any, candidates: int) -> str:
    return (
        f"| {label} | {metrics.recall_at_1:.4f} | {metrics.recall_at_5:.4f} | "
        f"{metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | "
        f"{metrics.median_rank:.2f} | {metrics.mean_rank:.2f} | "
        f"{metrics.query_count} | {candidates} |"
    )


def _difference_row(label: str, adapted: Any, zero_shot: Any) -> str:
    difference = _metric_difference(adapted, zero_shot)
    return (
        f"| {label} | {difference['recall_at_1']:+.4f} | "
        f"{difference['recall_at_5']:+.4f} | {difference['recall_at_10']:+.4f} | "
        f"{difference['mrr']:+.4f} | {difference['median_rank']:+.2f} | "
        f"{difference['mean_rank']:+.2f} |"
    )


def render_training_report(
    comparison: AdapterEvaluationComparison,
    metadata: dict[str, Any],
    train_cache: AdapterEmbeddingCache,
    validation_cache: AdapterEmbeddingCache,
) -> str:
    zero = comparison.zero_shot
    adapted = comparison.adapted
    config = metadata["training_config"]
    history = metadata["history"]
    history_rows = [
        (
            f"| {record['epoch']} | {record['training_loss']:.6f} | "
            f"{record['validation_mean_bidirectional_mrr']:.6f} |"
        )
        for record in history
    ]
    return "\n".join(
        [
            "# Contrastive Adapter Training Report",
            "",
            "Run state: **success**",
            "",
            "## Data and frozen encoder boundaries",
            "",
            f"- Training split: `{train_cache.metadata.split}`",
            f"- Training subset fingerprint: `{train_cache.metadata.subset_fingerprint}`",
            f"- Training images/captions: {train_cache.metadata.image_count} / "
            f"{train_cache.metadata.caption_count}",
            f"- Selection split: `{validation_cache.metadata.split}`",
            "- Validation subset status: `untouched for gradient updates`",
            f"- Validation subset fingerprint: `{validation_cache.metadata.subset_fingerprint}`",
            f"- Validation images/captions: {validation_cache.metadata.image_count} / "
            f"{validation_cache.metadata.caption_count}",
            "- Official test split accessed: `false`",
            f"- Source model: `{train_cache.metadata.model_name}`",
            f"- Source model revision: `{train_cache.metadata.model_revision}`",
            f"- Embedding dimension: {train_cache.metadata.embedding_dimension}",
            "- CLIP encoder frozen: `true`",
            "- Training input: `cached normalized embeddings only`",
            "",
            "## Fixed architecture and configuration",
            "",
            f"- Architecture: `{metadata['architecture']}`",
            f"- Bottleneck dimension: {metadata['bottleneck_dimension']}",
            "- Separate image/text parameters: `true`",
            f"- Parameter count: {metadata['parameter_count']}",
            f"- Seed: {config['seed']}",
            f"- Learning rate: {config['learning_rate']}",
            f"- Weight decay: {config['weight_decay']}",
            f"- Batch size: {config['batch_size']} unique images",
            f"- Temperature: {config['temperature']}",
            f"- Maximum epochs: {config['max_epochs']}",
            f"- Early-stopping patience: {config['early_stopping_patience']}",
            f"- Selected epoch: {metadata['selected_epoch']}",
            f"- Early stopped: `{str(metadata['early_stopped']).lower()}`",
            "- Selection metric: `validation mean bidirectional MRR`",
            "",
            "## Training history",
            "",
            "| Epoch | Training loss | Validation mean bidirectional MRR |",
            "| ---: | ---: | ---: |",
            *history_rows,
            "",
            "## Validation retrieval metrics",
            "",
            "| Representation and direction | R@1 | R@5 | R@10 | MRR | Median rank | "
            "Mean rank | Queries | Candidates |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _metric_row("Zero-shot text to image", zero.text_to_image.metrics, zero.text_to_image.candidate_count),
            _metric_row("Zero-shot image to text", zero.image_to_text.metrics, zero.image_to_text.candidate_count),
            _metric_row("Adapted text to image", adapted.text_to_image.metrics, adapted.text_to_image.candidate_count),
            _metric_row("Adapted image to text", adapted.image_to_text.metrics, adapted.image_to_text.candidate_count),
            "",
            "| Absolute adapted-minus-zero-shot difference | R@1 | R@5 | R@10 | MRR | "
            "Median rank | Mean rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _difference_row(
                "Text to image", adapted.text_to_image.metrics, zero.text_to_image.metrics
            ),
            _difference_row(
                "Image to text", adapted.image_to_text.metrics, zero.image_to_text.metrics
            ),
            "",
            "## Limitations",
            "",
            "This is one bounded adapter configuration over a small training subset. CLIP was",
            "not fine-tuned, and the official test split was not inspected. Validation selected",
            "the checkpoint and therefore is not an unbiased final benchmark. No hyperparameter",
            "search, reranking, LoRA, full-model gradients, or production-quality claim is included.",
            "A negative promotion result is expected and acceptable.",
            "",
        ]
    )


def render_promotion_report(comparison: AdapterEvaluationComparison) -> str:
    decision = comparison.promotion
    recommendation = (
        "Promote the bounded adapters for later test evaluation."
        if decision.promote
        else "Retain zero-shot CLIP; do not promote these adapters."
    )
    return "\n".join(
        [
            "# Contrastive Adapter Promotion Decision",
            "",
            "Run state: **success**",
            "",
            f"Decision: **{'promote' if decision.promote else 'retain zero-shot CLIP'}**",
            "",
            f"Mean bidirectional MRR difference: `{decision.mean_bidirectional_mrr_difference:+.6f}`",
            "",
            "## Conservative gates",
            "",
            *[f"- {reason}" for reason in decision.reasons],
            "",
            f"Recommendation: {recommendation}",
            "",
            "The decision uses only the official validation subset. The official test split remains",
            "untouched and is reserved for a later milestone.",
            "",
        ]
    )


def metrics_document(
    comparison: AdapterEvaluationComparison,
    metadata: dict[str, Any],
    train_cache: AdapterEmbeddingCache,
    validation_cache: AdapterEmbeddingCache,
) -> dict[str, Any]:
    zero = comparison.zero_shot
    adapted = comparison.adapted
    return {
        "run_state": "success",
        "data_boundaries": {
            "train_split": train_cache.metadata.split,
            "validation_split": validation_cache.metadata.split,
            "official_test_accessed": False,
            "train_images": train_cache.metadata.image_count,
            "train_captions": train_cache.metadata.caption_count,
            "validation_images": validation_cache.metadata.image_count,
            "validation_captions": validation_cache.metadata.caption_count,
            "train_subset_fingerprint": train_cache.metadata.subset_fingerprint,
            "validation_subset_fingerprint": validation_cache.metadata.subset_fingerprint,
        },
        "source_model": metadata["source_model"],
        "clip_frozen": metadata["clip_frozen"],
        "architecture": {
            "name": metadata["architecture"],
            "input_dimension": metadata["input_dimension"],
            "bottleneck_dimension": metadata["bottleneck_dimension"],
            "parameter_count": metadata["parameter_count"],
            "separate_text_and_image_adapters": True,
        },
        "training_config": metadata["training_config"],
        "training_history": metadata["history"],
        "selected_epoch": metadata["selected_epoch"],
        "early_stopped": metadata["early_stopped"],
        "zero_shot": {
            "text_to_image": metrics_payload(
                zero.text_to_image.metrics, zero.text_to_image.candidate_count
            ),
            "image_to_text": metrics_payload(
                zero.image_to_text.metrics, zero.image_to_text.candidate_count
            ),
        },
        "adapted": {
            "text_to_image": metrics_payload(
                adapted.text_to_image.metrics, adapted.text_to_image.candidate_count
            ),
            "image_to_text": metrics_payload(
                adapted.image_to_text.metrics, adapted.image_to_text.candidate_count
            ),
        },
        "differences": {
            "text_to_image": _metric_difference(
                adapted.text_to_image.metrics, zero.text_to_image.metrics
            ),
            "image_to_text": _metric_difference(
                adapted.image_to_text.metrics, zero.image_to_text.metrics
            ),
            "mean_bidirectional_mrr": comparison.promotion.mean_bidirectional_mrr_difference,
        },
        "promotion": asdict(comparison.promotion),
    }


def write_adapter_reports(
    comparison: AdapterEvaluationComparison,
    metadata: dict[str, Any],
    train_cache: AdapterEmbeddingCache,
    validation_cache: AdapterEmbeddingCache,
    training_report_path: Path,
    metrics_path: Path,
    promotion_path: Path,
) -> None:
    for path in (training_report_path, metrics_path, promotion_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    training_report_path.write_text(
        render_training_report(comparison, metadata, train_cache, validation_cache),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            metrics_document(comparison, metadata, train_cache, validation_cache),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    promotion_path.write_text(render_promotion_report(comparison), encoding="utf-8")


def write_adapter_failure_reports(
    state: str,
    detail: str,
    training_report_path: Path,
    metrics_path: Path,
    promotion_path: Path,
) -> None:
    if state not in ADAPTER_REPORT_STATES:
        raise ValueError(f"unsupported adapter report state: {state}")
    safe_detail = detail.replace("\n", " ")
    for path in (training_report_path, metrics_path, promotion_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    training_report_path.write_text(
        f"# Contrastive Adapter Training Report\n\nRun state: **{state}**\n\n"
        f"Detail: {safe_detail}\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": safe_detail, "run_state": state}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    promotion_path.write_text(
        "# Contrastive Adapter Promotion Decision\n\n"
        f"Run state: **{state}**\n\nDecision: **not evaluated**\n\n"
        f"Detail: {safe_detail}\n",
        encoding="utf-8",
    )
