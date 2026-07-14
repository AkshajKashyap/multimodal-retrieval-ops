"""Deterministic tracked reports for validation-only adapter failure analysis."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .contrastive_adapter_diagnostics import AdapterFailureAnalysis

DIAGNOSTIC_REPORT_STATES = frozenset(
    {
        "success",
        "not_run",
        "artifact_unavailable",
        "artifact_incompatible",
        "checkpoint_unavailable",
        "execution_failed",
    }
)


def _metric_row(label: str, direction: Any) -> str:
    metrics = direction.metrics
    return (
        f"| {label} | {metrics.recall_at_1:.4f} | {metrics.recall_at_5:.4f} | "
        f"{metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | "
        f"{metrics.median_rank:.2f} | {metrics.mean_rank:.2f} | "
        f"{metrics.query_count} | {direction.candidate_count} |"
    )


def _slice_metric_row(label: str, metrics: Any, candidates: int) -> str:
    return (
        f"| {label} | {metrics.recall_at_1:.4f} | {metrics.recall_at_5:.4f} | "
        f"{metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | "
        f"{metrics.median_rank:.2f} | {metrics.mean_rank:.2f} | "
        f"{metrics.query_count} | {candidates} |"
    )


def _outcome_lines(label: str, summary: Any) -> list[str]:
    return [
        f"### {label}",
        "",
        f"- Queries: {summary.query_count}",
        f"- Improved: {summary.improved_count} ({summary.improved_percentage:.2%})",
        f"- Unchanged: {summary.unchanged_count} ({summary.unchanged_percentage:.2%})",
        f"- Worsened: {summary.worsened_count} ({summary.worsened_percentage:.2%})",
        f"- Mean adapted-minus-zero-shot rank change: {summary.mean_rank_change:+.4f}",
        f"- Median rank change: {summary.median_rank_change:+.2f}",
        f"- Largest improvement: {summary.largest_improvement:+d} ranks",
        f"- Largest regression: {summary.largest_regression:+d} ranks",
        "",
    ]


def _margin_lines(label: str, summary: Any) -> list[str]:
    return [
        f"### {label}",
        "",
        f"- Zero-shot mean / median margin: {summary.zero_shot_mean_margin:+.6f} / "
        f"{summary.zero_shot_median_margin:+.6f}",
        f"- Adapted mean / median margin: {summary.adapted_mean_margin:+.6f} / "
        f"{summary.adapted_median_margin:+.6f}",
        f"- Positive-margin queries, zero-shot / adapted: "
        f"{summary.zero_shot_positive_margin_percentage:.2%} / "
        f"{summary.adapted_positive_margin_percentage:.2%}",
        f"- Margin improved / worsened: {summary.margin_improved_percentage:.2%} / "
        f"{summary.margin_worsened_percentage:.2%}",
        f"- Largest margin gain / loss: {summary.largest_margin_gain:+.6f} / "
        f"{summary.largest_margin_loss:+.6f}",
        "",
    ]


def _movement_lines(label: str, movement: Any) -> list[str]:
    return [
        f"| {label} | {movement.count} | {movement.mean:.6f} | {movement.median:.6f} | "
        f"{movement.minimum:.6f} | {movement.maximum:.6f} | "
        f"{movement.percentile_5:.6f} | {movement.percentile_95:.6f} |"
    ]


def _example_lines(label: str, examples: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {label}", ""]
    if not examples:
        return lines + ["No queries changed rank in this category.", ""]
    for item in examples:
        identity = item.get("caption_id", item.get("image_id", "unknown"))
        caption = str(item["caption_text"]).replace("\n", " ")
        lines.append(
            f"- `{identity}`: rank {item['zero_shot_rank']} → {item['adapted_rank']} "
            f"(change {item['rank_change']:+d}, margin {item['margin_change']:+.6f}); "
            f"caption: {caption}"
        )
    lines.append("")
    return lines


def render_failure_analysis(analysis: AdapterFailureAnalysis) -> str:
    compatibility = analysis.artifact_compatibility
    training = analysis.training_interpretation
    slice_rows = []
    for item in analysis.caption_length_slices:
        slice_rows.extend(
            [
                _slice_metric_row(
                    f"{item.group.title()} zero-shot",
                    item.zero_shot,
                    compatibility.validation_image_count,
                ),
                _slice_metric_row(
                    f"{item.group.title()} adapted",
                    item.adapted,
                    compatibility.validation_image_count,
                ),
            ]
        )
    return "\n".join(
        [
            "# Contrastive Adapter Failure Analysis",
            "",
            "Run state: **success**",
            "",
            "## Artifact compatibility",
            "",
            "- Compatibility: `passed`",
            f"- Model: `{compatibility.model_name}`",
            f"- Model revision: `{compatibility.model_revision}`",
            f"- Embedding dimension: {compatibility.embedding_dimension}",
            f"- Validation subset fingerprint: `{compatibility.validation_subset_fingerprint}`",
            f"- Validation images/captions: {compatibility.validation_image_count} / "
            f"{compatibility.validation_caption_count}",
            f"- Checkpoint architecture: `{compatibility.checkpoint_architecture}`",
            f"- Adapter parameters: {compatibility.adapter_parameter_count}",
            "- Cache, checkpoint, selected-ID, relationship, and recorded-metric checks: `passed`",
            "- Official test split accessed: `false`",
            "",
            "## Reproduced Milestone 9A validation metrics",
            "",
            "All reproduced values matched the tracked Milestone 9A metrics within `1e-6`.",
            "",
            "| Representation and direction | R@1 | R@5 | R@10 | MRR | Median rank | "
            "Mean rank | Queries | Candidates |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _metric_row("Zero-shot text to image", analysis.reproduced_zero_shot.text_to_image),
            _metric_row("Zero-shot image to text", analysis.reproduced_zero_shot.image_to_text),
            _metric_row("Adapted text to image", analysis.reproduced_adapted.text_to_image),
            _metric_row("Adapted image to text", analysis.reproduced_adapted.image_to_text),
            "",
            "## Per-query rank outcomes",
            "",
            *_outcome_lines("Text to image", analysis.text_outcomes),
            *_outcome_lines("Image to text", analysis.image_outcomes),
            "## Positive-versus-negative similarity margins",
            "",
            "For text queries, relevant similarity is the target image score. For image queries,",
            "it is the highest score among that image's relevant captions; irrelevant similarity",
            "is the highest score outside the relevant set.",
            "",
            *_margin_lines("Text to image", analysis.text_margins),
            *_margin_lines("Image to text", analysis.image_margins),
            "## Representation movement",
            "",
            "Cosine similarity compares each original frozen embedding with its adapted value.",
            "",
            "| Modality | Count | Mean | Median | Minimum | Maximum | P5 | P95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *_movement_lines("Image", analysis.image_movement),
            *_movement_lines("Text", analysis.text_movement),
            "",
            f"Text moved more aggressively than images: "
            f"`{str(analysis.text_moved_more_aggressively).lower()}`. Lower cosine means more movement.",
            "",
            "## Caption-length slices",
            "",
            "Whitespace token groups are fixed at short 1–7, medium 8–12, and long 13+.",
            "",
            "| Slice representation | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank | "
            "Queries | Candidates |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *slice_rows,
            "",
            "## Recorded training behavior",
            "",
            f"- Selected / stopping epoch: {training.selected_epoch} / {training.stopping_epoch}",
            f"- Best / final validation mean MRR: "
            f"{training.best_validation_mean_bidirectional_mrr:.6f} / "
            f"{training.final_validation_mean_bidirectional_mrr:.6f}",
            f"- Initial / selected / final training loss: {training.initial_training_loss:.6f} / "
            f"{training.selected_epoch_training_loss:.6f} / {training.final_training_loss:.6f}",
            f"- Loss trend: {training.training_loss_trend}",
            f"- Before selection: {training.validation_before_selected_epoch}",
            f"- After selection: {training.validation_after_selected_epoch}",
            f"- Supported interpretation: {training.supported_behavior}",
            "",
            "## Bounded qualitative examples",
            "",
            *_example_lines(
                "Five largest text-to-image regressions",
                analysis.examples["text_to_image_regressions"],
            ),
            *_example_lines(
                "Five largest text-to-image improvements",
                analysis.examples["text_to_image_improvements"],
            ),
            *_example_lines(
                "Five largest image-to-text regressions",
                analysis.examples["image_to_text_regressions"],
            ),
            *_example_lines(
                "Five largest image-to-text improvements",
                analysis.examples["image_to_text_improvements"],
            ),
            "## Supported diagnosis",
            "",
            *[
                f"- **{item.conclusion}** — {item.evidence}."
                for item in analysis.diagnoses
            ],
            "",
            "## Decision and limitations",
            "",
            "The Milestone 9A decision remains **retain zero-shot CLIP**. The adapter is not",
            "evaluated on the official test split and is not applied to serving indexes.",
            "This analysis is descriptive and validation-only. It uses one saved configuration",
            "and cannot establish causality or support a new quality claim. No CLIP inference,",
            "embedding generation, retraining, reranking, or parameter search was performed.",
            "",
        ]
    )


def render_decision_memo(analysis: AdapterFailureAnalysis) -> str:
    return "\n".join(
        [
            "# Contrastive Adapter Decision Memo",
            "",
            "Run state: **success**",
            "",
            "Decision: **retain zero-shot CLIP**",
            "",
            "The validation-only diagnostics reproduced the rejected Milestone 9A result.",
            "Supported findings:",
            "",
            *[f"- {item.conclusion}: {item.evidence}." for item in analysis.diagnoses],
            "",
            "No second model configuration was trained. The adapter remains excluded from serving",
            "indexes, and the official test split remains untouched for a later milestone.",
            "",
        ]
    )


def write_diagnostic_reports(
    analysis: AdapterFailureAnalysis,
    report_path: Path,
    metrics_path: Path,
    memo_path: Path,
) -> None:
    for path in (report_path, metrics_path, memo_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_failure_analysis(analysis), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "run_state": "success",
                "promotion_decision": "retain zero-shot CLIP",
                "official_test_accessed": False,
                "clip_inference_run": False,
                "embedding_generation_run": False,
                "adapter_retraining_run": False,
                **asdict(analysis),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    memo_path.write_text(render_decision_memo(analysis), encoding="utf-8")


def write_diagnostic_failure_reports(
    state: str,
    detail: str,
    report_path: Path,
    metrics_path: Path,
    memo_path: Path,
) -> None:
    if state not in DIAGNOSTIC_REPORT_STATES:
        raise ValueError(f"unsupported diagnostic report state: {state}")
    safe_detail = detail.replace("\n", " ")
    for path in (report_path, metrics_path, memo_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Contrastive Adapter Failure Analysis\n\nRun state: **{state}**\n\n"
        f"Detail: {safe_detail}\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": safe_detail, "run_state": state}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    memo_path.write_text(
        "# Contrastive Adapter Decision Memo\n\n"
        f"Run state: **{state}**\n\nDecision: **retain zero-shot CLIP**\n\n"
        "The prior decision is unchanged; diagnostics could not run.\n",
        encoding="utf-8",
    )
