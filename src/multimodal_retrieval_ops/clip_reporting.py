"""Deterministic optional-backend and retrieval reporting."""

from dataclasses import asdict
import json
from pathlib import Path

from .clip_backend import DEFAULT_CLIP_MODEL
from .evaluation import RetrievalMetrics


def render_clip_backend_report(
    status: str,
    *,
    model_name: str = DEFAULT_CLIP_MODEL,
    device: str = "cpu",
    dimension: int | None = None,
    item_count: int | None = None,
    cache_status: str = "not checked",
    detail: str | None = None,
) -> str:
    lines = [
        "# CLIP Backend Report",
        "",
        f"Execution status: **{status}**",
        "",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        f"- Embedding dimension: {dimension if dimension is not None else 'not loaded'}",
        f"- Item count: {item_count if item_count is not None else 'not indexed'}",
        f"- Cache: {cache_status}",
    ]
    if detail:
        lines.extend(["", f"Detail: {detail}"])
    lines.extend(
        [
            "",
            "The base installation does not require Torch, Transformers, Pillow, or model weights.",
            "Explicit CLIP model commands may download weights; use `--local-files-only` to forbid it.",
            "",
        ]
    )
    return "\n".join(lines)


def write_clip_backend_report(path: Path, **report_fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_clip_backend_report(**report_fields), encoding="utf-8")


def render_clip_retrieval_report(
    metrics: RetrievalMetrics,
    model_name: str,
    *,
    device: str,
    dimension: int,
    item_count: int,
) -> str:
    return "\n".join(
        [
            "# CLIP Retrieval Report",
            "",
            "Execution status: **successfully executed**",
            "",
            f"- Model: `{model_name}`",
            f"- Device: `{device}`",
            f"- Embedding dimension: {dimension}",
            f"- Indexed items: {item_count}",
            f"- Evaluation queries: {metrics.query_count}",
            "",
            "Results depend on real model weights. The tiny local fixture is not a quality benchmark.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Recall@1 | {metrics.recall_at_1:.4f} |",
            f"| Recall@5 | {metrics.recall_at_5:.4f} |",
            f"| Recall@10 | {metrics.recall_at_10:.4f} |",
            f"| MRR | {metrics.mrr:.4f} |",
            f"| Median rank | {metrics.median_rank:.2f} |",
            f"| Mean rank | {metrics.mean_rank:.2f} |",
            f"| Query count | {metrics.query_count} |",
            "",
        ]
    )


def write_clip_retrieval_reports(
    metrics: RetrievalMetrics,
    model_name: str,
    report_path: Path,
    metrics_path: Path,
    *,
    device: str,
    dimension: int,
    item_count: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_clip_retrieval_report(
            metrics,
            model_name,
            device=device,
            dimension=dimension,
            item_count=item_count,
        ),
        encoding="utf-8",
    )
    metrics_data = asdict(metrics) | {
        "status": "successfully_executed",
        "model_name": model_name,
        "device": device,
        "embedding_dimension": dimension,
        "item_count": item_count,
    }
    metrics_path.write_text(
        json.dumps(metrics_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_clip_failure_reports(
    status: str, detail: str, report_path: Path, metrics_path: Path
) -> None:
    """Record a stable failed/unavailable execution state without stale metrics."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# CLIP Retrieval Report",
                "",
                f"Execution status: **{status}**",
                "",
                f"Detail: {detail}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"detail": detail, "status": status.replace(" ", "_")}, indent=2)
        + "\n",
        encoding="utf-8",
    )
