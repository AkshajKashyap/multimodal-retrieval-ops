"""Deterministic optional-backend and retrieval reporting."""

from dataclasses import asdict
import json
from pathlib import Path

from .clip_backend import DEFAULT_CLIP_MODEL
from .evaluation import RetrievalMetrics


def render_clip_backend_report(available: bool) -> str:
    status = "available" if available else "not installed"
    return "\n".join(
        [
            "# CLIP Backend Report",
            "",
            f"Optional dependency status: **{status}**",
            "",
            f"Default model: `{DEFAULT_CLIP_MODEL}`",
            "",
            "The base installation does not require Torch, Transformers, Pillow, or model weights.",
            "CLIP model commands use locally cached weights unless `--allow-download` is provided.",
            "",
        ]
    )


def write_clip_backend_report(available: bool, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_clip_backend_report(available), encoding="utf-8")


def render_clip_retrieval_report(metrics: RetrievalMetrics, model_name: str) -> str:
    return "\n".join(
        [
            "# CLIP Retrieval Report",
            "",
            f"Model: `{model_name}`",
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
    metrics: RetrievalMetrics, model_name: str, report_path: Path, metrics_path: Path
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_clip_retrieval_report(metrics, model_name), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
