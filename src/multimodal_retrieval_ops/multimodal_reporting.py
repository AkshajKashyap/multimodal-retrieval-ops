"""Deterministic reporting for placeholder multimodal retrieval."""

from dataclasses import asdict
import json
from pathlib import Path

from .evaluation import RetrievalMetrics


def render_multimodal_report(metrics: RetrievalMetrics, backend_name: str) -> str:
    """Render a stable Markdown report without model-quality claims."""
    return "\n".join(
        [
            "# Multimodal Baseline Report",
            "",
            f"Backend: `{backend_name}`",
            "",
            "This deterministic placeholder hashes text tokens and local file features into a",
            "shared space. It is an architecture check, not CLIP or a measure of model quality.",
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


def write_multimodal_reports(
    metrics: RetrievalMetrics,
    backend_name: str,
    markdown_path: Path,
    metrics_path: Path,
) -> None:
    """Write stable human- and machine-readable evaluation reports."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_multimodal_report(metrics, backend_name), encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
