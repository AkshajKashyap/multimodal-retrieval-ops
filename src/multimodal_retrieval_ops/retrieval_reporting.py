"""Deterministic baseline retrieval reports."""

from dataclasses import asdict
import json
from pathlib import Path

from .evaluation import RetrievalMetrics


def render_retrieval_report(metrics: RetrievalMetrics) -> str:
    """Render a concise deterministic Markdown evaluation report."""
    return "\n".join(
        [
            "# Baseline Retrieval Report",
            "",
            "Lexical bag-of-words baseline with a vocabulary fitted on train captions only.",
            "Validation and test captions are evaluated against validation/test candidates.",
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


def write_retrieval_reports(
    metrics: RetrievalMetrics, markdown_path: Path, metrics_path: Path
) -> None:
    """Write deterministic Markdown and machine-readable metrics."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_retrieval_report(metrics), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
