"""Deterministic manifest reporting."""

from collections import Counter
from pathlib import Path

from .manifest import ManifestItem, VALID_SPLITS


def render_manifest_report(items: list[ManifestItem]) -> str:
    """Render a stable Markdown summary."""
    split_counts = Counter(item.split for item in items)
    source_counts = Counter(item.source for item in items)
    lines = [
        "# Demo Manifest Summary",
        "",
        f"Total image-caption pairs: **{len(items)}**",
        "",
        "## Splits",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {split} | {split_counts[split]} |" for split in VALID_SPLITS)
    lines.extend(["", "## Sources", "", "| Source | Rows |", "| --- | ---: |"])
    lines.extend(f"| {source} | {count} |" for source, count in sorted(source_counts.items()))
    return "\n".join(lines) + "\n"


def write_manifest_report(items: list[ManifestItem], path: Path) -> None:
    """Write a deterministic Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_manifest_report(items), encoding="utf-8")
