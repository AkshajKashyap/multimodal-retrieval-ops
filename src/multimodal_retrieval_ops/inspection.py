"""Dataset quality statistics and deterministic reports."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .manifest import ManifestRecord, SUPPORTED_IMAGE_EXTENSIONS, VALID_SPLITS, caption_identity


@dataclass(frozen=True)
class DatasetStatistics:
    row_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    caption_length_min: int
    caption_length_max: int
    caption_length_mean: float
    missing_image_count: int
    duplicate_item_id_count: int
    duplicate_caption_count: int


def inspect_items(items: list[ManifestRecord], base_path: Path = Path(".")) -> DatasetStatistics:
    """Compute lightweight, stable quality metrics for manifest rows."""
    lengths = [len(item.caption.split()) for item in items]
    item_counts = Counter(caption_identity(item) for item in items)
    caption_counts = Counter(item.caption for item in items)
    missing = 0
    for item in items:
        path = Path(item.image_path)
        resolved = path if path.is_absolute() else base_path / path
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS or not resolved.is_file():
            missing += 1
    return DatasetStatistics(
        row_count=len(items),
        split_counts=dict(Counter(item.split for item in items)),
        source_counts=dict(Counter(item.source for item in items)),
        caption_length_min=min(lengths, default=0),
        caption_length_max=max(lengths, default=0),
        caption_length_mean=sum(lengths) / len(lengths) if lengths else 0.0,
        missing_image_count=missing,
        duplicate_item_id_count=sum(count - 1 for count in item_counts.values() if count > 1),
        duplicate_caption_count=sum(count - 1 for count in caption_counts.values() if count > 1),
    )


def render_dataset_report(statistics: DatasetStatistics) -> str:
    """Render dataset quality metrics as deterministic Markdown."""
    lines = [
        "# Dataset Inspection Report",
        "",
        f"Total rows: **{statistics.row_count}**",
        "",
        "## Split counts",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {split} | {statistics.split_counts.get(split, 0)} |" for split in VALID_SPLITS
    )
    lines.extend(["", "## Source counts", "", "| Source | Rows |", "| --- | ---: |"])
    lines.extend(
        f"| {source} | {count} |" for source, count in sorted(statistics.source_counts.items())
    )
    lines.extend(
        [
            "",
            "## Caption lengths (words)",
            "",
            f"- Minimum: {statistics.caption_length_min}",
            f"- Maximum: {statistics.caption_length_max}",
            f"- Mean: {statistics.caption_length_mean:.2f}",
            "",
            "## Quality checks",
            "",
            f"- Missing images: {statistics.missing_image_count}",
            f"- Duplicate item IDs: {statistics.duplicate_item_id_count}",
            f"- Duplicate captions: {statistics.duplicate_caption_count}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_dataset_report(statistics: DatasetStatistics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dataset_report(statistics), encoding="utf-8")
