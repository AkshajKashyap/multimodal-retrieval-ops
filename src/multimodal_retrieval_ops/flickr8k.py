"""Local Flickr8k ingestion and grouped benchmark subsetting."""

import csv
from dataclasses import dataclass
from pathlib import Path
import random

from .manifest import ManifestItemV2, ManifestValidationError, image_identity, write_manifest
from .splitting import assign_splits


def _parse_token_lines(lines: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            reference, caption = line.rstrip("\n").split("\t", maxsplit=1)
        except ValueError as error:
            raise ManifestValidationError(
                [f"caption line {line_number}: expected tab-separated image and caption"]
            ) from error
        parsed.append((reference.split("#", maxsplit=1)[0], caption))
    return parsed


def parse_flickr8k_captions(path: Path) -> list[tuple[str, str]]:
    """Parse Flickr8k token TSV or common image/caption CSV metadata."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise ManifestValidationError([f"caption metadata is empty: {path}"])
    if "\t" in lines[0]:
        return _parse_token_lines(lines)
    reader = csv.DictReader(lines)
    fields = {field.lower(): field for field in (reader.fieldnames or [])}
    image_field = next(
        (fields[name] for name in ("image", "image_name", "filename", "file") if name in fields),
        None,
    )
    caption_field = next(
        (fields[name] for name in ("caption", "comment", "description") if name in fields),
        None,
    )
    if image_field is None or caption_field is None:
        raise ManifestValidationError(
            ["Flickr8k CSV requires an image/filename column and caption/comment column"]
        )
    return [(row[image_field], row[caption_field]) for row in reader]


def ingest_flickr8k(
    images_dir: Path,
    captions_file: Path,
    output_path: Path,
    *,
    seed: int = 42,
) -> list[ManifestItemV2]:
    """Create a schema-v2 manifest from user-provided local Flickr8k files."""
    pairs = parse_flickr8k_captions(captions_file)
    caption_counts: dict[str, int] = {}
    rows: list[ManifestItemV2] = []
    for filename, caption in pairs:
        filename = filename.strip()
        image_id = Path(filename).stem
        caption_counts[image_id] = caption_counts.get(image_id, 0) + 1
        rows.append(
            ManifestItemV2(
                image_id=image_id,
                caption_id=f"{image_id}-caption-{caption_counts[image_id]:03d}",
                image_path=(images_dir / filename).as_posix(),
                caption=caption.strip(),
                split="train",
                source="flickr8k",
            )
        )
    missing = sorted({row.image_path for row in rows if not Path(row.image_path).is_file()})
    if missing:
        preview = ", ".join(missing[:3])
        raise ManifestValidationError([f"missing Flickr8k images ({len(missing)}): {preview}"])
    split_rows = assign_splits(rows, seed=seed)
    result = [row for row in split_rows if isinstance(row, ManifestItemV2)]
    write_manifest(result, output_path)
    return result


def create_benchmark_subset(
    rows: list[ManifestItemV2],
    max_images: int,
    *,
    seed: int = 42,
) -> list[ManifestItemV2]:
    """Seed-select image groups and deterministically reassign grouped splits."""
    if max_images < 3:
        raise ValueError("benchmark subset requires at least three images")
    image_ids = sorted({image_identity(row) for row in rows})
    random.Random(seed).shuffle(image_ids)
    selected = set(image_ids[:max_images])
    subset = [row for row in rows if row.image_id in selected]
    split_rows = assign_splits(subset, seed=seed)
    return [row for row in split_rows if isinstance(row, ManifestItemV2)]


@dataclass(frozen=True)
class MultiCaptionStatistics:
    unique_images: int
    caption_queries: int
    captions_per_image_min: int
    captions_per_image_max: int
    captions_per_image_mean: float


def multi_caption_statistics(rows: list[ManifestItemV2]) -> MultiCaptionStatistics:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.image_id] = counts.get(row.image_id, 0) + 1
    values = list(counts.values())
    return MultiCaptionStatistics(
        unique_images=len(counts),
        caption_queries=len(rows),
        captions_per_image_min=min(values, default=0),
        captions_per_image_max=max(values, default=0),
        captions_per_image_mean=sum(values) / len(values) if values else 0.0,
    )


def render_flickr8k_report(rows: list[ManifestItemV2], title: str = "Flickr8k Dataset Report") -> str:
    stats = multi_caption_statistics(rows)
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- Unique images: {stats.unique_images}",
            f"- Caption queries: {stats.caption_queries}",
            f"- Captions per image (min/max/mean): "
            f"{stats.captions_per_image_min}/{stats.captions_per_image_max}/"
            f"{stats.captions_per_image_mean:.2f}",
            "",
        ]
    )
