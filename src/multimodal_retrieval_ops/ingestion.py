"""Local image-caption metadata ingestion."""

import csv
from pathlib import Path

from .manifest import ManifestItem, ManifestValidationError, validate_image_paths, write_manifest
from .splitting import assign_splits


def ingest_local_directory(
    directory: Path,
    output_path: Path,
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> list[ManifestItem]:
    """Ingest ``captions.csv`` and referenced images from a local directory."""
    metadata_path = directory / "captions.csv"
    if not metadata_path.is_file():
        raise ManifestValidationError([f"caption metadata does not exist: {metadata_path}"])
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        missing = [name for name in ("image_file", "caption") if name not in (reader.fieldnames or [])]
        if missing:
            raise ManifestValidationError(
                [f"missing local metadata columns: {', '.join(missing)}"]
            )
        items = [
            ManifestItem(
                item_id=(row.get("item_id") or f"local-{index:03d}").strip(),
                image_path=(directory / row["image_file"].strip()).as_posix(),
                caption=row["caption"].strip(),
                split="train",
                source=(row.get("source") or "local-fixture").strip(),
            )
            for index, row in enumerate(reader, start=1)
        ]
    items = assign_splits(items, fractions=fractions, seed=seed)
    validate_image_paths(items)
    write_manifest(items, output_path)
    return items
