"""Dataset manifest contracts and validation."""

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("item_id", "image_path", "caption", "split", "source")
VALID_SPLITS = ("train", "validation", "test")
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


@dataclass(frozen=True)
class ManifestItem:
    """One image-caption pair in a dataset manifest."""

    item_id: str
    image_path: str
    caption: str
    split: str
    source: str


class ManifestValidationError(ValueError):
    """Raised when a manifest violates its data contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Manifest validation failed:\n- " + "\n- ".join(errors))


def read_manifest_rows(path: Path) -> list[ManifestItem]:
    """Read canonical rows from a UTF-8 CSV, checking its schema only."""
    if not path.is_file():
        raise ManifestValidationError([f"manifest file does not exist: {path}"])
    with path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ManifestValidationError([f"missing required columns: {', '.join(missing)}"])
        return [
            ManifestItem(**{column: row[column] for column in REQUIRED_COLUMNS}) for row in reader
        ]


def read_manifest(path: Path) -> list[ManifestItem]:
    """Read and strictly validate a UTF-8 CSV manifest."""
    rows = read_manifest_rows(path)
    validate_items(rows)
    return rows


def validate_items(items: list[ManifestItem]) -> None:
    """Validate manifest rows and raise one error containing every issue."""
    errors: list[str] = []
    seen: set[str] = set()
    present_splits: set[str] = set()
    for row_number, item in enumerate(items, start=2):
        if not item.item_id.strip():
            errors.append(f"row {row_number}: item_id must be non-empty")
        elif item.item_id in seen:
            errors.append(f"row {row_number}: duplicate item_id '{item.item_id}'")
        seen.add(item.item_id)
        if item.split not in VALID_SPLITS:
            errors.append(f"row {row_number}: invalid split '{item.split}'")
        else:
            present_splits.add(item.split)
        if not item.caption.strip():
            errors.append(f"row {row_number}: caption must be non-empty")
        if not item.image_path.strip():
            errors.append(f"row {row_number}: image_path must be non-empty")
    for split in VALID_SPLITS:
        if split not in present_splits:
            errors.append(f"split '{split}' must contain at least one row")
    if errors:
        raise ManifestValidationError(errors)


def write_manifest(items: list[ManifestItem], path: Path) -> None:
    """Write manifest rows using stable column and row ordering."""
    validate_items(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.__dict__ for item in items)


def validate_image_paths(items: list[ManifestItem], base_path: Path = Path(".")) -> None:
    """Validate that image references use supported extensions and exist locally."""
    errors: list[str] = []
    for row_number, item in enumerate(items, start=2):
        image_path = Path(item.image_path)
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            errors.append(
                f"row {row_number}: unsupported image extension '{image_path.suffix}' "
                f"(supported: {supported})"
            )
        resolved = image_path if image_path.is_absolute() else base_path / image_path
        if not resolved.is_file():
            errors.append(f"row {row_number}: image does not exist: {item.image_path}")
    if errors:
        raise ManifestValidationError(errors)
