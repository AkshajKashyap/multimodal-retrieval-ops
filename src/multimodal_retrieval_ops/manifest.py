"""Dataset manifest contracts and validation."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("item_id", "image_path", "caption", "split", "source")
V2_REQUIRED_COLUMNS = ("image_id", "caption_id", "image_path", "caption", "split", "source")
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


@dataclass(frozen=True)
class ManifestItemV2:
    """Schema-v2 caption query targeting a stable image candidate."""

    image_id: str
    caption_id: str
    image_path: str
    caption: str
    split: str
    source: str

    @property
    def item_id(self) -> str:
        """Compatibility alias for legacy index code."""
        return self.image_id


ManifestRecord = ManifestItem | ManifestItemV2


def image_identity(item: ManifestRecord) -> str:
    return item.image_id if isinstance(item, ManifestItemV2) else item.item_id


def caption_identity(item: ManifestRecord) -> str:
    return item.caption_id if isinstance(item, ManifestItemV2) else item.item_id


class ManifestValidationError(ValueError):
    """Raised when a manifest violates its data contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Manifest validation failed:\n- " + "\n- ".join(errors))


def read_manifest_rows(path: Path) -> list[ManifestRecord]:
    """Read canonical rows from a UTF-8 CSV, checking its schema only."""
    if not path.is_file():
        raise ManifestValidationError([f"manifest file does not exist: {path}"])
    with path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        columns = tuple(reader.fieldnames or ())
        is_v2 = all(column in columns for column in V2_REQUIRED_COLUMNS)
        required = V2_REQUIRED_COLUMNS if is_v2 else REQUIRED_COLUMNS
        missing = [column for column in required if column not in columns]
        if missing:
            raise ManifestValidationError([f"missing required columns: {', '.join(missing)}"])
        item_class = ManifestItemV2 if is_v2 else ManifestItem
        return [
            item_class(**{column: row[column] for column in required}) for row in reader
        ]


def read_manifest(path: Path) -> list[ManifestRecord]:
    """Read and strictly validate a UTF-8 CSV manifest."""
    rows = read_manifest_rows(path)
    validate_items(rows)
    return rows


def validate_items(items: list[ManifestRecord]) -> None:
    """Validate manifest rows and raise one error containing every issue."""
    if items and isinstance(items[0], ManifestItemV2):
        validate_v2_items([item for item in items if isinstance(item, ManifestItemV2)])
        return
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


def validate_v2_items(items: list[ManifestItemV2]) -> None:
    """Validate caption identity and image-group invariants for schema v2."""
    errors: list[str] = []
    caption_ids: set[str] = set()
    image_paths: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    present_splits: set[str] = set()
    for row_number, item in enumerate(items, start=2):
        if not item.image_id.strip():
            errors.append(f"row {row_number}: image_id must be non-empty")
        if not item.caption_id.strip():
            errors.append(f"row {row_number}: caption_id must be non-empty")
        elif item.caption_id in caption_ids:
            errors.append(f"row {row_number}: duplicate caption_id '{item.caption_id}'")
        caption_ids.add(item.caption_id)
        if not item.caption.strip():
            errors.append(f"row {row_number}: caption must be non-empty")
        if not item.image_path.strip():
            errors.append(f"row {row_number}: image_path must be non-empty")
        elif Path(item.image_path).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            errors.append(f"row {row_number}: unsupported image extension '{Path(item.image_path).suffix}'")
        if item.split not in VALID_SPLITS:
            errors.append(f"row {row_number}: invalid split '{item.split}'")
        else:
            present_splits.add(item.split)
        previous_path = image_paths.setdefault(item.image_id, item.image_path)
        if previous_path != item.image_path:
            errors.append(f"row {row_number}: inconsistent image_path for image_id '{item.image_id}'")
        previous_split = image_splits.setdefault(item.image_id, item.split)
        if previous_split != item.split:
            errors.append(f"row {row_number}: inconsistent split for image_id '{item.image_id}'")
    for split in VALID_SPLITS:
        if split not in present_splits:
            errors.append(f"split '{split}' must contain at least one image")
    if errors:
        raise ManifestValidationError(errors)


def write_manifest(items: list[ManifestRecord], path: Path) -> None:
    """Write manifest rows using stable column and row ordering."""
    validate_items(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        is_v2 = bool(items) and isinstance(items[0], ManifestItemV2)
        fields = V2_REQUIRED_COLUMNS if is_v2 else REQUIRED_COLUMNS
        writer = csv.DictWriter(manifest_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(item) for item in items)


def migrate_to_v2(items: list[ManifestRecord]) -> list[ManifestItemV2]:
    """Deterministically migrate legacy rows; return v2 rows unchanged."""
    migrated: list[ManifestItemV2] = []
    for item in items:
        if isinstance(item, ManifestItemV2):
            migrated.append(item)
        else:
            migrated.append(
                ManifestItemV2(
                    image_id=item.item_id,
                    caption_id=f"{item.item_id}-caption-001",
                    image_path=item.image_path,
                    caption=item.caption,
                    split=item.split,
                    source=item.source,
                )
            )
    validate_v2_items(migrated)
    return migrated


def validate_image_paths(items: list[ManifestRecord], base_path: Path = Path(".")) -> None:
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
