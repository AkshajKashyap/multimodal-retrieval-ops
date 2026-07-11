"""Opt-in Hugging Face Flickr8k ingestion and provenance reporting."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from .flickr8k import multi_caption_statistics
from .manifest import ManifestItemV2, write_manifest

DEFAULT_HF_FLICKR8K_DATASET = "jxie/flickr8k"
HF_SOURCE = "hf:jxie/flickr8k"
SPLIT_ORDER = ("train", "validation", "test")


class HFFlickr8kError(ValueError):
    """Expected, actionable HF ingestion failure."""


class HFDatasetUnavailableError(HFFlickr8kError):
    """Dataset/network/cache failure."""


class HFDataExecutionError(HFFlickr8kError):
    """Materialization or row-shape failure."""


def hfdata_dependencies_available() -> bool:
    return importlib.util.find_spec("datasets") is not None and importlib.util.find_spec("PIL") is not None


def hfdata_dependency_message() -> str:
    return (
        "Hugging Face dataset dependencies are unavailable. Install them with "
        '`python -m pip install -e ".[dev,clip,hfdata]"`.'
    )


@dataclass(frozen=True)
class HFFlickr8kProvenance:
    dataset_name: str
    requested_revision: str
    resolved_fingerprint: str
    source_split_image_counts: dict[str, int]
    source_split_caption_counts: dict[str, int]
    materialized_split_image_counts: dict[str, int]
    materialized_split_caption_counts: dict[str, int]
    unique_image_count: int
    caption_count: int
    captions_per_image_min: int
    captions_per_image_max: int
    captions_per_image_mean: float
    image_format_counts: dict[str, int]
    missing_or_invalid_image_count: int
    licensing_status: str
    materialization_path: str
    resolved_revision: str = "unavailable"


def _captions_from_row(row: Mapping[str, Any]) -> list[str]:
    numbered = [
        str(row[name])
        for name in sorted(row)
        if name.startswith("caption_") and row[name]
    ]
    if numbered:
        return [caption.strip() for caption in numbered if caption.strip()]
    value = next(
        (row[name] for name in ("captions", "caption", "text", "sentences") if name in row),
        None,
    )
    if isinstance(value, str):
        captions = [value]
    elif isinstance(value, Mapping) and "raw" in value:
        raw = value["raw"]
        captions = [str(caption) for caption in (raw if isinstance(raw, list) else [raw])]
    elif isinstance(value, Iterable):
        captions = [str(caption) for caption in value]
    else:
        raise HFDataExecutionError("Flickr8k row has no recognized caption field")
    captions = [caption.strip() for caption in captions if caption.strip()]
    if not captions:
        raise HFDataExecutionError("Flickr8k row contains no non-empty captions")
    return captions


def stable_hf_image_id(split: str, row_index: int, row: Mapping[str, Any]) -> str:
    """Create a deterministic ID without depending on machine-local cache paths."""
    source_id = next(
        (str(row[name]) for name in ("image_id", "id", "filename", "file_name") if row.get(name)),
        None,
    )
    if source_id:
        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
        return f"{split}-{digest}"
    return f"{split}-{row_index:06d}"


def _valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def materialize_hf_image(image: Any, path: Path, force: bool = False) -> tuple[str, bool]:
    """Atomically save a decoded image, skipping an already-valid destination."""
    if not force and _valid_image(path):
        suffix = path.suffix.lstrip(".").lower()
        return ("jpeg" if suffix in {"jpg", "jpeg"} else suffix), False
    try:
        from PIL import Image

        if isinstance(image, Mapping) and image.get("path"):
            with Image.open(image["path"]) as opened:
                converted = opened.convert("RGB").copy()
        elif isinstance(image, Mapping) and image.get("bytes"):
            from io import BytesIO

            with Image.open(BytesIO(image["bytes"])) as opened:
                converted = opened.convert("RGB").copy()
        else:
            converted = image.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        converted.save(temporary, format="JPEG", quality=95)
        os.replace(temporary, path)
        if not _valid_image(path):
            raise ValueError("saved file did not pass image verification")
        return "jpeg", True
    except Exception as error:
        raise HFDataExecutionError(f"could not materialize image at {path}") from error


Materializer = Callable[[Any, Path, bool], tuple[str, bool]]


def convert_hf_split_rows(
    rows: Iterable[Mapping[str, Any]],
    split: str,
    images_dir: Path,
    *,
    max_images: int | None = None,
    force: bool = False,
    materializer: Materializer = materialize_hf_image,
) -> tuple[list[ManifestItemV2], Counter[str], int]:
    """Convert source rows while preserving source order and split identity."""
    converted: list[ManifestItemV2] = []
    formats: Counter[str] = Counter()
    invalid = 0
    seen_image_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if max_images is not None and row_index >= max_images:
            break
        image_id = stable_hf_image_id(split, row_index, row)
        if image_id in seen_image_ids:
            raise HFDataExecutionError(f"duplicate source image identity in split '{split}': {image_id}")
        seen_image_ids.add(image_id)
        image_path = images_dir / split / f"{image_id}.jpg"
        try:
            image_format, _ = materializer(row.get("image"), image_path, force)
            formats[image_format.lower()] += 1
        except HFFlickr8kError:
            invalid += 1
            raise
        captions = _captions_from_row(row)
        converted.extend(
            ManifestItemV2(
                image_id=image_id,
                caption_id=f"{image_id}-caption-{caption_index:03d}",
                image_path=image_path.as_posix(),
                caption=caption,
                split=split,
                source=HF_SOURCE,
            )
            for caption_index, caption in enumerate(captions, start=1)
        )
    converted.sort(key=lambda row: (row.image_id, row.caption_id))
    return converted, formats, invalid


def load_hf_flickr8k(
    dataset_name: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> Any:
    """Lazily load the requested dataset with actionable expected failures."""
    if not hfdata_dependencies_available():
        raise HFFlickr8kError(hfdata_dependency_message())
    try:
        from datasets import DownloadConfig, load_dataset

        download_config = DownloadConfig(local_files_only=True) if local_files_only else None
        return load_dataset(
            dataset_name,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            download_config=download_config,
        )
    except Exception as error:
        mode = "local Hugging Face cache" if local_files_only else "Hugging Face Hub"
        raise HFDatasetUnavailableError(
            f"Could not load dataset '{dataset_name}' from {mode}. "
            "Check network/cache availability and the requested revision."
        ) from error


def ingest_hf_flickr8k(
    *,
    dataset_name: str,
    revision: str | None,
    cache_dir: Path | None,
    output_manifest: Path,
    images_dir: Path,
    provenance_path: Path,
    max_images_per_split: int | None,
    local_files_only: bool,
    force: bool,
) -> tuple[list[ManifestItemV2], HFFlickr8kProvenance]:
    dataset = load_hf_flickr8k(
        dataset_name, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only
    )
    rows: list[ManifestItemV2] = []
    format_counts: Counter[str] = Counter()
    invalid_count = 0
    source_image_counts: dict[str, int] = {}
    source_caption_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}
    caption_counts: dict[str, int] = {}
    fingerprints: list[str] = []
    for split in SPLIT_ORDER:
        if split not in dataset:
            raise HFDataExecutionError(f"dataset '{dataset_name}' is missing required split '{split}'")
        source_image_counts[split] = len(dataset[split])
        source_caption_counts[split] = (
            len(_captions_from_row(dataset[split][0])) * len(dataset[split])
            if len(dataset[split])
            else 0
        )
        split_rows, split_formats, split_invalid = convert_hf_split_rows(
            dataset[split],
            split,
            images_dir,
            max_images=max_images_per_split,
            force=force,
        )
        rows.extend(split_rows)
        format_counts.update(split_formats)
        invalid_count += split_invalid
        image_counts[split] = len({row.image_id for row in split_rows})
        caption_counts[split] = len(split_rows)
        fingerprints.append(f"{split}:{getattr(dataset[split], '_fingerprint', 'unknown')}")
    rows.sort(key=lambda row: (SPLIT_ORDER.index(row.split), row.image_id, row.caption_id))
    write_manifest(rows, output_manifest)
    stats = multi_caption_statistics(rows)
    resolved = hashlib.sha256("|".join(fingerprints).encode("utf-8")).hexdigest()
    resolved_revision = "unavailable"
    for split in SPLIT_ORDER:
        cache_files = getattr(dataset[split], "cache_files", [])
        if cache_files:
            candidate = Path(cache_files[0]["filename"]).parent.name
            if len(candidate) >= 12:
                resolved_revision = candidate
                break
    provenance = HFFlickr8kProvenance(
        dataset_name=dataset_name,
        requested_revision=revision or "default",
        resolved_fingerprint=resolved,
        source_split_image_counts=source_image_counts,
        source_split_caption_counts=source_caption_counts,
        materialized_split_image_counts=image_counts,
        materialized_split_caption_counts=caption_counts,
        unique_image_count=stats.unique_images,
        caption_count=stats.caption_queries,
        captions_per_image_min=stats.captions_per_image_min,
        captions_per_image_max=stats.captions_per_image_max,
        captions_per_image_mean=stats.captions_per_image_mean,
        image_format_counts=dict(sorted(format_counts.items())),
        missing_or_invalid_image_count=invalid_count,
        licensing_status="unresolved; dataset source does not clearly expose licensing information",
        materialization_path=images_dir.as_posix(),
        resolved_revision=resolved_revision,
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, provenance


def load_hf_provenance(path: Path) -> HFFlickr8kProvenance:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("materialized_split_image_counts", data["source_split_image_counts"])
        data.setdefault("materialized_split_caption_counts", data["source_split_caption_counts"])
        data.setdefault("resolved_revision", "unavailable")
        return HFFlickr8kProvenance(**data)
    except (OSError, ValueError, TypeError) as error:
        raise HFDatasetUnavailableError(
            f"HF Flickr8k provenance is unavailable or invalid: {path}"
        ) from error


def render_hf_dataset_report(provenance: HFFlickr8kProvenance, status: str = "success") -> str:
    split_lines = [
        f"- {split}: {provenance.source_split_image_counts.get(split, 0)} images, "
        f"{provenance.source_split_caption_counts.get(split, 0)} captions source; "
        f"{provenance.materialized_split_image_counts.get(split, 0)} images, "
        f"{provenance.materialized_split_caption_counts.get(split, 0)} captions materialized"
        for split in SPLIT_ORDER
    ]
    return "\n".join(
        [
            "# Hugging Face Flickr8k Dataset Report",
            "",
            f"Run state: **{status}**",
            "",
            f"- Dataset: `{provenance.dataset_name}`",
            f"- Requested revision: `{provenance.requested_revision}`",
            f"- Resolved revision: `{provenance.resolved_revision}`",
            f"- Resolved fingerprint: `{provenance.resolved_fingerprint}`",
            f"- Unique images: {provenance.unique_image_count}",
            f"- Caption rows: {provenance.caption_count}",
            f"- Captions per image (min/max/mean): {provenance.captions_per_image_min}/"
            f"{provenance.captions_per_image_max}/{provenance.captions_per_image_mean:.2f}",
            f"- Image formats: {json.dumps(provenance.image_format_counts, sort_keys=True)}",
            f"- Missing or invalid images: {provenance.missing_or_invalid_image_count}",
            f"- Materialization path: `{provenance.materialization_path}`",
            f"- Licensing status: **{provenance.licensing_status}**",
            "",
            "## Official source splits",
            "",
            *split_lines,
            "",
            "Downloaded images remain local and are not redistributed by this repository.",
            "",
        ]
    )


def write_hf_failure_report(path: Path, status: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Hugging Face Flickr8k Dataset Report",
                "",
                f"Run state: **{status}**",
                "",
                f"Detail: {detail}",
                "",
                "Licensing status: **unresolved**",
                "",
            ]
        ),
        encoding="utf-8",
    )
