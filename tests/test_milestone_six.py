from dataclasses import replace
from pathlib import Path

import pytest

from multimodal_retrieval_ops.benchmark import run_clip_benchmark
from multimodal_retrieval_ops.deterministic_image_encoder import DeterministicImageEncoder
from multimodal_retrieval_ops.deterministic_text_encoder import DeterministicTextEncoder
from multimodal_retrieval_ops.flickr8k import (
    create_benchmark_subset,
    ingest_flickr8k,
    multi_caption_statistics,
)
from multimodal_retrieval_ops.manifest import (
    ManifestItem,
    ManifestItemV2,
    ManifestValidationError,
    migrate_to_v2,
    read_manifest,
    validate_items,
    write_manifest,
)
from multimodal_retrieval_ops.multimodal_evaluation import evaluate_multimodal_index
from multimodal_retrieval_ops.multimodal_index import build_multimodal_index
from multimodal_retrieval_ops.splitting import assign_splits
from test_milestone_one import run_cli

IMAGES = Path("tests/fixtures/local_dataset/images")
CAPTIONS = Path("tests/fixtures/multicaption/captions.txt")


def v2_rows() -> list[ManifestItemV2]:
    values = [
        ("red", "red-1", "red_square.jpg", "A red square.", "train"),
        ("red", "red-2", "red_square.jpg", "A crimson block.", "train"),
        ("blue", "blue-1", "blue_circle.jpeg", "A blue circle.", "validation"),
        ("blue", "blue-2", "blue_circle.jpeg", "A round blue shape.", "validation"),
        ("green", "green-1", "green_triangle.png", "A green triangle.", "test"),
        ("green", "green-2", "green_triangle.png", "A three-sided shape.", "test"),
    ]
    return [
        ManifestItemV2(image_id, caption_id, str(IMAGES / filename), caption, split, "fixture")
        for image_id, caption_id, filename, caption, split in values
    ]


def test_repeated_image_id_with_unique_captions_passes() -> None:
    validate_items(v2_rows())


def test_duplicate_caption_id_fails() -> None:
    rows = v2_rows()
    rows[-1] = replace(rows[-1], caption_id=rows[0].caption_id)
    with pytest.raises(ManifestValidationError, match="duplicate caption_id 'red-1'"):
        validate_items(rows)


def test_inconsistent_image_path_fails() -> None:
    rows = v2_rows()
    rows[1] = replace(rows[1], image_path=str(IMAGES / "blue_circle.jpeg"))
    with pytest.raises(ManifestValidationError, match="inconsistent image_path"):
        validate_items(rows)


def test_inconsistent_image_split_fails() -> None:
    rows = v2_rows()
    rows[1] = replace(rows[1], split="test")
    with pytest.raises(ManifestValidationError, match="inconsistent split"):
        validate_items(rows)


def test_grouped_split_is_deterministic_and_prevents_leakage() -> None:
    first = assign_splits(v2_rows(), seed=7)
    second = assign_splits(v2_rows(), seed=7)
    assert first == second
    splits_by_image: dict[str, set[str]] = {}
    for row in first:
        assert isinstance(row, ManifestItemV2)
        splits_by_image.setdefault(row.image_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in splits_by_image.values())


def test_index_has_one_candidate_per_image_and_all_caption_queries() -> None:
    rows = v2_rows()
    text_encoder = DeterministicTextEncoder()
    index = build_multimodal_index(rows, DeterministicImageEncoder(), text_encoder)
    assert len(index.entries) == 3
    assert len(index.queries or []) == 6
    assert {entry.item_id for entry in index.entries} == {"red", "blue", "green"}
    metrics, ranks = evaluate_multimodal_index(text_encoder, index)
    assert metrics.query_count == 4
    assert len(ranks) == 4


def test_legacy_migration_is_deterministic(tmp_path: Path) -> None:
    legacy = [
        ManifestItem("one", "one.jpg", "One caption", "train", "legacy"),
        ManifestItem("two", "two.png", "Two caption", "validation", "legacy"),
        ManifestItem("three", "three.webp", "Three caption", "test", "legacy"),
    ]
    first = migrate_to_v2(legacy)
    second = migrate_to_v2(legacy)
    assert first == second
    assert first[0].caption_id == "one-caption-001"
    path = tmp_path / "v2.csv"
    write_manifest(first, path)
    assert read_manifest(path) == first


def test_flickr_ingestion_and_subset_preserve_groups(tmp_path: Path) -> None:
    manifest = tmp_path / "flickr.csv"
    rows = ingest_flickr8k(IMAGES, CAPTIONS, manifest)
    stats = multi_caption_statistics(rows)
    assert stats.unique_images == 3
    assert stats.caption_queries == 6
    assert stats.captions_per_image_min == stats.captions_per_image_max == 2
    subset = create_benchmark_subset(rows, 3, seed=42)
    assert subset == create_benchmark_subset(rows, 3, seed=42)
    for image_id in {row.image_id for row in subset}:
        assert len({row.split for row in subset if row.image_id == image_id}) == 1


def test_schema_v2_cli_smoke(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.csv"
    migrated_path = tmp_path / "v2.csv"
    report = tmp_path / "migration.md"
    legacy = [
        ManifestItem("one", "one.jpg", "One", "train", "legacy"),
        ManifestItem("two", "two.png", "Two", "validation", "legacy"),
        ManifestItem("three", "three.webp", "Three", "test", "legacy"),
    ]
    write_manifest(legacy, legacy_path)
    migrated = run_cli(
        "migrate-manifest-v2",
        "--manifest",
        str(legacy_path),
        "--output",
        str(migrated_path),
        "--report-output",
        str(report),
    )
    assert migrated.returncode == 0, migrated.stderr
    assert run_cli("validate-manifest", "--manifest", str(migrated_path)).returncode == 0
    flickr_path = tmp_path / "flickr.csv"
    flickr_report = tmp_path / "flickr.md"
    ingested = run_cli(
        "ingest-flickr8k",
        "--images-dir",
        str(IMAGES),
        "--captions-file",
        str(CAPTIONS),
        "--output",
        str(flickr_path),
        "--report-output",
        str(flickr_report),
    )
    assert ingested.returncode == 0, ingested.stderr
    assert len(read_manifest(flickr_path)) == 6


class FakeSharedBackend:
    backend_name = "fake-shared"
    backend_version = "1"
    model_name = "fake/model"
    device = "cpu"
    dimension = 64

    def __init__(self) -> None:
        self.text = DeterministicTextEncoder(dimension=self.dimension)
        self.image = DeterministicImageEncoder(dimension=self.dimension)

    def encode_text(self, text: str) -> list[float]:
        return self.text.encode_text(text)

    def encode_image(self, image_path: str) -> list[float]:
        return self.image.encode_image(image_path)

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.encode_text(text) for text in texts]

    def encode_images(self, paths: list[str]) -> list[list[float]]:
        return [self.encode_image(path) for path in paths]


def test_benchmark_comparison_and_cache_are_download_free(tmp_path: Path) -> None:
    rows = v2_rows()
    paths = {
        "cache_path": tmp_path / "cache.json",
        "index_path": tmp_path / "index.json",
        "report_path": tmp_path / "report.md",
        "metrics_path": tmp_path / "metrics.json",
    }
    metrics, first_hit = run_clip_benchmark(rows, FakeSharedBackend(), **paths)
    _, second_hit = run_clip_benchmark(rows, FakeSharedBackend(), **paths)
    report = paths["report_path"].read_text(encoding="utf-8")
    assert metrics.query_count == 4
    assert first_hit is False and second_hit is True
    assert "Lexical" in report
    assert "Deterministic placeholder" in report
    assert "Zero-shot CLIP" in report
