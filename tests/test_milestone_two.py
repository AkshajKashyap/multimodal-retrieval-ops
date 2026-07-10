from dataclasses import replace
from pathlib import Path

import pytest

from multimodal_retrieval_ops.ingestion import ingest_local_directory
from multimodal_retrieval_ops.inspection import (
    inspect_items,
    render_dataset_report,
    write_dataset_report,
)
from multimodal_retrieval_ops.manifest import (
    ManifestItem,
    ManifestValidationError,
    read_manifest,
    validate_image_paths,
)
from multimodal_retrieval_ops.splitting import assign_splits
from test_milestone_one import run_cli

FIXTURE = Path("tests/fixtures/local_dataset")


def test_local_fixture_ingestion(tmp_path: Path) -> None:
    output = tmp_path / "ingested.csv"
    items = ingest_local_directory(FIXTURE, output)
    assert len(items) == 5
    assert len(read_manifest(output)) == 5
    assert {item.source for item in items} == {"local-fixture"}


def test_deterministic_splitting(tmp_path: Path) -> None:
    items = ingest_local_directory(FIXTURE, tmp_path / "source.csv")
    first = assign_splits(items, seed=7)
    second = assign_splits(items, seed=7)
    assert first == second
    assert {item.split for item in first} == {"train", "validation", "test"}
    assert len({item.item_id for item in first}) == len(first)


def test_image_path_validation() -> None:
    items = ingest_local_directory(FIXTURE, Path("/tmp/local-fixture-test.csv"))
    validate_image_paths(items)


def test_invalid_extension_handling(tmp_path: Path) -> None:
    invalid = tmp_path / "image.gif"
    invalid.write_text("placeholder", encoding="utf-8")
    item = ManifestItem("one", str(invalid), "Caption", "train", "test")
    with pytest.raises(ManifestValidationError, match="unsupported image extension '.gif'"):
        validate_image_paths([item])


def test_duplicate_item_id_detection(tmp_path: Path) -> None:
    items = ingest_local_directory(FIXTURE, tmp_path / "source.csv")
    duplicate = [*items, replace(items[0], caption="Another caption")]
    with pytest.raises(ValueError, match="duplicate item_id"):
        assign_splits(duplicate)
    assert inspect_items(duplicate).duplicate_item_id_count == 1


def test_dataset_inspection_report_generation(tmp_path: Path) -> None:
    items = ingest_local_directory(FIXTURE, tmp_path / "source.csv")
    statistics = inspect_items(items)
    report = tmp_path / "inspection.md"
    write_dataset_report(statistics, report)
    assert statistics.row_count == 5
    assert statistics.missing_image_count == 0
    assert statistics.duplicate_caption_count == 0
    assert report.read_text(encoding="utf-8") == render_dataset_report(statistics)


def test_milestone_two_cli_smoke(tmp_path: Path) -> None:
    ingested = tmp_path / "ingested.csv"
    split = tmp_path / "split.csv"
    report = tmp_path / "report.md"
    ingest_result = run_cli(
        "ingest-local-fixture", "--directory", str(FIXTURE), "--output", str(ingested)
    )
    assert ingest_result.returncode == 0, ingest_result.stderr
    split_result = run_cli("split-manifest", "--manifest", str(ingested), "--output", str(split))
    assert split_result.returncode == 0, split_result.stderr
    inspect_result = run_cli(
        "inspect-manifest", "--manifest", str(split), "--output", str(report)
    )
    assert inspect_result.returncode == 0, inspect_result.stderr
    assert "Missing images: 0" in report.read_text(encoding="utf-8")


def test_inspection_cli_reports_duplicates_without_rejecting_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    ingest_local_directory(FIXTURE, source)
    lines = source.read_text(encoding="utf-8").splitlines()
    source.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    report = tmp_path / "duplicates.md"
    result = run_cli("inspect-manifest", "--manifest", str(source), "--output", str(report))
    assert result.returncode == 0, result.stderr
    assert "Duplicate item IDs: 1" in report.read_text(encoding="utf-8")
