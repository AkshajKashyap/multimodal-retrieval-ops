import csv
from pathlib import Path
import subprocess
import sys

import pytest

from multimodal_retrieval_ops import __version__
from multimodal_retrieval_ops.demo import generate_demo_manifest
from multimodal_retrieval_ops.manifest import ManifestValidationError, read_manifest


def test_version_exists() -> None:
    assert __version__


def test_demo_manifest_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    generate_demo_manifest(first)
    generate_demo_manifest(second)
    assert first.read_bytes() == second.read_bytes()


def test_valid_manifest_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    generate_demo_manifest(manifest)
    assert len(read_manifest(manifest)) == 4


def test_invalid_manifest_fails_clearly(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["item_id", "image_path", "caption", "split", "source"])
        writer.writerow(["same", "", "", "wrong", "demo"])
        writer.writerow(["same", "image.jpg", "caption", "train", "demo"])
    with pytest.raises(ManifestValidationError) as error:
        read_manifest(manifest)
    message = str(error.value)
    assert "duplicate item_id 'same'" in message
    assert "caption must be non-empty" in message
    assert "image_path must be non-empty" in message
    assert "invalid split 'wrong'" in message
    assert "split 'validation' must contain at least one row" in message


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "multimodal_retrieval_ops.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_smoke_workflow(tmp_path: Path) -> None:
    manifest = tmp_path / "demo.csv"
    report = tmp_path / "report.md"
    assert run_cli("--version").returncode == 0
    assert run_cli("project-info").returncode == 0
    generated = run_cli("generate-demo-manifest", "--output", str(manifest))
    assert generated.returncode == 0
    assert run_cli("validate-manifest", "--manifest", str(manifest)).returncode == 0
    reported = run_cli(
        "generate-manifest-report",
        "--manifest",
        str(manifest),
        "--output",
        str(report),
    )
    assert reported.returncode == 0
    assert "Total image-caption pairs: **4**" in report.read_text(encoding="utf-8")


def test_cli_invalid_manifest_returns_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.csv"
    manifest.write_text("item_id,image_path\n1,x.jpg\n", encoding="utf-8")
    result = run_cli("validate-manifest", "--manifest", str(manifest))
    assert result.returncode != 0
    assert "missing required columns" in result.stderr
