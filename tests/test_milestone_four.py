import json
import math
from pathlib import Path

import pytest

from multimodal_retrieval_ops.deterministic_image_encoder import DeterministicImageEncoder
from multimodal_retrieval_ops.deterministic_text_encoder import DeterministicTextEncoder
from multimodal_retrieval_ops.evaluation import metrics_from_ranks
from multimodal_retrieval_ops.ingestion import ingest_local_directory
from multimodal_retrieval_ops.multimodal_evaluation import evaluate_multimodal_index
from multimodal_retrieval_ops.multimodal_index import (
    build_multimodal_index,
    load_multimodal_index,
    search_multimodal_index,
    write_multimodal_index,
)
from multimodal_retrieval_ops.multimodal_reporting import (
    render_multimodal_report,
    write_multimodal_reports,
)
from test_milestone_one import run_cli

FIXTURE = Path("tests/fixtures/local_dataset")
DIMENSION = 32


def test_encoder_output_dimension_and_determinism() -> None:
    text_encoder = DeterministicTextEncoder(dimension=DIMENSION)
    image_encoder = DeterministicImageEncoder(dimension=DIMENSION)
    image_path = str(FIXTURE / "images/red_square.jpg")
    assert len(text_encoder.encode_text("red square")) == DIMENSION
    assert text_encoder.encode_text("red square") == text_encoder.encode_text("red square")
    assert len(image_encoder.encode_image(image_path)) == DIMENSION
    assert image_encoder.encode_image(image_path) == image_encoder.encode_image(image_path)


def test_encoder_vectors_are_l2_normalized() -> None:
    text_encoder = DeterministicTextEncoder(dimension=DIMENSION)
    image_encoder = DeterministicImageEncoder(dimension=DIMENSION)
    text_vector = text_encoder.encode_text("red square")
    image_vector = image_encoder.encode_image(str(FIXTURE / "images/red_square.jpg"))
    assert math.sqrt(sum(value * value for value in text_vector)) == pytest.approx(1.0)
    assert math.sqrt(sum(value * value for value in image_vector)) == pytest.approx(1.0)


def test_multimodal_index_build_is_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    items = ingest_local_directory(FIXTURE, manifest)
    text_encoder = DeterministicTextEncoder(dimension=DIMENSION)
    image_encoder = DeterministicImageEncoder(dimension=DIMENSION)
    first = build_multimodal_index(items, image_encoder, text_encoder)
    second = build_multimodal_index(items, image_encoder, text_encoder)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_multimodal_index(first, first_path)
    write_multimodal_index(second, second_path)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert len({entry.item_id for entry in first.entries}) == len(items)


def test_multimodal_search_output_schema(tmp_path: Path) -> None:
    items = ingest_local_directory(FIXTURE, tmp_path / "manifest.csv")
    text_encoder = DeterministicTextEncoder(dimension=DIMENSION)
    index = build_multimodal_index(
        items, DeterministicImageEncoder(dimension=DIMENSION), text_encoder
    )
    result = search_multimodal_index("red square", text_encoder, index, k=1)[0]
    assert result.item_id == "fixture-001"
    assert isinstance(result.score, float)
    assert result.caption and result.image_path and result.split


def test_multimodal_metric_calculation(tmp_path: Path) -> None:
    items = ingest_local_directory(FIXTURE, tmp_path / "manifest.csv")
    text_encoder = DeterministicTextEncoder(dimension=DIMENSION)
    index = build_multimodal_index(
        items, DeterministicImageEncoder(dimension=DIMENSION), text_encoder
    )
    metrics, ranks = evaluate_multimodal_index(text_encoder, index)
    assert metrics == metrics_from_ranks(ranks)
    assert metrics.query_count == 2


def test_multimodal_report_is_deterministic(tmp_path: Path) -> None:
    metrics = metrics_from_ranks([1, 2])
    report = tmp_path / "report.md"
    metrics_path = tmp_path / "metrics.json"
    write_multimodal_reports(metrics, "placeholder", report, metrics_path)
    assert report.read_text(encoding="utf-8") == render_multimodal_report(
        metrics, "placeholder"
    )
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["mrr"] == 0.75


def test_multimodal_cli_smoke(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    index_path = tmp_path / "index.json"
    report = tmp_path / "report.md"
    metrics = tmp_path / "metrics.json"
    ingest_local_directory(FIXTURE, manifest)
    built = run_cli(
        "build-multimodal-baseline-index",
        "--manifest",
        str(manifest),
        "--output",
        str(index_path),
        "--dimension",
        str(DIMENSION),
    )
    assert built.returncode == 0, built.stderr
    assert load_multimodal_index(index_path).dimension == DIMENSION
    searched = run_cli(
        "search-multimodal-baseline",
        "--query",
        "red square",
        "--k",
        "2",
        "--index",
        str(index_path),
    )
    assert searched.returncode == 0, searched.stderr
    assert set(json.loads(searched.stdout)[0]) == {
        "item_id",
        "score",
        "caption",
        "image_path",
        "split",
    }
    evaluated = run_cli(
        "evaluate-multimodal-baseline",
        "--index",
        str(index_path),
        "--report-output",
        str(report),
        "--metrics-output",
        str(metrics),
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert report.is_file() and metrics.is_file()
