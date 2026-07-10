import json
import math
from pathlib import Path

import pytest

from multimodal_retrieval_ops.baseline_index import (
    build_index,
    exact_search,
    load_index,
    write_index,
)
from multimodal_retrieval_ops.evaluation import evaluate_index, metrics_from_ranks
from multimodal_retrieval_ops.ingestion import ingest_local_directory
from multimodal_retrieval_ops.manifest import ManifestItem
from multimodal_retrieval_ops.retrieval_reporting import (
    render_retrieval_report,
    write_retrieval_reports,
)
from multimodal_retrieval_ops.text_baseline import build_vocabulary, encode_text
from test_milestone_one import run_cli

FIXTURE = Path("tests/fixtures/local_dataset")


def sample_items() -> list[ManifestItem]:
    return [
        ManifestItem("train-1", "one.jpg", "red car red", "train", "test"),
        ManifestItem("train-2", "two.jpg", "blue boat", "train", "test"),
        ManifestItem("validation-1", "three.jpg", "leakageword car", "validation", "test"),
        ManifestItem("test-1", "four.jpg", "secretword boat", "test", "test"),
    ]


def test_vocabulary_is_deterministic_and_train_only() -> None:
    items = sample_items()
    expected = ["blue", "boat", "car", "red"]
    assert build_vocabulary(items) == expected
    assert build_vocabulary(list(reversed(items))) == expected
    assert "leakageword" not in expected
    assert "secretword" not in expected


def test_vector_normalization() -> None:
    vector = encode_text("red car red", ["car", "red"])
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)
    assert encode_text("unknown", ["car", "red"]) == [0.0, 0.0]


def test_exact_search_ranking() -> None:
    vocabulary, entries = build_index(sample_items())
    results = exact_search("red car", vocabulary, entries, k=2)
    assert [result.item_id for result in results] == ["train-1", "validation-1"]
    assert results[0].score > results[1].score


def test_recall_and_mrr_calculations() -> None:
    metrics = metrics_from_ranks([1, 2, 10, 11])
    assert metrics.recall_at_1 == pytest.approx(0.25)
    assert metrics.recall_at_5 == pytest.approx(0.5)
    assert metrics.recall_at_10 == pytest.approx(0.75)
    assert metrics.mrr == pytest.approx((1 + 1 / 2 + 1 / 10 + 1 / 11) / 4)
    assert metrics.median_rank == 6.0
    assert metrics.mean_rank == 6.0


def test_evaluation_excludes_train_candidates() -> None:
    vocabulary, entries = build_index(sample_items())
    metrics, ranks = evaluate_index(vocabulary, entries)
    assert metrics.query_count == 2
    assert len(ranks) == 2


def test_deterministic_index_and_report_generation(tmp_path: Path) -> None:
    vocabulary, entries = build_index(sample_items())
    first_index = tmp_path / "first-index.json"
    first_vocab = tmp_path / "first-vocab.json"
    second_index = tmp_path / "second-index.json"
    second_vocab = tmp_path / "second-vocab.json"
    write_index(vocabulary, entries, first_index, first_vocab)
    write_index(vocabulary, entries, second_index, second_vocab)
    assert first_index.read_bytes() == second_index.read_bytes()
    assert first_vocab.read_bytes() == second_vocab.read_bytes()
    metrics, _ = evaluate_index(vocabulary, entries)
    markdown = tmp_path / "report.md"
    metrics_json = tmp_path / "metrics.json"
    write_retrieval_reports(metrics, markdown, metrics_json)
    assert markdown.read_text(encoding="utf-8") == render_retrieval_report(metrics)
    assert json.loads(metrics_json.read_text(encoding="utf-8"))["query_count"] == 2


def test_baseline_cli_smoke(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    index = tmp_path / "index.json"
    vocab = tmp_path / "vocab.json"
    report = tmp_path / "report.md"
    metrics = tmp_path / "metrics.json"
    ingest_local_directory(FIXTURE, manifest)
    built = run_cli(
        "build-text-baseline-index",
        "--manifest",
        str(manifest),
        "--index-output",
        str(index),
        "--vocab-output",
        str(vocab),
    )
    assert built.returncode == 0, built.stderr
    loaded_vocab, loaded_entries = load_index(index, vocab)
    assert loaded_vocab
    assert len(loaded_entries) == 5
    searched = run_cli(
        "search-text-baseline",
        "--query",
        "red square",
        "--k",
        "2",
        "--index",
        str(index),
        "--vocab",
        str(vocab),
    )
    assert searched.returncode == 0, searched.stderr
    assert len(json.loads(searched.stdout)) == 2
    evaluated = run_cli(
        "evaluate-text-baseline",
        "--index",
        str(index),
        "--vocab",
        str(vocab),
        "--report-output",
        str(report),
        "--metrics-output",
        str(metrics),
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert report.is_file() and metrics.is_file()
