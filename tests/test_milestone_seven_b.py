from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import multimodal_retrieval_ops.faiss_flat as faiss_module
from multimodal_retrieval_ops.evaluation import RetrievalMetrics
from multimodal_retrieval_ops.faiss_flat import FaissDependencyError
from multimodal_retrieval_ops.faiss_hnsw import (
    ALLOWED_EF_SEARCH,
    HNSW_BACKEND_VERSION,
    HNSW_EF_CONSTRUCTION,
    HNSW_FORMAT_VERSION,
    HNSW_INDEX_TYPE,
    HNSW_M,
    HNSW_METRIC,
    ApproximationResult,
    DirectionBenchmark,
    HNSWIndexArtifact,
    HNSWIndexMetadata,
    LatencyResult,
    approximation_metrics,
    build_hnsw_index,
    hnsw_metadata_is_stale,
    load_hnsw_artifact,
    recommend_configuration,
    save_hnsw_artifact,
    search_hnsw_embedding,
    set_ef_search,
)
from multimodal_retrieval_ops.faiss_flat import require_faiss
from test_milestone_one import run_cli


def vectors() -> list[list[float]]:
    return [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]]


def metadata(ids: list[str]) -> HNSWIndexMetadata:
    return HNSWIndexMetadata(
        format_version=HNSW_FORMAT_VERSION,
        backend_name="faiss-cpu",
        backend_version=HNSW_BACKEND_VERSION,
        faiss_version="test",
        index_type=HNSW_INDEX_TYPE,
        metric=HNSW_METRIC,
        direction="text_to_image",
        embedding_dimension=2,
        m=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
        model_name="model",
        model_revision="revision",
        dataset_fingerprint="dataset",
        manifest_fingerprint="manifest",
        split="test",
        candidate_count=len(ids),
        candidate_ids=ids,
        preprocessing_version="normalized-clip-v1",
        source_cache_fingerprint="cache",
    )


def latency() -> LatencyResult:
    return LatencyResult(1, 5, 0.1, 0.1, 10.0)


def result(ef_search: int, *, difference: float = 0.0, overlap: float = 1.0) -> ApproximationResult:
    metrics = RetrievalMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2)
    differences = {
        "recall_at_1": difference,
        "recall_at_5": difference,
        "recall_at_10": difference,
        "mrr": difference,
        "median_rank": difference,
        "mean_rank": difference,
        "query_count": 0.0,
    }
    return ApproximationResult(
        ef_search, 1.0, overlap, overlap, 5.0 * overlap, 10.0 * overlap,
        metrics, differences, latency()
    )


def benchmark(configurations: list[ApproximationResult]) -> DirectionBenchmark:
    metrics = RetrievalMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2)
    return DirectionBenchmark(2, 3, metrics, latency(), configurations)


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_hnsw_creation_save_load_and_deterministic_metadata(tmp_path: Path) -> None:
    index = build_hnsw_index(vectors())
    artifact = HNSWIndexArtifact(metadata(["a", "b", "c"]), index)
    save_hnsw_artifact(artifact, tmp_path / "index.faiss", tmp_path / "index.json")
    loaded = load_hnsw_artifact(tmp_path / "index.faiss", tmp_path / "index.json")
    assert loaded.metadata == artifact.metadata
    assert loaded.index.ntotal == 3
    assert loaded.index.hnsw.efConstruction == 100


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_search_unique_candidates_and_applies_ef_search() -> None:
    artifact = HNSWIndexArtifact(metadata(["a", "b", "c"]), build_hnsw_index(vectors()))
    found = search_hnsw_embedding("q", {"q": [1.0, 0.0]}, artifact, 3, 32)
    assert len(found) == len({item["candidate_id"] for item in found}) == 3
    assert artifact.index.hnsw.efSearch == 32
    assert set(found[0]) == {"candidate_id", "score"}


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_only_bounded_ef_search_values_are_accepted() -> None:
    index = build_hnsw_index(vectors())
    for value in ALLOWED_EF_SEARCH:
        set_ef_search(index, value)
        assert index.hnsw.efSearch == value
    with pytest.raises(ValueError, match="efSearch must be one of"):
        set_ef_search(index, 128)
    with pytest.raises(ValueError, match="fixed at"):
        build_hnsw_index(vectors(), m=16)


def test_approximation_metrics() -> None:
    reference = np.asarray([[0, 1, 2], [2, 1, 0]])
    approximate = np.asarray([[0, 2, 1], [1, 2, 0]])
    values = approximation_metrics(reference, approximate)
    assert values["top1_agreement"] == 0.5
    assert values["top5_reference_set_recall"] == 1.0
    assert values["mean_overlap_at_5"] == 3


def test_recommendation_gate_prefers_lowest_passing_configuration() -> None:
    passing = [result(value) for value in ALLOWED_EF_SEARCH]
    recommendation, _ = recommend_configuration(benchmark(passing), benchmark(passing))
    assert recommendation == "HNSW efSearch=16"
    failing = [result(value, difference=0.006, overlap=0.97) for value in ALLOWED_EF_SEARCH]
    recommendation, _ = recommend_configuration(benchmark(failing), benchmark(failing))
    assert recommendation == "FlatIP"


def test_stale_detection_covers_required_metadata() -> None:
    original = metadata(["a", "b"])
    changes = (
        replace(original, source_cache_fingerprint="other"),
        replace(original, candidate_ids=["b", "a"]),
        replace(original, embedding_dimension=3),
        replace(original, model_name="other"),
        replace(original, model_revision="other"),
        replace(original, dataset_fingerprint="other"),
        replace(original, m=16),
        replace(original, ef_construction=200),
        replace(original, backend_version="2"),
    )
    assert all(hnsw_metadata_is_stale(original, changed) for changed in changes)


def test_missing_faiss_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(faiss_module, "faiss_available", lambda: False)
    with pytest.raises(FaissDependencyError, match=r"\[dev,faiss\]"):
        require_faiss()


@pytest.mark.parametrize(
    "command",
    [
        "build-faiss-hnsw-indexes",
        "evaluate-faiss-hnsw",
        "search-hnsw-text",
        "search-hnsw-image",
    ],
)
def test_hnsw_cli_help_smoke(command: str) -> None:
    completed = run_cli(command, "--help")
    assert completed.returncode == 0, completed.stderr
