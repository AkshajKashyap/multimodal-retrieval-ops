from dataclasses import replace
from pathlib import Path

import pytest

import multimodal_retrieval_ops.faiss_flat as faiss_module
from multimodal_retrieval_ops.faiss_flat import (
    FAISS_BACKEND_NAME,
    FAISS_BACKEND_VERSION,
    FAISS_FORMAT_VERSION,
    FAISS_INDEX_TYPE,
    PREPROCESSING_VERSION,
    FaissDependencyError,
    FaissIndexArtifact,
    FaissIndexMetadata,
    _compare_direction,
    build_flat_ip_index,
    index_metadata_is_stale,
    load_faiss_artifact,
    ordered_embeddings,
    require_faiss,
    save_faiss_artifact,
)
from test_milestone_one import run_cli


def normalized_vectors() -> list[list[float]]:
    return [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]]


def metadata(candidate_ids: list[str], direction: str = "text_to_image") -> FaissIndexMetadata:
    return FaissIndexMetadata(
        format_version=FAISS_FORMAT_VERSION,
        backend_name=FAISS_BACKEND_NAME,
        backend_version=FAISS_BACKEND_VERSION,
        faiss_version="test",
        index_type=FAISS_INDEX_TYPE,
        direction=direction,
        embedding_dimension=2,
        model_name="model",
        model_revision="revision",
        dataset_fingerprint="dataset",
        manifest_fingerprint="manifest",
        split="test",
        candidate_count=len(candidate_ids),
        candidate_ids=candidate_ids,
        preprocessing_version=PREPROCESSING_VERSION,
        source_cache_fingerprint="cache",
    )


def test_actionable_error_when_faiss_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(faiss_module, "faiss_available", lambda: False)
    with pytest.raises(FaissDependencyError, match=r"\[dev,faiss\]"):
        require_faiss()


def test_backend_info_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "faiss-info.txt"
    result = run_cli("faiss-backend-info", "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert "FAISS CPU backend" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "command",
    [
        "build-faiss-flat-indexes",
        "evaluate-faiss-flat",
        "search-faiss-text",
        "search-faiss-image",
    ],
)
def test_faiss_cli_command_help_smoke(command: str) -> None:
    result = run_cli(command, "--help")
    assert result.returncode == 0, result.stderr


def test_deterministic_candidate_id_ordering() -> None:
    ids, vectors = ordered_embeddings({"z": [0.0, 1.0], "a": [1.0, 0.0]})
    assert ids == ["a", "z"]
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_flat_ip_construction_save_and_load(tmp_path: Path) -> None:
    vectors = normalized_vectors()
    index = build_flat_ip_index(vectors)
    artifact = FaissIndexArtifact(metadata(["a", "b", "c"]), index)
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "index.json"
    save_faiss_artifact(artifact, index_path, metadata_path)
    loaded = load_faiss_artifact(index_path, metadata_path)
    assert loaded.index.ntotal == 3
    assert loaded.index.d == 2
    assert loaded.metadata == artifact.metadata


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_exact_cosine_and_faiss_topk_equivalence() -> None:
    vectors = normalized_vectors()
    index = build_flat_ip_index(vectors)
    comparison = _compare_direction(
        [[1.0, 0.0], [0.0, 1.0]],
        vectors,
        [{0}, {1}],
        index,
    )
    assert comparison.reference_metrics == comparison.faiss_metrics
    assert comparison.top1_agreement_rate == 1.0
    assert comparison.top5_set_agreement_rate == 1.0
    assert comparison.correctness_gate_passed


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_text_to_image_metric_equivalence() -> None:
    candidates = normalized_vectors()
    queries = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    comparison = _compare_direction(queries, candidates, [{0}, {0}, {1}], build_flat_ip_index(candidates))
    assert comparison.reference_metrics == comparison.faiss_metrics
    assert comparison.reference_metrics.recall_at_1 == 1.0


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_image_to_text_multiple_relevant_metric_equivalence() -> None:
    captions = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]
    images = [[1.0, 0.0], [0.0, 1.0]]
    comparison = _compare_direction(
        images,
        captions,
        [{0, 1}, {2, 3}],
        build_flat_ip_index(captions),
    )
    assert comparison.reference_metrics == comparison.faiss_metrics
    assert comparison.reference_metrics.recall_at_1 == 1.0


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_tied_score_disagreement_is_identified() -> None:
    candidates = [[1.0, 0.0], [1.0, 0.0]]
    comparison = _compare_direction(
        [[1.0, 0.0]], candidates, [{0, 1}], build_flat_ip_index(candidates)
    )
    disagreements = round((1.0 - comparison.top1_agreement_rate) * comparison.query_count)
    assert comparison.top1_tie_explained_disagreements == disagreements


def test_stale_index_detection() -> None:
    original = metadata(["a", "b"])
    for changed in (
        replace(original, source_cache_fingerprint="other"),
        replace(original, model_name="other"),
        replace(original, model_revision="other"),
        replace(original, embedding_dimension=3),
        replace(original, candidate_ids=["b", "a"]),
        replace(original, dataset_fingerprint="other"),
        replace(original, manifest_fingerprint="other"),
        replace(original, split="validation"),
        replace(original, backend_version="2"),
    ):
        assert index_metadata_is_stale(original, changed)


@pytest.mark.skipif(not faiss_module.faiss_available(), reason="optional faiss extra unavailable")
def test_incompatible_embedding_dimension_rejected() -> None:
    with pytest.raises(ValueError, match="does not match expected"):
        build_flat_ip_index(normalized_vectors(), expected_dimension=3)
