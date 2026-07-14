from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import multimodal_retrieval_ops.faiss_flat as faiss_module
import multimodal_retrieval_ops.faiss_reranking as reranking_module
from multimodal_retrieval_ops.evaluation import RetrievalMetrics
from multimodal_retrieval_ops.faiss_flat import (
    FaissDependencyError,
    FaissIndexArtifact,
    FaissIndexMetadata,
    require_faiss,
)
from multimodal_retrieval_ops.faiss_hnsw import HNSWIndexArtifact, HNSWIndexMetadata
from multimodal_retrieval_ops.faiss_reranking import (
    CANDIDATE_K,
    EF_SEARCH,
    DirectionRerankingResult,
    DirectionTiming,
    FaissRerankingError,
    RankChanges,
    RankedAgreement,
    RerankingArtifactIncompatibleError,
    ShortlistCoverage,
    TimingResult,
    classify_rank_changes,
    evaluate_promotion_gate,
    exact_rescore_shortlist,
    ranked_agreement,
    relevant_ranks,
    shortlist_coverage,
    validate_fixed_configuration,
)
from test_milestone_one import run_cli


def normalized_vectors() -> dict[str, list[float]]:
    root = 2**-0.5
    return {
        "x": [1.0, 0.0],
        "y": [0.0, 1.0],
        "diagonal": [root, root],
        "tie": [root, root],
    }


def test_exact_shortlist_rescoring_maps_ids_and_scores() -> None:
    candidates = normalized_vectors()
    results = exact_rescore_shortlist(
        [1.0, 0.0], ["y", "diagonal", "x"], candidates
    )
    assert [item["candidate_id"] for item in results] == ["x", "diagonal", "y"]
    assert len({item["candidate_id"] for item in results}) == 3
    for item in results:
        expected = float(np.dot([1.0, 0.0], candidates[str(item["candidate_id"])]))
        assert item["score"] == pytest.approx(expected, abs=1e-7)


def test_exact_rescoring_uses_original_shortlist_order_for_ties() -> None:
    results = exact_rescore_shortlist(
        [1.0, 0.0], ["tie", "diagonal"], normalized_vectors()
    )
    assert [item["candidate_id"] for item in results] == ["tie", "diagonal"]
    assert [item["shortlist_rank"] for item in results] == [1, 2]


def test_exact_rescoring_rejects_duplicates_missing_and_nonfinite_vectors() -> None:
    candidates = normalized_vectors()
    with pytest.raises(FaissRerankingError, match="duplicate"):
        exact_rescore_shortlist([1.0, 0.0], ["x", "x"], candidates)
    with pytest.raises(FaissRerankingError, match="missing"):
        exact_rescore_shortlist([1.0, 0.0], ["missing"], candidates)
    with pytest.raises(FaissRerankingError, match="non-finite"):
        exact_rescore_shortlist([float("nan"), 0.0], ["x"], candidates)
    invalid = {**candidates, "bad": [float("inf"), 0.0]}
    with pytest.raises(FaissRerankingError, match="non-finite"):
        exact_rescore_shortlist([1.0, 0.0], ["bad"], invalid)


def test_real_configuration_is_strictly_fixed() -> None:
    validate_fixed_configuration(CANDIDATE_K, EF_SEARCH)
    with pytest.raises(FaissRerankingError, match="candidate_k is fixed at 50"):
        validate_fixed_configuration(49, EF_SEARCH)
    with pytest.raises(FaissRerankingError, match="efSearch is fixed at 64"):
        validate_fixed_configuration(CANDIDATE_K, 32)


def test_shortlist_coverage_and_ranked_agreement() -> None:
    reference = np.asarray(
        [list(range(10)), list(range(10, 20))], dtype="int64"
    )
    shortlist = np.asarray(
        [list(range(10)), [10, 11, 12, 13, 14, 15, 16, 17, 99, 98]], dtype="int64"
    )
    coverage = shortlist_coverage(reference, shortlist)
    assert coverage.flat_top1_present_rate == 1.0
    assert coverage.mean_flat_top5_fraction == 1.0
    assert coverage.mean_flat_top10_fraction == 0.9
    assert coverage.complete_flat_top10_query_count == 1
    assert coverage.missing_flat_top1_query_count == 0
    agreement = ranked_agreement(reference, shortlist)
    assert agreement.top1_agreement == 1.0
    assert agreement.top5_set_agreement == 1.0
    assert agreement.top10_set_agreement == 0.5
    assert agreement.mean_overlap_at_10 == 9


def test_bidirectional_relevance_semantics() -> None:
    text_order = np.asarray([[1, 0, 2], [2, 1, 0]])
    assert relevant_ranks(text_order, [{0}, {2}], 4) == [2, 1]
    image_order = np.asarray([[3, 2, 1, 0], [0, 1, 2, 3]])
    assert relevant_ranks(image_order, [{1, 3}, {2, 3}], 5) == [1, 3]


def test_rank_change_classification() -> None:
    changes = classify_rank_changes([5, 2, 1], [3, 2, 4])
    assert changes.improved_query_count == 1
    assert changes.unchanged_query_count == 1
    assert changes.worsened_query_count == 1
    assert changes.largest_improvement == 2
    assert changes.largest_regression == -3


def _direction(
    *, coverage: float = 1.0, raw_agreement: float = 0.8, reranked_agreement: float = 0.9,
    raw_mrr: float = 0.5, reranked_mrr: float = 0.5,
    raw_recall: float = 0.8, reranked_recall: float = 0.8,
) -> DirectionRerankingResult:
    def metrics(mrr: float, recall: float) -> RetrievalMetrics:
        return RetrievalMetrics(0.4, 0.7, recall, mrr, 2.0, 3.0, 2)

    timing = TimingResult(1, 3, 0.1, 20.0)
    agreement_raw = RankedAgreement(0.8, 0.7, raw_agreement, 4.5, 9.0)
    agreement_reranked = RankedAgreement(0.9, 0.8, reranked_agreement, 4.8, 9.5)
    return DirectionRerankingResult(
        2,
        50,
        ShortlistCoverage(1.0, 1.0, coverage, 2, 1.0, 0),
        metrics(0.51, 0.81),
        metrics(raw_mrr, raw_recall),
        metrics(reranked_mrr, reranked_recall),
        agreement_raw,
        agreement_reranked,
        RankChanges(1, 1, 0, 0.5, 0.5, 1, 0),
        0.0,
        DirectionTiming(timing, timing, timing, timing),
    )


def test_promotion_gate_pass_and_rejections() -> None:
    passing = _direction()
    assert evaluate_promotion_gate(passing, passing).approved
    assert not evaluate_promotion_gate(_direction(coverage=0.98), passing).approved
    assert not evaluate_promotion_gate(
        _direction(reranked_agreement=0.7), passing
    ).approved
    assert not evaluate_promotion_gate(
        _direction(reranked_mrr=0.498), passing
    ).approved
    assert not evaluate_promotion_gate(
        _direction(reranked_recall=0.798), passing
    ).approved
    assert not evaluate_promotion_gate(
        passing, passing, rejected_adapter_embeddings_used=True
    ).approved


def _flat_metadata(ids: list[str]) -> FaissIndexMetadata:
    return FaissIndexMetadata(
        1, "faiss-cpu", "1", "test", "IndexFlatIP", "text_to_image", 2,
        "model", "revision", "dataset", "manifest", "test", len(ids), ids,
        "normalized-clip-v1", "cache",
    )


def _hnsw_metadata(ids: list[str]) -> HNSWIndexMetadata:
    return HNSWIndexMetadata(
        1, "faiss-cpu", "1", "test", "IndexHNSWFlat", "inner_product",
        "text_to_image", 2, 32, 100, "model", "revision", "dataset", "manifest",
        "test", len(ids), ids, "normalized-clip-v1", "cache",
    )


def test_stale_artifact_rejection_covers_candidate_order_and_model() -> None:
    flat = FaissIndexArtifact(_flat_metadata(["a", "b"]), object())
    hnsw = HNSWIndexArtifact(_hnsw_metadata(["a", "b"]), object())
    reranking_module._validate_artifact_pair(flat, hnsw)
    stale_order = replace(hnsw, metadata=replace(hnsw.metadata, candidate_ids=["b", "a"]))
    with pytest.raises(RerankingArtifactIncompatibleError, match="candidate_ids"):
        reranking_module._validate_artifact_pair(flat, stale_order)
    stale_model = replace(hnsw, metadata=replace(hnsw.metadata, model_name="other"))
    with pytest.raises(RerankingArtifactIncompatibleError, match="model_name"):
        reranking_module._validate_artifact_pair(flat, stale_model)


def test_rejected_adapter_module_is_not_part_of_reranking_runtime() -> None:
    assert "contrastive_adapters" not in reranking_module.__dict__
    assert "contrastive_adapter" not in Path(reranking_module.__file__).read_text(encoding="utf-8")


def test_missing_faiss_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(faiss_module, "faiss_available", lambda: False)
    with pytest.raises(FaissDependencyError, match=r"\[dev,faiss\]"):
        require_faiss()


@pytest.mark.parametrize(
    "command",
    [
        "evaluate-hnsw-reranking",
        "hnsw-reranking-info",
        "search-reranked-text",
        "search-reranked-image",
    ],
)
def test_reranking_cli_help_smoke(command: str) -> None:
    completed = run_cli(command, "--help")
    assert completed.returncode == 0, completed.stderr
