from dataclasses import asdict, replace
import inspect
import importlib.util
import json
import math
from pathlib import Path

import pytest

from multimodal_retrieval_ops.cli import main
from multimodal_retrieval_ops.contrastive_adapters import (
    ADAPTER_ARCHITECTURE,
    create_adapter_pair,
    file_digest,
    write_adapter_cache,
)
from multimodal_retrieval_ops.contrastive_adapter_diagnostics import (
    DiagnosticArtifactIncompatibleError,
    ImageQueryDiagnostic,
    TextQueryDiagnostic,
    TrainingHistoryInterpretation,
    ValidationCaption,
    calculate_query_diagnostics,
    caption_length_group,
    caption_slice_metrics,
    deterministic_examples,
    diagnose_failure,
    movement_statistics,
    read_validation_captions,
    run_adapter_failure_analysis,
    summarize_margins,
    summarize_rank_changes,
    validate_diagnostic_artifacts,
    verify_recorded_metrics,
)
from multimodal_retrieval_ops.evaluation import RetrievalMetrics
from multimodal_retrieval_ops.hf_clip_benchmark import BidirectionalResult, DirectionResult
from test_milestone_nine_a import cache
from test_milestone_one import run_cli


def direction(metrics: RetrievalMetrics, candidate_count: int) -> DirectionResult:
    return DirectionResult(metrics, candidate_count)


def result(mrr: float = 0.75) -> BidirectionalResult:
    return BidirectionalResult(
        direction(RetrievalMetrics(0.5, 1.0, 1.0, mrr, 1.5, 1.5, 2), 2),
        direction(RetrievalMetrics(0.5, 1.0, 1.0, mrr, 1.5, 1.5, 2), 2),
    )


def recorded_metrics(zero: BidirectionalResult, adapted: BidirectionalResult) -> dict:
    def document(value: DirectionResult) -> dict:
        return asdict(value.metrics) | {"candidate_count": value.candidate_count}

    return {
        "run_state": "success",
        "zero_shot": {
            "text_to_image": document(zero.text_to_image),
            "image_to_text": document(zero.image_to_text),
        },
        "adapted": {
            "text_to_image": document(adapted.text_to_image),
            "image_to_text": document(adapted.image_to_text),
        },
    }


def text_record(
    identity: str,
    zero_rank: int,
    adapted_rank: int,
    token_count: int = 8,
    margin_change: float = 0.0,
) -> TextQueryDiagnostic:
    return TextQueryDiagnostic(
        caption_id=identity,
        target_image_id="image-a",
        zero_shot_rank=zero_rank,
        adapted_rank=adapted_rank,
        rank_change=adapted_rank - zero_rank,
        zero_shot_reciprocal_rank=1 / zero_rank,
        adapted_reciprocal_rank=1 / adapted_rank,
        caption_text=" ".join(["word"] * token_count),
        caption_token_count=token_count,
        zero_shot_relevant_similarity=0.8,
        zero_shot_highest_irrelevant_similarity=0.7,
        zero_shot_margin=0.1,
        adapted_relevant_similarity=0.8 + margin_change,
        adapted_highest_irrelevant_similarity=0.7,
        adapted_margin=0.1 + margin_change,
        margin_change=margin_change,
    )


def image_record(
    identity: str, zero_rank: int, adapted_rank: int, margin_change: float = 0.0
) -> ImageQueryDiagnostic:
    return ImageQueryDiagnostic(
        image_id=identity,
        zero_shot_best_relevant_rank=zero_rank,
        adapted_best_relevant_rank=adapted_rank,
        rank_change=adapted_rank - zero_rank,
        zero_shot_reciprocal_rank=1 / zero_rank,
        adapted_reciprocal_rank=1 / adapted_rank,
        relevant_caption_ids=[f"{identity}-caption"],
        representative_caption_text="safe caption",
        zero_shot_relevant_similarity=0.8,
        zero_shot_highest_irrelevant_similarity=0.7,
        zero_shot_margin=0.1,
        adapted_relevant_similarity=0.8 + margin_change,
        adapted_highest_irrelevant_similarity=0.7,
        adapted_margin=0.1 + margin_change,
        margin_change=margin_change,
    )


def training_interpretation(overfit: bool) -> TrainingHistoryInterpretation:
    return TrainingHistoryInterpretation(
        selected_epoch=2,
        stopping_epoch=5,
        best_validation_mean_bidirectional_mrr=0.8,
        final_validation_mean_bidirectional_mrr=0.7,
        initial_training_loss=1.0,
        selected_epoch_training_loss=0.8,
        final_training_loss=0.4,
        training_loss_trend="decreased throughout the recorded run",
        validation_before_selected_epoch="validation improved",
        validation_after_selected_epoch="all later values were lower",
        supported_behavior=(
            "evidence consistent with overfitting after the selected epoch"
            if overfit
            else "mixed or inconclusive training-history evidence"
        ),
    )


@pytest.mark.skipif(not importlib.util.find_spec("torch"), reason="Torch missing")
def test_artifact_compatibility_validation(tmp_path: Path) -> None:
    train_cache = cache("train")
    validation_cache = cache("validation")
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    write_adapter_cache(train_cache, train_path)
    write_adapter_cache(validation_cache, validation_path)
    adapters = create_adapter_pair(4, bottleneck_dimension=2, seed=42)
    parameter_count = sum(parameter.numel() for parameter in adapters.parameters())
    source = {
        "backend_name": validation_cache.metadata.backend_name,
        "backend_version": validation_cache.metadata.backend_version,
        "model_name": validation_cache.metadata.model_name,
        "model_revision": validation_cache.metadata.model_revision,
        "preprocessing_identity": validation_cache.metadata.preprocessing_identity,
    }
    checkpoint = {
        "architecture": ADAPTER_ARCHITECTURE,
        "input_dimension": 4,
        "validation_subset_fingerprint": validation_cache.metadata.subset_fingerprint,
        "train_cache_fingerprint": file_digest(train_path),
        "validation_cache_fingerprint": file_digest(validation_path),
        "clip_frozen": True,
        "source_model": source,
        "parameter_count": parameter_count,
    }
    manifest = {
        caption_id: ValidationCaption(caption_id, image_id, "safe text")
        for caption_id, image_id in validation_cache.caption_image_ids.items()
    }
    manifest["unselected-caption"] = ValidationCaption(
        "unselected-caption", "unselected-image", "outside the selected subset"
    )
    recorded = {
        "data_boundaries": {
            "official_test_accessed": False,
            "validation_subset_fingerprint": validation_cache.metadata.subset_fingerprint,
        },
        "promotion": {"promote": False},
        "source_model": source,
        "architecture": {
            "name": ADAPTER_ARCHITECTURE,
            "input_dimension": 4,
            "parameter_count": parameter_count,
        },
    }
    compatibility = validate_diagnostic_artifacts(
        train_cache,
        validation_cache,
        checkpoint,
        recorded,
        manifest,
        adapters,
        train_path,
        validation_path,
    )
    assert compatibility.compatible is True
    assert compatibility.validation_caption_count == 4
    with pytest.raises(DiagnosticArtifactIncompatibleError, match="architecture"):
        validate_diagnostic_artifacts(
            train_cache,
            validation_cache,
            checkpoint | {"architecture": "wrong"},
            recorded,
            manifest,
            adapters,
            train_path,
            validation_path,
        )
    forbidden_test_cache = replace(
        validation_cache,
        metadata=replace(validation_cache.metadata, split="test"),
    )
    with pytest.raises(DiagnosticArtifactIncompatibleError, match="train and validation"):
        validate_diagnostic_artifacts(
            train_cache,
            forbidden_test_cache,
            checkpoint,
            recorded,
            manifest,
            adapters,
            train_path,
            validation_path,
        )


def test_metric_reproduction_tolerance_and_mismatch() -> None:
    zero = result(0.75)
    adapted = result(0.7500005)
    recorded = recorded_metrics(zero, adapted)
    verify_recorded_metrics(zero, adapted, recorded)
    changed = json.loads(json.dumps(recorded))
    changed["adapted"]["text_to_image"]["mrr"] = 0.70
    with pytest.raises(DiagnosticArtifactIncompatibleError, match="does not match"):
        verify_recorded_metrics(zero, adapted, changed)


def test_rank_deltas_classifications_and_positive_margins() -> None:
    image_ids = ["image-a", "image-b"]
    caption_ids = ["caption-a", "caption-b", "caption-c"]
    relationships = {
        "caption-a": "image-a",
        "caption-b": "image-b",
        "caption-c": "image-a",
    }
    zero = [[0.9, 0.8], [0.8, 0.9], [0.7, 0.8]]
    adapted = [[0.7, 0.8], [0.8, 0.9], [0.9, 0.8]]
    texts, images = calculate_query_diagnostics(
        zero,
        adapted,
        image_ids,
        caption_ids,
        relationships,
        {caption_id: "short safe caption" for caption_id in caption_ids},
    )
    summary = summarize_rank_changes(texts)
    assert (summary.improved_count, summary.unchanged_count, summary.worsened_count) == (1, 1, 1)
    assert texts[0].zero_shot_margin > 0
    assert texts[0].adapted_margin < 0
    margins = summarize_margins(texts)
    assert math.isclose(margins.margin_improved_percentage, 1 / 3)
    assert len(images) == 2


def test_representation_movement_statistics() -> None:
    value = 1 / math.sqrt(2)
    stats = movement_statistics(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [value, value]],
    )
    assert stats.count == 2
    assert math.isclose(stats.maximum, 1.0)
    assert math.isclose(stats.minimum, value)
    assert stats.percentile_5 <= stats.median <= stats.percentile_95


@pytest.mark.parametrize(
    ("count", "group"),
    [(1, "short"), (7, "short"), (8, "medium"), (12, "medium"), (13, "long")],
)
def test_caption_length_group_boundaries(count: int, group: str) -> None:
    assert caption_length_group(count) == group


def test_caption_length_slice_metrics() -> None:
    records = [
        text_record("short", 1, 2, token_count=7),
        text_record("medium", 2, 1, token_count=8),
        text_record("long", 6, 5, token_count=13),
    ]
    slices = caption_slice_metrics(records)
    assert [item.group for item in slices] == ["short", "medium", "long"]
    assert slices[0].zero_shot.recall_at_1 == 1.0
    assert slices[0].adapted.recall_at_1 == 0.0
    assert slices[2].adapted.recall_at_5 == 1.0


def test_deterministic_example_ordering() -> None:
    text = [
        text_record("b", 1, 4, margin_change=-0.1),
        text_record("a", 1, 4, margin_change=-0.1),
        text_record("c", 4, 1, margin_change=0.2),
    ]
    image = [
        image_record("image-b", 1, 3, -0.2),
        image_record("image-a", 3, 1, 0.2),
    ]
    examples = deterministic_examples(text, image)
    assert [item["caption_id"] for item in examples["text_to_image_regressions"]] == [
        "a",
        "b",
    ]
    assert examples["text_to_image_improvements"][0]["caption_id"] == "c"
    assert examples["image_to_text_regressions"][0]["image_id"] == "image-b"


def test_supported_diagnosis_rules() -> None:
    overfit = diagnose_failure(
        mean_mrr_difference=-0.01,
        text_mrr_difference=-0.02,
        image_mrr_difference=0.0,
        image_movement_mean=0.99,
        text_movement_mean=0.98,
        training=training_interpretation(True),
    )
    labels = {item.conclusion for item in overfit}
    assert "likely overfitting" in labels
    assert "likely optimization imbalance between modalities" in labels
    aggressive = diagnose_failure(
        mean_mrr_difference=0.006,
        text_mrr_difference=0.006,
        image_mrr_difference=0.006,
        image_movement_mean=0.94,
        text_movement_mean=0.93,
        training=training_interpretation(False),
    )
    assert "representation movement too aggressive" in {
        item.conclusion for item in aggressive
    }
    inconclusive = diagnose_failure(
        mean_mrr_difference=0.006,
        text_mrr_difference=0.006,
        image_mrr_difference=0.006,
        image_movement_mean=0.99,
        text_movement_mean=0.99,
        training=training_interpretation(False),
    )
    assert [item.conclusion for item in inconclusive] == ["mixed or inconclusive evidence"]


def test_official_test_rows_are_not_materialized(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_id,caption_id,image_path,caption,split,source\n"
        "validation-a,caption-a,a.jpg,safe validation,validation,test\n"
        "test-secret,test-caption,secret.jpg,do not inspect,test,test\n",
        encoding="utf-8",
    )
    captions = read_validation_captions(manifest)
    assert set(captions) == {"caption-a"}
    assert "test-caption" not in captions


def test_diagnostic_entrypoint_contains_no_training_or_clip_execution() -> None:
    source = inspect.getsource(run_adapter_failure_analysis)
    assert "train_adapters" not in source
    assert "ClipEmbeddingBackend" not in source
    assert "encode_image" not in source
    assert "encode_text" not in source


def test_cli_help_and_info_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    completed = run_cli("analyze-contrastive-adapter", "--help")
    assert completed.returncode == 0
    assert "--recorded-metrics" in completed.stdout
    assert main(["contrastive-adapter-diagnostics-info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["analysis_boundary"] == "validation-only"
    assert info["clip_inference"] is False
    assert info["retraining"] is False
