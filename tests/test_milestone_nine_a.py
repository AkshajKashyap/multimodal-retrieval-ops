from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

import multimodal_retrieval_ops.contrastive_adapters as adapter_module
from multimodal_retrieval_ops.cli import main
from multimodal_retrieval_ops.contrastive_adapters import (
    AdapterCacheMetadata,
    AdapterDependencyError,
    AdapterEmbeddingCache,
    EpochRecord,
    adapter_cache_is_stale,
    create_adapter_pair,
    deterministic_image_batches,
    load_adapter_checkpoint,
    make_adapter_cache_metadata,
    promotion_decision,
    prepare_adapter_cache,
    relationship_positive_mask,
    save_adapter_checkpoint,
    select_best_epoch,
    select_grouped_subset,
    symmetric_multi_positive_loss,
    AdapterTrainingConfig,
    evaluate_adapters,
    train_adapters,
    validate_adapter_cache,
    write_adapter_cache,
)
from multimodal_retrieval_ops.evaluation import RetrievalMetrics
from multimodal_retrieval_ops.hf_clip_benchmark import (
    BidirectionalResult,
    DirectionResult,
)
from multimodal_retrieval_ops.hf_flickr8k import HFFlickr8kProvenance
from multimodal_retrieval_ops.manifest import ManifestItemV2
from test_milestone_one import run_cli


def rows() -> list[ManifestItemV2]:
    result = []
    for split, image_count in (("train", 4), ("validation", 3), ("test", 2)):
        for image_index in range(image_count):
            image_id = f"{split}-{image_index}"
            for caption_index in range(2):
                result.append(
                    ManifestItemV2(
                        image_id=image_id,
                        caption_id=f"{image_id}-caption-{caption_index}",
                        image_path=f"ignored/{image_id}.jpg",
                        caption=f"{split} caption {image_index} {caption_index}",
                        split=split,
                        source="synthetic",
                    )
                )
    return result


def provenance() -> HFFlickr8kProvenance:
    return HFFlickr8kProvenance(
        dataset_name="synthetic",
        requested_revision="default",
        resolved_fingerprint="dataset-fingerprint",
        source_split_image_counts={"train": 4, "validation": 3, "test": 2},
        source_split_caption_counts={"train": 8, "validation": 6, "test": 4},
        materialized_split_image_counts={"train": 4, "validation": 3, "test": 2},
        materialized_split_caption_counts={"train": 8, "validation": 6, "test": 4},
        unique_image_count=9,
        caption_count=18,
        captions_per_image_min=2,
        captions_per_image_max=2,
        captions_per_image_mean=2.0,
        image_format_counts={"jpeg": 9},
        missing_or_invalid_image_count=0,
        licensing_status="synthetic",
        materialization_path="ignored",
        resolved_revision="revision-1",
    )


class FakeBackend:
    backend_name = "huggingface-clip"
    backend_version = "1"
    model_name = "synthetic-clip"
    model_revision = "revision-1"
    dimension = 4

    def __init__(self) -> None:
        self.ensure_calls = 0
        self.image_calls = 0
        self.text_calls = 0

    def ensure_loaded(self) -> None:
        self.ensure_calls += 1

    def encode_images(self, image_paths: list[str]) -> list[list[float]]:
        self.image_calls += 1
        return [
            [1.0, 0.0, 0.0, 0.0] if index % 2 == 0 else [0.0, 1.0, 0.0, 0.0]
            for index, _ in enumerate(image_paths)
        ]

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_calls += 1
        return [
            [1.0, 0.0, 0.0, 0.0] if index < 2 else [0.0, 1.0, 0.0, 0.0]
            for index, _ in enumerate(texts)
        ]


def metadata(split: str = "train") -> AdapterCacheMetadata:
    selected = select_grouped_subset(rows(), split, 2, 42)
    return make_adapter_cache_metadata(
        selected,
        FakeBackend(),
        provenance(),
        split=split,
        requested_image_count=2,
        seed=42,
        dimension=4,
    )


def cache(split: str = "train") -> AdapterEmbeddingCache:
    cache_metadata = metadata(split)
    image_embeddings = {
        image_id: ([1.0, 0.0, 0.0, 0.0] if index == 0 else [0.0, 1.0, 0.0, 0.0])
        for index, image_id in enumerate(cache_metadata.selected_image_ids)
    }
    selected_rows = select_grouped_subset(rows(), split, 2, 42)
    caption_embeddings = {}
    caption_image_ids = {}
    for row in selected_rows:
        caption_embeddings[row.caption_id] = list(image_embeddings[row.image_id])
        caption_image_ids[row.caption_id] = row.image_id
    return AdapterEmbeddingCache(
        cache_metadata, image_embeddings, caption_embeddings, caption_image_ids
    )


def result(
    *,
    t2i_mrr: float,
    i2t_mrr: float,
    t2i_r10: float = 0.9,
    i2t_r10: float = 0.9,
) -> BidirectionalResult:
    return BidirectionalResult(
        text_to_image=DirectionResult(
            RetrievalMetrics(0.5, 0.8, t2i_r10, t2i_mrr, 1.0, 2.0, 4), 2
        ),
        image_to_text=DirectionResult(
            RetrievalMetrics(0.5, 0.8, i2t_r10, i2t_mrr, 1.0, 2.0, 2), 4
        ),
    )


def test_grouped_subset_selection_is_seeded_and_preserves_all_captions() -> None:
    first = select_grouped_subset(rows(), "train", 3, 42)
    second = select_grouped_subset(rows(), "train", 3, 42)
    assert first == second
    assert {row.split for row in first} == {"train"}
    counts = {image_id: 0 for image_id in {row.image_id for row in first}}
    for row in first:
        counts[row.image_id] += 1
    assert counts == {image_id: 2 for image_id in counts}


@pytest.mark.parametrize(
    ("split", "count", "message"),
    [
        ("train", 501, "between 1 and 500"),
        ("validation", 101, "between 1 and 100"),
        ("test", 1, "only official train and validation"),
    ],
)
def test_subset_limits_and_test_split_rejection(
    split: str, count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_grouped_subset(rows(), split, count, 42)


def test_cache_metadata_identity_and_staleness(tmp_path: Path) -> None:
    frozen_cache = cache()
    validate_adapter_cache(frozen_cache)
    path = tmp_path / "cache.json"
    write_adapter_cache(frozen_cache, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["selected_image_ids"] == sorted(
        frozen_cache.metadata.selected_image_ids
    )
    assert payload["caption_image_ids"]
    assert adapter_cache_is_stale(frozen_cache, frozen_cache.metadata) is False
    changed = replace(frozen_cache.metadata, model_revision="different")
    assert adapter_cache_is_stale(frozen_cache, changed) is True


def test_compatible_cache_is_reused_without_encoder_invocation(tmp_path: Path) -> None:
    backend = FakeBackend()
    path = tmp_path / "train.json"
    created, first_hit = prepare_adapter_cache(
        rows(),
        backend,
        provenance(),
        split="train",
        image_count=2,
        seed=42,
        cache_path=path,
    )
    reused, second_hit = prepare_adapter_cache(
        rows(),
        backend,
        provenance(),
        split="train",
        image_count=2,
        seed=42,
        cache_path=path,
    )
    assert first_hit is False
    assert second_hit is True
    assert reused == created
    assert backend.ensure_calls == backend.image_calls == backend.text_calls == 1


@pytest.mark.skipif(not adapter_module.training_dependencies_available(), reason="Torch missing")
def test_adapter_dimension_normalization_and_identity_residual() -> None:
    torch = adapter_module.require_torch()
    adapters = create_adapter_pair(4, bottleneck_dimension=2, seed=42)
    inputs = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float32)
    output = adapters["image"](inputs)
    assert output.shape == (1, 4)
    assert math.isclose(float(output.norm(dim=-1).item()), 1.0, abs_tol=1e-6)
    expected = inputs / inputs.norm(dim=-1, keepdim=True)
    assert torch.allclose(output, expected)
    assert adapters["image"] is not adapters["text"]


@pytest.mark.skipif(not adapter_module.training_dependencies_available(), reason="Torch missing")
def test_positive_mask_and_multi_positive_loss() -> None:
    torch = adapter_module.require_torch()
    mask = relationship_positive_mask(
        ["image-a", "image-b"],
        ["image-a", "image-a", "image-b", "image-b"],
    )
    assert mask.tolist() == [[True, True, False, False], [False, False, True, True]]
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    captions = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    captions = torch.nn.functional.normalize(captions, dim=-1)
    loss = symmetric_multi_positive_loss(images, captions, mask, 0.07)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_deterministic_image_batches() -> None:
    image_ids = [f"image-{index}" for index in range(9)]
    first = deterministic_image_batches(image_ids, 4, 42, 1)
    second = deterministic_image_batches(reversed(image_ids), 4, 42, 1)
    assert first == second
    assert sorted(item for batch in first for item in batch) == sorted(image_ids)
    assert max(map(len, first)) == 4


@pytest.mark.skipif(not adapter_module.training_dependencies_available(), reason="Torch missing")
def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    adapters = create_adapter_pair(4, bottleneck_dimension=2, seed=42)
    checkpoint = tmp_path / "adapter.pt"
    sidecar = tmp_path / "adapter.json"
    metadata_document = {
        "input_dimension": 4,
        "bottleneck_dimension": 2,
        "training_config": {"seed": 42},
    }
    save_adapter_checkpoint(adapters, metadata_document, checkpoint, sidecar)
    loaded, loaded_metadata = load_adapter_checkpoint(checkpoint, sidecar)
    assert loaded_metadata == metadata_document
    assert set(loaded.state_dict()) == set(adapters.state_dict())
    for name, tensor in adapters.state_dict().items():
        assert adapter_module.require_torch().equal(tensor, loaded.state_dict()[name])


def test_best_checkpoint_selection_uses_validation_metric_only() -> None:
    history = [
        EpochRecord(1, training_loss=0.1, validation_mean_bidirectional_mrr=0.4),
        EpochRecord(2, training_loss=0.5, validation_mean_bidirectional_mrr=0.6),
        EpochRecord(3, training_loss=0.01, validation_mean_bidirectional_mrr=0.5),
    ]
    assert select_best_epoch(history) == 2


@pytest.mark.skipif(not adapter_module.training_dependencies_available(), reason="Torch missing")
def test_tiny_cached_training_and_validation_evaluation(tmp_path: Path) -> None:
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    checkpoint_path = tmp_path / "adapter.pt"
    metadata_path = tmp_path / "adapter.json"
    write_adapter_cache(cache("train"), train_path)
    write_adapter_cache(cache("validation"), validation_path)
    training = train_adapters(
        train_path,
        validation_path,
        checkpoint_path,
        metadata_path,
        AdapterTrainingConfig(max_epochs=2, early_stopping_patience=1, batch_size=2),
    )
    comparison, checkpoint_metadata = evaluate_adapters(
        train_path, validation_path, checkpoint_path, metadata_path
    )
    assert 1 <= training.selected_epoch <= 2
    assert checkpoint_metadata["clip_frozen"] is True
    assert comparison.zero_shot.text_to_image.metrics.query_count == 4
    assert comparison.adapted.image_to_text.metrics.query_count == 2


def test_promotion_gate_passes_only_when_every_condition_passes() -> None:
    zero_shot = result(t2i_mrr=0.5, i2t_mrr=0.5)
    adapted = result(t2i_mrr=0.51, i2t_mrr=0.51)
    decision = promotion_decision(zero_shot, adapted, evaluation_split="validation")
    assert decision.promote is True
    assert all(reason.startswith("PASS") for reason in decision.reasons)


def test_promotion_gate_rejects_directional_regression_and_nonvalidation() -> None:
    zero_shot = result(t2i_mrr=0.5, i2t_mrr=0.5)
    adapted = result(t2i_mrr=0.51, i2t_mrr=0.51, t2i_r10=0.89)
    regression = promotion_decision(zero_shot, adapted, evaluation_split="validation")
    wrong_split = promotion_decision(zero_shot, adapted, evaluation_split="test")
    assert regression.promote is False
    assert any(reason.startswith("FAIL") for reason in regression.reasons)
    assert wrong_split.promote is False


def test_missing_torch_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    original = adapter_module.importlib.util.find_spec
    monkeypatch.setattr(
        adapter_module.importlib.util,
        "find_spec",
        lambda name: None if name == "torch" else original(name),
    )
    with pytest.raises(AdapterDependencyError, match=r"\[dev,train\]"):
        adapter_module.require_torch()


def test_cli_help_and_info_are_download_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = run_cli("contrastive-adapter-info", "--help")
    assert completed.returncode == 0
    assert "--local-files-only" in completed.stdout
    assert main(["contrastive-adapter-info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["official_test_accessed"] is False
    assert info["local_files_only"] is True
