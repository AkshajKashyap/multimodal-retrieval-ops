from dataclasses import replace
from pathlib import Path

import pytest

import multimodal_retrieval_ops.hf_flickr8k as hf_module
from multimodal_retrieval_ops.clip_backend import ClipEmbeddingBackend
from multimodal_retrieval_ops.hf_clip_benchmark import (
    HFBenchmarkCache,
    bidirectional_ranks_reference,
    hf_benchmark_cache_is_stale,
    make_hf_cache_metadata,
    select_official_split,
)
from multimodal_retrieval_ops.hf_flickr8k import (
    HFFlickr8kError,
    HFFlickr8kProvenance,
    convert_hf_split_rows,
    load_hf_flickr8k,
    materialize_hf_image,
    stable_hf_image_id,
)


def fake_materializer(_image: object, path: Path, _force: bool) -> tuple[str, bool]:
    return path.suffix.lstrip("."), True


def mock_rows() -> list[dict[str, object]]:
    return [
        {
            "image": object(),
            "image_id": "source-image-1",
            "caption": [f"Caption {index}" for index in range(1, 6)],
        },
        {
            "image": object(),
            "image_id": "source-image-2",
            "caption": [f"Other caption {index}" for index in range(1, 6)],
        },
    ]


def test_hf_row_conversion_expands_five_unique_captions(tmp_path: Path) -> None:
    converted, formats, invalid = convert_hf_split_rows(
        mock_rows()[:1], "test", tmp_path, materializer=fake_materializer
    )
    assert len(converted) == 5
    assert len({row.caption_id for row in converted}) == 5
    assert {row.image_id for row in converted} == {converted[0].image_id}
    assert {row.split for row in converted} == {"test"}
    assert formats == {"jpg": 1}
    assert invalid == 0


def test_hf_ids_and_row_order_are_deterministic(tmp_path: Path) -> None:
    first, _, _ = convert_hf_split_rows(
        mock_rows(), "validation", tmp_path, materializer=fake_materializer
    )
    second, _, _ = convert_hf_split_rows(
        mock_rows(), "validation", tmp_path, materializer=fake_materializer
    )
    assert first == second
    assert [row.caption_id for row in first] == sorted(row.caption_id for row in first)
    assert stable_hf_image_id("test", 0, mock_rows()[0]) != stable_hf_image_id(
        "train", 0, mock_rows()[0]
    )


def test_source_split_selection_never_resplits() -> None:
    test_rows, _, _ = convert_hf_split_rows(
        mock_rows(), "test", Path("images"), materializer=fake_materializer
    )
    train_rows, _, _ = convert_hf_split_rows(
        mock_rows(), "train", Path("images"), materializer=fake_materializer
    )
    selected = select_official_split([*train_rows, *test_rows], "test", 1, 42)
    assert {row.split for row in selected} == {"test"}
    assert len({row.image_id for row in selected}) == 1
    assert len(selected) == 5


def test_safe_rematerialization_skips_valid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "already-valid.jpg"
    monkeypatch.setattr(hf_module, "_valid_image", lambda _path: True)
    image_format, written = materialize_hf_image(object(), destination, force=False)
    assert image_format == "jpeg"
    assert written is False


def test_bidirectional_reference_handles_multiple_relevant_captions() -> None:
    similarity = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8],
    ]
    text_ranks, image_ranks = bidirectional_ranks_reference(similarity, [0, 0, 1, 1])
    assert text_ranks == [1, 1, 1, 1]
    assert image_ranks == [1, 1]


def provenance() -> HFFlickr8kProvenance:
    return HFFlickr8kProvenance(
        dataset_name="jxie/flickr8k",
        requested_revision="default",
        resolved_fingerprint="dataset-fingerprint",
        source_split_image_counts={"train": 0, "validation": 0, "test": 2},
        source_split_caption_counts={"train": 0, "validation": 0, "test": 10},
        materialized_split_image_counts={"train": 0, "validation": 0, "test": 2},
        materialized_split_caption_counts={"train": 0, "validation": 0, "test": 10},
        unique_image_count=2,
        caption_count=10,
        captions_per_image_min=5,
        captions_per_image_max=5,
        captions_per_image_mean=5.0,
        image_format_counts={"jpg": 2},
        missing_or_invalid_image_count=0,
        licensing_status="unresolved",
        materialization_path="images",
        resolved_revision="resolved-revision",
    )


def test_hf_benchmark_cache_staleness_covers_relevant_inputs(tmp_path: Path) -> None:
    rows, _, _ = convert_hf_split_rows(
        mock_rows(), "test", tmp_path, materializer=fake_materializer
    )
    backend = ClipEmbeddingBackend(model_name="model", model_revision="revision")
    metadata = make_hf_cache_metadata(
        rows,
        backend,
        provenance(),
        split="test",
        max_images=100,
        seed=42,
        dimension=512,
    )
    cache = HFBenchmarkCache(metadata, {}, {})
    assert not hf_benchmark_cache_is_stale(cache, metadata)
    for field, value in [
        ("model_name", "other"),
        ("model_revision", "other-revision"),
        ("dataset_fingerprint", "other-data"),
        ("dataset_revision", "other-dataset-revision"),
        ("manifest_fingerprint", "other-manifest"),
        ("split", "validation"),
        ("max_images", None),
        ("seed", 7),
        ("backend_version", "2"),
    ]:
        assert hf_benchmark_cache_is_stale(cache, replace(metadata, **{field: value}))


def test_hf_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_module, "hfdata_dependencies_available", lambda: False)
    with pytest.raises(HFFlickr8kError) as error:
        load_hf_flickr8k(
            "jxie/flickr8k", revision=None, cache_dir=None, local_files_only=True
        )
    assert ".[dev,clip,hfdata]" in str(error.value)


def test_duplicate_source_image_is_rejected(tmp_path: Path) -> None:
    duplicate_rows = [mock_rows()[0], mock_rows()[0]]
    with pytest.raises(HFFlickr8kError, match="duplicate source image identity"):
        convert_hf_split_rows(
            duplicate_rows, "test", tmp_path, materializer=fake_materializer
        )
