from pathlib import Path

import pytest

import multimodal_retrieval_ops.clip_backend as clip_backend_module
from multimodal_retrieval_ops.clip_backend import (
    DEFAULT_CLIP_MODEL,
    ClipBackendError,
    ClipEmbeddingBackend,
)
from multimodal_retrieval_ops.clip_reporting import render_clip_backend_report
from multimodal_retrieval_ops.embedding_cache import (
    EmbeddingCache,
    cache_is_stale,
    load_embedding_cache,
    make_cache_metadata,
    write_embedding_cache,
)
from multimodal_retrieval_ops.manifest import ManifestItem
from multimodal_retrieval_ops.multimodal_index import (
    MultimodalIndex,
    MultimodalIndexEntry,
    load_multimodal_index,
    write_multimodal_index,
)
from test_milestone_one import run_cli


def cache_items() -> list[ManifestItem]:
    return [ManifestItem("one", "one.jpg", "red car", "train", "fixture")]


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clip_backend_module, "clip_dependencies_available", lambda: False)
    backend = ClipEmbeddingBackend()
    with pytest.raises(ClipBackendError) as error:
        backend.encode_text("red car")
    assert ".[dev,clip]" in str(error.value)
    assert "pip install" in str(error.value)


def test_clip_backend_metadata_contract() -> None:
    backend = ClipEmbeddingBackend(model_name="local/model", device="cpu", batch_size=4)
    assert backend.backend_name == "huggingface-clip"
    assert backend.backend_version
    assert callable(backend.encode_text)
    assert callable(backend.encode_image)
    assert backend.metadata() == {
        "backend_name": "huggingface-clip",
        "backend_version": "1",
        "model_name": "local/model",
        "device": "cpu",
        "batch_size": 4,
        "dimension": 0,
    }


def test_cache_metadata_creation_and_round_trip(tmp_path: Path) -> None:
    items = cache_items()
    metadata = make_cache_metadata(
        items,
        backend_name="clip",
        backend_version="1",
        model_name="model",
        embedding_dimension=2,
    )
    cache = EmbeddingCache(metadata, {"one": [1.0, 0.0]}, {"one": [0.0, 1.0]})
    path = tmp_path / "cache.json"
    write_embedding_cache(cache, path)
    assert load_embedding_cache(path) == cache
    assert metadata.item_count == 1
    assert len(metadata.manifest_hash) == 64


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("backend_name", "other"),
        ("backend_version", "2"),
        ("model_name", "other/model"),
        ("embedding_dimension", 3),
    ],
)
def test_stale_cache_detection(changed_field: str, changed_value: str | int) -> None:
    items = cache_items()
    original = make_cache_metadata(
        items,
        backend_name="clip",
        backend_version="1",
        model_name="model",
        embedding_dimension=2,
    )
    values = original.__dict__ | {changed_field: changed_value}
    changed = type(original)(**values)
    cache = EmbeddingCache(original, {"one": [1.0, 0.0]}, {"one": [0.0, 1.0]})
    assert cache_is_stale(cache, changed)


def test_manifest_change_marks_cache_stale() -> None:
    items = cache_items()
    original = make_cache_metadata(
        items,
        backend_name="clip",
        backend_version="1",
        model_name="model",
        embedding_dimension=2,
    )
    changed_items = [ManifestItem("one", "one.jpg", "blue car", "train", "fixture")]
    changed = make_cache_metadata(
        changed_items,
        backend_name="clip",
        backend_version="1",
        model_name="model",
        embedding_dimension=2,
    )
    cache = EmbeddingCache(original, {"one": [1.0, 0.0]}, {"one": [0.0, 1.0]})
    assert cache_is_stale(cache, changed)


def test_clip_index_schema_uses_shared_multimodal_contract(tmp_path: Path) -> None:
    index = MultimodalIndex(
        backend_name="huggingface-clip",
        backend_version="1",
        dimension=2,
        entries=[
            MultimodalIndexEntry(
                "one", "one.jpg", "red car", "test", "fixture", [1.0, 0.0]
            )
        ],
        model_name="local/model",
    )
    path = tmp_path / "index.json"
    write_multimodal_index(index, path)
    assert load_multimodal_index(path) == index


def test_clip_backend_info_cli_smoke(tmp_path: Path) -> None:
    report = tmp_path / "backend.md"
    result = run_cli("clip-backend-info", "--output", str(report))
    assert result.returncode == 0, result.stderr
    assert DEFAULT_CLIP_MODEL in result.stdout
    assert "CLIP Backend Report" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "status",
    [
        "successfully executed",
        "unavailable dependencies",
        "unavailable model weights",
        "execution failure",
    ],
)
def test_backend_report_distinguishes_execution_states(status: str) -> None:
    assert f"**{status}**" in render_clip_backend_report(status)
