from dataclasses import replace
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from multimodal_retrieval_ops.api.app import create_app
from multimodal_retrieval_ops.api.settings import ServiceSettings
from multimodal_retrieval_ops.embedding_cache import manifest_digest
from multimodal_retrieval_ops.faiss_flat import (
    FaissIndexArtifact,
    build_flat_ip_index,
    file_sha256,
    make_index_metadata,
    ordered_embeddings,
    require_faiss,
    save_faiss_artifact,
)
from multimodal_retrieval_ops.faiss_hnsw import (
    HNSWIndexArtifact,
    build_hnsw_index,
    make_hnsw_metadata,
    save_hnsw_artifact,
)
from multimodal_retrieval_ops.hf_clip_benchmark import (
    HFBenchmarkCache,
    HFBenchmarkCacheMetadata,
    write_hf_benchmark_cache,
)
from multimodal_retrieval_ops.manifest import ManifestItemV2, write_manifest
from test_milestone_one import run_cli


def _rows() -> list[ManifestItemV2]:
    return [
        ManifestItemV2("train-image", "train-caption", "train.jpg", "train", "train", "test"),
        ManifestItemV2(
            "validation-image",
            "validation-caption",
            "validation.jpg",
            "validation",
            "validation",
            "test",
        ),
        ManifestItemV2("image-a", "caption-a1", "a.jpg", "alpha", "test", "test"),
        ManifestItemV2("image-a", "caption-a2", "a.jpg", "alpha nearby", "test", "test"),
        ManifestItemV2("image-b", "caption-b1", "b.jpg", "beta", "test", "test"),
    ]


def _settings(tmp_path: Path, backend: str = "flat") -> ServiceSettings:
    artifact_root = tmp_path / "artifacts"
    cache_path = artifact_root / "clip" / "cache.json"
    manifest_path = tmp_path / "manifest.csv"
    rows = _rows()
    write_manifest(rows, manifest_path)
    test_rows = [row for row in rows if row.split == "test"]
    cache = HFBenchmarkCache(
        HFBenchmarkCacheMetadata(
            backend_name="huggingface-clip",
            backend_version="1",
            model_name="synthetic-model",
            model_revision="test",
            dataset_fingerprint="synthetic-dataset",
            manifest_fingerprint=manifest_digest(test_rows),
            split="test",
            max_images=None,
            seed=42,
            image_count=2,
            caption_count=3,
            embedding_dimension=2,
        ),
        image_embeddings={"image-a": [1.0, 0.0], "image-b": [0.0, 1.0]},
        caption_embeddings={
            "caption-a1": [1.0, 0.0],
            "caption-a2": [0.8, 0.6],
            "caption-b1": [0.0, 1.0],
        },
    )
    write_hf_benchmark_cache(cache, cache_path)
    fingerprint = file_sha256(cache_path)
    faiss = require_faiss()
    index_root = artifact_root / ("faiss" if backend == "flat" else "faiss_hnsw")
    for direction, embeddings in (
        ("text_to_image", cache.image_embeddings),
        ("image_to_text", cache.caption_embeddings),
    ):
        ids, vectors = ordered_embeddings(embeddings)
        if backend == "flat":
            metadata = make_index_metadata(
                cache,
                direction=direction,
                candidate_ids=ids,
                source_cache_fingerprint=fingerprint,
                faiss_version=faiss.__version__,
            )
            artifact = FaissIndexArtifact(metadata, build_flat_ip_index(vectors, 2))
            save_faiss_artifact(
                artifact,
                index_root / f"{direction}.faiss",
                index_root / f"{direction}.json",
            )
        else:
            metadata = make_hnsw_metadata(
                cache,
                direction=direction,
                candidate_ids=ids,
                source_cache_fingerprint=fingerprint,
                faiss_version=faiss.__version__,
            )
            artifact = HNSWIndexArtifact(metadata, build_hnsw_index(vectors, 2))
            save_hnsw_artifact(
                artifact,
                index_root / f"{direction}.faiss",
                index_root / f"{direction}.json",
            )
    return ServiceSettings(
        backend=backend,
        artifact_root=artifact_root,
        embedding_cache_path=cache_path,
        manifest_path=manifest_path,
        maximum_top_k=3,
        ef_search=64,
    )


def test_app_import_does_not_import_torch() -> None:
    import sys

    before = "torch" in sys.modules
    create_app(ServiceSettings())
    assert ("torch" in sys.modules) is before


def test_health_is_liveness_when_artifacts_are_missing(tmp_path: Path) -> None:
    settings = replace(ServiceSettings(), artifact_root=tmp_path, embedding_cache_path=tmp_path / "x")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "alive"}
        ready = client.get("/ready").json()
    assert ready["status"] == "unready"
    assert ready["artifact_validation"] == "artifact_unavailable"
    assert not any(str(tmp_path) in reason for reason in ready["reasons"])


def test_ready_and_index_info_with_valid_flat_artifacts(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/ready").json()["status"] == "ready"
        info = client.get("/index-info").json()
    assert info["backend"] == "flat"
    assert info["index_type"] == "IndexFlatIP"
    assert info["image_candidate_count"] == 2
    assert info["caption_candidate_count"] == 3
    assert info["ef_search"] is None


def test_stale_metadata_reports_unready_without_details(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    metadata_path = settings.index_artifacts_path / "text_to_image.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    data["dataset_fingerprint"] = "stale"
    metadata_path.write_text(json.dumps(data), encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready").json()
    assert ready["status"] == "unready"
    assert ready["artifact_validation"] == "artifact_incompatible"
    assert all("Traceback" not in reason for reason in ready["reasons"])


def test_flat_retrieval_in_both_directions(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        images = client.post(
            "/retrieve/images", json={"caption_id": "caption-a1", "top_k": 2}
        )
        captions = client.post(
            "/retrieve/captions", json={"image_id": "image-b", "top_k": 3}
        )
    assert images.status_code == 200
    assert images.json()["results"][0] == {
        "image_id": "image-a",
        "score": 1.0,
        "rank": 1,
        "relevant_target": True,
    }
    assert captions.status_code == 200
    first = captions.json()["results"][0]
    assert first["caption_id"] == "caption-b1"
    assert first["target_image_id"] == "image-b"
    assert first["caption_text"] == "beta"
    assert first["relevant_target"] is True


def test_hnsw_retrieval_applies_ef_search(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, "hnsw"))
    with TestClient(app) as client:
        response = client.post(
            "/retrieve/images", json={"caption_id": "caption-a1", "top_k": 2}
        )
        info = client.get("/index-info").json()
        applied = app.state.runtime.artifacts.text_to_image.index.hnsw.efSearch
    assert response.status_code == 200
    assert response.json()["backend"] == "hnsw"
    assert info["ef_search"] == applied == 64


@pytest.mark.parametrize(
    ("endpoint", "payload", "detail"),
    [
        ("/retrieve/images", {"caption_id": "missing", "top_k": 1}, "unknown caption_id"),
        ("/retrieve/captions", {"image_id": "missing", "top_k": 1}, "unknown image_id"),
    ],
)
def test_unknown_query_ids_return_404(
    tmp_path: Path, endpoint: str, payload: dict[str, object], detail: str
) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(endpoint, json=payload)
        metrics = client.get("/metrics").json()
    assert response.status_code == 404
    assert response.json()["detail"] == detail
    assert metrics["unknown_query_id_count"] == 1


@pytest.mark.parametrize("top_k", [0, -1, 4])
def test_invalid_top_k_is_typed_and_counted(tmp_path: Path, top_k: int) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/retrieve/images", json={"caption_id": "caption-a1", "top_k": top_k}
        )
        metrics = client.get("/metrics").json()
    assert response.status_code == 422
    assert "top_k" in response.json()["detail"]
    assert metrics["invalid_top_k_count"] == 1


def test_metrics_counters_and_latency_summary(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        client.post("/retrieve/images", json={"caption_id": "caption-a1", "top_k": 1})
        client.post("/retrieve/captions", json={"image_id": "image-a", "top_k": 1})
        metrics = client.get("/metrics").json()
    assert metrics["retrieval_requests_by_direction"] == {
        "caption_to_image": 1,
        "image_to_caption": 1,
    }
    assert metrics["requests_by_backend"]["flat"] == 2
    assert metrics["latency_observation_count"] == 2
    assert metrics["mean_latency_seconds"] >= 0
    assert metrics["p50_latency_seconds"] >= 0
    assert metrics["p95_latency_seconds"] >= metrics["p50_latency_seconds"]


def test_startup_does_not_rebuild_artifacts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paths = sorted(settings.index_artifacts_path.iterdir())
    before = {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in paths}
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").json()["status"] == "ready"
    after = {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in paths}
    assert after == before


def test_cli_info_and_in_process_smoke(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    common = (
        "--backend",
        "flat",
        "--artifact-root",
        str(settings.artifact_root),
        "--embedding-cache",
        str(settings.embedding_cache_path),
        "--manifest",
        str(settings.manifest_path),
        "--maximum-top-k",
        "3",
    )
    info = run_cli("retrieval-service-info", *common)
    assert info.returncode == 0, info.stderr
    assert json.loads(info.stdout)["ready"] is True
    report = tmp_path / "report.md"
    metrics = tmp_path / "metrics.json"
    smoke = run_cli(
        "retrieval-service-smoke",
        *common,
        "--report-output",
        str(report),
        "--metrics-output",
        str(metrics),
    )
    assert smoke.returncode == 0, smoke.stderr
    assert "Run state: **success**" in report.read_text(encoding="utf-8")
    assert json.loads(metrics.read_text(encoding="utf-8"))["run_state"] == "success"
