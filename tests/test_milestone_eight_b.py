from dataclasses import replace
from io import BytesIO
import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, features
import pytest

from multimodal_retrieval_ops.api.app import create_app
from multimodal_retrieval_ops.api.image_inference import ImageEmbeddingCache
from multimodal_retrieval_ops.api.settings import ServiceSettings
from multimodal_retrieval_ops.api.smoke import run_service_smoke
from multimodal_retrieval_ops.cli import main
from multimodal_retrieval_ops.clip_backend import ClipModelUnavailableError
from test_milestone_seven_c import _settings


class FakeImageEncoder:
    model_name = "synthetic-model"
    model_revision = "test"
    backend_name = "huggingface-clip"
    backend_version = "1"
    dimension = 2

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [3.0, 4.0]
        self.ensure_calls = 0
        self.encode_calls = 0

    def ensure_loaded(self) -> None:
        self.ensure_calls += 1

    def encode_image_object(self, image: Image.Image) -> list[float]:
        assert image.mode == "RGB"
        self.encode_calls += 1
        return list(self.vector)


class SharedFakeEncoder(FakeImageEncoder):
    def encode_text(self, text: str) -> list[float]:
        return [3.0, 4.0]


class UnavailableImageEncoder(FakeImageEncoder):
    def ensure_loaded(self) -> None:
        raise ClipModelUnavailableError("weights unavailable")


def enabled_settings(
    tmp_path: Path, backend: str = "flat", **changes: object
) -> ServiceSettings:
    values = {
        "enable_image_inference": True,
        "image_model_name": "synthetic-model",
        "image_model_revision": "test",
        "maximum_upload_bytes": 4096,
        "maximum_pixel_count": 100,
        "image_query_cache_size": 2,
        **changes,
    }
    return replace(_settings(tmp_path, backend), **values)


def image_bytes(image_format: str = "PNG", color: str = "red", size: tuple[int, int] = (2, 2)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=image_format)
    return output.getvalue()


def upload(
    client: TestClient,
    payload: bytes,
    content_type: str = "image/png",
    top_k: int | str = 2,
    filename: str = "query.png",
):
    return client.post(
        "/search/image",
        files={"image": (filename, payload, content_type)},
        data={"top_k": str(top_k)},
    )


def factory(encoder: FakeImageEncoder):
    return lambda _settings, _shared: encoder


def test_app_import_and_disabled_mode_do_not_load_clip_or_image_encoder(
    tmp_path: Path,
) -> None:
    called = False

    def image_factory(_settings: ServiceSettings, _shared: object) -> FakeImageEncoder:
        nonlocal called
        called = True
        return FakeImageEncoder()

    with TestClient(create_app(_settings(tmp_path), image_encoder_factory=image_factory)) as client:
        ready = client.get("/ready").json()
        response = upload(client, image_bytes())
    assert called is False
    assert ready["image_encoder_enabled"] is False
    assert ready["image_encoder_ready"] is False
    assert response.status_code == 503


def test_enabled_mode_readiness_and_index_info(tmp_path: Path) -> None:
    encoder = FakeImageEncoder()
    with TestClient(
        create_app(enabled_settings(tmp_path), image_encoder_factory=factory(encoder))
    ) as client:
        ready = client.get("/ready").json()
        info = client.get("/index-info").json()
    assert encoder.ensure_calls == 1
    assert ready["status"] == "ready"
    assert ready["retrieval_artifacts_ready"] is True
    assert ready["image_encoder_ready"] is True
    assert info["image_inference_enabled"] is True
    assert info["image_model_name"] == "synthetic-model"
    assert info["vision_embedding_dimension"] == 2
    assert info["accepted_image_formats"] == ["JPEG", "PNG", "WEBP"]
    assert info["maximum_upload_bytes"] == 4096
    assert "artifact_root" not in info


def test_text_and_image_towers_share_one_encoder_lifecycle(tmp_path: Path) -> None:
    shared = SharedFakeEncoder()
    settings = replace(
        enabled_settings(tmp_path),
        enable_text_inference=True,
        text_model_name="synthetic-model",
        text_model_revision="test",
    )
    app = create_app(settings, text_encoder_factory=lambda _: shared)
    with TestClient(app) as client:
        assert client.get("/ready").json()["status"] == "ready"
    assert app.state.runtime.text_encoder is shared
    assert app.state.runtime.image_encoder is shared
    assert shared.ensure_calls == 1


@pytest.mark.parametrize(
    ("image_format", "content_type", "filename"),
    [
        ("JPEG", "image/jpeg", "query.jpg"),
        ("PNG", "image/png", "query.png"),
        pytest.param(
            "WEBP",
            "image/webp",
            "query.webp",
            marks=pytest.mark.skipif(not features.check("webp"), reason="Pillow lacks WEBP"),
        ),
    ],
)
def test_supported_upload_formats(
    tmp_path: Path, image_format: str, content_type: str, filename: str
) -> None:
    encoder = FakeImageEncoder()
    with TestClient(
        create_app(enabled_settings(tmp_path), image_encoder_factory=factory(encoder))
    ) as client:
        response = upload(client, image_bytes(image_format), content_type, filename=filename)
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "flat"
    assert body["embedding_dimension"] == 2
    assert len(body["image_identifier"]) == 16
    assert body["results"][0]["caption_text"]


@pytest.mark.parametrize(
    ("payload", "content_type", "settings_changes", "message"),
    [
        (b"", "image/png", {}, "non-empty"),
        (b"x" * 200, "image/png", {"maximum_upload_bytes": 100}, "byte limit"),
        (b"hello", "application/octet-stream", {}, "content type"),
        (image_bytes("PNG"), "image/jpeg", {}, "does not match"),
        (b"not-an-image", "image/png", {}, "corrupt or undecodable"),
        (
            image_bytes("PNG", size=(11, 10)),
            "image/png",
            {"maximum_pixel_count": 100},
            "pixel limit",
        ),
    ],
)
def test_strict_upload_validation(
    tmp_path: Path,
    payload: bytes,
    content_type: str,
    settings_changes: dict[str, object],
    message: str,
) -> None:
    settings = enabled_settings(tmp_path, **settings_changes)
    with TestClient(
        create_app(settings, image_encoder_factory=factory(FakeImageEncoder()))
    ) as client:
        response = upload(client, payload, content_type)
        metrics = client.get("/metrics").json()
    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert metrics["image_validation_errors"] == 1


def test_nonfinite_vector_is_rejected_without_details(tmp_path: Path) -> None:
    encoder = FakeImageEncoder([float("nan"), 1.0])
    with TestClient(
        create_app(enabled_settings(tmp_path), image_encoder_factory=factory(encoder))
    ) as client:
        response = upload(client, image_bytes())
        metrics = client.get("/metrics").json()
    assert response.status_code == 503
    assert response.json()["detail"] == "arbitrary image inference execution failed"
    assert metrics["image_inference_errors"] == 1


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_search_normalizes_vector_and_returns_safe_caption_fields(
    tmp_path: Path, backend: str
) -> None:
    encoder = FakeImageEncoder()
    app = create_app(
        enabled_settings(tmp_path, backend), image_encoder_factory=factory(encoder)
    )
    with TestClient(app) as client:
        response = upload(client, image_bytes())
        cached_vector = next(iter(app.state.runtime.image_query_cache._values.values()))
        ef_search = (
            app.state.runtime.artifacts.image_to_text.index.hnsw.efSearch
            if backend == "hnsw"
            else None
        )
    assert response.status_code == 200
    assert math.isclose(math.sqrt(sum(value * value for value in cached_vector)), 1.0)
    body = response.json()
    assert body["backend"] == backend
    assert set(body["results"][0]) == {
        "caption_id",
        "target_image_id",
        "caption_text",
        "score",
        "rank",
        "split",
    }
    assert body["results"][0]["split"] == "test"
    assert str(tmp_path) not in json.dumps(body)
    assert "filename" not in json.dumps(body).lower()
    if backend == "hnsw":
        assert ef_search == 64


def test_identical_bytes_cache_hit_metrics_and_no_disk_write(tmp_path: Path) -> None:
    encoder = FakeImageEncoder()
    settings = enabled_settings(tmp_path)
    before = {path: path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()}
    with TestClient(create_app(settings, image_encoder_factory=factory(encoder))) as client:
        first = upload(client, image_bytes()).json()
        second = upload(client, image_bytes()).json()
        metrics = client.get("/metrics").json()
    after = {path: path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()}
    assert first["cached_query"] is False
    assert second["cached_query"] is True
    assert encoder.encode_calls == 1
    assert metrics["arbitrary_image_request_count"] == 2
    assert metrics["image_encoder_invocation_count"] == 1
    assert metrics["image_query_cache_misses"] == 1
    assert metrics["image_query_cache_hits"] == 1
    assert metrics["uploaded_byte_total"] == 2 * len(image_bytes())
    assert metrics["image_inference_latency_count"] == 2
    assert metrics["requests_by_backend"]["flat"] == 2
    assert after == before


def test_image_cache_is_bounded_lru() -> None:
    cache = ImageEmbeddingCache(2)
    cache.put("one", [1.0])
    cache.put("two", [2.0])
    assert cache.get("one") == [1.0]
    cache.put("three", [3.0])
    assert len(cache) == 2
    assert cache.get("two") is None


def test_encoder_failure_keeps_cached_id_retrieval_available(tmp_path: Path) -> None:
    app = create_app(
        enabled_settings(tmp_path),
        image_encoder_factory=factory(UnavailableImageEncoder()),
    )
    with TestClient(app) as client:
        ready = client.get("/ready").json()
        cached = client.post(
            "/retrieve/captions", json={"image_id": "image-a", "top_k": 1}
        )
        live = upload(client, image_bytes())
    assert ready["status"] == "unready"
    assert ready["retrieval_artifacts_ready"] is True
    assert ready["image_encoder_state"] == "model_unavailable"
    assert cached.status_code == 200
    assert live.status_code == 503


def test_image_smoke_and_cli_use_injected_encoder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    query_path = tmp_path / "query.png"
    query_path.write_bytes(image_bytes())
    settings = enabled_settings(tmp_path, smoke_image_path=query_path)
    encoder = FakeImageEncoder()
    result = run_service_smoke(settings, image_encoder_factory=factory(encoder))
    assert result.run_state == "success"
    assert result.image_search_responses[0]["cached_query"] is False
    assert result.image_search_responses[1]["cached_query"] is True
    assert result.observability["image_encoder_invocation_count"] == 1

    common = [
        "--backend",
        "flat",
        "--artifact-root",
        str(settings.artifact_root),
        "--embedding-cache",
        str(settings.embedding_cache_path),
        "--manifest",
        str(settings.manifest_path),
        "--enable-image-inference",
        "--image-model-name",
        "synthetic-model",
        "--image-model-revision",
        "test",
        "--maximum-top-k",
        "3",
    ]
    report = tmp_path / "report.md"
    metrics = tmp_path / "metrics.json"
    assert main(
        [
            "retrieval-service-smoke",
            *common,
            "--smoke-image-path",
            str(query_path),
            "--report-output",
            str(report),
            "--metrics-output",
            str(metrics),
        ],
        image_encoder_factory=factory(FakeImageEncoder()),
    ) == 0
    assert "Run state: **success**" in report.read_text(encoding="utf-8")
    assert main(
        ["search-live-image", *common, "--image-path", str(query_path), "--top-k", "2"],
        image_encoder_factory=factory(FakeImageEncoder()),
    ) == 0
    captured = capsys.readouterr().out
    assert "Image inference smoke passed" in captured
    assert '"caption_text"' in captured
