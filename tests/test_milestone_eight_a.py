from dataclasses import replace
import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from multimodal_retrieval_ops.api.app import create_app
from multimodal_retrieval_ops.api.settings import ServiceSettings
from multimodal_retrieval_ops.api.smoke import run_service_smoke
from multimodal_retrieval_ops.api.text_inference import QueryEmbeddingCache
from multimodal_retrieval_ops.cli import main
from multimodal_retrieval_ops.clip_backend import ClipModelUnavailableError
from test_milestone_seven_c import _settings


class FakeTextEncoder:
    model_name = "synthetic-model"
    model_revision = "test"
    backend_name = "huggingface-clip"
    backend_version = "1"

    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension
        self.ensure_calls = 0
        self.encode_calls = 0

    def ensure_loaded(self) -> None:
        self.ensure_calls += 1

    def encode_text(self, text: str) -> list[float]:
        self.encode_calls += 1
        return [3.0, 4.0] if self.dimension == 2 else [1.0] * self.dimension


class UnavailableTextEncoder(FakeTextEncoder):
    def ensure_loaded(self) -> None:
        raise ClipModelUnavailableError("weights unavailable")


def enabled_settings(
    tmp_path: Path, backend: str = "flat", **changes: object
) -> ServiceSettings:
    settings = _settings(tmp_path, backend)
    values = {
        "enable_text_inference": True,
        "text_model_name": "synthetic-model",
        "text_model_revision": "test",
        "maximum_text_length": 40,
        "text_query_cache_size": 2,
        **changes,
    }
    return replace(settings, **values)


def test_disabled_text_inference_does_not_instantiate_encoder(tmp_path: Path) -> None:
    called = False

    def factory(settings: ServiceSettings) -> FakeTextEncoder:
        nonlocal called
        called = True
        return FakeTextEncoder()

    with TestClient(create_app(_settings(tmp_path), text_encoder_factory=factory)) as client:
        ready = client.get("/ready").json()
        response = client.post("/search/text", json={"query": "dog", "top_k": 1})
    assert called is False
    assert ready["status"] == "ready"
    assert ready["text_encoder_enabled"] is False
    assert ready["text_encoder_ready"] is False
    assert response.status_code == 503


def test_enabled_encoder_initialization_and_readiness(tmp_path: Path) -> None:
    encoder = FakeTextEncoder()
    with TestClient(
        create_app(enabled_settings(tmp_path), text_encoder_factory=lambda _: encoder)
    ) as client:
        ready = client.get("/ready").json()
        info = client.get("/index-info").json()
    assert encoder.ensure_calls == 1
    assert ready["status"] == "ready"
    assert ready["retrieval_artifacts_ready"] is True
    assert ready["text_encoder_ready"] is True
    assert info["text_inference_enabled"] is True
    assert info["text_model_name"] == "synthetic-model"
    assert info["text_embedding_dimension"] == 2
    assert info["local_files_only"] is True


def test_model_and_dimension_mismatches_are_unready(tmp_path: Path) -> None:
    mismatched_model = enabled_settings(tmp_path / "model", text_model_name="other-model")
    with TestClient(
        create_app(mismatched_model, text_encoder_factory=lambda _: FakeTextEncoder())
    ) as client:
        model_ready = client.get("/ready").json()
    mismatched_dimension = enabled_settings(tmp_path / "dimension")
    with TestClient(
        create_app(
            mismatched_dimension,
            text_encoder_factory=lambda _: FakeTextEncoder(dimension=3),
        )
    ) as client:
        dimension_ready = client.get("/ready").json()
    assert model_ready["retrieval_artifacts_ready"] is True
    assert model_ready["text_encoder_state"] == "artifact_incompatible"
    assert dimension_ready["text_encoder_state"] == "artifact_incompatible"


def test_successful_flat_text_search_normalizes_vector_and_returns_safe_metadata(
    tmp_path: Path,
) -> None:
    encoder = FakeTextEncoder()
    app = create_app(enabled_settings(tmp_path), text_encoder_factory=lambda _: encoder)
    with TestClient(app) as client:
        response = client.post("/search/text", json={"query": "  Alpha   scene ", "top_k": 2})
        cached_vector = app.state.runtime.query_cache.get("alpha scene")
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Alpha scene"
    assert body["cached_query"] is False
    assert body["model_name"] == "synthetic-model"
    assert math.isclose(math.sqrt(sum(value * value for value in cached_vector)), 1.0)
    assert body["results"][0]["split"] == "test"
    assert body["results"][0]["image_path"].startswith("test/")
    assert not Path(body["results"][0]["image_path"]).is_absolute()
    assert body["results"][0]["captions"]
    assert str(tmp_path) not in json.dumps(body)


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("   ", 1, "query must be non-empty"),
        ("x" * 41, 1, "must not exceed 40 characters"),
        ("dog", 0, "top_k must be positive"),
        ("dog", 4, "configured maximum 3"),
    ],
)
def test_text_input_validation(
    tmp_path: Path, query: str, top_k: int, message: str
) -> None:
    app = create_app(
        enabled_settings(tmp_path), text_encoder_factory=lambda _: FakeTextEncoder()
    )
    with TestClient(app) as client:
        response = client.post("/search/text", json={"query": query, "top_k": top_k})
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_repeated_query_cache_hit_and_metrics(tmp_path: Path) -> None:
    encoder = FakeTextEncoder()
    app = create_app(enabled_settings(tmp_path), text_encoder_factory=lambda _: encoder)
    with TestClient(app) as client:
        first = client.post("/search/text", json={"query": "DOG running", "top_k": 1}).json()
        second = client.post(
            "/search/text", json={"query": " dog   RUNNING ", "top_k": 1}
        ).json()
        metrics = client.get("/metrics").json()
    assert first["cached_query"] is False
    assert second["cached_query"] is True
    assert encoder.encode_calls == 1
    assert metrics["arbitrary_text_request_count"] == 2
    assert metrics["text_encoder_invocation_count"] == 1
    assert metrics["text_query_cache_misses"] == 1
    assert metrics["text_query_cache_hits"] == 1
    assert metrics["text_inference_latency_count"] == 2
    assert metrics["requests_by_backend"]["flat"] == 2
    assert metrics["text_inference_latency_p95_seconds"] >= 0


def test_query_cache_is_bounded_lru() -> None:
    cache = QueryEmbeddingCache(2)
    cache.put("one", [1.0])
    cache.put("two", [2.0])
    assert cache.get("one") == [1.0]
    cache.put("three", [3.0])
    assert len(cache) == 2
    assert cache.get("two") is None
    assert cache.get("one") == [1.0]


def test_hnsw_text_search_applies_ef_search(tmp_path: Path) -> None:
    settings = enabled_settings(tmp_path, "hnsw")
    app = create_app(settings, text_encoder_factory=lambda _: FakeTextEncoder())
    with TestClient(app) as client:
        response = client.post("/search/text", json={"query": "alpha", "top_k": 2})
        ef_search = app.state.runtime.artifacts.text_to_image.index.hnsw.efSearch
    assert response.status_code == 200
    assert response.json()["backend"] == "hnsw"
    assert ef_search == 64


def test_encoder_failure_keeps_cached_retrieval_ready(tmp_path: Path) -> None:
    app = create_app(
        enabled_settings(tmp_path), text_encoder_factory=lambda _: UnavailableTextEncoder()
    )
    with TestClient(app) as client:
        ready = client.get("/ready").json()
        cached = client.post(
            "/retrieve/images", json={"caption_id": "caption-a1", "top_k": 1}
        )
        live = client.post("/search/text", json={"query": "dog", "top_k": 1})
    assert ready["status"] == "unready"
    assert ready["retrieval_artifacts_ready"] is True
    assert ready["text_encoder_state"] == "model_unavailable"
    assert cached.status_code == 200
    assert live.status_code == 503


def test_text_smoke_verifies_one_invocation_and_cache_hit(tmp_path: Path) -> None:
    encoder = FakeTextEncoder()
    result = run_service_smoke(
        enabled_settings(tmp_path), text_encoder_factory=lambda _: encoder
    )
    assert result.run_state == "success"
    assert result.text_search_responses[0]["cached_query"] is False
    assert result.text_search_responses[1]["cached_query"] is True
    assert result.observability["text_encoder_invocation_count"] == 1
    assert result.observability["text_query_cache_hits"] == 1


def test_cli_smoke_and_live_search_with_injected_encoder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = enabled_settings(tmp_path)
    common = [
        "--backend",
        "flat",
        "--artifact-root",
        str(settings.artifact_root),
        "--embedding-cache",
        str(settings.embedding_cache_path),
        "--manifest",
        str(settings.manifest_path),
        "--enable-text-inference",
        "--text-model-name",
        "synthetic-model",
        "--text-model-revision",
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
            "--report-output",
            str(report),
            "--metrics-output",
            str(metrics),
        ],
        text_encoder_factory=lambda _: FakeTextEncoder(),
    ) == 0
    assert "Run state: **success**" in report.read_text(encoding="utf-8")
    assert main(
        ["search-live-text", *common, "--query", "alpha", "--top-k", "2"],
        text_encoder_factory=lambda _: FakeTextEncoder(),
    ) == 0
    captured = capsys.readouterr().out
    assert "Text inference smoke passed" in captured
    assert '"cached_query": false' in captured
