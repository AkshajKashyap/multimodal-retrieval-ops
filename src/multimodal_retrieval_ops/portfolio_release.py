"""Deterministic synthetic release smoke for portfolio verification."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .api.settings import ServiceSettings
from .embedding_cache import manifest_digest
from .faiss_flat import (
    FaissIndexArtifact,
    build_flat_ip_index,
    file_sha256,
    make_index_metadata,
    ordered_embeddings,
    require_faiss,
    save_faiss_artifact,
)
from .hf_clip_benchmark import (
    HFBenchmarkCache,
    HFBenchmarkCacheMetadata,
    write_hf_benchmark_cache,
)
from .manifest import ManifestItemV2, write_manifest
from .retrieval_monitoring import analyze_events, read_telemetry

RELEASE_VERSION = "1.0.0"


@dataclass(frozen=True)
class PortfolioSmokeResult:
    release_version: str
    smoke_state: str
    synthetic_artifacts: bool
    neural_inference_used: bool
    retrieval_directions_exercised: list[str]
    service_endpoints_exercised: list[str]
    telemetry_exercised: bool
    telemetry_schema_version: int
    telemetry_event_count: int
    monitoring_health_decision: str
    supplied_test_count: int | None
    major_capabilities: list[str]
    known_limitations: list[str]


def _synthetic_rows() -> list[ManifestItemV2]:
    return [
        ManifestItemV2(
            "train-image", "train-caption", "train.jpg", "training caption", "train", "synthetic"
        ),
        ManifestItemV2(
            "validation-image",
            "validation-caption",
            "validation.jpg",
            "validation caption",
            "validation",
            "synthetic",
        ),
        ManifestItemV2(
            "image-alpha", "caption-alpha-1", "alpha.jpg", "alpha", "test", "synthetic"
        ),
        ManifestItemV2(
            "image-alpha",
            "caption-alpha-2",
            "alpha.jpg",
            "near alpha",
            "test",
            "synthetic",
        ),
        ManifestItemV2(
            "image-beta", "caption-beta-1", "beta.jpg", "beta", "test", "synthetic"
        ),
    ]


def prepare_synthetic_service_artifacts(root: Path) -> ServiceSettings:
    """Create tiny normalized FlatIP artifacts without images, CLIP, or network access."""
    artifact_root = root / "artifacts"
    cache_path = artifact_root / "clip" / "synthetic_cache.json"
    manifest_path = root / "synthetic_manifest.csv"
    rows = _synthetic_rows()
    write_manifest(rows, manifest_path)
    test_rows = [row for row in rows if row.split == "test"]
    cache = HFBenchmarkCache(
        HFBenchmarkCacheMetadata(
            backend_name="synthetic-release",
            backend_version="1",
            model_name="synthetic-normalized-vectors",
            model_revision="1",
            dataset_fingerprint="synthetic-portfolio-dataset-v1",
            manifest_fingerprint=manifest_digest(test_rows),
            split="test",
            max_images=None,
            seed=42,
            image_count=2,
            caption_count=3,
            embedding_dimension=2,
        ),
        image_embeddings={"image-alpha": [1.0, 0.0], "image-beta": [0.0, 1.0]},
        caption_embeddings={
            "caption-alpha-1": [1.0, 0.0],
            "caption-alpha-2": [0.8, 0.6],
            "caption-beta-1": [0.0, 1.0],
        },
    )
    write_hf_benchmark_cache(cache, cache_path)
    fingerprint = file_sha256(cache_path)
    faiss = require_faiss()
    index_root = artifact_root / "faiss"
    for direction, embeddings in (
        ("text_to_image", cache.image_embeddings),
        ("image_to_text", cache.caption_embeddings),
    ):
        candidate_ids, vectors = ordered_embeddings(embeddings)
        metadata = make_index_metadata(
            cache,
            direction=direction,
            candidate_ids=candidate_ids,
            source_cache_fingerprint=fingerprint,
            faiss_version=faiss.__version__,
        )
        save_faiss_artifact(
            FaissIndexArtifact(metadata, build_flat_ip_index(vectors, 2)),
            index_root / f"{direction}.faiss",
            index_root / f"{direction}.json",
        )
    return ServiceSettings(
        backend="flat",
        artifact_root=artifact_root,
        embedding_cache_path=cache_path,
        manifest_path=manifest_path,
        maximum_top_k=3,
        telemetry_enabled=True,
        telemetry_path=root / "telemetry" / "portfolio.jsonl",
        telemetry_max_bytes=100_000,
        telemetry_backup_count=1,
    )


def run_portfolio_smoke(*, supplied_test_count: int | None = None) -> PortfolioSmokeResult:
    """Exercise synthetic bidirectional retrieval, serving, and monitoring in process."""
    if supplied_test_count is not None and supplied_test_count < 0:
        raise ValueError("supplied test count must be non-negative")
    from fastapi.testclient import TestClient

    from .api.app import create_app

    with TemporaryDirectory(prefix="multimodal-release-") as temporary:
        settings = prepare_synthetic_service_artifacts(Path(temporary))
        app = create_app(settings)
        with TestClient(app) as client:
            health = client.get("/health")
            ready = client.get("/ready")
            images = client.post(
                "/retrieve/images",
                json={"caption_id": "caption-alpha-1", "top_k": 2},
            )
            captions = client.post(
                "/retrieve/captions",
                json={"image_id": "image-beta", "top_k": 3},
            )
        if health.json() != {"status": "alive"}:
            raise ValueError("synthetic service health check failed")
        if ready.json().get("status") != "ready":
            raise ValueError("synthetic service readiness check failed")
        if images.status_code != 200 or not images.json()["results"][0]["relevant_target"]:
            raise ValueError("synthetic text-to-image retrieval failed")
        if captions.status_code != 200 or not captions.json()["results"][0]["relevant_target"]:
            raise ValueError("synthetic image-to-text retrieval failed")
        summary = analyze_events(read_telemetry(settings.telemetry_path))
        if summary["health"]["decision"] != "insufficient_data":
            raise ValueError("synthetic telemetry must remain insufficient_data")
        event_count = int(summary["traffic"]["total_event_count"])
    return PortfolioSmokeResult(
        release_version=RELEASE_VERSION,
        smoke_state="success",
        synthetic_artifacts=True,
        neural_inference_used=False,
        retrieval_directions_exercised=["text_to_image", "image_to_text"],
        service_endpoints_exercised=[
            "/health",
            "/ready",
            "/retrieve/images",
            "/retrieve/captions",
        ],
        telemetry_exercised=True,
        telemetry_schema_version=1,
        telemetry_event_count=event_count,
        monitoring_health_decision="insufficient_data",
        supplied_test_count=supplied_test_count,
        major_capabilities=[
            "schema-v2 multi-caption relationships",
            "normalized bidirectional FlatIP retrieval",
            "in-process FastAPI cached-ID serving",
            "privacy-safe JSONL telemetry",
            "offline monitoring with minimum-sample safeguards",
        ],
        known_limitations=[
            "synthetic smoke validates integration, not model quality",
            "neural inference and real datasets are intentionally excluded",
            "monitoring smoke is insufficient for production-health conclusions",
            "the release is not evidence of internet-scale or deployed production operation",
        ],
    )


def render_portfolio_report(result: PortfolioSmokeResult) -> str:
    test_count = "not supplied" if result.supplied_test_count is None else result.supplied_test_count
    return "\n".join(
        [
            "# Portfolio Release 1.0.0",
            "",
            f"Release version: **{result.release_version}**",
            "",
            f"- Smoke state: **{result.smoke_state}**",
            f"- Synthetic artifacts only: **{str(result.synthetic_artifacts).lower()}**",
            f"- Neural inference used: **{str(result.neural_inference_used).lower()}**",
            "- Retrieval directions: " + ", ".join(result.retrieval_directions_exercised),
            "- Service endpoints: " + ", ".join(result.service_endpoints_exercised),
            f"- Telemetry exercised: **{str(result.telemetry_exercised).lower()}**",
            f"- Telemetry schema/events: {result.telemetry_schema_version}/{result.telemetry_event_count}",
            f"- Monitoring decision: **{result.monitoring_health_decision}**",
            f"- Supplied test count: {test_count}",
            "",
            "## Major capabilities",
            "",
            *(f"- {capability}" for capability in result.major_capabilities),
            "",
            "## Known limitations",
            "",
            *(f"- {limitation}" for limitation in result.known_limitations),
            "",
            "This deterministic smoke uses temporary two-dimensional normalized vectors and tiny",
            "FlatIP indexes. It performs no download, neural inference, dataset access, training,",
            "benchmarking, or real-artifact mutation.",
            "",
        ]
    )


def write_portfolio_outputs(
    result: PortfolioSmokeResult, report_path: Path, metrics_path: Path
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_portfolio_report(result), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_portfolio_output(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        raise ValueError("portfolio smoke output is missing; run portfolio-smoke first")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "release_version": RELEASE_VERSION,
        "smoke_state": "success",
        "synthetic_artifacts": True,
        "neural_inference_used": False,
        "telemetry_exercised": True,
        "monitoring_health_decision": "insufficient_data",
    }
    if any(data.get(name) != value for name, value in expected.items()):
        raise ValueError("portfolio smoke output is incompatible with release 1.0.0")
    return data
