"""One bounded in-process cached-ID telemetry smoke workflow."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from .app import create_app
from .settings import ServiceSettings
from ..retrieval_monitoring import RetrievalMonitoringError, read_telemetry


def _existing_telemetry_files(path: Path) -> list[Path]:
    return [candidate for candidate in path.parent.glob(f"{path.name}*") if candidate.is_file()]


def run_telemetry_smoke(settings: ServiceSettings) -> dict[str, Any]:
    """Write health, readiness, two cached retrievals, and one safe error event."""
    if not settings.telemetry_enabled:
        raise RetrievalMonitoringError("telemetry smoke requires telemetry_enabled")
    if _existing_telemetry_files(settings.telemetry_path):
        raise RetrievalMonitoringError("telemetry smoke requires an unused telemetry output path")
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        if ready.json().get("status") != "ready":
            raise RetrievalMonitoringError("persisted retrieval artifacts are not ready")
        artifacts = app.state.runtime.artifacts
        caption_id = sorted(artifacts.cache.caption_embeddings)[0]
        image_id = sorted(artifacts.cache.image_embeddings)[0]
        images = client.post(
            "/retrieve/images",
            json={
                "caption_id": caption_id,
                "top_k": min(10, artifacts.text_to_image.metadata.candidate_count),
            },
        )
        captions = client.post(
            "/retrieve/captions",
            json={
                "image_id": image_id,
                "top_k": min(10, artifacts.image_to_text.metadata.candidate_count),
            },
        )
        invalid = client.post(
            "/retrieve/images", json={"caption_id": "telemetry-missing-id", "top_k": 1}
        )
        service_metrics = client.get("/metrics").json()
    if (
        health.status_code != 200
        or images.status_code != 200
        or captions.status_code != 200
        or invalid.status_code != 404
    ):
        raise RetrievalMonitoringError("telemetry smoke request contract failed")
    read_result = read_telemetry(settings.telemetry_path)
    return {
        "backend": settings.backend,
        "event_count": len(read_result.events),
        "health_status": health.json()["status"],
        "ready_status": ready.json()["status"],
        "service_metrics": service_metrics,
    }
