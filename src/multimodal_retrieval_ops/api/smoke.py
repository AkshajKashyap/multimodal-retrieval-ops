"""Bounded in-process retrieval-service smoke workflow."""

from fastapi.testclient import TestClient

from .app import create_app
from .reporting import ServiceSmokeResult
from .settings import ServiceSettings


def run_service_smoke(settings: ServiceSettings) -> ServiceSmokeResult:
    """Exercise both directions once without binding a network socket."""
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health").json()
        ready = client.get("/ready").json()
        if ready["status"] != "ready":
            state = ready.get("artifact_validation", "execution_failed")
            if state not in {
                "dependency_unavailable",
                "artifact_unavailable",
                "artifact_incompatible",
            }:
                state = "execution_failed"
            return ServiceSmokeResult(
                run_state=state,
                backend=settings.backend,
                artifact_readiness="unready",
                index_info=None,
                health_response=health,
                ready_response=ready,
                image_retrieval_response=None,
                caption_retrieval_response=None,
                observability=client.get("/metrics").json(),
                detail="persisted artifacts did not pass readiness validation",
            )
        info = client.get("/index-info").json()
        artifacts = app.state.runtime.artifacts
        caption_id = sorted(artifacts.cache.caption_embeddings)[0]
        image_id = sorted(artifacts.cache.image_embeddings)[0]
        image_response = client.post(
            "/retrieve/images", json={"caption_id": caption_id, "top_k": 3}
        ).json()
        caption_response = client.post(
            "/retrieve/captions", json={"image_id": image_id, "top_k": 3}
        ).json()
        observability = client.get("/metrics").json()
    return ServiceSmokeResult(
        run_state="success",
        backend=settings.backend,
        artifact_readiness="ready",
        index_info=info,
        health_response=health,
        ready_response=ready,
        image_retrieval_response=image_response,
        caption_retrieval_response=caption_response,
        observability=observability,
        detail="in-process cached-ID smoke completed successfully",
    )
