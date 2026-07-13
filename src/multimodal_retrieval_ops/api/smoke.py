"""Bounded in-process retrieval-service smoke workflow."""

from typing import Callable

from fastapi.testclient import TestClient

from .app import create_app
from .reporting import ServiceSmokeResult
from .settings import ServiceSettings
from .text_inference import TextEncoder


def run_service_smoke(
    settings: ServiceSettings,
    text_encoder_factory: Callable[[ServiceSettings], TextEncoder] | None = None,
) -> ServiceSmokeResult:
    """Exercise both directions once without binding a network socket."""
    app = create_app(settings, text_encoder_factory=text_encoder_factory)
    with TestClient(app) as client:
        health = client.get("/health").json()
        ready = client.get("/ready").json()
        if ready["status"] != "ready":
            state = (
                ready.get("text_encoder_state", "execution_failed")
                if ready.get("retrieval_artifacts_ready")
                else ready.get("artifact_validation", "execution_failed")
            )
            if state not in {
                "dependency_unavailable",
                "model_unavailable",
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
        if settings.enable_text_inference:
            payload = {
                "query": "a dog running outside",
                "top_k": min(3, info["image_candidate_count"]),
            }
            first_response = client.post("/search/text", json=payload)
            second_response = client.post("/search/text", json=payload)
            observability = client.get("/metrics").json()
            if first_response.status_code != 200 or second_response.status_code != 200:
                return ServiceSmokeResult(
                    run_state="execution_failed",
                    backend=settings.backend,
                    artifact_readiness="ready",
                    index_info=info,
                    health_response=health,
                    ready_response=ready,
                    image_retrieval_response=None,
                    caption_retrieval_response=None,
                    observability=observability,
                    detail="arbitrary text smoke request failed",
                    text_search_responses=[first_response.json(), second_response.json()],
                )
            return ServiceSmokeResult(
                run_state="success",
                backend=settings.backend,
                artifact_readiness="ready",
                index_info=info,
                health_response=health,
                ready_response=ready,
                image_retrieval_response=None,
                caption_retrieval_response=None,
                observability=observability,
                detail="in-process arbitrary-text smoke completed successfully",
                text_search_responses=[first_response.json(), second_response.json()],
            )
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


def run_live_text_query(
    settings: ServiceSettings,
    query: str,
    top_k: int,
    text_encoder_factory: Callable[[ServiceSettings], TextEncoder] | None = None,
) -> dict[str, object]:
    """Run one arbitrary text request through the in-process application."""
    app = create_app(settings, text_encoder_factory=text_encoder_factory)
    with TestClient(app) as client:
        ready = client.get("/ready")
        if ready.json()["status"] != "ready":
            return {"status_code": 503, "response": ready.json()}
        response = client.post("/search/text", json={"query": query, "top_k": top_k})
        return {"status_code": response.status_code, "response": response.json()}
