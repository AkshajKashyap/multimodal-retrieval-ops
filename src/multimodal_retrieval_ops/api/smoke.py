"""Bounded in-process retrieval-service smoke workflow."""

from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient

from .app import create_app
from .image_inference import CONTENT_TYPES, ImageEncoder
from .reporting import ServiceSmokeResult
from .settings import ServiceSettings
from .text_inference import TextEncoder


def run_service_smoke(
    settings: ServiceSettings,
    text_encoder_factory: Callable[[ServiceSettings], TextEncoder] | None = None,
    image_encoder_factory: (
        Callable[[ServiceSettings, TextEncoder | None], ImageEncoder] | None
    ) = None,
) -> ServiceSmokeResult:
    """Exercise both directions once without binding a network socket."""
    app = create_app(
        settings,
        text_encoder_factory=text_encoder_factory,
        image_encoder_factory=image_encoder_factory,
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        ready = client.get("/ready").json()
        if ready["status"] != "ready":
            if not ready.get("retrieval_artifacts_ready"):
                state = ready.get("artifact_validation", "execution_failed")
            elif settings.enable_image_inference and not ready.get("image_encoder_ready"):
                state = ready.get("image_encoder_state", "execution_failed")
            else:
                state = ready.get("text_encoder_state", "execution_failed")
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
        if settings.enable_image_inference:
            if not settings.smoke_image_path.is_file():
                return ServiceSmokeResult(
                    run_state="invalid_input",
                    backend=settings.backend,
                    artifact_readiness="ready",
                    index_info=info,
                    health_response=health,
                    ready_response=ready,
                    image_retrieval_response=None,
                    caption_retrieval_response=None,
                    observability=client.get("/metrics").json(),
                    detail="configured smoke image is unavailable",
                )
            image_bytes = settings.smoke_image_path.read_bytes()
            suffix = settings.smoke_image_path.suffix.lower()
            content_type = {
                ".jpg": CONTENT_TYPES["JPEG"],
                ".jpeg": CONTENT_TYPES["JPEG"],
                ".png": CONTENT_TYPES["PNG"],
                ".webp": CONTENT_TYPES["WEBP"],
            }.get(suffix, "application/octet-stream")
            top_k = min(3, info["caption_candidate_count"])
            files = {"image": ("smoke-image" + suffix, image_bytes, content_type)}
            data = {"top_k": str(top_k)}
            first_response = client.post("/search/image", files=files, data=data)
            second_response = client.post("/search/image", files=files, data=data)
            observability = client.get("/metrics").json()
            responses = [first_response.json(), second_response.json()]
            if first_response.status_code != 200 or second_response.status_code != 200:
                return ServiceSmokeResult(
                    run_state=(
                        "invalid_input"
                        if first_response.status_code == 422
                        else "execution_failed"
                    ),
                    backend=settings.backend,
                    artifact_readiness="ready",
                    index_info=info,
                    health_response=health,
                    ready_response=ready,
                    image_retrieval_response=None,
                    caption_retrieval_response=None,
                    observability=observability,
                    detail="arbitrary image smoke request failed",
                    image_search_responses=responses,
                )
            cache_verified = (
                first_response.json()["cached_query"] is False
                and second_response.json()["cached_query"] is True
                and observability["image_encoder_invocation_count"] == 1
                and observability["image_query_cache_misses"] == 1
                and observability["image_query_cache_hits"] == 1
            )
            if not cache_verified:
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
                    detail="arbitrary image cache behavior did not meet the smoke contract",
                    image_search_responses=responses,
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
                detail="in-process arbitrary-image smoke completed successfully",
                image_search_responses=responses,
            )
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


def run_live_image_query(
    settings: ServiceSettings,
    image_path: str,
    top_k: int,
    image_encoder_factory: (
        Callable[[ServiceSettings, TextEncoder | None], ImageEncoder] | None
    ) = None,
) -> dict[str, object]:
    """Run one validated local image through the in-process application."""
    path = Path(image_path)
    if not path.is_file():
        return {"status_code": 422, "response": {"detail": "image path does not exist"}}
    suffix = path.suffix.lower()
    content_type = {
        ".jpg": CONTENT_TYPES["JPEG"],
        ".jpeg": CONTENT_TYPES["JPEG"],
        ".png": CONTENT_TYPES["PNG"],
        ".webp": CONTENT_TYPES["WEBP"],
    }.get(suffix, "application/octet-stream")
    app = create_app(settings, image_encoder_factory=image_encoder_factory)
    with TestClient(app) as client:
        ready = client.get("/ready")
        if ready.json()["status"] != "ready":
            return {"status_code": 503, "response": ready.json()}
        response = client.post(
            "/search/image",
            files={"image": ("query-image" + suffix, path.read_bytes(), content_type)},
            data={"top_k": str(top_k)},
        )
        return {"status_code": response.status_code, "response": response.json()}
