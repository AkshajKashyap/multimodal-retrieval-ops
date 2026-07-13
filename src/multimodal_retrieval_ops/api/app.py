"""FastAPI application factory for persisted cached-ID retrieval."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..faiss_flat import search_cached_embedding
from ..faiss_hnsw import search_hnsw_embedding
from .artifacts import ServiceArtifactError, ServiceArtifacts, load_service_artifacts
from .metrics import ServiceMetrics
from .schemas import (
    CaptionResult,
    CaptionRetrievalRequest,
    CaptionRetrievalResponse,
    HealthResponse,
    ImageResult,
    ImageRetrievalRequest,
    ImageRetrievalResponse,
    IndexInfoResponse,
    MetricsResponse,
    ReadyResponse,
)
from .settings import ServiceSettings


@dataclass
class ServiceRuntime:
    artifacts: ServiceArtifacts | None = None
    readiness_state: str = "not_loaded"
    readiness_reasons: tuple[str, ...] = ()


def _require_ready(app: FastAPI) -> ServiceArtifacts:
    runtime: ServiceRuntime = app.state.runtime
    if runtime.artifacts is None:
        raise HTTPException(status_code=503, detail="retrieval artifacts are not ready")
    return runtime.artifacts


def _validate_top_k(
    top_k: int, maximum_top_k: int, candidate_count: int, metrics: ServiceMetrics
) -> None:
    if top_k <= 0:
        metrics.record_invalid_top_k()
        raise HTTPException(status_code=422, detail="top_k must be positive")
    if top_k > maximum_top_k:
        metrics.record_invalid_top_k()
        raise HTTPException(
            status_code=422, detail=f"top_k must not exceed configured maximum {maximum_top_k}"
        )
    if top_k > candidate_count:
        metrics.record_invalid_top_k()
        raise HTTPException(
            status_code=422, detail=f"top_k must not exceed candidate count {candidate_count}"
        )


def _search(
    artifacts: ServiceArtifacts,
    query_id: str,
    query_embeddings: dict[str, list[float]],
    index: Any,
    top_k: int,
) -> list[dict[str, float | str]]:
    if artifacts.backend == "flat":
        return search_cached_embedding(query_id, query_embeddings, index, top_k)
    return search_hnsw_embedding(
        query_id,
        query_embeddings,
        index,
        top_k,
        artifacts.ef_search or 64,
    )


def create_app(settings: ServiceSettings) -> FastAPI:
    """Create an app without loading artifacts until its lifespan starts."""
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime: ServiceRuntime = app.state.runtime
        try:
            runtime.artifacts = load_service_artifacts(settings)
            runtime.readiness_state = "ready"
            runtime.readiness_reasons = ()
        except ServiceArtifactError as error:
            runtime.readiness_state = error.state
            runtime.readiness_reasons = (error.reason,)
        except Exception:
            runtime.readiness_state = "execution_failed"
            runtime.readiness_reasons = ("unexpected artifact-loading failure",)
        yield

    app = FastAPI(title="Multimodal Retrieval Ops", version="1", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = ServiceRuntime()
    app.state.metrics = metrics

    @app.middleware("http")
    async def count_requests(request: Request, call_next: Any):
        try:
            response = await call_next(request)
        except Exception:
            metrics.record_request(is_error=True)
            raise
        metrics.record_request(is_error=response.status_code >= 400)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        body = error.body
        if isinstance(body, dict) and "top_k" in body:
            metrics.record_invalid_top_k()
        return JSONResponse(status_code=422, content={"detail": error.errors()})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="alive")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse:
        runtime: ServiceRuntime = app.state.runtime
        ready_state = runtime.artifacts is not None
        return ReadyResponse(
            status="ready" if ready_state else "unready",
            backend=settings.backend,
            artifact_validation="passed" if ready_state else runtime.readiness_state,
            reasons=list(runtime.readiness_reasons),
        )

    @app.get("/index-info", response_model=IndexInfoResponse)
    async def index_info() -> IndexInfoResponse:
        artifacts = _require_ready(app)
        metadata = artifacts.text_to_image.metadata
        return IndexInfoResponse(
            backend=artifacts.backend,
            faiss_version=metadata.faiss_version,
            index_type=metadata.index_type,
            model_name=metadata.model_name,
            model_revision=metadata.model_revision,
            embedding_dimension=metadata.embedding_dimension,
            image_candidate_count=artifacts.text_to_image.metadata.candidate_count,
            caption_candidate_count=artifacts.image_to_text.metadata.candidate_count,
            dataset_fingerprint=metadata.dataset_fingerprint,
            split=metadata.split,
            ef_search=artifacts.ef_search,
        )

    @app.post("/retrieve/images", response_model=ImageRetrievalResponse)
    async def retrieve_images(request: ImageRetrievalRequest) -> ImageRetrievalResponse:
        artifacts = _require_ready(app)
        _validate_top_k(
            request.top_k,
            settings.maximum_top_k,
            artifacts.text_to_image.metadata.candidate_count,
            metrics,
        )
        if request.caption_id not in artifacts.cache.caption_embeddings:
            metrics.record_unknown_query()
            raise HTTPException(status_code=404, detail="unknown caption_id")
        started = time.perf_counter()
        found = _search(
            artifacts,
            request.caption_id,
            artifacts.cache.caption_embeddings,
            artifacts.text_to_image,
            request.top_k,
        )
        elapsed = time.perf_counter() - started
        metrics.record_retrieval("caption_to_image", artifacts.backend, elapsed)
        target = artifacts.caption_targets.get(request.caption_id)
        return ImageRetrievalResponse(
            query_caption_id=request.caption_id,
            backend=artifacts.backend,
            results=[
                ImageResult(
                    image_id=str(item["candidate_id"]),
                    score=float(item["score"]),
                    rank=rank,
                    relevant_target=str(item["candidate_id"]) == target if target else None,
                )
                for rank, item in enumerate(found, 1)
            ],
        )

    @app.post("/retrieve/captions", response_model=CaptionRetrievalResponse)
    async def retrieve_captions(request: CaptionRetrievalRequest) -> CaptionRetrievalResponse:
        artifacts = _require_ready(app)
        _validate_top_k(
            request.top_k,
            settings.maximum_top_k,
            artifacts.image_to_text.metadata.candidate_count,
            metrics,
        )
        if request.image_id not in artifacts.cache.image_embeddings:
            metrics.record_unknown_query()
            raise HTTPException(status_code=404, detail="unknown image_id")
        started = time.perf_counter()
        found = _search(
            artifacts,
            request.image_id,
            artifacts.cache.image_embeddings,
            artifacts.image_to_text,
            request.top_k,
        )
        elapsed = time.perf_counter() - started
        metrics.record_retrieval("image_to_caption", artifacts.backend, elapsed)
        results = []
        for rank, item in enumerate(found, 1):
            caption_id = str(item["candidate_id"])
            target_image_id = artifacts.caption_targets[caption_id]
            results.append(
                CaptionResult(
                    caption_id=caption_id,
                    target_image_id=target_image_id,
                    caption_text=artifacts.caption_text.get(caption_id),
                    score=float(item["score"]),
                    rank=rank,
                    relevant_target=target_image_id == request.image_id,
                )
            )
        return CaptionRetrievalResponse(
            query_image_id=request.image_id,
            backend=artifacts.backend,
            results=results,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def service_metrics() -> MetricsResponse:
        return MetricsResponse(**metrics.snapshot())

    return app
