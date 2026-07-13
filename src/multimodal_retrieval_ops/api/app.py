"""FastAPI application factory for persisted cached-ID retrieval."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..clip_backend import (
    ClipBackendError,
    ClipExecutionError,
    ClipModelUnavailableError,
)
from ..faiss_flat import _as_normalized_matrix, search_cached_embedding
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
    TextImageResult,
    TextSearchRequest,
    TextSearchResponse,
)
from .settings import ServiceSettings
from .text_inference import (
    QueryEmbeddingCache,
    TextEncoder,
    create_default_text_encoder,
    normalize_vector,
    normalized_query,
)


@dataclass
class ServiceRuntime:
    artifacts: ServiceArtifacts | None = None
    readiness_state: str = "not_loaded"
    readiness_reasons: tuple[str, ...] = ()
    text_encoder: TextEncoder | None = None
    text_encoder_state: str = "disabled"
    text_encoder_reasons: tuple[str, ...] = ()
    query_cache: QueryEmbeddingCache | None = None


def _require_ready(app: FastAPI) -> ServiceArtifacts:
    runtime: ServiceRuntime = app.state.runtime
    if runtime.artifacts is None:
        raise HTTPException(status_code=503, detail="retrieval artifacts are not ready")
    return runtime.artifacts


def _require_text_ready(app: FastAPI) -> tuple[ServiceArtifacts, TextEncoder, QueryEmbeddingCache]:
    artifacts = _require_ready(app)
    runtime: ServiceRuntime = app.state.runtime
    if runtime.text_encoder is None or runtime.query_cache is None:
        raise HTTPException(status_code=503, detail="arbitrary text inference is unavailable")
    return artifacts, runtime.text_encoder, runtime.query_cache


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


def _search_vector(
    artifacts: ServiceArtifacts, vector: list[float], top_k: int
) -> list[dict[str, float | str]]:
    query = _as_normalized_matrix(
        [vector], artifacts.text_to_image.metadata.embedding_dimension
    )
    scores, indices = artifacts.text_to_image.index.search(query, top_k)
    return [
        {
            "candidate_id": artifacts.text_to_image.metadata.candidate_ids[int(index)],
            "score": float(score),
        }
        for score, index in zip(scores[0], indices[0], strict=True)
        if index >= 0
    ]


def _validate_encoder_compatibility(
    settings: ServiceSettings, artifacts: ServiceArtifacts, encoder: TextEncoder
) -> None:
    cache_metadata = artifacts.cache.metadata
    configured_revision = settings.text_model_revision or "default"
    if settings.text_model_name != cache_metadata.model_name:
        raise ServiceArtifactError(
            "artifact_incompatible", "text model name does not match the image index"
        )
    if encoder.model_name != settings.text_model_name:
        raise ServiceArtifactError(
            "artifact_incompatible", "loaded text encoder model does not match configuration"
        )
    if configured_revision != cache_metadata.model_revision:
        raise ServiceArtifactError(
            "artifact_incompatible", "text model revision does not match the image index"
        )
    if (encoder.model_revision or "default") != configured_revision:
        raise ServiceArtifactError(
            "artifact_incompatible", "loaded text encoder revision does not match configuration"
        )
    if encoder.backend_name != cache_metadata.backend_name:
        raise ServiceArtifactError(
            "artifact_incompatible", "text backend does not match cached image embeddings"
        )
    if encoder.backend_version != cache_metadata.backend_version:
        raise ServiceArtifactError(
            "artifact_incompatible", "text backend version does not match image embeddings"
        )
    if encoder.dimension != cache_metadata.embedding_dimension:
        raise ServiceArtifactError(
            "artifact_incompatible", "text projection dimension does not match the image index"
        )


def create_app(
    settings: ServiceSettings,
    text_encoder_factory: Callable[[ServiceSettings], TextEncoder] | None = None,
) -> FastAPI:
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
        if settings.enable_text_inference and runtime.artifacts is not None:
            try:
                factory = text_encoder_factory or create_default_text_encoder
                encoder = factory(settings)
                encoder.ensure_loaded()
                _validate_encoder_compatibility(settings, runtime.artifacts, encoder)
                runtime.text_encoder = encoder
                runtime.query_cache = QueryEmbeddingCache(settings.text_query_cache_size)
                runtime.text_encoder_state = "ready"
                runtime.text_encoder_reasons = ()
            except ServiceArtifactError as error:
                runtime.text_encoder_state = error.state
                runtime.text_encoder_reasons = (error.reason,)
            except ClipModelUnavailableError:
                runtime.text_encoder_state = "model_unavailable"
                runtime.text_encoder_reasons = (
                    "configured CLIP model weights are not available locally",
                )
            except ClipExecutionError:
                runtime.text_encoder_state = "execution_failed"
                runtime.text_encoder_reasons = ("CLIP text encoder initialization failed",)
            except ClipBackendError:
                runtime.text_encoder_state = "dependency_unavailable"
                runtime.text_encoder_reasons = (
                    "optional CLIP text dependencies are unavailable",
                )
            except Exception:
                runtime.text_encoder_state = "execution_failed"
                runtime.text_encoder_reasons = ("text encoder initialization failed",)
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
        if request.url.path == "/search/text":
            metrics.record_text_request()
            metrics.record_text_error()
        if isinstance(body, dict) and "top_k" in body:
            metrics.record_invalid_top_k()
        return JSONResponse(status_code=422, content={"detail": error.errors()})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="alive")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse:
        runtime: ServiceRuntime = app.state.runtime
        artifacts_ready = runtime.artifacts is not None
        text_ready = runtime.text_encoder is not None
        ready_state = artifacts_ready and (
            not settings.enable_text_inference or text_ready
        )
        return ReadyResponse(
            status="ready" if ready_state else "unready",
            backend=settings.backend,
            artifact_validation="passed" if artifacts_ready else runtime.readiness_state,
            retrieval_artifacts_ready=artifacts_ready,
            text_encoder_enabled=settings.enable_text_inference,
            text_encoder_ready=text_ready,
            text_encoder_state=runtime.text_encoder_state,
            reasons=list(runtime.readiness_reasons + runtime.text_encoder_reasons),
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
            text_inference_enabled=settings.enable_text_inference,
            text_model_name=settings.text_model_name,
            text_model_revision=settings.text_model_revision or "default",
            text_encoder_ready=app.state.runtime.text_encoder is not None,
            text_embedding_dimension=(
                app.state.runtime.text_encoder.dimension
                if app.state.runtime.text_encoder is not None
                else None
            ),
            local_files_only=settings.local_files_only,
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

    @app.post("/search/text", response_model=TextSearchResponse)
    async def search_text(request: TextSearchRequest) -> TextSearchResponse:
        metrics.record_text_request()
        try:
            artifacts, encoder, query_cache = _require_text_ready(app)
            _validate_top_k(
                request.top_k,
                settings.maximum_top_k,
                artifacts.text_to_image.metadata.candidate_count,
                metrics,
            )
        except HTTPException:
            metrics.record_text_error()
            raise
        display_query, cache_key = normalized_query(request.query)
        if not display_query:
            metrics.record_text_error()
            raise HTTPException(status_code=422, detail="query must be non-empty")
        if len(display_query) > settings.maximum_text_length:
            metrics.record_text_error()
            raise HTTPException(
                status_code=422,
                detail=f"query must not exceed {settings.maximum_text_length} characters",
            )
        started = time.perf_counter()
        vector = query_cache.get(cache_key)
        cached_query = vector is not None
        metrics.record_text_cache(cached_query)
        try:
            if vector is None:
                metrics.record_text_encoder_invocation()
                vector = normalize_vector(
                    encoder.encode_text(display_query),
                    artifacts.text_to_image.metadata.embedding_dimension,
                )
                query_cache.put(cache_key, vector)
            found = _search_vector(artifacts, vector, request.top_k)
        except Exception:
            metrics.record_text_error()
            raise HTTPException(
                status_code=503, detail="arbitrary text inference execution failed"
            ) from None
        metrics.record_text_latency(time.perf_counter() - started, artifacts.backend)
        return TextSearchResponse(
            query=display_query,
            backend=artifacts.backend,
            model_name=encoder.model_name,
            embedding_dimension=len(vector),
            cached_query=cached_query,
            results=[
                TextImageResult(
                    image_id=str(item["candidate_id"]),
                    score=float(item["score"]),
                    rank=rank,
                    split=artifacts.image_splits[str(item["candidate_id"])],
                    image_path=artifacts.safe_image_paths[str(item["candidate_id"])],
                    captions=artifacts.image_captions[str(item["candidate_id"])],
                )
                for rank, item in enumerate(found, 1)
            ],
        )

    return app
