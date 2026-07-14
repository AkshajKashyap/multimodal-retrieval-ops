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
from .image_inference import (
    ImageEmbeddingCache,
    ImageEncoder,
    ImageValidationError,
    create_default_image_encoder,
    decode_and_validate_image,
    image_cache_identity,
    parse_multipart_image,
)
from .schemas import (
    CaptionResult,
    CaptionRetrievalRequest,
    CaptionRetrievalResponse,
    HealthResponse,
    ImageResult,
    ImageRetrievalRequest,
    ImageRetrievalResponse,
    ImageSearchResponse,
    IndexInfoResponse,
    MetricsResponse,
    ReadyResponse,
    LiveCaptionResult,
    TextImageResult,
    TextSearchRequest,
    TextSearchResponse,
)
from .settings import ServiceSettings
from .telemetry import (
    JsonlTelemetrySink,
    cached_relevance_metrics,
    make_retrieval_event,
    model_identity_fingerprint,
    safe_sha256,
    score_summary,
    unlabeled_relevance_metrics,
)
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
    image_encoder: ImageEncoder | None = None
    image_encoder_state: str = "disabled"
    image_encoder_reasons: tuple[str, ...] = ()
    image_query_cache: ImageEmbeddingCache | None = None


_TELEMETRY_ENDPOINTS = {
    "/health": ("health", "operational"),
    "/ready": ("readiness", "operational"),
    "/retrieve/images": ("cached_caption_to_image", "caption_to_image"),
    "/retrieve/captions": ("cached_image_to_caption", "image_to_caption"),
    "/search/text": ("arbitrary_text_to_image", "text_to_image"),
    "/search/image": ("arbitrary_image_to_caption", "image_to_text"),
}


def _set_telemetry(request: Request, **values: Any) -> None:
    context = getattr(request.state, "telemetry", {})
    context.update(values)
    request.state.telemetry = context


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


def _require_image_ready(
    app: FastAPI,
) -> tuple[ServiceArtifacts, ImageEncoder, ImageEmbeddingCache]:
    artifacts = _require_ready(app)
    runtime: ServiceRuntime = app.state.runtime
    if runtime.image_encoder is None or runtime.image_query_cache is None:
        raise HTTPException(status_code=503, detail="arbitrary image inference is unavailable")
    return artifacts, runtime.image_encoder, runtime.image_query_cache


def _validate_top_k(
    top_k: int,
    maximum_top_k: int,
    candidate_count: int,
    metrics: ServiceMetrics,
    telemetry_request: Request | None = None,
) -> None:
    if top_k <= 0:
        metrics.record_invalid_top_k()
        if telemetry_request is not None:
            _set_telemetry(telemetry_request, error_category="invalid_top_k")
        raise HTTPException(status_code=422, detail="top_k must be positive")
    if top_k > maximum_top_k:
        metrics.record_invalid_top_k()
        if telemetry_request is not None:
            _set_telemetry(telemetry_request, error_category="invalid_top_k")
        raise HTTPException(
            status_code=422, detail=f"top_k must not exceed configured maximum {maximum_top_k}"
        )
    if top_k > candidate_count:
        metrics.record_invalid_top_k()
        if telemetry_request is not None:
            _set_telemetry(telemetry_request, error_category="invalid_top_k")
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
    artifact: Any, vector: list[float], top_k: int
) -> list[dict[str, float | str]]:
    query = _as_normalized_matrix([vector], artifact.metadata.embedding_dimension)
    scores, indices = artifact.index.search(query, top_k)
    return [
        {
            "candidate_id": artifact.metadata.candidate_ids[int(index)],
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


def _validate_image_encoder_compatibility(
    settings: ServiceSettings, artifacts: ServiceArtifacts, encoder: ImageEncoder
) -> None:
    cache_metadata = artifacts.cache.metadata
    configured_revision = settings.image_model_revision or "default"
    if settings.image_model_name != cache_metadata.model_name:
        raise ServiceArtifactError(
            "artifact_incompatible", "image model name does not match the caption index"
        )
    if encoder.model_name != settings.image_model_name:
        raise ServiceArtifactError(
            "artifact_incompatible", "loaded image encoder model does not match configuration"
        )
    if configured_revision != cache_metadata.model_revision:
        raise ServiceArtifactError(
            "artifact_incompatible", "image model revision does not match the caption index"
        )
    if (encoder.model_revision or "default") != configured_revision:
        raise ServiceArtifactError(
            "artifact_incompatible", "loaded image encoder revision does not match configuration"
        )
    if encoder.backend_name != cache_metadata.backend_name:
        raise ServiceArtifactError(
            "artifact_incompatible", "image backend does not match cached caption embeddings"
        )
    if encoder.backend_version != cache_metadata.backend_version:
        raise ServiceArtifactError(
            "artifact_incompatible", "image backend version does not match caption embeddings"
        )
    if encoder.dimension != artifacts.image_to_text.metadata.embedding_dimension:
        raise ServiceArtifactError(
            "artifact_incompatible", "vision projection dimension does not match caption index"
        )


def create_app(
    settings: ServiceSettings,
    text_encoder_factory: Callable[[ServiceSettings], TextEncoder] | None = None,
    image_encoder_factory: (
        Callable[[ServiceSettings, TextEncoder | None], ImageEncoder] | None
    ) = None,
) -> FastAPI:
    """Create an app without loading artifacts until its lifespan starts."""
    metrics = ServiceMetrics()
    metrics.configure_telemetry(settings.telemetry_enabled)
    telemetry_sink = (
        JsonlTelemetrySink(
            settings.telemetry_path,
            settings.telemetry_max_bytes,
            settings.telemetry_backup_count,
            flush_each_event=settings.telemetry_flush_each_event,
            on_event=metrics.record_telemetry_event,
            on_failure=metrics.record_telemetry_failure,
            on_rotation=metrics.record_telemetry_rotation,
        )
        if settings.telemetry_enabled
        else None
    )

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
        if settings.enable_image_inference and runtime.artifacts is not None:
            try:
                image_factory = image_encoder_factory or create_default_image_encoder
                image_encoder = image_factory(settings, runtime.text_encoder)
                if image_encoder is not runtime.text_encoder:
                    image_encoder.ensure_loaded()
                _validate_image_encoder_compatibility(
                    settings, runtime.artifacts, image_encoder
                )
                runtime.image_encoder = image_encoder
                runtime.image_query_cache = ImageEmbeddingCache(
                    settings.image_query_cache_size
                )
                runtime.image_encoder_state = "ready"
                runtime.image_encoder_reasons = ()
            except ServiceArtifactError as error:
                runtime.image_encoder_state = error.state
                runtime.image_encoder_reasons = (error.reason,)
            except ClipModelUnavailableError:
                runtime.image_encoder_state = "model_unavailable"
                runtime.image_encoder_reasons = (
                    "configured CLIP model weights are not available locally",
                )
            except ClipExecutionError:
                runtime.image_encoder_state = "execution_failed"
                runtime.image_encoder_reasons = ("CLIP image encoder initialization failed",)
            except ClipBackendError:
                runtime.image_encoder_state = "dependency_unavailable"
                runtime.image_encoder_reasons = (
                    "optional CLIP image dependencies are unavailable",
                )
            except Exception:
                runtime.image_encoder_state = "execution_failed"
                runtime.image_encoder_reasons = ("image encoder initialization failed",)
        yield

    app = FastAPI(title="Multimodal Retrieval Ops", version="1", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = ServiceRuntime()
    app.state.metrics = metrics

    @app.middleware("http")
    async def record_telemetry(request: Request, call_next: Any):
        monitored = _TELEMETRY_ENDPOINTS.get(request.url.path)
        if telemetry_sink is None or monitored is None:
            return await call_next(request)
        endpoint, direction = monitored
        request.state.telemetry = {
            "endpoint": endpoint,
            "retrieval_direction": direction,
            **unlabeled_relevance_metrics(),
        }
        started = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            context = dict(request.state.telemetry)
            status_code = response.status_code if response is not None else 500
            context.setdefault("http_status_code", status_code)
            context.setdefault("request_status", "success" if status_code < 400 else "failed")
            if status_code >= 400:
                context.setdefault(
                    "error_category",
                    {
                        404: "unknown_cached_id",
                        422: "invalid_request",
                        503: "service_unavailable",
                    }.get(status_code, "internal_retrieval_failure"),
                )
            runtime: ServiceRuntime = app.state.runtime
            artifacts = runtime.artifacts
            if artifacts is not None:
                metadata = artifacts.text_to_image.metadata
                cache_metadata = artifacts.cache.metadata
                context.setdefault("backend", artifacts.backend)
                context.setdefault("embedding_dimension", metadata.embedding_dimension)
                context.setdefault(
                    "model_identity_fingerprint",
                    model_identity_fingerprint(
                        cache_metadata.backend_name,
                        cache_metadata.model_name,
                        cache_metadata.model_revision,
                        cache_metadata.embedding_dimension,
                    ),
                )
                context.setdefault("artifact_fingerprint", metadata.source_cache_fingerprint)
            else:
                context.setdefault("backend", settings.backend)
            context["latency_ms"] = (time.perf_counter() - started) * 1000.0
            try:
                telemetry_sink.write(make_retrieval_event(**context))
            except Exception:
                metrics.record_telemetry_failure()

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
        _set_telemetry(request, error_category="invalid_request")
        if request.url.path == "/search/text":
            metrics.record_text_request()
            metrics.record_text_error()
            if isinstance(body, dict) and isinstance(body.get("query"), str):
                normalized = " ".join(body["query"].split()).casefold()
                _set_telemetry(request, safe_query_hash=safe_sha256(normalized))
        elif request.url.path == "/retrieve/images" and isinstance(body, dict):
            identifier = body.get("caption_id")
            if isinstance(identifier, str):
                _set_telemetry(request, safe_query_hash=safe_sha256(identifier))
        elif request.url.path == "/retrieve/captions" and isinstance(body, dict):
            identifier = body.get("image_id")
            if isinstance(identifier, str):
                _set_telemetry(request, safe_query_hash=safe_sha256(identifier))
        if isinstance(body, dict) and "top_k" in body:
            metrics.record_invalid_top_k()
            _set_telemetry(request, error_category="invalid_top_k")
        return JSONResponse(status_code=422, content={"detail": error.errors()})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(status="alive")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready(request: Request) -> ReadyResponse:
        runtime: ServiceRuntime = app.state.runtime
        artifacts_ready = runtime.artifacts is not None
        text_ready = runtime.text_encoder is not None
        image_ready = runtime.image_encoder is not None
        ready_state = artifacts_ready and (
            not settings.enable_text_inference or text_ready
        ) and (not settings.enable_image_inference or image_ready)
        if not ready_state:
            _set_telemetry(
                request,
                request_status="failed",
                error_category="readiness_failure",
            )
        return ReadyResponse(
            status="ready" if ready_state else "unready",
            backend=settings.backend,
            artifact_validation="passed" if artifacts_ready else runtime.readiness_state,
            retrieval_artifacts_ready=artifacts_ready,
            text_encoder_enabled=settings.enable_text_inference,
            text_encoder_ready=text_ready,
            text_encoder_state=runtime.text_encoder_state,
            image_encoder_enabled=settings.enable_image_inference,
            image_encoder_ready=image_ready,
            image_encoder_state=runtime.image_encoder_state,
            reasons=list(
                runtime.readiness_reasons
                + runtime.text_encoder_reasons
                + runtime.image_encoder_reasons
            ),
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
            image_inference_enabled=settings.enable_image_inference,
            image_encoder_ready=app.state.runtime.image_encoder is not None,
            image_model_name=settings.image_model_name,
            image_model_revision=settings.image_model_revision or "default",
            vision_embedding_dimension=(
                app.state.runtime.image_encoder.dimension
                if app.state.runtime.image_encoder is not None
                else None
            ),
            accepted_image_formats=list(settings.allowed_image_formats),
            maximum_upload_bytes=settings.maximum_upload_bytes,
            maximum_pixel_count=settings.maximum_pixel_count,
        )

    @app.post("/retrieve/images", response_model=ImageRetrievalResponse)
    async def retrieve_images(
        request: ImageRetrievalRequest, http_request: Request
    ) -> ImageRetrievalResponse:
        _set_telemetry(
            http_request,
            top_k=request.top_k if request.top_k > 0 else None,
            safe_query_hash=safe_sha256(request.caption_id),
        )
        artifacts = _require_ready(app)
        _validate_top_k(
            request.top_k,
            settings.maximum_top_k,
            artifacts.text_to_image.metadata.candidate_count,
            metrics,
            http_request,
        )
        if request.caption_id not in artifacts.cache.caption_embeddings:
            metrics.record_unknown_query()
            _set_telemetry(http_request, error_category="unknown_cached_id")
            raise HTTPException(status_code=404, detail="unknown caption_id")
        started = time.perf_counter()
        try:
            found = _search(
                artifacts,
                request.caption_id,
                artifacts.cache.caption_embeddings,
                artifacts.text_to_image,
                request.top_k,
            )
        except Exception:
            _set_telemetry(http_request, error_category="internal_retrieval_failure")
            raise
        elapsed = time.perf_counter() - started
        metrics.record_retrieval("caption_to_image", artifacts.backend, elapsed)
        target = artifacts.caption_targets.get(request.caption_id)
        relevant = [str(item["candidate_id"]) == target for item in found]
        _set_telemetry(
            http_request,
            result_count=len(found),
            candidate_count=artifacts.text_to_image.metadata.candidate_count,
            **score_summary([float(item["score"]) for item in found]),
            **cached_relevance_metrics(relevant, request.top_k),
        )
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
    async def retrieve_captions(
        request: CaptionRetrievalRequest, http_request: Request
    ) -> CaptionRetrievalResponse:
        _set_telemetry(
            http_request,
            top_k=request.top_k if request.top_k > 0 else None,
            safe_query_hash=safe_sha256(request.image_id),
        )
        artifacts = _require_ready(app)
        _validate_top_k(
            request.top_k,
            settings.maximum_top_k,
            artifacts.image_to_text.metadata.candidate_count,
            metrics,
            http_request,
        )
        if request.image_id not in artifacts.cache.image_embeddings:
            metrics.record_unknown_query()
            _set_telemetry(http_request, error_category="unknown_cached_id")
            raise HTTPException(status_code=404, detail="unknown image_id")
        started = time.perf_counter()
        try:
            found = _search(
                artifacts,
                request.image_id,
                artifacts.cache.image_embeddings,
                artifacts.image_to_text,
                request.top_k,
            )
        except Exception:
            _set_telemetry(http_request, error_category="internal_retrieval_failure")
            raise
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
        _set_telemetry(
            http_request,
            result_count=len(found),
            candidate_count=artifacts.image_to_text.metadata.candidate_count,
            **score_summary([float(item["score"]) for item in found]),
            **cached_relevance_metrics(
                [
                    artifacts.caption_targets[str(item["candidate_id"])] == request.image_id
                    for item in found
                ],
                request.top_k,
            ),
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
    async def search_text(
        request: TextSearchRequest, http_request: Request
    ) -> TextSearchResponse:
        metrics.record_text_request()
        normalized_identity = " ".join(request.query.split()).casefold()
        _set_telemetry(
            http_request,
            top_k=request.top_k if request.top_k > 0 else None,
            safe_query_hash=safe_sha256(normalized_identity),
        )
        try:
            artifacts, encoder, query_cache = _require_text_ready(app)
            _validate_top_k(
                request.top_k,
                settings.maximum_top_k,
                artifacts.text_to_image.metadata.candidate_count,
                metrics,
                http_request,
            )
        except HTTPException:
            metrics.record_text_error()
            _set_telemetry(http_request, error_category="optional_encoder_unavailable")
            raise
        display_query, cache_key = normalized_query(request.query)
        if not display_query:
            metrics.record_text_error()
            _set_telemetry(http_request, error_category="invalid_request")
            raise HTTPException(status_code=422, detail="query must be non-empty")
        if len(display_query) > settings.maximum_text_length:
            metrics.record_text_error()
            _set_telemetry(http_request, error_category="invalid_request")
            raise HTTPException(
                status_code=422,
                detail=f"query must not exceed {settings.maximum_text_length} characters",
            )
        started = time.perf_counter()
        vector = query_cache.get(cache_key)
        cached_query = vector is not None
        metrics.record_text_cache(cached_query)
        _set_telemetry(http_request, cache_hit=cached_query)
        try:
            if vector is None:
                metrics.record_text_encoder_invocation()
                vector = normalize_vector(
                    encoder.encode_text(display_query),
                    artifacts.text_to_image.metadata.embedding_dimension,
                )
                query_cache.put(cache_key, vector)
            found = _search_vector(artifacts.text_to_image, vector, request.top_k)
        except Exception:
            metrics.record_text_error()
            _set_telemetry(http_request, error_category="internal_retrieval_failure")
            raise HTTPException(
                status_code=503, detail="arbitrary text inference execution failed"
            ) from None
        metrics.record_text_latency(time.perf_counter() - started, artifacts.backend)
        _set_telemetry(
            http_request,
            result_count=len(found),
            candidate_count=artifacts.text_to_image.metadata.candidate_count,
            **score_summary([float(item["score"]) for item in found]),
        )
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

    @app.post("/search/image", response_model=ImageSearchResponse)
    async def search_image(request: Request) -> ImageSearchResponse:
        metrics.record_image_request()
        try:
            artifacts, encoder, image_cache = _require_image_ready(app)
        except HTTPException:
            metrics.record_image_inference_error()
            _set_telemetry(request, error_category="optional_encoder_unavailable")
            raise
        try:
            upload = await parse_multipart_image(request, settings.maximum_upload_bytes)
            metrics.record_uploaded_bytes(len(upload.image_bytes))
            _set_telemetry(
                request,
                top_k=upload.top_k if upload.top_k > 0 else None,
                safe_query_hash=safe_sha256(upload.image_bytes),
            )
            _validate_top_k(
                upload.top_k,
                settings.maximum_top_k,
                artifacts.image_to_text.metadata.candidate_count,
                metrics,
                request,
            )
            image, _ = decode_and_validate_image(upload, settings)
        except ImageValidationError as error:
            metrics.record_image_validation_error()
            _set_telemetry(request, error_category="invalid_image_upload")
            raise HTTPException(status_code=422, detail=str(error)) from None
        except HTTPException:
            metrics.record_image_validation_error()
            _set_telemetry(request, error_category="invalid_top_k")
            raise
        identifier, cache_key = image_cache_identity(upload.image_bytes, encoder)
        started = time.perf_counter()
        vector = image_cache.get(cache_key)
        cached_query = vector is not None
        metrics.record_image_cache(cached_query)
        _set_telemetry(request, cache_hit=cached_query)
        try:
            if vector is None:
                metrics.record_image_encoder_invocation()
                vector = normalize_vector(
                    encoder.encode_image_object(image),
                    artifacts.image_to_text.metadata.embedding_dimension,
                )
                image_cache.put(cache_key, vector)
            found = _search_vector(artifacts.image_to_text, vector, upload.top_k)
        except Exception:
            metrics.record_image_inference_error()
            _set_telemetry(request, error_category="internal_retrieval_failure")
            raise HTTPException(
                status_code=503, detail="arbitrary image inference execution failed"
            ) from None
        metrics.record_image_latency(time.perf_counter() - started, artifacts.backend)
        _set_telemetry(
            request,
            result_count=len(found),
            candidate_count=artifacts.image_to_text.metadata.candidate_count,
            **score_summary([float(item["score"]) for item in found]),
        )
        return ImageSearchResponse(
            backend=artifacts.backend,
            model_name=encoder.model_name,
            embedding_dimension=len(vector),
            image_identifier=identifier,
            cached_query=cached_query,
            results=[
                LiveCaptionResult(
                    caption_id=str(item["candidate_id"]),
                    target_image_id=artifacts.caption_targets[str(item["candidate_id"])],
                    caption_text=artifacts.caption_text[str(item["candidate_id"])],
                    score=float(item["score"]),
                    rank=rank,
                    split=artifacts.cache.metadata.split,
                )
                for rank, item in enumerate(found, 1)
            ],
        )

    return app
