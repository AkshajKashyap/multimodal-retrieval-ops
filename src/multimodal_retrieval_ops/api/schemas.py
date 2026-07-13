"""Pydantic contracts exposed by the local retrieval API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    backend: str
    artifact_validation: str
    retrieval_artifacts_ready: bool
    text_encoder_enabled: bool
    text_encoder_ready: bool
    text_encoder_state: str
    reasons: list[str]


class IndexInfoResponse(BaseModel):
    backend: str
    faiss_version: str
    index_type: str
    model_name: str
    model_revision: str
    embedding_dimension: int
    image_candidate_count: int
    caption_candidate_count: int
    dataset_fingerprint: str
    split: str
    ef_search: int | None
    text_inference_enabled: bool
    text_model_name: str
    text_model_revision: str
    text_encoder_ready: bool
    text_embedding_dimension: int | None
    local_files_only: bool


class ImageRetrievalRequest(BaseModel):
    caption_id: str = Field(min_length=1)
    top_k: int


class CaptionRetrievalRequest(BaseModel):
    image_id: str = Field(min_length=1)
    top_k: int


class ImageResult(BaseModel):
    image_id: str
    score: float
    rank: int
    relevant_target: bool | None


class CaptionResult(BaseModel):
    caption_id: str
    target_image_id: str
    caption_text: str | None
    score: float
    rank: int
    relevant_target: bool


class ImageRetrievalResponse(BaseModel):
    query_caption_id: str
    backend: str
    results: list[ImageResult]


class CaptionRetrievalResponse(BaseModel):
    query_image_id: str
    backend: str
    results: list[CaptionResult]


class TextSearchRequest(BaseModel):
    query: str
    top_k: int


class TextImageResult(BaseModel):
    image_id: str
    score: float
    rank: int
    split: str
    image_path: str
    captions: list[str]


class TextSearchResponse(BaseModel):
    query: str
    backend: str
    model_name: str
    embedding_dimension: int
    cached_query: bool
    results: list[TextImageResult]


class MetricsResponse(BaseModel):
    total_requests: int
    retrieval_requests_by_direction: dict[str, int]
    requests_by_backend: dict[str, int]
    error_count: int
    unknown_query_id_count: int
    invalid_top_k_count: int
    latency_observation_count: int
    mean_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    arbitrary_text_request_count: int
    text_encoder_invocation_count: int
    text_query_cache_hits: int
    text_query_cache_misses: int
    text_inference_errors: int
    text_inference_latency_count: int
    text_inference_latency_mean_seconds: float
    text_inference_latency_p50_seconds: float
    text_inference_latency_p95_seconds: float
