"""Pydantic contracts exposed by the local retrieval API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    backend: str
    artifact_validation: str
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
