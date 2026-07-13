"""Deterministic reports for arbitrary-image service smoke execution."""

from dataclasses import asdict
import json
from pathlib import Path

from .reporting import ServiceSmokeResult
from .settings import ServiceSettings

IMAGE_REPORT_STATES = frozenset(
    {
        "success",
        "not_run",
        "dependency_unavailable",
        "model_unavailable",
        "artifact_unavailable",
        "artifact_incompatible",
        "invalid_input",
        "execution_failed",
    }
)


def render_image_inference_report(
    result: ServiceSmokeResult, settings: ServiceSettings
) -> str:
    info = result.index_info or {}
    metrics = result.observability or {}
    responses = result.image_search_responses or []
    first = responses[0] if responses else {}
    second = responses[1] if len(responses) > 1 else {}
    first_ids = ", ".join(item["caption_id"] for item in first.get("results", [])) or "none"
    return "\n".join(
        [
            "# Image Inference Service Report",
            "",
            f"Run state: **{result.run_state}**",
            "",
            f"- Backend: `{result.backend}`",
            f"- Model: `{settings.image_model_name}`",
            f"- Model revision: `{settings.image_model_revision or 'default'}`",
            f"- Local files only: `{str(settings.local_files_only).lower()}`",
            f"- Retrieval artifacts: `{result.artifact_readiness}`",
            f"- Image encoder ready: `{str(result.ready_response.get('image_encoder_ready', False)).lower()}`",
            f"- Caption candidates: {info.get('caption_candidate_count', 'unavailable')}",
            f"- Embedding dimension: {info.get('vision_embedding_dimension', 'unavailable')}",
            f"- Accepted formats: {', '.join(settings.allowed_image_formats)}",
            f"- Maximum upload bytes: {settings.maximum_upload_bytes}",
            f"- Maximum decoded pixels: {settings.maximum_pixel_count}",
            f"- Detail: {result.detail}",
            "",
            "## Smoke result",
            "",
            f"- Safe image identifier: `{first.get('image_identifier', 'not run')}`",
            f"- First request cached: `{str(first.get('cached_query', False)).lower()}`",
            f"- Repeated request cached: `{str(second.get('cached_query', False)).lower()}`",
            f"- Ranked caption IDs: `{first_ids}`",
            "",
            "## Input validation and metrics",
            "",
            "- Validation: byte size, MIME type, decoded format, corruption, and pixel count",
            f"- Arbitrary image requests: {metrics.get('arbitrary_image_request_count', 0)}",
            f"- Vision encoder invocations: {metrics.get('image_encoder_invocation_count', 0)}",
            f"- Image-cache hits: {metrics.get('image_query_cache_hits', 0)}",
            f"- Image-cache misses: {metrics.get('image_query_cache_misses', 0)}",
            f"- Validation errors: {metrics.get('image_validation_errors', 0)}",
            f"- Inference errors: {metrics.get('image_inference_errors', 0)}",
            f"- Uploaded bytes observed: {metrics.get('uploaded_byte_total', 0)}",
            f"- Image latency observations: {metrics.get('image_inference_latency_count', 0)}",
            "",
            "## Limitations",
            "",
            "Uploads are bounded, decoded, and processed entirely in memory; they are never saved",
            "to disk. The Hugging Face implementation loads one full CLIPModel object and invokes",
            "only its vision tower for image requests. CPU startup and inference may be slow.",
            "Caches and metrics are process-local. No OCR, augmentation, training, fine-tuning,",
            "reranking, index rebuilding, or new dataset is included.",
            "",
        ]
    )


def write_image_inference_reports(
    result: ServiceSmokeResult,
    settings: ServiceSettings,
    report_path: Path,
    metrics_path: Path,
) -> None:
    if result.run_state not in IMAGE_REPORT_STATES:
        raise ValueError(f"unsupported image inference report state: {result.run_state}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_image_inference_report(result, settings), encoding="utf-8")
    payload = {
        "settings": {
            "backend": settings.backend,
            "local_files_only": settings.local_files_only,
            "image_model_name": settings.image_model_name,
            "image_model_revision": settings.image_model_revision or "default",
            "maximum_upload_bytes": settings.maximum_upload_bytes,
            "maximum_pixel_count": settings.maximum_pixel_count,
            "allowed_image_formats": list(settings.allowed_image_formats),
        },
        **asdict(result),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
