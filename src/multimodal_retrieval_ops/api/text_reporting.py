"""Deterministic reports for real arbitrary-text service smoke execution."""

from dataclasses import asdict
import json
from pathlib import Path

from .reporting import ServiceSmokeResult
from .settings import ServiceSettings

TEXT_REPORT_STATES = frozenset(
    {
        "success",
        "not_run",
        "dependency_unavailable",
        "model_unavailable",
        "artifact_unavailable",
        "artifact_incompatible",
        "execution_failed",
    }
)


def render_text_inference_report(
    result: ServiceSmokeResult, settings: ServiceSettings
) -> str:
    info = result.index_info or {}
    metrics = result.observability or {}
    responses = result.text_search_responses or []
    first = responses[0] if responses else {}
    second = responses[1] if len(responses) > 1 else {}
    first_ids = ", ".join(item["image_id"] for item in first.get("results", [])) or "none"
    return "\n".join(
        [
            "# Text Inference Service Report",
            "",
            f"Run state: **{result.run_state}**",
            "",
            f"- Backend: `{result.backend}`",
            f"- Model: `{settings.text_model_name}`",
            f"- Model revision: `{settings.text_model_revision or 'default'}`",
            f"- Local files only: `{str(settings.local_files_only).lower()}`",
            f"- Retrieval artifacts: `{result.artifact_readiness}`",
            f"- Text encoder ready: `{str(result.ready_response.get('text_encoder_ready', False)).lower()}`",
            f"- Image candidates: {info.get('image_candidate_count', 'unavailable')}",
            f"- Embedding dimension: {info.get('text_embedding_dimension', 'unavailable')}",
            f"- Detail: {result.detail}",
            "",
            "## Smoke query",
            "",
            f"- Query: `{first.get('query', 'not run')}`",
            f"- First request cached: `{str(first.get('cached_query', False)).lower()}`",
            f"- Repeated request cached: `{str(second.get('cached_query', False)).lower()}`",
            f"- Ranked image IDs: `{first_ids}`",
            "",
            "## Process-local metrics",
            "",
            f"- Arbitrary text requests: {metrics.get('arbitrary_text_request_count', 0)}",
            f"- Text encoder invocations: {metrics.get('text_encoder_invocation_count', 0)}",
            f"- Query-cache hits: {metrics.get('text_query_cache_hits', 0)}",
            f"- Query-cache misses: {metrics.get('text_query_cache_misses', 0)}",
            f"- Text inference errors: {metrics.get('text_inference_errors', 0)}",
            f"- Text latency observations: {metrics.get('text_inference_latency_count', 0)}",
            "",
            "## Limitations",
            "",
            "The Hugging Face implementation loads the full CLIPModel object because text and",
            "vision projections share that model package, but this workflow invokes only the text",
            "tower. It never decodes or embeds images. Queries run on CPU by default and may be",
            "slow. The cache and metrics are bounded, process-local, and not persisted.",
            "No image upload, training, fine-tuning, reranking, or index rebuilding is included.",
            "",
        ]
    )


def write_text_inference_reports(
    result: ServiceSmokeResult,
    settings: ServiceSettings,
    report_path: Path,
    metrics_path: Path,
) -> None:
    if result.run_state not in TEXT_REPORT_STATES:
        raise ValueError(f"unsupported text inference report state: {result.run_state}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_text_inference_report(result, settings), encoding="utf-8")
    payload = {"settings": {
        "backend": settings.backend,
        "local_files_only": settings.local_files_only,
        "text_model_name": settings.text_model_name,
        "text_model_revision": settings.text_model_revision or "default",
    }, **asdict(result)}
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
