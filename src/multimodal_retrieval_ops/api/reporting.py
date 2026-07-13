"""Deterministic reports for the in-process retrieval-service smoke path."""

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

from .settings import ServiceSettings

REPORT_STATES = frozenset(
    {
        "success",
        "not_run",
        "dependency_unavailable",
        "artifact_unavailable",
        "artifact_incompatible",
        "execution_failed",
    }
)


def serving_dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("fastapi", "uvicorn", "httpx"))


def serving_dependency_message() -> str:
    return (
        "Serving dependencies are unavailable. Install them with "
        '`python -m pip install -e ".[dev,faiss,serve]"`.'
    )


@dataclass(frozen=True)
class ServiceSmokeResult:
    run_state: str
    backend: str
    artifact_readiness: str
    index_info: dict[str, Any] | None
    health_response: dict[str, Any]
    ready_response: dict[str, Any]
    image_retrieval_response: dict[str, Any] | None
    caption_retrieval_response: dict[str, Any] | None
    observability: dict[str, Any] | None
    detail: str


def _format_result_ids(response: dict[str, Any] | None, key: str) -> str:
    if response is None:
        return "none"
    return ", ".join(item[key] for item in response.get("results", [])) or "none"


def render_service_report(result: ServiceSmokeResult) -> str:
    info = result.index_info or {}
    metrics = result.observability or {}
    ef_search = info.get("ef_search")
    return "\n".join(
        [
            "# Retrieval Service Report",
            "",
            f"Run state: **{result.run_state}**",
            "",
            f"- Selected backend: `{result.backend}`",
            f"- Artifact readiness: `{result.artifact_readiness}`",
            f"- Detail: {result.detail}",
            f"- Index type: `{info.get('index_type', 'unavailable')}`",
            f"- FAISS version: `{info.get('faiss_version', 'unavailable')}`",
            f"- Model: `{info.get('model_name', 'unavailable')}` "
            f"(`{info.get('model_revision', 'unavailable')}`)",
            f"- Embedding dimension: {info.get('embedding_dimension', 'unavailable')}",
            f"- Image candidates: {info.get('image_candidate_count', 'unavailable')}",
            f"- Caption candidates: {info.get('caption_candidate_count', 'unavailable')}",
            f"- Split: `{info.get('split', 'unavailable')}`",
            f"- efSearch: {ef_search if ef_search is not None else 'not applicable'}",
            "",
            "## Smoke requests",
            "",
            f"- Health: `{result.health_response.get('status', 'unavailable')}`",
            f"- Ready: `{result.ready_response.get('status', 'unavailable')}`",
            "- Caption-to-image result IDs: "
            f"`{_format_result_ids(result.image_retrieval_response, 'image_id')}`",
            "- Image-to-caption result IDs: "
            f"`{_format_result_ids(result.caption_retrieval_response, 'caption_id')}`",
            "",
            "## Process-local observability",
            "",
            f"- Total requests observed: {metrics.get('total_requests', 0)}",
            f"- Retrieval latency observations: {metrics.get('latency_observation_count', 0)}",
            f"- Errors: {metrics.get('error_count', 0)}",
            f"- Unknown query IDs: {metrics.get('unknown_query_id_count', 0)}",
            f"- Invalid top-k requests: {metrics.get('invalid_top_k_count', 0)}",
            "",
            "## Limitations",
            "",
            "The service accepts cached caption and image IDs only. It does not encode arbitrary",
            "text, accept image uploads, run CLIP, rebuild indexes, or persist metrics. Counters and",
            "latency observations are bounded to one process and reset when that process exits.",
            "FlatIP remains the default correctness-oriented backend; HNSW must be selected",
            "explicitly and is not presented as universally faster or better.",
            "",
        ]
    )


def write_service_reports(
    result: ServiceSmokeResult, report_path: Path, metrics_path: Path
) -> None:
    if result.run_state not in REPORT_STATES:
        raise ValueError(f"unsupported retrieval service report state: {result.run_state}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_service_report(result), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(result.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def failure_result(settings: ServiceSettings, state: str, detail: str) -> ServiceSmokeResult:
    return ServiceSmokeResult(
        run_state=state,
        backend=settings.backend,
        artifact_readiness="unready",
        index_info=None,
        health_response={},
        ready_response={"status": "unready", "reasons": [detail]},
        image_retrieval_response=None,
        caption_retrieval_response=None,
        observability=None,
        detail=detail,
    )
