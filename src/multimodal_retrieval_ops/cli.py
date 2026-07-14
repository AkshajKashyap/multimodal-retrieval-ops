"""Command-line interface for project foundation workflows."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from . import __version__
from .baseline_index import build_index, exact_search, load_index, write_index
from .benchmark import run_clip_benchmark
from .clip_backend import (
    DEFAULT_CLIP_MODEL,
    ClipBackendError,
    ClipEmbeddingBackend,
    ClipExecutionError,
    ClipModelUnavailableError,
    clip_dependencies_available,
    clip_dependency_message,
)
from .clip_reporting import (
    write_clip_backend_report,
    write_clip_failure_reports,
    write_clip_retrieval_reports,
)
from .clip_workflow import build_clip_index
from .config import load_config
from .contrastive_adapters import (
    AdapterCacheIncompatibleError,
    AdapterDatasetUnavailableError,
    AdapterDependencyError,
    AdapterEvaluationError,
    AdapterTrainingConfig,
    AdapterTrainingError,
    ContrastiveAdapterError,
    evaluate_adapters,
    load_adapter_cache,
    prepare_adapter_cache,
    train_adapters,
    training_dependencies_available,
    training_dependency_message,
)
from .contrastive_adapter_reporting import (
    write_adapter_failure_reports,
    write_adapter_reports,
)
from .contrastive_adapter_diagnostics import (
    AdapterDiagnosticError,
    DiagnosticArtifactIncompatibleError,
    DiagnosticArtifactUnavailableError,
    DiagnosticCheckpointUnavailableError,
    run_adapter_failure_analysis,
)
from .contrastive_adapter_diagnostic_reporting import (
    write_diagnostic_failure_reports,
    write_diagnostic_reports,
)
from .demo import generate_demo_manifest
from .deterministic_image_encoder import DeterministicImageEncoder
from .deterministic_text_encoder import DeterministicTextEncoder
from .flickr8k import (
    create_benchmark_subset,
    ingest_flickr8k,
    multi_caption_statistics,
    render_flickr8k_report,
)
from .faiss_flat import (
    FaissCacheError,
    FaissDependencyError,
    FaissFlatError,
    FaissIndexStaleError,
    build_flickr8k_flat_artifacts,
    evaluate_flickr8k_faiss,
    faiss_available,
    faiss_dependency_message,
    load_flickr8k_artifacts,
    search_cached_embedding,
    write_correctness_outputs,
    write_faiss_failure,
)
from .faiss_hnsw import (
    ALLOWED_EF_SEARCH,
    build_flickr8k_hnsw_artifacts,
    evaluate_flickr8k_hnsw,
    load_flickr8k_hnsw_artifacts,
    search_hnsw_embedding,
    write_hnsw_failure,
    write_hnsw_outputs,
)
from .faiss_reranking import (
    CANDIDATE_K,
    EF_SEARCH,
    RerankingArtifactIncompatibleError,
    RerankingArtifactUnavailableError,
    evaluate_hnsw_reranking,
    load_persisted_reranking_info,
    load_reranking_artifacts,
    search_reranked_embedding,
    write_reranking_failure,
    write_reranking_outputs,
)
from .hf_clip_benchmark import run_hf_clip_benchmark, write_hf_clip_failure
from .hf_flickr8k import (
    DEFAULT_HF_FLICKR8K_DATASET,
    HFDataExecutionError,
    HFDatasetUnavailableError,
    HFFlickr8kError,
    ingest_hf_flickr8k,
    load_hf_provenance,
    render_hf_dataset_report,
    write_hf_failure_report,
)
from .ingestion import ingest_local_directory
from .inspection import inspect_items, write_dataset_report
from .manifest import (
    ManifestItemV2,
    ManifestValidationError,
    migrate_to_v2,
    read_manifest,
    read_manifest_rows,
    validate_image_paths,
    write_manifest,
)
from .multimodal_evaluation import evaluate_multimodal_index
from .multimodal_index import (
    build_multimodal_index,
    load_multimodal_index,
    search_multimodal_index,
    write_multimodal_index,
)
from .multimodal_reporting import write_multimodal_reports
from .reporting import write_manifest_report
from .evaluation import evaluate_index
from .retrieval_reporting import write_retrieval_reports
from .retrieval_monitoring import (
    HealthThresholds,
    RetrievalMonitoringError,
    TelemetryUnavailableError,
    analyze_events,
    read_telemetry,
    write_monitoring_failure,
    write_monitoring_outputs,
)
from .splitting import assign_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multimodal-retrieval-ops")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("project-info", help="show project information")
    generate = subparsers.add_parser("generate-demo-manifest", help="write deterministic demo CSV")
    generate.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate-manifest", help="validate a manifest CSV")
    validate.add_argument("--manifest", type=Path)
    report = subparsers.add_parser("generate-manifest-report", help="write a Markdown summary")
    report.add_argument("--manifest", type=Path)
    report.add_argument("--output", type=Path)
    ingest = subparsers.add_parser("ingest-local-fixture", help="ingest local image captions")
    ingest.add_argument("--directory", type=Path)
    ingest.add_argument("--output", type=Path)
    ingest.add_argument("--seed", type=int, default=42)
    split = subparsers.add_parser("split-manifest", help="deterministically reassign splits")
    split.add_argument("--manifest", type=Path)
    split.add_argument("--output", type=Path)
    split.add_argument("--train-fraction", type=float, default=0.7)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--test-fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    inspect = subparsers.add_parser("inspect-manifest", help="write dataset quality report")
    inspect.add_argument("--manifest", type=Path)
    inspect.add_argument("--output", type=Path)
    build = subparsers.add_parser(
        "build-text-baseline-index", help="build the lexical exact-search index"
    )
    build.add_argument("--manifest", type=Path)
    build.add_argument("--index-output", type=Path)
    build.add_argument("--vocab-output", type=Path)
    search = subparsers.add_parser("search-text-baseline", help="search the lexical index")
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--index", type=Path)
    search.add_argument("--vocab", type=Path)
    evaluate = subparsers.add_parser(
        "evaluate-text-baseline", help="evaluate held-out lexical retrieval"
    )
    evaluate.add_argument("--index", type=Path)
    evaluate.add_argument("--vocab", type=Path)
    evaluate.add_argument("--report-output", type=Path)
    evaluate.add_argument("--metrics-output", type=Path)
    multimodal_build = subparsers.add_parser(
        "build-multimodal-baseline-index", help="build the placeholder image embedding index"
    )
    multimodal_build.add_argument("--manifest", type=Path)
    multimodal_build.add_argument("--output", type=Path)
    multimodal_build.add_argument("--dimension", type=int, default=64)
    multimodal_search = subparsers.add_parser(
        "search-multimodal-baseline", help="search images with placeholder text embeddings"
    )
    multimodal_search.add_argument("--query", required=True)
    multimodal_search.add_argument("--k", type=int, default=5)
    multimodal_search.add_argument("--index", type=Path)
    multimodal_evaluate = subparsers.add_parser(
        "evaluate-multimodal-baseline", help="evaluate held-out text-to-image retrieval"
    )
    multimodal_evaluate.add_argument("--index", type=Path)
    multimodal_evaluate.add_argument("--report-output", type=Path)
    multimodal_evaluate.add_argument("--metrics-output", type=Path)
    clip_info = subparsers.add_parser(
        "clip-backend-info", help="show optional CLIP backend availability"
    )
    clip_info.add_argument("--output", type=Path)
    clip_info.add_argument("--model-name")
    clip_info.add_argument("--device", default="cpu")
    clip_info.add_argument("--batch-size", type=int, default=8)
    clip_info.add_argument("--local-files-only", action="store_true")
    clip_build = subparsers.add_parser("build-clip-index", help="build a cached CLIP image index")
    clip_build.add_argument("--manifest", type=Path)
    clip_build.add_argument("--output", type=Path)
    clip_build.add_argument("--cache", type=Path)
    clip_build.add_argument("--model-name", default=DEFAULT_CLIP_MODEL)
    clip_build.add_argument("--device", default="cpu")
    clip_build.add_argument("--batch-size", type=int, default=8)
    clip_build.add_argument("--allow-download", action="store_true")
    clip_build.add_argument("--local-files-only", action="store_true")
    clip_search = subparsers.add_parser("search-clip", help="search a CLIP image index")
    clip_search.add_argument("--query", required=True)
    clip_search.add_argument("--k", type=int, default=5)
    clip_search.add_argument("--index", type=Path)
    clip_search.add_argument("--model-name")
    clip_search.add_argument("--device", default="cpu")
    clip_search.add_argument("--batch-size", type=int, default=8)
    clip_search.add_argument("--allow-download", action="store_true")
    clip_search.add_argument("--local-files-only", action="store_true")
    clip_evaluate = subparsers.add_parser(
        "evaluate-clip", help="evaluate held-out CLIP text-to-image retrieval"
    )
    clip_evaluate.add_argument("--index", type=Path)
    clip_evaluate.add_argument("--model-name")
    clip_evaluate.add_argument("--device", default="cpu")
    clip_evaluate.add_argument("--batch-size", type=int, default=8)
    clip_evaluate.add_argument("--allow-download", action="store_true")
    clip_evaluate.add_argument("--local-files-only", action="store_true")
    clip_evaluate.add_argument("--report-output", type=Path)
    clip_evaluate.add_argument("--metrics-output", type=Path)
    migrate = subparsers.add_parser("migrate-manifest-v2", help="migrate a manifest to schema v2")
    migrate.add_argument("--manifest", type=Path)
    migrate.add_argument("--output", type=Path)
    migrate.add_argument("--report-output", type=Path)
    flickr = subparsers.add_parser("ingest-flickr8k", help="ingest local Flickr8k captions")
    flickr.add_argument("--images-dir", type=Path, required=True)
    flickr.add_argument("--captions-file", type=Path, required=True)
    flickr.add_argument("--output", type=Path)
    flickr.add_argument("--report-output", type=Path)
    flickr.add_argument("--seed", type=int, default=42)
    subset = subparsers.add_parser(
        "create-benchmark-subset", help="create a seeded image-group benchmark subset"
    )
    subset.add_argument("--manifest", type=Path)
    subset.add_argument("--output", type=Path)
    subset.add_argument("--max-images", type=int, default=500)
    subset.add_argument("--seed", type=int, default=42)
    clip_benchmark = subparsers.add_parser(
        "evaluate-clip-benchmark", help="run opt-in multi-caption CLIP benchmark"
    )
    clip_benchmark.add_argument("--manifest", type=Path)
    clip_benchmark.add_argument("--model-name", default=DEFAULT_CLIP_MODEL)
    clip_benchmark.add_argument("--device", default="cpu")
    clip_benchmark.add_argument("--batch-size", type=int, default=16)
    clip_benchmark.add_argument("--local-files-only", action="store_true")
    hf_ingest = subparsers.add_parser(
        "ingest-hf-flickr8k", help="materialize opt-in Hugging Face Flickr8k"
    )
    hf_ingest.add_argument("--dataset-name", default=DEFAULT_HF_FLICKR8K_DATASET)
    hf_ingest.add_argument("--revision")
    hf_ingest.add_argument("--cache-dir", type=Path)
    hf_ingest.add_argument("--output-manifest", type=Path)
    hf_ingest.add_argument("--images-dir", type=Path)
    hf_ingest.add_argument("--max-images-per-split", type=int)
    hf_ingest.add_argument("--local-files-only", action="store_true")
    hf_ingest.add_argument("--force", action="store_true")
    hf_inspect = subparsers.add_parser(
        "inspect-hf-flickr8k", help="inspect materialized Hugging Face Flickr8k"
    )
    hf_inspect.add_argument("--provenance", type=Path)
    hf_inspect.add_argument("--output", type=Path)
    hf_evaluate = subparsers.add_parser(
        "evaluate-clip-flickr8k", help="run bidirectional exact CLIP evaluation"
    )
    hf_evaluate.add_argument("--manifest", type=Path)
    hf_evaluate.add_argument("--provenance", type=Path)
    hf_evaluate.add_argument("--split", default="test")
    hf_evaluate.add_argument("--max-images", type=int)
    hf_evaluate.add_argument("--seed", type=int, default=42)
    hf_evaluate.add_argument("--model-name", default=DEFAULT_CLIP_MODEL)
    hf_evaluate.add_argument("--model-revision")
    hf_evaluate.add_argument("--device", default="cpu")
    hf_evaluate.add_argument("--batch-size", type=int, default=16)
    hf_evaluate.add_argument("--local-files-only", action="store_true")
    faiss_info = subparsers.add_parser(
        "faiss-backend-info", help="show optional FAISS CPU backend availability"
    )
    faiss_info.add_argument("--output", type=Path)
    faiss_build = subparsers.add_parser(
        "build-faiss-flat-indexes", help="build FlatIP indexes from cached embeddings"
    )
    faiss_build.add_argument("--cache", type=Path)
    faiss_build.add_argument("--artifacts-dir", type=Path)
    faiss_evaluate = subparsers.add_parser(
        "evaluate-faiss-flat", help="compare FAISS FlatIP against exact cosine"
    )
    faiss_evaluate.add_argument("--cache", type=Path)
    faiss_evaluate.add_argument("--manifest", type=Path)
    faiss_evaluate.add_argument("--artifacts-dir", type=Path)
    faiss_text = subparsers.add_parser(
        "search-faiss-text", help="search images using a cached caption embedding"
    )
    faiss_text.add_argument("--query-caption-id", required=True)
    faiss_text.add_argument("--k", type=int, default=10)
    faiss_text.add_argument("--cache", type=Path)
    faiss_text.add_argument("--artifacts-dir", type=Path)
    faiss_image = subparsers.add_parser(
        "search-faiss-image", help="search captions using a cached image embedding"
    )
    faiss_image.add_argument("--query-image-id", required=True)
    faiss_image.add_argument("--k", type=int, default=10)
    faiss_image.add_argument("--cache", type=Path)
    faiss_image.add_argument("--artifacts-dir", type=Path)
    hnsw_build = subparsers.add_parser(
        "build-faiss-hnsw-indexes", help="build bounded HNSW indexes from cached embeddings"
    )
    hnsw_build.add_argument("--cache", type=Path)
    hnsw_build.add_argument("--artifacts-dir", type=Path)
    hnsw_evaluate = subparsers.add_parser(
        "evaluate-faiss-hnsw", help="compare bounded HNSW search against FlatIP"
    )
    hnsw_evaluate.add_argument("--cache", type=Path)
    hnsw_evaluate.add_argument("--manifest", type=Path)
    hnsw_evaluate.add_argument("--flat-artifacts-dir", type=Path)
    hnsw_evaluate.add_argument("--hnsw-artifacts-dir", type=Path)
    hnsw_text = subparsers.add_parser(
        "search-hnsw-text", help="search HNSW images using a cached caption embedding"
    )
    hnsw_text.add_argument("--query-caption-id", required=True)
    hnsw_text.add_argument("--ef-search", type=int, choices=ALLOWED_EF_SEARCH, default=32)
    hnsw_text.add_argument("--k", type=int, default=10)
    hnsw_text.add_argument("--cache", type=Path)
    hnsw_text.add_argument("--artifacts-dir", type=Path)
    hnsw_image = subparsers.add_parser(
        "search-hnsw-image", help="search HNSW captions using a cached image embedding"
    )
    hnsw_image.add_argument("--query-image-id", required=True)
    hnsw_image.add_argument("--ef-search", type=int, choices=ALLOWED_EF_SEARCH, default=32)
    hnsw_image.add_argument("--k", type=int, default=10)
    hnsw_image.add_argument("--cache", type=Path)
    hnsw_image.add_argument("--artifacts-dir", type=Path)
    rerank_evaluate = subparsers.add_parser(
        "evaluate-hnsw-reranking", help="evaluate fixed exact reranking of HNSW top-50"
    )
    rerank_evaluate.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    rerank_evaluate.add_argument("--ef-search", type=int, default=EF_SEARCH)
    rerank_evaluate.add_argument("--cache", type=Path)
    rerank_evaluate.add_argument("--manifest", type=Path)
    rerank_evaluate.add_argument("--flat-artifacts-dir", type=Path)
    rerank_evaluate.add_argument("--hnsw-artifacts-dir", type=Path)
    subparsers.add_parser(
        "hnsw-reranking-info", help="load and summarize the persisted reranking result"
    )
    rerank_text = subparsers.add_parser(
        "search-reranked-text", help="rerank HNSW images for a cached caption ID"
    )
    rerank_text.add_argument("--query-caption-id", required=True)
    rerank_text.add_argument("--k", type=int, default=10)
    rerank_text.add_argument("--cache", type=Path)
    rerank_text.add_argument("--flat-artifacts-dir", type=Path)
    rerank_text.add_argument("--hnsw-artifacts-dir", type=Path)
    rerank_image = subparsers.add_parser(
        "search-reranked-image", help="rerank HNSW captions for a cached image ID"
    )
    rerank_image.add_argument("--query-image-id", required=True)
    rerank_image.add_argument("--k", type=int, default=10)
    rerank_image.add_argument("--cache", type=Path)
    rerank_image.add_argument("--flat-artifacts-dir", type=Path)
    rerank_image.add_argument("--hnsw-artifacts-dir", type=Path)
    for command, help_text in (
        ("serve-retrieval", "serve persisted retrieval artifacts over HTTP"),
        ("retrieval-service-info", "inspect retrieval-service artifact readiness"),
        ("retrieval-service-smoke", "run bounded in-process service requests"),
        ("search-live-text", "search persisted images with locally encoded text"),
        ("search-live-image", "search persisted captions with a locally encoded image"),
    ):
        service = subparsers.add_parser(command, help=help_text)
        service.add_argument("--backend", choices=("flat", "hnsw"), default="flat")
        service.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
        service.add_argument("--embedding-cache", type=Path)
        service.add_argument("--manifest", type=Path)
        service.add_argument("--ef-search", type=int, choices=ALLOWED_EF_SEARCH, default=64)
        service.add_argument("--maximum-top-k", type=int, default=100)
        service.add_argument("--host", default="127.0.0.1")
        service.add_argument("--port", type=int, default=8000)
        service.add_argument("--enable-text-inference", action="store_true")
        service.add_argument("--text-model-name", default=DEFAULT_CLIP_MODEL)
        service.add_argument("--text-model-revision")
        service.add_argument("--text-device", default="cpu")
        service.add_argument(
            "--local-files-only",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        service.add_argument("--maximum-text-length", type=int, default=512)
        service.add_argument("--text-query-cache-size", type=int, default=128)
        service.add_argument("--enable-image-inference", action="store_true")
        service.add_argument("--image-model-name", default=DEFAULT_CLIP_MODEL)
        service.add_argument("--image-model-revision")
        service.add_argument("--image-device", default="cpu")
        service.add_argument("--maximum-upload-bytes", type=int, default=10 * 1024 * 1024)
        service.add_argument("--maximum-pixel-count", type=int, default=20_000_000)
        service.add_argument("--image-query-cache-size", type=int, default=64)
        service.add_argument("--smoke-image-path", type=Path)
        service.add_argument("--enable-telemetry", action="store_true")
        service.add_argument("--telemetry-path", type=Path)
        service.add_argument("--telemetry-max-bytes", type=int, default=5 * 1024 * 1024)
        service.add_argument("--telemetry-backup-count", type=int, default=2)
        service.add_argument(
            "--telemetry-flush-each-event",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        if command == "retrieval-service-smoke":
            service.add_argument("--report-output", type=Path)
            service.add_argument("--metrics-output", type=Path)
        if command == "search-live-text":
            service.add_argument("--query", required=True)
            service.add_argument("--top-k", type=int, default=5)
        if command == "search-live-image":
            service.add_argument("--image-path", required=True)
            service.add_argument("--top-k", type=int, default=5)
    telemetry_info = subparsers.add_parser(
        "retrieval-telemetry-info", help="inspect local telemetry configuration"
    )
    telemetry_info.add_argument("--enabled", action="store_true")
    telemetry_info.add_argument("--telemetry-path", type=Path)
    telemetry_info.add_argument("--telemetry-max-bytes", type=int, default=5 * 1024 * 1024)
    telemetry_info.add_argument("--telemetry-backup-count", type=int, default=2)
    telemetry_info.add_argument(
        "--telemetry-flush-each-event",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    telemetry_analyze = subparsers.add_parser(
        "analyze-retrieval-telemetry", help="analyze local retrieval telemetry offline"
    )
    telemetry_analyze.add_argument("--telemetry-path", type=Path)
    telemetry_analyze.add_argument("--include-rotated", action="store_true")
    telemetry_analyze.add_argument("--maximum-error-rate", type=float, default=0.25)
    telemetry_analyze.add_argument("--maximum-readiness-failures", type=int, default=0)
    telemetry_analyze.add_argument("--maximum-p95-latency-ms", type=float)
    telemetry_analyze.add_argument("--minimum-labeled-recall-at-10", type=float, default=0.80)
    telemetry_analyze.add_argument("--minimum-labeled-mrr", type=float, default=0.50)
    telemetry_analyze.add_argument("--report-output", type=Path)
    telemetry_analyze.add_argument("--metrics-output", type=Path)
    telemetry_analyze.add_argument("--decision-output", type=Path)
    telemetry_smoke = subparsers.add_parser(
        "retrieval-telemetry-smoke", help="run one in-process cached-ID telemetry smoke"
    )
    telemetry_smoke.add_argument("--backend", choices=("flat", "hnsw"), default="flat")
    telemetry_smoke.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    telemetry_smoke.add_argument("--embedding-cache", type=Path)
    telemetry_smoke.add_argument("--manifest", type=Path)
    telemetry_smoke.add_argument("--ef-search", type=int, choices=ALLOWED_EF_SEARCH, default=64)
    telemetry_smoke.add_argument("--telemetry-path", type=Path)
    telemetry_smoke.add_argument("--telemetry-max-bytes", type=int, default=5 * 1024 * 1024)
    telemetry_smoke.add_argument("--telemetry-backup-count", type=int, default=2)
    adapter_prepare = subparsers.add_parser(
        "prepare-adapter-embeddings",
        help="cache bounded frozen train and validation CLIP embeddings",
    )
    adapter_prepare.add_argument("--manifest", type=Path)
    adapter_prepare.add_argument("--provenance", type=Path)
    adapter_prepare.add_argument("--train-cache", type=Path)
    adapter_prepare.add_argument("--validation-cache", type=Path)
    adapter_prepare.add_argument("--train-images", type=int, default=500)
    adapter_prepare.add_argument("--validation-images", type=int, default=100)
    adapter_prepare.add_argument("--seed", type=int, default=42)
    adapter_prepare.add_argument("--model-name", default=DEFAULT_CLIP_MODEL)
    adapter_prepare.add_argument("--model-revision")
    adapter_prepare.add_argument("--device", default="cpu")
    adapter_prepare.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    adapter_train = subparsers.add_parser(
        "train-contrastive-adapters",
        help="train one fixed residual-adapter configuration over frozen embeddings",
    )
    adapter_train.add_argument("--train-cache", type=Path)
    adapter_train.add_argument("--validation-cache", type=Path)
    adapter_train.add_argument("--checkpoint", type=Path)
    adapter_train.add_argument("--metadata-output", type=Path)
    adapter_train.add_argument("--seed", type=int, default=42)
    adapter_train.add_argument("--device", default="cpu")
    adapter_train.add_argument("--max-epochs", type=int, default=20)
    adapter_train.add_argument("--early-stopping-patience", type=int, default=4)
    adapter_train.add_argument("--batch-size", type=int, default=64)
    adapter_evaluate = subparsers.add_parser(
        "evaluate-contrastive-adapters",
        help="compare zero-shot and adapted retrieval on validation only",
    )
    adapter_evaluate.add_argument("--train-cache", type=Path)
    adapter_evaluate.add_argument("--validation-cache", type=Path)
    adapter_evaluate.add_argument("--checkpoint", type=Path)
    adapter_evaluate.add_argument("--metadata", type=Path)
    adapter_evaluate.add_argument("--device", default="cpu")
    adapter_evaluate.add_argument("--report-output", type=Path)
    adapter_evaluate.add_argument("--metrics-output", type=Path)
    adapter_evaluate.add_argument("--promotion-output", type=Path)
    adapter_info = subparsers.add_parser(
        "contrastive-adapter-info",
        help="inspect optional training support and generated adapter artifacts",
    )
    adapter_info.add_argument("--train-cache", type=Path)
    adapter_info.add_argument("--validation-cache", type=Path)
    adapter_info.add_argument("--checkpoint", type=Path)
    adapter_info.add_argument("--metadata", type=Path)
    adapter_info.add_argument("--model-name", default=DEFAULT_CLIP_MODEL)
    adapter_info.add_argument("--model-revision")
    adapter_info.add_argument("--device", default="cpu")
    adapter_info.add_argument("--seed", type=int, default=42)
    adapter_info.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    diagnostic = subparsers.add_parser(
        "analyze-contrastive-adapter",
        help="analyze the saved adapter failure on validation only",
    )
    diagnostic.add_argument("--train-cache", type=Path)
    diagnostic.add_argument("--validation-cache", type=Path)
    diagnostic.add_argument("--checkpoint", type=Path)
    diagnostic.add_argument("--metadata", type=Path)
    diagnostic.add_argument("--manifest", type=Path)
    diagnostic.add_argument("--recorded-metrics", type=Path)
    diagnostic.add_argument("--device", default="cpu")
    diagnostic.add_argument("--report-output", type=Path)
    diagnostic.add_argument("--metrics-output", type=Path)
    diagnostic.add_argument("--memo-output", type=Path)
    diagnostic_info = subparsers.add_parser(
        "contrastive-adapter-diagnostics-info",
        help="inspect read-only diagnostic artifact availability",
    )
    diagnostic_info.add_argument("--train-cache", type=Path)
    diagnostic_info.add_argument("--validation-cache", type=Path)
    diagnostic_info.add_argument("--checkpoint", type=Path)
    diagnostic_info.add_argument("--metadata", type=Path)
    diagnostic_info.add_argument("--manifest", type=Path)
    diagnostic_info.add_argument("--recorded-metrics", type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    text_encoder_factory: Any = None,
    image_encoder_factory: Any = None,
) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    try:
        if args.command == "project-info":
            print(f"multimodal-retrieval-ops {__version__}")
            print("Milestone: 11A (privacy-safe retrieval telemetry and offline monitoring)")
            print("Runtime: lightweight base install; optional CPU/GPU CLIP extra")
        elif args.command == "generate-demo-manifest":
            output = args.output or config.manifest_path
            items = generate_demo_manifest(output)
            print(f"Wrote {len(items)} rows to {output}")
        elif args.command == "validate-manifest":
            manifest = args.manifest or (
                config.dataset_manifest_path
                if config.dataset_manifest_path.is_file()
                else config.manifest_path
            )
            items = read_manifest(manifest)
            print(f"Valid manifest: {manifest} ({len(items)} rows)")
        elif args.command == "generate-manifest-report":
            manifest = args.manifest or config.manifest_path
            output = args.output or config.report_path
            items = read_manifest(manifest)
            write_manifest_report(items, output)
            print(f"Wrote manifest report to {output}")
        elif args.command == "ingest-local-fixture":
            directory = args.directory or config.fixture_path
            output = args.output or config.ingested_manifest_path
            items = ingest_local_directory(directory, output, seed=args.seed)
            print(f"Ingested {len(items)} image-caption pairs to {output}")
        elif args.command == "split-manifest":
            manifest = args.manifest or config.ingested_manifest_path
            output = args.output or config.dataset_manifest_path
            items = read_manifest(manifest)
            fractions = (
                args.train_fraction,
                args.validation_fraction,
                args.test_fraction,
            )
            split_items = assign_splits(items, fractions=fractions, seed=args.seed)
            validate_image_paths(split_items)
            write_manifest(split_items, output)
            print(f"Wrote {len(split_items)} deterministically split rows to {output}")
        elif args.command == "inspect-manifest":
            manifest = args.manifest or config.dataset_manifest_path
            output = args.output or config.inspection_report_path
            items = read_manifest_rows(manifest)
            statistics = inspect_items(items)
            write_dataset_report(statistics, output)
            print(f"Inspected {statistics.row_count} rows; wrote report to {output}")
        elif args.command == "build-text-baseline-index":
            manifest = args.manifest or config.dataset_manifest_path
            index_output = args.index_output or config.baseline_index_path
            vocab_output = args.vocab_output or config.baseline_vocab_path
            items = read_manifest(manifest)
            vocabulary, entries = build_index(items)
            write_index(vocabulary, entries, index_output, vocab_output)
            print(
                f"Built lexical index with {len(entries)} items and {len(vocabulary)} "
                f"train-vocabulary tokens at {index_output}"
            )
        elif args.command == "search-text-baseline":
            index_path = args.index or config.baseline_index_path
            vocab_path = args.vocab or config.baseline_vocab_path
            vocabulary, entries = load_index(index_path, vocab_path)
            results = exact_search(args.query, vocabulary, entries, k=args.k)
            print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
        elif args.command == "evaluate-text-baseline":
            index_path = args.index or config.baseline_index_path
            vocab_path = args.vocab or config.baseline_vocab_path
            report_output = args.report_output or config.baseline_report_path
            metrics_output = args.metrics_output or config.baseline_metrics_path
            vocabulary, entries = load_index(index_path, vocab_path)
            metrics, _ = evaluate_index(vocabulary, entries)
            write_retrieval_reports(metrics, report_output, metrics_output)
            print(
                f"Evaluated {metrics.query_count} held-out queries; "
                f"Recall@1={metrics.recall_at_1:.4f}, MRR={metrics.mrr:.4f}"
            )
        elif args.command == "build-multimodal-baseline-index":
            manifest = args.manifest or config.dataset_manifest_path
            output = args.output or config.multimodal_index_path
            items = read_manifest(manifest)
            text_encoder = DeterministicTextEncoder(dimension=args.dimension)
            image_encoder = DeterministicImageEncoder(dimension=args.dimension)
            index = build_multimodal_index(items, image_encoder, text_encoder)
            write_multimodal_index(index, output)
            print(
                f"Built {index.dimension}-dimensional multimodal placeholder index with "
                f"{len(index.entries)} items at {output}"
            )
        elif args.command == "search-multimodal-baseline":
            index_path = args.index or config.multimodal_index_path
            index = load_multimodal_index(index_path)
            text_encoder = DeterministicTextEncoder(dimension=index.dimension)
            results = search_multimodal_index(args.query, text_encoder, index, k=args.k)
            print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
        elif args.command == "evaluate-multimodal-baseline":
            index_path = args.index or config.multimodal_index_path
            report_output = args.report_output or config.multimodal_report_path
            metrics_output = args.metrics_output or config.multimodal_metrics_path
            index = load_multimodal_index(index_path)
            text_encoder = DeterministicTextEncoder(dimension=index.dimension)
            metrics, _ = evaluate_multimodal_index(text_encoder, index)
            write_multimodal_reports(
                metrics, index.backend_name, report_output, metrics_output
            )
            print(
                f"Evaluated {metrics.query_count} held-out text-to-image queries; "
                f"Recall@1={metrics.recall_at_1:.4f}, MRR={metrics.mrr:.4f}"
            )
        elif args.command == "clip-backend-info":
            available = clip_dependencies_available()
            output = args.output or config.clip_backend_report_path
            print(f"CLIP dependencies: {'available' if available else 'not installed'}")
            print(f"Default model: {DEFAULT_CLIP_MODEL}")
            if args.model_name and available:
                backend = ClipEmbeddingBackend(
                    model_name=args.model_name,
                    device=args.device,
                    batch_size=args.batch_size,
                    allow_download=not args.local_files_only,
                )
                backend.ensure_loaded()
                write_clip_backend_report(
                    output,
                    status="successfully executed",
                    model_name=backend.model_name,
                    device=backend.device,
                    dimension=backend.dimension,
                    detail="Processor and model loaded successfully.",
                )
                print(
                    f"Loaded {backend.model_name} on {backend.device}; "
                    f"embedding dimension={backend.dimension}"
                )
            elif not available:
                write_clip_backend_report(
                    output,
                    status="unavailable dependencies",
                    detail=clip_dependency_message(),
                )
                print(clip_dependency_message())
            else:
                write_clip_backend_report(
                    output,
                    status="dependencies available; model not loaded",
                )
            print(f"Wrote backend report to {output}")
        elif args.command == "build-clip-index":
            manifest = args.manifest or config.dataset_manifest_path
            output = args.output or config.clip_index_path
            cache_path = args.cache or config.clip_cache_path
            items = read_manifest(manifest)
            backend = ClipEmbeddingBackend(
                model_name=args.model_name,
                device=args.device,
                batch_size=args.batch_size,
                allow_download=args.allow_download or not args.local_files_only,
            )
            index, reused = build_clip_index(items, backend, cache_path)
            write_multimodal_index(index, output)
            write_clip_backend_report(
                config.clip_backend_report_path,
                status="successfully executed",
                model_name=index.model_name or args.model_name,
                device=args.device,
                dimension=index.dimension,
                item_count=len(index.entries),
                cache_status="hit" if reused else "miss (created)",
                detail="CLIP image index built successfully.",
            )
            print(
                f"Built {index.dimension}-dimensional CLIP index with {len(index.entries)} "
                f"items at {output} "
                f"(cache {'reused' if reused else 'created'})"
            )
        elif args.command == "search-clip":
            index_path = args.index or config.clip_index_path
            index = load_multimodal_index(index_path)
            model_name = args.model_name or index.model_name or DEFAULT_CLIP_MODEL
            backend = ClipEmbeddingBackend(
                model_name=model_name,
                device=args.device,
                batch_size=args.batch_size,
                allow_download=args.allow_download or not args.local_files_only,
            )
            backend.ensure_loaded()
            results = search_multimodal_index(args.query, backend, index, k=args.k)
            print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
        elif args.command == "evaluate-clip":
            index_path = args.index or config.clip_index_path
            index = load_multimodal_index(index_path)
            model_name = args.model_name or index.model_name or DEFAULT_CLIP_MODEL
            backend = ClipEmbeddingBackend(
                model_name=model_name,
                device=args.device,
                batch_size=args.batch_size,
                allow_download=args.allow_download or not args.local_files_only,
            )
            backend.ensure_loaded()
            metrics, _ = evaluate_multimodal_index(backend, index)
            report_output = args.report_output or config.clip_report_path
            metrics_output = args.metrics_output or config.clip_metrics_path
            write_clip_retrieval_reports(
                metrics,
                model_name,
                report_output,
                metrics_output,
                device=args.device,
                dimension=index.dimension,
                item_count=len(index.entries),
            )
            print(
                f"Evaluated {metrics.query_count} held-out CLIP queries; "
                f"Recall@1={metrics.recall_at_1:.4f}, MRR={metrics.mrr:.4f}"
            )
        elif args.command == "migrate-manifest-v2":
            manifest = args.manifest or config.dataset_manifest_path
            output = args.output or manifest
            report_output = args.report_output or config.schema_v2_report_path
            source_rows = read_manifest(manifest)
            rows = migrate_to_v2(source_rows)
            write_manifest(rows, output)
            stats = multi_caption_statistics(rows)
            report_output.parent.mkdir(parents=True, exist_ok=True)
            report_output.write_text(
                "\n".join(
                    [
                        "# Schema v2 Migration Report",
                        "",
                        "Status: **successfully migrated**",
                        "",
                        f"- Source rows: {len(source_rows)}",
                        f"- Unique images: {stats.unique_images}",
                        f"- Caption queries: {stats.caption_queries}",
                        f"- Output: `{output.as_posix()}`",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            print(
                f"Migrated {len(rows)} caption rows across {stats.unique_images} "
                f"images to {output}"
            )
        elif args.command == "ingest-flickr8k":
            output = args.output or config.flickr8k_manifest_path
            report_output = args.report_output or config.flickr8k_report_path
            rows = ingest_flickr8k(
                args.images_dir, args.captions_file, output, seed=args.seed
            )
            report_output.parent.mkdir(parents=True, exist_ok=True)
            report_output.write_text(render_flickr8k_report(rows), encoding="utf-8")
            stats = multi_caption_statistics(rows)
            print(
                f"Ingested {stats.caption_queries} captions for {stats.unique_images} "
                f"images to {output}"
            )
        elif args.command == "create-benchmark-subset":
            manifest = args.manifest or config.flickr8k_manifest_path
            output = args.output or config.benchmark_manifest_path
            raw_rows = read_manifest(manifest)
            rows = [row for row in raw_rows if isinstance(row, ManifestItemV2)]
            if len(rows) != len(raw_rows):
                raise ValueError("benchmark subsetting requires a schema-v2 manifest")
            subset_rows = create_benchmark_subset(rows, args.max_images, seed=args.seed)
            write_manifest(subset_rows, output)
            stats = multi_caption_statistics(subset_rows)
            print(
                f"Created benchmark subset with {stats.unique_images} images and "
                f"{stats.caption_queries} captions at {output}"
            )
        elif args.command == "evaluate-clip-benchmark":
            manifest = args.manifest or config.benchmark_manifest_path
            raw_rows = read_manifest(manifest)
            rows = [row for row in raw_rows if isinstance(row, ManifestItemV2)]
            if len(rows) != len(raw_rows):
                raise ValueError("CLIP benchmark requires a schema-v2 manifest")
            backend = ClipEmbeddingBackend(
                model_name=args.model_name,
                device=args.device,
                batch_size=args.batch_size,
                allow_download=not args.local_files_only,
            )
            metrics, cache_hit = run_clip_benchmark(
                rows,
                backend,
                cache_path=config.clip_benchmark_cache_path,
                index_path=config.clip_benchmark_index_path,
                report_path=config.clip_benchmark_report_path,
                metrics_path=config.clip_benchmark_metrics_path,
            )
            print(
                f"Evaluated {metrics.query_count} Flickr8k caption queries; "
                f"Recall@1={metrics.recall_at_1:.4f}; "
                f"cache={'hit' if cache_hit else 'miss'}"
            )
        elif args.command == "ingest-hf-flickr8k":
            output_manifest = args.output_manifest or config.hf_flickr8k_manifest_path
            images_dir = args.images_dir or config.hf_flickr8k_images_path
            rows, provenance = ingest_hf_flickr8k(
                dataset_name=args.dataset_name,
                revision=args.revision,
                cache_dir=args.cache_dir,
                output_manifest=output_manifest,
                images_dir=images_dir,
                provenance_path=config.hf_flickr8k_provenance_path,
                max_images_per_split=args.max_images_per_split,
                local_files_only=args.local_files_only,
                force=args.force,
            )
            config.hf_flickr8k_report_path.parent.mkdir(parents=True, exist_ok=True)
            config.hf_flickr8k_report_path.write_text(
                render_hf_dataset_report(provenance), encoding="utf-8"
            )
            print(
                f"Materialized {provenance.unique_image_count} images and "
                f"{len(rows)} captions to {output_manifest}"
            )
        elif args.command == "inspect-hf-flickr8k":
            provenance_path = args.provenance or config.hf_flickr8k_provenance_path
            output = args.output or config.hf_flickr8k_report_path
            provenance = load_hf_provenance(provenance_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_hf_dataset_report(provenance), encoding="utf-8")
            print(
                f"HF Flickr8k: {provenance.unique_image_count} images, "
                f"{provenance.caption_count} captions; report={output}"
            )
        elif args.command == "evaluate-clip-flickr8k":
            manifest = args.manifest or config.hf_flickr8k_manifest_path
            provenance_path = args.provenance or config.hf_flickr8k_provenance_path
            raw_rows = read_manifest(manifest)
            rows = [row for row in raw_rows if isinstance(row, ManifestItemV2)]
            if len(rows) != len(raw_rows):
                raise ValueError("HF Flickr8k evaluation requires a schema-v2 manifest")
            provenance = load_hf_provenance(provenance_path)
            integration = args.max_images is not None
            cache_path = (
                config.hf_integration_cache_path if integration else config.hf_test_cache_path
            )
            index_path = (
                config.hf_integration_index_path if integration else config.hf_test_index_path
            )
            report_path = (
                config.hf_integration_report_path if integration else config.hf_test_report_path
            )
            metrics_path = (
                config.hf_integration_metrics_path if integration else config.hf_test_metrics_path
            )
            backend = ClipEmbeddingBackend(
                model_name=args.model_name,
                model_revision=args.model_revision,
                device=args.device,
                batch_size=args.batch_size,
                allow_download=not args.local_files_only,
            )
            result, cache_hit = run_hf_clip_benchmark(
                rows,
                provenance,
                backend,
                split=args.split,
                max_images=args.max_images,
                seed=args.seed,
                cache_path=cache_path,
                index_path=index_path,
                report_path=report_path,
                metrics_path=metrics_path,
            )
            print(
                f"HF Flickr8k {args.split}: "
                f"T2I R@1={result.text_to_image.metrics.recall_at_1:.4f}, "
                f"I2T R@1={result.image_to_text.metrics.recall_at_1:.4f}; "
                f"cache={'hit' if cache_hit else 'miss'}"
            )
        elif args.command == "prepare-adapter-embeddings":
            manifest_path = args.manifest or config.hf_flickr8k_manifest_path
            provenance_path = args.provenance or config.hf_flickr8k_provenance_path
            train_cache_path = args.train_cache or config.adapter_train_cache_path
            validation_cache_path = (
                args.validation_cache or config.adapter_validation_cache_path
            )
            rows = read_manifest(manifest_path)
            if not all(isinstance(row, ManifestItemV2) for row in rows):
                raise ManifestValidationError(
                    "adapter preparation requires a schema-v2 Flickr8k manifest"
                )
            provenance = load_hf_provenance(provenance_path)
            backend = ClipEmbeddingBackend(
                model_name=args.model_name,
                model_revision=args.model_revision,
                device=args.device,
                batch_size=64,
                allow_download=not args.local_files_only,
            )
            train_cache, train_hit = prepare_adapter_cache(
                rows,
                backend,
                provenance,
                split="train",
                image_count=args.train_images,
                seed=args.seed,
                cache_path=train_cache_path,
            )
            validation_cache, validation_hit = prepare_adapter_cache(
                rows,
                backend,
                provenance,
                split="validation",
                image_count=args.validation_images,
                seed=args.seed,
                cache_path=validation_cache_path,
            )
            print(
                "Prepared frozen adapter embeddings: "
                f"train={train_cache.metadata.image_count} images/"
                f"{train_cache.metadata.caption_count} captions "
                f"({'hit' if train_hit else 'created'}), "
                f"validation={validation_cache.metadata.image_count} images/"
                f"{validation_cache.metadata.caption_count} captions "
                f"({'hit' if validation_hit else 'created'})"
            )
        elif args.command == "train-contrastive-adapters":
            result = train_adapters(
                args.train_cache or config.adapter_train_cache_path,
                args.validation_cache or config.adapter_validation_cache_path,
                args.checkpoint or config.adapter_checkpoint_path,
                args.metadata_output or config.adapter_checkpoint_metadata_path,
                AdapterTrainingConfig(
                    seed=args.seed,
                    max_epochs=args.max_epochs,
                    early_stopping_patience=args.early_stopping_patience,
                    batch_size=args.batch_size,
                    device=args.device,
                ),
            )
            print(
                f"Trained {len(result.history)} epochs; selected epoch "
                f"{result.selected_epoch}; early-stopped={str(result.early_stopped).lower()}; "
                f"parameters={result.parameter_count}"
            )
        elif args.command == "evaluate-contrastive-adapters":
            train_cache_path = args.train_cache or config.adapter_train_cache_path
            validation_cache_path = (
                args.validation_cache or config.adapter_validation_cache_path
            )
            comparison, metadata = evaluate_adapters(
                train_cache_path,
                validation_cache_path,
                args.checkpoint or config.adapter_checkpoint_path,
                args.metadata or config.adapter_checkpoint_metadata_path,
                args.device,
            )
            train_cache = load_adapter_cache(train_cache_path)
            validation_cache = load_adapter_cache(validation_cache_path)
            write_adapter_reports(
                comparison,
                metadata,
                train_cache,
                validation_cache,
                args.report_output or config.adapter_training_report_path,
                args.metrics_output or config.adapter_metrics_path,
                args.promotion_output or config.adapter_promotion_path,
            )
            print(
                "Adapter validation complete: "
                f"mean-MRR-difference="
                f"{comparison.promotion.mean_bidirectional_mrr_difference:+.6f}; "
                f"decision={'promote' if comparison.promotion.promote else 'retain-zero-shot'}"
            )
        elif args.command == "contrastive-adapter-info":
            paths = {
                "checkpoint": args.checkpoint or config.adapter_checkpoint_path,
                "metadata": args.metadata or config.adapter_checkpoint_metadata_path,
                "train_cache": args.train_cache or config.adapter_train_cache_path,
                "validation_cache": (
                    args.validation_cache or config.adapter_validation_cache_path
                ),
            }
            information = {
                "torch_available": training_dependencies_available(),
                "dependency_message": (
                    "available"
                    if training_dependencies_available()
                    else training_dependency_message()
                ),
                "model_name": args.model_name,
                "model_revision": args.model_revision or "default",
                "device": args.device,
                "seed": args.seed,
                "local_files_only": args.local_files_only,
                "artifacts": {name: path.is_file() for name, path in paths.items()},
                "official_test_accessed": False,
            }
            print(json.dumps(information, indent=2, sort_keys=True))
        elif args.command == "analyze-contrastive-adapter":
            analysis = run_adapter_failure_analysis(
                train_cache_path=args.train_cache or config.adapter_train_cache_path,
                validation_cache_path=(
                    args.validation_cache or config.adapter_validation_cache_path
                ),
                checkpoint_path=args.checkpoint or config.adapter_checkpoint_path,
                checkpoint_metadata_path=(
                    args.metadata or config.adapter_checkpoint_metadata_path
                ),
                manifest_path=args.manifest or config.hf_flickr8k_manifest_path,
                recorded_metrics_path=args.recorded_metrics or config.adapter_metrics_path,
                device=args.device,
            )
            write_diagnostic_reports(
                analysis,
                args.report_output or config.adapter_failure_report_path,
                args.metrics_output or config.adapter_failure_metrics_path,
                args.memo_output or config.adapter_decision_memo_path,
            )
            conclusions = ", ".join(item.conclusion for item in analysis.diagnoses)
            print(
                "Validation-only adapter diagnostics complete; "
                f"decision=retain-zero-shot; conclusions={conclusions}"
            )
        elif args.command == "contrastive-adapter-diagnostics-info":
            diagnostic_paths = {
                "checkpoint": args.checkpoint or config.adapter_checkpoint_path,
                "checkpoint_metadata": (
                    args.metadata or config.adapter_checkpoint_metadata_path
                ),
                "manifest": args.manifest or config.hf_flickr8k_manifest_path,
                "recorded_metrics": args.recorded_metrics or config.adapter_metrics_path,
                "train_cache": args.train_cache or config.adapter_train_cache_path,
                "validation_cache": (
                    args.validation_cache or config.adapter_validation_cache_path
                ),
            }
            print(
                json.dumps(
                    {
                        "artifacts": {
                            name: path.is_file() for name, path in diagnostic_paths.items()
                        },
                        "analysis_boundary": "validation-only",
                        "clip_inference": False,
                        "embedding_generation": False,
                        "official_test_access": False,
                        "retraining": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "faiss-backend-info":
            available = faiss_available()
            message = f"FAISS CPU backend: {'available' if available else 'not installed'}"
            print(message)
            if not available:
                print(faiss_dependency_message())
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(message + "\n", encoding="utf-8")
        elif args.command == "build-faiss-flat-indexes":
            cache_path = args.cache or config.hf_test_cache_path
            artifacts_dir = args.artifacts_dir or config.faiss_artifacts_path
            text_artifact, image_artifact = build_flickr8k_flat_artifacts(
                cache_path, artifacts_dir
            )
            print(
                f"Built FAISS {text_artifact.metadata.index_type} indexes: "
                f"{text_artifact.metadata.candidate_count} image candidates and "
                f"{image_artifact.metadata.candidate_count} caption candidates"
            )
        elif args.command == "evaluate-faiss-flat":
            cache_path = args.cache or config.hf_test_cache_path
            manifest = args.manifest or config.hf_flickr8k_manifest_path
            artifacts_dir = args.artifacts_dir or config.faiss_artifacts_path
            cache, text_artifact, image_artifact = load_flickr8k_artifacts(
                cache_path, artifacts_dir
            )
            result = evaluate_flickr8k_faiss(
                cache, text_artifact, image_artifact, manifest
            )
            write_correctness_outputs(
                result,
                text_artifact.metadata,
                config.faiss_report_path,
                config.faiss_metrics_path,
            )
            print(
                "FAISS Flat correctness: "
                f"T2I={'pass' if result.text_to_image.correctness_gate_passed else 'fail'}, "
                f"I2T={'pass' if result.image_to_text.correctness_gate_passed else 'fail'}"
            )
        elif args.command == "search-faiss-text":
            cache_path = args.cache or config.hf_test_cache_path
            artifacts_dir = args.artifacts_dir or config.faiss_artifacts_path
            cache, text_artifact, _ = load_flickr8k_artifacts(cache_path, artifacts_dir)
            results = search_cached_embedding(
                args.query_caption_id,
                cache.caption_embeddings,
                text_artifact,
                args.k,
            )
            print(json.dumps(results, indent=2, sort_keys=True))
        elif args.command == "search-faiss-image":
            cache_path = args.cache or config.hf_test_cache_path
            artifacts_dir = args.artifacts_dir or config.faiss_artifacts_path
            cache, _, image_artifact = load_flickr8k_artifacts(cache_path, artifacts_dir)
            results = search_cached_embedding(
                args.query_image_id,
                cache.image_embeddings,
                image_artifact,
                args.k,
            )
            print(json.dumps(results, indent=2, sort_keys=True))
        elif args.command == "build-faiss-hnsw-indexes":
            cache_path = args.cache or config.hf_test_cache_path
            artifacts_dir = args.artifacts_dir or config.faiss_hnsw_artifacts_path
            text_artifact, image_artifact = build_flickr8k_hnsw_artifacts(
                cache_path, artifacts_dir
            )
            print(
                f"Built FAISS {text_artifact.metadata.index_type} indexes: "
                f"{text_artifact.metadata.candidate_count} image candidates and "
                f"{image_artifact.metadata.candidate_count} caption candidates; "
                f"M={text_artifact.metadata.m}, "
                f"efConstruction={text_artifact.metadata.ef_construction}"
            )
        elif args.command == "evaluate-faiss-hnsw":
            result, metadata = evaluate_flickr8k_hnsw(
                args.cache or config.hf_test_cache_path,
                args.manifest or config.hf_flickr8k_manifest_path,
                args.flat_artifacts_dir or config.faiss_artifacts_path,
                args.hnsw_artifacts_dir or config.faiss_hnsw_artifacts_path,
            )
            write_hnsw_outputs(
                result,
                metadata,
                config.faiss_hnsw_report_path,
                config.faiss_hnsw_metrics_path,
            )
            print(f"FAISS HNSW comparison complete; recommendation={result.recommendation}")
        elif args.command == "search-hnsw-text":
            cache, text_artifact, _ = load_flickr8k_hnsw_artifacts(
                args.cache or config.hf_test_cache_path,
                args.artifacts_dir or config.faiss_hnsw_artifacts_path,
            )
            print(
                json.dumps(
                    search_hnsw_embedding(
                        args.query_caption_id,
                        cache.caption_embeddings,
                        text_artifact,
                        args.k,
                        args.ef_search,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "search-hnsw-image":
            cache, _, image_artifact = load_flickr8k_hnsw_artifacts(
                args.cache or config.hf_test_cache_path,
                args.artifacts_dir or config.faiss_hnsw_artifacts_path,
            )
            print(
                json.dumps(
                    search_hnsw_embedding(
                        args.query_image_id,
                        cache.image_embeddings,
                        image_artifact,
                        args.k,
                        args.ef_search,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "evaluate-hnsw-reranking":
            result, metadata = evaluate_hnsw_reranking(
                args.cache or config.hf_test_cache_path,
                args.manifest or config.hf_flickr8k_manifest_path,
                args.flat_artifacts_dir or config.faiss_artifacts_path,
                args.hnsw_artifacts_dir or config.faiss_hnsw_artifacts_path,
                candidate_k=args.candidate_k,
                ef_search=args.ef_search,
            )
            write_reranking_outputs(
                result,
                metadata,
                config.hnsw_reranking_report_path,
                config.hnsw_reranking_metrics_path,
                config.hnsw_reranking_decision_path,
            )
            decision = "pass" if result.promotion_gate.approved else "fail"
            print(
                "HNSW exact reranking evaluation complete; "
                f"candidate_k={metadata.candidate_k}, efSearch={metadata.ef_search}, "
                f"promotion_gate={decision}"
            )
        elif args.command == "hnsw-reranking-info":
            payload = load_persisted_reranking_info(config.hnsw_reranking_metrics_path)
            metadata = payload["metadata"]
            gate = payload["promotion_gate"]
            print(
                json.dumps(
                    {
                        "artifact_compatibility": metadata["artifact_compatibility"],
                        "candidate_k": metadata["candidate_k"],
                        "ef_search": metadata["ef_search"],
                        "promotion_approved": gate["approved"],
                        "recommendation": gate["recommendation"],
                        "rejected_adapter_embeddings_used": metadata[
                            "rejected_adapter_embeddings_used"
                        ],
                        "run_state": payload["run_state"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command in {"search-reranked-text", "search-reranked-image"}:
            artifacts = load_reranking_artifacts(
                args.cache or config.hf_test_cache_path,
                args.flat_artifacts_dir or config.faiss_artifacts_path,
                args.hnsw_artifacts_dir or config.faiss_hnsw_artifacts_path,
            )
            if args.command == "search-reranked-text":
                results = search_reranked_embedding(
                    args.query_caption_id,
                    artifacts.cache.caption_embeddings,
                    artifacts.cache.image_embeddings,
                    artifacts.hnsw_text_to_image,
                    k=args.k,
                )
            else:
                results = search_reranked_embedding(
                    args.query_image_id,
                    artifacts.cache.image_embeddings,
                    artifacts.cache.caption_embeddings,
                    artifacts.hnsw_image_to_text,
                    k=args.k,
                )
            print(json.dumps(results, indent=2, sort_keys=True))
        elif args.command == "retrieval-telemetry-info":
            from .api.settings import ServiceSettings

            telemetry_path = args.telemetry_path or config.retrieval_telemetry_path
            settings = ServiceSettings(
                telemetry_enabled=args.enabled,
                telemetry_path=telemetry_path,
                telemetry_max_bytes=args.telemetry_max_bytes,
                telemetry_backup_count=args.telemetry_backup_count,
                telemetry_flush_each_event=args.telemetry_flush_each_event,
            )
            settings.validate()
            print(
                json.dumps(
                    {
                        "backup_count": settings.telemetry_backup_count,
                        "enabled": settings.telemetry_enabled,
                        "flush_each_event": settings.telemetry_flush_each_event,
                        "maximum_bytes": settings.telemetry_max_bytes,
                        "path_name": settings.telemetry_path.name,
                        "runtime_file_exists": settings.telemetry_path.is_file(),
                        "schema_version": 1,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "retrieval-telemetry-smoke":
            from .api.settings import ServiceSettings
            from .api.telemetry_smoke import run_telemetry_smoke

            settings = ServiceSettings(
                backend=args.backend,
                artifact_root=args.artifact_root,
                embedding_cache_path=args.embedding_cache or config.hf_test_cache_path,
                manifest_path=args.manifest or config.hf_flickr8k_manifest_path,
                ef_search=args.ef_search,
                telemetry_enabled=True,
                telemetry_path=args.telemetry_path or config.retrieval_telemetry_path,
                telemetry_max_bytes=args.telemetry_max_bytes,
                telemetry_backup_count=args.telemetry_backup_count,
                telemetry_flush_each_event=True,
            )
            result = run_telemetry_smoke(settings)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "analyze-retrieval-telemetry":
            thresholds = HealthThresholds(
                maximum_error_rate=args.maximum_error_rate,
                maximum_readiness_failures=args.maximum_readiness_failures,
                maximum_p95_latency_ms=args.maximum_p95_latency_ms,
                minimum_labeled_recall_at_10=args.minimum_labeled_recall_at_10,
                minimum_labeled_mrr=args.minimum_labeled_mrr,
            )
            read_result = read_telemetry(
                args.telemetry_path or config.retrieval_telemetry_path,
                include_rotated=args.include_rotated,
            )
            summary = analyze_events(read_result, thresholds)
            write_monitoring_outputs(
                summary,
                args.report_output or config.retrieval_monitoring_report_path,
                args.metrics_output or config.retrieval_monitoring_metrics_path,
                args.decision_output or config.retrieval_monitoring_decision_path,
            )
            print(
                "Retrieval telemetry analysis complete; "
                f"events={summary['traffic']['total_event_count']}, "
                f"health={summary['health']['decision']}"
            )
        elif args.command in {
            "serve-retrieval",
            "retrieval-service-info",
            "retrieval-service-smoke",
            "search-live-text",
            "search-live-image",
        }:
            from .api.settings import ServiceSettings

            settings = ServiceSettings(
                backend=args.backend,
                artifact_root=args.artifact_root,
                embedding_cache_path=args.embedding_cache or config.hf_test_cache_path,
                manifest_path=args.manifest or config.hf_flickr8k_manifest_path,
                ef_search=args.ef_search,
                maximum_top_k=args.maximum_top_k,
                host=args.host,
                port=args.port,
                enable_text_inference=(
                    args.enable_text_inference or args.command == "search-live-text"
                ),
                text_model_name=args.text_model_name,
                text_model_revision=args.text_model_revision,
                text_device=args.text_device,
                local_files_only=args.local_files_only,
                maximum_text_length=args.maximum_text_length,
                text_query_cache_size=args.text_query_cache_size,
                enable_image_inference=(
                    args.enable_image_inference or args.command == "search-live-image"
                ),
                image_model_name=args.image_model_name,
                image_model_revision=args.image_model_revision,
                image_device=args.image_device,
                maximum_upload_bytes=args.maximum_upload_bytes,
                maximum_pixel_count=args.maximum_pixel_count,
                image_query_cache_size=args.image_query_cache_size,
                smoke_image_path=(
                    args.smoke_image_path or ServiceSettings.smoke_image_path
                ),
                telemetry_enabled=args.enable_telemetry,
                telemetry_path=args.telemetry_path or config.retrieval_telemetry_path,
                telemetry_max_bytes=args.telemetry_max_bytes,
                telemetry_backup_count=args.telemetry_backup_count,
                telemetry_flush_each_event=args.telemetry_flush_each_event,
            )
            if args.command == "retrieval-service-info":
                from .api.artifacts import ServiceArtifactError, load_service_artifacts

                if settings.enable_text_inference or settings.enable_image_inference:
                    from fastapi.testclient import TestClient

                    from .api.app import create_app

                    with TestClient(
                        create_app(
                            settings,
                            text_encoder_factory=text_encoder_factory,
                            image_encoder_factory=image_encoder_factory,
                        )
                    ) as client:
                        information = client.get("/ready").json()
                        if information["retrieval_artifacts_ready"]:
                            information["index_info"] = client.get("/index-info").json()
                else:
                    try:
                        artifacts = load_service_artifacts(settings)
                        metadata = artifacts.text_to_image.metadata
                        information = {
                            "artifact_validation": "passed",
                            "backend": artifacts.backend,
                            "caption_candidate_count": (
                                artifacts.image_to_text.metadata.candidate_count
                            ),
                            "dataset_fingerprint": metadata.dataset_fingerprint,
                            "ef_search": artifacts.ef_search,
                            "embedding_dimension": metadata.embedding_dimension,
                            "faiss_version": metadata.faiss_version,
                            "image_candidate_count": metadata.candidate_count,
                            "index_type": metadata.index_type,
                            "model_name": metadata.model_name,
                            "model_revision": metadata.model_revision,
                            "ready": True,
                            "reasons": [],
                            "split": metadata.split,
                        }
                    except ServiceArtifactError as error:
                        information = {
                            "artifact_validation": error.state,
                            "backend": settings.backend,
                            "ready": False,
                            "reasons": [error.reason],
                        }
                print(json.dumps(information, indent=2, sort_keys=True))
            else:
                from .api.reporting import (
                    failure_result,
                    serving_dependencies_available,
                    serving_dependency_message,
                    write_service_reports,
                )

                if not serving_dependencies_available():
                    detail = serving_dependency_message()
                    if args.command == "retrieval-service-smoke":
                        failure = failure_result(settings, "dependency_unavailable", detail)
                        if settings.enable_image_inference:
                            from .api.image_reporting import write_image_inference_reports

                            write_image_inference_reports(
                                failure,
                                settings,
                                args.report_output or config.image_inference_service_report_path,
                                args.metrics_output or config.image_inference_service_metrics_path,
                            )
                        elif settings.enable_text_inference:
                            from .api.text_reporting import write_text_inference_reports

                            write_text_inference_reports(
                                failure,
                                settings,
                                args.report_output or config.text_inference_service_report_path,
                                args.metrics_output or config.text_inference_service_metrics_path,
                            )
                        else:
                            write_service_reports(
                                failure,
                                args.report_output or config.retrieval_service_report_path,
                                args.metrics_output or config.retrieval_service_metrics_path,
                            )
                    build_parser().error(detail)
                if args.command == "serve-retrieval":
                    import uvicorn

                    from .api.app import create_app

                    uvicorn.run(
                        create_app(
                            settings,
                            text_encoder_factory=text_encoder_factory,
                            image_encoder_factory=image_encoder_factory,
                        ),
                        host=settings.host,
                        port=settings.port,
                    )
                elif args.command == "search-live-text":
                    from .api.smoke import run_live_text_query

                    live_result = run_live_text_query(
                        settings,
                        args.query,
                        args.top_k,
                        text_encoder_factory=text_encoder_factory,
                    )
                    if live_result["status_code"] != 200:
                        build_parser().error(
                            json.dumps(live_result["response"], sort_keys=True)
                        )
                    print(json.dumps(live_result["response"], indent=2, sort_keys=True))
                elif args.command == "search-live-image":
                    from .api.smoke import run_live_image_query

                    live_result = run_live_image_query(
                        settings,
                        args.image_path,
                        args.top_k,
                        image_encoder_factory=image_encoder_factory,
                    )
                    if live_result["status_code"] != 200:
                        build_parser().error(
                            json.dumps(live_result["response"], sort_keys=True)
                        )
                    print(json.dumps(live_result["response"], indent=2, sort_keys=True))
                else:
                    from .api.smoke import run_service_smoke

                    result = run_service_smoke(
                        settings,
                        text_encoder_factory=text_encoder_factory,
                        image_encoder_factory=image_encoder_factory,
                    )
                    if settings.enable_image_inference:
                        from .api.image_reporting import write_image_inference_reports

                        write_image_inference_reports(
                            result,
                            settings,
                            args.report_output or config.image_inference_service_report_path,
                            args.metrics_output or config.image_inference_service_metrics_path,
                        )
                    elif settings.enable_text_inference:
                        from .api.text_reporting import write_text_inference_reports

                        write_text_inference_reports(
                            result,
                            settings,
                            args.report_output or config.text_inference_service_report_path,
                            args.metrics_output or config.text_inference_service_metrics_path,
                        )
                    else:
                        write_service_reports(
                            result,
                            args.report_output or config.retrieval_service_report_path,
                            args.metrics_output or config.retrieval_service_metrics_path,
                        )
                    if result.run_state != "success":
                        build_parser().error(result.detail)
                    if settings.enable_image_inference:
                        print(
                            f"Image inference smoke passed for {result.backend}; "
                            "requests=2, encoder-invocations=1, cache-hits=1"
                        )
                    elif settings.enable_text_inference:
                        print(
                            f"Text inference smoke passed for {result.backend}; "
                            "requests=2, encoder-invocations=1, cache-hits=1"
                        )
                    else:
                        print(
                            f"Retrieval service smoke passed for {result.backend}; "
                            "caption-to-image=1, image-to-caption=1"
                        )
    except AdapterDiagnosticError as error:
        if isinstance(error, DiagnosticCheckpointUnavailableError):
            state = "checkpoint_unavailable"
        elif isinstance(error, DiagnosticArtifactUnavailableError):
            state = "artifact_unavailable"
        elif isinstance(error, DiagnosticArtifactIncompatibleError):
            state = "artifact_incompatible"
        else:
            state = "execution_failed"
        write_diagnostic_failure_reports(
            state,
            str(error),
            config.adapter_failure_report_path,
            config.adapter_failure_metrics_path,
            config.adapter_decision_memo_path,
        )
        build_parser().error(str(error))
    except ContrastiveAdapterError as error:
        if isinstance(error, AdapterDependencyError):
            state = "dependency_unavailable"
        elif isinstance(error, AdapterDatasetUnavailableError):
            state = "dataset_unavailable"
        elif isinstance(error, AdapterCacheIncompatibleError):
            state = "cache_incompatible"
        elif isinstance(error, AdapterTrainingError):
            state = "training_failed"
        elif isinstance(error, AdapterEvaluationError):
            state = "evaluation_failed"
        else:
            state = (
                "evaluation_failed"
                if args.command == "evaluate-contrastive-adapters"
                else "training_failed"
            )
        write_adapter_failure_reports(
            state,
            str(error),
            config.adapter_training_report_path,
            config.adapter_metrics_path,
            config.adapter_promotion_path,
        )
        build_parser().error(str(error))
    except FaissFlatError as error:
        if isinstance(error, FaissDependencyError):
            state = "dependency_unavailable"
        elif isinstance(error, FaissIndexStaleError):
            state = "cache_incompatible"
        elif isinstance(error, FaissCacheError):
            state = "cache_unavailable" if "missing" in str(error) else "cache_incompatible"
        else:
            state = "execution_failed"
        if args.command == "evaluate-hnsw-reranking":
            if isinstance(error, RerankingArtifactUnavailableError):
                state = "artifact_unavailable"
            elif isinstance(error, RerankingArtifactIncompatibleError):
                state = "artifact_incompatible"
            write_reranking_failure(
                config.hnsw_reranking_report_path,
                config.hnsw_reranking_metrics_path,
                config.hnsw_reranking_decision_path,
                state,
                str(error),
            )
        elif args.command.startswith(("build-faiss-hnsw", "evaluate-faiss-hnsw", "search-hnsw")):
            write_hnsw_failure(
                config.faiss_hnsw_report_path,
                config.faiss_hnsw_metrics_path,
                state,
                str(error),
            )
        elif args.command != "faiss-backend-info":
            write_faiss_failure(config.faiss_report_path, config.faiss_metrics_path, state, str(error))
        build_parser().error(str(error))
    except HFFlickr8kError as error:
        if isinstance(error, HFDatasetUnavailableError):
            state = "dataset_unavailable"
        elif isinstance(error, HFDataExecutionError):
            state = "execution_failed"
        else:
            state = "dependency_unavailable"
        write_hf_failure_report(config.hf_flickr8k_report_path, state, str(error))
        build_parser().error(str(error))
    except ClipBackendError as error:
        if isinstance(error, ClipModelUnavailableError):
            status = "unavailable model weights"
        elif isinstance(error, ClipExecutionError):
            status = "execution failure"
        else:
            status = "unavailable dependencies"
        write_clip_backend_report(
            config.clip_backend_report_path,
            status=status,
            model_name=getattr(args, "model_name", None) or DEFAULT_CLIP_MODEL,
            device=getattr(args, "device", "cpu"),
            detail=str(error),
        )
        if args.command == "prepare-adapter-embeddings":
            if isinstance(error, ClipModelUnavailableError):
                adapter_state = "model_unavailable"
            elif isinstance(error, ClipExecutionError):
                adapter_state = "training_failed"
            else:
                adapter_state = "dependency_unavailable"
            write_adapter_failure_reports(
                adapter_state,
                str(error),
                config.adapter_training_report_path,
                config.adapter_metrics_path,
                config.adapter_promotion_path,
            )
        if args.command == "evaluate-clip":
            write_clip_failure_reports(
                status,
                str(error),
                getattr(args, "report_output", None) or config.clip_report_path,
                getattr(args, "metrics_output", None) or config.clip_metrics_path,
            )
        elif args.command == "evaluate-clip-flickr8k":
            integration = getattr(args, "max_images", None) is not None
            if isinstance(error, ClipModelUnavailableError):
                hf_state = "model_unavailable"
            elif isinstance(error, ClipExecutionError):
                hf_state = "execution_failed"
            else:
                hf_state = "dependency_unavailable"
            write_hf_clip_failure(
                config.hf_integration_report_path if integration else config.hf_test_report_path,
                config.hf_integration_metrics_path if integration else config.hf_test_metrics_path,
                hf_state,
                str(error),
            )
        build_parser().error(str(error))
    except RetrievalMonitoringError as error:
        if isinstance(error, TelemetryUnavailableError):
            state = "telemetry_unavailable"
        elif args.command == "retrieval-telemetry-smoke" and "artifacts" in str(error):
            state = "artifact_unavailable"
        elif args.command == "analyze-retrieval-telemetry":
            state = "telemetry_invalid"
        else:
            state = "execution_failed"
        if args.command in {
            "retrieval-telemetry-smoke",
            "analyze-retrieval-telemetry",
        }:
            write_monitoring_failure(
                state,
                str(error),
                config.retrieval_monitoring_report_path,
                config.retrieval_monitoring_metrics_path,
                config.retrieval_monitoring_decision_path,
            )
        build_parser().error(str(error))
    except (ManifestValidationError, ValueError) as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
