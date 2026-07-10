"""Command-line interface for project foundation workflows."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from . import __version__
from .baseline_index import build_index, exact_search, load_index, write_index
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
from .demo import generate_demo_manifest
from .deterministic_image_encoder import DeterministicImageEncoder
from .deterministic_text_encoder import DeterministicTextEncoder
from .ingestion import ingest_local_directory
from .inspection import inspect_items, write_dataset_report
from .manifest import (
    ManifestValidationError,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    try:
        if args.command == "project-info":
            print(f"multimodal-retrieval-ops {__version__}")
            print("Milestone: 5.5 (verified CLIP retrieval and embedding cache)")
            print("Runtime: lightweight base install; optional CPU/GPU CLIP extra")
        elif args.command == "generate-demo-manifest":
            output = args.output or config.manifest_path
            items = generate_demo_manifest(output)
            print(f"Wrote {len(items)} rows to {output}")
        elif args.command == "validate-manifest":
            manifest = args.manifest or config.manifest_path
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
        if args.command == "evaluate-clip":
            write_clip_failure_reports(
                status,
                str(error),
                getattr(args, "report_output", None) or config.clip_report_path,
                getattr(args, "metrics_output", None) or config.clip_metrics_path,
            )
        build_parser().error(str(error))
    except (ManifestValidationError, ValueError) as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
