"""Command-line interface for project foundation workflows."""

import argparse
from pathlib import Path

from . import __version__
from .config import load_config
from .demo import generate_demo_manifest
from .ingestion import ingest_local_directory
from .inspection import inspect_items, write_dataset_report
from .manifest import (
    ManifestValidationError,
    read_manifest,
    read_manifest_rows,
    validate_image_paths,
    write_manifest,
)
from .reporting import write_manifest_report
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    try:
        if args.command == "project-info":
            print(f"multimodal-retrieval-ops {__version__}")
            print("Milestone: 2 (dataset ingestion and local image-caption registry)")
            print("Runtime: CPU-only; standard library")
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
    except (ManifestValidationError, ValueError) as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
