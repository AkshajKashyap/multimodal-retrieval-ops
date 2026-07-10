"""Command-line interface for project foundation workflows."""

import argparse
from pathlib import Path

from . import __version__
from .config import load_config
from .demo import generate_demo_manifest
from .manifest import ManifestValidationError, read_manifest
from .reporting import write_manifest_report


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    try:
        if args.command == "project-info":
            print(f"multimodal-retrieval-ops {__version__}")
            print("Milestone: 1 (project foundation and manifest validation)")
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
    except ManifestValidationError as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
