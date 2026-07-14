"""Create temporary, synthetic-only artifacts for the Docker smoke."""

import argparse
from pathlib import Path

from multimodal_retrieval_ops.portfolio_release import prepare_synthetic_service_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare_synthetic_service_artifacts(args.root)


if __name__ == "__main__":
    main()
