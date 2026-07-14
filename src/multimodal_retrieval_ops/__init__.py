"""Foundational tools for reproducible multimodal retrieval projects."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multimodal-retrieval-ops")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "1.0.0"
