"""Typed settings for the local retrieval service."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..faiss_hnsw import ALLOWED_EF_SEARCH

ServiceBackend = Literal["flat", "hnsw"]


@dataclass(frozen=True)
class ServiceSettings:
    """Paths and bounded runtime choices for serving persisted artifacts."""

    backend: ServiceBackend = "flat"
    artifact_root: Path = Path("artifacts")
    embedding_cache_path: Path = Path("artifacts/clip/hf_flickr8k_test_cache.json")
    manifest_path: Path = Path("data/processed/hf_flickr8k_manifest_v2.csv")
    ef_search: int = 64
    maximum_top_k: int = 100
    host: str = "127.0.0.1"
    port: int = 8000

    def validate(self) -> None:
        if self.backend not in ("flat", "hnsw"):
            raise ValueError("backend must be 'flat' or 'hnsw'")
        if self.backend == "hnsw" and self.ef_search not in ALLOWED_EF_SEARCH:
            allowed = ", ".join(str(value) for value in ALLOWED_EF_SEARCH)
            raise ValueError(f"HNSW efSearch must be one of: {allowed}")
        if self.maximum_top_k <= 0:
            raise ValueError("maximum_top_k must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @property
    def index_artifacts_path(self) -> Path:
        return self.artifact_root / ("faiss" if self.backend == "flat" else "faiss_hnsw")
