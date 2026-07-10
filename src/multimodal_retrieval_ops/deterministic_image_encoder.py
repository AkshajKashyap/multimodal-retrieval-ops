"""Deterministic file-feature image encoder placeholder."""

from dataclasses import dataclass
from pathlib import Path

from .embedding_backends import normalized_hashed_vector
from .text_baseline import tokenize


@dataclass(frozen=True)
class DeterministicImageEncoder:
    """Encode path and file-byte tokens into the shared hashing space."""

    dimension: int = 64
    backend_name: str = "deterministic-file-features"
    backend_version: str = "1"

    def encode_image(self, image_path: str) -> list[float]:
        path = Path(image_path)
        if not path.is_file():
            raise ValueError(f"image does not exist: {image_path}")
        byte_text = path.read_bytes().decode("utf-8", errors="ignore")
        features = [*tokenize(path.stem), *tokenize(byte_text)]
        return normalized_hashed_vector(features, self.dimension)
