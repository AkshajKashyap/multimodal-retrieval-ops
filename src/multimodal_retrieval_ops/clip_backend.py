"""Optional Hugging Face CLIP backend with lazy heavyweight imports."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
REQUIRED_CLIP_MODULES = ("torch", "transformers", "PIL")


class ClipBackendError(ValueError):
    """Actionable error raised when the optional backend cannot run."""


class ClipModelUnavailableError(ClipBackendError):
    """Raised when dependencies exist but requested weights cannot be loaded."""


class ClipExecutionError(ClipBackendError):
    """Raised when model execution fails after weights are available."""


def clip_dependencies_available() -> bool:
    """Return whether all optional CLIP packages are importable in principle."""
    return all(importlib.util.find_spec(module) is not None for module in REQUIRED_CLIP_MODULES)


def clip_dependency_message() -> str:
    return (
        "CLIP dependencies are unavailable. Install them with "
        '`python -m pip install -e ".[dev,clip]"`.'
    )


@dataclass
class ClipEmbeddingBackend:
    """Lazily loaded, normalized Hugging Face CLIP text/image encoder."""

    model_name: str = DEFAULT_CLIP_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    batch_size: int = 8
    allow_download: bool = False
    backend_name: str = "huggingface-clip"
    backend_version: str = "1"
    dimension: int = 0
    _torch: Any = None
    _processor: Any = None
    _model: Any = None
    _image_class: Any = None

    def metadata(self) -> dict[str, str | int]:
        """Return backend configuration without loading model weights."""
        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision or "default",
            "device": self.device,
            "batch_size": self.batch_size,
            "dimension": self.dimension,
        }

    def ensure_loaded(self) -> None:
        """Load optional dependencies and local model weights on first use."""
        if self._model is not None:
            return
        if not clip_dependencies_available():
            raise ClipBackendError(clip_dependency_message())
        if self.batch_size <= 0:
            raise ClipBackendError("CLIP batch size must be positive")
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except Exception as error:
            raise ClipBackendError(clip_dependency_message()) from error
        try:
            processor = CLIPProcessor.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                local_files_only=not self.allow_download,
            )
            model = CLIPModel.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                local_files_only=not self.allow_download,
            )
        except Exception as error:
            download_hint = " Retry without --local-files-only." if not self.allow_download else ""
            raise ClipModelUnavailableError(
                f"Could not load CLIP model weights for '{self.model_name}'."
                f" Ensure the model is accessible or cached locally.{download_hint}"
            ) from error
        try:
            model.to(self.device)
            model.eval()
        except Exception as error:
            raise ClipExecutionError(
                f"CLIP model '{self.model_name}' could not initialize on device '{self.device}'."
            ) from error
        self._torch = torch
        self._processor = processor
        self._model = model
        self._image_class = Image
        self.dimension = int(model.config.projection_dim)

    def _normalize_batch(self, tensor: Any) -> list[list[float]]:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return [
            [float(value) for value in row]
            for row in tensor.detach().cpu().tolist()
        ]

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode text in configured batches."""
        self.ensure_loaded()
        embeddings: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self._processor(text=batch, return_tensors="pt", padding=True)
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                with self._torch.inference_mode():
                    output = self._model.text_model(**inputs)
                    features = self._model.text_projection(output.pooler_output)
                embeddings.extend(self._normalize_batch(features))
        except Exception as error:
            raise ClipExecutionError("CLIP text embedding execution failed") from error
        return embeddings

    def encode_text(self, text: str) -> list[float]:
        return self.encode_texts([text])[0]

    def encode_images(self, image_paths: list[str]) -> list[list[float]]:
        """Decode and encode images in configured batches."""
        self.ensure_loaded()
        embeddings: list[list[float]] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = []
            for image_path in batch_paths:
                path = Path(image_path)
                if not path.is_file():
                    raise ClipBackendError(f"image does not exist: {image_path}")
                try:
                    with self._image_class.open(path) as image:
                        images.append(image.convert("RGB").copy())
                except Exception as error:
                    raise ClipExecutionError(f"could not decode image: {image_path}") from error
            try:
                inputs = self._processor(images=images, return_tensors="pt")
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                with self._torch.inference_mode():
                    output = self._model.vision_model(**inputs)
                    features = self._model.visual_projection(output.pooler_output)
                embeddings.extend(self._normalize_batch(features))
            except Exception as error:
                raise ClipExecutionError("CLIP image embedding execution failed") from error
        return embeddings

    def encode_image(self, image_path: str) -> list[float]:
        return self.encode_images([image_path])[0]
