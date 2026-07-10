"""Optional Hugging Face CLIP backend with lazy heavyweight imports."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
REQUIRED_CLIP_MODULES = ("torch", "transformers", "PIL")


class ClipBackendError(ValueError):
    """Actionable error raised when the optional backend cannot run."""


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

            processor = CLIPProcessor.from_pretrained(
                self.model_name, local_files_only=not self.allow_download
            )
            model = CLIPModel.from_pretrained(
                self.model_name, local_files_only=not self.allow_download
            )
            model.to(self.device)
            model.eval()
        except Exception as error:
            download_hint = " Retry with --allow-download" if not self.allow_download else ""
            raise ClipBackendError(
                f"Could not load CLIP model '{self.model_name}' on {self.device}."
                f" Ensure its weights are cached locally or choose another model.{download_hint}"
            ) from error
        self._torch = torch
        self._processor = processor
        self._model = model
        self._image_class = Image
        self.dimension = int(model.config.projection_dim)

    def _normalize(self, tensor: Any) -> list[float]:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return [float(value) for value in tensor[0].detach().cpu().tolist()]

    def encode_text(self, text: str) -> list[float]:
        self.ensure_loaded()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        return self._normalize(features)

    def encode_image(self, image_path: str) -> list[float]:
        self.ensure_loaded()
        path = Path(image_path)
        if not path.is_file():
            raise ClipBackendError(f"image does not exist: {image_path}")
        try:
            with self._image_class.open(path) as image:
                inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        except Exception as error:
            raise ClipBackendError(f"could not decode image: {image_path}") from error
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        return self._normalize(features)
