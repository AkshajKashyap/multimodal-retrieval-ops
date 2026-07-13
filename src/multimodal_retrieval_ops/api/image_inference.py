"""Strict in-memory image validation and injectable vision encoder contracts."""

from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
import hashlib
from io import BytesIO
from typing import Any, Protocol
import warnings

from PIL import Image, UnidentifiedImageError

from ..clip_backend import ClipEmbeddingBackend
from .settings import ServiceSettings
from .text_inference import QueryEmbeddingCache, TextEncoder

CONTENT_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
MULTIPART_OVERHEAD_LIMIT = 64 * 1024


class ImageValidationError(ValueError):
    pass


class ImageEncoder(Protocol):
    model_name: str
    model_revision: str | None
    backend_name: str
    backend_version: str
    dimension: int

    def ensure_loaded(self) -> None: ...

    def encode_image_object(self, image: Any) -> list[float]: ...


class ImageEmbeddingCache(QueryEmbeddingCache):
    pass


@dataclass(frozen=True)
class ParsedImageUpload:
    image_bytes: bytes
    content_type: str
    top_k: int


def create_default_image_encoder(
    settings: ServiceSettings, shared_encoder: TextEncoder | None = None
) -> ImageEncoder:
    if (
        shared_encoder is not None
        and hasattr(shared_encoder, "encode_image_object")
        and shared_encoder.model_name == settings.image_model_name
        and (shared_encoder.model_revision or "default")
        == (settings.image_model_revision or "default")
    ):
        return shared_encoder  # type: ignore[return-value]
    return ClipEmbeddingBackend(
        model_name=settings.image_model_name,
        model_revision=settings.image_model_revision,
        device=settings.image_device,
        batch_size=1,
        allow_download=not settings.local_files_only,
    )


async def parse_multipart_image(request: Any, maximum_upload_bytes: int) -> ParsedImageUpload:
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ImageValidationError("content type must be multipart/form-data")
    content_length = request.headers.get("content-length")
    maximum_body = maximum_upload_bytes + MULTIPART_OVERHEAD_LIMIT
    if content_length:
        try:
            if int(content_length) > maximum_body:
                raise ImageValidationError(
                    "uploaded image exceeds the configured byte limit"
                )
        except ValueError as error:
            raise ImageValidationError("content-length must be an integer") from error
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > maximum_body:
            raise ImageValidationError("uploaded image exceeds the configured byte limit")
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii", "ignore") + b"\r\n\r\n" + bytes(body)
    )
    image_bytes: bytes | None = None
    image_content_type = ""
    top_k: int | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name == "image":
            image_bytes = part.get_payload(decode=True) or b""
            image_content_type = part.get_content_type().lower()
        elif name == "top_k":
            try:
                top_k = int(part.get_content().strip())
            except (TypeError, ValueError) as error:
                raise ImageValidationError("top_k must be an integer") from error
    if image_bytes is None:
        raise ImageValidationError("multipart field 'image' is required")
    if top_k is None:
        raise ImageValidationError("multipart field 'top_k' is required")
    if not image_bytes:
        raise ImageValidationError("uploaded image must be non-empty")
    if len(image_bytes) > maximum_upload_bytes:
        raise ImageValidationError("uploaded image exceeds the configured byte limit")
    return ParsedImageUpload(image_bytes, image_content_type, top_k)


def decode_and_validate_image(
    upload: ParsedImageUpload, settings: ServiceSettings
) -> tuple[Any, str]:
    allowed_content_types = {CONTENT_TYPES[value] for value in settings.allowed_image_formats}
    if upload.content_type not in allowed_content_types:
        raise ImageValidationError("uploaded image content type is unsupported")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(upload.image_bytes)) as source:
                actual_format = (source.format or "").upper()
                width, height = source.size
                if actual_format not in settings.allowed_image_formats:
                    raise ImageValidationError("decoded image format is unsupported")
                if CONTENT_TYPES[actual_format] != upload.content_type:
                    raise ImageValidationError(
                        "uploaded content type does not match decoded image format"
                    )
                if width * height > settings.maximum_pixel_count:
                    raise ImageValidationError("decoded image exceeds the configured pixel limit")
                source.verify()
            with Image.open(BytesIO(upload.image_bytes)) as source:
                image = source.convert("RGB").copy()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageValidationError("decoded image exceeds safe Pillow limits") from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise ImageValidationError("uploaded image is corrupt or undecodable") from error
    return image, actual_format


def image_cache_identity(image_bytes: bytes, encoder: ImageEncoder) -> tuple[str, str]:
    digest = hashlib.sha256(image_bytes).hexdigest()
    identity = ":".join(
        (
            digest,
            encoder.backend_name,
            encoder.backend_version,
            encoder.model_name,
            encoder.model_revision or "default",
        )
    )
    return digest[:16], identity
