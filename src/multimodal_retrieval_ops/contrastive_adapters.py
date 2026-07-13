"""Bounded contrastive adapters trained only over frozen CLIP embeddings."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .embedding_cache import manifest_digest
from .evaluation import RetrievalMetrics
from .hf_clip_benchmark import BidirectionalResult, evaluate_bidirectional_vectorized
from .hf_flickr8k import HFFlickr8kProvenance
from .manifest import ManifestItemV2

MAX_TRAIN_IMAGES = 500
MAX_VALIDATION_IMAGES = 100
MAX_TRAIN_CAPTIONS = 2500
MAX_VALIDATION_CAPTIONS = 500
MAX_EPOCHS = 20
MAX_PATIENCE = 4
MAX_BATCH_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_BOTTLENECK_DIMENSION = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_TEMPERATURE = 0.07
ADAPTER_ARCHITECTURE = "two-layer-residual-gelu-l2-v1"


class ContrastiveAdapterError(ValueError):
    """Base expected error for the bounded adapter workflow."""


class AdapterDependencyError(ContrastiveAdapterError):
    """Optional Torch dependency is unavailable."""


class AdapterDatasetUnavailableError(ContrastiveAdapterError):
    """Required local train/validation data is unavailable."""


class AdapterCacheIncompatibleError(ContrastiveAdapterError):
    """A frozen embedding cache does not match the requested experiment."""


class AdapterTrainingError(ContrastiveAdapterError):
    """The fixed adapter training run failed."""


class AdapterEvaluationError(ContrastiveAdapterError):
    """The validation-only evaluation failed."""


def training_dependencies_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def training_dependency_message() -> str:
    return (
        "Adapter training requires optional Torch support. Install it with "
        '`python -m pip install -e ".[dev,train]"`.'
    )


def require_torch() -> Any:
    if not training_dependencies_available():
        raise AdapterDependencyError(training_dependency_message())
    try:
        import torch
    except Exception as error:
        raise AdapterDependencyError(training_dependency_message()) from error
    return torch


def _split_limit(split: str) -> tuple[int, int]:
    if split == "train":
        return MAX_TRAIN_IMAGES, MAX_TRAIN_CAPTIONS
    if split == "validation":
        return MAX_VALIDATION_IMAGES, MAX_VALIDATION_CAPTIONS
    raise ContrastiveAdapterError(
        "adapter experiments may access only official train and validation splits"
    )


def select_grouped_subset(
    rows: list[ManifestItemV2], split: str, image_count: int, seed: int
) -> list[ManifestItemV2]:
    """Select seeded image groups while retaining every caption for each image."""
    maximum_images, maximum_captions = _split_limit(split)
    if image_count <= 0 or image_count > maximum_images:
        raise ContrastiveAdapterError(
            f"{split} image count must be between 1 and {maximum_images}"
        )
    split_rows = [row for row in rows if row.split == split]
    available_ids = sorted({row.image_id for row in split_rows})
    if len(available_ids) < image_count:
        raise AdapterDatasetUnavailableError(
            f"official {split} split has only {len(available_ids)} available images"
        )
    random.Random(seed).shuffle(available_ids)
    selected_ids = set(available_ids[:image_count])
    selected = sorted(
        (row for row in split_rows if row.image_id in selected_ids),
        key=lambda row: (row.image_id, row.caption_id),
    )
    if len(selected) > maximum_captions:
        raise ContrastiveAdapterError(
            f"selected {split} subset exceeds the {maximum_captions}-caption limit"
        )
    if {row.split for row in selected} != {split}:
        raise ContrastiveAdapterError("official split preservation failed")
    return selected


def subset_fingerprint(rows: list[ManifestItemV2], selected_image_ids: list[str]) -> str:
    identity = "\n".join(selected_image_ids) + "\n" + manifest_digest(rows)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdapterCacheMetadata:
    backend_name: str
    backend_version: str
    preprocessing_identity: str
    model_name: str
    model_revision: str
    dataset_fingerprint: str
    dataset_revision: str
    manifest_fingerprint: str
    subset_fingerprint: str
    split: str
    requested_image_count: int
    seed: int
    selected_image_ids: list[str]
    image_count: int
    caption_count: int
    embedding_dimension: int


@dataclass(frozen=True)
class AdapterEmbeddingCache:
    metadata: AdapterCacheMetadata
    image_embeddings: dict[str, list[float]]
    caption_embeddings: dict[str, list[float]]
    caption_image_ids: dict[str, str]


def make_adapter_cache_metadata(
    rows: list[ManifestItemV2],
    backend: Any,
    provenance: HFFlickr8kProvenance,
    *,
    split: str,
    requested_image_count: int,
    seed: int,
    dimension: int,
) -> AdapterCacheMetadata:
    _split_limit(split)
    image_ids = sorted({row.image_id for row in rows})
    return AdapterCacheMetadata(
        backend_name=backend.backend_name,
        backend_version=backend.backend_version,
        preprocessing_identity=(
            f"{backend.backend_name}:{backend.backend_version}:"
            f"{backend.model_name}:{backend.model_revision or 'default'}:clip-processor"
        ),
        model_name=backend.model_name,
        model_revision=backend.model_revision or "default",
        dataset_fingerprint=provenance.resolved_fingerprint,
        dataset_revision=provenance.resolved_revision,
        manifest_fingerprint=manifest_digest(rows),
        subset_fingerprint=subset_fingerprint(rows, image_ids),
        split=split,
        requested_image_count=requested_image_count,
        seed=seed,
        selected_image_ids=image_ids,
        image_count=len(image_ids),
        caption_count=len(rows),
        embedding_dimension=dimension,
    )


def _validate_normalized_vectors(
    vectors: dict[str, list[float]], dimension: int, label: str
) -> None:
    for vector in vectors.values():
        if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
            raise AdapterCacheIncompatibleError(
                f"{label} embedding dimensions or values are incompatible"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, abs_tol=1e-4):
            raise AdapterCacheIncompatibleError(f"{label} embeddings are not L2-normalized")


def validate_adapter_cache(cache: AdapterEmbeddingCache) -> None:
    metadata = cache.metadata
    _split_limit(metadata.split)
    if metadata.selected_image_ids != sorted(metadata.selected_image_ids):
        raise AdapterCacheIncompatibleError("selected image IDs are not canonical")
    if set(cache.image_embeddings) != set(metadata.selected_image_ids):
        raise AdapterCacheIncompatibleError("image embedding IDs do not match cache metadata")
    if set(cache.caption_embeddings) != set(cache.caption_image_ids):
        raise AdapterCacheIncompatibleError("caption embedding relationships are incomplete")
    if not set(cache.caption_image_ids.values()) <= set(cache.image_embeddings):
        raise AdapterCacheIncompatibleError("caption relationship references an unknown image")
    if len(cache.image_embeddings) != metadata.image_count:
        raise AdapterCacheIncompatibleError("image count does not match cache metadata")
    if len(cache.caption_embeddings) != metadata.caption_count:
        raise AdapterCacheIncompatibleError("caption count does not match cache metadata")
    _validate_normalized_vectors(
        cache.image_embeddings, metadata.embedding_dimension, "image"
    )
    _validate_normalized_vectors(
        cache.caption_embeddings, metadata.embedding_dimension, "caption"
    )


def adapter_cache_is_stale(
    cache: AdapterEmbeddingCache, expected: AdapterCacheMetadata
) -> bool:
    try:
        validate_adapter_cache(cache)
    except AdapterCacheIncompatibleError:
        return True
    return cache.metadata != expected


def write_adapter_cache(cache: AdapterEmbeddingCache, path: Path) -> None:
    validate_adapter_cache(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cache), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_adapter_cache(path: Path) -> AdapterEmbeddingCache:
    if not path.is_file():
        raise AdapterDatasetUnavailableError("required frozen embedding cache is unavailable")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cache = AdapterEmbeddingCache(
            metadata=AdapterCacheMetadata(**data["metadata"]),
            image_embeddings=data["image_embeddings"],
            caption_embeddings=data["caption_embeddings"],
            caption_image_ids=data["caption_image_ids"],
        )
        validate_adapter_cache(cache)
        return cache
    except ContrastiveAdapterError:
        raise
    except Exception as error:
        raise AdapterCacheIncompatibleError(
            "frozen embedding cache could not be validated"
        ) from error


def prepare_adapter_cache(
    rows: list[ManifestItemV2],
    backend: Any,
    provenance: HFFlickr8kProvenance,
    *,
    split: str,
    image_count: int,
    seed: int,
    cache_path: Path,
) -> tuple[AdapterEmbeddingCache, bool]:
    """Create one bounded cache or reuse an exactly compatible existing cache."""
    selected = select_grouped_subset(rows, split, image_count, seed)
    if cache_path.is_file():
        candidate = load_adapter_cache(cache_path)
        expected = make_adapter_cache_metadata(
            selected,
            backend,
            provenance,
            split=split,
            requested_image_count=image_count,
            seed=seed,
            dimension=candidate.metadata.embedding_dimension,
        )
        if adapter_cache_is_stale(candidate, expected):
            raise AdapterCacheIncompatibleError(
                f"existing {split} cache is incompatible; choose a new output path"
            )
        return candidate, True
    backend.ensure_loaded()
    image_rows: dict[str, ManifestItemV2] = {}
    for row in selected:
        image_rows.setdefault(row.image_id, row)
    image_vectors = backend.encode_images(
        [image_rows[image_id].image_path for image_id in sorted(image_rows)]
    )
    caption_vectors = backend.encode_texts([row.caption for row in selected])
    image_embeddings = dict(zip(sorted(image_rows), image_vectors, strict=True))
    caption_embeddings = {
        row.caption_id: vector for row, vector in zip(selected, caption_vectors, strict=True)
    }
    cache = AdapterEmbeddingCache(
        metadata=make_adapter_cache_metadata(
            selected,
            backend,
            provenance,
            split=split,
            requested_image_count=image_count,
            seed=seed,
            dimension=backend.dimension,
        ),
        image_embeddings=image_embeddings,
        caption_embeddings=caption_embeddings,
        caption_image_ids={row.caption_id: row.image_id for row in selected},
    )
    write_adapter_cache(cache, cache_path)
    return cache, False


def deterministic_image_batches(
    image_ids: Iterable[str], batch_size: int, seed: int, epoch: int
) -> list[list[str]]:
    if batch_size <= 0 or batch_size > MAX_BATCH_SIZE:
        raise ContrastiveAdapterError(
            f"batch size must be between 1 and {MAX_BATCH_SIZE} unique images"
        )
    ordered = sorted(image_ids)
    random.Random(seed + epoch).shuffle(ordered)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def create_adapter_pair(
    input_dimension: int,
    bottleneck_dimension: int = DEFAULT_BOTTLENECK_DIMENSION,
    seed: int = DEFAULT_SEED,
) -> Any:
    """Create separate lazy-Torch residual adapters with an identity initialization."""
    torch = require_torch()
    torch.manual_seed(seed)

    class ResidualAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = torch.nn.Linear(input_dimension, bottleneck_dimension)
            self.activation = torch.nn.GELU()
            self.up = torch.nn.Linear(bottleneck_dimension, input_dimension)
            torch.nn.init.zeros_(self.up.weight)
            torch.nn.init.zeros_(self.up.bias)

        def forward(self, inputs: Any) -> Any:
            residual = inputs + self.up(self.activation(self.down(inputs)))
            return torch.nn.functional.normalize(residual, p=2, dim=-1)

    return torch.nn.ModuleDict({"image": ResidualAdapter(), "text": ResidualAdapter()})


def relationship_positive_mask(
    image_ids: list[str], caption_image_ids: list[str], device: str = "cpu"
) -> Any:
    torch = require_torch()
    mask = [
        [caption_image_id == image_id for caption_image_id in caption_image_ids]
        for image_id in image_ids
    ]
    result = torch.tensor(mask, dtype=torch.bool, device=device)
    if result.numel() == 0 or not result.any(dim=1).all() or not result.any(dim=0).all():
        raise ContrastiveAdapterError("every image and caption must have a positive relationship")
    return result


def symmetric_multi_positive_loss(
    image_vectors: Any, caption_vectors: Any, positive_mask: Any, temperature: float
) -> Any:
    torch = require_torch()
    if temperature <= 0:
        raise ContrastiveAdapterError("temperature must be positive")
    logits = image_vectors @ caption_vectors.T / temperature
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    image_loss = -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()
    caption_positive = positive_logits.T
    caption_logits = logits.T
    caption_loss = -(
        torch.logsumexp(caption_positive, dim=1)
        - torch.logsumexp(caption_logits, dim=1)
    ).mean()
    loss = (image_loss + caption_loss) / 2
    if not torch.isfinite(loss):
        raise AdapterTrainingError("contrastive loss became non-finite")
    return loss


def evaluate_cache(
    cache: AdapterEmbeddingCache, adapters: Any | None = None, device: str = "cpu"
) -> BidirectionalResult:
    if cache.metadata.split != "validation":
        raise AdapterEvaluationError("checkpoint selection and evaluation require validation data")
    torch = require_torch()
    image_ids = sorted(cache.image_embeddings)
    caption_ids = sorted(cache.caption_embeddings)
    image_vectors = torch.tensor(
        [cache.image_embeddings[item_id] for item_id in image_ids],
        dtype=torch.float32,
        device=device,
    )
    caption_vectors = torch.tensor(
        [cache.caption_embeddings[item_id] for item_id in caption_ids],
        dtype=torch.float32,
        device=device,
    )
    if adapters is not None:
        adapters.eval()
        with torch.inference_mode():
            image_vectors = adapters["image"](image_vectors)
            caption_vectors = adapters["text"](caption_vectors)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    relationships = [image_index[cache.caption_image_ids[item_id]] for item_id in caption_ids]
    return evaluate_bidirectional_vectorized(
        image_vectors.detach().cpu().tolist(),
        caption_vectors.detach().cpu().tolist(),
        relationships,
    )


def mean_bidirectional_mrr(result: BidirectionalResult) -> float:
    return (result.text_to_image.metrics.mrr + result.image_to_text.metrics.mrr) / 2


@dataclass(frozen=True)
class AdapterTrainingConfig:
    seed: int = DEFAULT_SEED
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    early_stopping_patience: int = MAX_PATIENCE
    batch_size: int = MAX_BATCH_SIZE
    temperature: float = DEFAULT_TEMPERATURE
    device: str = "cpu"
    bottleneck_dimension: int = DEFAULT_BOTTLENECK_DIMENSION

    def validate(self) -> None:
        if self.max_epochs <= 0 or self.max_epochs > MAX_EPOCHS:
            raise ContrastiveAdapterError(f"maximum epochs must be between 1 and {MAX_EPOCHS}")
        if not 1 <= self.early_stopping_patience <= MAX_PATIENCE:
            raise ContrastiveAdapterError(
                f"early-stopping patience must be between 1 and {MAX_PATIENCE}"
            )
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise ContrastiveAdapterError(
                f"batch size must be between 1 and {MAX_BATCH_SIZE} unique images"
            )
        if self.learning_rate != DEFAULT_LEARNING_RATE:
            raise ContrastiveAdapterError("learning rate is fixed at 1e-3")
        if self.weight_decay != DEFAULT_WEIGHT_DECAY:
            raise ContrastiveAdapterError("weight decay is fixed at 1e-4")
        if self.temperature != DEFAULT_TEMPERATURE:
            raise ContrastiveAdapterError("temperature is fixed at 0.07")
        if self.bottleneck_dimension != DEFAULT_BOTTLENECK_DIMENSION:
            raise ContrastiveAdapterError("adapter bottleneck dimension is fixed at 128")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_mean_bidirectional_mrr: float


@dataclass(frozen=True)
class AdapterTrainingResult:
    history: list[EpochRecord]
    selected_epoch: int
    selected_validation_mean_mrr: float
    early_stopped: bool
    parameter_count: int


def select_best_epoch(history: list[EpochRecord]) -> int:
    if not history:
        raise AdapterTrainingError("training history is empty")
    return max(
        history,
        key=lambda record: (record.validation_mean_bidirectional_mrr, -record.epoch),
    ).epoch


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(
    train_cache: AdapterEmbeddingCache,
    validation_cache: AdapterEmbeddingCache,
    train_cache_path: Path,
    validation_cache_path: Path,
    config: AdapterTrainingConfig,
    result: AdapterTrainingResult,
) -> dict[str, Any]:
    return {
        "architecture": ADAPTER_ARCHITECTURE,
        "input_dimension": train_cache.metadata.embedding_dimension,
        "bottleneck_dimension": config.bottleneck_dimension,
        "separate_text_and_image_adapters": True,
        "clip_frozen": True,
        "source_model": {
            "backend_name": train_cache.metadata.backend_name,
            "backend_version": train_cache.metadata.backend_version,
            "model_name": train_cache.metadata.model_name,
            "model_revision": train_cache.metadata.model_revision,
            "preprocessing_identity": train_cache.metadata.preprocessing_identity,
        },
        "train_cache_fingerprint": file_digest(train_cache_path),
        "validation_cache_fingerprint": file_digest(validation_cache_path),
        "train_subset_fingerprint": train_cache.metadata.subset_fingerprint,
        "validation_subset_fingerprint": validation_cache.metadata.subset_fingerprint,
        "training_config": asdict(config),
        "selected_epoch": result.selected_epoch,
        "selected_validation_mean_mrr": result.selected_validation_mean_mrr,
        "early_stopped": result.early_stopped,
        "parameter_count": result.parameter_count,
        "history": [asdict(record) for record in result.history],
    }


def save_adapter_checkpoint(
    adapters: Any, metadata: dict[str, Any], checkpoint_path: Path, metadata_path: Path
) -> None:
    torch = require_torch()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": adapters.state_dict(), "metadata": metadata}, checkpoint_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_adapter_checkpoint(
    checkpoint_path: Path, metadata_path: Path, device: str = "cpu"
) -> tuple[Any, dict[str, Any]]:
    torch = require_torch()
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise AdapterDatasetUnavailableError("adapter checkpoint artifacts are unavailable")
    try:
        expected_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if payload["metadata"] != expected_metadata:
            raise AdapterCacheIncompatibleError("checkpoint metadata sidecar does not match")
        adapters = create_adapter_pair(
            expected_metadata["input_dimension"],
            expected_metadata["bottleneck_dimension"],
            expected_metadata["training_config"]["seed"],
        )
        adapters.load_state_dict(payload["state_dict"])
        adapters.to(device)
        adapters.eval()
        return adapters, expected_metadata
    except ContrastiveAdapterError:
        raise
    except Exception as error:
        raise AdapterCacheIncompatibleError("adapter checkpoint could not be loaded") from error


def _validate_cache_pair(
    train_cache: AdapterEmbeddingCache, validation_cache: AdapterEmbeddingCache
) -> None:
    if train_cache.metadata.split != "train" or validation_cache.metadata.split != "validation":
        raise AdapterCacheIncompatibleError(
            "adapter training requires official train and validation caches"
        )
    identity_fields = (
        "backend_name",
        "backend_version",
        "preprocessing_identity",
        "model_name",
        "model_revision",
        "dataset_fingerprint",
        "dataset_revision",
        "embedding_dimension",
    )
    if any(
        getattr(train_cache.metadata, field) != getattr(validation_cache.metadata, field)
        for field in identity_fields
    ):
        raise AdapterCacheIncompatibleError("train and validation cache identities differ")


def train_adapters(
    train_cache_path: Path,
    validation_cache_path: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    config: AdapterTrainingConfig,
) -> AdapterTrainingResult:
    """Run exactly one bounded configuration without reopening images or CLIP."""
    config.validate()
    torch = require_torch()
    train_cache = load_adapter_cache(train_cache_path)
    validation_cache = load_adapter_cache(validation_cache_path)
    _validate_cache_pair(train_cache, validation_cache)
    try:
        torch.manual_seed(config.seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        adapters = create_adapter_pair(
            train_cache.metadata.embedding_dimension,
            config.bottleneck_dimension,
            config.seed,
        ).to(config.device)
        parameter_count = sum(parameter.numel() for parameter in adapters.parameters())
        optimizer = torch.optim.AdamW(
            adapters.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        captions_by_image: dict[str, list[str]] = {}
        for caption_id, image_id in sorted(train_cache.caption_image_ids.items()):
            captions_by_image.setdefault(image_id, []).append(caption_id)
        history: list[EpochRecord] = []
        best_state: dict[str, Any] | None = None
        best_mean_mrr = float("-inf")
        epochs_without_improvement = 0
        early_stopped = False
        for epoch in range(1, config.max_epochs + 1):
            adapters.train()
            losses: list[float] = []
            for image_ids in deterministic_image_batches(
                train_cache.image_embeddings,
                config.batch_size,
                config.seed,
                epoch,
            ):
                caption_ids = sorted(
                    caption_id
                    for image_id in image_ids
                    for caption_id in captions_by_image[image_id]
                )
                image_inputs = torch.tensor(
                    [train_cache.image_embeddings[item_id] for item_id in image_ids],
                    dtype=torch.float32,
                    device=config.device,
                )
                caption_inputs = torch.tensor(
                    [train_cache.caption_embeddings[item_id] for item_id in caption_ids],
                    dtype=torch.float32,
                    device=config.device,
                )
                positive_mask = relationship_positive_mask(
                    image_ids,
                    [train_cache.caption_image_ids[item_id] for item_id in caption_ids],
                    config.device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = symmetric_multi_positive_loss(
                    adapters["image"](image_inputs),
                    adapters["text"](caption_inputs),
                    positive_mask,
                    config.temperature,
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation = evaluate_cache(validation_cache, adapters, config.device)
            validation_mean = mean_bidirectional_mrr(validation)
            record = EpochRecord(epoch, sum(losses) / len(losses), validation_mean)
            history.append(record)
            if validation_mean > best_mean_mrr + 1e-12:
                best_mean_mrr = validation_mean
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in adapters.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    early_stopped = True
                    break
        if best_state is None:
            raise AdapterTrainingError("training did not produce a validation checkpoint")
        adapters.load_state_dict(best_state)
        selected_epoch = select_best_epoch(history)
        result = AdapterTrainingResult(
            history=history,
            selected_epoch=selected_epoch,
            selected_validation_mean_mrr=best_mean_mrr,
            early_stopped=early_stopped,
            parameter_count=parameter_count,
        )
        metadata = _checkpoint_metadata(
            train_cache,
            validation_cache,
            train_cache_path,
            validation_cache_path,
            config,
            result,
        )
        save_adapter_checkpoint(adapters, metadata, checkpoint_path, metadata_path)
        return result
    except ContrastiveAdapterError:
        raise
    except Exception as error:
        raise AdapterTrainingError("bounded adapter training failed") from error


def validate_checkpoint_cache_identity(
    metadata: dict[str, Any], train_cache_path: Path, validation_cache_path: Path
) -> None:
    if metadata.get("train_cache_fingerprint") != file_digest(train_cache_path):
        raise AdapterCacheIncompatibleError("training cache changed after checkpoint selection")
    if metadata.get("validation_cache_fingerprint") != file_digest(validation_cache_path):
        raise AdapterCacheIncompatibleError("validation cache changed after checkpoint selection")
    if metadata.get("clip_frozen") is not True:
        raise AdapterCacheIncompatibleError("checkpoint does not confirm a frozen CLIP encoder")


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    mean_bidirectional_mrr_difference: float
    reasons: list[str]


def promotion_decision(
    zero_shot: BidirectionalResult,
    adapted: BidirectionalResult,
    *,
    evaluation_split: str,
) -> PromotionDecision:
    mean_difference = mean_bidirectional_mrr(adapted) - mean_bidirectional_mrr(zero_shot)
    checks = [
        (
            evaluation_split == "validation",
            "evaluation used only the untouched official validation subset",
        ),
        (
            mean_difference >= 0.005 - 1e-12,
            "mean bidirectional MRR improved by at least 0.005",
        ),
        (
            adapted.text_to_image.metrics.recall_at_10
            - zero_shot.text_to_image.metrics.recall_at_10
            >= -0.005 - 1e-12,
            "text-to-image Recall@10 did not decrease by more than 0.005",
        ),
        (
            adapted.image_to_text.metrics.recall_at_10
            - zero_shot.image_to_text.metrics.recall_at_10
            >= -0.005 - 1e-12,
            "image-to-text Recall@10 did not decrease by more than 0.005",
        ),
        (
            adapted.text_to_image.metrics.mrr - zero_shot.text_to_image.metrics.mrr
            >= -0.005 - 1e-12,
            "text-to-image MRR did not decrease by more than 0.005",
        ),
        (
            adapted.image_to_text.metrics.mrr - zero_shot.image_to_text.metrics.mrr
            >= -0.005 - 1e-12,
            "image-to-text MRR did not decrease by more than 0.005",
        ),
    ]
    reasons = [f"{'PASS' if passed else 'FAIL'}: {detail}" for passed, detail in checks]
    return PromotionDecision(all(passed for passed, _ in checks), mean_difference, reasons)


@dataclass(frozen=True)
class AdapterEvaluationComparison:
    zero_shot: BidirectionalResult
    adapted: BidirectionalResult
    promotion: PromotionDecision


def evaluate_adapters(
    train_cache_path: Path,
    validation_cache_path: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    device: str = "cpu",
) -> tuple[AdapterEvaluationComparison, dict[str, Any]]:
    train_cache = load_adapter_cache(train_cache_path)
    validation_cache = load_adapter_cache(validation_cache_path)
    _validate_cache_pair(train_cache, validation_cache)
    adapters, metadata = load_adapter_checkpoint(checkpoint_path, metadata_path, device)
    validate_checkpoint_cache_identity(metadata, train_cache_path, validation_cache_path)
    try:
        zero_shot = evaluate_cache(validation_cache, None, device)
        adapted = evaluate_cache(validation_cache, adapters, device)
        promotion = promotion_decision(
            zero_shot, adapted, evaluation_split=validation_cache.metadata.split
        )
        return AdapterEvaluationComparison(zero_shot, adapted, promotion), metadata
    except ContrastiveAdapterError:
        raise
    except Exception as error:
        raise AdapterEvaluationError("adapter validation evaluation failed") from error


def metrics_payload(metrics: RetrievalMetrics, candidate_count: int) -> dict[str, Any]:
    return asdict(metrics) | {"candidate_count": candidate_count}
