"""Multi-caption benchmark evaluation and comparison reporting."""

from dataclasses import asdict
import json
from pathlib import Path

from .baseline_index import SearchResult
from .clip_backend import ClipEmbeddingBackend
from .clip_workflow import build_clip_index
from .deterministic_image_encoder import DeterministicImageEncoder
from .deterministic_text_encoder import DeterministicTextEncoder
from .evaluation import RetrievalMetrics, metrics_from_ranks
from .flickr8k import MultiCaptionStatistics, multi_caption_statistics
from .manifest import ManifestItemV2
from .multimodal_evaluation import evaluate_multimodal_index
from .multimodal_index import build_multimodal_index, write_multimodal_index
from .text_baseline import build_vocabulary, encode_text


def evaluate_lexical_multicaption(rows: list[ManifestItemV2]) -> RetrievalMetrics:
    """Evaluate caption queries against per-image aggregated caption vectors."""
    vocabulary = build_vocabulary(rows)
    captions: dict[str, list[str]] = {}
    split_by_image: dict[str, str] = {}
    for row in rows:
        captions.setdefault(row.image_id, []).append(row.caption)
        split_by_image[row.image_id] = row.split
    candidate_vectors = {
        image_id: encode_text(" ".join(image_captions), vocabulary)
        for image_id, image_captions in captions.items()
    }
    ranks: list[int] = []
    for row in rows:
        if row.split == "train":
            continue
        query = encode_text(row.caption, vocabulary)
        scores = [
            SearchResult(image_id, sum(a * b for a, b in zip(query, vector, strict=True)), "", "", split_by_image[image_id])
            for image_id, vector in candidate_vectors.items()
            if split_by_image[image_id] != "train"
        ]
        ranked = sorted(scores, key=lambda result: (-result.score, result.item_id))
        ranks.append(next(index for index, result in enumerate(ranked, 1) if result.item_id == row.image_id))
    return metrics_from_ranks(ranks)


def render_benchmark_report(
    stats: MultiCaptionStatistics,
    clip_metrics: RetrievalMetrics,
    lexical_metrics: RetrievalMetrics,
    placeholder_metrics: RetrievalMetrics,
    *,
    model_name: str,
    device: str,
    dimension: int,
    cache_hit: bool,
) -> str:
    return "\n".join(
        [
            "# CLIP Real Benchmark Report",
            "",
            "Execution status: **successfully executed**",
            "",
            f"- Model: `{model_name}`",
            f"- Device: `{device}`",
            f"- Embedding dimension: {dimension}",
            f"- Unique images: {stats.unique_images}",
            f"- Caption queries: {stats.caption_queries}",
            f"- Captions per image (min/max/mean): {stats.captions_per_image_min}/"
            f"{stats.captions_per_image_max}/{stats.captions_per_image_mean:.2f}",
            f"- Cache: {'hit' if cache_hit else 'miss'}",
            "",
            "| Backend | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _metric_row("Lexical", lexical_metrics),
            _metric_row("Deterministic placeholder", placeholder_metrics),
            _metric_row("Zero-shot CLIP", clip_metrics),
            "",
            "This opt-in subset benchmark is not run by normal tests.",
            "",
        ]
    )


def _metric_row(name: str, metrics: RetrievalMetrics) -> str:
    return (
        f"| {name} | {metrics.recall_at_1:.4f} | {metrics.recall_at_5:.4f} | "
        f"{metrics.recall_at_10:.4f} | {metrics.mrr:.4f} | {metrics.median_rank:.2f} | "
        f"{metrics.mean_rank:.2f} |"
    )


def run_clip_benchmark(
    rows: list[ManifestItemV2],
    backend: ClipEmbeddingBackend,
    *,
    cache_path: Path,
    index_path: Path,
    report_path: Path,
    metrics_path: Path,
) -> tuple[RetrievalMetrics, bool]:
    index, cache_hit = build_clip_index(rows, backend, cache_path)
    write_multimodal_index(index, index_path)
    clip_metrics, _ = evaluate_multimodal_index(backend, index)
    placeholder_text = DeterministicTextEncoder()
    placeholder_index = build_multimodal_index(rows, DeterministicImageEncoder(), placeholder_text)
    placeholder_metrics, _ = evaluate_multimodal_index(placeholder_text, placeholder_index)
    lexical_metrics = evaluate_lexical_multicaption(rows)
    stats = multi_caption_statistics(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_benchmark_report(
            stats,
            clip_metrics,
            lexical_metrics,
            placeholder_metrics,
            model_name=backend.model_name,
            device=backend.device,
            dimension=index.dimension,
            cache_hit=cache_hit,
        ),
        encoding="utf-8",
    )
    data = {
        "status": "successfully_executed",
        "model_name": backend.model_name,
        "device": backend.device,
        "embedding_dimension": index.dimension,
        "cache_hit": cache_hit,
        **asdict(stats),
        "clip": asdict(clip_metrics),
        "lexical": asdict(lexical_metrics),
        "deterministic_placeholder": asdict(placeholder_metrics),
    }
    metrics_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return clip_metrics, cache_hit
