"""Held-out text-to-image retrieval evaluation."""

from .embedding_backends import TextEncoder
from .evaluation import EVALUATION_SPLITS, RetrievalMetrics, metrics_from_ranks
from .multimodal_index import CaptionQuery, MultimodalIndex, search_multimodal_index


def evaluate_multimodal_index(
    text_encoder: TextEncoder,
    index: MultimodalIndex,
    evaluation_splits: set[str] | frozenset[str] = EVALUATION_SPLITS,
) -> tuple[RetrievalMetrics, list[int]]:
    """Evaluate held-out captions against held-out image candidates."""
    query_source = index.queries or [
        CaptionQuery(entry.item_id, entry.item_id, entry.caption, entry.split)
        for entry in index.entries
    ]
    queries = [query for query in query_source if query.split in evaluation_splits]
    if not queries:
        raise ValueError("index has no validation/test queries to evaluate")
    ranks: list[int] = []
    for query in queries:
        results = search_multimodal_index(
            query.caption,
            text_encoder,
            index,
            k=len(queries),
            allowed_splits=set(evaluation_splits),
        )
        rank = next(
            position
            for position, result in enumerate(results, start=1)
            if result.item_id == query.image_id
        )
        ranks.append(rank)
    return metrics_from_ranks(ranks), ranks
